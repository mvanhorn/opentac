"""Hug router — routes wires as close to the sensor field as possible.

Selected by GenerateRequest.router == "hug", the default. The legacy routers
in route_columns.py / route_rows.py are never touched by this module.

Model
-----
Rows and columns live on opposite faces of the flex, so each face is an
independent single-copper-layer planar problem. Within a face, planarity
forces everything: there is no search, only an order.

Every net is routed as one or more hops. A pad hop runs from the net's
connector pad to the segment of the net nearest the connector; a link hop
bridges two segments of a net that a concavity cut apart, hugging around the
concavity. Each hop hugs a known span of the sensor-field contour, so the hop's
contour interval is computable before any geometry exists.

The intervals of one face form a laminar family (nested or disjoint — anything
else means a forced crossing, which is reported, not routed). Routing
innermost-first is therefore simply shortest-span-first: each wire hugs the
union of the field and all previously routed copper on its face, and lane
stacking emerges from the order instead of being computed.

The module holds both phases: the plan (build_plan: terminals, contour,
spans, order, consistency checks) and the geometry emission that walks each
face's hops in plan order (route_face_wires, route_hug_all).
"""

from dataclasses import dataclass, field as dc_field

import math
import re

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, unary_union

from .types import Layout, Params, Route

# A row/column is split into segments wherever consecutive pixels are farther
# apart than this multiple of the pitch — same rule as
# assign_rowcol_nets_segmented, so the hug router's segments are exactly the
# pieces the segmented assignment would have made separate nets.
GAP_FACTOR = 1.25


@dataclass
class Hop:
    net: str
    face: str                  # "col" | "row"
    kind: str                  # "pad" | "link"
    a_xy: np.ndarray           # start terminal (pad position for pad hops)
    b_xy: np.ndarray           # end terminal (anchor / segment entry)
    u_a: float                 # contour param of a_xy's projection (pad hops: cut)
    u_b: float
    span: tuple                # (lo, hi) contour interval this hop hugs
    direction: str             # "right" (increasing u from the cut) | "left"
    pad_num: int | None = None
    gap_index: int = 0         # 0 = pad hop, k>=1 = link across the k-th gap
    depth: int = 0             # nesting depth within its face (0 = innermost tier)
    eff_len: float = 0.0       # length of the wire's real drawn interval (see analyze)


@dataclass
class NetChain:
    net: str
    face: str
    pad_num: int
    segments: list             # list of pixel-index arrays, in feed order
    hops: list                 # the net's hops, feed order (pad hop first)
    # Rows only: which flank the pad hop feeds from, hence which end of each
    # segment a link attaches to (see the link-tip extension in route_hug_all).
    feed_side: str = "left"


@dataclass
class HugPlan:
    ring: LineString           # field contour, re-based to start at the cut point
    length: float
    cut_xy: np.ndarray
    field_poly: Polygon
    chains: dict               # net -> NetChain
    order_col: list            # column-face hops in routing order
    order_row: list            # row-face hops in routing order
    warnings: list = dc_field(default_factory=list)
    # pad_num -> stub polyline [pad, ..., stub top] for pads whose straight
    # rise would leave a fixed board (see _pad_fan); hops start at the top
    pad_stubs: dict = dc_field(default_factory=dict)


# --------------------------------------------------------------------------
# Field contour
# --------------------------------------------------------------------------

def field_polygon(layout: Layout, params: Params, warnings: list | None = None) -> Polygon:
    """Union of pitch-sized cells around every pixel — the shape to hug.

    Pitch-sized (not pixel-sized) so adjacent cells share edges and the union
    is one connected polygon whose boundary follows the packed field.

    A packed field can come apart into islands: an outline with a narrow
    waist (a strut-joined wrap) at a pitch where no cell fits through the
    junction, or fixed mode's keep-out carving a corridor clean through.
    Hugging only the largest part is not enough: the island nets' anchors
    would project onto the main contour at meaningless params (u collapsing
    to the cut, span (0,0)) and their wires would be emitted as bare
    diagonals across the whole connector fan, crossing other wires.
    Instead, each island is bridged to the main body with a pitch-wide
    capsule across their narrowest gap, so one contour covers every pixel
    and island wires hug through the waist like any other lane — the same
    "hug around, don't teleport" approach the link-hop machinery takes for
    a split net.
    """
    # Cells are oversized by a 5µm weld: staggered packing routinely leaves
    # two cells touching only at a corner, and an exact union then reports
    # the field as "islands at distance zero". Bridging such a point-touch
    # would stamp a pitch/2-radius disc on the contour: every passing wire
    # and the board outline would hug around it, a half-circle bulge out of
    # a flat field edge over nothing. The weld fuses point-contacts into one
    # polygon — a ~10µm neck the base-clearance buffer smooths over — while
    # real gaps stay separate and still get capsule bridges below.
    weld = 0.005
    hx, hy = params.pitch_x_mm / 2 + weld, params.pitch_y_mm / 2 + weld
    cells = [box(x - hx, y - hy, x + hx, y + hy) for x, y in layout.placements_xy]
    merged = unary_union(cells)
    if merged.geom_type == "MultiPolygon":
        parts = sorted(merged.geoms, key=lambda g: -g.area)
        if warnings is not None:
            warnings.append(
                f"sensor field is split into {len(parts)} islands — hug contour "
                f"bridged across the narrowest gaps; check the waist routing")
        # The bridge must be a real corridor, pitch-wide, not a filament: a
        # hair-thin bridge puts the contour's two sides through the waist
        # ~0.1mm apart, making anchor projections and the arc-side chooser
        # ambiguous between them, and island wires then hairpin the wrong way
        # around. Pitch-wide, the junction behaves like one more column of
        # cells and lanes stack on each side of the waist as along any other
        # stretch of field.
        bridge_hw = min(params.pitch_x_mm, params.pitch_y_mm) / 2
        main = parts[0]
        rest = list(parts[1:])
        while rest:
            k = min(range(len(rest)), key=lambda i: main.distance(rest[i]))
            island = rest.pop(k)
            a, b = nearest_points(main, island)
            if a.distance(b) < 1e-9:
                bridge = a.buffer(bridge_hw, quad_segs=4)
            else:
                bridge = LineString([a, b]).buffer(bridge_hw, cap_style=2, join_style=2)
            main = unary_union([main, island, bridge])
            if main.geom_type == "MultiPolygon":     # pathological fallback
                main = max(main.geoms, key=lambda g: g.area)
        merged = main
    return orient(merged.buffer(0), 1.0)   # exterior CCW


def _cut_ring(field_poly: Polygon, cut_from_xy) -> tuple[LineString, np.ndarray]:
    """The field exterior as an open LineString starting/ending at the point
    nearest cut_from_xy (the connector), walked CCW. No hop crosses the cut, so
    every hop is a plain sub-interval of this ring's arc length."""
    ring = LineString(field_poly.exterior.coords)
    s_cut = ring.project(Point(cut_from_xy))
    cut_pt = ring.interpolate(s_cut)

    coords = list(field_poly.exterior.coords[:-1])
    # Distances of each vertex along the ring
    dists = [ring.project(Point(c)) for c in coords]
    after = [c for d, c in sorted(zip(dists, coords)) if d > s_cut + 1e-9]
    before = [c for d, c in sorted(zip(dists, coords)) if d < s_cut - 1e-9]
    new_coords = [(cut_pt.x, cut_pt.y)] + after + before + [(cut_pt.x, cut_pt.y)]
    return LineString(new_coords), np.array([cut_pt.x, cut_pt.y])


# --------------------------------------------------------------------------
# Segments and terminals
# --------------------------------------------------------------------------

def _net_segments(pix_idx, coords_along, gap_pitch):
    """Split a net's pixel indices into contiguous segments along one axis."""
    order = np.argsort(coords_along)
    idx = np.asarray(pix_idx)[order]
    vals = np.asarray(coords_along)[order]
    segments = [[idx[0]]]
    for k in range(1, len(idx)):
        if vals[k] - vals[k - 1] > gap_pitch * GAP_FACTOR:
            segments.append([])
        segments[-1].append(idx[k])
    return [np.array(s) for s in segments]


def _col_terminals(Pxy, seg, params):
    """(bottom entry, top exit) terminal points for one column segment."""
    spine_x = float(np.mean(Pxy[seg, 0])) + (
        params.pixel_w_mm / 2 - params.edge_clear_mm - params.trace_w_mm / 2)
    bottom = np.array([spine_x, float(np.min(Pxy[seg, 1])) - params.pixel_h_mm / 2
                       - params.trace_w_mm])
    top = np.array([spine_x, float(np.max(Pxy[seg, 1])) + params.pixel_h_mm / 2
                    + params.trace_w_mm])
    return bottom, top


def _row_via(Pxy, seg, params, side):
    """Legacy-style via anchor: near the left edge of the end pixel on either
    flank (route_rows.py puts the via at end-pixel-center + via_x_off for both
    sides; the wire enters it horizontally)."""
    y_mid = float(np.mean(Pxy[seg, 1]))
    off = -params.pixel_w_mm / 2 + params.edge_clear_mm + params.via_dia_mm / 2
    end_x = float(np.min(Pxy[seg, 0])) if side == "left" else float(np.max(Pxy[seg, 0]))
    return np.array([end_x + off, y_mid])


def _row_gap_end(Pxy, seg, params, which):
    """Gap-facing edge point of a row segment — where a link trace attaches."""
    y_mid = float(np.mean(Pxy[seg, 1]))
    reach = params.pixel_w_mm / 2 - params.edge_clear_mm - params.trace_w_mm / 2
    if which == "right":
        return np.array([float(np.max(Pxy[seg, 0])) + reach, y_mid])
    return np.array([float(np.min(Pxy[seg, 0])) - reach, y_mid])


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------

def _pad_fan(layout: Layout, params: Params, field_poly, pad_x_left: float, pad_y: float) -> dict:
    """Stubs that bring connector pads in under a narrow board, fixed mode only.

    Every wire leaves its pad straight up. On a fixed outline whose bottom is
    narrower than the connector (a jaw tapering into a wider tab), that
    straight rise from an outer pad clips the shoulder where the tab meets
    the outline. Widening the keep-out cannot fix it: only the field
    shrinks, the wire stays where it is, and fixed mode ends up falling back
    to its uniform conservative band, losing taxels for one such wire.

    So, in the tail below the shoulder, the used pads are fanned in: each is
    moved the least it can be so that all of them, still in order and as far
    apart as the room allows, rise through the stretch whose straight rise
    clears the board. A moved pad gets a stub: up, one slant across, up to a
    common top; the slant is shallow enough that neighbours keep a wire
    pitch. Returns {} (nothing changes) on a growing board, when no pad needs
    to move, or when the fan does not fit.
    """
    if params.auto_expand_board or not layout.pad_to_net or layout.board_poly is None:
        return {}
    # the board as built (outline, tail and the fill at the tail's
    # shoulders), less the copper-to-edge clearance — what the escape check
    # measures against
    board_in = layout.board_poly.buffer(-(params.trace_w_mm / 2 + params.board_edge_clear_mm))
    if board_in.is_empty:
        return {}
    base, pitch = _wire_clearances(params)
    corridor = field_poly.buffer(base)
    nums = sorted(int(n) for n in layout.pad_to_net)
    xs = np.array([pad_x_left + (n - 1) * params.pad_pitch_mm for n in nums])
    y_low = float(corridor.bounds[1])
    if y_low <= pad_y:
        return {}

    def reach(x):
        """where a wire rising at x meets the corridor round the field (its
        lowest point when nothing is straight above)"""
        hit = corridor.intersection(LineString([(x, pad_y), (x, pad_y + 1e4)]))
        return float(hit.bounds[1]) if not hit.is_empty else y_low

    def clean(x):
        return board_in.covers(LineString([(x, pad_y), (x, reach(x))]))

    if all(clean(x) for x in xs):
        return {}
    # the stretch of x whose straight rise clears the board, around the pads
    grid = np.arange(xs.min() - 5.0, xs.max() + 5.0, 0.02)
    ok = np.array([clean(x) for x in grid])
    if not ok.any():
        return {}
    mid = float(np.median(xs))
    runs, start = [], None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        if (not v or i == len(ok) - 1) and start is not None:
            end = i if v else i - 1
            runs.append((grid[start], grid[end]))
            start = None
    lo, hi = min(runs, key=lambda r: 0.0 if r[0] <= mid <= r[1] else min(abs(mid - r[0]), abs(mid - r[1])))
    n = len(xs)
    # Targets, in order, as far apart as the clean stretch allows (never
    # wider than the pads themselves), each moved the least it can be.
    gap = min(params.pad_pitch_mm, (hi - lo) / max(1, n - 1))
    if gap < pitch * 1.02:
        return {}                                  # no room to fan in
    lb = lo + gap * np.arange(n)
    ub = hi - gap * np.arange(n)[::-1]
    xt = np.clip(xs, lb, ub)
    move = np.abs(xt - xs) > 1e-6
    if not move.any():
        return {}
    # Every wire slants across on one band of the tail: straight up, one
    # straight slant to its target, straight up again. Neighbours start a pad
    # pitch apart and end at least `gap` apart, so a slant shallow enough
    # that gap * cos(angle) still clears a wire pitch keeps every pair apart.
    angle = 0.9 * math.acos(min(1.0, pitch / gap))
    shift = float(np.max(np.abs(xt - xs)))
    rise = shift / math.tan(angle) if angle > 1e-6 else float("inf")
    y_a = pad_y + pitch
    y_b = y_a + rise
    top = y_b + pitch
    if not math.isfinite(rise) or top >= y_low:
        return {}
    stubs = {}
    for i, num in enumerate(nums):
        if move[i]:
            pts = [(xs[i], pad_y), (xs[i], y_a), (xt[i], y_b), (xt[i], top)]
        else:
            pts = [(xs[i], pad_y), (xs[i], top)]
        stubs[num] = [(float(a_), float(b_)) for a_, b_ in pts]
    # the fan must itself be clean: inside the board, a pitch between pads
    lines = {k: LineString(v) for k, v in stubs.items()}
    keys = sorted(lines)
    for k in keys:
        if not board_in.covers(lines[k]):
            return {}
    for a_, b_ in zip(keys, keys[1:]):
        if lines[a_].distance(lines[b_]) < pitch - 1e-6:
            return {}
    return stubs


def build_plan(layout: Layout, outline_xy: np.ndarray, params: Params) -> HugPlan:
    Pxy = layout.placements_xy
    warnings = []
    fp = field_polygon(layout, params, warnings)

    x_mid = float(np.mean(outline_xy[:2, 0]))
    pad_x_left = x_mid - (layout.connector_pos - 1) * params.pad_pitch_mm / 2
    pad_y = layout.pad_y_mm

    pad_stubs = _pad_fan(layout, params, fp, pad_x_left, pad_y)
    cut_x = x_mid
    if pad_stubs:
        # the cut sits between the pad groups: move it with the fanned pads
        pairs = sorted((st[0][0], st[-1][0]) for st in pad_stubs.values())
        cut_x = float(np.interp(x_mid, [a for a, _ in pairs], [b for _, b in pairs]))
    ring, cut_xy = _cut_ring(fp, (cut_x, pad_y))
    L = ring.length

    def u_of(pt):
        return float(ring.project(Point(pt)))

    def pad_xy_of(pad_num):
        if pad_num in pad_stubs:
            return np.array(pad_stubs[pad_num][-1], dtype=float)
        return np.array([pad_x_left + (pad_num - 1) * params.pad_pitch_mm, pad_y])

    chains = {}

    # ---- per-net pixel index lists, joined names ----
    def net_pixels(per_pixel):
        nets = {}
        for i, n in enumerate(per_pixel):
            if n:
                nets.setdefault(n, []).append(i)
        return nets

    col_pixels = net_pixels(layout.col_net_per_pixel)
    row_pixels = net_pixels(layout.row_net_per_pixel)
    net_to_pad = {net: pad for pad, net in layout.pad_to_net.items()}

    # ---- row side rule (same top-half-left split the legacy pipeline uses) ----
    row_mean_y = {net: float(np.mean(Pxy[idx, 1])) for net, idx in row_pixels.items()}
    rows_by_height = sorted(row_pixels, key=lambda n: (-row_mean_y[n], n))
    top_count = (len(rows_by_height) + 1) // 2
    row_side = {net: ("left" if i < top_count else "right")
                for i, net in enumerate(rows_by_height)}

    def make_link_hop(net, face, a_xy, b_xy, gap_index):
        u_a, u_b = u_of(a_xy), u_of(b_xy)
        lo, hi = min(u_a, u_b), max(u_a, u_b)
        if hi - lo > L - (hi - lo):
            warnings.append(
                f"{net} link {gap_index}: shorter contour arc crosses the "
                f"connector cut (u={lo:.1f}..{hi:.1f}, L={L:.1f}) — check topology")
        return Hop(net=net, face=face, kind="link", a_xy=a_xy, b_xy=b_xy,
                   u_a=u_a, u_b=u_b, span=(lo, hi),
                   direction="right" if u_b >= u_a else "left",
                   gap_index=gap_index)

    # ---- columns: feed the segment nearest the connector (lowest), link upward ----
    for net in sorted(col_pixels):
        idx = col_pixels[net]
        pad = net_to_pad.get(net)
        if pad is None:
            warnings.append(f"{net}: no connector pad assigned")
            continue
        segments = _net_segments(idx, Pxy[np.asarray(idx), 1], params.pitch_y_mm)
        anchor, _ = _col_terminals(Pxy, segments[0], params)
        u = u_of(anchor)
        direction = "right" if u <= L - u else "left"
        span = (0.0, u) if direction == "right" else (u, L)
        hops = [Hop(net=net, face="col", kind="pad", a_xy=pad_xy_of(pad),
                    b_xy=anchor, u_a=0.0 if direction == "right" else L, u_b=u,
                    span=span, direction=direction, pad_num=pad)]
        for k in range(1, len(segments)):
            _, prev_top = _col_terminals(Pxy, segments[k - 1], params)
            nxt_bottom, _ = _col_terminals(Pxy, segments[k], params)
            hops.append(make_link_hop(net, "col", prev_top, nxt_bottom, k))
        chains[net] = NetChain(net=net, face="col", pad_num=pad,
                               segments=segments, hops=hops)

    # ---- column direction repair: a coherent split of the pad row ----
    # Each column pad hop above chose its direction independently ("shorter
    # way around"). On a plain field the per-wire optima are automatically a
    # contiguous split of the pad row and everything is consistent — but
    # when the contour cut lands on a small feature (an island bridged into
    # the contour sits right over the connector), wires to the far side of
    # the field are near-equidistant both ways, and independent choices
    # interleave directions across the pad row: neighbors take opposite
    # ways around the board and their fan verticals/arcs cross. The comb is
    # only sound when each direction group occupies a contiguous run of the
    # pad row, so choose the split, not the wires: pads left of the split go
    # "left", the rest go "right", picking the split with the smallest
    # total contour travel. On plain fields this reproduces the per-wire
    # choice exactly; in the ambiguous cases it keeps one bundle together.
    col_pads = sorted((c.hops[0] for c in chains.values() if c.face == "col"),
                      key=lambda h: h.a_xy[0])
    if len(col_pads) > 1:
        right_costs = [h.u_b for h in col_pads]
        left_costs = [L - h.u_b for h in col_pads]
        best_k, best_cost = 0, None
        for k in range(len(col_pads) + 1):
            cost = sum(left_costs[:k]) + sum(right_costs[k:])
            if best_cost is None or cost < best_cost - 1e-9:
                best_k, best_cost = k, cost
        for i, h in enumerate(col_pads):
            direction = "left" if i < best_k else "right"
            if direction != h.direction:
                h.direction = direction
                h.span = (0.0, h.u_b) if direction == "right" else (h.u_b, L)
                h.u_a = 0.0 if direction == "right" else L

    # ---- rows: feed from the flank the side rule picks, link across gaps ----
    for net in sorted(row_pixels):
        idx = row_pixels[net]
        pad = net_to_pad.get(net)
        if pad is None:
            warnings.append(f"{net}: no connector pad assigned")
            continue
        segments = _net_segments(idx, Pxy[np.asarray(idx), 0], params.pitch_x_mm)
        side = row_side[net]
        if side == "right":
            segments = segments[::-1]      # feed the rightmost segment first
        anchor = _row_via(Pxy, segments[0], params, side)
        # Project the approach point (the fed segment's edge on the feeding
        # side), not the via: the via can sit deep inside a narrow segment
        # (e.g. a waist row) and snap to the wrong contour wall. The wire hugs
        # the contour to this edge, then enters horizontally over its own
        # pixels to the via — same-net copper, same entry the legacy router uses.
        y_mid0 = float(np.mean(Pxy[segments[0], 1]))
        if side == "left":
            appr = (float(np.min(Pxy[segments[0], 0])) - params.pitch_x_mm / 2, y_mid0)
        else:
            appr = (float(np.max(Pxy[segments[0], 0])) + params.pitch_x_mm / 2, y_mid0)
        u = u_of(appr)
        direction = "left" if side == "left" else "right"
        span = (0.0, u) if direction == "right" else (u, L)
        hops = [Hop(net=net, face="row", kind="pad", a_xy=pad_xy_of(pad),
                    b_xy=anchor, u_a=0.0 if direction == "right" else L, u_b=u,
                    span=span, direction=direction, pad_num=pad)]
        for k in range(1, len(segments)):
            if side == "left":     # chain runs left -> right
                a = _row_gap_end(Pxy, segments[k - 1], params, "right")
                b = _row_gap_end(Pxy, segments[k], params, "left")
            else:                  # chain runs right -> left
                a = _row_gap_end(Pxy, segments[k - 1], params, "left")
                b = _row_gap_end(Pxy, segments[k], params, "right")
            hops.append(make_link_hop(net, "row", a, b, k))
        chains[net] = NetChain(net=net, face="row", pad_num=pad,
                               segments=segments, hops=hops, feed_side=side)

    # ---- per-face ordering + consistency checks ----
    def face_hops(face):
        return [h for c in chains.values() if c.face == face for h in c.hops]

    def analyze(face):
        hops = face_hops(face)
        # Laminarity: any partial overlap is a forced crossing.
        eps = 1e-6
        for i in range(len(hops)):
            for j in range(i + 1, len(hops)):
                a, b = hops[i].span, hops[j].span
                lo, hi = max(a[0], b[0]), min(a[1], b[1])
                if hi - lo <= eps:
                    continue                       # disjoint
                if (a[0] <= b[0] + eps and b[1] <= a[1] + eps) or \
                   (b[0] <= a[0] + eps and a[1] <= b[1] + eps):
                    continue                       # nested
                warnings.append(
                    f"{face}: spans of {hops[i].net}/{hops[i].kind}{hops[i].gap_index} "
                    f"and {hops[j].net}/{hops[j].kind}{hops[j].gap_index} partially "
                    f"overlap — forced crossing, topology not planar as ordered")
        # Nesting depth (for reporting/visuals): how many spans strictly contain mine.
        for h in hops:
            h.depth = sum(
                1 for o in hops
                if o is not h and o.span[0] <= h.span[0] + eps
                and h.span[1] <= o.span[1] + eps
                and (o.span[1] - o.span[0]) > (h.span[1] - h.span[0]) + eps)
        # Pad fan-out order must match peel order per direction.
        for direction, sign in (("right", 1), ("left", -1)):
            pads = [h for h in hops if h.kind == "pad" and h.direction == direction]
            pads.sort(key=lambda h: h.span[1] - h.span[0])   # peel order
            xs = [h.a_xy[0] for h in pads]
            for k in range(1, len(xs)):
                if sign * (xs[k] - xs[k - 1]) < -1e-9:
                    warnings.append(
                        f"{face}/{direction}: pad order breaks nesting between "
                        f"{pads[k-1].net} (x={xs[k-1]:.2f}) and {pads[k].net} "
                        f"(x={xs[k]:.2f}) — fan-out would cross")
        # Routing order: innermost first. For links that is shortest span
        # first. A pad hop's real run starts at its pad, not at the cut, so
        # its measure is the span corrected by where its pad sits relative
        # to the cut: a right-going wire whose pad lies right of the cut
        # enters the contour that much past the cut (shorter run), one
        # whose pad lies left of it enters before the cut (longer run) —
        # eff = span - o for "right", span + o for "left", with
        # o = pad_x - cut_x measured along the pad row. This is the same
        # fan-annulus refinement the legacy comb encodes (near the
        # connector a wire whose pad sits beyond another wire's anchor must
        # route after it), but computed from the pad's x directly. It is
        # not computed by projecting the pad onto the ring: a pad's nearest
        # ring point can be an unrelated feature — an island-bridge lobe,
        # the far wall of a notch — and the projected measure then inverts
        # the peel order and braids neighbouring wires. Plain span length
        # without the offset is no better: it drops the annulus refinement
        # and loses many sensors on shapes whose bundles interleave at the
        # cut. The measure is a length, so the absolute value is taken:
        # span - o goes negative when the pad sits beyond its own anchor,
        # and such a wire's run is |span - o| long on the far side;
        # ordering by the signed value routes the longest such wire first
        # and buries its neighbors' anchors under its lane.
        cut_x = float(cut_xy[0])
        for h in hops:
            span_len = h.span[1] - h.span[0]
            if h.kind == "pad":
                o = float(h.a_xy[0]) - cut_x
                h.eff_len = abs(span_len - o if h.direction == "right"
                                else span_len + o)
            else:
                h.eff_len = span_len
        return sorted(hops, key=lambda h: (h.eff_len, h.net, h.gap_index))

    order_col = analyze("col")
    order_row = analyze("row")

    return HugPlan(pad_stubs=pad_stubs, ring=ring, length=L, cut_xy=cut_xy, field_poly=fp,
                   chains=chains, order_col=order_col, order_row=order_row,
                   warnings=warnings)


# --------------------------------------------------------------------------
# Emission core: the hug walk
# --------------------------------------------------------------------------

def _pad_entry(ring: LineString, h) -> tuple[Point, tuple | None]:
    """Where a pad hop joins the corridor, as (entry point, elbow-or-None).

    The wire must start with a vertical rise at its own pad's x — that is the
    fan-comb invariant the whole connector zone relies on: pad order equals
    lane order, so vertical stubs at distinct x can never cross each other,
    and each stub stops one pitch short of every earlier wire's arc. (A
    globally-nearest projection instead of the vertical slides sideways onto
    neighboring wires' end caps as the bundle grows, skewing the fan and
    crossing earlier stubs.)

    Two cases keep that invariant for every pad:

    - The rising ray hits the corridor boundary: entry is the first point
      crossed (directly above the pad), elbow is None — the classic comb.
    - The ray misses: the pad sits beyond the corridor's footprint on this
      side (a wide connector next to a field that narrows toward the tail —
      e.g. a 30-pos connector under an arch whose bottom rows step inward).
      There is no boundary above the pad, but the comb generalizes: rise
      vertically at the pad's x to the height of the corridor's extreme
      point q on the pad's side — the ring point whose x is closest to the
      pad's x, lowest such — then run horizontally to q. The returned elbow
      is (pad_x, q.y) and entry is q. The horizontal run spans only x
      values beyond the corridor's extent, and every interior pad (one
      whose ray hits) lies under that extent, so the run provably cannot
      cross an interior pad's vertical; the Euclidean-nearest ring point
      does not have this property (it can sit past an interior vertical
      and lower, sending the horizontal straight through it).
      The caller prepends the elbow to the hugged arc so the run is
      buffered like any other routed copper; the next spilled pad's
      extreme point then lands on this wire's stub cap one pitch further
      out, and the bundle self-organizes into the same nested staircase
      lanes form everywhere else. A bare nearest-point projection here
      would instead draw a long diagonal from the pad across every
      neighbor's vertical.
    """
    x, y = float(h.a_xy[0]), float(h.a_xy[1])
    ray = LineString([(x, y), (x, y + 1e4)])
    hit = ray.intersection(ring)
    pts = []
    if not hit.is_empty:
        for g in getattr(hit, "geoms", [hit]):
            pts.extend([Point(c) for c in getattr(g, "coords", [])])
    pts = [p for p in pts if p.y > y]
    if pts:
        return min(pts, key=lambda p: p.y), None   # first boundary crossed rising
    coords = np.asarray(ring.coords)
    dx = np.abs(coords[:, 0] - x)
    near = coords[dx <= dx.min() + 0.02]
    qx, qy = near[np.argmin(near[:, 1])]
    return Point(float(qx), float(qy)), (x, float(qy))

def _hop_key(h) -> str:
    return f"{h.net}:{h.kind}{h.gap_index}"


def route_face_wires(plan: HugPlan, face: str, base_clear: float, wire_pitch: float,
                     sample_mm: float = 0.5, order=None, flip=frozenset()):
    """Route one face's hops in plan order; each wire hugs the union of the
    field and all previously routed wires on this face.

    The wire's path is a stretch of the boundary of
        buffer(field, base_clear) ∪ buffer(routed wires, wire_pitch)
    between the projections of its two terminals, walked on the side the plan's
    span dictates. A wire therefore cannot cross anything routed before it
    along its hugged arc — but a path also contains straight jumps (the pad
    stub, the anchor stub) whose safety rests on the fan-comb and disc
    arguments, and a plan whose projections are wrong for an unforeseen
    topology can degenerate the arc into more jumps. Emission therefore does
    not trust the model: each wire is checked against every previously
    emitted wire of the face, and one that would cross or come closer than
    the DRC minimum fails its whole net instead of being emitted. Callers
    drop or retry failed nets, so no crossing reaches the output.

    `order` replaces the plan's routing order and `flip` (hop keys) takes the
    other arc for those hops — both only for route_face_wires_repaired.

    Returns ([(hop, Nx2 coords), ...] in routing order, set of failed nets).
    """
    hops = order if order is not None else (plan.order_col if face == "col" else plan.order_row)
    obstacle = plan.field_poly.buffer(base_clear, quad_segs=8)
    # Verification threshold: the DRC centerline minimum. Lanes are built at
    # wire_pitch = min + DISCRETIZATION_MARGIN, so legitimate geometry passes
    # with the margin to spare and anything under the DRC line is refused.
    min_gap = wire_pitch - DISCRETIZATION_MARGIN - 1e-6
    emitted = []          # (net, LineString) accepted so far, this face
    failed_nets = set()

    # Field-side terminals of other nets are clearance discs: another net's
    # anchor can sit almost on top of my pad's vertical line (0.5mm pad grid vs
    # 4.2mm column grid, at the connector split), and a wire passing it must
    # clear the connection stub that will land there. Within a bundle nesting
    # already prevents this — a wire never passes a not-yet-routed wire's
    # anchor — so the discs matter exactly where the two directions straddle
    # the cut, and are covered by wire buffers everywhere else.
    terms = [(h2.net, tuple(h2.b_xy)) for h2 in hops]
    terms += [(h2.net, tuple(h2.a_xy)) for h2 in hops if h2.kind == "link"]
    nets = {h2.net for h2 in hops}
    discs_excl = {}
    for net in nets:
        pts = [Point(p) for n2, p in terms if n2 != net]
        discs_excl[net] = (unary_union([p.buffer(wire_pitch, quad_segs=8) for p in pts])
                           if pts else None)

    out = []
    for h in hops:
        if h.net in failed_nets:
            continue                       # net already refused on this face
        R = obstacle
        if discs_excl.get(h.net) is not None:
            R = R.union(discs_excl[h.net])
        if R.geom_type == "MultiPolygon":
            R = max(R.geoms, key=lambda g: g.area)
        ring = LineString(R.exterior.coords)
        L = ring.length
        # Project via the plan contour's approach points (u_a/u_b), not the raw
        # terminals: a terminal deep in a narrow region (a waist via) is nearer
        # the wrong wall, and a direct projection would send the wire the long
        # way around. Pad hops enter vertically above their own pad instead.
        elbow = None
        if h.kind == "pad":
            entry, elbow = _pad_entry(ring, h)
            pa = float(ring.project(entry))
        else:
            pa = float(ring.project(plan.ring.interpolate(h.u_a % plan.length)))
        pb = float(ring.project(plan.ring.interpolate(h.u_b % plan.length)))
        # The hop must hug the side of the field its span covers: pick the arc
        # whose midpoint lies nearest the span's midpoint on the plan contour.
        ref = plan.ring.interpolate((h.span[0] + h.span[1]) / 2)
        best = None
        for start, end, rev in ((pa, pb, False), (pb, pa, True)):
            dist = (end - start) % L
            n = max(2, int(dist / sample_mm))
            pts = [ring.interpolate((start + dist * k / n) % L) for k in range(n + 1)]
            d = pts[len(pts) // 2].distance(ref)
            if best is None or ((d < best[0]) != (_hop_key(h) in flip)):
                best = (d, pts, rev)
        _, pts, rev = best
        coords = [(p.x, p.y) for p in pts]
        if rev:
            coords.reverse()
        # A spilled pad (rising ray misses the corridor) joins via its elbow:
        # vertical pad->elbow, horizontal elbow->entry, then the hugged arc.
        # The elbow leads the coords so the horizontal run is buffered below.
        if elbow is not None:
            coords = [elbow] + coords
        path = np.array([tuple(h.a_xy)] + coords + [tuple(h.b_xy)])

        # The hard invariant: refuse any wire the model got wrong.
        path_ls = LineString([tuple(p) for p in path])
        clash = None
        for other_net, other_ls in emitted:
            if other_net == h.net:
                continue
            if path_ls.distance(other_ls) < min_gap:
                clash = other_net
                break
        if clash is not None:
            failed_nets.add(h.net)
            # Withdraw the net's already-accepted hops from the output (its
            # copper stays in `obstacle` — later wires keeping distance from
            # a withdrawn lane is safe, merely conservative).
            out = [(h2, p2) for h2, p2 in out if h2.net != h.net]
            emitted = [(n2, l2) for n2, l2 in emitted if n2 != h.net]
            plan.warnings.append(
                f"{face}: {h.net} {h.kind}{h.gap_index} would come within DRC "
                f"minimum of {clash} — net refused, not emitted")
            continue
        emitted.append((h.net, path_ls))
        out.append((h, path))
        # Buffer the arc plus the field-side stub (arc end -> anchor), so later
        # wires keep their pitch from the connection stub too — the discs above
        # guarantee no terminal ever ends up inside this buffer. The pad-side
        # vertical is deliberately not buffered: adjacent pads sit only one pad
        # pitch (0.5mm) apart, a hair over the wire pitch, and the ordered fan
        # comb already keeps pad verticals apart; buffering them leaves later
        # entry rays nothing to rise through. (A spilled pad's horizontal run
        # is in coords via its elbow and so is buffered — that is what makes
        # the next spilled pad's stub stack outside it instead of through it.)
        blocked = coords + [tuple(h.b_xy)]
        if h.kind == "link":
            blocked = [tuple(h.a_xy)] + blocked
        obstacle = obstacle.union(LineString(blocked).buffer(wire_pitch, quad_segs=8))
    return out, failed_nets


_REFUSAL = re.compile(r"^(col|row): (\S+) (pad|link)(\d+) would come within DRC minimum of (\S+) ")


def route_face_wires_repaired(plan: HugPlan, face: str, base_clear: float, wire_pitch: float,
                              sample_mm: float = 0.5, max_repairs: int = 12):
    """route_face_wires, then repairs driven by its own refusal report: for a
    refused hop ("X would come within DRC minimum of Y") try X's other arc,
    then X routed just before Y's first hop, and keep a change only when the
    face then refuses fewer nets. Bounded (max_repairs kept changes, two tries
    per refusal); every wire is still checked by the emission, so a repair can
    never produce a crossing. The plan's order is wrong where a pad sits beyond
    its own anchor near the cut (eff_len's straight-x offset against an arc
    length), and the midpoint arc test can take the long way round a U; both
    show up exactly as such a refusal."""
    attr = "order_col" if face == "col" else "order_row"
    key_w = list(plan.warnings)

    def run(order, flips):
        plan.warnings[:] = key_w
        out, failed = route_face_wires(plan, face, base_clear, wire_pitch, sample_mm,
                                       order=order, flip=flips)
        return out, failed, list(plan.warnings[len(key_w):])

    order, flips = list(getattr(plan, attr)), frozenset()
    out, failed, new_w = run(order, flips)
    tried, kept = set(), 0
    while failed and kept < max_repairs:
        improved = False
        for line in new_w:
            m = _REFUSAL.match(line)
            if not m:
                continue
            hk, clash = f"{m.group(2)}:{m.group(3)}{m.group(4)}", m.group(5)
            for kind in ("flip", "before"):
                if (hk, clash, kind) in tried:
                    continue
                tried.add((hk, clash, kind))
                o2, f2 = list(order), set(flips)
                if kind == "flip":
                    f2 ^= {hk}
                else:
                    # Positions by identity: list.index/remove compare by value, and
                    # a hop's numpy fields make that comparison raise.
                    ih = next((i for i, x in enumerate(o2) if _hop_key(x) == hk), None)
                    j = next((i for i, x in enumerate(o2) if x.net == clash), None)
                    if ih is None or j is None or ih < j:
                        continue
                    o2.insert(j, o2.pop(ih))
                res = run(o2, frozenset(f2))
                if len(res[1]) < len(failed):
                    order, flips = o2, frozenset(f2)
                    out, failed, new_w = res
                    kept += 1
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    plan.warnings[:] = key_w + new_w
    if kept:
        plan.warnings.append(f"{face}: {kept} refusal repair(s) kept")
    return out, failed


# The walk places wires by sampling a buffered boundary, and every step of that
# discretization (ring sampling chords, buffer polygonization, final simplify)
# pulls points slightly inside the exact offset. Routing at exactly the required
# clearance therefore fails DRC by a few hundredths everywhere. This allowance
# is added on top of the exact requirement so the discretized result still
# clears it. At 0.25mm sampling / quad_segs=8 / simplify 0.02 the worst
# shortfall is well under this.
DISCRETIZATION_MARGIN = 0.08


def _wire_clearances(params: Params) -> tuple[float, float]:
    """(base, pitch) the hug walk routes with — shared by route_hug_all and
    hug_keepout_widths so the keep-out models the identical geometry.

    Base lane: clear the pixel copper, and clear the column anchors that hang
    trace_w below the pixel edge (i.e. hang below the cell contour by
    trace_w - slack). A base lane closer than that traps a not-yet-connected
    anchor between the innermost passing wire and the field — the wire dips
    around the anchor's clearance disc and seals it into a pocket.
    """
    slack = min((params.pitch_x_mm - params.pixel_w_mm) / 2,
                (params.pitch_y_mm - params.pixel_h_mm) / 2)
    pitch = params.trace_w_mm + params.clearance_mm + DISCRETIZATION_MARGIN
    anchor_hang = max(0.0, params.trace_w_mm - max(0.0, slack))
    base = max(
        max(0.05, params.clearance_mm + params.trace_w_mm / 2 - max(0.0, slack))
        + DISCRETIZATION_MARGIN,
        anchor_hang + pitch + 0.02,
    )
    return base, pitch


def hug_keepout_widths(profile, params: Params, layout: Layout) -> np.ndarray:
    """Exact fixed-mode wiring demand from the hug plan — a drop-in
    replacement for keepout.analytic_widths when router="hug".

    The hug walk places the k-th wire passing a stretch of the field contour
    at base + k*pitch from it, so the band the drawn outline must reserve at
    each boundary sample is computable before any copper exists: project the
    sample onto the plan contour and count the hops whose span covers that
    param. Rows and columns live on opposite copper faces, so the band is the
    maximum of the two faces' stacks, not their sum. On top of the outermost
    wire's centerline: its own half-width + board edge clearance, plus one
    pitch of safety for the walk's rounded end caps and step-arounds. Samples
    no wire passes fall back to the explicit floor (0 by default — the model
    is exact, so it needs no hidden floor of its own).
    """
    from .keepout import fixed_keepout_floor
    if len(layout.placements_xy) == 0 or len(profile.samples) == 0:
        return profile.widths.copy()
    plan = build_plan(layout, layout.sensor_outline_xy, params)
    base, pitch = _wire_clearances(params)
    edge = params.trace_w_mm / 2 + params.board_edge_clear_mm
    safety = pitch
    floor = fixed_keepout_floor(params)
    eps = 1e-6
    col_spans = [h.span for h in plan.order_col]
    # A 4-layer board takes its rows off the field on the inner layers
    # (route_4layer.py): they need no band. A row that fits on neither inner
    # layer keeps B.Cu copper outside the field, and an escape grows the band
    # back where that happens, as for any other escaping wire.
    row_spans = ([] if int(getattr(params, "copper_layers", 2) or 2) == 4
                 else [h.span for h in plan.order_row])
    # Samples on the cable/tail region need no band at all: the fan near the
    # connector legitimately hangs through the cable edge into the tail, which
    # is board. Reserving the full fan depth inside the outline there is what
    # the demand would otherwise say, and it carves the bottom of the sensor
    # field away for room the wires never use. Escaping copper anywhere this
    # relief is too generous is caught by the retry loop's escape check and
    # grown back locally.
    relief = (layout.cable_poly.buffer(0.5)
              if layout.cable_poly is not None and not layout.cable_poly.is_empty
              else None)
    widths = np.empty(len(profile.samples), dtype=float)
    for i, s in enumerate(profile.samples):
        pt = Point(float(s[0]), float(s[1]))
        if relief is not None and relief.covers(pt):
            widths[i] = 0.0
            continue
        u = float(plan.ring.project(pt))
        n = max(
            sum(1 for lo, hi in col_spans if lo - eps <= u <= hi + eps),
            sum(1 for lo, hi in row_spans if lo - eps <= u <= hi + eps),
        )
        widths[i] = (base + (n - 1) * pitch + edge + safety) if n else 0.0
    return np.maximum(widths, floor)


def route_hug_all(layout: Layout, outline_xy: np.ndarray, params: Params, repair: bool = False):
    """Full hug routing at real clearances.
    Returns (col_routes, row_routes, plan, failed_pixels).

    repair=True routes each face with route_face_wires_repaired; the 4-layer
    router (route_4layer.route_hug_4layer) passes it, 2-layer boards do not.

    A net with links yields several Route entries sharing its net name — the
    pad hop plus one track per bridged gap; DRC skips same-net pairs. Base
    clearance is measured from the pitch-cell contour, so the slack between
    cell edge and pixel copper ((pitch - pixel)/2) is credited against the
    required copper-to-copper clearance.

    failed_pixels: pixel indices belonging to nets the emission refused
    (route_face_wires' hard invariant — a wire that would cross or
    under-clear previously routed copper is never emitted). Callers must
    treat these as routing problem pixels: drop them / grow the keep-out
    and retry, as for a legacy routing failure. Empty whenever plan and
    emission agree; where a topology opens a gap between them, the gap
    costs sensors, never crossings in the output.
    """
    plan = build_plan(layout, outline_xy, params)
    base, pitch = _wire_clearances(params)
    pad_xy_by_net = {c.net: np.asarray(plan.pad_stubs[c.pad_num][0] if c.pad_num in plan.pad_stubs
                                       else c.hops[0].a_xy, dtype=float)
                     for c in plan.chains.values()}
    col_routes, row_routes = [], []
    failed_nets = set()
    for face, bucket in (("col", col_routes), ("row", row_routes)):
        emit = route_face_wires_repaired if repair else route_face_wires
        wires, face_failed = emit(plan, face, base, pitch, sample_mm=0.25)
        failed_nets |= face_failed
        for h, path in wires:
            pts = [tuple(p) for p in path]
            anchor = np.asarray(h.b_xy, dtype=float)
            if face == "col":
                # Spine entry, like the legacy router's final segment
                # (route_columns.py appends anchor + pixel_h/2): a column
                # terminal sits trace_w outside the pixel edge, and the spine
                # copper starts inside it, so a wire that stops at the terminal
                # is ~2*trace_w short of actually connecting. Extend the tip
                # pixel_h/2 into its own pixel to overlap the spine. Pad hops
                # connect at their b end (a is the connector pad); links
                # connect at both ends. Same-net copper only — nothing else
                # is near these stubs, and DRC skips same-net pairs.
                half_h = params.pixel_h_mm / 2
                pts.append((h.b_xy[0], h.b_xy[1] + half_h))      # b: bottom terminal
                if h.kind == "link":
                    pts.insert(0, (h.a_xy[0], h.a_xy[1] - half_h))   # a: top terminal
            elif h.kind == "link":
                # Row links attach at a segment's gap-facing pixel edge, but a
                # row's only B.Cu copper is its bus, which spans via-to-via —
                # and the via sits near each pixel's left edge. So a link
                # landing on a segment's right end stops
                # pixel_w - 2*edge_clear - trace_w/2 - via_dia/2 short of any
                # row copper (2.53mm at 3mm pixels, 3.53mm at 4mm). DRC does
                # not detect this because it checks clearance, not
                # connectivity.
                #
                # Extend each link tip inward to its own segment's via, like
                # the column spine entry above. The plan's terminals are left
                # untouched, so routing order and every hugged path are
                # unaffected — this only appends copper along the row
                # centreline inside the net's own pixels, where nothing but its
                # own bus lives (DRC skips same-net pairs).
                #
                # The left-hand end already overlaps its via by a hair
                # (0.075mm short, closed by the 0.2mm trace width), but it is
                # extended too: "a link tip lands on its segment's via" is the
                # invariant, and it should not rest on an incidental overlap.
                chain = plan.chains[h.net]
                k = h.gap_index
                a_side = "right" if chain.feed_side == "left" else "left"
                b_side = "left" if chain.feed_side == "left" else "right"
                Pxy = layout.placements_xy
                pts.insert(0, tuple(_row_via(Pxy, chain.segments[k - 1], params, a_side)))
                b_via = _row_via(Pxy, chain.segments[k], params, b_side)
                pts.append(tuple(b_via))
                # The anchor is where the track lands on its net's copper,
                # which is the via: validate_routes checks a row route's last
                # point against it within 0.5mm, so an anchor left on the
                # pixel-edge terminal would report every linked row as
                # incomplete.
                anchor = b_via
            if h.kind == "pad" and plan.chains[h.net].pad_num in plan.pad_stubs:
                # the wire starts at its stub's top: run the stub down to the pad
                stub = plan.pad_stubs[plan.chains[h.net].pad_num]
                pts = [tuple(q) for q in stub[:-1]] + pts
            poly = np.asarray(LineString(pts).simplify(0.02).coords, dtype=float)
            bucket.append(Route(
                net_name=h.net,
                pad_num=plan.chains[h.net].pad_num,
                anchor_xy=anchor,
                pad_xy=pad_xy_by_net[h.net],
                polyline_xy=poly,
            ))
    # A net refused on one face leaves its pixels electrically dead either
    # way — fail the pixels, not just the face.
    failed_pixels = set()
    if failed_nets:
        for i, n in enumerate(layout.col_net_per_pixel):
            if n in failed_nets:
                failed_pixels.add(i)
        for i, n in enumerate(layout.row_net_per_pixel):
            if n in failed_nets:
                failed_pixels.add(i)
        col_routes = [r for r in col_routes if r.net_name not in failed_nets]
        row_routes = [r for r in row_routes if r.net_name not in failed_nets]
    return col_routes, row_routes, plan, failed_pixels
