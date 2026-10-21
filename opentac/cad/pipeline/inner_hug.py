"""Inner-layer rows for 4-layer boards: the hug router, inside the field.

route_hug.py routes every wire OUTSIDE the sensor field: each hugs the union of
the field and the wires laid before it, innermost first, so the wires of a
face nest with no search. Inner-layer copper has to stay UNDER the taxels, so
here the same walk runs INSIDE: a wire is a stretch of the boundary of the
free space — the taxel cells (inset from a fixed board's edge), less every
foreign via's keep-out disc, less the wires already laid on the layer, less
the terminal stubs of the rows still to come.

  In2.Cu  links: a row that a concavity cuts into pieces is joined across each
          gap. The links around one notch nest — the row nearest the notch tip
          runs closest to it, every further row around the last — so they go
          round the tab under the taxels, never round its tip off the board.
  In1.Cu  feeds: every row, from its end via on one flank, down the inside of
          the field to the exit above the connector, then straight down to a
          connector pad. Rows feed clockwise or counter-clockwise from the
          exit; each bundle nests innermost first, and the pads are handed out
          in the order the wires peel off the bottom edge.

Everything is decided by one contour — the field's, cut at the exit: spans,
routing order, side. Capacity is not counted anywhere. A wire is refused only
where the free space between its two ends has pinched shut, or when its walk
fails the independent clearance check; a refused row keeps its B.Cu hug wire
(the caller's fallback). Selected with OT_INNER_WIRE=hug (route_inner.py).
"""

import itertools
import math

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point
from shapely.geometry.polygon import orient
from shapely.ops import substring, unary_union

from opentac.cad.export.kicad_footprint import pixel_geometry
from .route_inner import _via_keepout, taxel_cells
from .types import Route

# Polygonal buffers sit INSIDE the true circle (by the chord's sagitta), so a
# walk along their boundary would come a few microns too close. Every
# obstacle is inflated by this much; the checks use the exact distances.
_M = 0.01


class _Space:
    """The free space of one inner layer and the contour that orders it."""

    def __init__(self, layout, params, cut_xy):
        self.s = params.trace_w_mm + params.clearance_mm
        self.a = _via_keepout(params)
        g = pixel_geometry(params.pixel_w_mm, params.pixel_h_mm, params.trace_w_mm,
                           params.gap_mm, params.center_clear_mm, params.edge_clear_mm,
                           params.via_dia_mm)
        P = layout.placements_xy
        self.vxy = np.column_stack([P[:, 0] + g.via_x, P[:, 1] - g.via_y])
        self.vnet = np.asarray(layout.row_net_per_pixel, dtype=object)
        region = taxel_cells(layout, params)
        if not params.auto_expand_board and layout.outline_poly is not None:
            edge_in = params.trace_w_mm / 2 + params.board_edge_clear_mm
            region = region.intersection(layout.outline_poly.buffer(-edge_in))
        self.region = region
        self.region_b = region.buffer(1e-3)
        self.discs = [Point(p).buffer(self.a + _M, quad_segs=8) for p in self.vxy]
        self.free0 = region.difference(unary_union(self.discs))
        # the ordering contour: the largest piece of the field, counter-
        # clockwise, starting at the exit (the point nearest the connector cut)
        main = max(getattr(region, "geoms", [region]), key=lambda q: q.area)
        ring = LineString(orient(main, 1.0).exterior.coords)
        u0 = ring.project(Point(cut_xy))
        self.ring = LineString(_sub(ring, u0, u0 - 1e-9) + [tuple(ring.interpolate(u0).coords[0])])
        self.L = self.ring.length
        self.main = main

    def u(self, pt):
        return float(self.ring.project(Point(pt)))

    def foreign(self, net):
        memo = self.__dict__.setdefault("_foreign", {})
        if net not in memo:
            memo[net] = MultiPoint([tuple(p) for p in self.vxy[self.vnet != net]])
        return memo[net]


def _sub(ring, u0, u1):
    """the closed ring's own vertices from param u0 forward to u1 (wrapping)"""
    L = ring.length
    u0, u1 = u0 % L, u1 % L
    parts = ([substring(ring, u0, u1)] if u1 >= u0
             else [substring(ring, u0, L), substring(ring, 0.0, u1)])
    pts = []
    for part in parts:
        coords = list(part.coords) if part.geom_type == "LineString" else [tuple(part.coords[0])]
        for q in coords:
            if not pts or math.dist(pts[-1], q) > 1e-9:
                pts.append(tuple(q))
    return pts


def _inside(sp, via, out_pt):
    """a terminal's stub corridor: from its via toward out_pt, the part
    inside the field (beyond the field it blocks nothing that matters)"""
    seg = LineString([tuple(via), tuple(out_pt)])
    cut = seg.intersection(sp.region)
    if cut.is_empty or cut.geom_type != "LineString":
        pts = [Point(via)] if cut.is_empty else [g for g in getattr(cut, "geoms", [cut])]
        cut = min(pts, key=lambda g: g.distance(Point(via)))
        if cut.geom_type == "Point":
            return LineString([tuple(via), tuple(via)])
    return cut


def _smooth(path, ok):
    """line of sight: from each kept point jump as far along the path as a
    straight segment passing ok() reaches (galloping, then bisecting — every
    accepted segment is checked, so only how far it reaches is heuristic).
    The endpoints stay."""
    out, i, n = [path[0]], 0, len(path)
    while i < n - 1:
        good, step = i + 1, 1
        while good + step < n and ok(LineString([path[i], path[good + step]])):
            good += step
            step *= 2
        lo, hi = good, min(good + step, n)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ok(LineString([path[i], path[mid]])):
                lo = mid
            else:
                hi = mid
        out.append(path[lo])
        i = lo
    return out


class _Layer:
    """One layer's walk state: the free space shrinks as wires are laid."""

    def __init__(self, sp, preplaced=()):
        self.sp = sp
        self.wires = [(net, LineString(p)) for net, p in preplaced]
        self.blocked = [w.buffer(sp.s + _M, quad_segs=8) for _, w in self.wires]

    def free(self, own_vias, corridors):
        """free space for a wire whose own vias may be touched, with the
        terminal corridors of rows still to come kept clear"""
        sp = self.sp
        f = sp.free0
        if own_vias:
            own = unary_union([sp.discs[i] for i in own_vias]).intersection(sp.region)
            f = f.union(own)
        cut = self.blocked + [c.buffer(sp.s + _M, quad_segs=8) for c in corridors]
        if cut:
            f = f.difference(unary_union(cut))
        return f

    def ok(self, net, corridors):
        """the independent check one segment (or path) of `net` must pass"""
        sp = self.sp
        foreign = sp.foreign(net)
        others = [w for n, w in self.wires if n != net]
        others = unary_union(others) if others else None
        cor = unary_union(corridors) if corridors else None

        def check(geom, in_region=True):
            if in_region and not sp.region_b.covers(geom):
                return "leaves the region"
            if not foreign.is_empty and geom.distance(foreign) < sp.a - 1e-6:
                return f"within {geom.distance(foreign):.3f} of a via"
            if others is not None and geom.distance(others) < sp.s - 1e-6:
                return f"within {geom.distance(others):.3f} of another wire"
            if cor is not None and geom.distance(cor) < sp.s - 1e-6:
                return "over the stub of a row still to come"
            return None
        return check

    def lay(self, net, path):
        line = LineString(path)
        self.wires.append((net, line))
        self.blocked.append(line.buffer(self.sp.s + _M, quad_segs=8))


def _walk(free, start_pt, end_pt, choose):
    """the boundary stretch of the free-space piece holding both ends, from
    start_pt's projection to end_pt's, on the side choose() picks; None when
    the ends lie in different pieces (the free space has pinched shut)"""
    if free.is_empty:
        return None
    pieces = list(getattr(free, "geoms", [free]))
    near = lambda pt: min(range(len(pieces)), key=lambda k: pieces[k].distance(Point(pt)))
    k = near(start_pt)
    if near(end_pt) != k:
        return None
    ring = LineString(orient(pieces[k], 1.0).exterior.coords)
    pa, pb = ring.project(Point(start_pt)), ring.project(Point(end_pt))
    fwd = _sub(ring, pa, pb)              # counter-clockwise
    back = _sub(ring, pb, pa)[::-1]       # clockwise
    return choose(fwd, back)


# ==========================================================================
# In2: links
# ==========================================================================

def route_links(layout, params, plan):
    """Returns (routes, nets_done, info) — the route_row_links_in2 contract:
    routes go on In2.Cu; for nets_done the hug router's B.Cu link hops are no
    longer needed (a net is done only when every one of its links is laid)."""
    sp = _Space(layout, params, plan.cut_xy)
    links = []
    for net, ch in plan.chains.items():
        if ch.face != "row" or len(ch.segments) < 2:
            continue
        for h in ch.hops:
            if h.kind != "link":
                continue
            A = np.asarray(ch.segments[h.gap_index - 1])
            B = np.asarray(ch.segments[h.gap_index])
            ia = int(A[np.argmin(np.linalg.norm(sp.vxy[A] - h.a_xy, axis=1))])
            ib = int(B[np.argmin(np.linalg.norm(sp.vxy[B] - h.b_xy, axis=1))])
            ua, ub = sp.u(h.a_xy), sp.u(h.b_xy)
            # the link hugs the hole between its pieces: the shorter way round
            fwd = (ub - ua) % sp.L
            start, length = (ua, fwd) if fwd <= sp.L - fwd else (ub, sp.L - fwd)
            links.append({"key": f"{net}:link{h.gap_index}", "net": net, "hop": h,
                          "ia": ia, "ib": ib, "start": start, "length": length,
                          "pad_num": ch.pad_num, "pad_xy": ch.hops[0].a_xy})
    if not links:
        return [], set(), {"links": 0}
    links.sort(key=lambda d: (d["length"], d["key"]))

    stub = {d["key"]: [_inside(sp, sp.vxy[d["ia"]], d["hop"].a_xy),
                       _inside(sp, sp.vxy[d["ib"]], d["hop"].b_xy)] for d in links}
    layer = _Layer(sp)
    paths, why = {}, {}
    for d in links:
        h = d["hop"]
        corridors = [c for d2 in links if d2["net"] != d["net"] and d2["key"] not in paths
                     for c in stub[d2["key"]]]
        free = layer.free([d["ia"], d["ib"]], corridors)

        def choose(fwd, back, d=d):
            def inside(pts):
                us = [sp.u(q) for q in pts[1:-1:max(1, len(pts) // 30)]] or [d["start"]]
                return sum(1 for u in us if (u - d["start"]) % sp.L <= d["length"] + 0.5) / len(us)
            return max((fwd, back), key=inside)
        arc = _walk(free, sp.vxy[d["ia"]], sp.vxy[d["ib"]], choose)
        if arc is None:
            why[d["key"]] = "the free space between its pieces has pinched shut"
            continue
        path = [tuple(sp.vxy[d["ia"]])] + arc + [tuple(sp.vxy[d["ib"]])]
        check = layer.ok(d["net"], corridors)
        bad = check(LineString(path))
        if bad:
            why[d["key"]] = "its walk " + bad
            continue
        path = _smooth(path, lambda seg: check(seg) is None)
        paths[d["key"]] = path
        layer.lay(d["net"], path)

    missing = {d["net"] for d in links if d["key"] not in paths}
    nets_done = {d["net"] for d in links} - missing
    routes = [Route(net_name=d["net"], pad_num=d["pad_num"],
                    anchor_xy=np.asarray(paths[d["key"]][-1], dtype=float),
                    # the net's pad, as for hug's link hops: a route whose first
                    # point IS its pad_xy is taken for a pad hop and extended
                    pad_xy=np.asarray(d["pad_xy"], dtype=float),
                    polyline_xy=np.asarray(paths[d["key"]], dtype=float))
              for d in links if d["key"] in paths and d["net"] in nets_done]
    info = {"links": len(links), "unplaced_links": len(links) - len(paths),
            "wire_model": ["hug"], "why": {d["net"]: why[d["key"]] for d in links if d["key"] in why},
            "nets_in2": len(nets_done), "nets_fallback_bcu": len(missing)}
    return routes, nets_done, info


# ==========================================================================
# In1: feeds
# ==========================================================================

def _bottom(sp, x_lo, x_hi, step=0.05):
    """the field's bottom profile: for each x, where a line rising from below
    first meets the field (inf where it misses)"""
    y0 = sp.region.bounds[1] - 1.0
    y1 = sp.region.bounds[3] + 1.0
    xs = np.arange(max(x_lo, sp.region.bounds[0]), min(x_hi, sp.region.bounds[2]) + 1e-9, step)
    yb = np.full(len(xs), np.inf)
    for k, x in enumerate(xs):
        hit = sp.region.intersection(LineString([(x, y0), (x, y1)]))
        if not hit.is_empty:
            yb[k] = float(hit.bounds[1])
    return xs, yb


def _slots(sp, xs, yb, rise):
    """Where a wire can come up into the field from below: points on the
    field's bottom edge whose straight rise through the first `rise` of the
    field clears every via — the gaps between the bottom row's vias. Returns
    [(x, y)] left to right.

    The bottom row's vias sit right above some pads, so a pad's own x is often
    not a slot: the wires fan from their pads to slots below the field. And a
    point in the thin channel UNDER a via is not a slot either: the first wire
    to reach it takes the channel, and the next one would have to climb out
    through its neighbour's entry gap."""
    out = []
    need = sp.a + _M + 1e-3
    for x, y in zip(xs, yb):
        if not np.isfinite(y):
            continue
        near = (np.abs(sp.vxy[:, 0] - x) < need) & (sp.vxy[:, 1] > y - need) & (sp.vxy[:, 1] < y + rise + need)
        if not near.any():
            out.append((float(x), float(y)))
    return out


def _fan_ok(start, slot, xs, yb, board=None):
    """the straight fan from `start` (below the field) up to `slot` stays
    under the field's bottom profile until it arrives — it must not cut
    through a lower step of the field on its way — and on the board (the
    drawn outline and the tail)"""
    (x0, y0), (x1, y1) = start, slot
    if abs(x1 - x0) > (y1 - y0) + 1e-9:
        return False          # no shallower than 45 degrees: up, not along
    if board is not None and not board.covers(LineString([start, slot])):
        return False
    if abs(x1 - x0) < 1e-9:
        return True
    lo, hi = min(x0, x1), max(x0, x1)
    m = (xs > lo + 1e-9) & (xs < hi - 1e-9) & (np.abs(xs - x1) > 0.026)
    if not m.any():
        return True
    yl = y0 + (xs[m] - x0) / (x1 - x0) * (y1 - y0)
    return bool(np.all(yl < yb[m] - 1e-6))


def _assign_slots(starts, cand, gap, xs, yb, lift, board=None, window=None):
    """Slots for wires whose fans start at `starts` (left to right): strictly
    left to right again, at least `gap` apart, each inside its window
    [lo, hi] of x, each fan clear of the field (_fan_ok), least total
    sideways shift. Plain DP over the candidates. None when they do not fit."""
    n, m = len(starts), len(cand)
    window = window or [(-np.inf, np.inf)] * n
    if n == 0:
        return [], 0.0
    if m < n:
        return None
    cx = np.array([c[0] for c in cand])
    INF = float("inf")
    def allowed(j, k):
        return (window[j][0] - 1e-9 <= cx[k] <= window[j][1] + 1e-9
                and _fan_ok(starts[j], (cand[k][0], cand[k][1] - lift), xs, yb, board))
    f = np.array([abs(cx[k] - starts[0][0]) if allowed(0, k) else INF for k in range(m)])
    back = []
    for j in range(1, n):
        g = np.full(m, INF)
        arg = np.full(m, -1)
        best, best_k, p = INF, -1, 0
        for k in range(m):
            while p < m and cx[p] <= cx[k] - gap + 1e-9:
                if f[p] < best:
                    best, best_k = f[p], p
                p += 1
            if best < INF and allowed(j, k):
                g[k] = best + abs(cx[k] - starts[j][0])
                arg[k] = best_k
        back.append(arg)
        f = g
    k = int(np.argmin(f))
    if not np.isfinite(f[k]):
        return None
    cost = float(f[k])
    pick = [k]
    for arg in reversed(back):
        k = int(arg[k])
        pick.append(k)
    return [cand[k] for k in reversed(pick)], cost


def route_feeds(layout, params, plan, layers=("In1.Cu",), preplaced=None):
    """Returns ({layer: [Route]}, failed_nets, info) — the route_rows_inner
    contract. Each Route runs pad -> tap via. Mutates layout.pad_to_net:
    placed rows get pads in peel order (failed rows keep theirs)."""
    rows = []
    for net, ch in plan.chains.items():
        if ch.face == "row":
            rows.append({"net": net, "chain": ch,
                         "pix": np.concatenate([np.asarray(sg, dtype=int) for sg in ch.segments])})
    info = {"rows": len(rows), "rows_failed": [], "fail_reasons": {}, "rows_bcu_fallback": 0,
            "per_layer": {L: 0 for L in layers}, "direct": 0, "channel_rows": 0, "arrival": 0,
            "fallback_ranks": [], "wire_model": {L: "hug" for L in layers}}
    if not rows:
        return {L: [] for L in layers}, set(), info
    sp = _Space(layout, params, plan.cut_xy)
    px, py = float(params.pitch_x_mm), float(params.pitch_y_mm)

    # both flanks of every row: the end via and the cell edge beyond it
    for r in rows:
        v = sp.vxy[r["pix"]]
        iL, iR = r["pix"][int(np.argmin(v[:, 0]))], r["pix"][int(np.argmax(v[:, 0]))]
        r["tap"] = {"cw": int(iL), "ccw": int(iR)}
        r["flank"] = {"cw": (float(sp.vxy[iL, 0]) - px / 2, float(sp.vxy[iL, 1])),
                      "ccw": (float(sp.vxy[iR, 0]) + px / 2, float(sp.vxy[iR, 1]))}
        r["uR"], r["uL"] = sp.u(r["flank"]["ccw"]), sp.u(r["flank"]["cw"])

    # pads: every row pad (stub top for fanned-in pads), left to right
    pad_xy = {int(r["chain"].pad_num): np.asarray(r["chain"].hops[0].a_xy, dtype=float) for r in rows}
    stubs = getattr(plan, "pad_stubs", {}) or {}
    orig_pad = {r["net"]: int(r["chain"].pad_num) for r in rows}

    def side_split(rs):
        """Which way each row is fed. Counter-clockwise rows must all end
        before every clockwise row starts (spans (0, uR) and (uL, L)
        disjoint), so a threshold t splits them. For each t every row takes
        its shorter way — except that rows whose two ways are about as long
        (within a few rows) are shared out between the bundles, the lower ones
        counter-clockwise, so neither side's bundle fills the field alone (a T
        stem holds about half its rows a side). Least total contour wins, an
        unbalanced row counting as `tie` of contour."""
        tie = 4.0 * max(px, py)
        best = None
        for t in sorted({0.0, sp.L} | {r["uR"] for r in rs} | {r["uL"] for r in rs}):
            side, flex = {}, []
            for r in rs:
                ok_ccw, ok_cw = r["uR"] <= t + 1e-9, r["uL"] >= t - 1e-9
                c_ccw, c_cw = r["uR"], sp.L - r["uL"]
                if ok_ccw and ok_cw and abs(c_ccw - c_cw) <= tie:
                    flex.append(r)
                elif ok_ccw and (not ok_cw or c_ccw <= c_cw):
                    side[r["net"]] = "ccw"
                elif ok_cw:
                    side[r["net"]] = "cw"
                else:
                    break
            else:
                n_ccw = sum(1 for v in side.values() if v == "ccw")
                n_cw = len(side) - n_ccw
                flex.sort(key=lambda r: (float(sp.vxy[r["tap"]["cw"], 1]), r["net"]))
                k = min(range(len(flex) + 1), key=lambda k: abs((n_ccw + k) - (n_cw + len(flex) - k)))
                for i, r in enumerate(flex):
                    side[r["net"]] = "ccw" if i < k else "cw"
                cost = sum(r["uR"] if side[r["net"]] == "ccw" else sp.L - r["uL"] for r in rs)
                # a row a side is worth `tie` of contour: one only crosses over
                # when its two ways are about as long
                n_ccw = sum(1 for v in side.values() if v == "ccw")
                cost += tie * abs(2 * n_ccw - len(rs)) / 2
                if best is None or cost < best[0] - 1e-9:
                    best = (cost, side)
        return best[1] if best else {r["net"]: ("ccw" if r["uR"] <= sp.L - r["uL"] else "cw") for r in rs}

    # on a fixed outline the fan from a pad up to the field runs on the drawn
    # board — the outline and the tail, never beside a narrow stem — inside
    # the edge clearance. A growing board is grown round it later (its tail
    # does not exist yet), so there only the field-bottom rules apply.
    board_in = None
    if not params.auto_expand_board:
        parts = [q for q in (layout.outline_poly,
                             layout.cable_poly.buffer(0.01, join_style=2) if layout.cable_poly is not None else None)
                 if q is not None and not q.is_empty]
        if parts:
            board_in = unary_union(parts).buffer(-(params.trace_w_mm / 2 + params.board_edge_clear_mm) + 1e-3)

    # slots along the field's bottom edge, over the pads and a field's width
    # either side of them
    pxs = [float(q[0]) for q in pad_xy.values()]
    width = sp.region.bounds[2] - sp.region.bounds[0]
    xs, yb = _bottom(sp, min(pxs) - width, max(pxs) + width)
    # only the field's bottom over the connector, give or take a couple of
    # rows of a ragged edge — not the underside of a T's bar beside its stem
    if np.isfinite(yb).any():
        yb = np.where(yb <= np.min(yb) + 2.5 * py, yb, np.inf)
    cand = _slots(sp, xs, yb, rise=1.5 * py)
    gap = sp.s + _M + 1e-3
    lift = sp.a + _M            # the fan reaches a slot straight from below

    def fan_start(pnum):
        """where a pad's fan leaves for its slot: a fanned-in pad's stub top,
        else a short rise straight up off the pad, clear of its neighbours"""
        q = pad_xy[pnum]
        if int(pnum) in stubs:
            return (float(q[0]), float(q[1]))
        return (float(q[0]), float(q[1]) + 1.0)

    left = list(rows)
    routes = {L: [] for L in layers}
    failed, why, taken = set(), {}, set()
    for L in layers:
        if not left:
            break
        for _round in range(len(left) + 1):
            pool = sorted(p for p in pad_xy if p not in taken | {orig_pad[n] for n in failed})
            todo = [r for r in left if r["net"] not in failed]
            if len(todo) > len(pool):
                todo.sort(key=lambda r: min(r["uR"], sp.L - r["uL"]))
                for r in todo[len(pool):]:
                    failed.add(r["net"]); why[r["net"]] = "more rows than free connector pads"
                todo = todo[:len(pool)]
            side = side_split(todo)
            cw = sorted((r for r in todo if side[r["net"]] == "cw"), key=lambda r: sp.L - r["uL"])
            ccw = sorted((r for r in todo if side[r["net"]] == "ccw"), key=lambda r: r["uR"])
            pads_sorted = sorted(pool, key=lambda p: pad_xy[p][0])
            pad_of = {}
            for k, r in enumerate(cw):                 # innermost -> leftmost pad
                pad_of[r["net"]] = pads_sorted[k]
            for k, r in enumerate(ccw):                # innermost -> rightmost pad
                pad_of[r["net"]] = pads_sorted[len(pads_sorted) - 1 - k]
            span = {r["net"]: (r["uR"] if side[r["net"]] == "ccw" else sp.L - r["uL"]) for r in todo}
            order = sorted(todo, key=lambda r: (span[r["net"]], r["net"]))
            # the wires come up into the field in peel order, left to right
            exits = cw + ccw[::-1]
            # A tap ON the bottom edge, among the slots (a narrow stem's bottom
            # rows) has no place in the peel order: its wire just comes up at
            # the slot nearest its tap, and every other wire comes up clear of
            # the short stretch between that slot and that tap (crossing it
            # would cross the walk). The rest keep peel order around them, and
            # the pads go to all slots left to right, so no fan crosses.
            lo_x = min(c[0] for c in cand) if cand else 0.0
            hi_x = max(c[0] for c in cand) if cand else 0.0
            zone, rest = [], []
            for r in exits:
                tx, ty = sp.vxy[r["tap"][side[r["net"]]]]
                ybt = float(np.interp(tx, xs, np.where(np.isfinite(yb), yb, 1e9))) if len(xs) else 1e9
                (zone if lo_x <= tx <= hi_x and ty - ybt <= 1.5 * py else rest).append(r)
            slot_of, banned = {}, []
            for r in sorted(zone, key=lambda r: sp.vxy[r["tap"][side[r["net"]]]][0]):
                tx = float(sp.vxy[r["tap"][side[r["net"]]]][0])
                free_c = [c for c in cand if all(abs(c[0] - q[0]) >= gap for q in slot_of.values())
                          and not any(lo < c[0] < hi for lo, hi in banned)]
                if not free_c:
                    continue
                c = min(free_c, key=lambda c: abs(c[0] - tx))
                slot_of[r["net"]] = c
                banned.append((min(c[0], tx) - gap, max(c[0], tx) + gap))
            zone = [r for r in zone if r["net"] in slot_of]
            for r in exits:
                if r not in zone and r not in rest:
                    rest.append(r)            # a zone row that found no slot: peel order
            rest = [r for r in exits if r in rest]
            cand2 = [c for c in cand if not any(lo < c[0] < hi for lo, hi in banned)]
            # pads the zone wires will take (nearest their slots), so the rest
            # plan their fans from the pads they will roughly get
            pads_left = list(pads_sorted)
            for r in zone:
                pk = min(pads_left, key=lambda p: abs(pad_xy[p][0] - slot_of[r["net"]][0]))
                pads_left.remove(pk)
            starts = [fan_start(p) for p in pads_left[:len(rest)]]
            got = _assign_slots(starts, cand2, gap, xs, yb, lift, board_in)
            if got is None:
                worst = max(rest or todo, key=lambda r: span[r["net"]])
                failed.add(worst["net"]); why[worst["net"]] = "no room along the field's bottom edge to come up into it"
                continue
            slot_of.update({r["net"]: sl for r, sl in zip(rest, got[0])})
            # every pad to its slot, both left to right
            in_order = sorted(todo, key=lambda r: slot_of[r["net"]][0])
            pad_of = {r["net"]: p for r, p in zip(in_order, pads_sorted)}
            corridor = {r["net"]: _inside(sp, sp.vxy[r["tap"][side[r["net"]]]], r["flank"][side[r["net"]]])
                        for r in todo}
            layer = _Layer(sp, [(n, p) for n, p in (preplaced or {}).get(L, [])])
            placed, newly = {}, set()
            for r in order:
                net, sd = r["net"], side[r["net"]]
                tap = r["tap"][sd]
                corridors = [corridor[n] for n in corridor if n != net and n not in placed]
                free = layer.free([tap], corridors)
                pxy = pad_xy[pad_of[net]]
                ent = slot_of[net]
                # below the field: up off the pad (a fanned-in pad's stub
                # already did), then straight to its slot
                fan = [tuple(pxy)]
                if int(pad_of[net]) not in stubs:
                    fan.append(fan_start(pad_of[net]))
                fan.append((ent[0], ent[1] - lift))
                # which way round: from where the entry and the tap sit on
                # the contour (a tap can lie between the exit and its own slot)
                ue, ut = sp.u(ent), sp.u(sp.vxy[tap])
                if sd == "ccw" and ue > sp.L / 2:
                    ue -= sp.L
                if sd == "cw" and ue < sp.L / 2:
                    ue += sp.L
                fwd = ut > ue
                arc = _walk(free, ent, sp.vxy[tap], lambda f, b: f if fwd else b)
                if arc is None:
                    newly.add(net); why[net] = "the free space to its flank has pinched shut"
                    continue
                inner = [tuple(ent)] + arc[1:] + [tuple(sp.vxy[tap])]
                check = layer.ok(net, corridors)
                bad = check(LineString(inner))
                bad = "its walk " + bad if bad else None
                drop = LineString(fan + [tuple(ent)])
                if not bad and check(drop, in_region=False):
                    bad = "its drop to the pad " + check(drop, in_region=False)
                if not bad and board_in is not None and not board_in.covers(drop):
                    bad = "its drop to the pad leaves the board"
                if bad:
                    newly.add(net); why[net] = bad
                    continue
                inner = _smooth(inner, lambda seg: check(seg) is None)
                placed[net] = (pad_of[net], fan + inner)
                layer.lay(net, placed[net][1])
            if not newly:
                break
            failed |= newly            # their pads go back to them; route the rest again
        for net, (pnum, path) in placed.items():
            stub = stubs.get(int(pnum))
            pts = [tuple(q) for q in stub[:-1]] + path if stub else path
            routes[L].append(Route(net_name=net, pad_num=pnum,
                                   anchor_xy=np.asarray(path[-1], dtype=float),
                                   pad_xy=np.asarray(stub[0] if stub else path[0], dtype=float),
                                   polyline_xy=np.asarray(pts, dtype=float)))
            layout.pad_to_net[pnum] = net
            taken.add(int(pnum))
        left = [r for r in left if r["net"] in failed]
        if L != layers[-1]:
            failed = set()             # the next layer tries them afresh
    failed = {r["net"] for r in left}
    info.update(rows_failed=sorted(failed), fail_reasons={n: why.get(n, "no route") for n in failed},
                rows_bcu_fallback=len(failed), per_layer={L: len(routes[L]) for L in layers})
    return routes, failed, info
