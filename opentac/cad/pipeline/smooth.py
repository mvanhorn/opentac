"""Route smoothing — collapse Manhattan staircases into long straight diagonals,
then chamfer any remaining sharp corners. Runs after routing + validation.

Two purely-geometric passes:

1. line-of-sight simplify (`los_simplify_layer`) — string-pulling. For each trace we
   replace the longest possible run of points with a single straight segment whenever
   that segment stays clear. In open space a whole staircase becomes one diagonal;
   where the path genuinely hugs pixels it stays stepped.
2. chamfer (`chamfer_layer`) — cut any leftover ~90° corners into 45° miters.

A candidate segment is accepted only if it keeps full clearance (>= G) from every
other trace on the same layer and this route's own non-replaced segments (so no real
trace short is ever created), stays inside the board, clears other-net pads, and does
not cut deep into pixel copper (shallow grazing is allowed — the ~0.1 mm tolerance the
router itself already uses). Checks run against every trace's current geometry, so the
passes never introduce a new trace-to-trace DRC violation.
"""

import numpy as np
from shapely.geometry import Point, LineString, box

from .geometry import seg_seg_dist, pt_seg_dist, remove_collinear

_PERP_DOT = 0.25   # |cos(angle)| below this counts as a ~90° corner
_PIX_ALLOW = 0.12  # mm a segment may sit inside pixel copper (router's own tolerance)


def taxel_copper_keepout_mm(params) -> float:
    """The pixel_keepout_mm for F.Cu (column) smoothing: it makes each pixel
    core the taxel's comb copper (its square inset by edge_clear_mm) grown by
    a wire's half width plus the clearance. On F.Cu both electrodes are combs,
    so the default core (grazing _PIX_ALLOW into the square) would let a
    column shortcut cut a neighbouring taxel's corner closer than the
    clearance to its row comb."""
    return (_PIX_ALLOW - params.edge_clear_mm + params.trace_w_mm / 2
            + params.clearance_mm + 0.01)


def _pixel_cores(layout, removed_pix, params, pixel_keepout_mm=0.0):
    Pxy = layout.placements_xy
    active = [i for i in range(len(Pxy)) if i not in removed_pix]
    hw = params.pixel_w_mm / 2
    hh = params.pixel_h_mm / 2
    inset = _PIX_ALLOW - pixel_keepout_mm
    return [(Pxy[i, 0] - hw + inset, Pxy[i, 1] - hh + inset,
             Pxy[i, 0] + hw - inset, Pxy[i, 1] + hh - inset) for i in active]


def _pad_positions(layout, outline_xy, params):
    conn_pos = layout.connector_pos
    pad_pitch = params.pad_pitch_mm
    x_mid = np.mean(outline_xy[:2, 0])
    pad_x_left = x_mid - (conn_pos - 1) * pad_pitch / 2
    pad_y = layout.pad_y_mm
    pts, nets = [], []
    for pad_num, net_name in layout.pad_to_net.items():
        pts.append(np.array([pad_x_left + (pad_num - 1) * pad_pitch, pad_y]))
        nets.append(net_name)
    return pts, nets


def _seg_clear(a, b, ri, polys, P_self, skip_lo, skip_hi, my_net,
               pix_core, board_poly, pad_pts, pad_nets, G):
    """True if straight segment a->b clears everything. Self segments with index
    in [skip_lo, skip_hi] are ignored (they are the ones being replaced / adjacent).

    board_poly here is the *safe* board (already eroded by the copper-to-edge
    clearance — see _board_safe), so plain containment of the shortcut is the
    clearance test. Testing raw containment instead would let a smoothed
    diagonal cut a corner to within a fraction of the clearance of the cut
    line: legal by containment, a DRC edge violation in fact.
    """
    if not board_poly.covers(LineString([a, b])):
        return False
    # Not too deep into pixel copper, or into an expanded pixel keepout.
    #
    # pix_core is a plain list of (xlo, ylo, xhi, yhi) tuples and never changes
    # during a pass. Building a shapely box() for every pixel on every call
    # would dominate the pass. A segment can only touch a box whose AABB the
    # segment's own AABB overlaps, so reject on cheap float comparisons first
    # and only build geometry for the handful that survive.
    seg_xlo, seg_xhi = (a[0], b[0]) if a[0] <= b[0] else (b[0], a[0])
    seg_ylo, seg_yhi = (a[1], b[1]) if a[1] <= b[1] else (b[1], a[1])
    candidate = None
    for (xlo, ylo, xhi, yhi) in pix_core:
        if xhi <= xlo or yhi <= ylo:
            continue
        if xhi < seg_xlo or xlo > seg_xhi or yhi < seg_ylo or ylo > seg_yhi:
            continue                       # AABBs disjoint -> cannot intersect
        if candidate is None:
            candidate = LineString([a, b])
        if candidate.intersects(box(xlo, ylo, xhi, yhi)):
            return False
    # Other-net pads
    for pt, net in zip(pad_pts, pad_nets):
        if net == my_net:
            continue
        if pt_seg_dist(pt, a, b) < G - 1e-9:
            return False
    # Other routes' current geometry
    for k, Pk in enumerate(polys):
        if k == ri:
            continue
        for s in range(len(Pk) - 1):
            if seg_seg_dist(a, b, Pk[s], Pk[s + 1]) < G - 1e-9:
                return False
    # Own non-replaced segments
    for s in range(len(P_self) - 1):
        if skip_lo <= s <= skip_hi:
            continue
        if seg_seg_dist(a, b, P_self[s], P_self[s + 1]) < G - 1e-9:
            return False
    return True


def _outer_row_fanout_nets(routes, layout):
    """Return the row nets whose early fanout should keep routed bundle shape."""
    row_means = {}
    for net in sorted({r.net_name for r in routes if r.net_name.startswith("ROW_")}):
        pix = [i for i, n in enumerate(layout.row_net_per_pixel) if n == net]
        if pix:
            row_means[net] = float(np.mean(layout.placements_xy[pix, 1]))

    if not row_means:
        row_means = {
            r.net_name: float(np.asarray(r.anchor_xy, dtype=float)[1])
            for r in routes
            if r.net_name.startswith("ROW_")
        }
    if not row_means:
        return set()

    # Tie-break on the net name. Without it, two rows at the same mean height
    # keep whatever order the set above happened to yield, which varies per
    # process because Python salts string hashing -- and since only the first and
    # last entries are protected from simplification, a tie flip changes which
    # trace gets smoothed. The same request then returns different copper
    # depending on which worker served it.
    ordered = sorted(row_means, key=lambda net: (-row_means[net], net))
    protected = {ordered[0]}
    if len(ordered) > 1:
        protected.add(ordered[-1])
    return protected


def _outer_col_fanout_nets(routes, layout):
    """Return the column nets whose early fanout should keep routed bundle shape."""
    col_means = {}
    for net in sorted({r.net_name for r in routes if r.net_name.startswith("COL_")}):
        pix = [i for i, n in enumerate(layout.col_net_per_pixel) if n == net]
        if pix:
            col_means[net] = float(np.mean(layout.placements_xy[pix, 0]))

    if not col_means:
        col_means = {
            r.net_name: float(np.asarray(r.anchor_xy, dtype=float)[0])
            for r in routes
            if r.net_name.startswith("COL_")
        }
    if not col_means:
        return set()

    ordered = sorted(col_means, key=lambda net: (col_means[net], net))   # see _outer_row_fanout_nets
    protected = {ordered[0]}
    if len(ordered) > 1:
        protected.add(ordered[-1])
    return protected


def _reject_row_fanout_shortcut(a, b, P, i, j, layout, params):
    if j <= i + 1:
        return False
    seg = b - a
    seg_len = float(np.linalg.norm(seg))
    trace_pitch = params.trace_w_mm + params.clearance_mm
    threshold = max(6.0, 4.0 * trace_pitch)
    if seg_len < threshold:
        return False
    n_out = np.asarray(layout.tail.get("n_out", [0.0, -1.0]), dtype=float)
    n_norm = float(np.linalg.norm(n_out))
    if n_norm < 1e-9:
        return False
    n_out = n_out / n_norm
    pad_center = np.asarray(layout.tail.get("pad_split_center") or layout.tail.get("pad_row_center") or [0.0, layout.pad_y_mm], dtype=float)
    depths = np.asarray([(p - pad_center) @ n_out for p in P], dtype=float)
    tail_reach = (
        float(layout.tail.get("length_mm", getattr(params, "cable_length_mm", 0.0)))
        + max(6.0, 12.0 * trace_pitch)
    )
    protected_prefix = np.where(np.abs(depths) <= tail_reach)[0]
    if len(protected_prefix) == 0:
        return False
    prefix_end = int(protected_prefix[-1])
    if i > prefix_end or j <= i + 1:
        return False
    skipped = np.asarray(P[i + 1:j], dtype=float)
    if len(skipped) == 0:
        return False
    line = LineString([a, b])
    max_dev = max(line.distance(Point(float(p[0]), float(p[1]))) for p in skipped)
    if max_dev <= max(0.6, 1.5 * trace_pitch):
        return False
    skipped_path = float(
        sum(np.linalg.norm(P[k + 1] - P[k]) for k in range(i, j))
    )
    return skipped_path - seg_len > max(1.0, 2.0 * trace_pitch)


def _board_safe(layout, params):
    """Board polygon eroded by the copper-to-edge clearance.

    Smoothing may only place copper where a trace centerline can legally sit,
    which is not the whole board: it has to stay trace_w/2 + board_edge_clear
    from the cut. Falls back to the raw board if the erosion collapses it (a
    very narrow tail), so smoothing degrades rather than dying.
    """
    clear = params.trace_w_mm / 2 + params.board_edge_clear_mm
    if clear <= 0:
        return layout.board_poly
    eroded = layout.board_poly.buffer(-clear, join_style='mitre')
    return eroded if not eroded.is_empty else layout.board_poly


def los_simplify_layer(
    routes,
    layout,
    removed_pix,
    outline_xy,
    params,
    passes=3,
    pixel_keepout_mm=0.0,
    protect_outer_row_fanout=False,
    protect_outer_col_fanout=False,
):
    """Collapse staircases into the longest clear straight segments (string-pulling)."""
    if not routes:
        return routes
    G = params.trace_w_mm + params.clearance_mm
    board_poly = _board_safe(layout, params)
    pix_core = _pixel_cores(layout, removed_pix, params, pixel_keepout_mm)
    pad_pts, pad_nets = _pad_positions(layout, outline_xy, params)

    protected_fanout_nets = set()
    if protect_outer_row_fanout:
        protected_fanout_nets |= _outer_row_fanout_nets(routes, layout)
    if protect_outer_col_fanout:
        protected_fanout_nets |= _outer_col_fanout_nets(routes, layout)
    polys = [np.asarray(r.polyline_xy, dtype=float) for r in routes]
    nets = [r.net_name for r in routes]

    for pass_i in range(passes):
        changed = False
        order = range(len(routes)) if pass_i % 2 == 0 else range(len(routes) - 1, -1, -1)
        for ri in order:
            P = polys[ri]
            m = len(P)
            if m < 3:
                continue
            keep = [0]
            i = 0
            while i < m - 1:
                best = i + 1  # original next point is always valid
                for j in range(m - 1, i + 1, -1):
                    if nets[ri] in protected_fanout_nets and _reject_row_fanout_shortcut(
                        P[i], P[j], P, i, j, layout, params):
                        continue
                    if _seg_clear(P[i], P[j], ri, polys, P, i - 1, j, nets[ri],
                                  pix_core, board_poly, pad_pts, pad_nets, G):
                        best = j
                        break
                keep.append(best)
                i = best
            newP = remove_collinear(P[keep])
            if len(newP) != len(P):
                changed = True
            polys[ri] = newP
        if not changed:
            break

    for ri, route in enumerate(routes):
        route.polyline_xy = polys[ri]
    return routes


def chamfer_layer(routes, layout, removed_pix, outline_xy, params,
                  max_miter_mm=2.5, passes=6, pixel_keepout_mm=0.0,
                  angle_dot_limit=_PERP_DOT,
                  fractions=(1.0, 0.7, 0.45, 0.25, 0.15, 0.1, 0.06)):
    """Cut leftover ~90° corners into 45° miters (multi-pass)."""
    if not routes:
        return routes
    G = params.trace_w_mm + params.clearance_mm
    board_poly = _board_safe(layout, params)
    pix_core = _pixel_cores(layout, removed_pix, params, pixel_keepout_mm)
    pad_pts, pad_nets = _pad_positions(layout, outline_xy, params)

    polys = [np.asarray(r.polyline_xy, dtype=float) for r in routes]
    nets = [r.net_name for r in routes]

    for pass_i in range(passes):
        changed = False
        order = range(len(routes)) if pass_i % 2 == 0 else range(len(routes) - 1, -1, -1)
        for ri in order:
            P = polys[ri]
            if len(P) < 3:
                continue
            result = [P[0]]
            for vi in range(1, len(P) - 1):
                p = np.asarray(result[-1], dtype=float)
                v = P[vi]
                nxt = P[vi + 1]
                din = v - p
                dout = nxt - v
                lin = np.linalg.norm(din)
                lout = np.linalg.norm(dout)
                if lin < 1e-9 or lout < 1e-9:
                    continue
                din = din / lin
                dout = dout / lout
                cross = din[0] * dout[1] - din[1] * dout[0]
                if abs(cross) < 1e-6:
                    result.append(v)
                    continue
                if abs(float(din @ dout)) > angle_dot_limit:
                    result.append(v)
                    continue
                accepted = False
                for frac in fractions:
                    d = min(max_miter_mm, 0.5 * lin, 0.5 * lout) * frac
                    if d < 1e-3:
                        break
                    a = v - d * din
                    b = v + d * dout
                    if _seg_clear(a, b, ri, polys, P, vi - 1, vi, nets[ri],
                                  pix_core, board_poly, pad_pts, pad_nets, G):
                        result.append(a)
                        result.append(b)
                        accepted = True
                        changed = True
                        break
                if not accepted:
                    result.append(v)
            result.append(P[-1])
            polys[ri] = remove_collinear(np.array(result, dtype=float))
        if not changed:
            break

    for ri, route in enumerate(routes):
        route.polyline_xy = polys[ri]
    return routes
