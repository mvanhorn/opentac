"""Column routing — greedy multi-turn fan-out from pad to pixel anchor."""

import numpy as np
from shapely.geometry import Point

from .types import Layout, Route, Params

# Cap on the post-route L-step repair loop, per route.
#
# Each pass of that loop replaces one 2-point vertical segment with a 4-point
# L-jog, so the polyline *grows* every time it fires. If the new jog lands too
# close to yet another trace the next pass jogs again, and nothing else bounds
# the recursion: on a waisted outline (two lobes joined by a neck) an uncapped
# loop need not terminate, so /generate would never return.
#
# The horizontal-crossing repair loop further down this file has the same cap
# of 20, and route_rows.py's L-step loop shares this constant. Hitting the cap
# leaves the route as it stands, which the DRC/validate pass reports and the
# removal loop acts on -- the same contract as any other unroutable net.
MAX_LSTEP_FIXES = 20

from .geometry import remove_collinear


def route_columns_grid(
    layout: Layout,
    outline_xy: np.ndarray,
    params: Params,
    external_skip_pix: set[int] | None = None,
) -> tuple[list[Route], set[int]]:
    """Route column nets. Returns (routes, removed_pixel_indices)."""
    G = params.trace_w_mm + params.clearance_mm
    board_poly = layout.board_poly
    skip_pix = set(external_skip_pix) if external_skip_pix else set()
    if params.preserve_sensors:
        routes, viol_pix = _route_once(
            layout, outline_xy, params, G, board_poly, skip_pix)
        return routes, viol_pix

    max_removals = len(layout.placements_xy)

    for attempt in range(max_removals):
        routes, viol_pix = _route_once(
            layout, outline_xy, params, G, board_poly, skip_pix)
        if not viol_pix:
            break
        skip_pix |= viol_pix

    return routes, skip_pix


def _route_once(layout, outline_xy, params, G, board_poly, skip_pix):
    """Route all columns once, return routes and violation pixels."""
    col_nets, pad_nums, anchor_xy, pad_xy, col_pixel_idx = _build_col_info(
        layout, outline_xy, params, skip_pix)
    N = len(col_nets)

    if N == 0:
        return [], set()

    Pxy = layout.placements_xy
    n_pix = len(Pxy)
    px_hw = params.pixel_w_mm / 2
    px_hh = params.pixel_h_mm / 2
    trace_w = params.trace_w_mm
    pad_y_mm = layout.pad_y_mm
    active_pix = set(range(n_pix)) - skip_pix
    pixel_h = params.pixel_h_mm

    displacement = np.abs(anchor_xy[:, 0] - pad_xy[:, 0])
    order = np.argsort(displacement)

    routes = [None] * N

    for idx in range(N):
        i = order[idx]
        ax, ay = anchor_xy[i]
        my_pix = col_pixel_idx[i]
        other_pix = active_pix - my_pix

        cur_x = pad_xy[i, 0]
        cur_y = pad_y_mm
        P = [[cur_x, cur_y]]

        for _iter in range(100):
            # === Phase 1: go up at cur_x ===
            best_y = float('inf')
            best_xlo = float('-inf')
            best_xhi = float('inf')
            found = False

            my_xlo = min(cur_x, ax)
            my_xhi = max(cur_x, ax)

            # (a) Wire horizontal segments from already-routed traces
            for jj in range(idx):
                k = order[jj]
                Pk = routes[k].polyline_xy
                for s in range(len(Pk) - 1):
                    if abs(Pk[s][1] - Pk[s + 1][1]) > 1e-9:
                        continue
                    T = Pk[s][1]
                    xlo = min(Pk[s][0], Pk[s + 1][0])
                    xhi = max(Pk[s][0], Pk[s + 1][0])
                    if T <= cur_y + 1e-9:
                        continue
                    if T > ay + 1e-9:
                        continue
                    if my_xlo > xhi + 1e-9 or my_xhi < xlo - 1e-9:
                        continue
                    if T < best_y:
                        best_y = T
                        best_xlo = xlo
                        best_xhi = xhi
                        found = True

            # (b) Pixel bottom edges — offset by traceWmm for spine clearance
            for pp in other_pix:
                bot_y = Pxy[pp, 1] - px_hh - trace_w
                if bot_y <= cur_y + 1e-9:
                    continue
                if bot_y > ay + 1e-9:
                    continue
                xlo = Pxy[pp, 0] - px_hw
                xhi = Pxy[pp, 0] + px_hw
                if my_xlo > xhi + 1e-9 or my_xhi < xlo - 1e-9:
                    continue
                if bot_y < best_y:
                    best_y = bot_y
                    best_xlo = xlo
                    best_xhi = xhi
                    found = True

            if not found:
                if abs(cur_x - ax) > 1e-9:
                    # Check if horizontal jog at ay crosses any pixel
                    jog_y = ay
                    jog_xlo = min(cur_x, ax)
                    jog_xhi = max(cur_x, ax)
                    for pp in other_pix:
                        px_xlo = Pxy[pp, 0] - px_hw
                        px_xhi = Pxy[pp, 0] + px_hw
                        px_ylo = Pxy[pp, 1] - px_hh
                        px_yhi = Pxy[pp, 1] + px_hh
                        if jog_xhi < px_xlo - 1e-9 or jog_xlo > px_xhi + 1e-9:
                            continue
                        if jog_y > px_ylo - 1e-9 and jog_y < px_yhi + 1e-9:
                            jog_y = px_ylo - G
                    if abs(jog_y - ay) > 1e-9:
                        P.extend([[cur_x, jog_y], [ax, jog_y], [ax, ay]])
                    else:
                        P.extend([[cur_x, ay], [ax, ay]])
                else:
                    P.append([cur_x, ay])
                break

            turn_y = best_y - G
            P.append([cur_x, turn_y])

            if ax > cur_x:
                dodge_x = min(best_xhi + G, ax)
            elif ax < cur_x:
                dodge_x = max(best_xlo - G, ax)
            else:
                if (best_xhi - cur_x) < (cur_x - best_xlo):
                    dodge_x = best_xhi + G
                else:
                    dodge_x = best_xlo - G

            # === Phase 2: go horizontal — check vertical blockers ===
            d = 1 if dodge_x > cur_x else (-1 if dodge_x < cur_x else 0)
            actual_x = dodge_x

            if abs(d) > 0:
                # (a) Pixel side edges
                for pp in other_pix:
                    top_y = Pxy[pp, 1] + px_hh
                    bot_y = Pxy[pp, 1] - px_hh
                    if turn_y < bot_y - 1e-9 or turn_y > top_y + 1e-9:
                        continue
                    if d > 0:
                        edge_x = Pxy[pp, 0] - px_hw
                        if edge_x > cur_x + 1e-9 and edge_x - G < actual_x:
                            actual_x = edge_x - G
                    else:
                        edge_x = Pxy[pp, 0] + px_hw
                        if edge_x < cur_x - 1e-9 and edge_x + G > actual_x:
                            actual_x = edge_x + G

                # (b) Wire vertical segments at turn_y level
                for jj in range(idx):
                    k = order[jj]
                    Pk = routes[k].polyline_xy
                    for s in range(len(Pk) - 1):
                        if abs(Pk[s][0] - Pk[s + 1][0]) > 1e-9:
                            continue
                        Vx = Pk[s][0]
                        ylo = min(Pk[s][1], Pk[s + 1][1])
                        yhi = max(Pk[s][1], Pk[s + 1][1])
                        if turn_y < ylo - 1e-9 or turn_y > yhi + 1e-9:
                            continue
                        if d > 0:
                            if Vx > cur_x + 1e-9 and Vx - G < actual_x:
                                actual_x = Vx - G
                        else:
                            if Vx < cur_x - 1e-9 and Vx + G > actual_x:
                                actual_x = Vx + G

                # (c) Post-dodge clearance push
                for jj in range(idx):
                    k = order[jj]
                    Pk = routes[k].polyline_xy
                    for s in range(len(Pk) - 1):
                        if abs(Pk[s][0] - Pk[s + 1][0]) > 1e-9:
                            continue
                        Vx = Pk[s][0]
                        ylo = min(Pk[s][1], Pk[s + 1][1])
                        yhi = max(Pk[s][1], Pk[s + 1][1])
                        if yhi < turn_y - G or ylo > ay + G:
                            continue
                        if abs(Vx - actual_x) < G - 1e-9:
                            if d > 0:
                                actual_x = max(actual_x, Vx + G)
                            else:
                                actual_x = min(actual_x, Vx - G)

            P.append([actual_x, turn_y])
            cur_x = actual_x
            cur_y = turn_y

            if abs(cur_x - ax) < 1e-9:
                P.append([ax, ay])
                break

        # Append final vertical segment up into the pixel spine
        spine_entry_y = anchor_xy[i, 1] + pixel_h / 2
        P.append([anchor_xy[i, 0], spine_entry_y])

        P = np.array(P)
        P = remove_collinear(P)

        routes[i] = Route(
            net_name=col_nets[i],
            pad_num=pad_nums[i],
            anchor_xy=anchor_xy[i],
            pad_xy=pad_xy[i],
            polyline_xy=P,
        )

    # Convert polyline_xy to list-of-lists for L-step fix mutation
    for ri in range(N):
        routes[ri].polyline_xy = routes[ri].polyline_xy.tolist()

    # === Post-route: L-step fix for vertical clearance violations ===
    for idx in range(1, N):
        i = order[idx]
        fix_done = True
        n_lstep = 0
        while fix_done and n_lstep < MAX_LSTEP_FIXES:
            fix_done = False
            n_lstep += 1
            Pi = routes[i].polyline_xy
            for si in range(len(Pi) - 1):
                if abs(Pi[si][0] - Pi[si + 1][0]) > 1e-9:
                    continue
                xi = Pi[si][0]
                ylo_i = min(Pi[si][1], Pi[si + 1][1])
                yhi_i = max(Pi[si][1], Pi[si + 1][1])

                for jj in range(idx):
                    j = order[jj]
                    Pj = routes[j].polyline_xy
                    for sj in range(len(Pj) - 1):
                        if abs(Pj[sj][0] - Pj[sj + 1][0]) > 1e-9:
                            continue
                        xj = Pj[sj][0]
                        ylo_j = min(Pj[sj][1], Pj[sj + 1][1])
                        yhi_j = max(Pj[sj][1], Pj[sj + 1][1])
                        if yhi_j < ylo_i - 1e-9 or ylo_j > yhi_i + 1e-9:
                            continue
                        if abs(xi - xj) >= G - 1e-9:
                            continue

                        new_x = xj + G if xi >= xj else xj - G

                        overlap_y = max(ylo_i, ylo_j)
                        split_y = overlap_y - G
                        seg_lo = min(Pi[si][1], Pi[si + 1][1])
                        seg_hi = max(Pi[si][1], Pi[si + 1][1])
                        split_y = max(split_y, seg_lo)
                        split_y = min(split_y, seg_hi)

                        # Avoid pixel copper
                        xlo_jog = min(xi, new_x)
                        xhi_jog = max(xi, new_x)
                        for pp in active_pix:
                            px_xlo = Pxy[pp, 0] - px_hw
                            px_xhi = Pxy[pp, 0] + px_hw
                            px_ylo = Pxy[pp, 1] - px_hh
                            px_yhi = Pxy[pp, 1] + px_hh
                            if xhi_jog < px_xlo - 1e-9 or xlo_jog > px_xhi + 1e-9:
                                continue
                            if split_y > px_ylo - 1e-9 and split_y < px_yhi + 1e-9:
                                split_y = px_ylo - G
                        split_y = max(split_y, seg_lo)
                        split_y = min(split_y, seg_hi)

                        y_lo = Pi[si][1]
                        y_hi = Pi[si + 1][1]
                        replacement = [[xi, y_lo], [xi, split_y], [new_x, split_y], [new_x, y_hi]]
                        Pi = Pi[:si] + replacement + Pi[si + 2:]
                        Pi = remove_collinear(np.array(Pi)).tolist()
                        routes[i].polyline_xy = Pi
                        fix_done = True
                        break
                    if fix_done:
                        break
                if fix_done:
                    break

    # === Post-route: fix horizontal segments that cross pixel copper ===
    for idx in range(N):
        i = order[idx]
        my_pix = col_pixel_idx[i]
        other_active_pix = active_pix - my_pix
        changed = True
        max_fix = 20
        n_fix = 0
        while changed and n_fix < max_fix:
            changed = False
            Pi = routes[i].polyline_xy
            for si in range(len(Pi) - 1):
                if abs(Pi[si][1] - Pi[si + 1][1]) > 1e-9:
                    continue
                seg_y = Pi[si][1]
                seg_xlo = min(Pi[si][0], Pi[si + 1][0])
                seg_xhi = max(Pi[si][0], Pi[si + 1][0])

                for pp in other_active_pix:
                    px_xlo = Pxy[pp, 0] - px_hw
                    px_xhi = Pxy[pp, 0] + px_hw
                    px_ylo = Pxy[pp, 1] - px_hh
                    px_yhi = Pxy[pp, 1] + px_hh
                    if seg_xhi < px_xlo - 1e-9 or seg_xlo > px_xhi + 1e-9:
                        continue
                    if seg_y < px_ylo - 1e-9 or seg_y > px_yhi + 1e-9:
                        continue
                    safe_y = px_ylo - G
                    Pi[si][1] = safe_y
                    Pi[si + 1][1] = safe_y
                    Pi = remove_collinear(np.array(Pi)).tolist()
                    routes[i].polyline_xy = Pi
                    changed = True
                    n_fix += 1
                    break
                if changed:
                    break

    # Convert back to numpy arrays
    for ri in range(N):
        routes[ri].polyline_xy = np.array(routes[ri].polyline_xy)

    # === Violation check ===
    viol_pix = set()
    for ri in range(N):
        P = routes[ri].polyline_xy
        my_pix = col_pixel_idx[ri]
        other_active = active_pix - my_pix

        # (1) Boundary check
        for s in range(len(P) - 1):
            seg_len = np.linalg.norm(P[s + 1] - P[s])
            n_samp = max(3, int(np.ceil(seg_len / 0.5)))
            for q in range(n_samp):
                t = q / max(1, n_samp - 1)
                sx = P[s, 0] + t * (P[s + 1, 0] - P[s, 0])
                sy = P[s, 1] + t * (P[s + 1, 1] - P[s, 1])
                if board_poly.contains(Point(sx, sy)):
                    continue
                if not other_active:
                    continue
                dists = np.sum((Pxy[list(other_active)] - [sx, sy]) ** 2, axis=1)
                mi = np.argmin(dists)
                viol_pix.add(list(other_active)[mi])

        # (2) Incomplete route
        end_pt = P[-1]
        spine_top = np.array([anchor_xy[ri, 0], anchor_xy[ri, 1] + pixel_h / 2])
        dist_to_anchor = np.linalg.norm(end_pt - anchor_xy[ri])
        dist_to_spine = np.linalg.norm(end_pt - spine_top)
        if min(dist_to_anchor, dist_to_spine) > G:
            if other_active:
                dists = np.sum((Pxy[list(other_active)] - anchor_xy[ri]) ** 2, axis=1)
                mi = np.argmin(dists)
                viol_pix.add(list(other_active)[mi])

        # (3) Edge proximity against the final routable board boundary
        for s in range(len(P) - 1):
            n_samp = max(3, int(np.ceil(np.linalg.norm(P[s + 1] - P[s]) / 0.5)))
            for qi in range(n_samp + 1):
                t = qi / n_samp
                pt = P[s] + t * (P[s + 1] - P[s])
                pt_obj = Point(pt[0], pt[1])
                if not board_poly.contains(pt_obj):
                    continue
                d = board_poly.boundary.distance(pt_obj)
                if d < trace_w / 2 + params.board_edge_clear_mm:
                    all_active = list(active_pix)
                    dists = np.sum((Pxy[all_active] - pt) ** 2, axis=1)
                    mi = np.argmin(dists)
                    viol_pix.add(all_active[mi])

    return routes, viol_pix


def _build_col_info(layout, outline_xy, params, skip_pix):
    """Extract column net info: nets, pad positions, anchors, pixel indices."""
    Pxy = layout.placements_xy
    net_px = layout.col_net_per_pixel
    pad_to_net = layout.pad_to_net
    conn_pos = layout.connector_pos
    trace_w = params.trace_w_mm
    pixel_w = params.pixel_w_mm
    pixel_h = params.pixel_h_mm
    pad_pitch = params.pad_pitch_mm
    pad_y = layout.pad_y_mm
    edge_clear = params.edge_clear_mm

    x_mid = np.mean(outline_xy[:2, 0])
    pad_x_left = x_mid - (conn_pos - 1) * pad_pitch / 2

    col_nets = []
    col_pads = []
    for pad_num, net_name in pad_to_net.items():
        if net_name.startswith("COL_"):
            col_nets.append(net_name)
            col_pads.append(pad_num)

    if not col_nets:
        return [], [], np.zeros((0, 2)), np.zeros((0, 2)), []

    # Sort by pad number
    sort_idx = np.argsort(col_pads)
    col_pads = [col_pads[i] for i in sort_idx]
    col_nets = [col_nets[i] for i in sort_idx]

    N = len(col_nets)
    pad_nums = col_pads[:]
    anchor_xy = np.zeros((N, 2))
    pad_xy = np.zeros((N, 2))
    col_pixel_idx = [set() for _ in range(N)]

    dead_cols = [False] * N
    for i in range(N):
        pix_idx = set(j for j, n in enumerate(net_px) if n == col_nets[i]) - skip_pix
        col_pixel_idx[i] = pix_idx
        if not pix_idx:
            dead_cols[i] = True
            continue
        pts = Pxy[list(pix_idx)]
        spine_x_off = pixel_w / 2 - edge_clear - trace_w / 2
        anchor_xy[i] = [np.mean(pts[:, 0]) + spine_x_off,
                        np.min(pts[:, 1]) - pixel_h / 2 - trace_w]
        pad_xy[i] = [pad_x_left + (col_pads[i] - 1) * pad_pitch, pad_y]

    # Filter dead columns
    keep = [i for i in range(N) if not dead_cols[i]]
    col_nets = [col_nets[i] for i in keep]
    pad_nums = [pad_nums[i] for i in keep]
    anchor_xy = anchor_xy[keep]
    pad_xy = pad_xy[keep]
    col_pixel_idx = [col_pixel_idx[i] for i in keep]

    return col_nets, pad_nums, anchor_xy, pad_xy, col_pixel_idx
