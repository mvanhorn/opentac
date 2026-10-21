"""Row routing — greedy reactive router: each wire climbs from its pad and, at
each sensor-row edge it passes, is pulled toward its anchor as far as the
clearances allow."""

from dataclasses import replace

import numpy as np
import shapely
from shapely.geometry import Point
from shapely.affinity import scale

from .types import Layout, Route, Params
from .geometry import remove_collinear
from .route_columns import MAX_LSTEP_FIXES


def route_rows_grid(
    layout: Layout,
    outline_xy: np.ndarray,
    params: Params,
    external_skip_pix: set[int] | None = None,
) -> tuple[list[Route], set[int]]:
    """Route row nets. Returns (routes, removed_pixel_indices)."""
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


def make_row_entries_horizontal(routes: list[Route]) -> list[Route]:
    """Force generated row-route endpoint connections to enter anchors horizontally."""
    for route in routes:
        route.polyline_xy = _make_final_entry_horizontal(
            np.asarray(route.polyline_xy, dtype=float),
            np.asarray(route.anchor_xy, dtype=float),
        )
    return routes


def _route_once(layout, outline_xy, params, G, board_poly, skip_pix):
    """Route all rows once, return routes and violation pixels."""
    row_nets, pad_nums, anchor_xy, pad_xy, row_pixel_idx = _build_row_info(
        layout, outline_xy, params, skip_pix)
    row_sides = _row_route_sides(row_nets, row_pixel_idx, layout.placements_xy)
    N = len(row_nets)

    if N == 0:
        return [], set()

    left_idx = [i for i, side in enumerate(row_sides) if side == "left"]
    right_idx = [i for i, side in enumerate(row_sides) if side == "right"]
    layout.routing["row_routes_left"] = len(left_idx)
    layout.routing["row_routes_right"] = len(right_idx)

    routes: list[Route | None] = [None] * N
    viol_pix = set()

    if left_idx:
        left_routes, left_viol = _route_once_side(
            layout, outline_xy, params, G, board_poly, skip_pix,
            _subset_row_info(row_nets, pad_nums, anchor_xy, pad_xy, row_pixel_idx, left_idx),
        )
        for src_i, route in zip(left_idx, left_routes):
            routes[src_i] = route
        viol_pix |= left_viol

    if right_idx:
        mirrored_layout = replace(
            layout,
            placements_xy=_mirror_points(layout.placements_xy),
            board_poly=scale(board_poly, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0)),
        )
        right_info = _subset_row_info(
            row_nets, pad_nums,
            _mirror_points(anchor_xy), _mirror_points(pad_xy), row_pixel_idx, right_idx)
        right_routes, right_viol = _route_once_side(
            mirrored_layout, _mirror_points(outline_xy), params, G,
            mirrored_layout.board_poly, skip_pix, right_info,
        )
        for route in right_routes:
            route.anchor_xy = _mirror_points(route.anchor_xy)
            route.pad_xy = _mirror_points(route.pad_xy)
            route.polyline_xy = _mirror_points(route.polyline_xy)
        for src_i, route in zip(right_idx, right_routes):
            routes[src_i] = route
        viol_pix |= right_viol

    return [route for route in routes if route is not None], viol_pix


def _route_once_side(layout, outline_xy, params, G, board_poly, skip_pix, row_info):
    """Route a preselected row group with the left-side row router (the
    right-side group is passed in mirrored)."""
    row_nets, pad_nums, anchor_xy, pad_xy, row_pixel_idx = row_info
    N = len(row_nets)

    if N == 0:
        return [], set()

    Pxy = layout.placements_xy
    n_pix = len(Pxy)
    px_hw = params.pixel_w_mm / 2
    px_hh = params.pixel_h_mm / 2
    trace_w = params.trace_w_mm
    row_sensor_gap = max(G, 0.5)
    pad_y_mm = layout.pad_y_mm
    active_pix = set(range(n_pix)) - skip_pix

    # Pre-compute unique pixel-row y-levels
    active_y = Pxy[list(active_pix), 1]
    all_row_y = np.sort(np.unique(active_y))

    order = np.argsort(anchor_xy[:, 1])

    routes = [None] * N

    for idx in range(N):
        i = order[idx]
        ax, ay = anchor_xy[i]
        my_pix = row_pixel_idx[i]
        other_pix = active_pix - my_pix

        cur_x = pad_xy[i, 0]
        cur_y = pad_y_mm
        P = [[cur_x, cur_y]]

        for _iter in range(100):
            # === Go up at cur_x: find lowest horizontal blocker ===
            best_y = float('inf')
            found = False

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
                    if cur_x < xlo - 1e-9 or cur_x > xhi + 1e-9:
                        continue
                    if T < best_y:
                        best_y = T
                        found = True

            for pp in active_pix:
                bot_y = Pxy[pp, 1] - px_hh
                if bot_y <= cur_y + 1e-9:
                    continue
                if bot_y > ay + 1e-9:
                    continue
                xlo = Pxy[pp, 0] - px_hw
                xhi = Pxy[pp, 0] + px_hw
                if cur_x < xlo - 1e-9 or cur_x > xhi + 1e-9:
                    continue
                if bot_y < best_y:
                    best_y = bot_y
                    found = True

            if found:
                limit_y = max(best_y - row_sensor_gap, cur_y)
            else:
                limit_y = ay

            # === Walk up through pull-points, pulling at each ===
            pull_ys = list(all_row_y - px_hh)
            for jj in range(idx):
                k = order[jj]
                pull_ys.append(anchor_xy[k, 1] + 1e-6)
            pull_ys = sorted(set(pull_ys))

            for py_val in pull_ys:
                if py_val <= cur_y + 1e-9:
                    continue
                if py_val > limit_y + 1e-9:
                    break

                if abs(py_val - cur_y) > 1e-9:
                    P.append([cur_x, py_val])
                    cur_y = py_val

                target_x = ax
                if target_x <= cur_x + 1e-9:
                    continue

                pull_x = target_x

                # Check pixel left edges
                for pp in other_pix:
                    top_y = Pxy[pp, 1] + px_hh
                    bot_y2 = Pxy[pp, 1] - px_hh
                    if cur_y < bot_y2 - 1e-9 or cur_y > top_y + 1e-9:
                        continue
                    edge_x = Pxy[pp, 0] - px_hw
                    if edge_x > cur_x + 1e-9 and edge_x - row_sensor_gap < pull_x:
                        pull_x = edge_x - row_sensor_gap

                # Check other traces' vertical segments
                for jj in range(idx):
                    k = order[jj]
                    Pk = routes[k].polyline_xy
                    for s in range(len(Pk) - 1):
                        if abs(Pk[s][0] - Pk[s + 1][0]) > 1e-9:
                            continue
                        Vx = Pk[s][0]
                        ylo = min(Pk[s][1], Pk[s + 1][1])
                        yhi = max(Pk[s][1], Pk[s + 1][1])
                        if cur_y < ylo - 1e-9 or cur_y > yhi + 1e-9:
                            continue
                        if Vx > cur_x + 1e-9 and Vx - G < pull_x:
                            pull_x = Vx - G

                if pull_x > cur_x + 1e-9:
                    P.append([pull_x, cur_y])
                    cur_x = pull_x

                    # Re-check blockers at new cur_x
                    new_best_y = float('inf')
                    new_found = False
                    for jj2 in range(idx):
                        k2 = order[jj2]
                        Pk2 = routes[k2].polyline_xy
                        for s2 in range(len(Pk2) - 1):
                            if abs(Pk2[s2][1] - Pk2[s2 + 1][1]) > 1e-9:
                                continue
                            T2 = Pk2[s2][1]
                            xlo2 = min(Pk2[s2][0], Pk2[s2 + 1][0])
                            xhi2 = max(Pk2[s2][0], Pk2[s2 + 1][0])
                            if T2 <= cur_y + 1e-9:
                                continue
                            if T2 > ay + 1e-9:
                                continue
                            if cur_x < xlo2 - 1e-9 or cur_x > xhi2 + 1e-9:
                                continue
                            if T2 < new_best_y:
                                new_best_y = T2
                                new_found = True

                    for pp2 in active_pix:
                        bot_y2 = Pxy[pp2, 1] - px_hh
                        if bot_y2 <= cur_y + 1e-9:
                            continue
                        if bot_y2 > ay + 1e-9:
                            continue
                        xlo2 = Pxy[pp2, 0] - px_hw
                        xhi2 = Pxy[pp2, 0] + px_hw
                        if cur_x < xlo2 - 1e-9 or cur_x > xhi2 + 1e-9:
                            continue
                        if bot_y2 < new_best_y:
                            new_best_y = bot_y2
                            new_found = True

                    if new_found:
                        new_limit = max(new_best_y - row_sensor_gap, cur_y)
                        if new_limit < limit_y - 1e-9:
                            limit_y = new_limit
                            best_y = new_best_y
                            found = True

            if not found:
                if abs(cur_x - ax) > 1e-9:
                    P.append([ax, cur_y])
                    cur_x = ax
                if abs(cur_y - ay) > 1e-9:
                    P.append([ax, ay])
                break

            # Step up to limit_y
            if abs(limit_y - cur_y) > 1e-9:
                P.append([cur_x, limit_y])
                cur_y = limit_y

            # === Dodge left past blocker at best_y ===
            agg_xlo = float('inf')
            for pp in active_pix:
                bot_y = Pxy[pp, 1] - px_hh
                if abs(bot_y - best_y) > 1e-9:
                    continue
                agg_xlo = min(agg_xlo, Pxy[pp, 0] - px_hw)

            for jj in range(idx):
                k = order[jj]
                Pk = routes[k].polyline_xy
                for s in range(len(Pk) - 1):
                    if abs(Pk[s][1] - Pk[s + 1][1]) > 1e-9:
                        continue
                    if abs(Pk[s][1] - best_y) > 1e-9:
                        continue
                    agg_xlo = min(agg_xlo, min(Pk[s][0], Pk[s + 1][0]))

            dodge_x = agg_xlo - row_sensor_gap
            actual_x = dodge_x

            for pp in other_pix:
                top_y = Pxy[pp, 1] + px_hh
                bot_y = Pxy[pp, 1] - px_hh
                if cur_y < bot_y - 1e-9 or cur_y > top_y + 1e-9:
                    continue
                edge_x = Pxy[pp, 0] + px_hw
                if edge_x < cur_x - 1e-9 and edge_x + row_sensor_gap > actual_x:
                    actual_x = edge_x + row_sensor_gap

            for jj in range(idx):
                k = order[jj]
                Pk = routes[k].polyline_xy
                for s in range(len(Pk) - 1):
                    if abs(Pk[s][0] - Pk[s + 1][0]) > 1e-9:
                        continue
                    Vx = Pk[s][0]
                    ylo = min(Pk[s][1], Pk[s + 1][1])
                    yhi = max(Pk[s][1], Pk[s + 1][1])
                    if cur_y < ylo - 1e-9 or cur_y > yhi + 1e-9:
                        continue
                    if Vx < cur_x - 1e-9 and Vx + G > actual_x:
                        actual_x = Vx + G

            if abs(actual_x - cur_x) > 1e-9:
                P.append([actual_x, cur_y])
            cur_x = actual_x

            if abs(cur_x - ax) < 1e-9:
                P.append([ax, ay])
                break

        P = np.array(P)
        P = remove_collinear(P)
        P = _make_final_entry_horizontal(P, anchor_xy[i])

        routes[i] = Route(
            net_name=row_nets[i],
            pad_num=pad_nums[i],
            anchor_xy=anchor_xy[i],
            pad_xy=pad_xy[i],
            polyline_xy=P,
        )

    # Convert for L-step fix mutation
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

    # Convert back to numpy
    for ri in range(N):
        routes[ri].polyline_xy = np.array(routes[ri].polyline_xy)

    # === Boundary + edge proximity check ===
    #
    # Both passes below sample every route segment every 0.5mm and test each
    # sample against the board. Testing one point at a time -- Point(x, y)
    # then board_poly.contains(...) -- makes hundreds of thousands of Shapely
    # calls on a large board, each paying the per-call overhead of crossing
    # into GEOS. Sampling every segment up front and testing the whole array
    # with shapely.contains_xy gives the same result in vectorised calls.
    viol_pix = set()

    def _sample_points(P):
        """Every 0.5mm sample on a polyline, plus the index of its segment."""
        pts, seg_id = [], []
        for s in range(len(P) - 1):
            seg_len = float(np.linalg.norm(P[s + 1] - P[s]))
            n_samp = max(3, int(np.ceil(seg_len / 0.5)))
            t = np.arange(n_samp) / max(1, n_samp - 1)
            pts.append(P[s] + t[:, None] * (P[s + 1] - P[s]))
            seg_id.append(np.full(n_samp, s))
        if not pts:
            return np.zeros((0, 2)), np.zeros(0, dtype=int)
        return np.vstack(pts), np.concatenate(seg_id)

    # Pass 1: copper outside the board blames the nearest *other* row's pixel.
    for ri in range(N):
        pts, _ = _sample_points(routes[ri].polyline_xy)
        if len(pts) == 0:
            continue
        outside = ~shapely.contains_xy(board_poly, pts[:, 0], pts[:, 1])
        if not outside.any():
            continue
        other_active = active_pix - row_pixel_idx[ri]
        if not other_active:
            continue
        oa_list = list(other_active)
        oa_pts = Pxy[oa_list]
        for sx, sy in pts[outside]:
            dists = np.sum((oa_pts - [sx, sy]) ** 2, axis=1)
            viol_pix.add(oa_list[int(np.argmin(dists))])

    # Pass 2: copper inside the board but too close to its edge blames the
    # nearest active pixel. Note the sampling differs from pass 1 (n_samp + 1
    # points, t = qi / n_samp).
    boundary = board_poly.boundary          # computed once, not per sample
    edge_min = trace_w / 2 + params.board_edge_clear_mm
    all_active = list(active_pix)
    all_active_pts = Pxy[all_active] if all_active else np.zeros((0, 2))
    for ri in range(N):
        P = routes[ri].polyline_xy
        pts = []
        for s in range(len(P) - 1):
            n_samp = max(3, int(np.ceil(np.linalg.norm(P[s + 1] - P[s]) / 0.5)))
            t = np.arange(n_samp + 1) / n_samp
            pts.append(P[s] + t[:, None] * (P[s + 1] - P[s]))
        if not pts or not all_active:
            continue
        pts = np.vstack(pts)
        inside = shapely.contains_xy(board_poly, pts[:, 0], pts[:, 1])
        if not inside.any():
            continue
        cand = pts[inside]
        d = shapely.distance(boundary, shapely.points(cand))
        for pt in cand[d < edge_min]:
            dists = np.sum((all_active_pts - pt) ** 2, axis=1)
            viol_pix.add(all_active[int(np.argmin(dists))])

    return routes, viol_pix


def _subset_row_info(row_nets, pad_nums, anchor_xy, pad_xy, row_pixel_idx, indices):
    """Take a stable subset of row routing metadata."""
    return (
        [row_nets[i] for i in indices],
        [pad_nums[i] for i in indices],
        np.asarray(anchor_xy, dtype=float)[indices].copy(),
        np.asarray(pad_xy, dtype=float)[indices].copy(),
        [row_pixel_idx[i] for i in indices],
    )


def _mirror_points(points):
    """Mirror point array horizontally around the Y axis."""
    mirrored = np.asarray(points, dtype=float).copy()
    if mirrored.ndim == 1:
        mirrored[0] *= -1
    else:
        mirrored[:, 0] *= -1
    return mirrored


def _row_route_sides(row_nets, row_pixel_idx, placements_xy):
    """Split row nets by top-to-bottom order: top half left, bottom half right."""
    if not row_nets:
        return []

    mean_y = {}
    for i, net in enumerate(row_nets):
        pix_idx = list(row_pixel_idx[i])
        if pix_idx:
            mean_y[net] = float(np.mean(placements_xy[pix_idx, 1]))

    ordered = sorted(row_nets, key=lambda net: -mean_y.get(net, 0.0))
    top_count = (len(ordered) + 1) // 2
    left_nets = set(ordered[:top_count])
    return ["left" if net in left_nets else "right" for net in row_nets]


def _make_final_entry_horizontal(P, anchor):
    """Replace a final vertical row-node entry with a horizontal via entry."""
    if len(P) < 2:
        return P
    end = np.asarray(P[-1], dtype=float)
    if np.linalg.norm(end - anchor) > 1e-6:
        return P
    prev = np.asarray(P[-2], dtype=float)
    if abs(prev[1] - end[1]) < 1e-9:
        return P
    P2 = np.array(P, dtype=float)
    direction = -1.0 if prev[0] <= end[0] else 1.0
    P2[-2] = [prev[0] + direction * 0.1, end[1]]
    return remove_collinear(P2)


def _build_row_info(layout, outline_xy, params, skip_pix):
    """Extract row net info: nets, pad positions, anchors, pixel indices."""
    Pxy = layout.placements_xy
    net_px = layout.row_net_per_pixel
    pad_to_net = layout.pad_to_net
    conn_pos = layout.connector_pos
    pixel_w = params.pixel_w_mm
    pad_pitch = params.pad_pitch_mm
    pad_y = layout.pad_y_mm
    edge_clear = params.edge_clear_mm
    via_dia = params.via_dia_mm

    x_mid = np.mean(outline_xy[:2, 0])
    pad_x_left = x_mid - (conn_pos - 1) * pad_pitch / 2

    row_nets = []
    row_pads = []
    for pad_num, net_name in pad_to_net.items():
        if net_name.startswith("ROW_"):
            row_nets.append(net_name)
            row_pads.append(pad_num)

    if not row_nets:
        return [], [], np.zeros((0, 2)), np.zeros((0, 2)), []

    sort_idx = np.argsort(row_pads)
    row_pads = [row_pads[i] for i in sort_idx]
    row_nets = [row_nets[i] for i in sort_idx]

    N = len(row_nets)
    pad_nums = row_pads[:]
    anchor_xy = np.zeros((N, 2))
    pad_xy = np.zeros((N, 2))
    row_pixel_idx = [set() for _ in range(N)]

    dead_rows = [False] * N
    for i in range(N):
        pix_idx = set(j for j, n in enumerate(net_px) if n == row_nets[i]) - skip_pix
        row_pixel_idx[i] = pix_idx
        if not pix_idx:
            dead_rows[i] = True
            continue
        pts = Pxy[list(pix_idx)]
        via_x_off = -pixel_w / 2 + edge_clear + via_dia / 2
        anchor_xy[i] = [np.min(pts[:, 0]) + via_x_off, np.mean(pts[:, 1])]
        pad_xy[i] = [pad_x_left + (row_pads[i] - 1) * pad_pitch, pad_y]

    keep = [i for i in range(N) if not dead_rows[i]]
    row_nets = [row_nets[i] for i in keep]
    pad_nums = [pad_nums[i] for i in keep]
    anchor_xy = anchor_xy[keep]
    pad_xy = pad_xy[keep]
    row_pixel_idx = [row_pixel_idx[i] for i in keep]

    row_sides = _row_route_sides(row_nets, row_pixel_idx, Pxy)
    for i, side in enumerate(row_sides):
        pts = Pxy[list(row_pixel_idx[i])]
        y_mid = np.mean(pts[:, 1])
        via_x_off = -pixel_w / 2 + edge_clear + via_dia / 2
        if side == "right":
            anchor_xy[i] = [np.max(pts[:, 0]) + via_x_off, y_mid]
        else:
            anchor_xy[i] = [np.min(pts[:, 0]) + via_x_off, y_mid]

    return row_nets, pad_nums, anchor_xy, pad_xy, row_pixel_idx
