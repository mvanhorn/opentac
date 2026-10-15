"""Route validation — find incomplete/missing routes and pad crossings."""

import numpy as np
from .types import Layout, Route
from .geometry import pt_seg_dist


def validate_routes(
    layout: Layout,
    col_routes: list[Route],
    row_routes: list[Route],
    removed_pix: set[int],
    outline_xy: np.ndarray,
    pad_pitch_mm: float,
    trace_w_mm: float,
    clearance_mm: float,
) -> tuple[set[int], list[str]]:
    """Find pixels whose routes are missing, incomplete, or cross pads.

    Returns:
        remove_pix: pixel indices to remove
        incomplete_nets: net names that failed
    """
    tol = 0.5
    n_pix = len(layout.placements_xy)
    active_pix = set(range(n_pix)) - removed_pix
    G = trace_w_mm + clearance_mm

    active_col_nets = set(layout.col_net_per_pixel[i] for i in active_pix)
    active_row_nets = set(layout.row_net_per_pixel[i] for i in active_pix)

    # Build pad positions
    pad_to_net = layout.pad_to_net
    conn_pos = layout.connector_pos
    x_mid = np.mean(outline_xy[:2, 0])
    pad_x_left = x_mid - (conn_pos - 1) * pad_pitch_mm / 2
    pad_y_mm = layout.pad_y_mm

    pad_positions = {}  # pad_num -> (x, y)
    pad_nets = {}       # pad_num -> net_name
    for pad_num, net_name in pad_to_net.items():
        pad_positions[pad_num] = np.array([
            pad_x_left + (pad_num - 1) * pad_pitch_mm, pad_y_mm
        ])
        pad_nets[pad_num] = net_name

    bad_nets = set()

    # Check column route completeness
    routed_col_nets = set()
    for route in col_routes:
        P = route.polyline_xy
        if P is None or len(P) < 2:
            continue
        anchor = route.anchor_xy
        end_pt = P[-1]
        if abs(end_pt[0] - anchor[0]) < tol and end_pt[1] >= anchor[1] - tol:
            routed_col_nets.add(route.net_name)
        else:
            bad_nets.add(route.net_name)

    # Check row route completeness
    routed_row_nets = set()
    for route in row_routes:
        P = route.polyline_xy
        if P is None or len(P) < 2:
            continue
        anchor = route.anchor_xy
        end_pt = P[-1]
        if np.linalg.norm(end_pt - anchor) < tol:
            routed_row_nets.add(route.net_name)
        else:
            bad_nets.add(route.net_name)

    # Check for pad crossings
    body_below = 3.25
    all_routes = list(col_routes) + list(row_routes)

    for route in all_routes:
        P = route.polyline_xy
        net_name = route.net_name
        if P is None or len(P) == 0:
            continue
        already_bad = False

        for s in range(len(P) - 1):
            if already_bad:
                break
            seg_y_hi = max(P[s][1], P[s + 1][1])
            if seg_y_hi > pad_y_mm + 1:
                continue

            for pad_num, pad_net in pad_nets.items():
                if pad_net == net_name:
                    continue
                pad_pt = pad_positions[pad_num]
                d = pt_seg_dist(pad_pt, P[s], P[s + 1])
                if d < G:
                    bad_nets.add(net_name)
                    already_bad = True
                    break

    # Missing routes
    missing_col = active_col_nets - routed_col_nets
    missing_row = active_row_nets - routed_row_nets
    bad_nets |= missing_col | missing_row

    if not bad_nets:
        return set(), []

    # Find pixels on bad nets
    remove_pix = set()
    for net in bad_nets:
        if net.startswith("COL_"):
            pix = set(i for i, n in enumerate(layout.col_net_per_pixel) if n == net)
        else:
            pix = set(i for i, n in enumerate(layout.row_net_per_pixel) if n == net)
        pix &= active_pix
        remove_pix |= pix

    return remove_pix, list(bad_nets)
