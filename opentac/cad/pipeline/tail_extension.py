"""Post-generation custom tail extension helpers."""

import os
import re

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from .geometry import (
    extension_tail_poly, offset_path, remove_collinear, remove_collinear_closed, tail_offset_mm,
)


def tail_extension_from_layout(layout):
    ext = layout.tail.get("tail_extension_resolved") if layout.tail else None
    return ext if ext and ext.get("enabled") else None


def _unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _segment_normals(path):
    normals = []
    dirs = []
    for i in range(len(path) - 1):
        d = _unit(path[i + 1] - path[i])
        if d is None:
            return None, None
        dirs.append(d)
        normals.append(np.array([-d[1], d[0]], dtype=float))
    return dirs, normals


# A pad's lane: the path moved sideways by the pad's offset from the split point
# (shared with the corridor around the lanes, geometry.extension_tail_poly).
_offset_path = offset_path


def _largest_polygon(poly):
    if isinstance(poly, Polygon):
        return poly
    if hasattr(poly, "geoms"):
        polys = [g for g in poly.geoms if isinstance(g, Polygon) and not g.is_empty]
        if polys:
            return max(polys, key=lambda g: g.area)
    return Polygon()


def _active_split_pad(layout):
    row_pads = sorted(int(p) for p, net in layout.pad_to_net.items() if str(net).startswith("ROW_"))
    col_pads = sorted(int(p) for p, net in layout.pad_to_net.items() if str(net).startswith("COL_"))
    if row_pads and col_pads:
        return 0.5 * (max(row_pads) + min(col_pads))
    return (int(layout.connector_pos) + 1) / 2.0


def _route_pad_offset(route, layout, params, start, start_t):
    """Return the route lane offset from the active row/column split point."""
    pad_num = int(route.pad_num)
    split_pad = _active_split_pad(layout)
    return float((pad_num - split_pad) * params.pad_pitch_mm)


def _connector_drill_reach_mm(layout, connector_dir):
    """Return distance from the tail-side pad edge to the connector drill center."""
    if not connector_dir:
        return None
    mod_file = os.path.join(connector_dir, f"{int(layout.connector_pos)}_pos_0.5mm_FPC.kicad_mod")
    if not os.path.exists(mod_file):
        return None
    with open(mod_file, "r") as f:
        text = f.read()
    pad = _first_pad_block(text)
    if not pad:
        return None
    cy = pad["at_y"]
    sy = pad["size_y"]
    drill_y = cy + pad["drill_offset_y"]
    anchor_y = _tail_side_pad_edge_y(text, cy, sy)
    reach = anchor_y - drill_y
    return max(0.0, float(reach))


def _first_pad_block(text):
    m = re.search(r'\(pad\s+"1"\s+\S+\s+\S+.*?\n\t\)', text, re.DOTALL)
    if not m:
        m = re.search(r'\(pad\s+"\d+"\s+\S+\s+\S+.*?\n\t\)', text, re.DOTALL)
    if not m:
        return None
    block = m.group(0)
    at = re.search(r'\(at\s+([-\d.]+)\s+([-\d.]+)', block)
    size = re.search(r'\(size\s+([-\d.]+)\s+([-\d.]+)', block)
    offset = re.search(r'\(offset\s+([-\d.]+)\s+([-\d.]+)\)', block)
    if not (at and size):
        return None
    return {
        "at_y": float(at.group(2)),
        "size_y": float(size.group(2)),
        "drill_offset_y": float(offset.group(2)) if offset else 0.0,
    }


def _tail_side_pad_edge_y(text, pad_center_y, pad_size_y):
    open_y = _open_edge_mid_y(text)
    edge_a = pad_center_y - pad_size_y / 2
    edge_b = pad_center_y + pad_size_y / 2
    if open_y is None:
        return max(edge_a, edge_b)
    return edge_a if abs(edge_a - open_y) < abs(edge_b - open_y) else edge_b


def _open_edge_mid_y(text):
    segments = []
    for m in re.finditer(
        r'fp_line\s*\(\s*start\s+([-\d.]+)\s+([-\d.]+)\s*\)\s*'
        r'\(\s*end\s+([-\d.]+)\s+([-\d.]+)\s*\).*?Edge\.Cuts',
        text, re.DOTALL):
        segments.append(((float(m.group(1)), float(m.group(2))),
                         (float(m.group(3)), float(m.group(4)))))
    for m in re.finditer(
        r'fp_arc\s*\(\s*start\s+([-\d.]+)\s+([-\d.]+)\s*\)\s*'
        r'\(\s*mid\s+([-\d.]+)\s+([-\d.]+)\s*\)\s*'
        r'\(\s*end\s+([-\d.]+)\s+([-\d.]+)\s*\).*?Edge\.Cuts',
        text, re.DOTALL):
        segments.append(((float(m.group(1)), float(m.group(2))),
                         (float(m.group(5)), float(m.group(6)))))
    if not segments:
        return None

    nodes = []
    degree = []

    def node_index(pt):
        for i, existing in enumerate(nodes):
            if np.linalg.norm(np.array(pt) - np.array(existing)) < 0.01:
                return i
        nodes.append(pt)
        degree.append(0)
        return len(nodes) - 1

    for a, b in segments:
        ia = node_index(a)
        ib = node_index(b)
        degree[ia] += 1
        degree[ib] += 1

    open_nodes = [nodes[i] for i, d in enumerate(degree) if d == 1]
    if len(open_nodes) < 2:
        return None
    return float(np.mean([p[1] for p in open_nodes]))


def apply_tail_extension(layout, routes, params, connector_dir=""):
    """Move connector pads to the end of a custom tail path and extend routes."""
    ext = layout.tail.get("tail_extension") if layout.tail else None
    if not ext or ext.get("enabled") is False:
        layout.tail["tail_extension_resolved"] = {"enabled": False}
        return False

    path = np.array(ext.get("path") or [], dtype=float)
    if len(path) < 2:
        layout.tail["tail_extension_resolved"] = {"enabled": False}
        return False

    if np.any(np.linalg.norm(np.diff(path, axis=0), axis=1) < 1e-6):
        raise ValueError("Tail extension contains a zero-length segment.")

    dirs, normals = _segment_normals(path)
    if dirs is None:
        raise ValueError("Tail extension path is invalid.")

    resolved = layout.tail.get("tail_extension_resolved") if layout.tail else None
    true_start = path[0] if resolved and resolved.get("enabled") else (
        layout.tail.get("pad_split_center") or layout.tail.get("pad_row_center") if layout.tail else None
    )
    if true_start is not None:
        true_start = np.asarray(true_start, dtype=float)
        if true_start.shape == (2,):
            path = path.copy()
            path[0] = true_start
            dirs, normals = _segment_normals(path)
            if dirs is None:
                raise ValueError("Tail extension path is invalid.")

    start = path[0]
    end = path[-1]
    start_t = normals[0]
    end_t = normals[-1]
    end_n = dirs[-1]
    width = float(ext.get("width_mm") or layout.tail.get("width_mm") or 0.0)
    clearance = float(ext.get("clearance_mm") or 0.0)
    # Kept in sync by hand with export.connector_geometry.SUPPORTED_CONNECTOR_EXPORT_POS
    # (can't import it here: that module imports from this one).
    connector_export_enabled = int(layout.connector_pos) in {20, 30, 32, 40, 50, 60, 70, 80, 90, 100}
    drill_reach = _connector_drill_reach_mm(layout, connector_dir) if connector_export_enabled else None
    lane_end_reach = drill_reach if drill_reach is not None else max(0.05, min(0.15, params.trace_w_mm * 0.5))

    for route in routes:
        P = np.asarray(route.polyline_xy, dtype=float)
        # Only tracks that actually BEGIN at the connector belong to the tail
        # fan. The hug router emits several tracks per net: the pad hop starts
        # at the pad, but link tracks (bridging a concavity gap) start at a
        # sensor-field terminal — extending those bolted a full tail lane onto
        # the middle of the board. Legacy routes always start at their pad
        # (and re-application starts at the previously moved pad, which
        # pad_xy tracks too), so this skip changes nothing for them.
        if len(P) == 0 or np.linalg.norm(P[0] - np.asarray(route.pad_xy, dtype=float)) > 1.0:
            continue
        offset = _route_pad_offset(route, layout, params, start, start_t)
        lane = _offset_path(path, offset)
        if lane is None:
            raise ValueError("Tail extension path cannot be offset.")
        pad_edge = lane[-1]
        new_pad = pad_edge + lane_end_reach * end_n
        extension = np.vstack([new_pad, lane[::-1]])
        if len(P) > 0 and np.linalg.norm(P[0] - new_pad) < 0.25:
            P = P.copy()
            P[0] = new_pad
            route.pad_xy = new_pad
            route.polyline_xy = remove_collinear(P)
            continue
        if len(P) > 1 and np.linalg.norm(extension[-1] - P[0]) < 0.25:
            new_poly = np.vstack([extension, P[1:]])
        else:
            new_poly = np.vstack([extension, P])
        route.pad_xy = new_pad
        route.polyline_xy = remove_collinear(new_poly)

    # The custom tail corridor stops at the connector base; physical connector
    # outline/pads are stitched in during export when a supported footprint is present.
    # It runs over the lanes the way the tail strip it continues runs over the
    # pads: centred where that strip is (geometry.fit_tail_strip), which is off
    # the split point the lanes are laid out from when the used pads are.
    split_pad = _active_split_pad(layout)
    centre_pad = (int(layout.connector_pos) + 1) / 2.0
    corridor_offset = tail_offset_mm(layout.tail) + (centre_pad - split_pad) * params.pad_pitch_mm
    corridor = extension_tail_poly(path, width, clearance, layout.tail.get("connector_min_width_mm"),
                                   corridor_offset)
    expanded = unary_union([layout.board_poly, corridor])
    layout.board_poly = expanded
    final_poly = _largest_polygon(expanded)
    if not final_poly.is_empty:
        layout.final_outline_xy = remove_collinear_closed(np.array(final_poly.exterior.coords[:-1], dtype=float))

    pad1_world = end + (1.0 - split_pad) * params.pad_pitch_mm * end_t
    layout.pad_y_mm = float(end[1])
    layout.tail["tail_extension_resolved"] = {
        "enabled": True,
        "path": path.tolist(),
        "width_mm": width,
        "clearance_mm": clearance,
        "offset_mm": float(corridor_offset),
        "t_hat": end_t.tolist(),
        "n_out": end_n.tolist(),
        "pad1_world": pad1_world.tolist(),
        "pad_row_center": end.tolist(),
        "length_mm": float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1))),
        "connector_export_enabled": connector_export_enabled,
        "lane_end_reach_mm": lane_end_reach,
    }
    return True
