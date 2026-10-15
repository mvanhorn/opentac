"""Direct Gerber RS-274X file generation."""

import os
import re
import numpy as np
from opentac.cad.pipeline.types import Layout, Route, Params
from opentac.cad.pipeline.geometry import contiguous_runs
from opentac.cad.pipeline.tail_extension import tail_extension_from_layout
from opentac.cad.export.kicad_footprint import pixel_geometry
from opentac.cad.export.connector_geometry import (
    connector_edge_geometries,
    connector_pad_geometries,
    connector_user_geometries,
    stitched_outline_segments,
)
from opentac.cad.export.silk_label import label_strokes, silk_label
from opentac.cad.export.silk_backmark import backmark


def _coord(v_mm: float) -> int:
    """Convert mm to Gerber integer (format 4.6: 1mm = 1000000)."""
    return round(v_mm * 1_000_000)


def _write_header(lines, layer_name):
    """Write Gerber header."""
    lines.append(f"G04 Layer: {layer_name}*")
    lines.append("%FSLAX46Y46*%")
    lines.append("%MOMM*%")
    lines.append("%LPD*%")


def _write_footer(lines):
    lines.append("M02*")


class ApertureManager:
    """Manage Gerber aperture definitions."""

    def __init__(self):
        self._apertures = {}  # key -> D-code
        self._defs = []
        self._next_d = 10

    def circle(self, diameter: float) -> int:
        key = f"C,{diameter:.6f}"
        if key not in self._apertures:
            d = self._next_d
            self._next_d += 1
            self._apertures[key] = d
            self._defs.append(f"%ADD{d}C,{diameter:.6f}*%")
        return self._apertures[key]

    def rect(self, w: float, h: float) -> int:
        key = f"R,{w:.6f}X{h:.6f}"
        if key not in self._apertures:
            d = self._next_d
            self._next_d += 1
            self._apertures[key] = d
            self._defs.append(f"%ADD{d}R,{w:.6f}X{h:.6f}*%")
        return self._apertures[key]

    def definitions(self):
        return self._defs


def _draw_line(lines, x1, y1, x2, y2):
    """Draw a line segment (D01 interpolation)."""
    lines.append(f"X{_coord(x1)}Y{_coord(-y1)}D02*")
    lines.append(f"X{_coord(x2)}Y{_coord(-y2)}D01*")


def _flash(lines, x, y):
    """Flash aperture at position."""
    lines.append(f"X{_coord(x)}Y{_coord(-y)}D03*")


def generate_gerbers(
    layout: Layout,
    outline_xy: np.ndarray,
    col_routes: list[Route],
    row_routes: list[Route],
    removed_pix: set[int],
    params: Params,
    connector_dir: str = "",
    manifest_json: str | None = None,
) -> dict[str, str]:
    """Generate all Gerber layer files. Returns {filename: content}.

    `manifest_json` is optional; supplying it adds board-B_Silkscreen.gbo with
    the taxel-core QR and registration grid. This is the pure-Python fallback
    path -- when kicad-cli is available the same marks come out of the
    .kicad_pcb instead, which is why both exporters read the same
    silk_backmark geometry.
    """
    files = {}

    active_pix = sorted(set(range(len(layout.placements_xy))) - removed_pix)
    Pxy = layout.placements_xy
    g = pixel_geometry(params.pixel_w_mm, params.pixel_h_mm, params.trace_w_mm,
                       params.gap_mm, params.center_clear_mm, params.edge_clear_mm,
                       params.via_dia_mm)
    connector_pads = connector_pad_geometries(layout, outline_xy, params, connector_dir)

    # F.Cu — front copper
    files["board-F_Cu.gtl"] = _generate_copper_layer(
        "F.Cu", col_routes, layout, active_pix, Pxy, g, params, is_front=True,
        connector_pads=connector_pads)

    # B.Cu — back copper (the row wires that are not on an inner layer)
    bcu_routes = [r for r in row_routes if (getattr(r, "layer", None) or "B.Cu") == "B.Cu"]
    files["board-B_Cu.gbl"] = _generate_copper_layer(
        "B.Cu", bcu_routes, layout, active_pix, Pxy, g, params, is_front=False,
        connector_pads=connector_pads)

    # In1.Cu / In2.Cu — a 4-layer board's inner row wiring, with every
    # through-hole pad (taxel vias, connector pins) as KiCad plots them
    from opentac.cad.pipeline.route_4layer import copper_layers_of
    if copper_layers_of(params, row_routes) == 4:
        for layer, fname in (("In1.Cu", "board-In1_Cu.g1"), ("In2.Cu", "board-In2_Cu.g2")):
            files[fname] = _generate_copper_layer(
                layer, [r for r in row_routes if getattr(r, "layer", None) == layer],
                layout, active_pix, Pxy, g, params, is_front=False,
                connector_pads=connector_pads, buses=False)

    # Edge.Cuts — board outline + connector body
    files["board-Edge_Cuts.gm1"] = _generate_edge_cuts(layout, outline_xy, params, connector_dir)

    # Mask openings over pixel pads and connector pads
    files["board-F_Mask.gts"] = _generate_mask(
        "F.Mask", layout, active_pix, Pxy, g, params, connector_pads, include_pixels=True)
    files["board-B_Mask.gbs"] = _generate_mask(
        "B.Mask", layout, active_pix, Pxy, g, params, connector_pads, include_pixels=False)

    # F.Silkscreen -- the design name, so a fabricated board says what it is.
    silk = _generate_front_silk(layout, outline_xy, params, connector_dir)
    if silk:
        files["board-F_Silkscreen.gto"] = silk

    # B.Silkscreen -- taxel-core QR + registration grid, so a board separated
    # from its manifest still says what it is and can be measured from a photo.
    back_silk = _generate_back_silk(manifest_json)
    if back_silk:
        files["board-B_Silkscreen.gbo"] = back_silk

    dwgs_user = _generate_dwgs_user(layout, outline_xy, params, connector_dir)
    if dwgs_user:
        files["board-User_Drawings.gbr"] = dwgs_user
    cmts_user = _generate_cmts_user(layout, outline_xy, params, connector_dir)
    if cmts_user:
        files["board-User_Comments.gbr"] = cmts_user

    return files


def _generate_copper_layer(layer_name, routes, layout, active_pix, Pxy, g, params, is_front, connector_pads=None,
                           buses=True):
    """Generate copper layer Gerber. buses=False (inner layers): no row or
    column bus lines, only pads, vias and the given routes."""
    lines = []
    _write_header(lines, layer_name)

    am = ApertureManager()
    trace_ap = am.circle(params.trace_w_mm)

    # Pixel pads — draw interdigitated fingers
    for k in active_pix:
        cx, cy = Pxy[k]
        if is_front:
            # Pad 2 (col net) — right spine + odd fingers
            _draw_pixel_pad(lines, am, cx, cy, g, params.trace_w_mm, pad_num=2)
            # Pad 1 (row net) — left spine + even fingers
            _draw_pixel_pad(lines, am, cx, cy, g, params.trace_w_mm, pad_num=1)
        else:
            # Via pad on B.Cu
            via_ap = am.circle(params.via_dia_mm)
            lines.extend(am.definitions()[-1:])  # ensure def written
            lines.append(f"D{via_ap}*")
            _flash(lines, cx + g.via_x, cy + g.via_y)

    # Write all aperture definitions at top
    header_lines = []
    _write_header(header_lines, layer_name)
    header_lines.extend(am.definitions())
    lines = header_lines + lines[len([l for l in lines if l.startswith("G04") or l.startswith("%")])+1:]

    # Rebuild properly
    lines = []
    _write_header(lines, layer_name)
    lines.extend(am.definitions())

    # Draw pixel pads
    trace_ap = am.circle(params.trace_w_mm)
    lines.append(f"D{trace_ap}*")

    for k in active_pix:
        cx, cy = Pxy[k]
        if is_front:
            _draw_pixel_copper(lines, cx, cy, g, params.trace_w_mm)
        else:
            via_ap = am.circle(params.via_dia_mm)
            lines.append(f"D{via_ap}*")
            _flash(lines, cx + g.via_x, cy + g.via_y)

    # Connector pad copper, including through-hole rectangular terminals.
    # Use rectangular flashes instead of filled regions; region polygons can
    # produce viewer-dependent wedge fills around the pinout.
    _flash_connector_pads(lines, am, connector_pads or [], layer_name)

    # Route traces
    lines.append(f"D{trace_ap}*")
    for route in routes:
        P = route.polyline_xy
        if P is None or len(P) < 2:
            continue
        for s in range(len(P) - 1):
            _draw_line(lines, P[s][0], P[s][1], P[s + 1][0], P[s + 1][1])

    # Bus lines
    if not buses:
        pass
    elif is_front:
        unique_nets = set(layout.col_net_per_pixel[i] for i in active_pix)
        for cnet in sorted(unique_nets):
            pix_idx = [i for i in active_pix if layout.col_net_per_pixel[i] == cnet]
            if len(pix_idx) < 2:
                continue
            pts = Pxy[pix_idx]
            pts = pts[np.argsort(pts[:, 1])]
            # One bus per contiguous pixel run — a joined net (hug router) can
            # span a concavity gap; a single min-to-max bus would jump across.
            for run in contiguous_runs(pts[:, 1], params.pitch_y_mm * 1.25):
                if len(run) < 2:
                    continue
                seg = pts[run]
                bus_x = np.mean(seg[:, 0]) + g.xR
                _draw_line(lines, bus_x, seg[0, 1] - g.spine_y_hi,
                           bus_x, seg[-1, 1] + g.spine_y_hi)
    else:
        unique_nets = set(layout.row_net_per_pixel[i] for i in active_pix)
        for rnet in sorted(unique_nets):
            pix_idx = [i for i in active_pix if layout.row_net_per_pixel[i] == rnet]
            if len(pix_idx) < 2:
                continue
            pts = Pxy[pix_idx]
            pts = pts[np.argsort(pts[:, 0])]
            for run in contiguous_runs(pts[:, 0], params.pitch_x_mm * 1.25):
                if len(run) < 2:
                    continue
                seg = pts[run]
                row_y = np.mean(seg[:, 1])
                _draw_line(lines, seg[0, 0] + g.via_x, row_y,
                           seg[-1, 0] + g.via_x, row_y)

    _write_footer(lines)
    return '\n'.join(lines)


def _draw_pixel_copper(lines, cx, cy, g, trace_w):
    """Draw all copper primitives for one pixel at world position (cx, cy)."""
    # Left spine (pad 1)
    _draw_line(lines, cx + g.xL, cy + g.spine_y_lo, cx + g.xL, cy + g.spine_y_hi)
    # Right spine (pad 2)
    _draw_line(lines, cx + g.xR, cy + g.spine_y_lo, cx + g.xR, cy + g.spine_y_hi)
    # Fingers
    for i in range(g.total_slots):
        fy = g.y0 + i * g.pitch
        if i % 2 == 0:
            # Pad 1 finger (left -> right)
            _draw_line(lines, cx + g.xL, cy + fy, cx + g.x_end_left_max, cy + fy)
        else:
            # Pad 2 finger (right -> left)
            _draw_line(lines, cx + g.xR, cy + fy, cx + g.col_end_x[i], cy + fy)


def _draw_pixel_pad(lines, am, cx, cy, g, trace_w, pad_num):
    """Draw copper for one pad of a pixel."""
    pass  # covered by _draw_pixel_copper



def _append_new_aperture_defs(lines, am, before_count):
    for definition in am.definitions()[before_count:]:
        lines.append(definition)


def _connector_pad_bbox(pad):
    corners = np.asarray(pad["corners"], dtype=float)
    min_xy = corners.min(axis=0)
    max_xy = corners.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    width = max(max_xy[0] - min_xy[0], 0.001)
    height = max(max_xy[1] - min_xy[1], 0.001)
    return center, width, height


def _pad_axes(pad):
    corners = np.asarray(pad["corners"], dtype=float)
    center = np.asarray(pad["center"], dtype=float)
    e_x = corners[1] - corners[0]
    e_y = corners[2] - corners[1]
    len_x = float(np.linalg.norm(e_x))
    len_y = float(np.linalg.norm(e_y))
    if len_x < 1e-9 or len_y < 1e-9:
        return center, None, 0.0, 0.0
    if len_x >= len_y:
        return center, e_x / len_x, len_x, len_y
    return center, e_y / len_y, len_y, len_x


def _is_axis_aligned_pad(pad):
    corners = np.asarray(pad["corners"], dtype=float)
    for edge in (corners[1] - corners[0], corners[2] - corners[1]):
        norm = float(np.linalg.norm(edge))
        if norm < 1e-9:
            return False
        unit = np.abs(edge / norm)
        if max(unit[0], unit[1]) < 0.999:
            return False
    return True


def _flash_rect(lines, am, center, width, height):
    before = len(am.definitions())
    aperture = am.rect(width, height)
    _append_new_aperture_defs(lines, am, before)
    lines.append(f"D{aperture}*")
    _flash(lines, center[0], center[1])


def _draw_stroked_pad(lines, am, pad):
    center, axis, length, width = _pad_axes(pad)
    if axis is None:
        return
    before = len(am.definitions())
    aperture = am.circle(max(width, 0.001))
    _append_new_aperture_defs(lines, am, before)
    lines.append(f"D{aperture}*")
    draw_len = max(length - width, 0.001)
    p1 = center - 0.5 * draw_len * axis
    p2 = center + 0.5 * draw_len * axis
    _draw_line(lines, p1[0], p1[1], p2[0], p2[1])


def _draw_region_polygon(lines, points):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return
    lines.append("G36*")
    lines.append(f"X{_coord(pts[0][0])}Y{_coord(-pts[0][1])}D02*")
    for p in pts[1:]:
        lines.append(f"X{_coord(p[0])}Y{_coord(-p[1])}D01*")
    lines.append(f"X{_coord(pts[0][0])}Y{_coord(-pts[0][1])}D01*")
    lines.append("G37*")


def _draw_connector_pad(lines, am, pad):
    # Connector pads come from the transformed source-footprint rectangle.
    # Emit that exact rectangle as a Gerber region so direct Gerbers match
    # KiCad native pad geometry for both straight and angled tail extensions.
    _draw_region_polygon(lines, pad["corners"])


def _flash_connector_pads(lines, am, connector_pads, layer_name):
    for pad in connector_pads:
        layers = pad["layers"]
        if "*.Cu" not in layers and layer_name not in layers:
            continue
        _draw_connector_pad(lines, am, pad)


def _flash_connector_mask(lines, am, connector_pads, layer_name):
    for pad in connector_pads:
        layers = pad["layers"]
        if "*.Mask" not in layers and layer_name not in layers:
            continue
        _draw_connector_pad(lines, am, pad)


def _local_to_world(lx, ly, t_hat, n_out, pad1_world):
    """Transform footprint-local coordinate to world (Y-up) coordinate.

    Accounts for X-negation, KiCad rotation, and Y-down → Y-up mapping.
    Formula: world = pad1_world + lx * t_hat - ly * n_out
    """
    wx = pad1_world[0] + lx * t_hat[0] - ly * n_out[0]
    wy = pad1_world[1] + lx * t_hat[1] - ly * n_out[1]
    return wx, wy


def _arc_to_segments(start, mid, end, n_segs=16):
    """Convert a 3-point arc (start, mid, end) to line segments.

    Finds the circle through the 3 points, then interpolates.
    """
    ax, ay = start
    bx, by = mid
    cx, cy = end

    # Find circumscribed circle center
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-12:
        return [(ax, ay), (cx, cy)]

    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) +
          (cx**2 + cy**2) * (ay - by)) / D
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) +
          (cx**2 + cy**2) * (bx - ax)) / D
    r = np.hypot(ax - ux, ay - uy)

    # Angles
    a_start = np.arctan2(ay - uy, ax - ux)
    a_mid = np.arctan2(by - uy, bx - ux)
    a_end = np.arctan2(cy - uy, cx - ux)

    # Determine direction (CW vs CCW) based on midpoint
    def normalize_angle(a, ref):
        while a - ref > np.pi:
            a -= 2 * np.pi
        while a - ref < -np.pi:
            a += 2 * np.pi
        return a

    a_mid_n = normalize_angle(a_mid, a_start)
    a_end_n = normalize_angle(a_end, a_start)

    # If mid is between start and end in the correct direction
    if (a_end_n - a_start) != 0:
        if (a_mid_n - a_start) * (a_end_n - a_start) < 0:
            # Wrong direction, go the other way
            if a_end_n > a_start:
                a_end_n -= 2 * np.pi
            else:
                a_end_n += 2 * np.pi

    pts = []
    for i in range(n_segs + 1):
        t = i / n_segs
        a = a_start + t * (a_end_n - a_start)
        pts.append((ux + r * np.cos(a), uy + r * np.sin(a)))
    return pts


def _generate_edge_cuts(layout, outline_xy, params, connector_dir=""):
    """Generate Edge.Cuts Gerber layer."""
    from opentac.cad.pipeline.geometry import tail_geometry

    lines = []
    _write_header(lines, "Edge.Cuts")
    am = ApertureManager()
    edge_ap = am.circle(0.05)
    lines.extend(am.definitions())
    lines.append(f"D{edge_ap}*")

    outline_poly = layout.outline_poly
    p1 = outline_xy[0]
    p2 = outline_xy[1]

    conn_pos = layout.connector_pos
    tail = tail_geometry(outline_poly, p1, p2, conn_pos, params)
    t_hat = tail["t_hat"]
    n_out = tail["n_out"]
    A = tail["A"]
    B = tail["B"]
    C = tail["C"]
    D = tail["D"]
    tail_ext = tail_extension_from_layout(layout)
    if tail_ext:
        t_hat = np.array(tail_ext["t_hat"], dtype=float)
        n_out = np.array(tail_ext["n_out"], dtype=float)

    # Board outline with cable cutout. Whenever a final board polygon
    # exists, export it stitched to the connector's open edge (all board
    # modes) — see export/kicad_pcb.py's matching branch for the fixed-mode
    # Edge.Cuts gap this closes. Legacy branch only when no final outline.
    if layout.final_outline_xy is not None:
        outline_segments = stitched_outline_segments(layout, outline_xy, params, connector_dir)
        if outline_segments is None:
            final_xy = np.asarray(layout.final_outline_xy, dtype=float)
            outline_segments = [(final_xy[e], final_xy[(e + 1) % len(final_xy)]) for e in range(len(final_xy))]
        for v1, v2 in outline_segments:
            _draw_line(lines, v1[0], v1[1], v2[0], v2[1])
    else:
        n_verts = len(outline_xy)
        for e in range(n_verts):
            v1 = outline_xy[e]
            v2 = outline_xy[(e + 1) % n_verts]
            if e == 0:
                if np.linalg.norm(v1 - A) > 0.01:
                    _draw_line(lines, v1[0], v1[1], A[0], A[1])
                if np.linalg.norm(B - v2) > 0.01:
                    _draw_line(lines, B[0], B[1], v2[0], v2[1])
            else:
                _draw_line(lines, v1[0], v1[1], v2[0], v2[1])

        _draw_line(lines, A[0], A[1], D[0], D[1])
        _draw_line(lines, B[0], B[1], C[0], C[1])

    connector_edges = connector_edge_geometries(layout, outline_xy, params, connector_dir)
    for p1_edge, p2_edge in connector_edges["lines"]:
        _draw_line(lines, p1_edge[0], p1_edge[1], p2_edge[0], p2_edge[1])
    for start, mid, end in connector_edges["arcs"]:
        pts = _arc_to_segments(start, mid, end, n_segs=12)
        for p1_arc, p2_arc in zip(pts, pts[1:]):
            _draw_line(lines, p1_arc[0], p1_arc[1], p2_arc[0], p2_arc[1])

    _write_footer(lines)
    return '\n'.join(lines)


def _generate_mask(layer_name, layout, active_pix, Pxy, g, params, connector_pads, include_pixels):
    """Generate solder mask openings."""
    lines = []
    _write_header(lines, layer_name)
    am = ApertureManager()
    pw = params.pixel_w_mm
    ph = params.pixel_h_mm
    mask_ap = am.rect(pw, ph)
    lines.extend(am.definitions())
    lines.append(f"D{mask_ap}*")

    if include_pixels:
        for k in active_pix:
            _flash(lines, Pxy[k, 0], Pxy[k, 1])

    _flash_connector_mask(lines, am, connector_pads, layer_name)

    _write_footer(lines)
    return '\n'.join(lines)


def _generate_front_silk(layout, outline_xy, params, connector_dir):
    """Front silkscreen: the design-name label, stroked with our own font.

    KiCad would render this text for us, but only on the kicad-cli path --
    which needs KiCad installed, and the hosted deployment is a plain Python
    image. This is the path every hosted export actually takes, so the label
    has to be real geometry here or it does not exist on most boards.

    One round aperture, one pen width: silkscreen is screen-printed, and a
    single consistent stroke is what survives it.
    """
    label = silk_label(layout, outline_xy, params, connector_dir)
    if not label:
        return None
    strokes = label_strokes(label)
    if not strokes:
        return None

    lines = []
    _write_header(lines, "F.Silkscreen")
    am = ApertureManager()
    ap = am.circle(label["thickness_mm"])
    lines.extend(am.definitions())
    lines.append(f"D{ap}*")
    for poly in strokes:
        for (x1, y1), (x2, y2) in zip(poly, poly[1:]):
            _draw_line(lines, x1, y1, x2, y2)
    _write_footer(lines)
    return "\n".join(lines)


def _generate_back_silk(manifest_json):
    """B.Silkscreen: the taxel-core QR and the registration grid.

    Region fills, not strokes. The QR is a knockout whose ink spans are
    rectangles of many different widths, and one aperture per width would be
    both larger and less exact than G36/G37 regions.
    """
    mark = backmark(manifest_json)
    rects = mark["rects"]
    strokes = mark.get("strokes") or []
    if not rects and not strokes:
        return None

    lines = []
    _write_header(lines, "B.Silkscreen")
    for x0, y0, x1, y1 in rects:
        _draw_region_polygon(lines, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    if strokes:
        # One round aperture, one pen width -- the same choice _generate_front_silk
        # makes, and for the same reason: silkscreen is screen-printed, and a
        # single consistent stroke is what survives it.
        am = ApertureManager()
        ap = am.circle(mark.get("stroke_w", 0.2))
        lines.extend(am.definitions())
        lines.append(f"D{ap}*")
        for poly in strokes:
            for (x1, y1), (x2, y2) in zip(poly, poly[1:]):
                _draw_line(lines, x1, y1, x2, y2)
    _write_footer(lines)
    return "\n".join(lines)


def _generate_dwgs_user(layout, outline_xy, params, connector_dir):
    """Generate Dwgs.User Gerber layer with connector stiffener outline."""
    user = connector_user_geometries(layout, outline_xy, params, connector_dir)
    user_lines = user["lines"]
    if not user_lines:
        return None

    lines = []
    _write_header(lines, "Dwgs.User")
    am = ApertureManager()
    widths = sorted({round(float(item.get("width", 0.1)), 6) for item in user_lines})
    apertures = {w: am.circle(w) for w in widths}
    lines.extend(am.definitions())
    current_ap = None
    for item in user_lines:
        width = round(float(item.get("width", 0.1)), 6)
        ap = apertures[width]
        if ap != current_ap:
            lines.append(f"D{ap}*")
            current_ap = ap
        x1, y1 = item["start"]
        x2, y2 = item["end"]
        _draw_line(lines, x1, y1, x2, y2)

    _write_footer(lines)
    return '\n'.join(lines)


def _generate_cmts_user(layout, outline_xy, params, connector_dir):
    """Generate Cmts.User Gerber comments with connector stiffener notes."""
    user = connector_user_geometries(layout, outline_xy, params, connector_dir)
    user_texts = user["texts"]
    if not user_texts:
        return None

    lines = []
    _write_header(lines, "Cmts.User")
    for item in user_texts:
        text = item.get("text", "")
        for part in text.replace("\\n", "\n").splitlines():
            clean = part.replace("*", " ").strip()
            if clean:
                lines.append(f"G04 {clean}*")
    _write_footer(lines)
    return '\n'.join(lines)
