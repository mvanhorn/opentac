"""KiCad .kicad_pcb board export."""

import os
import re
import uuid
import numpy as np

from opentac.cad.pipeline.types import Layout, Route, Params
from opentac.cad.pipeline.geometry import contiguous_runs
from opentac.cad.pipeline.tail_extension import tail_extension_from_layout
from opentac.cad.export.kicad_footprint import pixel_geometry, fmt
from opentac.cad.export.connector_geometry import connector_export_dir, connector_geometries, stitched_outline_segments
from opentac.cad.export.silk_label import silk_label
from opentac.cad.export.silk_backmark import backmark


def make_uuid():
    return str(uuid.uuid4())


def generate_kicad_pcb(
    layout: Layout,
    outline_xy: np.ndarray,
    col_routes: list[Route],
    row_routes: list[Route],
    removed_pix: set[int],
    params: Params,
    connector_dir: str,
    manifest_json: str | None = None,
) -> str:
    """Generate complete KiCad .kicad_pcb file content as string.

    `manifest_json` is what export_manifest() produced for this same board. It
    is optional so every existing caller and test keeps working; when it is
    supplied the back silkscreen carries the taxel-core QR and the registration
    grid (see export/silk_backmark.py).
    """
    from opentac.cad.pipeline.geometry import tail_geometry

    Pxy = layout.placements_xy
    n_pix = len(Pxy)
    conn_pos = layout.connector_pos
    pad_to_net = layout.pad_to_net
    active_pix = sorted(set(range(n_pix)) - removed_pix)

    trace_w = params.trace_w_mm
    clearance = params.clearance_mm
    via_drill = params.via_drill_mm
    via_dia = params.via_dia_mm
    pad_pitch = params.pad_pitch_mm
    pixel_w = params.pixel_w_mm
    pixel_h = params.pixel_h_mm

    # Cable geometry
    outline_poly = layout.outline_poly
    p1 = outline_xy[0]
    p2 = outline_xy[1]
    tail = tail_geometry(outline_poly, p1, p2, conn_pos, params,
                         fit_width=(layout.tail or {}).get("fit_width_mm"),
                         fit_offset=(layout.tail or {}).get("fit_offset_mm"))
    t_hat = tail["t_hat"]
    A = tail["A"]
    B = tail["B"]
    C = tail["C"]
    D = tail["D"]
    pad1_world = tail["pad1_world"]
    tail_ext = tail_extension_from_layout(layout)
    if tail_ext:
        t_hat = np.array(tail_ext["t_hat"], dtype=float)
        pad1_world = np.array(tail_ext["pad1_world"], dtype=float)
    conn_at = [pad1_world[0], -pad1_world[1]]
    conn_rot = np.degrees(np.arctan2(-t_hat[1], t_hat[0]))

    # Net map
    all_nets_set = set()
    for i in active_pix:
        all_nets_set.add(layout.col_net_per_pixel[i])
        all_nets_set.add(layout.row_net_per_pixel[i])

    col_nets = sorted([n for n in all_nets_set if n.startswith("COL_")])
    row_nets = sorted([n for n in all_nets_set if n.startswith("ROW_")])
    all_nets = col_nets + row_nets
    net_map = {name: idx + 1 for idx, name in enumerate(all_nets)}

    # Pixel geometry
    g = pixel_geometry(pixel_w, pixel_h, trace_w, params.gap_mm,
                       params.center_clear_mm, params.edge_clear_mm, via_dia)

    lines = []

    # Header
    lines.append('(kicad_pcb (version 20211014) (generator "python_autogen")')
    lines.append('  (general (thickness 1.6))')
    lines.append('  (paper "A4")')
    lines.append('  (layers')
    lines.append('    (0 "F.Cu" signal)')
    from opentac.cad.pipeline.route_4layer import copper_layers_of
    if copper_layers_of(params, row_routes) == 4:
        # 4-layer board: the row wiring's inner layers (see pipeline/route_4layer.py)
        lines.append('    (1 "In1.Cu" signal)')
        lines.append('    (2 "In2.Cu" signal)')
    lines.append('    (31 "B.Cu" signal)')
    lines.append('    (36 "B.SilkS" user)')
    lines.append('    (37 "F.SilkS" user)')
    lines.append('    (38 "B.Mask" user)')
    lines.append('    (39 "F.Mask" user)')
    lines.append('    (40 "Dwgs.User" user)')
    lines.append('    (41 "Cmts.User" user)')
    lines.append('    (44 "Edge.Cuts" user)')
    lines.append('    (49 "F.Fab" user)')
    lines.append('    (51 "User.2" user)')
    lines.append('  )')

    # Nets
    lines.append('  (net 0 "")')
    for name in all_nets:
        lines.append(f'  (net {net_map[name]} "{name}")')

    # Setup
    lines.append('  (setup')
    lines.append(f'    (last_trace_width {fmt(trace_w)})')
    lines.append(f'    (trace_clearance {fmt(clearance)})')
    lines.append(f'    (via_size {fmt(via_dia)})')
    lines.append(f'    (via_drill {fmt(via_drill)})')
    lines.append('    (edge_width 0.15)')
    lines.append(f'    (user_trace_width {fmt(trace_w)})')
    lines.append('  )')
    lines.append('  (net_class "Default" "default"')
    lines.append(f'    (clearance {fmt(clearance)})')
    lines.append(f'    (trace_width {fmt(trace_w)})')
    lines.append(f'    (via_dia {fmt(via_dia)})')
    lines.append(f'    (via_drill {fmt(via_drill)})')
    lines.append('    (uvia_dia 0.3)')
    lines.append('    (uvia_drill 0.1)')
    for name in all_nets:
        lines.append(f'    (add_net "{name}")')
    lines.append('  )')

    # Connector footprint: exported whenever the size has a folder-backed
    # footprint — with or without a custom tail extension.
    connector_dir = connector_export_dir(layout, connector_dir)
    conn_text = _build_connector_resolved(layout, outline_xy, params, connector_dir,
                                           pad_to_net, net_map)
    if conn_text:
        lines.append(conn_text)

    # Board outline (Edge.Cuts). Whenever the layout carries a final board
    # polygon, export THAT, stitched to the connector's open edge — for
    # every board mode. Fixed mode used to fall through to the legacy
    # hand-stitched branch below, whose tail walls stop at cable_length
    # while the connector outline starts at the pad row
    # (CONNECTOR_BODY_ABOVE_MM deeper): a 2.75mm open gap per side in the
    # exported Edge.Cuts, gerber-confirmed on the fingertip wrap band. The
    # final outline already reaches the full connector-body envelope
    # (tail_geometry's total_length), so the stitched cut lands exactly on
    # the connector's open endpoints, same as expand mode always did. The
    # legacy branch remains only for layouts with no final outline at all.
    if layout.final_outline_xy is not None:
        outline_segments = stitched_outline_segments(layout, outline_xy, params, connector_dir)
        if outline_segments is None:
            final_xy = np.asarray(layout.final_outline_xy, dtype=float)
            outline_segments = [(final_xy[e], final_xy[(e + 1) % len(final_xy)]) for e in range(len(final_xy))]
        for start_edge, end_edge in outline_segments:
            lines.append(_edge_line(start_edge, end_edge))
    else:
        n_verts = len(outline_xy)
        for e in range(n_verts):
            v1 = outline_xy[e]
            v2 = outline_xy[(e + 1) % n_verts]
            if e == 0:
                if np.linalg.norm(v1 - A) > 0.01:
                    lines.append(_edge_line(v1, A))
                if np.linalg.norm(B - v2) > 0.01:
                    lines.append(_edge_line(B, v2))
            else:
                lines.append(_edge_line(v1, v2))
        lines.append(_edge_line(A, D))
        lines.append(_edge_line(B, C))

    # Pixel footprints
    pix_num = 0
    for k in active_pix:
        pix_num += 1
        cx = Pxy[k, 0]
        cy = Pxy[k, 1]
        row_net = layout.row_net_per_pixel[k]
        col_net = layout.col_net_per_pixel[k]
        row_net_id = net_map[row_net]
        col_net_id = net_map[col_net]
        ref = f"PX{pix_num:04d}"
        lines.append(_pixel_footprint(cx, -cy, ref, g, pixel_w, pixel_h, trace_w,
                                       via_drill, via_dia, row_net_id, row_net,
                                       col_net_id, col_net))

    # Column route segments (F.Cu)
    for route in col_routes:
        P = route.polyline_xy
        if route.net_name not in net_map:
            continue
        net_id = net_map[route.net_name]
        for s in range(len(P) - 1):
            lines.append(_segment(P[s], P[s + 1], trace_w, "F.Cu", net_id))

    # Row route segments (B.Cu, or the route's inner layer on a 4-layer board)
    for route in row_routes:
        P = route.polyline_xy
        if route.net_name not in net_map:
            continue
        net_id = net_map[route.net_name]
        layer = getattr(route, "layer", None) or "B.Cu"
        for s in range(len(P) - 1):
            lines.append(_segment(P[s], P[s + 1], trace_w, layer, net_id))

    # Intra-column bus lines (F.Cu)
    unique_col_nets = set(layout.col_net_per_pixel[i] for i in active_pix)
    for cnet in sorted(unique_col_nets):
        if cnet not in net_map:
            continue
        net_id = net_map[cnet]
        pix_idx = [i for i in active_pix if layout.col_net_per_pixel[i] == cnet]
        if len(pix_idx) < 2:
            continue
        pts = Pxy[pix_idx]
        pts = pts[np.argsort(pts[:, 1])]
        # One bus per contiguous pixel run: a joined net (hug router) can span
        # a concavity gap, and a single min-to-max bus would jump across it.
        for run in contiguous_runs(pts[:, 1], params.pitch_y_mm * 1.25):
            if len(run) < 2:
                continue          # a lone pixel's spine lives in its footprint
            seg = pts[run]
            bus_x = np.mean(seg[:, 0]) + g.xR
            ylo = seg[0, 1] - g.spine_y_hi  # spine_y_hi is positive half
            yhi = seg[-1, 1] + g.spine_y_hi
            lines.append(_segment([bus_x, ylo], [bus_x, yhi], trace_w, "F.Cu", net_id))

    # Intra-row bus lines (B.Cu)
    unique_row_nets = set(layout.row_net_per_pixel[i] for i in active_pix)
    for rnet in sorted(unique_row_nets):
        if rnet not in net_map:
            continue
        net_id = net_map[rnet]
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
            bus_xlo = seg[0, 0] + g.via_x
            bus_xhi = seg[-1, 0] + g.via_x
            lines.append(_segment([bus_xlo, row_y], [bus_xhi, row_y], trace_w, "B.Cu", net_id))

    # Design-name label on the front silkscreen, inboard of the connector.
    lines.extend(_silk_label_lines(layout, outline_xy, params, connector_dir))

    # Taxel-core QR + registration grid on the back silkscreen.
    lines.extend(_backmark_lines(manifest_json))

    lines.append(')')
    return '\n'.join(lines)


def _silk_label_lines(layout, outline_xy, params, connector_dir) -> list:
    """The design name as a gr_text on F.SilkS.

    KiCad strokes this with its own font, which is why nothing here mentions
    glyphs: when kicad-cli plots the Gerbers from this file it renders the
    text itself. The pure-Python Gerber path in export/gerber.py cannot, and
    draws the same string from export/stroke_font.py instead -- both read
    their placement from silk_label(), so the two land in the same spot.
    """
    label = silk_label(layout, outline_xy, params,
                       connector_export_dir(layout, connector_dir))
    if not label:
        return []
    cx, cy = label["center"]
    h = label["height_mm"]
    return [
        f'  (gr_text "{label["text"]}" '
        f'(at {fmt(cx)} {fmt(-cy)} {fmt(label["rot_deg"])}) (layer "F.SilkS")',
        f'    (effects (font (size {fmt(h)} {fmt(h)}) '
        f'(thickness {fmt(label["thickness_mm"])})))',
        '  )',
    ]


def _backmark_lines(manifest_json) -> list:
    """The back-silkscreen marks as filled gr_poly on B.SilkS.

    Filled polygons rather than text or strokes: the QR is a knockout, and
    KiCad unions overlapping silk instead of subtracting it, so the artwork has
    to arrive as the ink itself. Y is negated here exactly as it is for every
    other primitive in this file.
    """
    mark = backmark(manifest_json)
    out = []
    for x0, y0, x1, y1 in mark["rects"]:
        out.append(
            f'  (gr_poly (pts (xy {fmt(x0)} {fmt(-y0)}) (xy {fmt(x1)} {fmt(-y0)}) '
            f'(xy {fmt(x1)} {fmt(-y1)}) (xy {fmt(x0)} {fmt(-y1)})) '
            f'(stroke (width 0) (type solid)) (fill solid) (layer "B.SilkS"))')
    # The grid's dimension callout is stroked, not filled: it comes from the
    # same single-stroke font export/gerber.py draws the front label with, so
    # both export paths render identical geometry.
    w = mark.get("stroke_w", 0.2)
    for poly in mark.get("strokes", []):
        for (x1, y1), (x2, y2) in zip(poly, poly[1:]):
            out.append(
                f'  (gr_line (start {fmt(x1)} {fmt(-y1)}) (end {fmt(x2)} {fmt(-y2)}) '
                f'(stroke (width {fmt(w)}) (type solid)) (layer "B.SilkS"))')
    return out


def _edge_line(v1, v2):
    return (f'  (gr_line (start {fmt(v1[0])} {fmt(-v1[1])}) '
            f'(end {fmt(v2[0])} {fmt(-v2[1])}) (layer "Edge.Cuts") (width 0.15))')


def _segment(p1, p2, width, layer, net_id):
    return (f'  (segment (start {fmt(p1[0])} {fmt(-p1[1])}) '
            f'(end {fmt(p2[0])} {fmt(-p2[1])}) '
            f'(width {fmt(width)}) (layer "{layer}") (net {net_id}) '
            f'(tstamp "{make_uuid()}"))')


def _pixel_footprint(kx, ky, ref, g, pixel_w, pixel_h, trace_w,
                      via_drill, via_dia, row_net_id, row_net, col_net_id, col_net):
    """Generate inline pixel footprint."""
    x1 = -pixel_w / 2
    y1 = -pixel_h / 2
    x2 = pixel_w / 2
    y2 = pixel_h / 2
    uid = make_uuid()

    lines = []
    lines.append('  (footprint "tactile_pixel"')
    lines.append('    (layer "F.Cu")')
    lines.append(f'    (tstamp "{uid}")')
    lines.append(f'    (at {fmt(kx)} {fmt(ky)} 0)')
    lines.append(f'    (fp_text reference "{ref}" (at 0 -2.6 0) (layer "F.SilkS") hide '
                 f'(effects (font (size 1 1) (thickness 0.15))))')
    lines.append(f'    (fp_text value "tactile_pixel" (at 0 2.6 0) (layer "User.2") hide '
                 f'(effects (font (size 1 1) (thickness 0.15))))')
    lines.append('    (attr smd)')
    lines.append(f'    (fp_rect (start {fmt(x1)} {fmt(y1)}) (end {fmt(x2)} {fmt(y2)}) '
                 f'(stroke (width 0.1) (type default)) (fill no) (layer "User.2"))')

    # Pad 1 (row net)
    lines.append(f'    (pad "1" smd custom (net {row_net_id} "{row_net}") '
                 f'(at {fmt(g.xL)} {fmt(g.ymid)}) (size 0.05 0.05) '
                 f'(layers "F.Cu" "F.Mask") (options (clearance outline) (anchor circle)) (primitives')
    lines.append(f'    (gr_poly (pts (xy -0.02 -0.02) (xy 0.02 -0.02) '
                 f'(xy 0.02 0.02) (xy -0.02 0.02)) (width 0) (fill yes))')
    lines.append(f'    (gr_line (start 0 {fmt(g.spine_y_lo - g.ymid)}) '
                 f'(end 0 {fmt(g.spine_y_hi - g.ymid)}) (width {fmt(trace_w)}))')
    for i in range(g.total_slots):
        if i % 2 == 0:
            fy = g.y0 + i * g.pitch
            lines.append(f'    (gr_line (start 0 {fmt(fy - g.ymid)}) '
                         f'(end {fmt(g.x_end_left_max - g.xL)} {fmt(fy - g.ymid)}) '
                         f'(width {fmt(trace_w)}))')
    lines.append('  ))')

    # Pad 2 (col net)
    lines.append(f'    (pad "2" smd custom (net {col_net_id} "{col_net}") '
                 f'(at {fmt(g.xR)} {fmt(g.ymid)}) (size 0.05 0.05) '
                 f'(layers "F.Cu" "F.Mask") (options (clearance outline) (anchor circle)) (primitives')
    lines.append(f'    (gr_poly (pts (xy -0.02 -0.02) (xy 0.02 -0.02) '
                 f'(xy 0.02 0.02) (xy -0.02 0.02)) (width 0) (fill yes))')
    lines.append(f'    (gr_line (start 0 {fmt(g.spine_y_lo - g.ymid)}) '
                 f'(end 0 {fmt(g.spine_y_hi - g.ymid)}) (width {fmt(trace_w)}))')
    for i in range(g.total_slots):
        if i % 2 == 1:
            fy = g.y0 + i * g.pitch
            lines.append(f'    (gr_line (start 0 {fmt(fy - g.ymid)}) '
                         f'(end {fmt(g.col_end_x[i] - g.xR)} {fmt(fy - g.ymid)}) '
                         f'(width {fmt(trace_w)}))')
    lines.append('  ))')

    # Via
    lines.append(f'    (pad "1" thru_hole circle (net {row_net_id} "{row_net}") '
                 f'(at {fmt(g.via_x)} {fmt(g.via_y)}) '
                 f'(size {fmt(via_dia)} {fmt(via_dia)}) (drill {fmt(via_drill)}) '
                 f'(layers "*.Cu" "*.Mask"))')

    lines.append('  )')
    return '\n'.join(lines)


def _kicad_xy(pt):
    return np.array([float(pt[0]), -float(pt[1])], dtype=float)


def _connector_custom_pad_line(pad, pad_num, net_text, drill):
    anchor_world = pad["drill_center"] if drill else pad["center"]
    anchor = _kicad_xy(anchor_world)
    corner_pts = []
    for corner in pad["corners"]:
        p = _kicad_xy(corner) - anchor
        corner_pts.append(f'(xy {fmt(p[0])} {fmt(p[1])})')

    drill_text = f' (drill {fmt(drill)})' if drill else ''
    # The pad's own layers from the connector footprint: contacts are on the
    # BOTTOM copper (Molex 505110 is bottom-contact), so the mask opens on
    # B.Mask only and the top stays covered under the stiffener.
    if pad.get("layers"):
        layers = " ".join(f'"{name}"' for name in pad["layers"])
    else:
        layers = '"*.Cu" "B.Mask"' if drill else '"F.Cu" "F.Mask"'
    kind = pad.get("kind") or ('thru_hole' if drill else 'smd')

    return (f'    (pad "{pad_num}" {kind} custom{net_text} '
            f'(at {fmt(anchor[0])} {fmt(anchor[1])} 0) '
            f'(size 0.05 0.05){drill_text} (layers {layers}) '
            f'(options (clearance outline) (anchor rect)) '
            f'(primitives (gr_poly (pts {" ".join(corner_pts)}) '
            f'(width 0) (fill yes))))')


def _build_connector_resolved(layout, outline_xy, params, conn_dir, pad_to_net, net_map):
    """Emit connector geometry already resolved into board coordinates.

    Direct Gerber export is the working reference for angled tail extensions.
    Writing the KiCad connector in the same resolved coordinate space avoids
    KiCad-specific rotation/drill-offset interpretation differences.
    """
    model = connector_geometries(layout, outline_xy, params, conn_dir)
    if not model:
        return None

    conn_pos = int(layout.connector_pos)
    lines = []
    lines.append(f'  (footprint "{conn_pos}_pos_0.5mm_FPC_resolved"')
    lines.append('    (layer "F.Cu")')
    lines.append(f'    (tstamp "{make_uuid()}")')
    lines.append('    (at 0 0 0)')
    lines.append(f'    (property "Reference" "J1" (at 0 0 0) (layer "F.SilkS") hide '
                 f'(effects (font (size 1 1) (thickness 0.15))))')
    lines.append(f'    (property "Value" "{conn_pos}_pos_0.5mm_FPC" (at 0 1.2 0) '
                 f'(layer "F.Fab") hide (effects (font (size 1 1) (thickness 0.15))))')
    lines.append('    (attr smd)')

    # Every pad is emitted as a custom polygon anchored at its own
    # already-resolved world-space drill/pad center (pad["drill_center"] /
    # pad["center"], both computed via connector_geometry.local_to_world —
    # the same reference transform the Gerber exporter uses directly).
    # A native KiCad "(at x y rot)" + local drill (offset ox oy) shortcut
    # used to be emitted here for axis-aligned tails, on the theory that
    # local_to_world's rotation could be reproduced by a plain KiCad pad
    # rotation. That relies on exactly matching KiCad's own pad-rotation
    # handedness convention, which isn't independently verified anywhere in
    # this codebase — asymmetric features (like this connector family's
    # off-center drill offset, "(offset 0 -1.5)") are exactly what would
    # silently come out mirrored if that assumption were ever wrong. Always
    # emitting the resolved-coordinate custom polygon instead removes that
    # risk entirely by construction, matching the Gerber path byte-for-byte.
    for pad in sorted(model["pads"], key=lambda p: p["pad_num"]):
        pad_num = int(pad["pad_num"])
        net_name = pad_to_net.get(pad_num, pad_to_net.get(str(pad_num)))
        net_text = ''
        if net_name in net_map:
            net_text = f' (net {net_map[net_name]} "{net_name}")'
        drill = pad.get("drill")
        lines.append(_connector_custom_pad_line(pad, pad_num, net_text, drill))

    for p1, p2 in model["edge_lines"]:
        a = _kicad_xy(p1)
        b = _kicad_xy(p2)
        lines.append(f'    (fp_line (start {fmt(a[0])} {fmt(a[1])}) '
                     f'(end {fmt(b[0])} {fmt(b[1])}) '
                     f'(stroke (width 0.05) (type default)) (layer "Edge.Cuts"))')

    for start, mid, end in model["edge_arcs"]:
        s = _kicad_xy(start)
        m = _kicad_xy(mid)
        e = _kicad_xy(end)
        lines.append(f'    (fp_arc (start {fmt(s[0])} {fmt(s[1])}) '
                     f'(mid {fmt(m[0])} {fmt(m[1])}) '
                     f'(end {fmt(e[0])} {fmt(e[1])}) '
                     f'(stroke (width 0.05) (type default)) (layer "Edge.Cuts"))')

    for item in model.get("user_lines", []):
        a = _kicad_xy(item["start"])
        b = _kicad_xy(item["end"])
        width = item.get("width", 0.1)
        layer = item.get("layer", "Dwgs.User")
        lines.append(f'    (fp_line (start {fmt(a[0])} {fmt(a[1])}) '
                     f'(end {fmt(b[0])} {fmt(b[1])}) '
                     f'(stroke (width {fmt(width)}) (type default)) (layer "{layer}"))')

    for item in model.get("user_arcs", []):
        s = _kicad_xy(item["start"])
        m = _kicad_xy(item["mid"])
        e = _kicad_xy(item["end"])
        width = item.get("width", 0.1)
        layer = item.get("layer", "Dwgs.User")
        lines.append(f'    (fp_arc (start {fmt(s[0])} {fmt(s[1])}) '
                     f'(mid {fmt(m[0])} {fmt(m[1])}) '
                     f'(end {fmt(e[0])} {fmt(e[1])}) '
                     f'(stroke (width {fmt(width)}) (type default)) (layer "{layer}"))')

    for item in model.get("user_texts", []):
        p = _kicad_xy(item["at"])
        sx, sy = item.get("size", (1.0, 1.0))
        thickness = item.get("thickness", 0.15)
        layer = item.get("layer", "Cmts.User")
        justify = item.get("justify", "")
        justify_text = f' (justify {justify})' if justify else ''
        lines.append(f'    (fp_text user "{item["text"]}" '
                     f'(at {fmt(p[0])} {fmt(p[1])} {fmt(item.get("rot", 0.0))}) '
                     f'(layer "{layer}") '
                     f'(effects (font (size {fmt(sx)} {fmt(sy)}) '
                     f'(thickness {fmt(thickness)})){justify_text}))')

    lines.append('  )')
    return '\n'.join(lines)


def _build_connector(conn_dir, conn_pos, conn_at, conn_rot, pad_to_net, net_map, reverse_pad_order=False):
    """Load the exact connector footprint and place it at the tail end."""
    mod_file = os.path.join(conn_dir, f"{conn_pos}_pos_0.5mm_FPC.kicad_mod")
    if not os.path.exists(mod_file):
        return None

    with open(mod_file, 'r') as f:
        text = f.read()

    src_lines = text.strip().split('\n')
    while src_lines and not src_lines[-1].strip():
        src_lines.pop()

    out = []
    out.append(f'  (footprint "{conn_pos}_pos_0.5mm_FPC"')

    added_instance = False
    in_pad = False
    pad_paren_depth = 0

    for li in range(1, len(src_lines)):
        L = src_lines[li]
        L_trim = L.strip()

        if li == len(src_lines) - 1 and L_trim == ')':
            continue

        if not added_instance and '(layer "F.Cu")' in L_trim:
            out.append(L)
            out.append(f'  \t(tstamp "{make_uuid()}")')
            out.append(f'  \t(at {fmt(conn_at[0])} {fmt(conn_at[1])} {fmt(conn_rot)})')
            added_instance = True
            continue

        if '(property "Reference"' in L_trim:
            L = L.replace('"REF**"', '"J1"')

        # Detect pad opening
        pad_match = re.search(r'^\(pad "(\d+)"', L_trim)
        if pad_match:
            in_pad = True
            pad_paren_depth = L_trim.count('(') - L_trim.count(')')

            pad_num = int(pad_match.group(1))
            logical_pad_num = int(conn_pos) + 1 - pad_num if reverse_pad_order else pad_num
            if logical_pad_num in pad_to_net:
                net_name = pad_to_net[logical_pad_num]
                if net_name in net_map:
                    net_id = net_map[net_name]
                    L = re.sub(
                        r'(\(pad "\d+"\s+\w+\s+\w+)',
                        rf'\1\n\t  (net {net_id} "{net_name}")',
                        L, count=1
                    )

        if in_pad and not pad_match:
            pad_paren_depth += L_trim.count('(') - L_trim.count(')')
            if reverse_pad_order:
                L = re.sub(r'\(offset\s+0\s+-1\.5\)', '(offset 0 1.5)', L)

            if pad_paren_depth <= 0:
                in_pad = False

        if not L_trim:
            continue

        out.append(f'  {L}')

    out.append('  )')
    return '\n'.join(out)
