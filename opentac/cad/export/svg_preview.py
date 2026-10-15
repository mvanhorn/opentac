"""SVG preview generation for browser display."""

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from opentac.cad.pipeline.types import Layout, Route, Params
from opentac.cad.export.silk_label import label_strokes, silk_label


def generate_svg(
    layout: Layout,
    outline_xy: np.ndarray,
    col_routes: list[Route],
    row_routes: list[Route],
    removed_pix: set[int],
    params: Params,
    connector_dir: str = "",
) -> str:
    """Generate SVG preview of the routed board."""
    Pxy = layout.placements_xy
    n_pix = len(Pxy)
    active_pix = set(range(n_pix)) - removed_pix
    px_hw = params.pixel_w_mm / 2
    px_hh = params.pixel_h_mm / 2

    # Compute bounding box (board poly bounds + margin)
    board_bounds = layout.board_poly.bounds  # (minx, miny, maxx, maxy)
    margin = 2.0
    xmin = board_bounds[0] - margin
    ymin = board_bounds[1] - margin
    xmax = board_bounds[2] + margin
    ymax = board_bounds[3] + margin
    width = xmax - xmin
    height = ymax - ymin

    # SVG uses Y-down, our coordinates are Y-up
    # Transform: svg_y = ymax - y (flip vertically)
    def tx(x):
        return x - xmin

    def ty(y):
        return ymax - y

    svg_w = 800
    svg_h = svg_w * height / width if width > 0 else 600
    scale = svg_w / width

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'viewBox="{0} {0} {width:.2f} {height:.2f}" '
                 f'width="{svg_w:.0f}" height="{svg_h:.0f}">')
    lines.append('<style>')
    lines.append('  .outline { fill: none; stroke: #333; stroke-width: 0.3; }')
    lines.append('  .sensor-outline { fill: none; stroke: #6688aa; stroke-width: 0.16; stroke-dasharray: 0.7,0.35; }')
    lines.append('  .routing-area { fill: #eef8ef; stroke: none; opacity: 0.55; }')
    lines.append('  .route-keepout { fill: none; stroke: #2d7d46; stroke-width: 0.12; stroke-dasharray: 0.8,0.45; opacity: 0.55; }')
    lines.append('  .cable { fill: none; stroke: #999; stroke-width: 0.2; stroke-dasharray: 0.5; }')
    lines.append('  .pixel { fill: #ddeeff; stroke: #6688aa; stroke-width: 0.1; }')
    lines.append('  .pixel-removed { fill: #ffdddd; stroke: #cc4444; stroke-width: 0.1; stroke-dasharray: 0.3; }')
    lines.append('  .col-route { fill: none; stroke: #cc3333; stroke-width: 0.15; }')
    lines.append('  .row-route { fill: none; stroke: #3333cc; stroke-width: 0.15; stroke-dasharray: 0.3,0.15; }')
    # 4-layer boards: the inner-layer row wiring
    lines.append('  .in1-route { fill: none; stroke: #c98a00; stroke-width: 0.15; }')
    lines.append('  .in2-route { fill: none; stroke: #2e9e57; stroke-width: 0.15; }')
    lines.append('  .connector { fill: #ffffcc; stroke: #999; stroke-width: 0.15; }')
    lines.append('  .pad-line { stroke: #cc0000; stroke-width: 0.1; stroke-dasharray: 0.5; }')
    lines.append('  .silk { fill: none; stroke: #b06a1f; stroke-width: 0.15; '
                 'stroke-linecap: round; stroke-linejoin: round; }')
    lines.append('</style>')

    # Background
    lines.append(f'<rect x="0" y="0" width="{width:.2f}" height="{height:.2f}" fill="white"/>')

    # Board outline. Expanded preserve mode shows the final board edge and the
    # original sensor outline separately so users can see the added route area.
    final_outline = layout.final_outline_xy if layout.final_outline_xy is not None else outline_xy
    if layout.routing_margin_applied_mm > 0:
        final_pts_fill = ' '.join(f'{tx(x):.3f},{ty(y):.3f}' for x, y in final_outline)
        lines.append(f'<polygon points="{final_pts_fill}" class="routing-area"/>')

    outline_pts = ' '.join(f'{tx(x):.3f},{ty(y):.3f}' for x, y in final_outline)
    lines.append(f'<polygon points="{outline_pts}" class="outline"/>')
    if layout.routing_margin_applied_mm > 0:
        sensor_pts = ' '.join(f'{tx(x):.3f},{ty(y):.3f}' for x, y in outline_xy)
        lines.append(f'<polygon points="{sensor_pts}" class="sensor-outline"/>')

    # Route edge-padding guide: trace centerlines should stay inside this guide.
    keepout_offset = params.board_edge_clear_mm + params.trace_w_mm / 2
    if keepout_offset > 0:
        safe_poly = layout.board_poly.buffer(-keepout_offset, join_style='mitre')
        safe_geoms = []
        if isinstance(safe_poly, Polygon):
            safe_geoms = [safe_poly]
        elif isinstance(safe_poly, MultiPolygon):
            safe_geoms = list(safe_poly.geoms)
        for geom in safe_geoms:
            if geom.is_empty or geom.area <= 0:
                continue
            guide_pts = ' '.join(f'{tx(x):.3f},{ty(y):.3f}' for x, y in geom.exterior.coords)
            lines.append(f'<polygon points="{guide_pts}" class="route-keepout"/>')

    # Cable outline
    if layout.cable_poly and not layout.cable_poly.is_empty:
        cable_coords = list(layout.cable_poly.exterior.coords)
        cable_pts = ' '.join(f'{tx(x):.3f},{ty(y):.3f}' for x, y in cable_coords)
        lines.append(f'<polygon points="{cable_pts}" class="cable"/>')

    # Pixels
    for k in range(n_pix):
        cx, cy = Pxy[k]
        x = tx(cx - px_hw)
        y = ty(cy + px_hh)
        w = params.pixel_w_mm
        h = params.pixel_h_mm
        cls = "pixel" if k in active_pix else "pixel-removed"
        lines.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" class="{cls}"/>')

    # Column routes (F.Cu)
    for route in col_routes:
        P = route.polyline_xy
        if P is None or len(P) < 2:
            continue
        pts = ' '.join(f'{tx(p[0]):.3f},{ty(p[1]):.3f}' for p in P)
        lines.append(f'<polyline points="{pts}" class="col-route"/>')

    # Row routes: B.Cu ones as .row-route, in order (the route editor pairs
    # them with edit_data's B.Cu routes by position); a 4-layer board's inner
    # ones after them, as .in1-route / .in2-route
    inner_cls = {"In1.Cu": "in1-route", "In2.Cu": "in2-route"}
    for route in sorted(row_routes, key=lambda r: getattr(r, "layer", None) in inner_cls):
        P = route.polyline_xy
        if P is None or len(P) < 2:
            continue
        cls = inner_cls.get(getattr(route, "layer", None), "row-route")
        pts = ' '.join(f'{tx(p[0]):.3f},{ty(p[1]):.3f}' for p in P)
        lines.append(f'<polyline points="{pts}" class="{cls}"/>')

    # Pad row line
    pad_y = layout.pad_y_mm
    lines.append(f'<line x1="0" y1="{ty(pad_y):.3f}" x2="{width:.2f}" y2="{ty(pad_y):.3f}" class="pad-line"/>')

    # Silkscreen design-name label, drawn from the same placement the exporters
    # use so the preview cannot promise a label the board does not get.
    label = silk_label(layout, outline_xy, params, connector_dir)
    if label:
        for poly in label_strokes(label):
            pts = ' '.join(f'{tx(x):.3f},{ty(y):.3f}' for x, y in poly)
            lines.append(f'<polyline points="{pts}" class="silk"/>')

    lines.append('</svg>')
    return '\n'.join(lines)
