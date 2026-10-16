"""Final-board outline helper that follows main layout and route silhouette."""

import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon, box
from shapely.ops import unary_union

from .geometry import (
    CONNECTOR_BODY_ABOVE_MM, CONNECTOR_BODY_BELOW_MM, extension_tail_poly, fillet_inside_corners,
    make_cable_rect, make_tail_poly, remove_collinear_closed, tail_offset_mm, tail_side_edges,
    tail_step_fillet, tail_without_landing,
)
from .route_4layer import INNER_LAYERS, copper_layers_of


def _largest_polygon(poly) -> Polygon:
    if isinstance(poly, Polygon):
        return poly
    if hasattr(poly, "geoms"):
        polys = [g for g in poly.geoms if isinstance(g, Polygon) and not g.is_empty]
        if polys:
            return max(polys, key=lambda g: g.area)
    return Polygon()


def _line_geoms(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if hasattr(geom, "geoms"):
        out = []
        for child in geom.geoms:
            out.extend(_line_geoms(child))
        return out
    return []


def _inboard_clip_polygon(outline_xy, n_out, board_poly, routes, outward_mm=0.0):
    p1 = np.asarray(outline_xy[0], dtype=float)
    p2 = np.asarray(outline_xy[1], dtype=float)
    t_hat = p2 - p1
    norm = float(np.linalg.norm(t_hat))
    if norm < 1e-9:
        return Polygon()
    t_hat = t_hat / norm
    n_out = np.asarray(n_out, dtype=float)
    n_norm = float(np.linalg.norm(n_out))
    if n_norm < 1e-9:
        return Polygon()
    n_out = n_out / n_norm

    minx, miny, maxx, maxy = board_poly.bounds
    for route in routes:
        P = np.asarray(route.polyline_xy, dtype=float)
        if len(P):
            minx = min(minx, float(np.min(P[:, 0])))
            miny = min(miny, float(np.min(P[:, 1])))
            maxx = max(maxx, float(np.max(P[:, 0])))
            maxy = max(maxy, float(np.max(P[:, 1])))
    span = max(maxx - minx, maxy - miny, norm, 1.0) + 100.0

    a = p1 - span * t_hat + outward_mm * n_out
    b = p2 + span * t_hat + outward_mm * n_out
    c = b - span * n_out
    d = a - span * n_out
    return Polygon([a, b, c, d])


def _tail_extension_corridor(layout):
    ext = layout.tail.get("tail_extension_resolved") if layout.tail else None
    if not ext or not ext.get("enabled"):
        return None
    path = np.asarray(ext.get("path") or [], dtype=float)
    if len(path) < 2:
        return None
    width = float(ext.get("width_mm") or 0.0)
    clearance = float(ext.get("clearance_mm") or 0.0)
    if width <= 0:
        return None
    return extension_tail_poly(path, width, clearance, layout.tail.get("connector_min_width_mm"),
                               float(ext.get("offset_mm") or 0.0), tail_step_fillet(layout.tail),
                               lead_dir=layout.tail.get("n_out"))


def _poly_to_coords(poly):
    if poly is None or poly.is_empty:
        return None
    largest = _largest_polygon(poly)
    if largest.is_empty:
        return None
    return [[float(x), float(y)] for x, y in largest.exterior.coords[:-1]]


def _poly_from_coords(coords):
    if not coords or len(coords) < 3:
        return None
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    return _largest_polygon(poly)


def _locked_tail_pieces(layout, ext_active):
    lock = layout.tail.get("tail_geometry_lock") if layout.tail else None
    if not lock or bool(lock.get("tail_extension_enabled")) != bool(ext_active):
        return []
    pieces = []
    for coords in lock.get("polygons", []):
        poly = _poly_from_coords(coords)
        if poly is not None and not poly.is_empty:
            pieces.append(poly)
    return pieces


def _store_tail_lock(layout, tail_pieces, ext_active):
    polygons = []
    for poly in tail_pieces:
        coords = _poly_to_coords(poly)
        if coords:
            polygons.append(coords)
    layout.tail["tail_geometry_lock"] = {
        "tail_extension_enabled": bool(ext_active),
        "polygons": polygons,
    }


def _tail_shoulder_guard(outline_xy, tail, padding):
    p1 = np.asarray(outline_xy[0], dtype=float)
    p2 = np.asarray(outline_xy[1], dtype=float)
    t_hat = np.asarray(tail.get("t_hat"), dtype=float)
    n_out = np.asarray(tail.get("n_out"), dtype=float)
    if t_hat.shape != (2,) or n_out.shape != (2,):
        return None
    t_norm = float(np.linalg.norm(t_hat))
    n_norm = float(np.linalg.norm(n_out))
    if t_norm < 1e-9 or n_norm < 1e-9:
        return None
    t_hat = t_hat / t_norm
    n_out = n_out / n_norm

    tail_width = float(tail.get("width_mm") or 0.0)
    tail_length = float(tail.get("length_mm") or 0.0)
    half_width = tail_width / 2.0 + padding
    depth = tail_length + padding
    mid = 0.5 * (p1 + p2)
    a = mid - half_width * t_hat
    b = mid + half_width * t_hat
    c = b + depth * n_out
    d = a + depth * n_out
    return Polygon([a, b, c, d])


def _narrow_tail(tail) -> bool:
    """The tail was cut to its wires (narrow_tail.narrow_routed_tail), or is
    narrower than the connector it carries (see geometry.make_tail_poly)."""
    width = float((tail or {}).get("width_mm") or 0.0)
    conn = float((tail or {}).get("connector_min_width_mm") or 0.0)
    return (tail or {}).get("mode") == "fit" or 0.0 < width < conn - 1e-6


def _cable_piece(layout, outline_xy, ext_active):
    """The tail as the board takes it: without its connector landing when the
    connector has moved to the end of a drawn extension."""
    if not ext_active or not _narrow_tail(layout.tail):
        return layout.cable_poly
    return cable_without_landing(layout.cable_poly, outline_xy, layout.tail)


def _strip_aware(tail) -> bool:
    """The tail was narrowed by the strip-over-the-pads code (it records
    fit_offset_mm, geometry.fit_tail_strip). Designs saved before that keep the
    padding pass they were previewed with."""
    return (tail or {}).get("mode") == "fit" and "fit_offset_mm" in (tail or {})


def _tail_strip(layout, outline_xy, ext_active, cable_piece):
    """The part of a narrowed tail whose wires the padding pass leaves alone.

    Its own wires are held by the strip itself (narrow_tail.wire_edge_mm from
    its edge). Wire outside it -- the fan running in below the cable edge,
    or up a sloping outline beside the strip -- is padded like any other wire:
    the tail piece holds that wire only route_containment_mm out and cut off
    at the old tail's side, which against the padding just past it left a
    step, and a flare running into the step a horn."""
    t = layout.tail or {}
    if not _strip_aware(t) or t.get("t_hat") is None or t.get("n_out") is None:
        return cable_piece
    mid = 0.5 * (np.asarray(outline_xy[0], dtype=float) + np.asarray(outline_xy[1], dtype=float))
    t_hat = np.asarray(t["t_hat"], dtype=float)
    n_out = np.asarray(t["n_out"], dtype=float)
    width = float(t.get("width_mm") or 0.0)
    wide = float(t.get("connector_min_width_mm") or 0.0)
    cable = float(t.get("length_mm") or 0.0)
    total = cable + CONNECTOR_BODY_ABOVE_MM + CONNECTOR_BODY_BELOW_MM
    offset = tail_offset_mm(t)
    if width <= 0:
        return cable_piece
    if ext_active:
        # the strip runs on to the extension (cable_without_landing), whose
        # corridor holds the wire lanes turning with it
        strip_mid = mid + offset * t_hat
        strip = make_cable_rect(strip_mid, strip_mid, t_hat, n_out, width, total)
        corridor = _tail_extension_corridor(layout)
        return strip if corridor is None or corridor.is_empty else unary_union([strip, corridor])
    return make_tail_poly(mid, t_hat, n_out, width, wide, cable, total, offset, tail_step_fillet(t))


def cable_without_landing(cable_poly, outline_xy, tail):
    t = tail or {}
    if t.get("t_hat") is None or t.get("n_out") is None:
        return cable_poly
    mid = 0.5 * (np.asarray(outline_xy[0], dtype=float) + np.asarray(outline_xy[1], dtype=float))
    cable = float(t.get("length_mm") or 0.0)
    return tail_without_landing(cable_poly, mid, t["t_hat"], t["n_out"], float(t.get("width_mm") or 0.0),
                                float(t.get("connector_min_width_mm") or 0.0), cable,
                                cable + CONNECTOR_BODY_ABOVE_MM + CONNECTOR_BODY_BELOW_MM, tail_offset_mm(t),
                                tail_step_fillet(t), t.get("landing_cover") or ())


def fillet_tail_junction(board, outline_xy, tail, params, piece=None):
    """Round the two inside corners where a narrow tail leaves the board.

    Computed from the board itself on every pass rather than stored with the
    tail pieces, so it always fits the current board edge. Only whole fillet
    pieces touching the cable's sides are kept: other inside corners of the
    board keep the shape they have. The sides are the strip's straight edges,
    its step out to the tab (geometry.tail_side_edges) where the board reaches
    down past where it starts, and the edge of the tail `piece` itself between
    the cable edge and the landing, which is where the board meets the tail
    when the tail holds a wire beside the strip (a 4-layer inner feed running
    down the tail before it turns to its pad)."""
    r = max(0.0, float(getattr(params, "connector_shoulder_fillet_mm", 0.0) or 0.0))
    if r <= 0 or tail.get("t_hat") is None or tail.get("n_out") is None:
        return board
    t_hat = np.asarray(tail["t_hat"], dtype=float)
    n_out = np.asarray(tail["n_out"], dtype=float)
    mid = 0.5 * (np.asarray(outline_xy[0], dtype=float) + np.asarray(outline_xy[1], dtype=float))
    # the strip is centred off the connector's centre line when it carries an
    # off-centre block of pads (geometry.fit_tail_strip)
    offset = tail_offset_mm(tail)
    width = float(tail.get("width_mm") or 0.0)
    wide = float(tail.get("connector_min_width_mm") or 0.0)
    length = float(tail.get("length_mm") or 0.0)
    lo, hi = offset - width / 2.0, offset + width / 2.0
    lines = [[mid + s * t_hat, mid + s * t_hat + length * n_out] for s in (lo, hi)]
    near = [MultiLineString(lines)]
    if _strip_aware(tail) and 0.0 < width < wide - 1e-6:
        sides = tail_side_edges(width, wide, max(0.5, length - 1.0), offset, tail_step_fillet(tail))
        near.append(MultiLineString([[mid + s * t_hat + (length - u) * n_out for u, s in side]
                                     for side in sides if len(side) >= 2]))
        if piece is not None and not piece.is_empty:
            big = 1e4
            band = Polygon([mid - big * t_hat, mid + big * t_hat,
                            mid + big * t_hat + length * n_out, mid - big * t_hat + length * n_out])
            near.append(piece.boundary.intersection(band))
    return fillet_inside_corners(board, r, near=unary_union(near))


def route_containment_mm(params) -> float:
    """How far every wire centreline must sit inside the board edge: the
    post-route safety net (main._apply_post_route_containment) grows the board
    wherever a wire comes closer."""
    return (float(params.trace_w_mm) / 2 + float(params.board_edge_clear_mm)
            + max(0.5, float(params.routing_margin_mm) * 0.25))


def _taxel_clearance(layout, params):
    """The taxel field grown just enough that its copper clears the board
    edge (copper sits edge_clear_mm inside each taxel square). Grow-mode
    packing already keeps it well inside the drawn outline; this only holds
    for a field that reaches the outline."""
    P = np.asarray(layout.placements_xy if layout.placements_xy is not None else [], dtype=float).reshape(-1, 2)
    if len(P) == 0:
        return Polygon()
    w = float(getattr(params, "pixel_w_mm", 0.0)) / 2.0
    h = float(getattr(params, "pixel_h_mm", 0.0)) / 2.0
    grow = max(0.0, float(getattr(params, "board_edge_clear_mm", 0.0)) - float(getattr(params, "edge_clear_mm", 0.0)))
    field = unary_union([box(x - w, y - h, x + w, y + h) for x, y in P])
    return field.buffer(grow, join_style=2) if grow > 0 else field


def _solid_exterior(poly):
    largest = _largest_polygon(poly)
    if largest.is_empty:
        return Polygon()
    return Polygon(largest.exterior.coords)


def apply_main_follow_padding(layout, routes, params):
    """Replace final outline with main-layout route-follow shell plus unchanged tail."""
    if not getattr(params, "follow_main_padding", True):
        layout.tail["follow_main_padding"] = False
        layout.tail["follow_main_padding_mm"] = float(getattr(params, "follow_main_padding_mm", 2.0))
        layout.tail["smooth_follow_padding"] = bool(getattr(params, "smooth_follow_padding", True))
        layout.tail["final_outline_mode"] = "rectangular"
        return False

    padding = max(0.0, float(getattr(params, "follow_main_padding_mm", 2.0)))
    outline_xy = np.asarray(
        layout.sensor_outline_xy if layout.sensor_outline_xy is not None else [],
        dtype=float,
    )
    if len(outline_xy) < 3 or padding <= 0:
        layout.tail["follow_main_padding"] = bool(padding > 0)
        layout.tail["follow_main_padding_mm"] = padding
        layout.tail["final_outline_mode"] = "rectangular"
        return False

    outline_poly = layout.outline_poly if layout.outline_poly is not None else Polygon(outline_xy)
    if outline_poly.is_empty:
        layout.tail["follow_main_padding"] = True
        layout.tail["follow_main_padding_mm"] = padding
        layout.tail["final_outline_mode"] = "rectangular"
        return False

    n_out = layout.tail.get("n_out")
    if n_out is None:
        n_out = np.asarray(outline_xy[0], dtype=float) - np.asarray(outline_xy[1], dtype=float)
        n_out = np.array([-n_out[1], n_out[0]], dtype=float)
    ext = layout.tail.get("tail_extension_resolved", {})
    ext_active = bool(ext.get("enabled"))
    outward_mm = max(0.0, float(getattr(params, "cable_length_mm", 0.0)))
    if ext_active:
        p1 = np.asarray(outline_xy[0], dtype=float)
        p2 = np.asarray(outline_xy[1], dtype=float)
        edge_mid = 0.5 * (p1 + p2)
        n_vec = np.asarray(n_out, dtype=float)
        n_len = float(np.linalg.norm(n_vec))
        path = np.asarray(ext.get("path") or [], dtype=float)
        if n_len > 1e-9 and len(path) >= 1:
            n_vec = n_vec / n_len
            start_depth = float((path[0] - edge_mid) @ n_vec)
            outward_mm = max(outward_mm, start_depth)
    clip = _inboard_clip_polygon(outline_xy, n_out, layout.board_poly, routes, outward_mm=outward_mm)
    if clip.is_empty:
        layout.tail["follow_main_padding"] = True
        layout.tail["follow_main_padding_mm"] = padding
        layout.tail["final_outline_mode"] = "rectangular"
        return False

    smooth = bool(getattr(params, "smooth_follow_padding", True))
    cap_style = 1 if smooth else 2
    join_style = 1 if smooth else 2
    trace_buffer = float(getattr(params, "trace_w_mm", 0.0)) / 2.0 + padding
    # On 4 copper layers the rows leave the field on the inner layers, so no
    # wiring runs round most of the perimeter: the drawn shape keeps its own
    # edge there, and gets the padding only where wiring runs at it (the
    # column fan to the connector, a row kept on B.Cu). Every wire still
    # clears the edge. 2 layers: the whole shape is padded.
    four_layer = copper_layers_of(params, routes) == 4
    # Wires that stay inside get the safety net's distance, so it adds nothing
    # (+0.02: a buffer is a polygon, its arcs fall a hair inside the radius).
    edge_buffer = route_containment_mm(params) + 0.02
    padded = outline_poly.buffer(padding, cap_style=cap_style, join_style=join_style).intersection(clip)
    if four_layer:
        pieces = [outline_poly.intersection(clip), _taxel_clearance(layout, params).intersection(clip)]
        edge_band = padded.difference(outline_poly.buffer(-padding))
        at_edge = []
    else:
        pieces = [padded]

    # A narrowed tail already holds its wires at this padding (narrow_tail.py);
    # padding them again here would blend the tail into the board over the
    # closing radius below instead of the tail's own fillet.
    narrow = _narrow_tail(layout.tail)
    cable_piece = _cable_piece(layout, outline_xy, ext_active)
    tail_strip = (_tail_strip(layout, outline_xy, ext_active, cable_piece)
                  if narrow and cable_piece is not None else None)
    for route in routes:
        P = np.asarray(route.polyline_xy, dtype=float)
        if len(P) < 2:
            continue
        clipped = LineString(P).intersection(clip)
        if tail_strip is not None:
            clipped = clipped.difference(tail_strip)
        for line in _line_geoms(clipped):
            if line.length <= 1e-6:
                continue
            if not four_layer:
                pieces.append(line.buffer(trace_buffer, cap_style=cap_style, join_style=join_style))
                continue
            pieces.append(line.buffer(edge_buffer, cap_style=cap_style, join_style=join_style))
            for out in _line_geoms(line.difference(outline_poly)):
                if out.length > 1e-6:
                    pieces.append(out.buffer(trace_buffer, cap_style=cap_style, join_style=join_style))
            # an inner-layer wire near the edge is not perimeter wiring: a row
            # feed may start at a via by the edge and head into the field
            if getattr(route, "layer", None) not in INNER_LAYERS and line.intersects(edge_band):
                at_edge.append(line.intersection(edge_band))

    if four_layer and at_edge:
        # The shape's own padding along the stretch of edge that wiring runs
        # at, ramping back to the drawn edge past it (a step would read as a
        # notch): round the margin's ends, then close it against the shape
        # with a wide radius, kept inside the padding band and near it.
        margin = padded.intersection(unary_union(at_edge).buffer(2.0 * padding))
        margin = margin.buffer(-padding).buffer(padding)
        ramp = 5.0 * padding
        ramped = unary_union([outline_poly, margin]).buffer(ramp).buffer(-ramp)
        pieces.append(ramped.intersection(padded).intersection(margin.buffer(2.0 * ramp)))
    raw_main = unary_union([p for p in pieces if p is not None and not p.is_empty])
    close = max(padding * 1.25, float(getattr(params, "trace_w_mm", 0.0)) + padding)
    if close > 0:
        closed = raw_main.buffer(close, cap_style=cap_style, join_style=join_style)
        closed = closed.buffer(-close, cap_style=cap_style, join_style=join_style)
        if four_layer:
            # Only round the board where it grows past the drawn shape;
            # elsewhere the closing would fill the shape's own notches.
            grown = raw_main.difference(outline_poly)
            reach = grown.buffer(close) if not grown.is_empty else Polygon()
            closed = unary_union([closed.intersection(unary_union([raw_main, reach])), raw_main])
        raw_main = closed
    main_shell = _solid_exterior(raw_main)
    if main_shell.is_empty:
        layout.tail["follow_main_padding"] = True
        layout.tail["follow_main_padding_mm"] = padding
        layout.tail["final_outline_mode"] = "rectangular"
        return False

    tail_pieces = _locked_tail_pieces(layout, ext_active)
    if not tail_pieces:
        # A narrow tail is already sized to keep its wires clear of the edge
        # (geometry.fit_tail_strip); the padded guard would widen it again.
        guard = None if narrow else _tail_shoulder_guard(outline_xy, layout.tail, padding)
        if not ext_active and guard is not None and not guard.is_empty:
            tail_pieces.append(guard)
        if cable_piece is not None and not cable_piece.is_empty:
            tail_pieces.append(cable_piece)
        ext_corridor = _tail_extension_corridor(layout)
        if ext_corridor is not None and not ext_corridor.is_empty:
            tail_pieces.append(ext_corridor)
        _store_tail_lock(layout, tail_pieces, ext_active)

    final = unary_union([main_shell, *tail_pieces]).buffer(0)
    if narrow:
        final = fillet_tail_junction(final, outline_xy, layout.tail, params, cable_piece)
    final_shell = _solid_exterior(final)
    if final_shell.is_empty:
        return False

    layout.board_poly = final_shell
    layout.final_outline_xy = remove_collinear_closed(
        np.array(final_shell.exterior.coords[:-1], dtype=float)
    )
    layout.routing_margin_applied_mm = max(float(layout.routing_margin_applied_mm), padding)
    layout.tail["follow_main_padding"] = True
    layout.tail["follow_main_padding_mm"] = padding
    layout.tail["smooth_follow_padding"] = smooth
    layout.tail["follow_padding_tail_reach_mm"] = outward_mm
    layout.tail["final_outline_mode"] = "follow"
    return True
