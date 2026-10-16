"""Cut the cable tail down to the wires it carries (Params.narrow_tail).

The board is laid out and routed with a tail as wide as the connector, exactly as
without this option, so the taxels and the wires are the same either way. Only
then is the tail narrowed: a strip just wide enough for its wires, over the
connector pads in use (which sit off the connector's centre whenever rows and
columns differ in number), stepping out to the connector's tab right before the
connector (geometry.make_tail_poly), plus the board around any wire that fans in
inside the tail rather than in the sensor.
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from .follow_padding import fillet_tail_junction, route_containment_mm
from .geometry import (
    fillet_inside_corners, fit_tail_strip, make_tail_poly, remove_collinear_closed, tail_geometry,
    tail_widening_length,
)
from .layout import _largest_polygon, cable_shoulder_fill, shoulder_fill_depth

# Straight cable a narrowed tail keeps between the board and its step: the
# follow-padding margin it leaves the board through, plus room to grip it. With
# the step at most fillet + corner (1.5 mm) long, every tail of the 12 mm
# minimum narrows fully; only a shorter typed length can be too short.
MIN_STRAIGHT_MM = 4.0


def _follow_padding_on(params) -> bool:
    return bool(params.auto_expand_board and getattr(params, "follow_main_padding", True)
                and float(getattr(params, "follow_main_padding_mm", 0.0) or 0.0) > 0)


def wire_edge_mm(params) -> float:
    """How far the tail's edge sits from its outermost wire: the padding the rest
    of a grown board gets (the follow-padding shell around the tail's own wires
    is then exactly the strip), or on a fixed outline the edge distance every
    wire is held to."""
    if _follow_padding_on(params):
        return float(params.trace_w_mm) / 2 + float(params.follow_main_padding_mm)
    return route_containment_mm(params)


def step_fillet_mm(params) -> float:
    """The inside radius of the narrow tail's step to the connector's tab: the
    one its corners where it leaves the board get (fillet_tail_junction)."""
    return max(0.0, float(getattr(params, "connector_shoulder_fillet_mm", 0.0) or 0.0))


def strip_within_cable(width: float, offset: float, wide: float, cable_length: float, fillet: float):
    """The strip (width, offset) if the cable is long enough to narrow, else None.

    The step back to the connector's tab (geometry.tail_side_edges) is at most
    `fillet` + TAIL_STEP_CORNER_MM long whatever its width, and the cable keeps
    MIN_STRAIGHT_MM of straight run past where it leaves the board (a case clips
    onto that run). A cable shorter than both keeps the connector's width."""
    if width >= wide - 1e-6:
        return width, offset
    if cable_length >= tail_widening_length(width, wide, float("inf"), offset, fillet) + MIN_STRAIGHT_MM:
        return width, offset
    return None


def narrow_routed_tail(layout, routes, params) -> bool:
    """Replace the routed board's connector-width tail by the narrow one.

    Only an automatic tail is narrowed: a width the user typed, or side padding,
    is kept as given. Returns whether the tail changed."""
    tail = layout.tail
    if not getattr(params, "narrow_tail", True) or layout.cable_poly is None:
        return False
    if tail.get("mode") != "auto":
        return False
    wide = float(tail.get("connector_min_width_mm") or 0.0)
    strip = fit_tail_strip(layout.pad_to_net, layout.connector_pos, params, wire_edge_mm(params))
    if strip is None:
        return False
    r = step_fillet_mm(params)
    strip = strip_within_cable(*strip, wide, float(params.cable_length_mm), r)
    if strip is None:
        return False
    fit, offset = strip
    tail.pop("landing_cover", None)
    # fit == wide (wires across the whole connector) still drops the padded
    # guard follow padding would add around a connector-width tail.

    outline_xy = np.asarray(layout.sensor_outline_xy, dtype=float)
    outline_poly = layout.outline_poly
    geom = tail_geometry(outline_poly, outline_xy[0], outline_xy[1], layout.connector_pos, params,
                         fit_width=fit, fit_offset=offset)
    mid, t_hat, n_out = geom["mid"], geom["t_hat"], geom["n_out"]
    piece = make_tail_poly(mid, t_hat, n_out, fit, wide, geom["cable_length"], geom["total_length"], offset,
                           step_fillet=r)

    # Wire the router ran outside the drawn outline in the old tail (or in the
    # shoulder fill beside it) keeps the board around it, and so does wire inside
    # the outline close enough to the cable edge to have counted on the old tail
    # below it: beside a narrower strip that edge is the board's edge again (a
    # column running along it just inside a fixed outline came within 0.05 mm).
    # On a grown board the padding pass then pads the wire outside the strip like
    # any other wire (follow_padding._tail_strip), so this only holds what the
    # strip's own wires need where the strip pokes past the drawn outline.
    contain = route_containment_mm(params)
    old = unary_union([layout.cable_poly, *cable_shoulder_fill(outline_poly, mid, t_hat, n_out, wide)])
    lines = [LineString(np.asarray(r.polyline_xy, dtype=float)) for r in routes if len(r.polyline_xy) >= 2]
    near_old = old.difference(outline_poly).buffer(contain)
    outside = unary_union(lines).intersection(near_old) if lines else None
    if outside is not None and not outside.is_empty:
        # Buffer each piece and union the results, never the MultiLineString in one
        # call: hundreds of parallel wire pieces in one GEOS buffer is ~300 MB and
        # seconds (a 29 x 28 grown insole: +266 MB on Linux, +345 MB on macOS),
        # enough to get the 512 MB hosted server OOM-killed mid-request. Per piece
        # it is a few MB and a few ms, and the same region.
        pieces = shapely.get_parts(outside)
        cover = shapely.union_all(shapely.buffer(pieces, contain)).intersection(old)
        piece = unary_union([piece, cover]).buffer(0)
        # The part of it in the step and the landing, kept for when a drawn
        # tail extension takes them to the moved connector: the strip running
        # on in their place (geometry.tail_without_landing) still has to hold
        # a wire that comes down beside it (a fixed outline's pad fan turns in
        # within millimetres of the pads).
        start = geom["cable_length"] - tail_widening_length(fit, wide, max(0.5, geom["cable_length"] - 1.0),
                                                            offset, r) - contain
        big = 1e4
        beyond = Polygon([mid - big * t_hat + start * n_out, mid + big * t_hat + start * n_out,
                          mid + big * t_hat + big * n_out, mid - big * t_hat + big * n_out])
        landing_cover = [g for g in getattr(cover.intersection(beyond), "geoms", [cover.intersection(beyond)])
                         if g.geom_type == "Polygon" and g.area > 1e-6]
        if landing_cover:
            tail["landing_cover"] = [np.round(np.asarray(g.exterior.coords[:-1], dtype=float), 4).tolist()
                                     for g in landing_cover]
    if r > 0:
        piece = fillet_inside_corners(piece, r)
    piece = _largest_polygon(piece)

    # The fill beside the strip reaches as far into the board side as the one
    # the wires were routed in: sized by the strip it would stop short of a wire
    # the router ran up a sloping outline above it.
    shoulder_depth = shoulder_fill_depth(wide)
    tail.update({"mode": "fit", "width_mm": fit, "fit_width_mm": fit, "fit_offset_mm": offset,
                 "routed_width_mm": wide, "shoulder_depth_mm": shoulder_depth, "step_fillet_mm": r})
    layout.cable_poly = piece

    if not params.auto_expand_board:
        # A fixed outline: the board is the drawn shape plus the tail, so rebuild
        # it (the same union layout_with_connector and edit-data replay make).
        # Grown boards are rebuilt from the tail by follow padding.
        board = unary_union([outline_poly, piece.buffer(0.001, join_style=2),
                             *cable_shoulder_fill(outline_poly, mid + offset * t_hat, t_hat, n_out, fit,
                                                  depth=shoulder_depth)])
        filleted = fillet_tail_junction(board, outline_xy, tail, params, piece)
        # Keep the junction fillets in the tail itself so the edit-data replay,
        # which rebuilds this board from the outline and the tail, has them too.
        extra = filleted.difference(board)
        if not extra.is_empty:
            layout.cable_poly = _largest_polygon(unary_union([piece, extra.buffer(0.001, join_style=2)]))
        layout.board_poly = filleted
        final = _largest_polygon(filleted)
        layout.final_outline_xy = remove_collinear_closed(np.array(final.exterior.coords[:-1], dtype=float))
    return True
