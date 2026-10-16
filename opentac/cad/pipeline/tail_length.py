"""Automatic cable tail length (Params.cable_length_auto).

A tail of a fixed 12 mm was too short for most sensors: the printed case, which
wraps the board and lets the tail out through a slot, swallowed part of the
connector body or left it no flex to plug in with (23 of the 31 example designs).

The board is laid out and routed with the minimum tail, AUTO_TAIL_MIN_MM, so the
taxels and wires are exactly those of a 12 mm board. Only then is the tail made
as long as the sensor needs (main._auto_tail_length):

  * fit width -- long enough for the narrow tail to reach the width of the wires
    it carries: its step back to the connector plus MIN_STRAIGHT_MM of straight
    cable (narrow_tail.strip_within_cable). The step is at most 1.5 mm long, so
    the 12 mm minimum always has room; this only counts with a fillet radius
    (connector_shoulder_fillet_mm) far above its 1 mm default;
  * printed case clearance -- long enough that the printed case built with its
    default settings leaves the connector body clear with its own minimum of
    free flex (case.case_cable_check, the 2D half of build_case);

rounded up to TAIL_LENGTH_STEP_MM. Lengthening a routed tail needs no second
routing pass (stretch_routed_tail): the connector, its landing and its pads move
further out along the tail, and every wire that starts at a pad gets a straight
run of the same length into it, in the tail where nothing else runs. Redo padding
with a Tail length typed in does the same (main._relength_routed_tail), and the
reverse to shorten a tail (trim_routed_tail) as far as every wire's straight run
into its pad allows (tail_trim_room).
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .geometry import (
    fit_tail_strip, make_tail_poly, remove_collinear, remove_collinear_closed, tail_geometry,
    tail_widening_length,
)
from .layout import _largest_polygon, cable_shoulder_fill
from .narrow_tail import MIN_STRAIGHT_MM, step_fillet_mm, wire_edge_mm

AUTO_TAIL_MIN_MM = 12.0
TAIL_LENGTH_STEP_MM = 0.5
# The straight run every wire keeps into its pad when a routed tail is shortened
# without routing it again: the router's own shortest (a 4-layer row feed turns
# into its pad 1 mm before it).
TRIM_KEEP_MM = 1.0

REASON_MIN = "minimum 12 mm"
REASON_FIT = "fit width"
REASON_CASE = "printed case clearance"
REASON_TYPED = "typed"


def round_up_length(length_mm: float) -> float:
    """Up to the next TAIL_LENGTH_STEP_MM."""
    return math.ceil(float(length_mm) / TAIL_LENGTH_STEP_MM - 1e-9) * TAIL_LENGTH_STEP_MM


def fit_tail_length(layout, params) -> float:
    """Cable the narrow tail needs to reach its full fit width: its step back
    to the connector plus MIN_STRAIGHT_MM. 0 for a tail that is not cut to its
    wires (narrow_tail off, or a width the user typed)."""
    if not getattr(params, "narrow_tail", True) or (layout.tail or {}).get("mode") != "auto":
        return 0.0
    strip = fit_tail_strip(layout.pad_to_net, layout.connector_pos, params, wire_edge_mm(params))
    wide = float((layout.tail or {}).get("connector_min_width_mm") or 0.0)
    if strip is None or strip[0] >= wide - 1e-6:
        return 0.0
    return tail_widening_length(strip[0], wide, float("inf"), strip[1], step_fillet_mm(params)) + MIN_STRAIGHT_MM


def _starts_at_pad(route) -> bool:
    """The track begins at its connector pad (link tracks between split runs of
    a net start in the sensor field instead; apply_tail_extension's test)."""
    P = np.asarray(route.polyline_xy, dtype=float)
    return len(P) > 0 and np.linalg.norm(P[0] - np.asarray(route.pad_xy, dtype=float)) <= 1.0


def _move_pads(layout, routes, params, delta_mm: float) -> dict:
    """Move the connector `delta_mm` along the tail (out when positive), with
    every pad wire's end; returns the tail geometry at params' length."""
    tail = layout.tail
    n_out = np.asarray(tail["n_out"], dtype=float)
    shift = float(delta_mm) * n_out
    for r in routes:
        if not _starts_at_pad(r):
            continue
        P = np.asarray(r.polyline_xy, dtype=float)
        if delta_mm > 0:
            # a straight run of delta_mm from the moved pad back to where it was
            r.polyline_xy = remove_collinear(np.vstack([P[0] + shift, P]))
        else:
            # the pad slides back along the wire's own straight run into it
            # (tail_trim_room has checked that run is long enough)
            P = P.copy()
            P[0] = P[0] + shift
            r.polyline_xy = remove_collinear(P)
        r.pad_xy = np.asarray(r.pad_xy, dtype=float) + shift

    outline_xy = np.asarray(layout.sensor_outline_xy, dtype=float)
    geom = tail_geometry(layout.outline_poly, outline_xy[0], outline_xy[1], layout.connector_pos, params)
    layout.pad_y_mm = float(geom["pad_row_center"][1])
    tail["length_mm"] = float(params.cable_length_mm)
    tail["pad_row_center"] = np.asarray(geom["pad_row_center"], dtype=float).tolist()
    return geom


def _routed_tail(geom):
    """The tail as it is routed (connector width, or the width the user typed)."""
    return make_tail_poly(geom["mid"], geom["t_hat"], geom["n_out"], geom["tail_width"],
                          geom["connector_width"], geom["cable_length"], geom["total_length"])


def _set_board(layout, board) -> None:
    layout.board_poly = board
    final = _largest_polygon(board)
    if not final.is_empty:
        layout.final_outline_xy = remove_collinear_closed(np.array(final.exterior.coords[:-1], dtype=float))


def stretch_routed_tail(layout, routes, params, extra_mm: float) -> None:
    """Lengthen a routed tail by `extra_mm`, in place. `params` already carries
    the new cable_length_mm.

    The connector, its landing and its pad row move `extra_mm` further out along
    the tail. Every wire that starts at a pad (the same test apply_tail_extension
    uses: link tracks between split runs start in the sensor field) gets a
    straight run of `extra_mm` from its moved pad back to where its pad was, so
    the rest of the wire is untouched. The runs are as far apart as the pads,
    in the tail beyond the old pad row where no other copper is, so they keep
    every clearance the pads themselves keep."""
    if extra_mm <= 1e-9:
        return
    geom = _move_pads(layout, routes, params, extra_mm)
    # The routed (connector-width) tail, as long as the new cable: the board the
    # rest of the pipeline narrows and pads from (layout_with_connector).
    cable = _routed_tail(geom)
    layout.cable_poly = cable
    _set_board(layout, unary_union([layout.board_poly, cable.buffer(0.001, join_style=2)]))


def tail_trim_room(layout, routes) -> float:
    """How far a routed tail can be shortened without routing it again: every
    wire that starts at a pad must run straight along the tail into it, and keep
    TRIM_KEEP_MM of that run. 0 when any does not."""
    n_out = np.asarray(layout.tail["n_out"], dtype=float)
    room = math.inf
    for r in routes:
        if not _starts_at_pad(r):
            continue
        P = np.asarray(r.polyline_xy, dtype=float)
        if len(P) < 2:
            return 0.0
        run = P[1] - P[0]
        length = float(np.linalg.norm(run))
        if length < 1e-9 or float(run @ n_out) / length > -1.0 + 1e-6:
            return 0.0
        room = min(room, length - TRIM_KEEP_MM)
    return max(0.0, room) if math.isfinite(room) else 0.0


def trim_routed_tail(layout, routes, params, cut_mm: float) -> None:
    """Shorten a routed tail by `cut_mm`, in place: stretch_routed_tail in
    reverse. `params` already carries the new cable_length_mm, and
    tail_trim_room has allowed the cut.

    The connector, its landing and its pad row move `cut_mm` back along the
    tail, every pad wire's end sliding back along its own straight run into its
    pad. The board loses the old tail (and whatever was grown around the wires
    in it) and gets the routed tail at the new length."""
    if cut_mm <= 1e-9:
        return
    if cut_mm > tail_trim_room(layout, routes) + 1e-9:
        raise ValueError("the wires' straight runs into the connector are too short to shorten the tail")
    tail = layout.tail
    wide = max(float(tail.get("connector_min_width_mm") or 0.0), float(tail.get("width_mm") or 0.0))
    geom = _move_pads(layout, routes, params, -cut_mm)
    cable = _routed_tail(geom)
    layout.cable_poly = cable
    # everything past the cable edge across the old tail and a little beside it
    # (the containment grown round its outermost wires)
    mid, t_hat, n_out = geom["mid"], geom["t_hat"], geom["n_out"]
    half = wide / 2.0 + 2.0
    big = 1e4
    band = Polygon([mid - half * t_hat, mid + half * t_hat,
                    mid + half * t_hat + big * n_out, mid - half * t_hat + big * n_out])
    _set_board(layout, unary_union([layout.board_poly.difference(band), layout.outline_poly,
                                    cable.buffer(0.001, join_style=2),
                                    *cable_shoulder_fill(layout.outline_poly, mid, t_hat, n_out,
                                                         geom["tail_width"])]))
