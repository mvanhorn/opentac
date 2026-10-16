"""Automatic cable tail length (Params.cable_length_auto).

A tail of a fixed 12 mm was too short for most sensors twice over: the narrow
tail could not flare back to the connector in it (a 60-pin tail over an
off-centre pad block needs 14-22 mm), and the printed case, which wraps the
board and lets the tail out through a slot, swallowed part of the connector body
or left it no flex to plug in with (23 of the 31 example designs).

The board is laid out and routed with the minimum tail, AUTO_TAIL_MIN_MM, so the
taxels and wires are exactly those of a 12 mm board. Only then is the tail made
as long as the sensor needs (main._auto_tail_length):

  * fit width -- long enough for the narrow tail to reach the width of the wires
    it carries: its flare at 45 degrees plus MIN_STRAIGHT_MM of straight cable
    (narrow_tail.strip_within_cable);
  * printed case clearance -- long enough that the printed case built with its
    default settings leaves the connector body clear with its own minimum of
    free flex (case.case_cable_check, the 2D half of build_case);

rounded up to TAIL_LENGTH_STEP_MM. Lengthening a routed tail needs no second
routing pass (stretch_routed_tail): the connector, its landing and its pads move
further out along the tail, and every wire that starts at a pad gets a straight
run of the same length into it, in the tail where nothing else runs.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.ops import unary_union

from .geometry import (
    fit_tail_strip, make_tail_poly, remove_collinear, remove_collinear_closed, tail_flare_length, tail_geometry,
)
from .layout import _largest_polygon
from .narrow_tail import MIN_STRAIGHT_MM, wire_edge_mm

AUTO_TAIL_MIN_MM = 12.0
TAIL_LENGTH_STEP_MM = 0.5

REASON_MIN = "minimum 12 mm"
REASON_FIT = "fit width"
REASON_CASE = "printed case clearance"
REASON_TYPED = "typed"


def round_up_length(length_mm: float) -> float:
    """Up to the next TAIL_LENGTH_STEP_MM."""
    return math.ceil(float(length_mm) / TAIL_LENGTH_STEP_MM - 1e-9) * TAIL_LENGTH_STEP_MM


def fit_tail_length(layout, params) -> float:
    """Cable the narrow tail needs to reach its full fit width: the flare back
    to the connector at 45 degrees plus MIN_STRAIGHT_MM. 0 for a tail that is
    not cut to its wires (narrow_tail off, or a width the user typed)."""
    if not getattr(params, "narrow_tail", True) or (layout.tail or {}).get("mode") != "auto":
        return 0.0
    strip = fit_tail_strip(layout.pad_to_net, layout.connector_pos, params, wire_edge_mm(params))
    wide = float((layout.tail or {}).get("connector_min_width_mm") or 0.0)
    if strip is None or strip[0] >= wide - 1e-6:
        return 0.0
    return tail_flare_length(strip[0], wide, float("inf"), strip[1]) + MIN_STRAIGHT_MM


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
    tail = layout.tail
    n_out = np.asarray(tail["n_out"], dtype=float)
    shift = float(extra_mm) * n_out
    for r in routes:
        P = np.asarray(r.polyline_xy, dtype=float)
        pad = np.asarray(r.pad_xy, dtype=float)
        if len(P) == 0 or np.linalg.norm(P[0] - pad) > 1.0:
            continue
        r.polyline_xy = remove_collinear(np.vstack([P[0] + shift, P]))
        r.pad_xy = pad + shift

    outline_xy = np.asarray(layout.sensor_outline_xy, dtype=float)
    geom = tail_geometry(layout.outline_poly, outline_xy[0], outline_xy[1], layout.connector_pos, params)
    layout.pad_y_mm = float(geom["pad_row_center"][1])
    tail["length_mm"] = float(params.cable_length_mm)
    tail["pad_row_center"] = np.asarray(geom["pad_row_center"], dtype=float).tolist()
    # The routed (connector-width) tail, as long as the new cable: the board the
    # rest of the pipeline narrows and pads from (layout_with_connector).
    cable = make_tail_poly(geom["mid"], geom["t_hat"], geom["n_out"], geom["tail_width"],
                           geom["connector_width"], geom["cable_length"], geom["total_length"])
    layout.cable_poly = cable
    layout.board_poly = unary_union([layout.board_poly, cable.buffer(0.001, join_style=2)])
    final = _largest_polygon(layout.board_poly)
    if not final.is_empty:
        layout.final_outline_xy = remove_collinear_closed(np.array(final.exterior.coords[:-1], dtype=float))
