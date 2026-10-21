"""End-to-end pipeline tests on a square outline."""

from pathlib import Path

import numpy as np
from shapely.geometry import LineString

import opentac.cad
from opentac.cad.pipeline.types import Params
from opentac.cad.pipeline.layout import layout_fixed_shape_keepout, layout_with_connector, max_pack_count
from opentac.cad.pipeline.route_columns import route_columns_grid
from opentac.cad.pipeline.route_rows import route_rows_grid
from opentac.cad.pipeline.validate import validate_routes
from opentac.cad.pipeline.drc import drc_check
from opentac.cad.export.kicad_pcb import generate_kicad_pcb
from opentac.cad.export.manifest import export_manifest
from opentac.cad.export.svg_preview import generate_svg
from opentac.cad.export.gerber import generate_gerbers
from opentac.cad.export.excellon import generate_drill_file

CONNECTOR_DIR = Path(opentac.cad.__file__).parent / "connectors"


def test_default_square_end_to_end():
    """The default square through layout, routing, validation, DRC and every
    export, with the connector footprint on the board."""
    outline = np.array([[-20, -20], [20, -20], [20, 20], [-20, 20]], dtype=float)
    params = Params()

    layout = layout_with_connector(outline, params)
    col_routes, col_removed = route_columns_grid(layout, outline, params)
    row_routes, row_removed = route_rows_grid(layout, outline, params, col_removed)

    removed = col_removed | row_removed
    if row_removed - col_removed:
        # Re-route the columns around the pixels the rows removed.
        col_routes, cr2 = route_columns_grid(layout, outline, params, removed)
        removed |= cr2

    for pass_num in range(10):
        bad_pix, incomplete = validate_routes(
            layout, col_routes, row_routes, removed,
            outline, params.pad_pitch_mm, params.trace_w_mm, params.clearance_mm)
        if not bad_pix:
            break
        removed |= bad_pix
        col_routes, cr = route_columns_grid(layout, outline, params, removed)
        row_routes, rr = route_rows_grid(layout, outline, params, removed)
        removed |= cr | rr

    violations = drc_check(col_routes, row_routes, params.trace_w_mm, params.clearance_mm)
    assert len(violations) == 0, f"{len(violations)} DRC violations: {violations[:4]}"

    assert CONNECTOR_DIR.is_dir()
    pcb_text = generate_kicad_pcb(
        layout, outline, col_routes, row_routes, removed, params, str(CONNECTOR_DIR))
    assert f'(footprint "{layout.connector_pos}_pos_0.5mm_FPC_resolved"' in pcb_text, \
        "the board was exported without its connector footprint"

    assert export_manifest(layout, removed, params.pixel_w_mm, params.pixel_h_mm, outline)
    assert generate_svg(layout, outline, col_routes, row_routes, removed, params)
    gerbers = generate_gerbers(layout, outline, col_routes, row_routes, removed, params)
    assert gerbers and all(gerbers.values())
    assert generate_drill_file(layout, removed, params)


def _assert_contiguous_grid(P):
    """No interior holes: every row and column is one unbroken run of pixels.

    The tapered keep-out lets the array be a staircase (wider up top, where few
    wires pass), so a strict rectangle is the wrong test — but a gap *inside* a
    row or column would mean pixels were punched out individually, which is
    exactly what this mode exists to avoid.
    """
    xs = np.unique(np.round(P[:, 0], 6))
    ys = np.unique(np.round(P[:, 1], 6))
    x_index = {x: i for i, x in enumerate(xs)}
    y_index = {y: i for i, y in enumerate(ys)}

    rows = {}
    cols = {}
    for x, y in np.round(P, 6):
        rows.setdefault(y, []).append(x_index[x])
        cols.setdefault(x, []).append(y_index[y])
    for y, idxs in rows.items():
        idxs = sorted(idxs)
        assert idxs == list(range(idxs[0], idxs[-1] + 1)), f"gap inside row y={y}"
    for x, idxs in cols.items():
        idxs = sorted(idxs)
        assert idxs == list(range(idxs[0], idxs[-1] + 1)), f"gap inside column x={x}"
    return len(xs), len(ys)


def test_fixed_keepout_square(half=35.0):
    """Fixed-outline keep-out mode.

    The drawn outline is the board: the sensor grid shrinks instead, and it
    shrinks by *not packing* pixels into the reserved wiring band rather than by
    deleting placed pixels. The band follows the outline's shape and tapers —
    thin up top where few wires pass, widest near the tail where they all pile
    up — so the surviving array is a staircase, never a grid with holes.
    """
    outline = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=float)
    params = Params(board_mode="fixed_keepout", auto_expand_board=False)

    # Baseline: the pixel count at maximum packing.
    n_max = max_pack_count(outline, params)

    layout, profile, info = layout_fixed_shape_keepout(outline, params)
    n_pix = len(layout.placements_xy)
    summary = profile.summary()

    assert info["converged"], "keep-out fixed point did not converge"
    assert 0 < n_pix < n_max, f"expected fewer than {n_max} pixels, got {n_pix}"
    # Not compared against routing_margin_mm, which is grow mode's outward
    # expansion: used as this band's floor, it would pin every sample at 3.0 mm
    # and mask the shape-following model (see keepout.fixed_keepout_floor). The
    # property checked is that wherever wiring runs, the band carries at least
    # the model's own safety term.
    safety = max(1.0, 2 * (params.trace_w_mm + params.clearance_mm))
    assert summary["max_mm"] >= safety - 1e-9, "keep-out narrower than the wiring safety term"
    assert summary["min_mm"] >= 0.0, "negative keep-out width"
    assert summary["max_mm"] > summary["min_mm"] + 1e-6, "keep-out did not taper"

    # board_poly is the drawn outline unioned with the cable tail; the outline
    # itself must not have grown in x, and must not have grown upward in y.
    minx, miny, maxx, maxy = layout.board_poly.bounds
    o_minx, o_miny, o_maxx, o_maxy = layout.outline_poly.bounds
    assert abs(minx - o_minx) < 1e-6 and abs(maxx - o_maxx) < 1e-6, "board grew sideways"
    assert maxy <= o_maxy + 1e-6, "board grew past the outline"

    # The grid is top-aligned and has no interior holes.
    P = layout.placements_xy
    ys = np.unique(np.round(P[:, 1], 6))
    top_gap = o_maxy - (ys[-1] + params.pixel_h_mm / 2)
    assert top_gap >= summary["min_mm"] - 1e-6, "top row intrudes into the keep-out"
    _assert_contiguous_grid(P)

    # Routing and validation delete nothing.
    col_routes, col_problem = route_columns_grid(layout, outline, params)
    row_routes, row_problem = route_rows_grid(layout, outline, params)
    bad_pix, incomplete = validate_routes(
        layout, col_routes, row_routes, set(), outline,
        params.pad_pitch_mm, params.trace_w_mm, params.clearance_mm)
    assert not (col_problem | row_problem | bad_pix), "fixed keep-out board did not route cleanly"

    # A fixed outline is never grown, so every trace fits inside it with
    # clearance.
    required = params.trace_w_mm / 2 + params.board_edge_clear_mm
    worst = min(
        float(LineString(np.asarray(r.polyline_xy, dtype=float)).distance(layout.board_poly.boundary))
        for r in [*col_routes, *row_routes] if len(r.polyline_xy) >= 2
    )
    assert all(layout.board_poly.covers(LineString(np.asarray(r.polyline_xy, dtype=float)))
               for r in [*col_routes, *row_routes] if len(r.polyline_xy) >= 2), "copper outside Edge.Cuts"
    assert worst >= required - 1e-6, "copper too close to the board edge"

    violations = drc_check(col_routes, row_routes, params.trace_w_mm, params.clearance_mm,
                           outline_xy=outline, board_edge_clear_mm=params.board_edge_clear_mm,
                           board_poly=layout.board_poly)
    assert not violations
