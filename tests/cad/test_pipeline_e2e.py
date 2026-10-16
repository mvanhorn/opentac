"""End-to-end pipeline test with the default square shape."""

import sys
import os

import numpy as np
from opentac.cad.pipeline.types import Params
from shapely.geometry import LineString
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


def test_default_square():
    outline = np.array([[-20, -20], [20, -20], [20, 20], [-20, 20]], dtype=float)
    params = Params()

    print("1. Layout...")
    layout = layout_with_connector(outline, params)
    n_pix = len(layout.placements_xy)
    print(f"   {n_pix} pixels packed, connector {layout.connector_pos}-pos, swap={layout.connector_swap}")

    print("2. Route columns...")
    col_routes, col_removed = route_columns_grid(layout, outline, params)
    print(f"   {len(col_routes)} col routes, {len(col_removed)} removed")

    print("3. Route rows...")
    row_routes, row_removed = route_rows_grid(layout, outline, params, col_removed)
    print(f"   {len(row_routes)} row routes, {len(row_removed)} removed")

    removed = col_removed | row_removed
    if row_removed - col_removed:
        print("   Re-routing columns due to row removals...")
        col_routes, cr2 = route_columns_grid(layout, outline, params, removed)
        removed |= cr2

    print("4. Validate...")
    for pass_num in range(10):
        bad_pix, incomplete = validate_routes(
            layout, col_routes, row_routes, removed,
            outline, params.pad_pitch_mm, params.trace_w_mm, params.clearance_mm)
        if not bad_pix:
            print(f"   Pass {pass_num + 1}: OK")
            break
        print(f"   Pass {pass_num + 1}: removing {len(bad_pix)} pixels")
        removed |= bad_pix
        col_routes, cr = route_columns_grid(layout, outline, params, removed)
        row_routes, rr = route_rows_grid(layout, outline, params, removed)
        removed |= cr | rr

    print("5. DRC...")
    violations = drc_check(col_routes, row_routes, params.trace_w_mm, params.clearance_mm)
    print(f"   {len(violations)} violations")

    n_active = n_pix - len(removed)
    print(f"\n   RESULT: {n_active} active pixels, {len(removed)} removed, "
          f"{len(col_routes)} col + {len(row_routes)} row routes")

    print("6. KiCad export...")
    connector_dir = os.path.join(os.path.dirname(__file__), '..', 'connectors')
    pcb_text = generate_kicad_pcb(
        layout, outline, col_routes, row_routes, removed, params, connector_dir)
    print(f"   PCB: {len(pcb_text)} chars")

    print("7. Manifest...")
    manifest = export_manifest(layout, removed, params.pixel_w_mm, params.pixel_h_mm, outline)
    print(f"   Manifest: {len(manifest)} chars")

    print("8. SVG preview...")
    svg = generate_svg(layout, outline, col_routes, row_routes, removed, params)
    print(f"   SVG: {len(svg)} chars")

    print("9. Gerber files...")
    gerbers = generate_gerbers(layout, outline, col_routes, row_routes, removed, params)
    for name, content in gerbers.items():
        print(f"   {name}: {len(content)} chars")

    print("10. Drill file...")
    drill = generate_drill_file(layout, removed, params)
    print(f"   Drill: {len(drill)} chars")

    print("\n=== TEST PASSED ===")


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
    up — so the surviving array is a clean staircase, never a grid with holes.
    """
    outline = np.array([[-half, -half], [half, -half], [half, half], [-half, half]], dtype=float)
    params = Params(board_mode="fixed_keepout", auto_expand_board=False)

    print("\n=== FIXED KEEP-OUT MODE ===")
    print("1. Baseline (max packing)...")
    n_max = max_pack_count(outline, params)
    print(f"   {n_max} pixels at max packing")

    print("2. Keep-out layout...")
    layout, profile, info = layout_fixed_shape_keepout(outline, params)
    n_pix = len(layout.placements_xy)
    summary = profile.summary()
    print(f"   {n_pix} pixels, keep-out {summary['min_mm']:.2f}..{summary['max_mm']:.2f} mm "
          f"(mean {summary['mean_mm']:.2f}), {info['iterations']} carve iterations, "
          f"converged={info['converged']}")

    assert info["converged"], "keep-out fixed point did not converge"
    assert 0 < n_pix < n_max, f"expected fewer than {n_max} pixels, got {n_pix}"
    # Deliberately NOT compared against routing_margin_mm. That parameter is
    # grow-board mode's outward expansion; using it as this band's floor pinned
    # every sample at 3.0mm and masked the shape-following model completely (see
    # keepout.fixed_keepout_floor). The real invariant is that wherever wiring
    # actually runs, the band carries at least the model's own safety term.
    safety = max(1.0, 2 * (params.trace_w_mm + params.clearance_mm))
    assert summary["max_mm"] >= safety - 1e-9, "keep-out narrower than the wiring safety term"
    assert summary["min_mm"] >= 0.0, "negative keep-out width"
    assert summary["max_mm"] > summary["min_mm"] + 1e-6, "keep-out did not taper"

    print("3. Outline is unchanged...")
    # board_poly is the drawn outline unioned with the cable tail; the outline
    # itself must not have grown in x, and must not have grown upward in y.
    minx, miny, maxx, maxy = layout.board_poly.bounds
    o_minx, o_miny, o_maxx, o_maxy = layout.outline_poly.bounds
    assert abs(minx - o_minx) < 1e-6 and abs(maxx - o_maxx) < 1e-6, "board grew sideways"
    assert maxy <= o_maxy + 1e-6, "board grew past the outline"
    print(f"   outline x[{o_minx:.2f},{o_maxx:.2f}] board x[{minx:.2f},{maxx:.2f}] (tail extends -y only)")

    print("4. Grid is top-aligned and has no interior holes...")
    P = layout.placements_xy
    ys = np.unique(np.round(P[:, 1], 6))
    top_gap = o_maxy - (ys[-1] + params.pixel_h_mm / 2)
    assert top_gap >= summary["min_mm"] - 1e-6, "top row intrudes into the keep-out"
    n_cols, n_rows = _assert_contiguous_grid(P)
    print(f"   up to {n_cols} cols x {n_rows} rows, no holes, {top_gap:.2f} mm clear above the top row")

    print("5. Route + validate (nothing gets deleted)...")
    col_routes, col_problem = route_columns_grid(layout, outline, params)
    row_routes, row_problem = route_rows_grid(layout, outline, params)
    bad_pix, incomplete = validate_routes(
        layout, col_routes, row_routes, set(), outline,
        params.pad_pitch_mm, params.trace_w_mm, params.clearance_mm)
    print(f"   {len(col_routes)} col + {len(row_routes)} row routes, "
          f"{len(col_problem | row_problem | bad_pix)} problem pixels, {len(incomplete)} warning nets")
    assert not (col_problem | row_problem | bad_pix), "fixed keep-out board did not route cleanly"

    print("6. All copper inside Edge.Cuts...")
    # The whole promise of a fixed outline: nothing may be solved by growing the
    # board, so every trace has to fit inside it with clearance.
    required = params.trace_w_mm / 2 + params.board_edge_clear_mm
    worst = min(
        float(LineString(np.asarray(r.polyline_xy, dtype=float)).distance(layout.board_poly.boundary))
        for r in [*col_routes, *row_routes] if len(r.polyline_xy) >= 2
    )
    assert all(layout.board_poly.covers(LineString(np.asarray(r.polyline_xy, dtype=float)))
               for r in [*col_routes, *row_routes] if len(r.polyline_xy) >= 2), "copper outside Edge.Cuts"
    print(f"   closest copper-to-edge {worst:.3f} mm (need {required:.3f})")
    assert worst >= required - 1e-6, "copper too close to the board edge"

    print("7. DRC...")
    violations = drc_check(col_routes, row_routes, params.trace_w_mm, params.clearance_mm,
                           outline_xy=outline, board_edge_clear_mm=params.board_edge_clear_mm,
                           board_poly=layout.board_poly)
    print(f"   {len(violations)} violations")
    assert not violations

    print("\n=== TEST PASSED ===")


if __name__ == "__main__":
    test_default_square()
    test_fixed_keepout_square()
