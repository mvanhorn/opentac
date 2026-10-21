"""JSON manifest export for MCU."""

import json
import numpy as np
from opentac.cad.export.connector_geometry import stitched_outline_segments
from opentac.cad.export.design import design_block
from opentac.cad.pipeline.bump_sheet import bump_centers_from_taxels, bump_index_per_point
from opentac.cad.pipeline.types import Layout


def _edge_cuts_points(layout, outline_xy, params, connector_dir):
    """The board's real cut boundary up to the connector, as one closed ring of
    points: the grown board and the cable tail, closed straight across the
    connector's open edge. The connector footprint draws the tab beyond that
    edge (the Gerber Edge.Cuts has the whole cut line).

    Built from `stitched_outline_segments` — the exact same call the KiCad and
    Gerber exporters use — rather than re-deriving a polygon here, because the
    manufactured edge is not the drawn outline: it carries the grown board, the
    cable tail and the connector-body envelope, with the connector's landing
    zone cut away. Do not re-derive "the board's physical edge" from raw
    outline_xy; sharing the exporters' own source keeps the manifest and the
    Gerbers in agreement.
    """
    if params is None or layout.final_outline_xy is None:
        return []
    try:
        segments = stitched_outline_segments(layout, outline_xy, params, connector_dir)
    except Exception:
        segments = None
    if segments is None:
        final_xy = np.asarray(layout.final_outline_xy, dtype=float)
        segments = [(final_xy[e], final_xy[(e + 1) % len(final_xy)])
                    for e in range(len(final_xy))]
    # Segments come as (start, end) pairs walking the boundary; emit the start
    # of each, plus the end of a segment the walk does not continue from: the
    # walk is open at the connector (the opening segment is dropped), and
    # emitting only starts would lose that corner, so the ring would cut
    # diagonally across the tail to the far side of the opening.
    ring = []
    for i, (a, b) in enumerate(segments):
        ring.append(a)
        nxt = segments[(i + 1) % len(segments)][0]
        if np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(nxt, dtype=float)) > 1e-6:
            ring.append(b)
    return [[round(float(x), 4), round(float(y), 4)] for x, y in ring]


def _shear_block(layout, active_pix, pixel_xy, params):
    """Which taxels share a shear bump — the grouping, not a per-taxel flag.

    A flag would say almost nothing: every 2x2 block holding any taxel gets a
    full-size bump (`bump_centers_from_taxels`), so "may have a bump" is true
    of every taxel. What no consumer can
    recover on its own is the grouping, and that is what shear actually needs:
    a bump converts a sideways load into an asymmetric normal load across the
    four taxels beneath it, so the direction only exists as a differential
    within one group.

    It cannot be re-derived from the fields already here, either. `pixels`
    carries row/col indices in pad order, which is deliberately not geometric
    order (split_row_pin_order reverses the bottom half of the rows), so
    grouping by `index // 2` pairs taxels that are nowhere near each other —
    the reason the bump sheet derives row/col from position instead (see
    bump_centers_from_taxels).

    Centres come from the taxel grid, on `params.pitch_*_mm` — the same two numbers `/bump-sheet`
    passes — never on the manifest's own `px`/`py`, which are pad-order
    derived and can be negative.

    This describes the bumps' plan geometry only. Bump height, base fraction
    and taper are not chosen until `/bump-sheet` or `/case` runs, long after
    this file is written, so recording a guess would be worse than recording
    nothing.
    """
    if params is None or not pixel_xy:
        return None
    pitch_x = float(getattr(params, "pitch_x_mm", 0.0) or 0.0)
    pitch_y = float(getattr(params, "pitch_y_mm", 0.0) or 0.0)
    if pitch_x <= 0.0 or pitch_y <= 0.0:
        return None

    centers, _partial = bump_centers_from_taxels(
        layout.placements_xy, sorted(active_pix), pitch_x, pitch_y)
    if not len(centers):
        return None

    return {
        # Which rule produced the centres: always the uniform 2x2 taxel grid.
        # The key is kept so readers written for older manifests still find it.
        "src": "taxel_2x2",
        "bumps": [[round(float(x), 4), round(float(y), 4)] for x, y in centers],
        "b": bump_index_per_point([[p[0], p[1]] for p in pixel_xy],
                                  centers, pitch_x, pitch_y),
    }


_SIM_README_LINES = (
    "'sim3d' places this sensor in 3D on the part it was unrolled from, for"
    " simulation. Its coordinates are in the SOURCE MESH's frame, NOT the"
    " flat cable frame every other field uses - do not overlay the two.",
    "'sim3d.taxels3d' is index-parallel to 'pixels'. 'pos' lies ON the part"
    " surface; a contacting object meets the sensing surface, which is"
    " 'sim3d.stack.surface_to_bump_tip_mm' further along 'normal'.",
    "'sim3d.taxels3d.tangent' is the +column direction carried onto the"
    " surface, so shear resolved against it matches the column index.",
    "'sim3d.fit' reports how well the 2D drawing and the 3D map agreed."
    " Both numbers should be ~0; if not, 'sim3d.warnings' says what is wrong"
    " and the placement should not be trusted.",
)


def export_manifest(
    layout: Layout,
    removed_pix: set[int],
    pixel_w_mm: float,
    pixel_h_mm: float,
    outline_xy: np.ndarray,
    params=None,
    connector_dir=None,
    fill_region=None,
    sim3d=None,
) -> str:
    """Generate compact JSON manifest for MCU upload. Returns JSON string."""
    n_pix = len(layout.placements_xy)
    active_pix = set(range(n_pix)) - removed_pix

    # Separate row/col pads
    row_pads = []
    row_nets = []
    col_pads = []
    col_nets = []
    for pad_num, net in sorted(layout.pad_to_net.items()):
        if net.startswith("ROW_"):
            row_pads.append(pad_num)
            row_nets.append(net)
        else:
            col_pads.append(pad_num)
            col_nets.append(net)

    n_rows = len(row_pads)
    n_cols = len(col_pads)

    # Build bitmask
    mask = [[0] * n_cols for _ in range(n_rows)]
    cell_xy = [[None] * n_cols for _ in range(n_rows)]

    for pk in active_pix:
        r_idx = None
        c_idx = None
        row_net = layout.row_net_per_pixel[pk]
        col_net = layout.col_net_per_pixel[pk]
        for ri, rn in enumerate(row_nets):
            if rn == row_net:
                r_idx = ri
                break
        for ci, cn in enumerate(col_nets):
            if cn == col_net:
                c_idx = ci
                break
        if r_idx is not None and c_idx is not None:
            mask[r_idx][c_idx] = 1
            cell_xy[r_idx][c_idx] = layout.placements_xy[pk]

    # Derive pitch
    pitch_x = pixel_w_mm + 0.2
    pitch_y = pixel_h_mm + 0.2
    for ci in range(n_cols - 1):
        for ri in range(n_rows):
            if mask[ri][ci] and mask[ri][ci + 1]:
                pitch_x = cell_xy[ri][ci + 1][0] - cell_xy[ri][ci][0]
                break
        if pitch_x != pixel_w_mm + 0.2:
            break
    for ri in range(n_rows - 1):
        for ci in range(n_cols):
            if mask[ri][ci] and mask[ri + 1][ci]:
                pitch_y = cell_xy[ri + 1][ci][1] - cell_xy[ri][ci][1]
                break
        if pitch_y != pixel_h_mm + 0.2:
            break

    # Origin
    ox = oy = 0.0
    for ri in range(n_rows):
        for ci in range(n_cols):
            if mask[ri][ci]:
                ox = cell_xy[ri][ci][0] - ci * pitch_x
                oy = cell_xy[ri][ci][1] - ri * pitch_y
                break
        if ox != 0.0:
            break

    # Bitmask strings
    mask_rows = [''.join(str(c) for c in row) for row in mask]

    # Explicit geometry, so the sensor drawing can be reproduced exactly rather
    # than inferred from the grid fields — which cannot do it.
    #
    # The mask's rows are indexed by *pad order*, because this function walks
    # sorted(layout.pad_to_net). Pad order is not geometric order:
    # split_row_pin_order (pipeline/layout.py) assigns the top half of the rows
    # in geometric order and then the bottom half reversed, so the array's pads
    # meet the routing on the correct side. `oy + ri*py` assumes mask index and
    # geometric index are the same thing, so every row in that reversed bottom
    # half lands at the wrong Y (the bottom half comes out mirrored).
    #
    # On a full rectangular array this is invisible if you only check the point
    # cloud — mirroring evenly spaced rows maps the set of Y values onto itself
    # — but the row *indexing* is still wrong, and as soon as the rows have
    # different bit patterns the positions are wrong too.
    #
    # The grid fields keep pad order: the MCU scans by pad, so pad order is the
    # right order for `m`/`rp`/`cp`. `pixels` is the authority on where each
    # one physically sits.
    pixel_xy = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            if mask[ri][ci]:
                x, y = cell_xy[ri][ci]
                pixel_xy.append([round(float(x), 4), round(float(y), 4), ri, ci])

    outline_pts = []
    if outline_xy is not None:
        outline_pts = [[round(float(x), 4), round(float(y), 4)]
                       for x, y in np.asarray(outline_xy, dtype=float)]

    edge_cuts_pts = _edge_cuts_points(layout, outline_xy, params, connector_dir)
    # Built from `pixel_xy`, so `shear["b"]` is index-parallel to `pixels`.
    _shear = _shear_block(layout, active_pix, pixel_xy, params)

    # Documentation lives in the file itself, as string fields rather than
    # `//` comments: JSON has no comment syntax, and this file is meant to be
    # re-read (by OpenTac-CAD to rebuild a design, and by whatever drives the
    # MCU), so json.loads / JSON.parse must keep working on it. Unknown keys are
    # ignored by every consumer, so a "_readme" costs nothing but is visible
    # the moment someone opens the file.
    readme = [
        "OpenTac-CAD sensor manifest. Lengths in mm.",
        "Coordinates are Y-UP and share one frame (cable edge horizontal at the"
        " bottom), so 'outline', 'edge_cuts' and 'pixels' overlay directly."
        " The KiCad export negates Y.",
        "'edge_cuts' is the board that gets MANUFACTURED (grown board + cable"
        " tail), closed straight across the connector's open edge; the"
        " connector tab beyond it is in the Gerber Edge.Cuts. 'outline' is"
        " only the shape the user drew - smaller, and not the cut boundary.",
        "'pixels' is [x, y, row_index, col_index] per active taxel and is the"
        " authority on where taxels physically sit.",
        "'ox'/'oy'/'px'/'py'/'m'/'rp'/'cp' are for MCU scan order, NOT"
        " geometry: their row index is pad order, which is deliberately not"
        " geometric order, so 'py' can be negative and ox+ci*px does not"
        " reproduce positions. Use 'pixels' for geometry.",
        "'w'/'h' are one taxel's footprint size; 'footprint' holds the rest of"
        " the parameters needed to redraw its interdigitated copper.",
        "'shear' says which taxels share a shear bump. 'shear.bumps' are bump"
        " centres in the same frame as 'pixels'; 'shear.b' is parallel to"
        " 'pixels' and gives each taxel's bump index, or -1 for a taxel with"
        " no bump above it (pressure only, no shear). Shear direction is the"
        " differential across the taxels of one group, signed by (pixel -"
        " bump centre); a group of fewer than 4 is a block the outline cut"
        " short and is degenerate on any axis with taxels on one side only."
        " The bump SHEET is optional and its height/taper are chosen when it"
        " is built, so they are not here - this is only where bumps would go.",
        *( _SIM_README_LINES if sim3d else () ),
        "'design' is the RECIPE: every input OpenTac-CAD needs to rebuild"
        " this board. Open this file in the app to keep editing the design."
        " Everything above it describes the RESULT and is ignored on import."
        " design.json, beside this file in the ZIP, additionally restores"
        " hand-edited routes and is tied to the pipeline version that wrote"
        " it; 'design' here survives a version change.",
    ]
    data = {
        "_readme": readme,
        # 4 added the "design" block, 5 the "shear" block, 6 the optional
        # "sim3d" block (written only when a caller supplies it). All are
        # additive (no key removed or given a new meaning), so MCU code
        # written against v3 keeps working; importers key off design.schema,
        # not this.
        "v": 6,
        "units": "mm",
        # --- sensor drawing: outline, pixel centers, pixel size ---
        # Same coordinate frame as each other (the canonical frame
        # rotate_for_cable() produces, with the cable edge horizontal at the
        # bottom), so they overlay directly. This frame is Y-up; the KiCad
        # export negates Y on the way out.
        "outline": outline_pts,      # the shape the user drew
        # The board's real cut boundary — grown board, cable tail and
        # connector envelope included, connector landing zone cut away. This,
        # not "outline", is what gets manufactured, and it is larger than the
        # drawn shape.
        "edge_cuts": edge_cuts_pts,
        "pixels": pixel_xy,          # [x, y, row_index, col_index] per active pixel
        "w": round(pixel_w_mm, 2),   # pixel width  (footprint, not pitch)
        "h": round(pixel_h_mm, 2),   # pixel height
        # Everything else needed to redraw one pixel's interdigitated copper.
        # w/h alone only give its bounding box; these are the parameters
        # kicad_footprint.pixel_geometry() takes, so the footprint can be
        # rebuilt exactly from this file.
        **({"footprint": {
            "trace_w": round(params.trace_w_mm, 4),
            "gap": round(params.gap_mm, 4),
            "center_clear": round(params.center_clear_mm, 4),
            "edge_clear": round(params.edge_clear_mm, 4),
            "via_drill": round(params.via_drill_mm, 4),
            "via_dia": round(params.via_dia_mm, 4),
        }} if params is not None else {}),
        # Which taxels sit under the same shear bump. Geometry, so it belongs
        # with the drawing above rather than with the scan order below.
        **({"shear": _shear} if _shear is not None else {}),
        # Where this sensor sits on the part it was unrolled from: per-taxel
        # 3D position, normal and tangent, plus the z-stack. Optional: written
        # only when the caller supplies `sim3d` (built from the source mesh);
        # a flat, hand-drawn or DXF board has no 3D placement.
        **({"sim3d": sim3d} if sim3d else {}),
        # --- matrix scanning: the MCU reads these ---
        "r": n_rows,
        "c": n_cols,
        "px": round(pitch_x, 2),
        "py": round(pitch_y, 2),
        "ox": round(ox, 3),
        "oy": round(oy, 3),
        "rp": row_pads,
        "cp": col_pads,
        "m": mask_rows,
    }

    # --- re-opening the design: every /generate input, so this file can be
    # loaded back into OpenTac-CAD and edited further. Last, because everything above
    # describes the manufactured result and this describes how to rebuild it.
    if params is not None:
        data["design"] = design_block(layout, params, removed_pix, fill_region)

    return json.dumps(data, indent=2)
