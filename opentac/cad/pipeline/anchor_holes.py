"""Anchor holes: Edge.Cuts cutouts at 2x2 block centres, for bump posts.

Why they exist. The shear bump sheet is held to the board by its skirt, at the
perimeter only. Under a strong tangential load the whole cover translates, and
because every bump translates with it the array reads a uniform shear field --
which is exactly what a real uniform shear load looks like. Bulk slip is
therefore not noise, it is degenerate with the measurand, and it ratchets:
stick-slip leaves the cover somewhere new and the zero walks.

A cutout under each bump lets the sheet's post pass through the board, so the
cover is keyed to it across the whole array instead of only at the rim. The
cutouts go on Edge.Cuts (milled openings), NOT drills -- nothing is plated and
nothing needs an annular ring, and the opening has to be a real hole in the
board for a post to pass.

Placement. One cutout per 2x2 taxel block, at the block centre -- the same
lattice `bump_sheet.bump_centers_from_taxels` already uses, so the post and the
hole cannot drift apart. In plan view that is the interstitial site:

        x   x        x = taxel
          o          o = cutout, bump sits directly over it
        x   x

What sets the pitch. Not the taxel copper -- the copper stops `edge_clear`
inside the taxel box, so its corner has room to spare. The binding constraint is
the intra-column bus: `export/kicad_pcb.py` joins every taxel in a column with a
continuous F.Cu trace at `pixel_centre_x + g.xR`, and for the left-hand column of
each block that bus runs straight past the cutout. See `required_half_gap_x`.

Measured, not assumed: the bus constraint applies to X only. The row bus on
B.Cu runs along the taxel centre line, a full half-pitch from the cutout, so the
Y pitch does not have to grow at all. Growing Y anyway costs taxels for no
clearance -- but leaving it makes the 2x2 block oblong, and the bump sized from
it elliptical, which biases shear sensitivity along one axis. `square_cells`
picks: True keeps the response isotropic, False keeps the taxels.
"""

from __future__ import annotations

import numpy as np

from opentac.cad.export.kicad_footprint import pixel_geometry


def _pixel_g(params):
    return pixel_geometry(params.pixel_w_mm, params.pixel_h_mm, params.trace_w_mm,
                          params.gap_mm, params.center_clear_mm,
                          params.edge_clear_mm, params.via_dia_mm)


def required_half_gap_x(params) -> float:
    """Half the intra-block X pitch, i.e. the cutout centre's X offset.

    Two constraints, both measured against real emitted geometry:

    bus     the column bus sits at +g.xR from the taxel centre and is
            `trace_w` wide, so its edge must clear the cutout edge.
    corner  the taxel copper corner sits at (w/2 - edge_clear); it is the
            diagonal distance to the cutout that matters, so this one couples
            X and Y and is checked in `_corner_ok` once both are known.
    """
    g = _pixel_g(params)
    r = params.anchor_hole_dia_mm / 2.0
    c = params.anchor_hole_clearance_mm
    return g.xR + params.trace_w_mm / 2.0 + r + c


def required_half_gap_y(params) -> float:
    """Half the intra-block Y pitch.

    The row bus runs along the taxel centre line, so it only needs the cutout
    to clear it; the taxels themselves need the pitch to exceed the taxel
    height plus a copper gap, which is what actually binds here.
    """
    r = params.anchor_hole_dia_mm / 2.0
    c = params.anchor_hole_clearance_mm
    bus = params.trace_w_mm / 2.0 + r + c
    taxel = (params.pixel_h_mm + params.gap_mm) / 2.0
    return max(bus, taxel)


def _corner_ok(params, hx: float, hy: float) -> bool:
    """Does the cutout clear the nearest taxel copper corner?"""
    r = params.anchor_hole_dia_mm / 2.0
    c = params.anchor_hole_clearance_mm
    cx = params.pixel_w_mm / 2.0 - params.edge_clear_mm
    cy = params.pixel_h_mm / 2.0 - params.edge_clear_mm
    return np.hypot(max(0.0, hx - cx), max(0.0, hy - cy)) >= r + c - 1e-9


def resolve_pitches(params) -> dict:
    """Wide/narrow pitch per axis for the requested lattice.

    Returns a dict with `wide_x/narrow_x/wide_y/narrow_y` plus the pattern
    lists the packer cycles through, and `pitch_x/pitch_y` -- the value the
    REST of the pipeline should keep calling `params.pitch_*_mm`. That value is
    the wide pitch, deliberately: every downstream use of pitch is a tolerance
    (`cluster_axis` at 0.55, bus-run segmentation at 1.25, the hug router's
    cell size), and overstating it is the safe direction for all three.
    """
    hx = required_half_gap_x(params)
    hy = required_half_gap_y(params)

    wide_x = 2.0 * hx
    wide_y = 2.0 * hy
    if params.anchor_hole_square_cells:
        wide_x = wide_y = max(wide_x, wide_y)
        hx = hy = wide_x / 2.0

    # The narrow (inter-block) gap carries no cutout, so it only has to keep
    # the taxels apart -- that is the whole density argument for "paired".
    tight_x = params.pixel_w_mm + params.gap_mm
    tight_y = params.pixel_h_mm + params.gap_mm

    if params.anchor_hole_lattice == "paired":
        narrow_x, narrow_y = tight_x, tight_y
    else:
        narrow_x, narrow_y = wide_x, wide_y

    # Never let the "narrow" step exceed the wide one: with a small taxel and a
    # big cutout the uniform pitch is already the wide one, and with a huge
    # taxel the taxel spacing dominates both.
    narrow_x = min(narrow_x, wide_x)
    narrow_y = min(narrow_y, wide_y)

    return {
        "wide_x": wide_x, "narrow_x": narrow_x,
        "wide_y": wide_y, "narrow_y": narrow_y,
        "pattern_x": [wide_x, narrow_x] if narrow_x < wide_x - 1e-9 else [wide_x],
        "pattern_y": [wide_y, narrow_y] if narrow_y < wide_y - 1e-9 else [wide_y],
        "pitch_x": wide_x, "pitch_y": wide_y,
        "half_gap_x": hx, "half_gap_y": hy,
        "corner_ok": _corner_ok(params, hx, hy),
    }


def _axis_lattice(vals: np.ndarray, tol: float) -> np.ndarray:
    """Distinct lattice coordinates along one axis, ascending.

    Groups values within `tol` of each other and returns each group's MEAN.
    Rounding to a multiple of `tol` instead would be a snap, not a
    de-duplication: at a 6.1mm pitch the tolerance is ~1.5mm, and quantising
    the columns to that grid moved every cutout by up to 0.175mm -- enough to
    eat half the clearance to the column bus, and invisible in DRC because
    DRC never looks at the cutouts.
    """
    v = np.sort(np.asarray(vals, dtype=float))
    if not len(v):
        return v
    groups = [[v[0]]]
    for x in v[1:]:
        if x - groups[-1][-1] > tol:
            groups.append([x])
        else:
            groups[-1].append(x)
    return np.asarray([float(np.mean(gp)) for gp in groups])


def _pair_midpoints(coords: np.ndarray, wide: float, narrow: float,
                    paired: bool) -> np.ndarray:
    """Midpoints of the gaps that carry a cutout.

    Paired lattice: measured, not counted. The wide gaps ARE the blocks, so
    finding them by spacing means the result cannot disagree with whatever the
    packer actually laid down -- including at a board edge that truncated a
    block, or after a phase search shifted the grid. A gap counts as wide when
    it is nearer the wide step than the narrow one.

    Uniform lattice: every gap is the same width, so there is nothing to
    measure and blocks have to be counted off instead. Counting runs from the
    HIGH end because `pack_pixels_in_polygon` anchors top-right; starting from
    the low end would pair columns across block boundaries whenever the count
    is odd, putting every cutout half a pitch off.
    """
    if len(coords) < 2:
        return np.zeros(0)
    if paired and wide - narrow > 1e-6:
        gaps = np.diff(coords)
        mid = (coords[:-1] + coords[1:]) / 2.0
        return mid[gaps >= 0.5 * (wide + narrow)]
    desc = coords[::-1]
    return np.asarray([(desc[i] + desc[i + 1]) / 2.0
                       for i in range(0, len(desc) - 1, 2)])[::-1]


def anchor_hole_centers(placements_xy, active_idx, params, pitches,
                        board_poly=None) -> tuple[np.ndarray, int]:
    """Cutout centres for the placed array, plus the count dropped at the edge.

    A cutout is kept only when its full disc plus clearance lies inside the
    board. A block the outline cut short still gets its bump (bump_sheet keeps
    those deliberately, so the surviving taxels still resolve shear) but it
    cannot always get its hole: near a concave edge the disc can hang over the
    boundary, and an Edge.Cuts circle that crosses the outline is not a hole,
    it is a bite out of the board edge. Those bumps simply lose their post.
    """
    P = np.asarray(placements_xy, dtype=float)
    idx = np.asarray(sorted(active_idx), dtype=int)
    if not len(idx):
        return np.zeros((0, 2)), 0
    pts = P[idx]

    paired = params.anchor_hole_lattice == "paired"
    xs = _axis_lattice(pts[:, 0], min(pitches["narrow_x"], pitches["wide_x"]) * 0.25)
    ys = _axis_lattice(pts[:, 1], min(pitches["narrow_y"], pitches["wide_y"]) * 0.25)
    hx = _pair_midpoints(xs, pitches["wide_x"], pitches["narrow_x"], paired)
    hy = _pair_midpoints(ys, pitches["wide_y"], pitches["narrow_y"], paired)
    if not len(hx) or not len(hy):
        return np.zeros((0, 2)), 0

    # Only blocks that actually hold a taxel. A cutout under empty board is a
    # hole with no bump over it -- structurally weakening for nothing.
    occupied = set(zip(np.round(pts[:, 0], 4), np.round(pts[:, 1], 4)))
    half_x = pitches["wide_x"] / 2.0 + 1e-6
    half_y = pitches["wide_y"] / 2.0 + 1e-6
    kept, dropped = [], 0
    r_clear = params.anchor_hole_dia_mm / 2.0 + params.anchor_hole_clearance_mm
    from shapely.geometry import Point
    for cx in hx:
        for cy in hy:
            near = [(x, y) for (x, y) in occupied
                    if abs(x - cx) <= half_x and abs(y - cy) <= half_y]
            if not near:
                continue
            if board_poly is not None and not board_poly.contains(
                    Point(float(cx), float(cy)).buffer(r_clear)):
                dropped += 1
                continue
            kept.append([float(cx), float(cy)])
    return np.asarray(kept, dtype=float).reshape(-1, 2), dropped


def apply_to_params(params) -> dict | None:
    """Resolve the anchor-hole lattice into `params`, in place.

    Overwrites `pitch_*_mm` with the wide pitch and fills `pack_pattern_*`.
    Returns the pitch dict, or None when the feature is off -- so a caller can
    branch on it without re-reading the flag.

    Called once per request, before the layout runs. Every packer call site
    then reads the pattern off params, which keeps the phase search, the
    fixed-keepout trial packs and the real pack on one lattice; if they
    disagreed the cutouts would be measured off a grid the board does not have.
    """
    if not getattr(params, "anchor_holes", False):
        params.pack_pattern_x = None
        params.pack_pattern_y = None
        return None
    r = resolve_pitches(params)
    params.pitch_x_mm = r["pitch_x"]
    params.pitch_y_mm = r["pitch_y"]
    params.pack_pattern_x = r["pattern_x"]
    params.pack_pattern_y = r["pattern_y"]
    return r
