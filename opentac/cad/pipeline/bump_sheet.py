"""Shear-sensing bump sheet: a printable TPU overlay that sits on a finished
sensor and puts one bump on each 2x2 group of taxels.

Why bumps: a flat taxel array reads normal force only. A bump standing above a
2x2 group converts a sideways (shear) load into an asymmetric normal load
across those four taxels, so their differential says which way the surface was
pushed.

The stack, bottom-up, all in the sensor's own coordinate frame:

    z = 0                    skirt bottom and sensor bottom, flush on the
                             mounting surface
    z = recess_depth         ceiling of the recess the sensor sits in
    z = recess_depth
        + sheet_thickness    top of the sheet; bumps rise from here
    ... + bump_height        bump tops

In plan view there are two outlines: `A`, the sensor footprint plus a fit
tolerance (the recess), and `B`, `A` grown by the skirt width (the whole
sheet). The skirt is B - A, solid over the full height, and it is what gives
the part a rigid frame to be glued or clamped by.

Both outlines are cut off flat at the cable edge, so the recess reaches it and
the frame is a U rather than a closed ring. The tail and the connector have to
leave the part, and a skirt wall across that edge would sit straight in their
way; with it open the board also slides in from that side instead of having to
be stretched over the lip.

Bump decoupling. A continuous sheet would drag every neighbouring bump when one
is pushed sideways, which is exactly the crosstalk the bumps exist to resolve.
So the sheet layer is cut into one tile per bump by through-slits, and the tiles
are rejoined by a few narrow tabs -- tab width and count are the stiffness knob.
Slits and tabs are plain 2D boolean work, unlike a free-standing lattice, and
FDM TPU prints a ~1mm tab far more reliably than it prints thin lattice strands.
The perimeter is slit too, so the outermost bumps are suspended from the frame
rather than rigidly joined to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union


@dataclass
class BumpParams:
    """Every dimension in mm. Defaults follow the taper ratio reported best for
    uniform stress transfer in Micromachines 2024, 15(4), 486, and a sheet thin
    enough to stay compliant in printed TPU."""
    skirt_width_mm: float = 3.0
    recess_depth_mm: float = 1.0        # sensor thickness
    recess_tolerance_mm: float = 0.2    # slip fit around the board edge
    sheet_thickness_mm: float = 0.8
    bump_height_mm: float = 2.0
    bump_base_frac: float = 0.8         # of the bump pitch (2 x taxel pitch)
    bump_top_ratio: float = 0.7         # top/base diameter
    slit_width_mm: float = 0.8
    tab_width_mm: float = 1.2
    tabs_per_edge: int = 1
    segments: int = 48                  # bump tessellation


@dataclass
class BumpSheetResult:
    mesh: trimesh.Trimesh
    bump_centers: np.ndarray
    bump_base_dia_mm: tuple      # (x, y) - equal for square taxels
    bump_top_dia_mm: tuple
    total_height_mm: float
    partial_groups: int                 # bumps whose 2x2 block was cut short
    warnings: list = field(default_factory=list)


# Outlines that came from an STL unfold carry duplicate/zero-length edges, and
# buffering them keeps those. Extruding such a polygon yields degenerate
# triangles and a mesh that is not a closed volume — which makes the whole
# boolean union bail out. Measured on the Yubi finger pad: min edge length
# 0.000000mm over 1038 points, extrude -> is_volume False; simplifying at 1um
# (well under any printer's resolution) fixes it and more than halves the
# triangle count. Every polygon is passed through this before extrusion.
CLEAN_TOL_MM = 1e-3


def _clean(poly):
    out = poly.simplify(CLEAN_TOL_MM).buffer(0)
    return _largest_polygon(out) if not isinstance(out, Polygon) else out


def _largest_polygon(geom):
    """Shapely booleans can yield a MultiPolygon; the sheet is the biggest part.
    Slivers here are offset artefacts, not real geometry."""
    if isinstance(geom, Polygon):
        if geom.is_empty:
            raise ValueError("bump sheet outline collapsed to nothing")
        return geom
    # A GeometryCollection turns up when the tail clip grazes an edge and
    # leaves stray lines/points alongside the real area; keep only polygons.
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty]
    if polys:
        return max(polys, key=lambda g: g.area)
    raise ValueError("bump sheet outline collapsed to nothing")


def _above_cable(poly, cable_y: float) -> Polygon:
    """Keep only what lies on the sensor side of the cable edge."""
    minx, miny, maxx, maxy = poly.bounds
    keep = box(minx - 10.0, cable_y, maxx + 10.0, maxy + 10.0)
    return _largest_polygon(poly.intersection(keep))


def sensor_body_polygon(final_outline_xy, cable_y: float) -> Polygon:
    """The manufactured board with the cable tail cut off.

    The tail carries no taxels and the user does not want it covered, so the
    sheet stops at the cable edge. Clipping the FINAL outline (not the drawn
    one) matters: the board that exists is the grown one, and its sensor-side
    boundary is what the skirt has to wrap.
    """
    poly = Polygon(np.asarray(final_outline_xy, dtype=float)).buffer(0)
    return _above_cable(poly, cable_y)


def bump_centers_from_taxels(placements_xy, active_idx, pitch_x, pitch_y):
    """One bump per 2x2 block that contains ANY taxel.

    A block the board outline cut short still gets a full-size bump, placed
    where the complete block would have been. Part of that bump then overhangs
    the space the missing taxels would have occupied, which is intended: the
    taxels that ARE present still sit under bump material and still see the
    asymmetric load when it tips, so they keep resolving shear. Dropping the
    bump instead left whole strips along every irregular edge reading pressure
    only — on the Yubi pad that was 16 of 26 blocks.

    Row/column indices are derived from position rather than taken from the
    layout's net ordering: pad order is deliberately not geometric order (the
    bottom half of the rows is reversed so the array meets the routing on the
    right side), so grouping by it would pair taxels that are nowhere near each
    other.
    """
    P = np.asarray(placements_xy, dtype=float)
    if not len(active_idx):
        return np.zeros((0, 2)), 0
    pts = P[np.asarray(sorted(active_idx), dtype=int)]
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    col = np.rint((pts[:, 0] - x0) / pitch_x).astype(int)
    row = np.rint((pts[:, 1] - y0) / pitch_y).astype(int)

    counts: dict = {}
    for r, c in zip(row, col):
        key = (int(r) // 2, int(c) // 2)
        counts[key] = counts.get(key, 0) + 1

    centers, partial = [], 0
    for (gr, gc), n in sorted(counts.items()):
        # The block's LATTICE centre, not the mean of whichever taxels happen
        # to be present: averaging would drag a partial block's bump sideways
        # onto its surviving taxels, so it would no longer straddle them and
        # could not tip against them. It also keeps every centre exactly on one
        # grid, which the slit spacing below depends on.
        centers.append([x0 + (2 * gc + 0.5) * pitch_x,
                        y0 + (2 * gr + 0.5) * pitch_y])
        if n < 4:
            partial += 1
    return np.asarray(centers, dtype=float), partial


def bump_index_per_point(points_xy, centers, pitch_x, pitch_y):
    """Which bump each point sits under: an index into `centers`, or -1.

    The rule is the TILE, not raw distance. A bump owns the 2x2 box around its
    centre — literally the `box(cx +- pitch_x, cy +- pitch_y)` that
    `_slit_geometry` cuts its tile from — so a taxel is under a bump exactly
    when it falls inside that box, and where two tiles overlap the nearer
    centre wins. They do overlap on the "paired" anchor lattice, whose
    inter-block gap is deliberately narrower than the tile.

    -1 happens only when the centres come from ANCHOR CUTOUTS, and then it is
    routine rather than exotic. 2x2 blocks cannot pair an odd number of taxel
    lines; here that leftover line gets no cutout (`_pair_midpoints` emits
    midpoints of pairs only) and so no bump. Measured exact over eight boards:
    `bumpless = (cols odd ? rows : 0) + (rows odd ? cols : 0) - (both odd ? 1
    : 0)`. A cutout dropped for hanging over the board edge orphans its block
    the same way.

    Centres derived from the taxel grid never orphan anything: an odd line
    forms its own partial block there and keeps a full-size bump, which is
    invariant 2 of this module. The anchor path not honouring it is a known
    gap — see NOTES.md, "Manifest `shear` block".

    Either way those taxels read pressure alone, and nothing downstream could
    tell them apart from the taxels that do resolve shear. This is what says so.
    """
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    C = np.asarray(centers, dtype=float).reshape(-1, 2)
    if not len(pts) or not len(C):
        return [-1] * len(pts)
    dx = pts[:, None, 0] - C[None, :, 0]
    dy = pts[:, None, 1] - C[None, :, 1]
    # 1e-6 so a taxel sitting exactly on a tile edge (which the uniform
    # lattice puts it nowhere near, but the paired one can) is not rejected by
    # float noise.
    inside = (np.abs(dx) <= pitch_x + 1e-6) & (np.abs(dy) <= pitch_y + 1e-6)
    d2 = np.where(inside, dx * dx + dy * dy, np.inf)
    best = np.argmin(d2, axis=1)
    return [int(b) if np.isfinite(d2[i, b]) else -1 for i, b in enumerate(best)]


def _frustum(cx, cy, z0, height, rb, rt, segments):
    """Watertight ELLIPTICAL frustum; `rb`/`rt` are (rx, ry) at base and top.

    Elliptical rather than circular because taxels are not necessarily square:
    the bump is sized from its own 2x2 block, so a 4x2mm taxel grid gets a bump
    twice as wide as it is deep. A circle sized to the smaller axis would leave
    the long axis of every block uncovered, and one sized to the larger axis
    would overhang into the neighbouring block. A circle is just the case
    rx == ry, so there is one code path.
    """
    rxb, ryb = rb
    rxt, ryt = rt
    th = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    bottom = np.column_stack([cx + rxb * np.cos(th), cy + ryb * np.sin(th),
                              np.full(segments, z0)])
    top = np.column_stack([cx + rxt * np.cos(th), cy + ryt * np.sin(th),
                           np.full(segments, z0 + height)])
    cb, ct = [cx, cy, z0], [cx, cy, z0 + height]
    V = np.vstack([bottom, top, [cb], [ct]])
    ib, it = 2 * segments, 2 * segments + 1
    F = []
    for i in range(segments):
        j = (i + 1) % segments
        F.append([i, j, segments + j])                 # side
        F.append([i, segments + j, segments + i])
        F.append([ib, j, i])                           # bottom cap
        F.append([it, segments + i, segments + j])     # top cap
    return trimesh.Trimesh(vertices=V, faces=np.asarray(F), process=True)


def _tab_offsets(n: int, span: float):
    """Where `n` tabs sit along one tile edge of length `span`.

    Symmetric about the middle of the edge and kept off the corners: a tab at a
    tile corner braces two cuts at once, which stiffens the tile far more than
    the same tab placed mid-edge and defeats the point of the count being a
    tunable.
    """
    if n <= 1:
        return [0.0]
    usable = span * 0.7
    return [-usable / 2.0 + usable * i / (n - 1) for i in range(n)]


def _slit_geometry(recess_poly, centers, pitch_x, pitch_y, p: BumpParams,
                   open_edge_y: float | None = None):
    """Through-cuts that separate one bump tile from the next, minus the tabs
    that hold them together. Cut lines run midway between bump centres, and the
    outer edge of the tile field is cut as well, so a border bump gets the same
    2x2 tile as an interior one and hangs from the frame on tabs instead of
    being welded to it."""
    if not len(centers):
        return None
    half = p.slit_width_mm / 2.0
    minx, miny, maxx, maxy = recess_poly.bounds
    xs = np.unique(np.round(centers[:, 0], 4))
    ys = np.unique(np.round(centers[:, 1], 4))

    cuts = []
    for x in xs[:-1] + pitch_x:                       # midway to the next column
        cuts.append(box(x - half, miny - 1, x + half, maxy + 1))
    for y in ys[:-1] + pitch_y:                       # midway to the next row
        cuts.append(box(minx - 1, y - half, maxx + 1, y + half))
    grid = unary_union(cuts).intersection(recess_poly) if cuts else None

    # The OUTER edge of the tile field, not the board's edge.
    #
    # Cutting at the board edge instead left every perimeter bump welded to the
    # margin of sheet between the outermost tile and the board: measured 7.3%
    # of an outward-facing tile edge actually cut, against 82% for edges facing
    # another bump, so border bumps barely moved independently at all. Slitting
    # around the tile field gives every bump the same 2x2 tile regardless of
    # where it sits, and the leftover margin merges into the rigid frame, which
    # is what it should be doing anyway.
    #
    # Clipping the field to the recess before taking its boundary handles the
    # other case in the same expression: where the tile field runs past the
    # board edge, the clipped boundary follows the board instead, which is the
    # old behaviour exactly where it was the right one.
    field = unary_union([box(cx - pitch_x, cy - pitch_y, cx + pitch_x, cy + pitch_y)
                         for cx, cy in centers]).intersection(recess_poly)
    ring = field.boundary.buffer(half) if not field.is_empty else None
    if ring is not None and open_edge_y is not None:
        # The cable edge is where the part simply ends. Slitting along it would
        # only trim the bottom row of tiles: there is nothing on the far side
        # to detach them from.
        ring = ring.difference(box(minx - 10.0, open_edge_y - half - 0.01,
                                   maxx + 10.0, open_edge_y + half + 0.01))

    # Tabs: short bridges back across every cut. Width and count together are
    # the stiffness knob — how freely one bump moves without carrying its
    # neighbours with it.
    tw = p.tab_width_mm / 2.0
    n_tabs = max(1, int(p.tabs_per_edge))
    off_y = _tab_offsets(n_tabs, 2.0 * pitch_y)       # one tile edge is 2 pitches
    off_x = _tab_offsets(n_tabs, 2.0 * pitch_x)
    grid_tabs = []
    for x in xs[:-1] + pitch_x:                       # across vertical cuts
        for y in ys:
            for dy in off_y:
                grid_tabs.append(box(x - half - 0.1, y + dy - tw, x + half + 0.1, y + dy + tw))
    for y in ys[:-1] + pitch_y:                       # across horizontal cuts
        for x in xs:
            for dx in off_x:
                grid_tabs.append(box(x + dx - tw, y - half - 0.1, x + dx + tw, y + half + 0.1))

    # Frame tabs, walked along the REAL boundary rather than the bounding box.
    # Placing them on the bbox only works for a rectangular board: on the Yubi
    # pad (a long tapered outline) not one bbox tab touched the boundary, every
    # border tile was cut free, and the sheet came apart into 11 pieces.
    # Every ring of the tile field is walked, holes included, so a tile beside a
    # gap in the array is held on too.
    ring_tabs = []
    step = max(2.0 * p.tab_width_mm, min(pitch_x, pitch_y) * 2.0) / n_tabs
    for poly in (field.geoms if hasattr(field, "geoms") else [field]):
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        for line in [poly.exterior, *poly.interiors]:
            n = max(4, int(round(line.length / step)))
            for i in range(n):
                # Half-step offset so tabs land mid-edge rather than at the
                # corners where the grid cuts meet this boundary.
                pt = line.interpolate((i + 0.5) * line.length / n)
                ring_tabs.append(box(pt.x - tw, pt.y - tw, pt.x + tw, pt.y + tw)
                                 .buffer(half, join_style=2))   # span the slit either side

    # Each tab group is subtracted only from the cut it belongs to. Subtracting
    # one merged tab set from one merged slit set let a frame tab plug the END
    # of a grid cut, which is precisely how a perimeter tile ends up still
    # attached on the side that was supposed to be cut.
    parts = []
    if grid is not None and not grid.is_empty:
        parts.append(grid.difference(unary_union(grid_tabs)) if grid_tabs else grid)
    if ring is not None and not ring.is_empty:
        parts.append(ring.difference(unary_union(ring_tabs)) if ring_tabs else ring)
    if not parts:
        return None
    return unary_union(parts)


def build_bump_sheet(final_outline_xy, cable_y, placements_xy, active_idx,
                     pitch_x, pitch_y, p: BumpParams | None = None,
                     centers_override=None) -> BumpSheetResult:
    """`centers_override`: bump centres to use instead of deriving them from
    the taxel grid. Passed when the board carries anchor cutouts -- the bump
    has to sit on its hole, and re-deriving the lattice here would put it
    somewhere else the moment the pitch is non-uniform (the paired lattice),
    because the derivation assumes a constant step.
    """
    p = p or BumpParams()
    warnings = []

    body = sensor_body_polygon(final_outline_xy, cable_y)
    # Both the pocket and the sheet stop dead at the cable edge, so the recess
    # runs right out to it. Buffering without re-clipping wrapped a skirt wall
    # a full skirt-width + tolerance thick across that edge — directly in the
    # path of the tail and connector, which have to leave the part. Ending the
    # sheet there instead leaves the frame a U, open at the cable, and the
    # board slides in from that side rather than having to be stretched over.
    recess = _above_cable(_largest_polygon(body.buffer(p.recess_tolerance_mm)), cable_y)
    sheet_2d = _above_cable(_largest_polygon(recess.buffer(p.skirt_width_mm)), cable_y)

    if centers_override is not None and len(centers_override):
        centers = np.asarray(centers_override, dtype=float).reshape(-1, 2)
        partial = 0
    else:
        centers, partial = bump_centers_from_taxels(
            placements_xy, active_idx, pitch_x, pitch_y)
    if not len(centers):
        warnings.append("The board has no taxels, so the sheet has no bumps.")

    # Sized per axis from that axis's own 2x2 span, so a non-square taxel grid
    # yields an oval bump that actually covers its block.
    base_dia_x = p.bump_base_frac * 2.0 * pitch_x
    base_dia_y = p.bump_base_frac * 2.0 * pitch_y
    top_dia_x = p.bump_top_ratio * base_dia_x
    top_dia_y = p.bump_top_ratio * base_dia_y
    # These two are user-facing now, so an unusable value has to say why rather
    # than quietly producing a sheet that looks fine and prints wrong.
    if len(centers) > 1 and p.bump_base_frac >= 1.0:
        warnings.append(
            "Bump width is 100% of the block or more, so neighbouring bumps run "
            "into each other and can no longer move independently.")
    if p.bump_top_ratio > 1.0:
        warnings.append(
            "Bump taper is over 100%, so each bump flares outward and overhangs "
            "its own base — it will need support to print.")
    recess_top = p.recess_depth_mm
    top_z = recess_top + p.sheet_thickness_mm

    parts = []
    # Frame: the skirt ring, full height, the only rigid part of the sheet.
    frame_2d = sheet_2d.difference(recess)
    if not frame_2d.is_empty:
        parts.append(trimesh.creation.extrude_polygon(_clean(_largest_polygon(frame_2d)),
                                                      height=top_z))

    # Sheet layer, slit into one tile per bump.
    slits = _slit_geometry(recess, centers, pitch_x, pitch_y, p, open_edge_y=cable_y)
    # The sheet layer is cut by SUBTRACTING solid slits from a plain slab,
    # rather than triangulating a polygon that already has the slits as holes.
    # That polygon has 30+ interior rings which pinch against each other and
    # against the outer boundary, and it extrudes to a mesh that is not a
    # closed volume -- so the sliver filter threw away the entire sheet and
    # left a frame with free-floating bumps (seen as "50 bumps -> 51 bodies").
    # Extruding a simple outline and letting the boolean engine do the cutting
    # avoids the whole class of problem.
    slab = trimesh.creation.extrude_polygon(_clean(sheet_2d), height=p.sheet_thickness_mm)
    slab.apply_translation((0, 0, recess_top))
    if slits is not None and not slits.is_empty:
        cutters = []
        for piece in (slits.geoms if isinstance(slits, MultiPolygon) else [slits]):
            if piece.area <= 1e-9:
                continue
            try:
                c = trimesh.creation.extrude_polygon(_clean(piece),
                                                     height=p.sheet_thickness_mm + 1.0)
            except Exception:
                continue
            if not c.is_volume:
                continue
            c.apply_translation((0, 0, recess_top - 0.5))
            cutters.append(c)
        if cutters:
            try:
                slab = trimesh.boolean.difference([slab] + cutters)
            except Exception as exc:
                warnings.append(f"Slit cutting failed ({exc}); sheet left uncut, "
                                "so bumps will not be mechanically decoupled.")
    parts.append(slab)

    # Each bump gets a short plug of its own base radius driven DOWN into the
    # sheet before the frustum starts. Sitting a bump exactly on the sheet's
    # top plane makes the two solids touch over zero volume, and the boolean
    # then keeps them apart: a 50-bump board came out as 51 separate bodies.
    # The plug is buried inside the sheet, so the exposed bump profile is still
    # exactly base -> top over bump_height.
    plug = min(0.3, p.sheet_thickness_mm * 0.5)
    rb = (base_dia_x / 2.0, base_dia_y / 2.0)
    rt = (top_dia_x / 2.0, top_dia_y / 2.0)
    for cx, cy in centers:
        parts.append(_frustum(cx, cy, top_z - plug, plug, rb, rb, p.segments))
        parts.append(_frustum(cx, cy, top_z, p.bump_height_mm, rb, rt, p.segments))

    # A real boolean union, not a concatenation: the frame, the sheet layer and
    # every bump overlap where they meet, and merely stacking their triangle
    # lists leaves those shared faces inside the solid. Slicers mostly cope,
    # but the result is not a closed manifold and cannot be checked as one.
    # manifold3d does this in well under a second at these sizes.
    try:
        mesh = trimesh.boolean.union(parts)
    except Exception as exc:                     # keep an inspectable part
        warnings.append(f"Boolean union unavailable ({exc}); "
                        "returning overlapping solids, which most slicers still union.")
        mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    if not mesh.is_watertight:
        # The union can leave a few single-face edges where many coplanar
        # pieces meet — 3 out of 11097 on the Yubi pad. Filling them closes the
        # solid with the volume unchanged to the decimal, so it is a crack in
        # the surface, not a missing chunk. Only report if it survives.
        before = float(mesh.volume)
        try:
            mesh.fill_holes()
        except Exception as exc:
            # Repair is an optimisation, never a requirement: the caller still
            # gets a usable solid with a warning. Letting this raise turned a
            # cosmetic crack into a failed request — trimesh reaches for
            # networkx here, and only here, so an environment missing it broke
            # exactly the boards that needed repair while every clean board
            # worked. The dependency is pinned now; this keeps the next missing
            # optional dep from taking the whole feature down with it.
            warnings.append(f"Could not repair a hairline crack ({exc}); "
                            "the solid is usable but should be inspected.")
        if not mesh.is_watertight:
            warnings.append("Mesh is not watertight — inspect before printing.")
        elif abs(float(mesh.volume) - before) > 1e-3:
            warnings.append("Mesh needed hole-filling that changed its volume — inspect.")
    return BumpSheetResult(
        mesh=mesh, bump_centers=centers,
        bump_base_dia_mm=(base_dia_x, base_dia_y),
        bump_top_dia_mm=(top_dia_x, top_dia_y),
        total_height_mm=top_z + p.bump_height_mm,
        partial_groups=partial, warnings=warnings)
