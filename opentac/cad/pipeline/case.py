"""Two-piece printed case for a finished sensor: a recessed bottom and a lid.

The stack, bottom-up, all in the sensor's own coordinate frame:

    z = 0                          underside of the case
    z = plate                      pocket floor — the flex sits on this
    z = plate + recess             top of the skirt; the lid lands here
    z = plate + recess + lid       top of the lid; bumps rise from here

Geometry comes from `bump_sheet`'s own helpers rather than a second copy:
`sensor_body_polygon` clips the manufactured board at the cable edge, and the
pocket and skirt are two buffers of that. Reusing them means the tail is
handled the same way the bump sheet already handles it — every part of the case
simply stops at the cable edge, so the skirt is a U and the connector leaves
through the open side with nothing to detect or carve.

How the lid is held on. A rib rises from the skirt and passes through a slot in
the lid; its head is wider than the slot, so the lid is retained by SHAPE. That
matters because TPU creeps: anything held by a squeeze force goes slack in
weeks, while a shoulder wider than its slot stays wider than its slot.

Why the rib is broken. An unbroken rib needs an unbroken slot, and an unbroken
slot cuts a closed loop through the lid -- severing the inner panel that covers
the sensor from the outer flange. Measured on a 42x42 board, that panel ends up
hanging off a 1.4mm neck at the cable edge and would tear off in handling. Every
gap in the rib is therefore a bridge in the lid, and the default 10/10 duty is
chosen so the bridges carry as much material as the rib does. Duty is the knob:
at a fixed period, lengthening the rib buys grip and shortens the bridges, and
it does NOT change the number of segment ends -- which is where the lid tears
from. Shortening the period instead adds ends for no gain.

Printing the pocket. At a 0.10mm flex and 0.10mm Velostat the pocket is 0.20mm
deep -- exactly one layer at a 0.2mm layer height, and the entire adjustment
range is finer than the printer's resolution. One layer is a neutral fit; zero
layers crushes the stack by its whole thickness; two layers doubles the slop.
There is no middle setting to print. If the fit needs tuning, shim under the
board with cut film rather than editing this number, or drop the bottom piece
to a 0.1mm layer height.

Placement is a modulo on arc length and nothing else. Each segment is a swept
curve, not a straight tab, so it follows the outline's facets and corners with
no corner test; a gap that lands on a corner is simply a bridge at a corner. An
earlier straight-tab version needed an arc walk, a corner test, a nudge pass and
a gap-repair pass, and still skipped 43mm of a 197mm perimeter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import unary_union

from .geometry import CONNECTOR_BODY_ABOVE_MM, CONNECTOR_BODY_BELOW_MM
from .bump_sheet import (BumpParams, _above_cable, _clean, _frustum,
                         _largest_polygon, _slit_geometry, _tab_offsets,
                         bump_centers_from_taxels, sensor_body_polygon)


# Below this a bridge is a tear waiting to happen, whatever the duty cycle.
MIN_BRIDGE_MM = 2.5
# The lid's perimeter slit and the rib slot are two concentric cuts with a
# ligament between them, and that ligament carries every load from the flange
# to the tile field. Narrower than this and it is under two extrusion widths.
MIN_WEB_MM = 1.0
# How far below the cable edge to look for the tail neck, and how decisive the
# step has to be before it counts as one.
# The scan runs the FULL depth below the cable edge, not a fixed window. A 9mm
# window was enough for a 42x42 board and wrong twice on a 65x65: with follow
# padding off the board is still full width at -10.5, and at 20x20 taxels the
# tail settles at -8.0 but confirming "stable for 2mm" needs samples down to
# -10.0, which the window did not have. Searching deeper is safe because the
# rule takes the FIRST stable narrow run going down -- the tail is always found
# before the connector's own narrower step further along.
# The bottom piece is a stack -- sheet, wall on it, rib on the wall. Stacked face
# to face, two solids are fused through a shared plane, which the boolean can
# keep as a pinch where an edge of one meets the other's outline (the tail
# slot's corner): closed in memory, open once the STL's float32 vertices are
# merged on load (16 of 310 generated boards). Each layer rises this far out of
# the one below instead, as the lid's bumps plug into its slab; the solid is the
# same.
FUSE_MM = 0.10
# Rib pieces under this area (in units of the rib head squared) are slivers the
# tail slot clips off a segment's end; extruded they are not a closed solid and
# the union refuses the whole bottom.
MIN_RIB_PIECE = 0.25
NECK_MAX_FRAC = 0.85      # tail must be clearly narrower than the board
NECK_STABLE_MM = 2.0      # ...and hold that width for this far
NECK_FLAT_MM = 0.25       # ...to within this much


@dataclass
class CaseParams:
    """Every dimension in mm."""
    # Wider than BumpParams' 3.0 on purpose. The bump sheet puts ONE cut near
    # the pocket edge; the case puts two concentric ones -- the perimeter slit
    # and the rib slot -- and the ligament between them scales with the skirt
    # at about half a millimetre per millimetre. At 3.0 that ligament is
    # 0.65mm, under two extrusion widths, and it carries the whole tile field.
    skirt_width_mm: float = 4.0
    recess_tolerance_mm: float = 0.20   # slip fit around the board edge
    # Measured, not estimated: flex ~0.10 + Velostat ~0.10. An earlier 0.30 came
    # from assuming a 0.15mm flex and left the board 0.1mm loose in a joint where
    # contact pressure IS the signal. Note this is now ONE 0.2mm layer -- see the
    # printing note in the module docstring before changing it.
    recess_depth_mm: float = 0.20
    plate_thickness_mm: float = 0.60    # pocket floor
    lid_thickness_mm: float = 0.80
    rib_wall_mm: float = 0.80
    rib_head_mm: float = 1.20           # shoulder width; undercut = (head-wall)/2
    rib_proud_mm: float = 0.60          # head standing above the lid
    slot_width_mm: float = 0.90
    rib_period_mm: float = 20.0
    rib_gap_mm: float = 10.0            # 10/10 — bridges carry as much as the rib
    slot_end_clearance_mm: float = 0.30  # slot longer than rib, or the ends bind
    min_bridges: int = 8                # small boards need a shorter period
    # Wrap the cable edge and cut only a slot for the tail, instead of leaving
    # the whole side open. The board is full width at the cable line -- the tail
    # does not narrow until a few mm below it -- so clipping there opens ~46mm
    # of wall for a ~15mm tail. False restores the open-edge U.
    wrap_cable_edge: bool = True
    tail_clearance_mm: float = 0.60     # each side of the tail slot
    tail_grip_mm: float = 1.00          # tail captured in the pocket: strain relief
    # Flexible cable that must still be exposed between the case and the start
    # of the connector body. The case eats skirt + tolerance + grip of tail
    # before anything emerges, and the connector body itself is another 6mm
    # that cannot bend -- so without this check a short cable produces a case
    # that swallows the connector whole and looks fine in the STL.
    min_tail_clear_mm: float = 3.00
    # Hatch the lid: cut it on the taxel boundaries so a press lands on the taxel
    # under it instead of spreading to its neighbours. Material stays over every
    # taxel -- a hole THERE would remove the only thing that can push on the
    # pixel and put the load path on the boundaries, which is backwards.
    hatch: bool = False
    hatch_groove_mm: float = 0.90
    # Through-cut by default. A groove only THINS the boundary: at half the lid
    # thickness that is 12.5% stiffness -- 8x softer, but still a continuous
    # membrane carrying moment across every edge. A through-cut carries nothing
    # but the tabs, which is the point. hatch_through=False gives the groove
    # instead: a sealed lid with nothing to tear, at the cost of decoupling.
    hatch_through: bool = True
    hatch_tab_mm: float = 1.20          # bridges across each cut; through-cut only
    hatch_tabs_per_edge: int = 1
    hatch_depth_mm: float = 0.40        # groove mode only: leaves lid - depth of web
    # One small bump per taxel on the hatched lid. Default TOP, and the reasoning
    # is worth keeping because the alternative looks tempting.
    #
    # A pip on the UNDERSIDE would raise pressure at the sensing junction --
    # tile force over contact area, so ~6x at these sizes. But it contradicts
    # the rule the whole lid is built on: the underside stays dead flat against
    # the Velostat. A pip is a permanent contact point, so it preloads the
    # junction, gives a non-zero baseline and then creeps; and it caps the
    # signal, since only the comb fingers beneath it ever get bridged.
    #
    # A bump on TOP gives a discrete, repeatable contact instead of somewhere
    # ambiguous on a flat tile. Against a flat rigid indenter it does not change
    # the junction pressure -- but a flat rigid indenter is the worst case for
    # it, not the usual one. Against a finger or any curved or compliant object
    # the bumps are what the object contacts, so load concentrates into
    # individual tiles instead of spreading across the plate.
    #
    # Only one face prints without supports either way: the lid goes
    # flat-face-down, so top bumps print upward cleanly and a bottom pip means
    # printing it inverted. The two cannot be combined.
    hatch_bumps: bool = False
    hatch_bump_face: str = "top"        # "top" | "bottom"
    hatch_bump_dia_mm: float = 1.80
    hatch_bump_height_mm: float = 0.60
    hatch_bump_top_ratio: float = 0.70
    bumps: bool = False
    # Slit the lid into one tile per bump. Without this a bump drags its
    # neighbours when pushed sideways, which is exactly the crosstalk the bumps
    # exist to resolve -- so it is on whenever there are bumps.
    slit_bumps: bool = True
    bump_params: BumpParams = field(default_factory=BumpParams)
    segments: int = 48


@dataclass
class CaseResult:
    bottom: trimesh.Trimesh
    top: trimesh.Trimesh
    stats: dict
    warnings: list = field(default_factory=list)


def _tail_neck(final_outline_xy, cable_y: float):
    """Where the board settles into its cable tail: (clip_y, x_lo, x_hi) or None.

    The cable edge is NOT the right place to open the wall. At that line the
    board is still full width -- the routing fan-out shoulder sits just below it
    and only then does the outline step down to the tail, which is sized by the
    connector rather than by the board. Clipping at the cable edge therefore
    opens ~46mm of wall to let a ~15mm tail out.

    Two things make this awkward to detect and both were found by measuring:

    * The step is not a clean edge. Across roughly a millimetre the scan line
      grazes the taper and returns two or three slivers, so anything that keys
      off the largest single drop lands inside that zone and reports a tail
      33mm wide. What is reliable is the STABLE width below the step.
    * "Narrow" cannot be a fraction of the board. An 18x18 board's tail is 62%
      of its width, so any fixed fraction that catches a big board misses a
      small one. The test is only that the tail is clearly narrower than full.

    Where the board steps in to the connector's tab at the connector's opening
    (_neck_floor), the tab past it is not looked at.
    """
    poly = Polygon(np.asarray(final_outline_xy, dtype=float)).buffer(0)
    if poly.is_empty:
        return None
    minx, miny, maxx, maxy = poly.bounds

    def width(y):
        """Widest CONTIGUOUS run at this height, not the overall extent."""
        g = poly.intersection(LineString([(minx - 1.0, y), (maxx + 1.0, y)]))
        if g.is_empty:
            return 0.0, None
        runs = [r for r in (g.geoms if hasattr(g, "geoms") else [g]) if r.length > 1e-9]
        if not runs:
            return 0.0, None
        b = max(runs, key=lambda r: r.length).bounds
        return b[2] - b[0], (b[0], b[2])

    full = max((width(y)[0] for y in np.linspace(cable_y + 0.5, maxy - 0.5, 25)), default=0.0)
    if full <= 0:
        return None

    step = 0.25
    ys = np.arange(cable_y, _neck_floor(poly, miny) + step, -step)
    prof = [(y,) + width(y) for y in ys]
    hold = int(round(NECK_STABLE_MM / step))
    for i in range(len(prof) - hold):
        y, w, xr = prof[i]
        if xr is None or w <= 0.5 or w > NECK_MAX_FRAC * full:
            continue
        window = prof[i:i + hold]
        if any(v[1] <= 0.5 or abs(v[1] - w) > NECK_FLAT_MM for v in window):
            continue
        return float(y), float(xr[0]), float(xr[1])
    return None


def _neck_floor(poly: Polygon, miny: float) -> float:
    """The lowest y the tail neck is looked for at: the tail's end, unless the
    board steps in to the connector's tab at the connector's opening, which the
    export puts CONNECTOR_BODY_BELOW_MM before the end. A tail wider than the
    tab there (one typed wider than the connector, or the routing canvas of a
    board without follow padding) is cut to the tab past the opening
    (connector_geometry.clip_beside_tab), and that tab, the connector itself,
    then read as the neck: the case sealed the connector in. Any other tail is
    as wide as the tab before the opening as after it, and is looked at whole,
    as it always was."""
    y_open = miny + CONNECTOR_BODY_BELOW_MM
    minx, _, maxx, _ = poly.bounds

    def widest(y):
        g = poly.intersection(LineString([(minx - 1.0, y), (maxx + 1.0, y)]))
        return max((r.length for r in getattr(g, "geoms", [g]) if not r.is_empty), default=0.0)

    return y_open if widest(y_open + 0.25) - widest(y_open - 0.25) > NECK_FLAT_MM else miny


def _polygons(geom):
    """The polygons in a shapely result, whatever type it came back as.

    Unions and differences of cut shapes can return a GeometryCollection (a
    polygon plus a stray sliver line or point) instead of a MultiPolygon. Treating
    that as one piece skipped every cut in it: the hatched lid of a plain square
    patch came out almost uncut.
    """
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return [g for part in getattr(geom, "geoms", []) for g in _polygons(part)]


def _hatch_geometry(recess_poly, pts, pitch_x, pitch_y, width,
                    tab_width: float = 0.0, tabs_per_edge: int = 1):
    """Groove lines midway between taxels, plus one round the tile field.

    The field ring matters for the same reason it does in `_slit_geometry`:
    without it a border pad is stiffened by the flange it is welded to, and only
    the interior taxels get the compliance.

    Column and row lines are taken as the midpoint of each ADJACENT PAIR rather
    than centre+pitch, so an uneven lattice
    grooves in the right places instead of drifting across the array.
    """
    pts = np.asarray(pts, dtype=float)
    if not len(pts):
        return None
    half = width / 2.0
    minx, miny, maxx, maxy = recess_poly.bounds
    xs = np.unique(np.round(pts[:, 0], 4))
    ys = np.unique(np.round(pts[:, 1], 4))

    cuts = []
    for x in (xs[:-1] + np.diff(xs) / 2.0):
        cuts.append(box(x - half, miny - 1.0, x + half, maxy + 1.0))
    for y in (ys[:-1] + np.diff(ys) / 2.0):
        cuts.append(box(minx - 1.0, y - half, maxx + 1.0, y + half))
    grid = unary_union(cuts).intersection(recess_poly) if cuts else None

    field = unary_union([box(cx - pitch_x / 2, cy - pitch_y / 2,
                             cx + pitch_x / 2, cy + pitch_y / 2)
                         for cx, cy in pts]).intersection(recess_poly)
    ring = field.boundary.buffer(half) if not field.is_empty else None

    if tab_width <= 0:
        keep = [g for g in (grid, ring) if g is not None and not g.is_empty]
        return unary_union(keep) if keep else None

    # A through-cut needs bridges back across every cut or the lid falls into one
    # loose tile per taxel. Each tab group is subtracted only from the cut it
    # belongs to: merging them lets a ring tab plug the END of a grid cut, which
    # is exactly how a border tile ends up still welded on the side that was
    # supposed to come free.
    tw = tab_width / 2.0
    n = max(1, int(tabs_per_edge))
    off_x, off_y = _tab_offsets(n, pitch_x), _tab_offsets(n, pitch_y)
    grid_tabs = []
    for x in (xs[:-1] + np.diff(xs) / 2.0):
        for y in ys:
            for dy in off_y:
                grid_tabs.append(box(x - half - 0.1, y + dy - tw, x + half + 0.1, y + dy + tw))
    for y in (ys[:-1] + np.diff(ys) / 2.0):
        for x in xs:
            for dx in off_x:
                grid_tabs.append(box(x + dx - tw, y - half - 0.1, x + dx + tw, y + half + 0.1))

    ring_tabs = []
    step = max(2.0 * tab_width, min(pitch_x, pitch_y)) / n
    for poly in (field.geoms if hasattr(field, "geoms") else [field]):
        if not isinstance(poly, Polygon) or poly.is_empty:
            continue
        for line in [poly.exterior, *poly.interiors]:
            k = max(4, int(round(line.length / step)))
            for i in range(k):
                pt = line.interpolate((i + 0.5) * line.length / k)
                ring_tabs.append(box(pt.x - tw, pt.y - tw, pt.x + tw, pt.y + tw)
                                 .buffer(half, join_style=2))

    keep = []
    if grid is not None and not grid.is_empty:
        keep.append(grid.difference(unary_union(grid_tabs)) if grid_tabs else grid)
    if ring is not None and not ring.is_empty:
        keep.append(ring.difference(unary_union(ring_tabs)) if ring_tabs else ring)
    return unary_union(keep) if keep else None


def _no_tail_reason(final_outline_xy, cable_y: float) -> str:
    """Say WHY no tail was found, because the two causes need different fixes."""
    return _no_tail(final_outline_xy, cable_y)[1]


def _no_tail(final_outline_xy, cable_y: float) -> tuple[str, str]:
    """(kind, message) for a board with no tail neck: "full_width" (no narrower
    tail at all -- a longer cable cannot help) or "unsteady" (the tail never
    holds a width for NECK_STABLE_MM -- a longer cable gives it one)."""
    poly = Polygon(np.asarray(final_outline_xy, dtype=float)).buffer(0)
    minx, miny, maxx, maxy = poly.bounds

    def width(y):
        g = poly.intersection(LineString([(minx - 1.0, y), (maxx + 1.0, y)]))
        runs = [r for r in (g.geoms if hasattr(g, "geoms") else [g]) if r.length > 1e-9]
        return (max(runs, key=lambda r: r.length).bounds[2]
                - max(runs, key=lambda r: r.length).bounds[0]) if runs else 0.0

    full = max((width(y) for y in np.linspace(cable_y + 0.5, maxy - 0.5, 25)), default=0.0)
    below = [width(y) for y in np.arange(cable_y, _neck_floor(poly, miny) + 0.5, -0.5)]
    narrowest = min([w for w in below if w > 0.5], default=full)
    if full > 0 and narrowest > NECK_MAX_FRAC * full:
        return "full_width", (
            "The board is full width all the way down to the connector, so there is no "
            "narrower tail to cut a slot around — the connector sits inside the outline "
            "rather than on a protruding tail. The cable edge is left open, which is the only "
            "thing this shape allows. Generating with follow-padding on (follow_main_padding) "
            "gives the board a real tail.")
    return "unsteady", (
        "No cable tail was found: below the cable edge the outline never settles to a width "
        f"that holds steady for {NECK_STABLE_MM:g}mm. The case leaves the whole cable edge "
        "open rather than guessing where to cut a slot.")


def _ring(centre: Polygon, width: float) -> Polygon:
    """A band `width` wide centred on `centre`'s boundary."""
    return centre.buffer(width / 2).difference(centre.buffer(-width / 2))


def _arc_samples(poly: Polygon, step: float = 0.2):
    """Even arc-length samples round a ring, as (distance, point).

    Arc length, not vertices: a generated outline is a few hundred points, most
    of them sub-millimetre facets approximating a curve, and anything indexed by
    segment silently ignores most of the perimeter.
    """
    coords = list(poly.exterior.coords)[:-1]
    out, acc = [], 0.0
    for a, b in zip(coords, coords[1:] + coords[:1]):
        a, b = np.asarray(a), np.asarray(b)
        L = float(np.hypot(*(b - a)))
        if L < 1e-9:
            continue
        for k in range(max(1, int(L / step))):
            out.append((acc + k * step, a + (b - a) * (k * step / L)))
        acc += L
    return out, acc


def _gap_cutters(samples, total: float, period: float, gap: float, width: float):
    """Solid regions to subtract from rib and slot, one per gap."""
    runs, cur = [], []
    for s, p in samples:
        if (s % period) < gap:
            cur.append(p)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    pieces = [LineString(r).buffer(width, cap_style=2) for r in runs if len(r) >= 2]
    return (unary_union(pieces) if pieces else None), len(runs)


def tail_extension_length(tail_extension: dict | None) -> float:
    """Arc length of a drawn tail extension, or 0.

    The extension is NOT in the board outline. `apply_tail_extension` moves the
    connector pads and re-routes to them, but `final_outline_xy` is unchanged --
    so anything that measures the outline is blind to it, and a user who adds a
    15mm extension sees the same "cable too short" warning as before.
    """
    if not tail_extension or tail_extension.get("enabled") is False:
        return 0.0
    path = np.asarray(tail_extension.get("path") or [], dtype=float)
    if path.ndim != 2 or len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _case_plan(final_outline_xy, cable_y, p: CaseParams) -> dict:
    """The case's footprint around the board, 2D only: where the wall opens for
    the tail (the neck), the pocket (recess), the sheet under it, the cut that
    opens the wall, and why the tail could not be found if it could not."""
    # Where the wall opens. Wrapping the cable edge and cutting only a tail slot
    # means clipping BELOW the fan-out shoulder, not at the cable line -- at the
    # cable line the board is still full width, so the wall would open across
    # the whole side to pass a tail a third that wide.
    neck = _tail_neck(final_outline_xy, cable_y) if p.wrap_cable_edge else None
    no_tail = None
    if neck is not None:
        clip_y = neck[0] - p.tail_grip_mm     # bite a little tail: strain relief
        tail_x = (neck[1] - p.tail_clearance_mm, neck[2] + p.tail_clearance_mm)
    else:
        clip_y, tail_x = cable_y, None
        if p.wrap_cable_edge:
            no_tail = _no_tail(final_outline_xy, cable_y)

    body = sensor_body_polygon(final_outline_xy, clip_y)
    if tail_x is None:
        recess = _above_cable(_largest_polygon(body.buffer(p.recess_tolerance_mm)), clip_y)
        sheet = _above_cable(_largest_polygon(recess.buffer(p.skirt_width_mm)), clip_y)
        edge_cut = box(-1e4, -1e4, 1e4, clip_y + p.rib_head_mm)
    else:
        # Closed rings: the wall goes all the way round and only the slot opens.
        recess = _largest_polygon(body.buffer(p.recess_tolerance_mm))
        sheet = _largest_polygon(recess.buffer(p.skirt_width_mm))
        edge_cut = box(tail_x[0], sheet.bounds[1] - 10.0, tail_x[1], clip_y + 0.01)
    return {"neck": neck, "no_tail": no_tail, "clip_y": clip_y, "tail_x": tail_x,
            "recess": recess, "sheet": sheet, "edge_cut": edge_cut}


def _cable_clearance(final_outline_xy, sheet, p: CaseParams, tail_extension: dict | None) -> dict:
    """How much tail actually escapes the case, and how much of that is the rigid
    connector body. Measured from the shipped outline rather than from
    cable_length_mm, because the neck (and so the clip) moves with the board."""
    case_bottom = float(sheet.bounds[1])
    board_bottom = float(Polygon(np.asarray(final_outline_xy, dtype=float)).buffer(0).bounds[1])
    conn_body = CONNECTOR_BODY_ABOVE_MM + CONNECTOR_BODY_BELOW_MM
    # A drawn extension adds its full arc length of flexible tail, wherever it
    # runs -- it carries the connector away from the board, so every millimetre
    # of it is cable between the case and the connector.
    ext_mm = tail_extension_length(tail_extension)
    tail_free = case_bottom - board_bottom + ext_mm
    tail_flex = tail_free - conn_body
    tail_short = max(0.0, p.min_tail_clear_mm - tail_flex)
    warning = None
    if tail_short > 0.05:
        if tail_free <= 0:
            warning = (
                f"The case extends {-tail_free:.1f}mm PAST the end of the board — the connector "
                f"would be sealed inside it. Lengthen the cable by at least {tail_short:.1f}mm, "
                f"or draw a tail extension at least {tail_short:.1f}mm long.")
        elif tail_flex <= 0:
            warning = (
                f"The case covers {min(conn_body, -tail_flex):.1f}mm of the {conn_body:.0f}mm "
                f"connector body, so the connector cannot plug in. Lengthen the cable by at "
                f"least {tail_short:.1f}mm, or draw a tail extension at least {tail_short:.1f}mm long.")
        else:
            warning = (
                f"Only {tail_flex:.1f}mm of flexible cable clears the case before the "
                f"{conn_body:.0f}mm connector body begins, so there is almost nothing to bend or "
                f"grip. Lengthen the cable by at least {tail_short:.1f}mm, or draw a tail extension "
                f"at least {tail_short:.1f}mm long.")
    return {"tail_free_mm": tail_free, "tail_flex_mm": tail_flex,
            "short_mm": tail_short if tail_short > 0.05 else 0.0, "warning": warning}


def case_cable_check(final_outline_xy, cable_y, p: CaseParams | None = None,
                     tail_extension: dict | None = None) -> dict:
    """The printed case's cable checks without building the case (2D only, no
    meshing): the same neck, sheet and free-flex measurement build_case makes.

    Returns short_mm (how much longer the cable must be for the connector body to
    clear the case with min_tail_clear_mm of flex, 0 when it does), no_tail
    (None, or "full_width" / "unsteady" when no tail neck was found) and the
    cable warnings build_case would raise."""
    p = p or CaseParams()
    plan = _case_plan(final_outline_xy, cable_y, p)
    clear = _cable_clearance(final_outline_xy, plan["sheet"], p, tail_extension)
    warnings = [m for m in ((plan["no_tail"] or (None, None))[1], clear["warning"]) if m]
    return {"short_mm": clear["short_mm"], "tail_flex_mm": clear["tail_flex_mm"],
            "no_tail": plan["no_tail"][0] if plan["no_tail"] else None,
            "neck": plan["neck"], "warnings": warnings}


def build_case(final_outline_xy, cable_y, placements_xy, active_idx,
               pitch_x, pitch_y, p: CaseParams | None = None,
               tail_extension: dict | None = None) -> CaseResult:
    p = p or CaseParams()
    warnings = []

    plan = _case_plan(final_outline_xy, cable_y, p)
    neck, clip_y, tail_x = plan["neck"], plan["clip_y"], plan["tail_x"]
    recess, sheet, edge_cut = plan["recess"], plan["sheet"], plan["edge_cut"]
    if plan["no_tail"]:
        warnings.append(plan["no_tail"][1])
    frame = sheet.difference(recess).difference(edge_cut)
    centre = recess.buffer(p.skirt_width_mm / 2)

    samples, total = _arc_samples(centre)

    # A fixed period does not scale down: a fingertip-sized board has too short
    # a perimeter to fit enough gaps, and the lid panel is then held on by two
    # or three bridges. Shrink the period until there are at least min_bridges,
    # and shrink the gap by the SAME factor so the duty cycle is preserved --
    # scaling the period alone either starves the rib or, if the gap is used as
    # a floor, silently makes the period longer instead of shorter.
    period, gap = p.rib_period_mm, p.rib_gap_mm
    if total / max(period, 1e-6) < p.min_bridges:
        needed = (total / p.min_bridges) / period
        floor = MIN_BRIDGE_MM / gap if gap > 0 else 1.0
        scale = min(1.0, max(needed, floor))
        period, gap = period * scale, gap * scale
        warnings.append(
            f"Perimeter is only {total:.0f}mm, so the rib pattern was scaled to "
            f"{period:.1f}mm period / {gap:.1f}mm gap to keep enough bridges holding the "
            "lid together.")
    gap = min(gap, period * 0.7)
    if gap < MIN_BRIDGE_MM:
        warnings.append(
            f"A {gap:.1f}mm gap leaves bridges that narrow in the lid; under about "
            f"{MIN_BRIDGE_MM:g}mm they tear where the slot ends. Widen the gap.")

    clear = _cable_clearance(final_outline_xy, sheet, p, tail_extension)
    if clear["warning"]:
        warnings.append(clear["warning"])

    open_edge = edge_cut
    rib_cut, n_gaps = _gap_cutters(samples, total, period, gap, p.rib_head_mm)
    # The slot must be LONGER than the rib at every end or the two butt together
    # and the lid will not seat. Cutting the slot with a shorter cutter leaves
    # slot_end_clearance of free space beyond each rib end.
    slot_cut, _ = _gap_cutters(samples, total, period,
                               max(0.2, gap - 2 * p.slot_end_clearance_mm), p.rib_head_mm)

    def band(width, cutter):
        g = _ring(centre, width).difference(open_edge)
        return g.difference(cutter) if cutter is not None else g

    rib = band(p.rib_head_mm, rib_cut)
    slot = band(p.slot_width_mm, slot_cut)
    if rib.is_empty:
        raise ValueError("The rib came out empty — check skirt width against the board size.")

    z_floor = p.plate_thickness_mm
    z_seat = z_floor + p.recess_depth_mm
    rib_h = p.lid_thickness_mm + p.rib_proud_mm

    # ---- bottom piece -----------------------------------------------------
    parts = [trimesh.creation.extrude_polygon(_clean(sheet), height=z_floor)]
    if not frame.is_empty:
        wall = trimesh.creation.extrude_polygon(
            _clean(_largest_polygon(frame)), height=p.recess_depth_mm + FUSE_MM)
        wall.apply_translation((0, 0, z_floor - FUSE_MM))
        parts.append(wall)
    for piece in (rib.geoms if isinstance(rib, MultiPolygon) else [rib]):
        if piece.area <= MIN_RIB_PIECE * p.rib_head_mm ** 2:
            continue
        seg = trimesh.creation.extrude_polygon(_clean(piece), height=rib_h + FUSE_MM)
        seg.apply_translation((0, 0, z_seat - FUSE_MM))
        parts.append(seg)
    bottom = trimesh.boolean.union(parts) if len(parts) > 1 else parts[0]

    # ---- top piece --------------------------------------------------------
    # Both the rib slot and the bump slits are cut by SUBTRACTING solid cutters
    # from a plain slab, never by extruding a polygon that already carries them
    # as interior rings. bump_sheet learned this the hard way: with 30+ rings
    # pinching against each other and the outer boundary, the extrusion is not a
    # closed volume and the whole sheet gets thrown away.
    centers = np.zeros((0, 2))
    slits = None
    hatch = None
    n_bumps = 0
    bp = p.bump_params
    if p.hatch and not p.bumps:
        if not p.hatch_through:
            web = p.lid_thickness_mm - p.hatch_depth_mm
            if web < 0.15:
                warnings.append(
                    f"A {p.hatch_depth_mm:g}mm groove in a {p.lid_thickness_mm:g}mm lid leaves "
                    f"only {web:.2f}mm of web — too thin to survive handling.")
        active_pts = np.asarray(placements_xy, dtype=float)[np.asarray(sorted(active_idx), dtype=int)]             if len(active_idx) else np.zeros((0, 2))
        hatch = _hatch_geometry(recess, active_pts, pitch_x, pitch_y, p.hatch_groove_mm,
                                tab_width=p.hatch_tab_mm if p.hatch_through else 0.0,
                                tabs_per_edge=p.hatch_tabs_per_edge)
    elif p.hatch and p.bumps:
        warnings.append(
            "Hatching is for a flat lid: a groove still couples neighbours in shear, so a bumped "
            "lid uses through-slits instead. The hatch was ignored.")
    if p.bumps:
        centers, _partial = bump_centers_from_taxels(
            placements_xy, active_idx, pitch_x, pitch_y)
        if p.slit_bumps and len(centers):
            slits = _slit_geometry(recess, centers, pitch_x, pitch_y, bp,
                                   open_edge_y=None if tail_x is not None else clip_y)

    # The bump slits and the rib slot are concentric cuts near the pocket edge.
    # What is left between them is a thin continuous ring carrying flange loads
    # into the tile field, and it narrows as the skirt narrows -- so measure it
    # rather than trusting that two independently sensible numbers compose.
    web = None
    if slits is not None and not slits.is_empty and not slot.is_empty:
        web = float(slits.distance(slot))
        if web < MIN_WEB_MM:
            warnings.append(
                f"Only {web:.2f}mm of lid is left between the bump slits and the rib slot, and "
                "that ring carries the whole tile field. Widen the skirt (each extra 1mm of "
                "skirt adds about 0.5mm here) or narrow the bump slit.")

    lid_2d = sheet.difference(slot)
    # Every THROUGH cut has to count here, or the piece check silently ignores
    # the one most likely to fall apart: a hatch whose tabs failed leaves one
    # loose tile per taxel. A groove is not a through cut and must not count.
    lid_cut_2d = lid_2d
    for cut in (slits, hatch if p.hatch_through else None):
        if cut is not None and not cut.is_empty:
            lid_cut_2d = lid_cut_2d.difference(cut)
    n_lid = len(lid_cut_2d.geoms) if isinstance(lid_cut_2d, MultiPolygon) else 1

    slab = trimesh.creation.extrude_polygon(_clean(sheet), height=p.lid_thickness_mm)
    cutters = []
    for geom in (g for g in (slot, slits) if g is not None and not g.is_empty):
        for piece in _polygons(geom):
            if piece.area <= 1e-9:
                continue
            try:
                c = trimesh.creation.extrude_polygon(_clean(piece),
                                                     height=p.lid_thickness_mm + 1.0)
            except Exception:
                continue
            if not c.is_volume:
                continue
            c.apply_translation((0, 0, -0.5))
            cutters.append(c)
    # Grooves are PARTIAL depth and open at the top: the underside has to stay
    # dead flat against the Velostat, or the lid presses along the boundaries —
    # exactly where the compliance is supposed to be.
    if hatch is not None and not hatch.is_empty:
        through = p.hatch_through
        depth = p.lid_thickness_mm if through else p.hatch_depth_mm
        for piece in _polygons(hatch):
            if piece.area <= 1e-9:
                continue
            try:
                c = trimesh.creation.extrude_polygon(_clean(piece), height=depth + 1.0)
            except Exception:
                continue
            if not c.is_volume:
                continue
            c.apply_translation((0, 0, p.lid_thickness_mm - depth - (0.5 if through else 0.0)))
            cutters.append(c)
    if cutters:
        try:
            slab = trimesh.boolean.difference([slab] + cutters)
        except Exception as exc:
            warnings.append(f"Cutting the lid failed ({exc}); it is uncut, so the rib has no "
                            "slot to pass through and the bumps are not decoupled.")
    tops = [slab]

    if p.hatch and p.hatch_bumps and hatch is not None:
        face = p.hatch_bump_face if p.hatch_bump_face in ("top", "bottom") else "top"
        tile = min(pitch_x, pitch_y) - p.hatch_groove_mm
        if p.hatch_bump_dia_mm > tile - 0.4:
            warnings.append(
                f"A {p.hatch_bump_dia_mm:g}mm bump does not fit a {tile:.1f}mm tile with clearance; "
                "it will run into the cut around it.")
        rb = (p.hatch_bump_dia_mm / 2.0, p.hatch_bump_dia_mm / 2.0)
        rt = (p.hatch_bump_top_ratio * rb[0], p.hatch_bump_top_ratio * rb[1])
        plug = min(0.25, p.lid_thickness_mm * 0.4)
        pts = np.asarray(placements_xy, dtype=float)[np.asarray(sorted(active_idx), dtype=int)] \
            if len(active_idx) else np.zeros((0, 2))
        for cx, cy in pts:
            if face == "top":
                tops.append(_frustum(cx, cy, p.lid_thickness_mm - plug, plug, rb, rb, p.segments))
                tops.append(_frustum(cx, cy, p.lid_thickness_mm, p.hatch_bump_height_mm,
                                     rb, rt, p.segments))
            else:
                # Built upward from below the lid so the wide end meets it: the
                # pip narrows AWAY from the lid, same profile as a top bump seen
                # from the other side.
                tops.append(_frustum(cx, cy, -p.hatch_bump_height_mm, p.hatch_bump_height_mm,
                                     rt, rb, p.segments))
                tops.append(_frustum(cx, cy, 0.0, plug, rb, rb, p.segments))
        n_pips = len(pts)
    else:
        n_pips = 0

    if p.bumps and len(centers):
        # A bump needs lid under it: one over a slit or the rib slot has nothing
        # to stand on.
        keep = [c for c in centers
                if lid_cut_2d.intersects(Polygon([(c[0] - 0.1, c[1] - 0.1), (c[0] + 0.1, c[1] - 0.1),
                                                  (c[0] + 0.1, c[1] + 0.1), (c[0] - 0.1, c[1] + 0.1)]))]
        n_bumps = len(keep)
        if len(keep) < len(centers):
            warnings.append(f"{len(centers) - len(keep)} bump(s) had no lid under them and were "
                            "dropped.")
        rb = (bp.bump_base_frac * pitch_x, bp.bump_base_frac * pitch_y)
        rt = (bp.bump_top_ratio * rb[0], bp.bump_top_ratio * rb[1])
        plug = min(0.3, p.lid_thickness_mm * 0.5)
        for cx, cy in keep:
            tops.append(_frustum(cx, cy, p.lid_thickness_mm - plug, plug, rb, rb, p.segments))
            tops.append(_frustum(cx, cy, p.lid_thickness_mm, bp.bump_height_mm, rb, rt, p.segments))
    top = trimesh.boolean.union(tops) if len(tops) > 1 else tops[0]

    if n_lid > 1:
        warnings.append(
            f"The cuts leave the lid in {n_lid} loose pieces. Widen the rib gap, or the hatch "
            "or bump tabs, so the bridges hold it together.")

    engaged = rib.area / p.rib_head_mm if p.rib_head_mm else 0.0
    stats = {
        "rib_segments": len(rib.geoms) if isinstance(rib, MultiPolygon) else 1,
        "rib_engaged_mm": round(float(engaged), 1),
        "bridges": int(n_gaps),
        "bridge_mm": round(float(max(0.0, total - (sheet.bounds[2] - sheet.bounds[0]) - engaged)), 1),
        "rib_period_mm": round(float(period), 1),
        "rib_gap_mm": round(float(gap), 1),
        "perimeter_mm": round(float(total), 1),
        "lid_pieces": int(n_lid),
        "closed_height_mm": round(z_seat + p.lid_thickness_mm, 2),
        "case_w_mm": round(float(sheet.bounds[2] - sheet.bounds[0]), 1),
        "case_h_mm": round(float(sheet.bounds[3] - sheet.bounds[1]), 1),
        "bumps": int(n_bumps),
        "slit": bool(slits is not None and not slits.is_empty),
        "hatch": bool(hatch is not None and not hatch.is_empty),
        "hatch_through": bool(hatch is not None and p.hatch_through),
        "hatch_bumps": int(n_pips),
        "hatch_bump_face": p.hatch_bump_face if n_pips else None,
        "hatch_web_mm": (None if hatch is None or p.hatch_through
                         else round(p.lid_thickness_mm - p.hatch_depth_mm, 2)),
        "slit_to_slot_mm": round(web, 2) if web is not None else None,
        "undercut_mm": round((p.rib_head_mm - p.rib_wall_mm) / 2, 2),
        "cable_edge": "slot" if tail_x is not None else "open side",
        "tail_free_mm": round(clear["tail_free_mm"], 1),
        "tail_extension_mm": round(tail_extension_length(tail_extension), 1),
        "tail_flex_mm": round(clear["tail_flex_mm"], 1),
        "tail_short_by_mm": round(clear["short_mm"], 1) if clear["short_mm"] else None,
        "tail_slot_mm": round(tail_x[1] - tail_x[0], 1) if tail_x else None,
    }
    return CaseResult(bottom=bottom, top=top, stats=stats, warnings=warnings)
