"""Polygon geometry utilities — wrappers around Shapely."""

import math

import numpy as np
from shapely.geometry import LineString, Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from shapely import union_all

CONNECTOR_BODY_ABOVE_MM = 2.75
CONNECTOR_BODY_BELOW_MM = 3.25


def make_polygon(outline_xy: np.ndarray) -> Polygon:
    """Create a Shapely Polygon from Nx2 vertices."""
    poly = Polygon(outline_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def polygon_buffer(poly: Polygon, distance: float) -> Polygon:
    """Shrink (negative distance) or grow polygon. Returns Polygon or MultiPolygon."""
    result = poly.buffer(distance, join_style='mitre')
    if result.is_empty:
        return Polygon()
    return result


def round_convex_corners(poly: Polygon, radius: float, quad_segs: int = 16):
    """Round a polygon's *convex* corners by a morphological opening.

    Shrink-then-grow with round joins: each outward 90° corner comes back as a
    quarter arc of `radius`, straight edges are unchanged. The radius is capped
    against the shape's own short dimension so the erode step can never eat the
    polygon away (a rectangle narrower than 2*radius would otherwise vanish),
    and the whole thing falls back to the input if the result degenerates.
    """
    if radius <= 0 or poly.is_empty:
        return poly
    minx, miny, maxx, maxy = poly.bounds
    r = min(radius, 0.45 * min(maxx - minx, maxy - miny))
    if r <= 1e-6:
        return poly
    opened = poly.buffer(-r, join_style="round", quad_segs=quad_segs) \
                 .buffer(r, join_style="round", quad_segs=quad_segs)
    if opened.is_empty or opened.area <= 0:
        return poly
    return opened


def fillet_inside_corners(poly, radius: float, near=None):
    """`poly` with its inside corners rounded to `radius`.

    Adds whole fillet wedges from concave_fillet_infill, skipping the slivers a
    buffer leaves along an already-curved edge (they are far smaller than the
    r^2 (1 - pi/4) wedge of a square corner). With `near`, only wedges touching
    it."""
    infill = concave_fillet_infill(poly, radius)
    areas = [g for g in getattr(infill, "geoms", [infill])
             if g.geom_type == "Polygon" and g.area > 0.05 * radius * radius
             and (near is None or g.distance(near) < 1e-6)]
    if not areas:
        return poly
    # A wedge meets the edge it rounds tangentially, a hair off the line: the
    # plain union can keep a zero-width spike along it, which KiCad reads as a
    # self-intersecting outline. Closing by 10 nm merges it.
    eps = 1e-5
    return unary_union([poly, *areas]).buffer(eps, join_style=2).buffer(-eps, join_style=2)


def concave_fillet_infill(poly, radius: float, zone=None, quad_segs: int = 16):
    """Material that rounds a polygon's *concave* corners, as its own polygon.

    A morphological closing (grow-then-shrink, round joins) turns every inward
    corner into a fillet arc; the difference against the original is exactly the
    small wedge of material that fillet adds. Returned separately rather than
    applied so callers can union it in as one more tail piece — the
    tail_geometry_lock replay in pipeline/follow_padding.py can only reproduce
    *additive* pieces, so a fillet expressed as an in-place shape edit would be
    silently dropped on the next request.

    `zone` limits the result to a region of interest, leaving concave corners
    elsewhere on the board untouched.
    """
    if radius <= 0 or poly.is_empty:
        return Polygon()
    closed = poly.buffer(radius, join_style="round", quad_segs=quad_segs) \
                 .buffer(-radius, join_style="round", quad_segs=quad_segs)
    if closed.is_empty:
        return Polygon()
    added = closed.difference(poly)
    if zone is not None and not zone.is_empty:
        added = added.intersection(zone)
    if added.is_empty or added.area <= 0:
        return Polygon()
    return added


def inside_or_on_polygon(poly, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
    """Check if points are inside OR on boundary of polygon (handles MultiPolygon)."""
    n = len(xq)
    result = np.zeros(n, dtype=bool)

    if isinstance(poly, MultiPolygon):
        geoms = list(poly.geoms)
    elif isinstance(poly, Polygon):
        geoms = [poly]
    else:
        return result

    # Vectorized but predicate-identical to the original per-point loop:
    # shapely.contains_xy is the same GEOS contains, and the boundary
    # tolerance keeps the strict `distance < 1e-9` comparison (NOT dwithin,
    # whose <= differs at exactly 1e-9). Profiled: the per-point version was
    # the packer's whole cost, and packing dominates fixed-mode carves.
    import shapely as _shapely
    xa = np.asarray(xq, dtype=float)
    ya = np.asarray(yq, dtype=float)
    for geom in geoms:
        todo = np.flatnonzero(~result)
        if len(todo) == 0:
            break
        inside = _shapely.contains_xy(geom, xa[todo], ya[todo])
        result[todo[inside]] = True
        miss = todo[~inside]
        if len(miss):
            pts = _shapely.points(np.column_stack([xa[miss], ya[miss]]))
            near = _shapely.distance(geom.boundary, pts) < 1e-9
            result[miss[near]] = True

    return result


def poly_bounds(poly) -> tuple[float, float, float, float]:
    """Return (xmin, xmax, ymin, ymax) of polygon."""
    b = poly.bounds  # (minx, miny, maxx, maxy)
    return b[0], b[2], b[1], b[3]


def edge_tangent_and_outward_normal(outline_poly: Polygon, p1: np.ndarray, p2: np.ndarray):
    """Compute unit tangent and outward normal for edge p1->p2."""
    e = p2 - p1
    t_hat = e / np.linalg.norm(e)

    n1 = np.array([-t_hat[1], t_hat[0]])
    n2 = -n1

    mid = 0.5 * (p1 + p2)
    eps = 0.5
    test1 = mid + eps * n1
    test2 = mid + eps * n2

    in1 = outline_poly.contains(Point(test1[0], test1[1]))
    in2 = outline_poly.contains(Point(test2[0], test2[1]))

    if in1 and not in2:
        n_out = n2
    elif in2 and not in1:
        n_out = n1
    else:
        n_out = n2

    return t_hat, n_out


def make_cable_rect(p1, p2, t_hat, n_out, width_mm, length_mm) -> Polygon:
    """Build cable rectangle extending from edge midpoint."""
    mid = 0.5 * (p1 + p2)
    A = mid - (width_mm / 2) * t_hat
    B = mid + (width_mm / 2) * t_hat
    C = B + length_mm * n_out
    D = A + length_mm * n_out
    return Polygon([A, B, C, D])


def connector_required_tail_width(conn_pos: int, pad_pitch_mm: float) -> float:
    """Minimum tail width needed to expose the connector edge pads."""
    return connector_dimensions(conn_pos, pad_pitch_mm)["tab_tip_width_mm"]


def connector_dimensions(conn_pos: int, pad_pitch_mm: float) -> dict:
    """Return connector width dimensions derived from pin count and pitch."""
    pad_center_span = max(0, conn_pos - 1) * pad_pitch_mm
    copper_span = pad_center_span + 0.3
    tab_tip_width = (conn_pos + 1) * pad_pitch_mm
    ear_stiffener_width = tab_tip_width + 1.1
    return {
        "pad_center_span_mm": pad_center_span,
        "copper_span_mm": copper_span,
        "tab_tip_width_mm": tab_tip_width,
        "ear_stiffener_width_mm": ear_stiffener_width,
    }


def fit_tail_strip(pad_to_net: dict, conn_pos: int, params, edge_mm: float) -> tuple[float, float] | None:
    """Narrowest tail strip that carries the connector pads in use: (width, offset).

    The readout boards fix which connector half carries the rows and which the
    columns (60-pin: rows on 1-30, columns on 31-60; the 30-pin board the other
    way round), and each side's used pads are packed against the connector's
    centre, so the used pads form one contiguous block that sits off-centre
    whenever rows and columns differ in number (a finger pad with 8 rows and 27
    columns: pads 23-57). The strip covers that block, each outermost wire running
    straight into its pad `edge_mm` from the cut edge, so it is centred on the
    block rather than on the connector: `offset` is how far its centre sits from
    the connector's centre line, along t_hat (pad numbers grow along +t_hat).
    It never reaches past the connector's own tab tip on either side; the tail
    widens back to that at the connector (make_tail_poly).
    """
    pads = [int(p) for p in (pad_to_net or {})]
    if not pads:
        return None
    centre = (conn_pos + 1) / 2.0
    pitch = float(params.pad_pitch_mm)
    half_tab = connector_required_tail_width(conn_pos, pitch) / 2.0
    lo = round(max(-half_tab, (min(pads) - centre) * pitch - edge_mm), 4)
    hi = round(min(half_tab, (max(pads) - centre) * pitch + edge_mm), 4)
    return round(hi - lo, 4), round((lo + hi) / 2.0, 4)


def tail_offset_mm(tail) -> float:
    """Where a narrowed tail's strip is centred, along t_hat from the connector's
    centre line (fit_tail_strip). 0 for a tail that was not cut to its wires,
    and for designs saved before the strip could sit off-centre."""
    t = tail or {}
    if t.get("mode") != "fit":
        return 0.0
    return float(t.get("fit_offset_mm") or 0.0)


# How a narrow tail widens back to the connector's tab, per side.
#
# The tail cut to its wires after routing (narrow_tail.narrow_routed_tail) steps
# out to the tab right before it, as on the boards fabricated before: the cable
# runs straight into the tab and each side turns out square, through a concave
# fillet as round as the tail's corners where it leaves the board
# (Params.connector_shoulder_fillet_mm, stored with the tail as step_fillet_mm),
# then round a small convex corner (TAIL_STEP_CORNER_MM) onto the tab's edge.
# No inside corner is sharper than the fillet, so the flex has nowhere to tear.
TAIL_STEP_CORNER_MM = 0.5
TAIL_STEP_ARC_STEPS = 12     # segments per quarter turn
# A tail of a width the user typed is routed with its widening (layout.py), and
# the router fans wires in through it to pads outside the typed width, so it
# keeps the long half-cosine flare it was routed with: steepest 45 degrees, tangent
# to the cable and to the landing at its ends. So do tails narrowed by the
# designs saved before the step, so they come back as they were made.
TAIL_FLARE_MAX_SLOPE = 1.0
TAIL_FLARE_STEPS = 16


def _side_steps(lo: float, hi: float, wide_w: float) -> tuple[float, float]:
    """How far each side of a strip [lo, hi] (along t_hat from the connector's
    centre line) has to move out to the tab, `wide_w` wide around that line."""
    return max(0.0, lo + wide_w / 2.0), max(0.0, wide_w / 2.0 - hi)


def _flare_sides(narrow_w: float, wide_w: float, room: float, offset: float):
    """The half-cosine flare: both sides take the same length, set by the side
    that has further to go, kept within `room`."""
    lo, hi = offset - narrow_w / 2.0, offset + narrow_w / 2.0
    d_lo, d_hi = _side_steps(lo, hi, wide_w)
    length = max(0.1, min(math.pi / 2.0 * max(d_lo, d_hi) / TAIL_FLARE_MAX_SLOPE, room))
    n = TAIL_FLARE_STEPS
    f = [(1.0 - math.cos(math.pi * i / n)) / 2.0 for i in range(n + 1)]
    u = [length * (1.0 - i / n) for i in range(n + 1)]
    return ([(u[i], lo - d_lo * f[i]) for i in range(n + 1)],
            [(u[i], hi + d_hi * f[i]) for i in range(n + 1)])


def _step_side(edge: float, tab: float, sign: float, fillet: float, corner: float, room: float):
    """One side of the step, from the strip's edge `edge` out to the tab's edge
    `tab` (`sign` the way it moves along t_hat), as (distance before the
    landing, position along t_hat) points.

    Leaves the strip round the concave fillet, runs straight across (square to
    the tail), and meets the tab's edge round the convex corner exactly at the
    landing. A step shorter than the two radii keeps them and turns less than a
    right angle: an S of the same two arcs, so its inside radius is still the
    fillet. Both radii shrink, together, only when the cable has less than their
    sum before the landing (`room`)."""
    d = abs(tab - edge)
    if d < 1e-6:
        return [(0.0, tab)]
    r1, r2 = max(0.0, fillet), max(0.0, corner)
    total = r1 + r2
    if total < 1e-9:
        return [(0.0, edge), (0.0, tab)]
    if room < total:
        k = max(room, 1e-3) / total
        r1, r2, total = r1 * k, r2 * k, total * k
    theta = math.pi / 2.0 if d >= total else math.acos(1.0 - d / total)
    u0 = total * math.sin(theta)
    n1 = max(2, math.ceil(TAIL_STEP_ARC_STEPS * theta / (math.pi / 2.0)))
    pts = [(u0 - r1 * math.sin(theta * i / n1), edge + sign * r1 * (1.0 - math.cos(theta * i / n1)))
           for i in range(n1 + 1)]
    pts += [(r2 * math.sin(theta * (1.0 - j / n1)), tab - sign * r2 * (1.0 - math.cos(theta * (1.0 - j / n1))))
            for j in range(n1 + 1)]
    out = [pts[0]]
    for p in pts[1:]:
        if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-7:
            out.append(p)
    return out


def tail_side_edges(narrow_w: float, wide_w: float, room: float, offset: float = 0.0,
                    step_fillet: float | None = None):
    """How each side of a narrow tail widens to the connector's full-width
    landing: (low side, high side), each a list of (distance before the landing,
    position along t_hat from the connector's centre line) points from where it
    leaves the strip to where it meets the tab at the landing (distance 0).

    The strip is `narrow_w` wide centred `offset` off that line, the landing
    `wide_w` wide centred on it, so each side moves out by its own amount (0: it
    runs straight on into the tab). `step_fillet` given: the quick step
    (_step_side) with that inside radius. None: the half-cosine flare a typed
    width is routed with. Kept within `room` before the landing.
    """
    if step_fillet is None:
        return _flare_sides(narrow_w, wide_w, room, offset)
    lo, hi = offset - narrow_w / 2.0, offset + narrow_w / 2.0
    return (_step_side(lo, min(lo, -wide_w / 2.0), -1.0, step_fillet, TAIL_STEP_CORNER_MM, room),
            _step_side(hi, max(hi, wide_w / 2.0), 1.0, step_fillet, TAIL_STEP_CORNER_MM, room))


def tail_widening_length(narrow_w: float, wide_w: float, room: float, offset: float = 0.0,
                         step_fillet: float | None = None) -> float:
    """How far before the landing the widening (tail_side_edges) starts."""
    low, high = tail_side_edges(narrow_w, wide_w, room, offset, step_fillet)
    return max(low[0][0], high[0][0])


def tail_step_fillet(tail) -> float | None:
    """The inside radius a stored tail's step was made with (tail_side_edges),
    or None for a tail that flares: one of a typed width, or one narrowed by a
    design saved before the step."""
    r = (tail or {}).get("step_fillet_mm")
    return None if r is None else float(r)


def make_tail_poly(mid, t_hat, n_out, tail_width: float, connector_width: float,
                   cable_length: float, total_length: float, offset: float = 0.0,
                   step_fillet: float | None = None) -> Polygon:
    """The tail from the board edge to the connector's far end.

    A tail as wide as the connector is a plain rectangle. A narrower one runs at
    its own width for the cable, centred `offset` off the connector's centre
    line (fit_tail_strip), then widens (tail_side_edges: a quick step with
    `step_fillet`, else the flare) to the connector's tab width so the connector
    lands on a full-width tab, centred on the connector, which the connector
    footprint's own Edge.Cuts continue from (export stitched_outline_segments).
    The landing starts at cable_length, ahead of the footprint's open edge.
    """
    mid, t_hat, n_out = (np.asarray(v, dtype=float) for v in (mid, t_hat, n_out))
    w, W = float(tail_width), float(connector_width)
    if w >= W - 1e-6:
        return make_cable_rect(mid, mid, t_hat, n_out, w, total_length)
    low, high = tail_side_edges(w, W, max(0.5, cable_length - 1.0), offset, step_fillet)
    left = [mid + (offset - w / 2) * t_hat]
    left += [mid + s * t_hat + (cable_length - u) * n_out for u, s in low]
    left.append(mid - (W / 2) * t_hat + total_length * n_out)
    right = [mid + (W / 2) * t_hat + total_length * n_out]
    right += [mid + s * t_hat + (cable_length - u) * n_out for u, s in reversed(high)]
    right.append(mid + (offset + w / 2) * t_hat)
    return Polygon(left + right)


def tail_without_landing(cable_poly, mid, t_hat, n_out, width: float, connector_width: float,
                         cable_length: float, total_length: float, offset: float = 0.0,
                         step_fillet: float | None = None, landing_cover=()) -> Polygon:
    """A narrow tail once its connector has moved to the end of a drawn tail
    extension: the widening and landing (make_tail_poly) belong to the
    connector, so the plain strip runs on in their place to where the extension
    starts, at the pad row; the extension's corridor carries on from there
    (extension_tail_poly), turning with the path. Run on to the old landing's
    end, the strip stuck out past the outside of a path that turns at once.
    `landing_cover`: the board the narrowed tail kept round wires in its step
    and landing (narrow_tail.narrow_routed_tail), kept up to the pad row."""
    if cable_poly is None or width >= connector_width - 1e-6:
        return cable_poly
    mid, t_hat, n_out = (np.asarray(v, dtype=float) for v in (mid, t_hat, n_out))
    start = cable_length - tail_widening_length(width, connector_width, max(0.5, cable_length - 1.0), offset,
                                                step_fillet)
    big = 1e4
    before = Polygon([mid - big * t_hat - big * n_out, mid + big * t_hat - big * n_out,
                      mid + big * t_hat + start * n_out, mid - big * t_hat + start * n_out])
    strip_mid = mid + offset * t_hat
    pad_row = cable_length + CONNECTOR_BODY_ABOVE_MM
    strip = make_cable_rect(strip_mid, strip_mid, t_hat, n_out, width, pad_row)
    pieces = [cable_poly.intersection(before), strip]
    if landing_cover:
        to_pad_row = Polygon([mid - big * t_hat - big * n_out, mid + big * t_hat - big * n_out,
                              mid + big * t_hat + pad_row * n_out, mid - big * t_hat + pad_row * n_out])
        pieces += [Polygon(c).buffer(0).intersection(to_pad_row) for c in landing_cover if len(c) >= 3]
    return unary_union(pieces).buffer(0)


def offset_path(path, offset: float):
    """`path` moved `offset` sideways, to the left of its direction of travel
    (the normal (-dy, dx) of each segment), with mitred corners: where a tail
    extension's lane for a pad `offset` from the split point runs. None for a
    path with a zero-length segment."""
    path = np.asarray(path, dtype=float)
    dirs, normals = [], []
    for i in range(len(path) - 1):
        d = path[i + 1] - path[i]
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return None
        d = d / n
        dirs.append(d)
        normals.append(np.array([-d[1], d[0]], dtype=float))
    if not dirs:
        return path.copy()
    out = []
    for i, p in enumerate(path):
        if i == 0:
            out.append(p + offset * normals[0])
        elif i == len(path) - 1:
            out.append(p + offset * normals[-1])
        else:
            p1 = p + offset * normals[i - 1]
            p2 = p + offset * normals[i]
            mat = np.column_stack((dirs[i - 1], -dirs[i]))
            det = float(np.linalg.det(mat))
            if abs(det) < 1e-9:
                out.append(p1)
            else:
                t = np.linalg.solve(mat, p2 - p1)[0]
                out.append(p1 + t * dirs[i - 1])
    return np.array(out, dtype=float)


def _side_band(line, lo: float, hi: float):
    """The points between `lo` and `hi` to the left of a polyline (negative:
    to its right), mitred at its corners and cut square at its ends. Built
    from one-sided buffers rather than an offset centre line: the inside of a
    turn after a segment shorter than the offset folds an offset line back on
    itself, and its buffer then misses the corner."""
    ls = LineString(np.asarray(line, dtype=float))

    def side(d):
        return ls.buffer(d, single_sided=True, join_style=2) if abs(d) > 1e-9 else Polygon()

    if lo < 0.0 < hi:
        # the two sides meet along the line itself: a plain buffer over it
        # keeps a hair crack between them out of the union
        core = ls.buffer(min(-lo, hi), cap_style=2, join_style=2)
        return unary_union([side(hi), side(lo), core])
    if lo >= 0.0:
        return side(hi).difference(side(lo))
    return side(lo).difference(side(hi))


def extension_tail_poly(path, width: float, clearance: float, connector_width,
                        offset: float = 0.0, step_fillet: float | None = None, lead_dir=None) -> Polygon:
    """A drawn tail extension's board: its corridor, ending in a connector landing
    when the corridor is narrower than the connector (connector_landing_poly).

    The corridor runs `offset` to the left of the drawn path (offset_path), so
    it stays over the wire lanes when the tail strip it continues sits off the
    connector's centre (fit_tail_strip); the landing is centred on the path's
    end, where the connector goes. `lead_dir`, the direction the tail runs in
    before the path, makes the corridor carry on from the tail and mitre where
    the path turns away from it, as the wire lanes do
    (tail_extension.apply_tail_extension): it starts back along the tail, far
    enough that the inside of the turn is board too."""
    path = np.asarray(path, dtype=float)
    line = path
    if lead_dir is not None and len(path) >= 2:
        lead = np.asarray(lead_dir, dtype=float)
        lead = lead / max(float(np.linalg.norm(lead)), 1e-12)
        # the first turn away from the tail's direction (a path may start with
        # a straight run down the tail, tail_extension._lane_lead_in)
        half_turn = 0.0
        for a, b in zip(path[:-1], path[1:]):
            seg = (b - a) / max(float(np.linalg.norm(b - a)), 1e-12)
            if float(lead @ seg) < 1.0 - 1e-9:
                half_turn = 0.5 * math.acos(max(-1.0, min(1.0, float(lead @ seg))))
                break
        reach = (abs(offset) + width / 2 + clearance) * math.tan(min(half_turn, math.pi / 4))
        line = np.vstack([path[0] - (reach + 0.5) * lead, path])
    corridor = _side_band(line, offset - width / 2 - clearance, offset + width / 2 + clearance)
    narrow = width + 2 * clearance
    W = float(connector_width or 0.0)
    if len(path) < 2 or W <= narrow + 1e-6:
        return corridor
    d = path[-1] - path[-2]
    seg = float(np.linalg.norm(d))
    if seg < 1e-9:
        return corridor
    n_out = d / seg
    t_hat = np.array([-n_out[1], n_out[0]])
    landing = connector_landing_poly(path[-1], t_hat, n_out, narrow, W, room=seg - CONNECTOR_BODY_ABOVE_MM,
                                     offset=offset, step_fillet=step_fillet)
    return unary_union([corridor, landing])


def connector_landing_poly(pad_row_center, t_hat, n_out, narrow_w: float, connector_width: float,
                           room: float, offset: float = 0.0, step_fillet: float | None = None) -> Polygon:
    """The widening and full-width landing alone, for a connector at the end of
    a drawn tail extension whose corridor is narrower than the connector: the
    same shape make_tail_poly ends in, placed around the moved pad row, widening
    from a corridor centred `offset` off it."""
    c, t_hat, n_out = (np.asarray(v, dtype=float) for v in (pad_row_center, t_hat, n_out))
    low, high = tail_side_edges(narrow_w, connector_width, max(0.5, room), offset, step_fillet)
    W = float(connector_width)
    land = -CONNECTOR_BODY_ABOVE_MM
    left = [c + s * t_hat + (land - u) * n_out for u, s in low]
    left.append(c - (W / 2) * t_hat + CONNECTOR_BODY_BELOW_MM * n_out)
    right = [c + (W / 2) * t_hat + CONNECTOR_BODY_BELOW_MM * n_out]
    right += [c + s * t_hat + (land - u) * n_out for u, s in reversed(high)]
    return Polygon(left + right)


def resolve_tail(params, p1, p2, conn_pos: int, fit_width: float | None = None,
                 fit_offset: float | None = None) -> dict:
    """Resolve tail dimensions.

    Automatic width: `fit_width` (the narrow tail that just carries the traces,
    centred `fit_offset` off the connector's centre line, see fit_tail_strip)
    when given, else the connector's tab tip width. Manual user widths are
    respected, including values smaller than the connector tip, and are centred
    on the connector.
    """
    required = connector_required_tail_width(conn_pos, params.pad_pitch_mm)
    dimensions = connector_dimensions(conn_pos, params.pad_pitch_mm)
    requested = getattr(params, "tail_width_mm", None)
    side_padding = max(0.0, getattr(params, "tail_side_padding_mm", 0.0))
    edge_len = float(np.linalg.norm(p2 - p1))
    available = edge_len - 2 * side_padding

    requested_width = None if requested is None or requested <= 0 else requested
    mode = "auto"
    clamp_reason = None

    if requested_width is not None:
        mode = "manual"
        width = requested_width
    elif side_padding > 0:
        mode = "side_padding"
        width = available
    elif fit_width is not None:
        mode = "fit"
        width = fit_width
    else:
        width = required

    return {
        "mode": mode,
        "requested_width_mm": requested_width,
        "width_mm": width,
        "connector_min_width_mm": required,
        **dimensions,
        "side_padding_mm": side_padding,
        "edge_length_mm": edge_len,
        "available_width_mm": available,
        "clamp_reason": clamp_reason,
        "length_mm": params.cable_length_mm,
        # Frozen with the layout, so every exporter draws the same tail.
        "fit_width_mm": fit_width,
        "fit_offset_mm": (fit_offset or 0.0) if mode == "fit" else None,
    }


def tail_geometry(outline_poly: Polygon, p1, p2, conn_pos: int, params,
                  fit_width: float | None = None, fit_offset: float | None = None) -> dict:
    """Compute shared connector-tail geometry.

    The connector stays centered on the first outline edge (`mid`, `pad1_world`,
    `pad_row_center`); a narrowed tail's strip (A-D) may sit `fit_offset` off
    that line, over the pads it carries (fit_tail_strip). Edge cuts use
    cable_length_mm; layout/preview board union uses the total connector body
    envelope so routing can legally enter the connector area.
    """
    t_hat, n_out = edge_tangent_and_outward_normal(outline_poly, p1, p2)
    mid = 0.5 * (p1 + p2)

    resolved = resolve_tail(params, p1, p2, conn_pos, fit_width, fit_offset)
    tail_width = resolved["width_mm"]
    tail_offset = float(resolved["fit_offset_mm"] or 0.0)
    connector_width = resolved["connector_min_width_mm"]
    cable_length = params.cable_length_mm
    total_length = cable_length + CONNECTOR_BODY_ABOVE_MM + CONNECTOR_BODY_BELOW_MM

    strip_mid = mid + tail_offset * t_hat
    A = strip_mid - (tail_width / 2) * t_hat
    B = strip_mid + (tail_width / 2) * t_hat
    D = A + cable_length * n_out
    C = B + cable_length * n_out

    At = strip_mid - (tail_width / 2) * t_hat
    Bt = strip_mid + (tail_width / 2) * t_hat
    Dt = At + total_length * n_out
    Ct = Bt + total_length * n_out

    connector_left = mid - (connector_width / 2) * t_hat
    pad1_world = (
        connector_left
        + params.pad_pitch_mm * t_hat
        + (cable_length + CONNECTOR_BODY_ABOVE_MM) * n_out
    )
    pad_row_center = mid + (cable_length + CONNECTOR_BODY_ABOVE_MM) * n_out

    return {
        "t_hat": t_hat,
        "n_out": n_out,
        "mid": mid,
        "tail_width": tail_width,
        "tail_offset": tail_offset,
        "connector_width": connector_width,
        "cable_length": cable_length,
        "total_length": total_length,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "At": At,
        "Bt": Bt,
        "Ct": Ct,
        "Dt": Dt,
        "pad1_world": pad1_world,
        "pad_row_center": pad_row_center,
        "resolved": resolved,
    }


def pt_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Point-to-segment distance.

    Written out in scalars rather than numpy vector ops. These are 2D points and
    this is one of the hottest functions in the router (~250k calls on a 16x16
    board); at that size np.dot/np.linalg.norm are dominated by their own
    per-call dispatch, and math.hypot is ~5x faster than np.linalg.norm on a
    2-vector. Same arithmetic, same result.
    """
    ax, ay = float(a[0]), float(a[1])
    abx, aby = float(b[0]) - ax, float(b[1]) - ay
    px, py = float(p[0]) - ax, float(p[1]) - ay
    len2 = abx * abx + aby * aby
    if len2 < 1e-24:
        return math.hypot(px, py)
    t = (px * abx + py * aby) / len2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return math.hypot(px - t * abx, py - t * aby)


def seg_seg_dist(a1, a2, b1, b2) -> float:
    """Segment-to-segment minimum distance."""
    if segments_intersect(a1, a2, b1, b2):
        return 0.0
    return min(
        pt_seg_dist(a1, b1, b2),
        pt_seg_dist(a2, b1, b2),
        pt_seg_dist(b1, a1, a2),
        pt_seg_dist(b2, a1, a2),
    )


def segments_intersect(a1, a2, b1, b2) -> bool:
    """Check if two segments properly intersect (cross-product method)."""
    def cross2d(u, v):
        return u[0] * v[1] - u[1] * v[0]

    d1 = cross2d(b2 - b1, a1 - b1)
    d2 = cross2d(b2 - b1, a2 - b1)
    d3 = cross2d(a2 - a1, b1 - a1)
    d4 = cross2d(a2 - a1, b2 - a1)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def remove_collinear(P: np.ndarray) -> np.ndarray:
    """Remove collinear intermediate points from a polyline."""
    if len(P) < 3:
        return P
    keep = [True] * len(P)
    for i in range(1, len(P) - 1):
        a = P[i] - P[i - 1]
        b = P[i + 1] - P[i]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            keep[i] = False
            continue
        if abs(a[0] * b[1] - a[1] * b[0]) < 1e-6 * na * nb:
            keep[i] = False
    return P[keep]


# A ring vertex the outline turns back at, by more than SPIKE_TURN_DEG, next to
# an edge shorter than SPIKE_EDGE_MM, is the tip of a hairline spike a polygon
# union leaves where two nearly parallel edges meet (a containment buffer along
# a 4-layer board's edge: 0.02 mm long). Exported, it is a zero-width spike in
# Edge.Cuts (export/edge_cuts_check.py).
SPIKE_TURN_DEG = 170.0
SPIKE_EDGE_MM = 0.1


def drop_hairline_spikes(P: np.ndarray) -> np.ndarray:
    """A closed ring without the tips of hairline spikes (see SPIKE_TURN_DEG).
    A ring with none comes back unchanged."""
    pts = [np.asarray(p, dtype=float) for p in P]
    limit = math.cos(math.radians(SPIKE_TURN_DEG))
    dropped = False
    i = 0
    while len(pts) > 3 and i < len(pts):
        n = len(pts)
        # the turn at pts[i], looking past repeated points on either side
        j = (i - 1) % n
        while j != i and float(np.linalg.norm(pts[i] - pts[j])) <= 1e-9:
            j = (j - 1) % n
        k = (i + 1) % n
        while k != i and float(np.linalg.norm(pts[k] - pts[i])) <= 1e-9:
            k = (k + 1) % n
        a, b = pts[i] - pts[j], pts[k] - pts[i]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if (j != i and k != i and min(na, nb) < SPIKE_EDGE_MM
                and float(a @ b) < limit * na * nb):
            # the tip and its repeats: the run strictly between j and k
            run = {(j + 1 + m) % n for m in range((k - j - 1) % n)}
            pts = [p for m, p in enumerate(pts) if m not in run]
            dropped = True
            i = max(0, min(i, len(pts)) - 1)   # the vertex before may be a tip now
            continue
        i += 1
    return np.array(pts, dtype=float) if dropped else np.asarray(P)


def remove_collinear_closed(P: np.ndarray) -> np.ndarray:
    """Remove collinear points from a closed polygon ring without duplicating the
    first point, and the tips of hairline spikes (drop_hairline_spikes)."""
    if len(P) < 3:
        return P
    P = drop_hairline_spikes(P)
    keep = []
    n = len(P)
    for i in range(n):
        prev_pt = P[(i - 1) % n]
        curr = P[i]
        next_pt = P[(i + 1) % n]
        a = curr - prev_pt
        b = next_pt - curr
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            continue
        if abs(a[0] * b[1] - a[1] * b[0]) < 1e-6 * na * nb:
            continue
        keep.append(curr)
    return np.array(keep, dtype=float) if keep else P


def ensure_ccw(outline_xy: np.ndarray) -> np.ndarray:
    """Ensure polygon vertices are in counter-clockwise order.

    Uses the shoelace formula: positive signed area = CCW.
    If CW, reverses vertices 1..N while keeping vertex 0 in place.
    """
    n = len(outline_xy)
    area2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        area2 += outline_xy[i, 0] * outline_xy[j, 1] - outline_xy[j, 0] * outline_xy[i, 1]
    if area2 < 0:
        # CW — reverse but keep vertex 0 as vertex 0
        outline_xy = np.vstack([outline_xy[0:1], outline_xy[1:][::-1]])
    return outline_xy


def cable_transform(outline_xy: np.ndarray):
    """The rigid map `rotate_for_cable` applies to POINTS, captured on its own.

    Split out so anything drawn in the pre-rotation frame can be carried into
    the canonical frame *later, from a different request* — the 3D unfold's uv
    map is produced by /unroll-mesh and needed again at export time, long after
    the rotation was chosen inside /generate. Recovering the transform by
    fitting the outline before and after would be the "reconstruct it later"
    mistake `_edge_cuts_points` documents; re-running this pure function on the
    same outline gives the identical answer by construction.

    Returns (R, mid, flip). A point p maps to `s * (R @ (p - mid))`, with
    s = -1 when `flip`. The flip negates x and y together, which is a 180
    degree rotation rather than a mirror, so the whole map is rigid and
    orientation-preserving — meaning a *direction* carries through as
    `s * (R @ v)` with no translation.
    """
    outline_xy = ensure_ccw(np.asarray(outline_xy, dtype=float))
    p1 = outline_xy[0]
    p2 = outline_xy[1]
    edge_vec = p2 - p1
    ang = np.arctan2(edge_vec[1], edge_vec[0])

    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s], [s, c]])
    mid = 0.5 * (p1 + p2)

    rotated = (R @ (outline_xy - mid).T).T
    # Centroid must end up above the cable edge.
    flip = bool(rotated.mean(axis=0)[1] < 0)
    return R, mid, flip


def apply_cable_transform(pts: np.ndarray, tf) -> np.ndarray:
    """Carry POSITIONS through `cable_transform`'s map."""
    R, mid, flip = tf
    out = (R @ (np.asarray(pts, dtype=float) - mid).T).T
    return -out if flip else out


def rotate_for_cable(outline_xy: np.ndarray, extra_xy: np.ndarray = None):
    """Rotate outline so edge 1->2 is horizontal at the bottom.

    The rest of the polygon ends up above the cable edge.
    Ensures CCW winding first, then matches MATLAB rotateForCable().

    If extra_xy is given (e.g. a sensor-fill region drawn in the same
    coordinate frame as outline_xy), it is carried through the identical
    rigid transform so the two stay aligned. Returns (rotated, extra_rotated)
    in that case, otherwise just rotated.

    The transform itself lives in `cable_transform` — use that directly when
    you need to carry other geometry into this frame later.
    """
    outline_xy = ensure_ccw(outline_xy)
    tf = cable_transform(outline_xy)
    rotated = np.ascontiguousarray(apply_cable_transform(outline_xy, tf))

    extra_rotated = None
    if extra_xy is not None:
        extra_rotated = np.ascontiguousarray(apply_cable_transform(extra_xy, tf))

    # Ensure p1 is left of p2 (t_hat points right). Reorders the array's first
    # two rows only — it moves no point, so it is not part of the transform.
    if rotated[0, 0] > rotated[1, 0]:
        rotated[[0, 1]] = rotated[[1, 0]]

    if extra_xy is not None:
        return rotated, extra_rotated
    return rotated


def contiguous_runs(sorted_vals, max_step):
    """Split an ascending 1D sequence into runs of consecutive values no more
    than max_step apart. Returns lists of indices into sorted_vals.

    Used by the exporters to draw one intra-column/row bus line per contiguous
    pixel run: a joined net (hug router) can span a concavity gap, and a single
    min-to-max bus would jump straight across it — the routed link trace is
    what connects the runs, around the concavity. Same gap rule as
    assign_rowcol_nets_segmented (pitch * 1.25 at the call sites).
    """
    runs = [[0]]
    for k in range(1, len(sorted_vals)):
        if sorted_vals[k] - sorted_vals[k - 1] > max_step:
            runs.append([])
        runs[-1].append(k)
    return runs
