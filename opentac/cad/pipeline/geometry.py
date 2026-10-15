"""Polygon geometry utilities — wrappers around Shapely."""

import math

import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon
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


def make_connector_ear_polys(center, t_hat, n_out, width_mm):
    """Build clean connector side ears for the final manufacturing outline.

    The ears replace raw connector-footprint Edge.Cuts fragments. Dimensions are
    based on the existing FPC connector footprint ear style but kept simple so
    all connector sizes share one manufacturable outer contour.
    """
    center = np.asarray(center, dtype=float)
    t_hat = np.asarray(t_hat, dtype=float)
    n_out = np.asarray(n_out, dtype=float)
    half_w = float(width_mm) / 2
    ear_out = 0.55
    ear_h = 1.05
    chamfer = 0.28
    # Place ears near the terminal end, similar to the source FPC connector.
    y0 = -2.55
    y1 = y0 + ear_h

    def world(tx, ny):
        return center + tx * t_hat + ny * n_out

    left_inner = -half_w
    left_outer = left_inner - ear_out
    right_inner = half_w
    right_outer = right_inner + ear_out

    left = Polygon([
        world(left_inner, y0),
        world(left_inner, y1),
        world(left_outer + chamfer, y1),
        world(left_outer, y1 - chamfer),
        world(left_outer, y0 + chamfer),
        world(left_outer + chamfer, y0),
    ])
    right = Polygon([
        world(right_inner, y0),
        world(right_outer - chamfer, y0),
        world(right_outer, y0 + chamfer),
        world(right_outer, y1 - chamfer),
        world(right_outer - chamfer, y1),
        world(right_inner, y1),
    ])
    return [left, right]


def add_connector_ears_to_board(board_poly, center, t_hat, n_out, width_mm):
    """Union connector ears into a board polygon and return a cleaned polygon."""
    from shapely.ops import unary_union

    ears = make_connector_ear_polys(center, t_hat, n_out, width_mm)
    expanded = unary_union([board_poly, *ears])
    if not expanded.is_valid:
        expanded = expanded.buffer(0)
    return expanded


def trim_connector_front_space(board_poly, center, t_hat, n_out, width_mm, keep_mm=0.3):
    """Remove unused board area beyond the connector terminal line.

    The generated connector pads/routes end at the pad row. Keep a tiny amount
    of board beyond that line so endpoints remain inside the manufacturable
    outline, but trim the large empty strip from the tail end.
    """
    center = np.asarray(center, dtype=float)
    t_hat = np.asarray(t_hat, dtype=float)
    n_out = np.asarray(n_out, dtype=float)
    minx, miny, maxx, maxy = board_poly.bounds
    half_w = max(float(width_mm) / 2 + 1.0, float(np.hypot(maxx - minx, maxy - miny)))
    trim_start = center + float(keep_mm) * n_out
    trim_len = CONNECTOR_BODY_BELOW_MM + max(5.0, half_w)
    a = trim_start - half_w * t_hat
    b = trim_start + half_w * t_hat
    c = b + trim_len * n_out
    d = a + trim_len * n_out
    trimmed = board_poly.difference(Polygon([a, b, c, d]))
    if not trimmed.is_valid:
        trimmed = trimmed.buffer(0)
    return trimmed


def trim_connector_side_space(board_poly, center, t_hat, n_out, width_mm, keep_mm=0.3, side_keep_mm=0.3):
    """Tighten the connector-end side edges around the gold-finger zone."""
    center = np.asarray(center, dtype=float)
    t_hat = np.asarray(t_hat, dtype=float)
    n_out = np.asarray(n_out, dtype=float)
    minx, miny, maxx, maxy = board_poly.bounds
    far = float(np.hypot(maxx - minx, maxy - miny)) + 5.0
    half_terminal = float(width_mm) / 2 + float(side_keep_mm)
    # Keep the side trim at the insertion edge only; routes fan out just behind
    # this line and must remain inside the flex outline.
    y_front = float(keep_mm)
    y_back = 0.0
    left_inner0 = center - half_terminal * t_hat + y_back * n_out
    left_inner1 = center - half_terminal * t_hat + y_front * n_out
    left_outer1 = center - far * t_hat + y_front * n_out
    left_outer0 = center - far * t_hat + y_back * n_out
    right_inner0 = center + half_terminal * t_hat + y_back * n_out
    right_outer0 = center + far * t_hat + y_back * n_out
    right_outer1 = center + far * t_hat + y_front * n_out
    right_inner1 = center + half_terminal * t_hat + y_front * n_out
    cut = Polygon([left_inner0, left_inner1, left_outer1, left_outer0])
    cut = cut.union(Polygon([right_inner0, right_outer0, right_outer1, right_inner1]))
    trimmed = board_poly.difference(cut)
    if not trimmed.is_valid:
        trimmed = trimmed.buffer(0)
    return trimmed


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


def resolve_tail_width(p1, p2, conn_pos: int, params) -> float:
    """Resolve manual/automatic tail width."""
    return resolve_tail(params, p1, p2, conn_pos)["width_mm"]


def resolve_tail(params, p1, p2, conn_pos: int) -> dict:
    """Resolve tail dimensions.

    The connector table value is the automatic/default tip width. Manual user
    widths are respected, including values smaller than the connector tip.
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
    }


def tail_geometry(outline_poly: Polygon, p1, p2, conn_pos: int, params) -> dict:
    """Compute shared connector-tail geometry.

    The tail stays centered on the first outline edge. Edge cuts use
    cable_length_mm; layout/preview board union uses the total connector body
    envelope so routing can legally enter the connector area.
    """
    t_hat, n_out = edge_tangent_and_outward_normal(outline_poly, p1, p2)
    mid = 0.5 * (p1 + p2)

    resolved = resolve_tail(params, p1, p2, conn_pos)
    tail_width = resolved["width_mm"]
    connector_width = resolved["connector_min_width_mm"]
    cable_length = params.cable_length_mm
    total_length = cable_length + CONNECTOR_BODY_ABOVE_MM + CONNECTOR_BODY_BELOW_MM

    A = mid - (tail_width / 2) * t_hat
    B = mid + (tail_width / 2) * t_hat
    D = A + cable_length * n_out
    C = B + cable_length * n_out

    At = mid - (tail_width / 2) * t_hat
    Bt = mid + (tail_width / 2) * t_hat
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


def remove_collinear_closed(P: np.ndarray) -> np.ndarray:
    """Remove collinear points from a closed polygon ring without duplicating the first point."""
    if len(P) < 3:
        return P
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
    `s * (R @ v)` with no translation. See `apply_cable_transform_dir`.
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


def apply_cable_transform_dir(vecs: np.ndarray, tf) -> np.ndarray:
    """Carry DIRECTIONS through the same map — rotation only, no translation.

    Tangents must go through this, not `apply_cable_transform`: subtracting
    `mid` from a direction vector would silently turn it into a point.
    """
    R, _mid, flip = tf
    out = (R @ np.asarray(vecs, dtype=float).T).T
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
