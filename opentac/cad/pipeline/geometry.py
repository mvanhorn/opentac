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


# The flare that widens a narrow tail back to the connector's tab: steepest
# (45 degrees) at its middle, tangent to the cable and to the landing at its ends.
TAIL_FLARE_MAX_SLOPE = 1.0
TAIL_FLARE_STEPS = 16


def tail_flare_edges(narrow_w: float, wide_w: float, room: float,
                     offset: float = 0.0) -> list[tuple[float, float, float]]:
    """(distance before the landing, low edge, high edge) samples of the flare,
    from the narrow cable to the full-width landing, the edges along t_hat from
    the connector's centre line.

    The cable is `narrow_w` wide centred `offset` off that line, the landing
    `wide_w` wide centred on it, so each side flares by its own amount; both
    take the same length, set by the side that has further to go. Each side is
    a half cosine: it leaves the cable and meets the landing without a corner
    at either end, so there is no sharp inside corner for the flex to tear at.
    Kept within `room` (the straight cable it replaces), steeper when the cable
    is short.
    """
    lo, hi = offset - narrow_w / 2.0, offset + narrow_w / 2.0
    d_lo = max(0.0, lo + wide_w / 2.0)
    d_hi = max(0.0, wide_w / 2.0 - hi)
    length = max(0.1, min(math.pi / 2.0 * max(d_lo, d_hi) / TAIL_FLARE_MAX_SLOPE, room))
    n = TAIL_FLARE_STEPS
    out = []
    for i in range(n + 1):
        f = (1.0 - math.cos(math.pi * i / n)) / 2.0
        out.append((length * (1.0 - i / n), lo - d_lo * f, hi + d_hi * f))
    return out


def tail_flare_length(narrow_w: float, wide_w: float, room: float, offset: float = 0.0) -> float:
    """How far before the landing the flare (tail_flare_edges) starts."""
    return tail_flare_edges(narrow_w, wide_w, room, offset)[0][0]


def make_tail_poly(mid, t_hat, n_out, tail_width: float, connector_width: float,
                   cable_length: float, total_length: float, offset: float = 0.0) -> Polygon:
    """The tail from the board edge to the connector's far end.

    A tail as wide as the connector is a plain rectangle. A narrower one runs at
    its own width for the cable, centred `offset` off the connector's centre
    line (fit_tail_strip), then flares (tail_flare_edges) to the connector's tab
    width so the connector lands on a full-width tab, centred on the connector,
    which the connector footprint's own Edge.Cuts continue from (export
    stitched_outline_segments). The landing starts at cable_length, ahead of the
    footprint's open edge.
    """
    mid, t_hat, n_out = (np.asarray(v, dtype=float) for v in (mid, t_hat, n_out))
    w, W = float(tail_width), float(connector_width)
    if w >= W - 1e-6:
        return make_cable_rect(mid, mid, t_hat, n_out, w, total_length)
    prof = tail_flare_edges(w, W, room=max(0.5, cable_length - 1.0), offset=offset)
    left = [mid + (offset - w / 2) * t_hat]
    left += [mid + lo * t_hat + (cable_length - d) * n_out for d, lo, _hi in prof]
    left.append(mid - (W / 2) * t_hat + total_length * n_out)
    right = [mid + (W / 2) * t_hat + total_length * n_out]
    right += [mid + hi * t_hat + (cable_length - d) * n_out for d, _lo, hi in reversed(prof)]
    right.append(mid + (offset + w / 2) * t_hat)
    return Polygon(left + right)


def tail_without_landing(cable_poly, mid, t_hat, n_out, width: float, connector_width: float,
                         cable_length: float, total_length: float, offset: float = 0.0) -> Polygon:
    """A narrow tail once its connector has moved to the end of a drawn tail
    extension: the flare and landing (make_tail_poly) belong to the connector,
    so the plain strip runs on in their place to where the extension starts."""
    if cable_poly is None or width >= connector_width - 1e-6:
        return cable_poly
    mid, t_hat, n_out = (np.asarray(v, dtype=float) for v in (mid, t_hat, n_out))
    start = cable_length - tail_flare_length(width, connector_width, max(0.5, cable_length - 1.0), offset)
    big = 1e4
    before = Polygon([mid - big * t_hat - big * n_out, mid + big * t_hat - big * n_out,
                      mid + big * t_hat + start * n_out, mid - big * t_hat + start * n_out])
    strip_mid = mid + offset * t_hat
    strip = make_cable_rect(strip_mid, strip_mid, t_hat, n_out, width, total_length)
    return unary_union([cable_poly.intersection(before), strip]).buffer(0)


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


def extension_tail_poly(path, width: float, clearance: float, connector_width,
                        offset: float = 0.0) -> Polygon:
    """A drawn tail extension's board: its corridor, ending in a connector landing
    when the corridor is narrower than the connector (connector_landing_poly).

    The corridor runs `offset` to the left of the drawn path (offset_path), so
    it stays over the wire lanes when the tail strip it continues sits off the
    connector's centre (fit_tail_strip); the landing is centred on the path's
    end, where the connector goes."""
    path = np.asarray(path, dtype=float)
    centre = offset_path(path, offset) if offset and len(path) >= 2 else path
    if centre is None:
        centre = path
    corridor = LineString(centre).buffer(width / 2 + clearance, cap_style=2, join_style=2)
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
                                     offset=offset)
    return unary_union([corridor, landing])


def connector_landing_poly(pad_row_center, t_hat, n_out, narrow_w: float, connector_width: float,
                           room: float, offset: float = 0.0) -> Polygon:
    """The flare and full-width landing alone, for a connector at the end of a
    drawn tail extension whose corridor is narrower than the connector: the same
    shape make_tail_poly ends in, placed around the moved pad row, flaring from a
    corridor centred `offset` off it."""
    c, t_hat, n_out = (np.asarray(v, dtype=float) for v in (pad_row_center, t_hat, n_out))
    prof = tail_flare_edges(narrow_w, connector_width, room=max(0.5, room), offset=offset)
    W = float(connector_width)
    land = -CONNECTOR_BODY_ABOVE_MM
    left = [c + lo * t_hat + (land - d) * n_out for d, lo, _hi in prof]
    left.append(c - (W / 2) * t_hat + CONNECTOR_BODY_BELOW_MM * n_out)
    right = [c + (W / 2) * t_hat + CONNECTOR_BODY_BELOW_MM * n_out]
    right += [c + hi * t_hat + (land - d) * n_out for d, _lo, hi in reversed(prof)]
    return Polygon(left + right)


def resolve_tail_width(p1, p2, conn_pos: int, params) -> float:
    """Resolve manual/automatic tail width."""
    return resolve_tail(params, p1, p2, conn_pos)["width_mm"]


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
