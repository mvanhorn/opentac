"""Design rule checks for routed traces."""

from .types import Route, DrcViolation
from .geometry import seg_seg_dist
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, nearest_points, unary_union


def drc_check(
    col_routes: list[Route],
    row_routes: list[Route],
    trace_w_mm: float,
    clearance_mm: float,
    outline_xy: np.ndarray | None = None,
    board_edge_clear_mm: float = 0.0,
    board_poly=None,
    taxels=None,
) -> list[DrcViolation]:
    """Check same-layer route clearance and optional board-edge clearance.

    Each copper layer is checked on its own: a row route with .layer set (the
    inner layers of a 4-layer board) is never compared with B.Cu copper it
    merely overlaps in plan view.

    taxels (optional, a TaxelCopper) also checks the F.Cu wires against the
    taxels' own copper — see _check_taxels."""
    G = trace_w_mm + clearance_mm
    violations = []

    if col_routes:
        violations.extend(_check_layer(col_routes, G, "F.Cu"))
        if taxels is not None:
            violations.extend(_check_taxels(col_routes, taxels, trace_w_mm / 2 + clearance_mm))
    for layer, routes in _by_layer(row_routes, "B.Cu"):
        violations.extend(_check_layer(routes, G, layer))
    if board_edge_clear_mm > 0 and (outline_xy is not None or board_poly is not None):
        edge_min = trace_w_mm / 2 + board_edge_clear_mm
        violations.extend(_check_board_edge(col_routes, outline_xy, edge_min, "F.Cu", board_poly))
        for layer, routes in _by_layer(row_routes, "B.Cu"):
            violations.extend(_check_board_edge(routes, outline_xy, edge_min, layer, board_poly))

    return violations


class TaxelCopper:
    """The taxels' F.Cu copper for _check_taxels: each active taxel's square
    inset by edge_clear_mm (the comb's outer frame), with its column net."""

    def __init__(self, placements_xy, col_net_per_pixel, removed, pixel_w_mm, pixel_h_mm, edge_clear_mm):
        from shapely.geometry import box
        hw, hh = pixel_w_mm / 2 - edge_clear_mm, pixel_h_mm / 2 - edge_clear_mm
        self.items = [(col_net_per_pixel[i], box(x - hw, y - hh, x + hw, y + hh))
                      for i, (x, y) in enumerate(np.asarray(placements_xy, dtype=float))
                      if i not in removed]

    @classmethod
    def of(cls, layout, removed, params):
        return cls(layout.placements_xy, layout.col_net_per_pixel, set(removed or ()),
                   params.pixel_w_mm, params.pixel_h_mm, params.edge_clear_mm)


def _check_taxels(col_routes: list[Route], taxels: TaxelCopper, need: float) -> list[DrcViolation]:
    """F.Cu wires against the taxels' combs: a column wire passing a taxel
    must clear its copper like any other copper (on F.Cu both electrodes are
    combs, so grazing one shorts a column to a row). A wire's own column's
    taxels are skipped — it lands on them by design. Measured: route
    smoothing let a column cut a taxel's corner at 0.025-0.05 mm that the
    wire-to-wire check never saw; only KiCad did."""
    from shapely.strtree import STRtree
    if not taxels.items or not col_routes:
        return []
    boxes = [b for _, b in taxels.items]
    tree = STRtree(boxes)
    out = []
    for r in col_routes:
        P = np.asarray(r.polyline_xy, dtype=float)
        for s in range(len(P) - 1):
            seg = LineString([P[s], P[s + 1]])
            if seg.length < 1e-9:
                continue
            for k in tree.query(seg.buffer(need)):
                net, b = taxels.items[int(k)]
                if net == r.net_name:
                    continue
                d = seg.distance(b)
                if d < need - 1e-6:
                    out.append(DrcViolation(
                        layer="F.Cu", net1=r.net_name, net2=f"TAXEL:{net}", type="taxel_clearance",
                        seg1=np.array([P[s], P[s + 1]]), seg2=np.array(b.exterior.coords[:2]),
                        distance=float(d), min_required=float(need)))
    return out


def _by_layer(routes: list[Route], default: str) -> list[tuple[str, list[Route]]]:
    """routes grouped by copper layer (Route.layer, or `default`), in first-seen order"""
    groups: dict[str, list[Route]] = {}
    for r in routes or ():
        groups.setdefault(getattr(r, "layer", None) or default, []).append(r)
    return list(groups.items())


def _check_layer(routes: list[Route], G: float, layer_name: str) -> list[DrcViolation]:
    """Check all route pairs on one layer.

    Exact-result bounding-box prefilters at the track and segment level: a
    pair whose boxes are separated by >= G cannot come within G, so skipping
    it changes nothing. Without them this loop is O(tracks^2 * segments^2)
    in pure Python and dominated fixed-mode generates (profiled: 25s of a
    48s request went to the attempt-scoring calls into this function).
    """
    violations = []
    N = len(routes)

    prepared = []
    for r in routes:
        P = r.polyline_xy
        if P is None or len(P) < 2:
            prepared.append(None)
            continue
        P = np.asarray(P, dtype=float)
        lo = np.minimum(P[:-1], P[1:])
        hi = np.maximum(P[:-1], P[1:])
        prepared.append((P, lo, hi, P.min(axis=0), P.max(axis=0)))

    for i in range(N):
        if prepared[i] is None:
            continue
        Pi, ilo, ihi, imin, imax = prepared[i]
        for j in range(i + 1, N):
            if routes[j].net_name == routes[i].net_name:
                continue   # same net may touch (hug router emits several tracks per net)
            if prepared[j] is None:
                continue
            Pj, jlo, jhi, jmin, jmax = prepared[j]
            if (imin[0] > jmax[0] + G or jmin[0] > imax[0] + G
                    or imin[1] > jmax[1] + G or jmin[1] > imax[1] + G):
                continue
            for si in range(len(Pi) - 1):
                cand = np.flatnonzero(
                    (jlo[:, 0] <= ihi[si, 0] + G) & (jhi[:, 0] >= ilo[si, 0] - G)
                    & (jlo[:, 1] <= ihi[si, 1] + G) & (jhi[:, 1] >= ilo[si, 1] - G))
                for sj in cand:
                    d = seg_seg_dist(Pi[si], Pi[si + 1], Pj[sj], Pj[sj + 1])
                    if d < G - 1e-9:
                        typ = "crossing" if d < 1e-9 else "clearance"
                        violations.append(DrcViolation(
                            layer=layer_name,
                            net1=routes[i].net_name,
                            net2=routes[j].net_name,
                            type=typ,
                            seg1=np.array([Pi[si], Pi[si + 1]]),
                            seg2=np.array([Pj[sj], Pj[sj + 1]]),
                            distance=d,
                            min_required=G,
                        ))

    return violations


def _check_board_edge(
    routes: list[Route],
    outline_xy: np.ndarray,
    min_required: float,
    layer_name: str,
    board_poly=None,
) -> list[DrcViolation]:
    """Check trace centerlines against the final board boundary.

    Every stretch of a segment that lies outside the board, or inside it but
    closer than min_required to the edge, is its own violation, reported at
    that stretch's worst point. Distance is the centerline's distance to the
    edge, negative when the centerline is outside the board.

    This used to sample points along each segment, skip every sample outside
    the board, and keep one result per segment. Copper that left the board was
    then never measured, and a harmless near-miss at the connector end of a
    long segment stood in for the real fault further along it — which
    _drop_connector_zone_edge_flags then discarded. A column wire running
    straight through the eight preset's waist, 0.25mm outside the outline,
    reported DRC 0 that way.
    """
    violations = []
    outline_poly = board_poly if board_poly is not None else Polygon(outline_xy)
    if outline_poly.is_empty:
        return violations
    boundary = outline_poly.boundary
    band = boundary.buffer(min_required)

    for route in routes:
        P = route.polyline_xy
        if P is None or len(P) < 2:
            continue
        P = np.asarray(P, dtype=float)
        for si in range(len(P) - 1):
            seg = LineString([P[si], P[si + 1]])
            if seg.length <= 0:
                continue
            bad = seg.intersection(band).union(seg.difference(outline_poly))
            if not bad.is_empty and bad.geom_type != "LineString":
                bad = unary_union(_line_pieces(bad))
                if bad.geom_type == "MultiLineString":
                    bad = linemerge(bad)   # one fault, one flag
            for piece in _line_pieces(bad):
                worst_pt, dist = _worst_point(piece, outline_poly, boundary)
                if dist >= min_required - 1e-9:
                    continue
                violations.append(DrcViolation(
                    layer=layer_name,
                    net1=route.net_name,
                    net2="BOARD_EDGE",
                    type="edge_clearance",
                    seg1=np.array([P[si], P[si + 1]]),
                    seg2=np.array([worst_pt, worst_pt]),
                    distance=dist,
                    min_required=min_required,
                ))

    return violations


def _line_pieces(geom):
    """The non-degenerate LineStrings in a (multi)line intersection result."""
    if geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom] if geom.length > 1e-9 else []
    return [g for g in getattr(geom, "geoms", []) for g in _line_pieces(g)]


def _worst_point(piece, outline_poly, boundary, step=0.05):
    """(point, signed distance) where `piece` is worst: deepest outside the
    board if any of it is outside, else closest to the edge."""
    outside = _line_pieces(piece.difference(outline_poly))
    if outside:
        best = None
        for part in outside:
            n = max(2, int(np.ceil(part.length / step)) + 1)
            for t in np.linspace(0.0, 1.0, n):
                pt = part.interpolate(t, normalized=True)
                d = boundary.distance(pt)
                if best is None or d > best[1]:
                    best = (np.array([pt.x, pt.y]), d)
        return best[0], -float(best[1])
    p, _ = nearest_points(piece, boundary)
    return np.array([p.x, p.y]), float(piece.distance(boundary))
