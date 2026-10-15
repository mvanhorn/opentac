"""Outline construction and validation for OpenTactileCAD.

Everything here exists to keep a generated outline out of the failure modes
documented in NOTES.md ("The cable edge must be a real edge") and in
pipeline/geometry.py's winding handling. Build outlines with these functions
rather than emitting raw coordinate lists.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, box


# ---------------------------------------------------------------- winding

def signed_area(pts: np.ndarray) -> float:
    """Shoelace signed area. Positive = counter-clockwise."""
    x, y = np.asarray(pts, float).T
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def orient_ccw(pts: np.ndarray) -> np.ndarray:
    """Make the outline counter-clockwise *without* changing which edge is first.

    This matters more than it looks. pipeline/geometry.py::ensure_ccw reverses
    vertices 1..N while pinning vertex 0, so handing the pipeline a clockwise
    polygon does not merely flip its winding — it replaces edge 0->1 with the
    *other* edge meeting vertex 0. The board still generates; the cable just
    comes out somewhere nobody asked for, with no warning.

    Reversing the whole ring and re-rotating keeps the same geometric edge
    first, only traversed the other way round.
    """
    pts = np.asarray(pts, float)
    if signed_area(pts) >= 0:
        return pts
    return np.vstack([pts[1::-1], pts[:1:-1]])


# ------------------------------------------------------------- resampling

def resample_closed(pts: np.ndarray, n: int = 240, smooth: bool = True) -> np.ndarray:
    """Resample a closed outline to n evenly spaced points, optionally splined.

    Linear interpolation between hand-placed control points leaves visible
    faceting on anything organic. A periodic spline pass removes it without
    needing shape-specific control point tuning.
    """
    pts = np.asarray(pts, float)
    closed = np.vstack([pts, pts[:1]])

    if smooth:
        try:
            from scipy.interpolate import splprep, splev
            # s=0 interpolates the control points exactly; per=1 closes the curve.
            tck, _ = splprep([closed[:, 0], closed[:, 1]], s=0, per=1,
                             k=min(3, len(pts) - 1))
            u = np.linspace(0, 1, n, endpoint=False)
            curve = np.column_stack(splev(u, tck))
            # A spline through sparse vertices overshoots between them. That is
            # what you want on control points sketching an organic shape, and
            # destructive on geometry that already has real straight edges: an
            # 8-vertex face unrolled from an STL went from 28x79mm to 51x83mm.
            # If the curve leaves the source outline's envelope, it is
            # inventing shape, so keep the corners instead.
            grew = np.ptp(curve, axis=0) / np.maximum(np.ptp(pts, axis=0), 1e-9)
            if np.all(grew < 1.02):
                return curve
        except Exception:
            pass  # fall through to linear

    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    u = np.linspace(0, t[-1], n, endpoint=False)
    return np.column_stack([np.interp(u, t, closed[:, 0]),
                            np.interp(u, t, closed[:, 1])])


# ------------------------------------------------------------- cable edge

_SIDES = {"bottom": 0, "top": 1, "left": 2, "right": 3}


def with_flat_cable_edge(pts: np.ndarray, side: str = "bottom",
                         width_mm: float = 20.0,
                         max_trim_frac: float = 0.35) -> np.ndarray:
    """Give an outline an explicit flat first edge on `side`, for the connector.

    Edge 0->1 is the cable edge, and rotate_for_cable() spins the whole board
    to lay that edge horizontal. Sample a curve naively and edge 0->1 is a
    short chord at whatever angle the sampling happened to land on — the board
    rotates by that angle and the axis-aligned taxel grid skews across the
    shape. Measured on a 270mm insole: a 20.4 degree accidental rotation cost
    40% of the taxels, grew the board from 100mm to 147mm wide, and produced
    61 DRC violations.

    So the flat edge is cut, not hoped for: slice the shape with a straight
    line far enough in that the cut is `width_mm` wide, keep the larger piece,
    and rotate the vertex array so that cut is edge 0->1.
    """
    if side not in _SIDES:
        raise ValueError(f"side must be one of {sorted(_SIDES)}, got {side!r}")

    pts = orient_ccw(np.asarray(pts, float))
    # Work in a frame where the requested side is "bottom", then rotate back.
    theta = -0.5 * np.pi * _SIDES[side]
    work = _rotate(pts, theta)

    poly = Polygon(work)
    if not poly.is_valid:
        poly = poly.buffer(0)
    minx, miny, maxx, maxy = poly.bounds
    span = maxy - miny

    cut_y = None
    for frac in np.linspace(0.002, max_trim_frac, 400):
        y = miny + frac * span
        piece = poly.intersection(box(minx - 10, y, maxx + 10, maxy + 10))
        if piece.is_empty or piece.geom_type != "Polygon":
            continue
        w = _chord_width(piece, y)
        if w >= width_mm:
            cut_y, cut_poly = y, piece
            break

    if cut_y is None:
        raise ValueError(
            f"Cannot cut a {width_mm:.1f}mm flat edge on the {side} of this "
            f"outline within {max_trim_frac:.0%} of its {span:.1f}mm extent. "
            "The shape is too narrow there for the connector — put the cable "
            "on a wider side, or use fewer rows/columns so a smaller "
            "connector fits.")

    ring = np.array(cut_poly.exterior.coords[:-1], float)
    ring = _clean_ring(ring, cut_y)
    ring = orient_ccw(ring)
    ring = _rotate_to_flat_edge(ring, cut_y)
    return _rotate(ring, -theta)


def _clean_ring(ring: np.ndarray, cut_y: float, min_step_mm: float = 0.15,
                sliver_mm: float = 0.15) -> np.ndarray:
    """Drop the slivers a straight clip leaves behind.

    Where the resampled curve crosses the cut line within a fraction of a
    millimetre of the clip corner, shapely emits a vertex a hair above the
    cut. That leaves an almost-zero-length edge adjacent to the cable edge:
    geometrically valid, but it makes edge 0->1 ambiguous and gives the
    router a degenerate corner to work around.
    """
    on_cut = np.abs(ring[:, 1] - cut_y) <= 1e-9
    near_cut = np.abs(ring[:, 1] - cut_y) <= sliver_mm
    keep = [i for i in range(len(ring)) if on_cut[i] or not near_cut[i]]
    ring = ring[keep]

    out = [ring[0]]
    for pt in ring[1:]:
        if np.hypot(*(pt - out[-1])) >= min_step_mm or abs(pt[1] - cut_y) <= 1e-9:
            out.append(pt)
    if np.hypot(*(out[0] - out[-1])) < min_step_mm and len(out) > 3:
        out.pop()
    return np.array(out)


def _rotate(pts: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return pts @ np.array([[c, s], [-s, c]])


def _chord_width(poly: Polygon, y: float, tol: float = 1e-7) -> float:
    xs = [x for x, yy in poly.exterior.coords if abs(yy - y) <= 1e-6]
    return (max(xs) - min(xs)) if len(xs) >= 2 else 0.0


def _rotate_to_flat_edge(ring: np.ndarray, y: float, tol: float = 1e-6) -> np.ndarray:
    """Roll the vertex array so the flat edge at height `y` becomes edge 0->1."""
    on = [i for i, (_, yy) in enumerate(ring) if abs(yy - y) <= tol]
    n = len(ring)
    for i in on:
        if (i + 1) % n in on:
            return np.roll(ring, -i, axis=0)
    raise ValueError("Flat cable edge vanished during clipping.")


# -------------------------------------------------------------- validation

def validate(pts: np.ndarray, rows: int | None = None, cols: int | None = None,
             pad_pitch_mm: float = 0.5, flat_tol_mm: float = 0.05) -> list[str]:
    """Return a list of problems with this outline. Empty list means good.

    Runs before generating, because every check here costs microseconds and
    the generate it guards costs 7-40 seconds.
    """
    problems: list[str] = []
    pts = np.asarray(pts, float)

    if pts.ndim != 2 or pts.shape[1] != 2:
        return ["Outline must be an Nx2 array of millimetre coordinates."]
    if len(pts) < 3:
        return ["Outline needs at least 3 vertices."]
    if np.allclose(pts[0], pts[-1]):
        problems.append("Outline must not repeat its first point at the end; "
                        "the polygon is implicitly closed.")

    poly = Polygon(pts)
    if not poly.is_valid:
        problems.append("Outline is self-intersecting or otherwise invalid: "
                        f"{poly.buffer(0).is_valid and 'self-intersection' or 'degenerate'}.")
    if signed_area(pts) < 0:
        problems.append("Outline is clockwise. Pass it through orient_ccw() "
                        "first — ensure_ccw() in the pipeline would silently "
                        "swap your cable edge for the other edge at vertex 0.")

    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)
    w, h = maxx - minx, maxy - miny
    if min(w, h) < 5:
        problems.append(f"Outline is only {w:.1f} x {h:.1f}mm — too small to "
                        "route. Are the units millimetres?")
    if max(w, h) > 500:
        problems.append(f"Outline is {w:.1f} x {h:.1f}mm — larger than any "
                        "standard fab panel. Are the units millimetres?")

    # Edge 0->1 is the cable edge. It has to be straight, and wide enough for
    # the connector the requested grid implies.
    edge = pts[1] - pts[0]
    edge_len = float(np.hypot(*edge))
    n_hat = np.array([-edge[1], edge[0]]) / (edge_len or 1.0)
    if edge_len < 1e-9:
        problems.append("Edge 0->1 (the cable edge) is zero length.")
    else:
        # Neighbouring vertices must not curve away from the cable edge line.
        for j in (2, len(pts) - 1):
            off = abs(float(np.dot(pts[j % len(pts)] - pts[0], n_hat)))
            if off < flat_tol_mm:
                problems.append(
                    "Edge 0->1 is collinear with its neighbour — the cable "
                    "edge is not a distinct edge.")
                break

    if rows is not None and cols is not None:
        need = min_cable_edge_mm(rows, cols, pad_pitch_mm)
        if need is None:
            problems.append(
                f"{rows} rows x {cols} columns needs {rows + cols} channels, "
                "more than any connector in the family carries (50 per side "
                "is the ceiling). Reduce the grid.")
        elif edge_len + 1e-6 < need:
            problems.append(
                f"Cable edge is {edge_len:.1f}mm but a {rows}x{cols} grid "
                f"needs a {need:.1f}mm connector. Widen the flat edge, move "
                "it to a wider side, or use a smaller grid.")

    return problems


def min_cable_edge_mm(rows: int, cols: int, pad_pitch_mm: float = 0.5) -> float | None:
    """Width the flat cable edge must have for a rows x cols grid.

    Derived from the pipeline's own connector tables rather than tabulated
    here, so it cannot drift from what select_connector_for_nets() will pick.
    Returns None if no connector in the family fits.
    """
    from opentac.cad.pipeline.layout import select_connector_for_nets
    from opentac.cad.pipeline.geometry import connector_dimensions
    try:
        conn_pos, _swap = select_connector_for_nets(rows, cols)
    except ValueError:
        for override in (80, 90, 100):
            if _fits(override, rows, cols):
                return connector_dimensions(override, pad_pitch_mm)["ear_stiffener_width_mm"]
        return None
    return connector_dimensions(conn_pos, pad_pitch_mm)["ear_stiffener_width_mm"]


def _fits(conn_pos: int, rows: int, cols: int) -> bool:
    from opentac.cad.pipeline.layout import _best_fit_swap
    return bool(_best_fit_swap(conn_pos, rows, cols)[0])
