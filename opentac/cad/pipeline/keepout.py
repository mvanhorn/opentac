"""Variable-width wiring keep-out that follows the sensor outline.

Fixed-outline mode reserves room for wiring *inside* the drawn outline. That
reserved band is deliberately not a constant-width ring:

  * Wires accumulate as they run down toward the connector tail. A row net high
    up in the array occupies its side corridor for the whole way down, so the
    corridor must be widest near the tail and can hug the real edge near the
    top, where only a couple of wires pass.
  * Row nets split top-half-left / bottom-half-right (see _row_route_sides in
    route_rows.py), so the two sides carry different counts at the same height
    and the band is genuinely asymmetric.
  * The outline is not necessarily a rectangle, so the band has to follow the
    actual edge rather than an inset of the bounding box.

The band is therefore stored as a width per boundary sample point
(KeepoutProfile) and realized as a variable-distance erosion: every sample
contributes a disc of its own radius, and the sensor region is whatever is left
of the outline after subtracting their union. Widths only ever grow (callers
take an element-wise max), which is what makes the surrounding fixed-point loop
terminate.
"""

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .geometry import make_polygon

# Boundary sampling step. The erosion envelope is the union of discs centered on
# these samples, so the inner edge scallops by w - sqrt(w^2 - (spacing/2)^2);
# at 1 mm spacing and any realistic width that is well under 10 um.
BOUNDARY_SPACING_MM = 1.0

# Widths are smoothed with a running max over this window so the band never has
# a narrow notch a wire cannot actually turn into.
SMOOTH_WINDOW_MM = 3.0


@dataclass
class KeepoutProfile:
    """Inward keep-out width at each sampled point of the outline boundary."""

    samples: np.ndarray  # Nx2 points on the outline exterior
    widths: np.ndarray   # N widths in mm, measured inward from the boundary

    def copy(self) -> "KeepoutProfile":
        return KeepoutProfile(self.samples.copy(), self.widths.copy())

    def summary(self) -> dict:
        return {
            "min_mm": float(np.min(self.widths)) if len(self.widths) else 0.0,
            "max_mm": float(np.max(self.widths)) if len(self.widths) else 0.0,
            "mean_mm": float(np.mean(self.widths)) if len(self.widths) else 0.0,
        }


def sample_boundary(outline_poly: Polygon, spacing: float = BOUNDARY_SPACING_MM) -> np.ndarray:
    """Evenly spaced points along the outline exterior."""
    ring = outline_poly.exterior
    n = max(8, int(np.ceil(ring.length / max(spacing, 1e-3))))
    ds = ring.length / n
    return np.array([ring.interpolate(i * ds).coords[0] for i in range(n)], dtype=float)


def make_profile(outline_xy: np.ndarray, base_width: float) -> KeepoutProfile:
    """A uniform keep-out of base_width — the starting point before any taper."""
    samples = sample_boundary(make_polygon(outline_xy))
    return KeepoutProfile(samples, np.full(len(samples), max(0.0, float(base_width))))


def region(outline_poly: Polygon, profile: KeepoutProfile):
    """Erode the outline by the profile's variable width.

    Returns the sensor region (possibly a MultiPolygon, possibly empty).
    """
    discs = [
        Point(s).buffer(float(w), quad_segs=8)
        for s, w in zip(profile.samples, profile.widths)
        if w > 1e-9
    ]
    if not discs:
        return outline_poly
    return outline_poly.difference(unary_union(discs))


def smooth_widths(profile: KeepoutProfile, window_mm: float = SMOOTH_WINDOW_MM) -> None:
    """Running max along the boundary, in place.

    Keeps the band's inner edge navigable: a wire needs room on the approach to
    a wide spot, not only at the spot itself.
    """
    n = len(profile.widths)
    if n == 0:
        return
    half = max(1, int(round(window_mm / BOUNDARY_SPACING_MM / 2)))
    if half * 2 + 1 >= n:
        profile.widths[:] = profile.widths.max()
        return
    # The boundary is a closed loop, so wrap rather than clamp at the ends.
    padded = np.concatenate([profile.widths[-half:], profile.widths, profile.widths[:half]])
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    profile.widths[:] = windows.max(axis=1)


def fixed_keepout_floor(params) -> float:
    """Minimum wiring keep-out width in fixed-outline mode.

    Deliberately not routing_margin_mm. That parameter is how far grow-board
    mode pushes the outline *outward*; used as the floor of a band cut
    *inward*, its 3.0 mm default gives a flat ring that can exceed the
    modelled demand everywhere, overriding the shape-following taper that
    keepout.analytic_widths computes. The top edge, where no wires run, would
    then reserve as much as the tail.

    None (the default) means "let the model decide": analytic_widths already
    carries its own safety term, which is the real floor. An explicit value is
    honoured so a board can still be forced conservative.
    """
    explicit = getattr(params, "wiring_keepout_floor_mm", None)
    if explicit is not None:
        return max(0.0, float(explicit))
    return 0.0

_TOO_SMALL_MSG = (
    "Outline too small to fit both sensors and wiring at this pitch. "
    "Enlarge the outline, coarsen the pitch, or switch to grow-board mode."
)


def analytic_widths(profile: KeepoutProfile, params, layout) -> np.ndarray:
    """Model the wiring demand at each boundary sample from an existing layout.

    Row nets run out to one side and then down that side to the pad row, so the
    demand at a sample is set by how many of that side's rows sit at or above
    it — few near the top, all of them near the tail. Samples along the bottom
    edge are below every row, so they already inherit that side's full count;
    an extra tail fan-in term for the row nets would double-count it. What the
    fan-in actually needs beyond this is discovered, not guessed: escaping copper
    grows the profile locally through keepout.grow_near().

    Everything is computed in the canonical frame rotate_for_cable() produces:
    the tail edge is horizontal at the bottom and the shape sits above it.
    """
    P = np.asarray(layout.placements_xy, dtype=float)
    if len(P) == 0:
        return profile.widths.copy()

    base = fixed_keepout_floor(params)
    trace_pitch = params.trace_w_mm + params.clearance_mm
    safety = max(1.0, 2 * trace_pitch)

    row_nets = [n for n in set(layout.row_net_per_pixel) if n]
    col_nets = [n for n in set(layout.col_net_per_pixel) if n]
    n_row, n_col = len(row_nets), len(col_nets)

    # Mean height of each row net, then the same top-half-left / bottom-half-right
    # split the row router applies.
    row_y = []
    for net in row_nets:
        idx = [i for i, n in enumerate(layout.row_net_per_pixel) if n == net]
        row_y.append(float(np.mean(P[idx, 1])))
    order = np.argsort(-np.asarray(row_y)) if row_y else np.array([], dtype=int)
    ordered_y = np.asarray([row_y[i] for i in order]) if len(order) else np.zeros(0)
    top_count = (n_row + 1) // 2
    left_y = np.sort(ordered_y[:top_count])      # rows that exit left
    right_y = np.sort(ordered_y[top_count:])     # rows that exit right

    x_mid = 0.5 * (P[:, 0].min() + P[:, 0].max())

    sx = profile.samples[:, 0]
    sy = profile.samples[:, 1]

    # Number of same-side rows at or above each sample height.
    left_count = len(left_y) - np.searchsorted(left_y, sy, side="left")
    right_count = len(right_y) - np.searchsorted(right_y, sy, side="left")
    side_count = np.where(sx <= x_mid, left_count, right_count)
    row_term = side_count * trace_pitch + safety

    # Column fan-in. Every column net converges on the pad row, so the tail end
    # of the board carries all of them at once — the same demand grow mode
    # reserves as _routing_demand's `n_col * trace_pitch * 0.35` bottom padding
    # and `* 0.25` side padding (pipeline/layout.py). The default floor is zero
    # (see fixed_keepout_floor), so without this term column traces escape the
    # board edge on shapes whose edge recedes toward the tail (a disc, a
    # trapezoid, a narrow wide board).
    #
    # Weighted by how close the sample is to the pad row: full demand at the
    # array's bottom edge and below, tapering to nothing at the top, since a
    # column has not fanned in yet up there.
    y_lo, y_hi = float(P[:, 1].min()), float(P[:, 1].max())
    span = max(y_hi - y_lo, 1e-6)
    tail_frac = np.clip(1.0 - (sy - y_lo) / span, 0.0, 1.0)
    col_term = n_col * trace_pitch * 0.35 * tail_frac

    return np.maximum(base, np.maximum(row_term, col_term))


def grow_near(profile: KeepoutProfile, points, amounts, window_mm: float = 6.0) -> bool:
    """Widen the keep-out around specific trouble spots. Returns True if changed.

    Used when routing actually fails: the boundary near a wire that escaped the
    board (or near a pixel the router could not wire) is exactly where more
    room is needed, and nowhere else has to pay for it.
    """
    if len(profile.samples) == 0:
        return False
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.size == 0:
        return False
    amts = np.atleast_1d(np.asarray(amounts, dtype=float))
    if len(amts) == 1:
        amts = np.repeat(amts, len(pts))

    before = profile.widths.copy()
    for pt, amt in zip(pts, amts):
        if amt <= 0:
            continue
        d = np.hypot(profile.samples[:, 0] - pt[0], profile.samples[:, 1] - pt[1])
        near = d <= window_mm
        if not np.any(near):
            near = d == d.min()
        profile.widths[near] = np.maximum(profile.widths[near], profile.widths[near] + amt)
    return bool(np.any(profile.widths > before + 1e-12))
