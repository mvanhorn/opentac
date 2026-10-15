"""Foam contact model: a rigid indenter pressed into a flat skin with no bumps.

The skin is velostat under a foam layer. Pressure follows a Winkler
foundation,

    p(x, y) = k * max(delta - z(x, y), 0)

where z is the height of the indenter's surface above the skin and delta is
the indentation depth. The pressure field is then blurred by a Gaussian of
width sigma (the foam spreads load sideways) and integrated over each taxel.
delta is solved so the pressure integrates to the requested force, so the
contact grows with load.

Two global parameters, fitted once in the OpenTac paper on one skin
(robotiq_long, 96 taxels at 4 mm) and held fixed on every other skin:

    sigma = 3.00 mm,  k = 0.00866 N/mm^3

Zero-shot on three other skins, this model scored a median shape F1 of 0.73
against a repeat-press ceiling of 0.80 (docs/sim.md).

The returned forces are at physical scale, not renormalised to the requested
total: load that falls outside the taxels (an indenter overhanging the edge,
foam spreading past the outline) is not read, so the model predicts what the
skin actually sees.

Ported from the lab script `opentac_indent_compare.py` (indenter_height,
gaussian_blur, sim_taxel_forces); the numbers are unchanged, which
tests/sim/test_foam.py checks against the original.
"""
from __future__ import annotations

import numpy as np

GRID_MM = 0.5                         # sampling of the pressure field
SIGMA_MM = 3.0                        # fitted on robotiq_long (paper)
K_N_PER_MM3 = 0.008659643233600654    # fitted on robotiq_long (paper)

#: The indenters of the paper's experiments, by name.
INDENTERS = {"sphere_R25": {"kind": "sphere", "R": 25.0},
             "sphere_R10": {"kind": "sphere", "R": 10.0},
             "alignment_tip_R3": {"kind": "sphere", "R": 3.0},
             "punch_D10": {"kind": "punch", "R": 5.0},
             "edge_R5_L40": {"kind": "bar", "R": 5.0, "L": 40.0}}


def indenter_height(geom, dx, dy, angle_deg=0.0):
    """Height of the indenter surface above the flat skin, at offsets dx, dy (mm).

    `geom` is {"kind": "sphere", "R": r}, {"kind": "punch", "R": r} (a flat
    disc of radius r) or {"kind": "bar", "R": r, "L": length} (a cylinder of
    radius r lying on the skin, its axis at `angle_deg` from +x). Infinite
    where the indenter is not above the skin at all.
    """
    kind = geom["kind"]
    if kind == "sphere":
        R = geom["R"]
        r2 = dx ** 2 + dy ** 2
        z = np.where(r2 < R ** 2, R - np.sqrt(np.clip(R ** 2 - r2, 0, None)), np.inf)
        return z
    if kind == "punch":
        r2 = dx ** 2 + dy ** 2
        return np.where(r2 < geom["R"] ** 2, 0.0, np.inf)
    if kind == "bar":
        th = np.deg2rad(angle_deg)
        s = dx * np.cos(th) + dy * np.sin(th)            # along the bar
        u = -dx * np.sin(th) + dy * np.cos(th)           # across it
        R, L = geom["R"], geom["L"]
        z = np.where(np.abs(u) < R, R - np.sqrt(np.clip(R ** 2 - u ** 2, 0, None)), np.inf)
        return np.where(np.abs(s) <= L / 2, z, np.inf)
    raise ValueError(kind)


def gaussian_blur(field, sigma_px):
    """Separable Gaussian, no scipy dependency."""
    if sigma_px <= 0:
        return field
    k = int(np.ceil(3 * sigma_px))
    x = np.arange(-k, k + 1)
    g = np.exp(-0.5 * (x / sigma_px) ** 2)
    g /= g.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 0, field)
    return np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 1, out)


def taxel_forces(xy_mm, pitch_mm, geom, centre_mm, total_n, sigma_mm=SIGMA_MM,
                 angle_deg=0.0, k_n_per_mm3=K_N_PER_MM3):
    """N per taxel for a rigid indenter pressed into foam, at a total force.

    `xy_mm` is (N, 2) taxel centres and `pitch_mm` the taxel pitch. The
    indentation depth is solved so the integrated pressure equals `total_n`;
    each taxel then reads the blurred pressure over a square of about one
    pitch around its centre.
    """
    xy, pitch = np.asarray(xy_mm, dtype=float), float(pitch_mm)
    lo = xy.min(0) - 3 * pitch - 2 * sigma_mm
    hi = xy.max(0) + 3 * pitch + 2 * sigma_mm
    gx = np.arange(lo[0], hi[0] + GRID_MM, GRID_MM)
    gy = np.arange(lo[1], hi[1] + GRID_MM, GRID_MM)
    X, Y = np.meshgrid(gx, gy)
    z = indenter_height(geom, X - centre_mm[0], Y - centre_mm[1], angle_deg)
    cell = GRID_MM ** 2

    def force_at(delta):
        return k_n_per_mm3 * np.clip(delta - z, 0, None).sum() * cell

    lo, hi = 0.0, 0.05
    while force_at(hi) < total_n and hi < 200.0:
        hi *= 2
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if force_at(mid) < total_n:
            lo = mid
        else:
            hi = mid
    p = k_n_per_mm3 * np.clip(0.5 * (lo + hi) - z, 0, None)
    p = gaussian_blur(p, sigma_mm / GRID_MM)
    if p.sum() <= 0:
        return np.zeros(len(xy))
    # integrate over each taxel's cell
    ix = np.clip(np.round((xy[:, 0] - gx[0]) / GRID_MM).astype(int), 0, len(gx) - 1)
    iy = np.clip(np.round((xy[:, 1] - gy[0]) / GRID_MM).astype(int), 0, len(gy) - 1)
    half = max(1, int(round(pitch / 2 / GRID_MM)))
    cum = p.cumsum(0).cumsum(1)
    cum = np.pad(cum, ((1, 0), (1, 0)))
    y0, y1 = np.clip(iy - half, 0, len(gy)), np.clip(iy + half + 1, 0, len(gy))
    x0, x1 = np.clip(ix - half, 0, len(gx)), np.clip(ix + half + 1, 0, len(gx))
    box = cum[y1, x1] - cum[y0, x1] - cum[y1, x0] + cum[y0, x0]
    # Physical scale, NOT renormalised to total_n: whatever falls outside the taxels
    # (the bar overhangs the skin, the foam spreads past the outline) is not read,
    # so the model predicts the force the skin actually sees.
    return np.clip(box, 0, None) * cell
