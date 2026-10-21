# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SO-101 MuJoCo Tactile contributors.
# From TactileSim (tactilesim/sensor/loader.py); modified for OpenTac.
"""A real OpenTac-CAD sensor from its manifest, with the analytic bump readout.

A real board is not a regular lattice: it has an irregular outline, removed
taxels, 2x2 blocks the outline cut short, and taxels with no bump above them.
Everything needed for a flat board is in its manifest: `pixels` gives each
taxel's position and scan row and column, and `shear` gives the bump centres
and the bump above each taxel.

Flat sensors only. A manifest with a `sim3d` block (a sensor wrapped on a
curved part) is refused rather than flattened, because a wrapped sensor
modelled as a flat pad has every contact normal in the wrong direction.

Units: the manifest is in millimetres and this package in SI. The conversion
happens once, here; `describe()` prints both.

The bump height, taper and sheet thickness are chosen when the bump sheet is
built, after export, so the manifest does not carry them. They arrive as a
`BumpParams` (OpenTac-CAD's defaults unless given), or as a `bump_params`
block in the manifest.

The manifest is checked on loading: lengths in mm, finite positions, integer
and unique scan indices inside the scan grid, unique bump centres, a bump
index in [-1, number of bumps) for every taxel, at least one taxel under
every bump, footprints that do not overlap at the pitch, and bump bases that
do not overlap.
"""
import dataclasses
import json
import os

import numpy as np

from opentac.manifest import ManifestError

from .params import MM, BumpParams
from .surrogate import Surrogate

#: Taxels the manifest marks as having no bump above them (shear.b == -1).
NO_BUMP = -1

#: Footprints may touch (OpenTac-CAD allows a zero gap); one larger than the
#: pitch by more than this, in metres, overlaps its neighbour. The margin
#: covers the rounding of manifest coordinates.
FOOTPRINT_TOLERANCE_M = 1e-5


class TactileSensor:
    """One OpenTac sensor, ready to read.

    Taxel positions, bump centres and the taxel-to-bump grouping come from the
    manifest. The bump-level physics is the Winkler surrogate; the
    distribution onto taxels, which depends on the board's layout, is done here.
    """

    def __init__(self, manifest: dict, bump: BumpParams = None, name=None):
        self.manifest = manifest
        self.name = name
        # The version is reported, not checked: a feature is detected by its block.
        self.v = int(manifest.get("v", 0))
        pixels = manifest.get("pixels")
        if not pixels:
            raise ManifestError("manifest has no 'pixels'; nothing to simulate.")
        if manifest.get("units", "mm") != "mm":
            raise ManifestError("Sensor manifest lengths must use mm")
        px = np.asarray(pixels, dtype=float)
        if px.ndim != 2 or px.shape[1] < 4 or not np.isfinite(px).all():
            raise ManifestError("pixels must contain finite x, y, row, col values")
        if np.any(px[:, 2:4] != np.floor(px[:, 2:4])):
            raise ManifestError("Pixel row and column indices must be integers")
        if len(np.unique(px[:, :2], axis=0)) != len(px) or len(np.unique(px[:, 2:4], axis=0)) != len(px):
            raise ManifestError("Pixel positions and scan indices must be unique")
        self.xy = px[:, :2] * MM                    # (T, 2) metres
        self.rc = px[:, 2:4].astype(int)            # (T, 2) scan row, col
        self.n_rows = int(manifest["r"])
        self.n_cols = int(manifest["c"])
        self.taxel_size = (float(manifest["w"]) * MM, float(manifest["h"]) * MM)
        if self.n_rows <= 0 or self.n_cols <= 0 or np.any(self.rc < 0) or np.any(self.rc >= [self.n_rows, self.n_cols]):
            raise ManifestError("Pixel scan indices are outside the sensor dimensions")
        if not np.isfinite(self.taxel_size).all() or min(self.taxel_size) <= 0:
            raise ManifestError("Taxel dimensions must be positive and finite")

        shear = manifest.get("shear") or {}
        bumps = shear.get("bumps")
        if not bumps:
            raise ManifestError(
                f"manifest (v{self.v}) has no usable 'shear' block, so there is no way to know "
                "which taxels share a bump. OpenTac-CAD added it in v5; re-export from a current "
                "version. Without it the board may still be real, but this package cannot place "
                "bumps on it.")
        if shear.get("b") is None:
            raise ManifestError("'shear' has bump centres but no 'b', so no taxel can be assigned to a bump.")
        self.bump_xy = np.asarray(bumps, dtype=float) * MM   # (B, 2) metres
        if self.bump_xy.ndim != 2 or self.bump_xy.shape[1] != 2 or not np.isfinite(self.bump_xy).all():
            raise ManifestError("shear.bumps must contain finite XY centres")
        if len(np.unique(self.bump_xy, axis=0)) != len(self.bump_xy):
            raise ManifestError("Bump centres must be unique")
        raw_b = np.asarray(shear.get("b"), dtype=float)
        if raw_b.ndim != 1 or not np.isfinite(raw_b).all() or np.any(raw_b != np.floor(raw_b)):
            raise ManifestError("shear.b must contain integer bump indices")
        if np.any(raw_b < NO_BUMP) or np.any(raw_b >= len(self.bump_xy)):
            raise ManifestError("shear.b index is outside [-1, number of bumps)")
        self.taxel_bump = raw_b.astype(int)
        self._no_bump = np.flatnonzero(self.taxel_bump == NO_BUMP)
        if len(self.taxel_bump) != len(self.xy):
            raise ManifestError(
                f"'shear.b' has {len(self.taxel_bump)} entries but there are {len(self.xy)} taxels; "
                "the manifest promises they are parallel.")
        self._reject_curved(manifest)

        # Pitch from the spacing of taxel positions, never from px/py, which
        # are the scan direction.
        self.pitch = self._derive_pitch()
        if not np.isfinite(self.pitch).all() or min(self.pitch) <= 0:
            raise ManifestError("Taxel pitch must be positive and finite")
        if np.any(np.asarray(self.taxel_size) > np.asarray(self.pitch) + FOOTPRINT_TOLERANCE_M):
            raise ManifestError("Taxel footprints overlap at the derived pitch")
        extra = manifest.get("bump_params", {})
        if "taxel_pitch_mm" in extra:
            raise ManifestError("bump_params cannot set taxel_pitch_mm; the pitch comes from the taxel positions.")
        self._p = bump or BumpParams(taxel_pitch_mm=(self.pitch[0] / MM, self.pitch[1] / MM), **extra)
        self._check_bump_bases(self._p)
        self.surrogate = Surrogate(self._p, self.bump_xy)
        if np.any(self.group_sizes() == 0):
            raise ManifestError("Every bump must have at least one assigned taxel")
        # Minimum-norm couple weights about each group's centroid: the full
        # covariance handles asymmetric and three-taxel groups, and a
        # rank-deficient group reports only the moment component its geometry
        # can observe.
        d = self._centred_offsets()
        self._moment_weights = np.zeros((self.n_taxels, 2))
        for k in range(self.n_bumps):
            members = np.flatnonzero(self.taxel_bump == k)
            self._moment_weights[members] = np.linalg.pinv(d[members].T)

    def _check_bump_bases(self, p):
        """Refuse bump bases that overlap; touching bases are accepted.

        Scaled by the base semi-axes, two equal ellipses overlap when their
        centres are less than 2 apart.
        """
        if self.n_bumps > 1:
            from scipy.spatial.distance import pdist
            if pdist(self.bump_xy / np.asarray(p.r_base)).min() < 2.0 - 1e-9:
                raise ManifestError("Neighbouring bump bases overlap; reduce bump_base_frac.")

    # -- parameters ------------------------------------------------------

    @property
    def p(self) -> BumpParams:
        """The bump parameters. Assigning to this propagates to the surrogate."""
        return self._p

    @p.setter
    def p(self, value: BumpParams):
        self._check_bump_bases(value)
        self._p = value
        self.surrogate.reparameterize(value)

    def set_params(self, **overrides):
        """Return a new sensor with some BumpParams fields replaced.

        The entry point for domain randomisation; `params.ESTIMATE_RANGES`
        gives the plausible spans of the estimated fields. A new object is
        returned, so several parameter sets can coexist without aliasing.
        """
        return TactileSensor(self.manifest, bump=dataclasses.replace(self._p, **overrides), name=self.name)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_manifest(cls, path, bump: BumpParams = None):
        """Load from a manifest.json path, or an already-parsed dict."""
        if isinstance(path, dict):
            return cls(path, bump=bump)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data, bump=bump, name=os.path.basename(str(path)))

    def _reject_curved(self, manifest):
        """Refuse a sensor whose 3-D placement would have to be invented."""
        sim = manifest.get("sim3d")
        if sim and sim.get("taxels3d"):
            raise ManifestError(
                "this sensor is mounted on a curved surface (the manifest has a 'sim3d' block). "
                "Curved placement is not supported; this model handles flat sensors only.")

    def _derive_pitch(self):
        """Spacing between neighbouring taxels, from their positions.

        Neighbours are found in geometric order (`geometric_rc`), not scan
        order: OpenTac-CAD's pad order reverses half of the rows, and any scan
        order gives the same pitch.
        """
        by_rc = {(int(r), int(c)): i for i, (r, c) in enumerate(self.geometric_rc())}
        dx, dy = [], []
        for (r, c), i in by_rc.items():
            j = by_rc.get((r, c + 1))
            if j is not None:
                dx.append(abs(self.xy[j, 0] - self.xy[i, 0]))
            j = by_rc.get((r + 1, c))
            if j is not None:
                dy.append(abs(self.xy[j, 1] - self.xy[i, 1]))
        if not dx or not dy:
            raise ManifestError("cannot derive taxel pitch: this board has no two taxels adjacent in a row and a column.")
        return float(np.median(dx)), float(np.median(dy))

    # -- geometry --------------------------------------------------------

    @property
    def n_taxels(self):
        return len(self.xy)

    @property
    def n_bumps(self):
        return len(self.bump_xy)

    @property
    def shape(self):
        """(rows, cols) of the scanned image, matching the firmware's."""
        return (self.n_rows, self.n_cols)

    def group_sizes(self):
        """How many taxels sit under each bump.

        A full block has 4. Fewer means the outline cut the block short; such
        a bump cannot report shear along an axis with taxels on one side only.
        """
        bumped = self.taxel_bump[self.taxel_bump != NO_BUMP]
        return np.bincount(bumped, minlength=self.n_bumps)

    def describe(self):
        pit = (self.pitch[0] / MM, self.pitch[1] / MM)
        sizes = self.group_sizes()
        n_partial = int((sizes < 4).sum())
        n_nobump = int((self.taxel_bump == NO_BUMP).sum())
        return "\n".join([
            f"OpenTac sensor{' ' + self.name if self.name else ''} (manifest v{self.v})",
            f"  {self.n_taxels} taxels on a {self.n_rows}x{self.n_cols} scan grid",
            f"  pitch {pit[0]:.2f} x {pit[1]:.2f} mm   taxel {self.taxel_size[0] / MM:.2f} x {self.taxel_size[1] / MM:.2f} mm",
            f"  {self.n_bumps} bumps, {n_partial} on partial blocks, {n_nobump} taxels with no bump",
            f"  bump height {self.p.bump_height_mm:.2f} mm, E_eff {self.p.e_eff_mpa:.1f} MPa (estimate; randomise it)",
            "  units: manifest mm -> SI m",
        ])

    # -- reading ---------------------------------------------------------

    def offsets(self):
        """Each taxel's (dx, dy) from the centre of the bump above it; zero for a taxel with no bump."""
        d = np.zeros((self.n_taxels, 2))
        has = self.taxel_bump != NO_BUMP
        d[has] = self.xy[has] - self.bump_xy[self.taxel_bump[has]]
        return d

    def _apply_moments(self, forces, base, moments):
        """Add each bump's base moment as a couple over its taxels, then restore its compressive load.

        Clipping is physical (a taxel cannot pull) but adds load, so each
        group is rescaled to its bump's base force; the moment that clipping
        removes is not restored, so decoded shear saturates as a real board's does.
        """
        m = np.asarray(moments, dtype=float)
        if m.shape != (self.n_bumps, 2) or not np.isfinite(m).all():
            raise ValueError("Moments need finite (Mx, My) pairs for every bump")
        idx = np.flatnonzero(self.taxel_bump != NO_BUMP)
        b = self.taxel_bump[idx]
        weights = self._moment_weights[idx]
        couple = weights[:, 0] * m[b, 1] - weights[:, 1] * m[b, 0]
        forces[idx] = np.clip(forces[idx] + couple, 0.0, None)
        total = np.bincount(b, weights=forces[idx], minlength=self.n_bumps)
        scale = np.divide(base, total, out=np.zeros(self.n_bumps), where=total > 0)
        forces[idx] *= scale[b]
        return forces

    def taxel_forces(self, surface, moments=None):
        """Normal force at every taxel, in taxel order, for a surface.

        `surface(x, y) -> z` is the object's underside in metres in the
        sensor's frame, with z measured from the top of the sheet (the bumps
        occupy 0 .. bump height). Each bump's base force is split over its
        taxels by the compression at each taxel's own position, which is the
        pressure gradient across the bump. `moments` (from `shear_step` via
        `surrogate.base_moments`) adds the friction-driven base moment as a
        couple, which makes the image checkered under shear. A taxel with no
        bump above it feels the sheet, a full bump height lower.
        """
        base = self.surrogate.base_forces(surface)
        out = np.zeros(self.n_taxels)
        has = self.taxel_bump != NO_BUMP
        if has.any():
            idx = np.nonzero(has)[0]
            b = self.taxel_bump[idx]
            z = surface(self.xy[idx, 0], self.xy[idx, 1])
            w = np.clip(self.p.height - z, 0.0, None)
            tot = np.zeros(self.n_bumps)
            np.add.at(tot, b, w)
            good = tot[b] > 1e-12
            share = np.zeros(len(idx))
            share[good] = w[good] / tot[b][good]
            # A bump whose taxels all have zero weight still carries its base
            # force; it is split evenly.
            sizes = self.group_sizes()
            share[~good] = 1.0 / np.maximum(sizes[b][~good], 1)
            out[idx] = base[b] * share
        if (~has).any():
            idx = np.nonzero(~has)[0]
            z = surface(self.xy[idx, 0], self.xy[idx, 1])
            out[idx] = self.surrogate.k * np.clip(-z, 0.0, None)
        return self._apply_moments(out, base, moments) if moments is not None else out

    def supported_forces(self, surface):
        """Every load the sensor carries for a surface: each bump's base force, then each bump-less taxel's.

        Their sum equals the sum of `taxel_forces(surface)`, since a bump's
        taxels share its base force exactly, at a fraction of the cost; the
        force solve uses it.
        """
        base = self.surrogate.base_forces(surface)
        if not len(self._no_bump):
            return base
        z = surface(self.xy[self._no_bump, 0], self.xy[self._no_bump, 1])
        return np.concatenate([base, self.surrogate.k * np.clip(-z, 0.0, None)])

    def taxel_forces_from_wrench(self, base_fz, moments):
        """Distribute per-bump (Fz, Mx, My) onto this board's taxels, conserving each bump's load.

        The counterpart of `taxel_forces` for the full MuJoCo scene, where the
        wrench is read off each bump's joints: an even share of the base force
        plus the moment as a couple, over each bump's actual member taxels.
        """
        fz = np.asarray(base_fz, dtype=float)
        if fz.shape != (self.n_bumps,) or not np.isfinite(fz).all() or np.any(fz < 0):
            raise ValueError("Wrenches need finite nonnegative bump loads")
        out = np.zeros(self.n_taxels)
        has = self.taxel_bump != NO_BUMP
        b = self.taxel_bump[has]
        out[has] = fz[b] / np.maximum(self.group_sizes()[b], 1)
        return self._apply_moments(out, fz, moments)

    def _centred_offsets(self):
        """Taxel offsets from each group's own centroid.

        On a block the outline cut short, the offsets from the bump centre do
        not sum to zero, and a pure normal press would decode as shear.
        Referencing the centroid makes a uniform load read zero for any group
        shape and changes nothing on a full 2x2 block.
        """
        d = self.offsets()
        has = self.taxel_bump != NO_BUMP
        b = self.taxel_bump[has]
        sizes = self.group_sizes()
        cx = np.zeros(self.n_bumps)
        cy = np.zeros(self.n_bumps)
        np.add.at(cx, b, d[has, 0])
        np.add.at(cy, b, d[has, 1])
        n = np.maximum(sizes, 1)
        out = d.copy()
        out[has, 0] -= (cx / n)[b]
        out[has, 1] -= (cy / n)[b]
        return out

    def degenerate_axes(self):
        """(B, 2) bool: axes on which a group cannot report shear at all.

        True where every taxel of the group lies on one side of its centroid
        along that axis; `decode_shear` components there are noise.
        """
        d = self._centred_offsets()
        has = self.taxel_bump != NO_BUMP
        b = self.taxel_bump[has]
        spread = np.zeros((self.n_bumps, 2))
        np.add.at(spread, b, np.abs(d[has]))
        return spread < 1e-09

    def decode_shear(self, per_taxel):
        """Per-bump shear force (B, 2) in N, from the normal-force image alone.

        This is what the firmware can compute: the first moment of each
        group's readings about its centroid gives the base moment, and a
        guided cantilever turns it into a lateral force, Ft = 2 M / H.
        Check `degenerate_axes()` before trusting a component.
        """
        f = np.asarray(per_taxel, dtype=float)
        has = self.taxel_bump != NO_BUMP
        d = self._centred_offsets()
        b = self.taxel_bump[has]
        my = np.zeros(self.n_bumps)
        mx = np.zeros(self.n_bumps)
        np.add.at(my, b, f[has] * d[has, 0])
        np.add.at(mx, b, -f[has] * d[has, 1])
        h = self.p.height
        return np.column_stack([2.0 * my / h, -2.0 * mx / h])

    def checkerness(self, per_taxel, per_bump=False):
        """How far each bump's load sits off-centre within its group: the normalised dipole.

        0 when the taxels share evenly; 1 when a single corner taxel carries
        everything; a fully one-sided edge split of a 2x2 block reads 0.707.
        The dipole, not the variance, is measured, because a ball loads the
        middle of a block more than its corners through curvature alone.
        """
        f = np.asarray(per_taxel, dtype=float)
        has = self.taxel_bump != NO_BUMP
        b = self.taxel_bump[has]
        d = self._centred_offsets()[has]
        tot = np.zeros(self.n_bumps)
        np.add.at(tot, b, f[has])
        mx = np.zeros(self.n_bumps)
        my = np.zeros(self.n_bumps)
        np.add.at(mx, b, f[has] * d[:, 0])
        np.add.at(my, b, f[has] * d[:, 1])
        reach = np.zeros(self.n_bumps)
        np.maximum.at(reach, b, np.hypot(d[:, 0], d[:, 1]))
        denom = tot * reach
        live = (tot > 1e-12) & (denom > 1e-18)
        val = np.zeros(self.n_bumps)
        val[live] = np.hypot(mx[live], my[live]) / denom[live]
        if per_bump:
            return val
        return float(val[live].mean()) if live.any() else 0.0

    def geometric_rc(self):
        """Each taxel's (row, col) by position, top row first; not the scan order.

        The manifest's row and column are pad order, in which OpenTac-CAD
        reverses half of the rows; plotted in that order, one round contact
        splits into two blobs. Use geometric order for anything spatial and
        scan order to compare with what the firmware streams.
        """
        ux = np.unique(np.round(self.xy[:, 0], 9))
        uy = np.unique(np.round(self.xy[:, 1], 9))
        col = np.searchsorted(ux, np.round(self.xy[:, 0], 9))
        row = np.searchsorted(uy, np.round(self.xy[:, 1], 9))
        return np.column_stack([len(uy) - 1 - row, col])

    def to_image(self, per_taxel, fill=0.0, order="geometric"):
        """Scatter per-taxel values into a 2-D image, in "geometric" or "scan" order."""
        per_taxel = np.asarray(per_taxel, dtype=float)
        if order == "scan":
            img = np.full(self.shape, fill, dtype=float)
            img[self.rc[:, 0], self.rc[:, 1]] = per_taxel
            return img
        if order != "geometric":
            raise ValueError(f"order must be 'geometric' or 'scan', not {order!r}")
        g = self.geometric_rc()
        img = np.full((g[:, 0].max() + 1, g[:, 1].max() + 1), fill, dtype=float)
        img[g[:, 0], g[:, 1]] = per_taxel
        return img

    def centre(self):
        """A contact point on a bump: the bump centre nearest the centroid of the bumps.

        The centroid of the taxels falls between four bumps on a regular grid,
        where a small ball can touch nothing.
        """
        c = self.bump_xy.mean(axis=0)
        i = int(np.argmin(np.hypot(self.bump_xy[:, 0] - c[0], self.bump_xy[:, 1] - c[1])))
        return float(self.bump_xy[i, 0]), float(self.bump_xy[i, 1])

    def _make_surface(self, shape, radius_mm, cx, cy, tilt):
        from . import surrogate as sg
        if not np.isfinite([radius_mm, cx, cy, *tilt]).all() or radius_mm <= 0:
            raise ValueError("Probe radius must be positive and all coordinates finite")
        if shape == "sphere":
            return lambda z: sg.sphere_surface(radius_mm * MM, z, cx, cy)
        if shape == "cylinder":
            return lambda z: sg.cylinder_surface(radius_mm * MM, z, "y", cx, cy)
        if shape == "plane":
            return lambda z: sg.plane_surface(z, tilt[0], tilt[1])
        raise ValueError(f"unknown probe shape {shape!r}; expected sphere, cylinder or plane")

    def press(self, shape="sphere", force_n=5.0, radius_mm=10.0, cx_mm=None, cy_mm=None, tilt=(0.0, 0.0)):
        """Press an object in to a total normal force; returns (surface, z).

        Force, not depth: at a fixed depth the object's own curvature decides
        which bumps it reaches. The total includes the load on taxels with no
        bump. A load that would compress a bump by its full height is outside
        the lumped model and raises.
        """
        cx, cy = self.centre()
        if cx_mm is not None:
            cx = cx_mm * MM
        if cy_mm is not None:
            cy = cy_mm * MM
        make = self._make_surface(shape, radius_mm, cx, cy, tilt)
        z, _ = self.surrogate.press_to_force(make, force_n, force_fn=self.supported_forces)
        if np.any(self.surrogate.compressions(make(z)) >= self.p.height):
            raise ValueError("Requested load would flatten a bump; outside the lumped model")
        return make(z), z

    def probe(self, shape="sphere", force_n=5.0, radius_mm=10.0, cx_mm=None, cy_mm=None, tilt=(0.0, 0.0), as_image=True):
        """Press a simple object in and return what the sensor reads, as an image or in taxel order."""
        surf, _ = self.press(shape, force_n, radius_mm, cx_mm, cy_mm, tilt)
        f = self.taxel_forces(surf)
        return self.to_image(f) if as_image else f

    def press_and_twist(self, yaw_deg, shape="sphere", force_n=5.0, radius_mm=10.0, cx_mm=None, cy_mm=None, steps=None):
        """Press an object in, then rotate it about the contact normal; one reading per yaw.

        The stick anchors advance continuously through `yaw_deg`, so stick,
        breakaway and slip appear in order. The normal force is held constant
        while the object turns. Each reading is a dict with the taxel forces
        and images, the true and decoded per-bump shear (raw, and less the
        no-shear baseline that a curved object already produces), and the
        checkerness.
        """
        surf, _ = self.press(shape, force_n, radius_mm, cx_mm, cy_mm)
        cx, cy = self.centre()
        if cx_mm is not None:
            cx = cx_mm * MM
        if cy_mm is not None:
            cy = cy_mm * MM
        angles = np.asarray(yaw_deg, dtype=float)
        if steps:
            angles = np.linspace(0.0, angles.max(), steps)
        self.surrogate.reset_shear()
        base_f = self.taxel_forces(surf)
        base_decode = self.decode_shear(base_f)
        baseline_checkerness = self.checkerness(base_f)
        baseline_groups = self.checkerness(base_f, per_bump=True)
        out = []
        for a in angles:
            yaw = np.radians(a)
            # shear_step measures motion about the sensor origin; the twist is
            # about the contact point.
            obj_xy = (cy * yaw, -cx * yaw)
            ft, _ = self.surrogate.shear_step(surf, obj_xy, yaw)
            m = self.surrogate.base_moments()
            f = self.taxel_forces(surf, moments=m)
            decoded = self.decode_shear(f)
            out.append({
                "yaw_deg": float(a),
                "forces": f,
                "baseline": base_f,
                "image": self.to_image(f),
                "image_scan": self.to_image(f, order="scan"),
                "shear_true": ft,
                "shear_decoded_raw": decoded,
                "shear_decoded": decoded - base_decode,
                "checkerness": self.checkerness(f),
                "checkerness_baseline": baseline_checkerness,
                "checkerness_shear": float(np.mean(np.abs(self.checkerness(f, per_bump=True) - baseline_groups))),
                "total_n": float(f.sum()),
                "stuck": int(self.surrogate.stuck.sum()),
            })
        return out
