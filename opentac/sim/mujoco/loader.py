"""Load a real OpenTac CAD sensor from its manifest.

This is the link that was missing: everything in this package described a
regular lattice of identical bumps, and a real OpenTac board is not that. It has an
irregular outline, taxels that were removed, 2x2 blocks the outline cut short,
and taxels with no bump above them at all.

v1 handles FLAT sensors -- a pad on a parallel-jaw gripper, which is most RL
grippers. A flat board's manifest already carries everything needed, so nothing
new has to be exported: `pixels` gives every taxel's position and row/column,
`shear` gives the bump centres and which taxels sit under each, and `r`/`c`/
`rp`/`cp`/`m` give the scan topology that ghosting comes out of.

Curved sensors need the `sim3d` block (manifest v6) for their 3D placement.
`from_manifest` refuses those rather than silently flattening them -- a wrapped
fingertip modelled as a flat pad gets the contact geometry wrong everywhere and
nothing downstream would notice.

Units
-----
The manifest is in **millimetres**; everything in this package is **SI**. The
conversion happens once, here, and `describe()` prints both so a factor of 1000
cannot hide. That mistake is silent in MuJoCo: a sensor 1000x too big still
simulates, it just never touches anything.

What is NOT in the manifest
---------------------------
Bump height and taper, and the sheet/recess thicknesses. Those are chosen when
the bump sheet is built, which happens after export -- the manifest's own
_readme says so. They arrive as a `BumpParams`, defaulting to OpenTac's defaults.
If the user changed them when they printed, they have to say so here too.
"""
import json
import os

import numpy as np

from opentac.manifest import ManifestError

from .params import BumpParams
from .surrogate import Surrogate

MM = 1e-3

#: Taxels the manifest marks as having no bump above them (shear.b == -1).
NO_BUMP = -1


class TactileSensor:
    """One OpenTac sensor, ready to read.

    Geometry is real -- taxel positions, bump centres and the taxel/bump
    grouping all come from the manifest rather than from an idealised lattice.
    The bump-level physics is the validated Winkler surrogate; only the
    distribution onto taxels is done here, because that is the part that
    depends on the board's actual layout.
    """

    def __init__(self, manifest: dict, bump: BumpParams = None, name=None):
        self.manifest = manifest
        self.name = name
        # Check for the BLOCK, not the version number. A real export from a
        # deployed OpenTac turned up carrying "v": 3 with a full shear block --
        # version numbers travel differently from features, and refusing that
        # file would have been refusing a sensor we can simulate perfectly.
        # The version is still reported in describe() so it is not invisible.
        self.v = int(manifest.get("v", 0))

        pixels = manifest.get("pixels")
        if not pixels:
            raise ManifestError("manifest has no 'pixels'; nothing to simulate.")

        px = np.asarray(pixels, dtype=float)
        self.xy = px[:, :2] * MM                    # (T, 2) metres
        self.rc = px[:, 2:4].astype(int)            # (T, 2) row, col
        self.n_rows = int(manifest["r"])
        self.n_cols = int(manifest["c"])
        self.taxel_size = (float(manifest["w"]) * MM, float(manifest["h"]) * MM)

        shear = manifest.get("shear") or {}
        bumps = shear.get("bumps")
        if not bumps:
            raise ManifestError(
                f"manifest (v{self.v}) has no usable 'shear' block, so there is "
                "no way to know which taxels share a bump. OpenTac CAD added "
                "it in v5; re-export from a current version. Without it the "
                "board may still be real, but this package cannot place bumps "
                "on it.")
        if shear.get("b") is None:
            raise ManifestError(
                "'shear' has bump centres but no 'b', so no taxel can be "
                "assigned to a bump.")
        self.bump_xy = np.asarray(bumps, dtype=float) * MM   # (B, 2) metres
        self.taxel_bump = np.asarray(shear.get("b"), dtype=int)
        if len(self.taxel_bump) != len(self.xy):
            raise ManifestError(
                f"'shear.b' has {len(self.taxel_bump)} entries but there are "
                f"{len(self.xy)} taxels; the manifest promises they are parallel.")

        self._reject_curved(manifest)

        # Taxel pitch from the SPACING of pixels, never from px/py -- those are
        # pad-scan order, and the manifest's own _readme says so.
        self.pitch = self._derive_pitch()
        self._p = bump or BumpParams(taxel_pitch_mm=(self.pitch[0] / MM,
                                                     self.pitch[1] / MM))
        self.surrogate = Surrogate(self._p, self.bump_xy)

    # -- parameters ------------------------------------------------------

    @property
    def p(self) -> BumpParams:
        """The bump parameters. Assigning to this propagates to the surrogate.

        A plain attribute here was a trap: the surrogate caches stiffness at
        construction, so `sensor.p = something_else` changed nothing and said
        nothing. Anything randomising per episode would have been training
        against one fixed stiffness while believing otherwise.
        """
        return self._p

    @p.setter
    def p(self, value: BumpParams):
        self._p = value
        self.surrogate.reparameterize(value)

    def set_params(self, **overrides):
        """Return a NEW sensor with some BumpParams fields replaced.

        The per-episode entry point for domain randomisation. What to
        randomise and over what distribution is the RL designer's call, not
        ours -- our job is only that changing it is cheap and that it actually
        takes effect. `params.ESTIMATE_RANGES` documents which of these are
        measurements and which are guesses.

        Returns a new object rather than mutating, so a vectorised env can
        hold several stiffnesses at once without them aliasing.
        """
        import dataclasses

        return TactileSensor(self.manifest,
                             bump=dataclasses.replace(self._p, **overrides),
                             name=self.name)

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
        """Refuse a sensor whose 3D placement we would have to invent.

        A board unrolled from a mesh is flat only in the manifest's drawing
        frame; on the part it wraps. Modelling it as a flat pad puts every
        contact normal in the wrong direction, and the readings would still
        look entirely reasonable.
        """
        sim = manifest.get("sim3d")
        if sim and sim.get("taxels3d"):
            raise ManifestError(
                "this sensor is mounted on a curved surface (the manifest has a "
                "'sim3d' block). Curved placement is not supported yet; v1 "
                "handles flat sensors only.")

    def _derive_pitch(self):
        """Spacing between neighbouring taxels, from their real positions."""
        by_rc = {(int(r), int(c)): i for i, (r, c) in enumerate(self.rc)}
        dx, dy = [], []
        for (r, c), i in by_rc.items():
            j = by_rc.get((r, c + 1))
            if j is not None:
                dx.append(abs(self.xy[j, 0] - self.xy[i, 0]))
            j = by_rc.get((r + 1, c))
            if j is not None:
                dy.append(abs(self.xy[j, 1] - self.xy[i, 1]))
        if not dx or not dy:
            raise ManifestError(
                "cannot derive taxel pitch: this board has no two taxels "
                "adjacent in a row and a column.")
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

        A full block is 4. Fewer means the outline cut the block short, and
        that bump is degenerate on any axis with taxels on one side only --
        its shear reading along that axis is not trustworthy.
        """
        sizes = np.zeros(self.n_bumps, dtype=int)
        bumped = self.taxel_bump[self.taxel_bump != NO_BUMP]
        np.add.at(sizes, bumped, 1)
        return sizes

    def describe(self):
        pit = (self.pitch[0] / MM, self.pitch[1] / MM)
        sizes = self.group_sizes()
        n_partial = int((sizes < 4).sum())
        n_nobump = int((self.taxel_bump == NO_BUMP).sum())
        return "\n".join([
            f"OpenTac sensor{' ' + self.name if self.name else ''} "
            f"(manifest v{self.v})",
            f"  {self.n_taxels} taxels on a {self.n_rows}x{self.n_cols} scan grid",
            f"  pitch {pit[0]:.2f} x {pit[1]:.2f} mm   "
            f"taxel {self.taxel_size[0]/MM:.2f} x {self.taxel_size[1]/MM:.2f} mm",
            f"  {self.n_bumps} bumps, {n_partial} on partial blocks, "
            f"{n_nobump} taxels with no bump",
            f"  bump height {self.p.bump_height_mm:.2f} mm, "
            f"E_eff {self.p.e_eff_mpa:.1f} MPa (ESTIMATE - randomise it)",
            "  units: manifest mm -> SI m",
        ])

    # -- reading ---------------------------------------------------------

    def offsets(self):
        """Each taxel's (dx, dy) from the centre of the bump above it.

        Zero for a taxel with no bump. These are the REAL offsets, which is
        what lets a partial 2x2 block work: the surrogate's own taxel_forces
        assumes four taxels at exactly +/-pitch/2, and a block the outline cut
        short has neither the count nor the symmetry.
        """
        d = np.zeros((self.n_taxels, 2))
        has = self.taxel_bump != NO_BUMP
        d[has] = self.xy[has] - self.bump_xy[self.taxel_bump[has]]
        return d

    def _group_second_moments(self):
        """Sum of dx^2 and dy^2 within each bump's group, for the moment split."""
        d = self.offsets()
        has = self.taxel_bump != NO_BUMP
        sxx = np.zeros(self.n_bumps)
        syy = np.zeros(self.n_bumps)
        np.add.at(sxx, self.taxel_bump[has], d[has, 0] ** 2)
        np.add.at(syy, self.taxel_bump[has], d[has, 1] ** 2)
        return sxx, syy

    def taxel_forces(self, surface, moments=None):
        """Normal force at every taxel, in taxel order, for a surface.

        `surface(x, y) -> z` is the object's underside, in metres, in the
        sensor's own frame, with z measured from the top of the TPU sheet (so
        the bumps occupy 0 .. bump_height).

        Two contributions, exactly as the full model has:

          1. the pressure GRADIENT across the bump's footprint -- a bump on a
             slope loads its downhill side harder, present with no friction
             at all; and
          2. the friction-driven BASE MOMENT, if `moments` is given (from
             `shear_step` via `surrogate.base_moments`). A bump loaded
             sideways tips, and the taxels under it split its load
             asymmetrically. This is the term that makes the normal-force
             image go checkered under shear, and it is the whole reason the
             2x2 grouping exists.

        The bump-level physics is the validated surrogate; only the
        distribution onto taxels happens here, because that is the part that
        depends on the board's real layout.
        """
        base = self.surrogate.base_forces(surface)          # (B,)
        out = np.zeros(self.n_taxels)
        has = self.taxel_bump != NO_BUMP

        if has.any():
            idx = np.nonzero(has)[0]
            b = self.taxel_bump[idx]
            # Local compression at each taxel's OWN position, as the weight.
            z = surface(self.xy[idx, 0], self.xy[idx, 1])
            w = np.clip(self.p.height - z, 0.0, None)
            tot = np.zeros(self.n_bumps)
            np.add.at(tot, b, w)
            good = tot[b] > 1e-12
            share = np.zeros(len(idx))
            share[good] = w[good] / tot[b][good]
            # A bump whose taxels all read zero weight still carries its base
            # force; split it evenly rather than dropping it.
            sizes = self.group_sizes()
            share[~good] = 1.0 / np.maximum(sizes[b][~good], 1)
            out[idx] = base[b] * share

            if moments is not None:
                m = np.asarray(moments, dtype=float)
                d = self.offsets()[idx]
                sxx, syy = self._group_second_moments()
                # f_i = My*dx_i/sum(dx^2) - Mx*dy_i/sum(dy^2). Reduces to the
                # surrogate's 4*ax^2 form on a symmetric full block, and stays
                # correct on a partial one, where those sums are different.
                fx = np.divide(m[b, 1] * d[:, 0], sxx[b],
                               out=np.zeros(len(idx)), where=sxx[b] > 1e-18)
                fy = np.divide(m[b, 0] * d[:, 1], syy[b],
                               out=np.zeros(len(idx)), where=syy[b] > 1e-18)
                couple = fx - fy
                # A moment is a couple: it must redistribute the bump's load,
                # never change its total. Subtracting the group mean makes the
                # sum exactly zero BY CONSTRUCTION, which matters on a
                # degenerate group whose offsets do not cancel.
                gsum = np.zeros(self.n_bumps)
                np.add.at(gsum, b, couple)
                couple -= (gsum / np.maximum(sizes, 1))[b]
                out[idx] = np.clip(out[idx] + couple, 0.0, None)
                # Clipping is physical -- a taxel cannot pull -- but it breaks
                # the couple: it both loses moment (so the decoded shear
                # SATURATES, by ~10-20% once bumps start lifting) and adds
                # normal force (measured 20.0 N -> 21.2 N at 2 degrees of
                # twist). Restoring the bump's own base force keeps the grip
                # force honest, which is what an RL policy is standing on; the
                # shear saturation is left visible rather than papered over,
                # because a real board saturates the same way.
                tot2 = np.zeros(self.n_bumps)
                np.add.at(tot2, b, out[idx])
                scale = np.divide(base, tot2, out=np.ones(self.n_bumps),
                                  where=tot2 > 1e-15)
                out[idx] *= scale[b]

        # Taxels with no bump above them feel the sheet, which sits a full
        # bump height lower -- so an object has to press that much deeper
        # before they read anything. Correct in kind; the sheet's own
        # stiffness is not modelled (see NOTES section 6, base-sheet flex).
        if (~has).any():
            idx = np.nonzero(~has)[0]
            z = surface(self.xy[idx, 0], self.xy[idx, 1])
            out[idx] = self.surrogate.k * np.clip(-z, 0.0, None)
        return np.clip(out, 0.0, None)

    def taxel_forces_from_wrench(self, base_fz, moments):
        """Distribute per-bump (Fz, Mx, My) onto this board's real taxels.

        The counterpart of `taxel_forces` for the FULL MuJoCo lattice, where
        the wrench is read off each bump's joints rather than evaluated from
        an analytic surface. Same split as sim.Lattice.taxel_forces -- an even
        share of the base force plus the moment as a couple -- but over each
        bump's actual member taxels instead of an assumed 2x2 at +/-pitch/2.

        That generalisation is the point: a block the outline cut short has
        neither four taxels nor a symmetric offset, and the assumed version
        would put load on taxels that do not exist.
        """
        fz = np.asarray(base_fz, dtype=float)
        m = np.asarray(moments, dtype=float)
        out = np.zeros(self.n_taxels)
        has = self.taxel_bump != NO_BUMP
        if not has.any():
            return out

        idx = np.nonzero(has)[0]
        b = self.taxel_bump[idx]
        d = self.offsets()[idx]
        sizes = self.group_sizes()
        sxx, syy = self._group_second_moments()

        share = fz[b] / np.maximum(sizes[b], 1)
        fx = np.divide(m[b, 1] * d[:, 0], sxx[b],
                       out=np.zeros(len(idx)), where=sxx[b] > 1e-18)
        fy = np.divide(m[b, 0] * d[:, 1], syy[b],
                       out=np.zeros(len(idx)), where=syy[b] > 1e-18)
        couple = fx - fy
        gsum = np.zeros(self.n_bumps)
        np.add.at(gsum, b, couple)
        couple -= (gsum / np.maximum(sizes, 1))[b]
        out[idx] = np.clip(share + couple, 0.0, None)
        return out

    def _centred_offsets(self):
        """Taxel offsets measured from each GROUP's own centroid.

        Not from the bump centre, and the difference only shows up on a block
        the outline cut short. Such a block has taxels on one side only, so
        its offsets do not sum to zero -- and then a PURE NORMAL PRESS
        produces a non-zero first moment and decodes as shear that is not
        there. Measured on a 41-taxel disc with 9 partial blocks: a flat press
        decoded 18.4 N of phantom shear before this.

        Referencing the centroid makes a uniform load read exactly zero for
        any group shape, and changes nothing on a full 2x2, where the centroid
        IS the bump centre. It does not rescue the degenerate axis -- see
        `degenerate_axes` -- it just stops it inventing a signal.
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

        True where every taxel of the group sits on one side of its centroid
        along that axis -- there is no differential left to read. The
        manifest's own _readme says the same thing about groups of fewer than
        four. Read this alongside `decode_shear`: those components are noise,
        not measurement.
        """
        d = self._centred_offsets()
        has = self.taxel_bump != NO_BUMP
        b = self.taxel_bump[has]
        spread = np.zeros((self.n_bumps, 2))
        np.add.at(spread, b, np.abs(d[has]))
        return spread < 1e-9

    def decode_shear(self, per_taxel):
        """Recover per-bump shear force from the normal-force image alone.

        This is what the FIRMWARE can do: it never sees a moment, only four
        numbers per 2x2 cell. Taking their differential, signed by each
        taxel's offset, gives the base moment back, and a guided cantilever
        turns that into a lateral force -- a stretch v puts k*v*H/2 into the
        base, so Ft = 2*M/H.

        Returns (B, 2) in newtons. Check `degenerate_axes()` before trusting a
        component: a block the outline cut short has no differential on its
        degenerate axis, so that number means nothing.
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
        """How far each bump's load sits OFF-CENTRE within its own 2x2.

        This is the "checkered" pattern as a number: the normalised first
        moment (the dipole) of the group's load. 0.0 when the taxels share
        evenly; 1.0 when a single CORNER taxel carries everything.

        An EDGE-loaded cell tops out at 1/sqrt(2) = 0.707, not 1.0, because
        the normaliser is the largest dipole the geometry allows -- all the
        load at the most distant taxel, which is a diagonal -- while an edge
        split puts the dipole along an axis. So a fully one-sided 2x2 reads
        0.707, and a 3:1 edge split reads exactly 0.25*sqrt(2) = 0.354.

        Measuring the dipole specifically, rather than any old unevenness, is
        the point. A ball already loads the middle of a cell harder than its
        corners through pure curvature, and that is a MONOPOLE: it is large,
        it has nothing to do with shear, and it swamps a plain variance. The
        dipole is what tipping produces, and it is the same quantity
        `decode_shear` reads -- so this rises monotonically with twist until
        the bumps slip, which a variance does not.
        """
        f = np.asarray(per_taxel, dtype=float)
        has = self.taxel_bump != NO_BUMP
        b = self.taxel_bump[has]
        # Centroid-referenced, so a cut-short block does not read as permanently
        # checkered under a uniform load. See _centred_offsets.
        d = self._centred_offsets()[has]

        tot = np.zeros(self.n_bumps)
        np.add.at(tot, b, f[has])
        mx = np.zeros(self.n_bumps)
        my = np.zeros(self.n_bumps)
        np.add.at(mx, b, f[has] * d[:, 0])
        np.add.at(my, b, f[has] * d[:, 1])

        # Normalise by the largest dipole the geometry could produce: all the
        # load at the most distant taxel of the group.
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
        """Each taxel's (row, col) by POSITION, not by scan order.

        This is not the same as `pixels`' row/col, and the difference is not
        cosmetic. Those indices are PAD order: `split_row_pin_order` assigns
        the top half of the rows in geometric order and then the bottom half
        REVERSED, so the pads meet the routing on the correct side. Plotting a
        single round contact in pad order splits it into two separate blobs at
        opposite ends of the image -- and a convolution over that image is
        reading a scrambled picture, which is exactly what an RL policy would
        be handed.

        So: geometric order for anything spatial (plots, CNNs), pad order for
        comparing against what the firmware actually streams.
        """
        ux = np.unique(np.round(self.xy[:, 0], 9))
        uy = np.unique(np.round(self.xy[:, 1], 9))
        col = np.searchsorted(ux, np.round(self.xy[:, 0], 9))
        row = np.searchsorted(uy, np.round(self.xy[:, 1], 9))
        # Top row first, the way an image is normally read.
        return np.column_stack([len(uy) - 1 - row, col])

    def to_image(self, per_taxel, fill=0.0, order="geometric"):
        """Scatter per-taxel values into a 2D image.

        `order="geometric"` lays taxels out by position -- use it for plots and
        for anything a convolution will see. `order="scan"` uses the manifest's
        own row/col, which is what the firmware streams, so code compared
        against hardware should use that.
        """
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
        """A contact point that actually lands ON a bump.

        The centroid of the taxels is the WRONG default: on a regular grid it
        falls exactly between four bumps, which is the worst case. Measured on
        a real 48-taxel board, an 8 mm ball pressed 1 mm at the centroid reads
        exactly zero -- the nearest bump is 4.2 mm away laterally, by which
        point the ball has curved back up 1.19 mm and cleared the bump tops
        entirely. That is a real property of a bump lattice (NOTES 4.2), but
        it makes a terrible default.
        """
        c = self.bump_xy.mean(axis=0)
        i = int(np.argmin(np.hypot(self.bump_xy[:, 0] - c[0],
                                   self.bump_xy[:, 1] - c[1])))
        return float(self.bump_xy[i, 0]), float(self.bump_xy[i, 1])

    def _make_surface(self, shape, radius_mm, cx, cy, tilt):
        from . import surrogate as sg
        if shape == "sphere":
            return lambda z: sg.sphere_surface(radius_mm * MM, z, cx, cy)
        if shape == "cylinder":
            return lambda z: sg.cylinder_surface(radius_mm * MM, z, "y", cx, cy)
        if shape == "plane":
            return lambda z: sg.plane_surface(z, tilt[0], tilt[1])
        raise ValueError(
            f"unknown probe shape {shape!r}; expected sphere, cylinder or plane")

    def press(self, shape="sphere", force_n=5.0, radius_mm=10.0,
              cx_mm=None, cy_mm=None, tilt=(0.0, 0.0)):
        """Press an object in to a total NORMAL FORCE. Returns (surface, z).

        Force, not depth. NOTES puts this first among the two things to know
        before changing anything: at a fixed depth the object's own curvature
        decides which bumps it reaches, so every stiffness engages an
        identical set and any comparison looks flat. It also keeps the numbers
        physical -- 0.6 mm of flat plate on this board is 916 N.
        """
        cx, cy = self.centre()
        if cx_mm is not None:
            cx = cx_mm * MM
        if cy_mm is not None:
            cy = cy_mm * MM
        make = self._make_surface(shape, radius_mm, cx, cy, tilt)
        z, _ = self.surrogate.press_to_force(make, force_n)
        return make(z), z

    def probe(self, shape="sphere", force_n=5.0, radius_mm=10.0,
              cx_mm=None, cy_mm=None, tilt=(0.0, 0.0), as_image=True):
        """Press a simple object in and return what the sensor reads.

        The zero-setup path: no scene, no gripper, no solver. Enough to see
        whether a design resolves the features you care about before anything
        is printed.
        """
        surf, _ = self.press(shape, force_n, radius_mm, cx_mm, cy_mm, tilt)
        f = self.taxel_forces(surf)
        return self.to_image(f) if as_image else f

    def press_and_twist(self, yaw_deg, shape="sphere", force_n=5.0,
                        radius_mm=10.0, cx_mm=None, cy_mm=None, steps=None):
        """Press an object in, then rotate it about the contact normal.

        Yields one reading per yaw in `yaw_deg`, as dicts with the taxel
        image, the decoded per-bump shear, and the checkerness. The stick
        anchors advance continuously across the sequence, so stick, breakaway
        and slip all appear in order -- twisting straight to a final angle
        would skip the stick phase entirely, which is the same trap as
        teleporting an indenter in the full model.

        The normal force is held constant while the object turns, so anything
        that changes in the image is shear, not a harder press.
        """
        surf, _ = self.press(shape, force_n, radius_mm, cx_mm, cy_mm)
        cx, cy = self.centre()
        if cx_mm is not None:
            cx = cx_mm * MM
        if cy_mm is not None:
            cy = cy_mm * MM

        angles = np.asarray(yaw_deg, dtype=float)
        if steps:                      # refine so the stick phase is resolved
            angles = np.linspace(0.0, angles.max(), steps)
        self.surrogate.reset_shear()

        # The no-shear reading. A ball already loads the middle of each bump
        # harder than its edges through pure curvature, so the image is not
        # flat at yaw=0 and the differential across a 2x2 is not zero. Keeping
        # the baseline lets the shear contribution be separated from the
        # pressure gradient it is confounded with.
        base_f = self.taxel_forces(surf)
        base_decode = self.decode_shear(base_f)

        out = []
        for a in angles:
            yaw = np.radians(a)
            # shear_step measures the object's motion about the sensor origin;
            # the twist is about the CONTACT point, so shift into that frame.
            obj_xy = (cy * yaw, -cx * yaw)
            ft, _ = self.surrogate.shear_step(surf, obj_xy, yaw)
            m = self.surrogate.base_moments(surf)
            f = self.taxel_forces(surf, moments=m)
            out.append({
                "yaw_deg": float(a),
                "forces": f,
                "baseline": base_f,
                "image": self.to_image(f),                    # geometric
                "image_scan": self.to_image(f, order="scan"),  # as streamed
                "shear_true": ft,
                # What firmware could compute from one frame: contaminated by
                # the pressure gradient, which it cannot see separately.
                "shear_decoded_raw": self.decode_shear(f),
                # The same, with the no-contact-shear baseline removed. Not
                # available on hardware without a reference frame, but it is
                # what isolates the shear channel for validation.
                "shear_decoded": self.decode_shear(f) - base_decode,
                "checkerness": self.checkerness(f),
                "checkerness_baseline": self.checkerness(base_f),
                # The shear-INDUCED checkering. The raw dipole above starts
                # high and dips before it climbs, because a ball's curvature
                # already makes a dipole in every off-centre cell, pointing
                # radially inward -- and the shear dipole is tangential, so it
                # first partly cancels that. Subtracting the baseline isolates
                # the part twisting is responsible for, and THAT rises
                # monotonically until the bumps slip.
                "checkerness_shear": float(np.mean(np.abs(
                    self.checkerness(f, per_bump=True)
                    - self.checkerness(base_f, per_bump=True)))),
                "total_n": float(f.sum()),
                "stuck": int(self.surrogate.stuck.sum()),
            })
        return out
