"""Quasi-static bump loads (an analytic Winkler foundation) and a Coulomb brush model for shear.

Bumps are mechanically independent: the slits of OpenTac-CAD's bump sheet cut
each bump free of its neighbours. A bump's base force then follows from the
object's surface height at the bump alone:

    F_i = k * max(0, H - z_surface(x_i, y_i))

with k the stiffness an object feels (`BumpParams.k_axial_effective`). No
bodies, contacts or time step are involved; one evaluation takes microseconds.

The surrogate has no dynamics: it answers what the sensor reads for an object
at a given pose, not where the object goes. Shear is the lateral displacement
of each bump top, which for a rigid object follows from its motion while the
bump sticks and saturates at the Coulomb limit once it slips (`shear_step`).
"""

import numpy as np

from .params import BumpParams


# ---- analytic object surfaces -------------------------------------------
# Height of the object's underside above the pad plane at (x, y), in metres,
# with z = 0 on the sheet and the bump tops at the bump height. A point the
# object does not cover returns +inf, so that it is never in contact.

def sphere_surface(R, z_low, cx=0.0, cy=0.0):
    """Lower hemisphere of a sphere of radius R whose lowest point is at height z_low."""

    def f(x, y):
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        return np.where(r2 <= R * R,
                        z_low + R - np.sqrt(np.clip(R * R - r2, 0.0, None)), np.inf)
    return f


def plane_surface(z0, tilt_x=0.0, tilt_y=0.0):
    """A plane through height z0 at the origin.

    Sign convention: `tilt_x` is a rotation about +Y, as in the MJCF quaternion.
    A right-handed rotation about +Y carries +Z toward +X, so the underside is
    lower where x is larger.
    """

    def f(x, y):
        return z0 - x * np.tan(tilt_x) - y * np.tan(tilt_y)
    return f


def cylinder_surface(R, z_low, axis="y", cx=0.0, cy=0.0):
    """Lower half of a cylinder of radius R lying on the skin, its axis along `axis` ("y" or "x")."""

    def f(x, y):
        d = (x - cx) if axis == "y" else (y - cy)
        return np.where(d * d <= R * R,
                        z_low + R - np.sqrt(np.clip(R * R - d * d, 0.0, None)), np.inf)
    return f


class Surrogate:
    """Winkler readout over a set of bump positions."""

    def __init__(self, p: BumpParams, positions):
        self.xy = np.asarray(positions, dtype=float)
        self.reparameterize(p)

    def reparameterize(self, p: BumpParams):
        """Point this surrogate at new parameters and refresh what it caches.

        `k` and `h` are cached because the force solve evaluates them many
        times; going through this method (or `TactileSensor.set_params`)
        keeps the cache consistent with the parameters.
        """
        self.p = p
        self.k = p.k_axial_effective
        self.h = p.height
        return self

    def compressions(self, surface):
        z = surface(self.xy[:, 0], self.xy[:, 1])
        return np.clip(self.h - z, 0.0, None)

    def base_forces(self, surface):
        return self.k * self.compressions(surface)

    def press_to_force(self, make_surface, target_n, lo=-0.001, hi=0.003, iters=60, force_fn=None):
        """The object height at which the total force is `target_n`; returns (z, total).

        `make_surface(z)` gives the object's surface at height z. `force_fn`
        maps a surface to the forces that sum to the total (default: the bump
        base forces). The search interval is widened until it brackets the
        target, then bisected; a target that no height reaches raises.
        """
        if not np.isfinite(target_n) or target_n < 0 or iters < 1:
            raise ValueError("target_n must be finite and nonnegative, and iters positive")
        lo, hi = min(lo, self.h - 0.001), max(hi, self.h + 0.001)

        def total(z):
            force = float((force_fn or self.base_forces)(make_surface(z)).sum())
            if not np.isfinite(force):
                raise ValueError("Probe surface produces non-finite force")
            return force
        for _ in range(32):
            f_lo, f_hi = total(lo), total(hi)
            if f_hi <= target_n <= f_lo:
                break
            width = hi - lo
            if f_hi > target_n:
                hi += 2 * width
            if f_lo < target_n:
                lo -= 2 * width
        else:
            raise ValueError("Requested force cannot be bracketed on this sensor")
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            force = total(mid)
            if abs(force - target_n) <= max(1e-9, 1e-8 * target_n):
                return mid, force
            if force < target_n:
                hi = mid
            else:
                lo = mid
        raise RuntimeError("Force solve did not converge to the requested load")

    def reset_shear(self):
        """Clear every bump's stick anchor. Call before a new contact episode."""
        self.anchor = np.zeros((len(self.xy), 2))
        self.stuck = np.zeros(len(self.xy), dtype=bool)
        self.delta_t = np.zeros((len(self.xy), 2))

    def shear_step(self, surface, obj_xy, obj_yaw=0.0):
        """Advance the brush model to an object pose; returns (Ft, delta_t).

        `obj_xy` and `obj_yaw` are the object's in-plane translation and its
        rotation about the contact normal, relative to where contact began.
        While a bump sticks, its top travels with the object; once k_shear
        times the stretch exceeds mu Fn the bump slips, the force saturates at
        mu Fn, and the stick anchor follows so that the stretch stays on the
        Coulomb limit. A bump that loses contact drops its anchor.
        """
        if not hasattr(self, "anchor"):
            self.reset_shear()
        fn = self.base_forces(surface)
        r = self.xy
        u = np.column_stack([obj_xy[0] - obj_yaw * r[:, 1], obj_xy[1] + obj_yaw * r[:, 0]])
        live = fn > 1e-9
        self.anchor[~live] = u[~live]
        d = u - self.anchor
        ks = np.array([self.p.k_shear_x, self.p.k_shear_y])
        ft = d * ks
        mag = np.hypot(ft[:, 0], ft[:, 1])
        lim = self.p.mu * fn
        slip = live & (mag > lim) & (mag > 1e-12)
        scale = np.ones(len(r))
        scale[slip] = lim[slip] / mag[slip]
        ft = ft * scale[:, None]
        self.anchor[slip] = u[slip] - ft[slip] / ks
        self.stuck = live & ~slip
        self.delta_t = ft / ks
        return ft, self.delta_t

    def base_moments(self, surface=None):
        """(Mx, My) per bump, N m: the shear channel.

        A guided cantilever displaced laterally by v puts k v H / 2 into its
        base. Uses the state of the latest `shear_step`; `surface` is accepted
        for the call form of `TactileSensor` and not used.
        """
        if not hasattr(self, "delta_t"):
            self.reset_shear()
        h = self.p.height
        my = self.p.k_shear_x * self.delta_t[:, 0] * h / 2.0
        mx = -self.p.k_shear_y * self.delta_t[:, 1] * h / 2.0
        return np.column_stack([mx, my])

    def taxel_forces(self, surface):
        """Per-taxel normal force on an idealised full 2x2 block, shape (n_bumps, 2, 2), indexed [bump][iy][ix].

        The four taxels sit at +/- half a taxel pitch from the bump centre.
        The base force is split by the surface height at each taxel and the
        friction-driven base moment is added as a couple. For a real board,
        with partial blocks, use `TactileSensor.taxel_forces`.
        """
        f = self.base_forces(surface)
        ax = self.p.taxel_pitch_mm[0] * 1e-3 / 2.0
        ay = self.p.taxel_pitch_mm[1] * 1e-3 / 2.0
        x, y = self.xy[:, 0], self.xy[:, 1]
        out = np.zeros((len(self.xy), 2, 2))
        for iy, yo in enumerate((-ay, ay)):
            for ix, xo in enumerate((-ax, ax)):
                out[:, iy, ix] = self.k * np.clip(self.h - surface(x + xo, y + yo), 0.0, None)
        tot = out.sum(axis=(1, 2))
        good = tot > 1e-12
        out[good] *= (f[good] / tot[good])[:, None, None]
        out[~good] = 0.0
        m = self.base_moments()
        for iy, yo in enumerate((-ay, ay)):
            for ix, xo in enumerate((-ax, ax)):
                out[:, iy, ix] += m[:, 1] * xo / (4 * ax * ax) - m[:, 0] * yo / (4 * ay * ay)
        return np.clip(out, 0.0, None)
