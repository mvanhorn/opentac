"""A cheap surrogate for the bump lattice: an analytic Winkler foundation.

The lattice model already assumes bumps are mechanically independent -- that is
the Winkler assumption, and NOTES section 1 argues at length for why the slits
in OpenTac's bump sheet make it a good one. If it holds, then a bump's base force
does not depend on the solver at all:

    F_i = k_axial * max(0, H - z_surface(x_i, y_i))

...the object's surface height sampled at each bump, times the bump's stiffness.
No bodies, no contacts, no timestep. Microseconds instead of milliseconds.

This is the surrogate to train RL against. `scripts/surrogate_validation.py`
measures how far it drifts from the full MuJoCo lattice and how much faster it
is; read that before trusting it.

WHAT IT DOES NOT DO
  - No dynamics. It answers "given the object HERE, what does the sensor read",
    not "where does the object go". For a policy consuming tactile images under
    a scripted or externally-simulated motion, that is the whole question. For
    contact-rich manipulation where the pad's compliance changes the object's
    trajectory, it is not, and the full lattice is still needed.
  - No inter-bump coupling, by construction -- the assumption is the model.

Shear IS available (`shear_step`).  Shear is nothing but the bump's lateral
displacement, and for a rigid object that displacement is known analytically:
while the bump top sticks it moves with the object, and once Coulomb is
exceeded it saturates.  That needs one 2-vector of state per bump -- the stick
anchor -- not a solver.  It is the brush/bristle model from tyre mechanics.
"""

import numpy as np

from .params import BumpParams


# ---- analytic object surfaces -------------------------------------------
# Height of the object's UNDERSIDE above the pad plane, at (x, y).  Anything
# that can answer this can drive the surrogate; for a mesh, a downward raycast
# per bump does the same job at similar cost.

def sphere_surface(R, z_low, cx=0.0, cy=0.0):
    def f(x, y):
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        return z_low + R - np.sqrt(np.clip(R * R - r2, 0.0, None))
    return f


def plane_surface(z0, tilt_x=0.0, tilt_y=0.0):
    """SIGN CONVENTION: `tilt_x` is a rotation about +Y, matching the MJCF quat.

    A right-handed rotation about +Y carries +Z toward +X, so the plate's
    underside gets LOWER as x grows -- hence the minus.  Getting this backwards
    mirrors the pressure pattern about the pad centre, which leaves the engaged
    COUNT and the total force both correct and only shows up as a large,
    suspiciously constant per-bump error (63% at every pitch and modulus).
    That constancy is the tell: a real modelling failure would vary with them.
    """
    def f(x, y):
        return z0 - x * np.tan(tilt_x) - y * np.tan(tilt_y)
    return f


def cylinder_surface(R, z_low, axis="y", cx=0.0, cy=0.0):
    def f(x, y):
        d = (x - cx) if axis == "y" else (y - cy)
        return z_low + R - np.sqrt(np.clip(R * R - d * d, 0.0, None))
    return f


class Surrogate:
    """Winkler readout over a set of bump positions."""

    def __init__(self, p: BumpParams, positions):
        self.xy = np.asarray(positions, dtype=float)
        self.reparameterize(p)

    def reparameterize(self, p: BumpParams):
        """Point this surrogate at new parameters and refresh what it caches.

        `k` and `h` are cached rather than read from `p` on every call because
        `press_to_force` bisects sixty times and `k_axial_effective` chains
        through several properties. The cost of caching is that MUTATING a
        BumpParams after construction used to do nothing at all, silently --
        which is exactly what an RL loop randomising stiffness per episode
        would try. Go through here (or `TactileSensor.set_params`) and the
        cache cannot go stale.
        """
        self.p = p
        # k_axial_effective, not k_axial: a real object feels the bump, the
        # sheet AND the local surface flattening in series, and the full
        # lattice sim includes all three.  Using the joint stiffness alone
        # makes the surrogate systematically stiff by ~13%.
        self.k = p.k_axial_effective
        self.h = p.height
        return self

    def compressions(self, surface):
        z = surface(self.xy[:, 0], self.xy[:, 1])
        return np.clip(self.h - z, 0.0, None)

    def base_forces(self, surface):
        return self.k * self.compressions(surface)

    def press_to_force(self, make_surface, target_n, lo=-1e-3, hi=3e-3,
                       iters=60):
        """Find the object height giving `target_n` total.

        Plain bisection is fine here: an evaluation is ~10 microseconds, so
        there is nothing to gain from being clever, and bisection cannot
        overshoot the way the secant did in the full model.
        """
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            tot = self.base_forces(make_surface(mid)).sum()
            if tot < target_n:
                hi = mid
            else:
                lo = mid
        return mid, float(self.base_forces(make_surface(mid)).sum())

    def reset_shear(self):
        """Clear every bump's stick anchor.  Call before a new contact episode."""
        self.anchor = np.zeros((len(self.xy), 2))
        self.stuck = np.zeros(len(self.xy), dtype=bool)
        self.dt = np.zeros((len(self.xy), 2))

    def shear_step(self, surface, obj_xy, obj_yaw=0.0):
        """Advance the brush model to an object pose; return (Ft, delta_t).

        `obj_xy` is the object's in-plane translation and `obj_yaw` its rotation
        about the grip axis, both relative to where contact began.

        Per bump: while it sticks, its top travels with the object, so the
        lateral spring stretch is the object's motion at that point measured
        from the stick anchor.  Once k_shear*stretch exceeds mu*Fn the contact
        slips, the force saturates at mu*Fn, and the anchor is dragged forward
        so the stretch sits exactly on the Coulomb limit.  Releasing normal
        force drops the anchor entirely.

        This reproduces stick, breakaway and sliding without a contact solver
        -- the same three regimes Stage 0 measured on the full lattice.
        """
        if not hasattr(self, "anchor"):
            self.reset_shear()
        fn = self.base_forces(surface)
        r = self.xy
        # rigid-body motion of the object evaluated at each bump
        u = np.column_stack([obj_xy[0] - obj_yaw * r[:, 1],
                             obj_xy[1] + obj_yaw * r[:, 0]])
        live = fn > 1e-9
        self.anchor[~live] = u[~live]          # not touching: no stored stretch
        d = u - self.anchor
        ks = np.array([self.p.k_shear_x, self.p.k_shear_y])
        ft = d * ks
        mag = np.hypot(ft[:, 0], ft[:, 1])
        lim = self.p.mu * fn
        slip = live & (mag > lim) & (mag > 1e-12)
        scale = np.ones(len(r))
        scale[slip] = lim[slip] / mag[slip]
        ft = ft * scale[:, None]
        # drag the anchor so the stretch sits on the limit, not past it
        self.anchor[slip] = u[slip] - (ft[slip] / ks)
        self.stuck = live & ~slip
        self.dt = ft / ks
        return ft, self.dt

    def base_moments(self, surface):
        """(Mx, My) per bump, N*m -- the shear channel.

        Same guided-cantilever reconstruction the full model uses: a lateral
        stretch v puts k*v*H/2 into the base.  Requires shear_step to have run.
        """
        if not hasattr(self, "dt"):
            self.reset_shear()
        h = self.p.height
        my = self.p.k_shear_x * self.dt[:, 0] * h / 2.0
        mx = -self.p.k_shear_y * self.dt[:, 1] * h / 2.0
        return np.column_stack([mx, my])

    def taxel_forces(self, surface):
        """Per-taxel normal force, shape (n_bumps, 2, 2).

        The base moment comes from the pressure GRADIENT across the bump rather
        than from a solver state: a bump straddling a slope carries more load on
        its downhill side.  Estimated by differencing the surface across the
        bump's own footprint, which is the same first-moment argument the full
        model's base_moments makes, minus the shear term (no friction here).
        """
        f = self.base_forces(surface)
        ax = self.p.taxel_pitch_mm[0] * 1e-3 / 2.0
        ay = self.p.taxel_pitch_mm[1] * 1e-3 / 2.0
        x, y = self.xy[:, 0], self.xy[:, 1]
        out = np.zeros((len(self.xy), 2, 2))
        for iy, yo in enumerate((-ay, ay)):
            for ix, xo in enumerate((-ax, ax)):
                zc = np.clip(self.h - surface(x + xo, y + yo), 0.0, None)
                out[:, iy, ix] = self.k * zc
        # renormalise so the four quadrants carry the bump's own base force
        tot = out.sum(axis=(1, 2))
        good = tot > 1e-12
        out[good] *= (f[good] / tot[good])[:, None, None]
        out[~good] = 0.0
        # ...then add the friction-driven moment on top, which the pressure
        # gradient above cannot see: a stuck bump loaded sideways tips, and the
        # four taxels under it split the load asymmetrically.
        m = self.base_moments(surface)
        for iy, yo in enumerate((-ay, ay)):
            for ix, xo in enumerate((-ax, ax)):
                out[:, iy, ix] += (m[:, 1] * xo / (4 * ax * ax)
                                   - m[:, 0] * yo / (4 * ay * ay))
        return np.clip(out, 0.0, None)
