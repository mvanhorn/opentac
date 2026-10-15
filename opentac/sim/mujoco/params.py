"""Bump-lattice parameters and the lumped-element stiffnesses derived from them.

Geometry names mirror OpenTac's `pipeline/bump_sheet.py::BumpParams` on purpose, so
a design exported from OpenTac CAD can be dropped in without a translation
layer.  See NOTES.md, "Mapping to OpenTac".

The whole point of this model is that a bump is NOT simulated as a deformable
solid.  It is a rigid frustum on a 3-DoF compliant joint (press + tip about two
axes), and the stiffnesses of that joint come from slender-body formulas below.
Those formulas are the thing an offline single-bump FEM run would later replace
with a measured shape factor -- see NOTES.md, "Replacing the beam formulas".
"""

from dataclasses import dataclass
import math


# --- named material estimates for PolyFlex TPU95, all UNMEASURED ---
#
# Basis for each, so they can be argued with rather than trusted:
#
# TPU95_SOLID    Gent correlation from Shore 95A.  Bulk elastomer, small
#                strain.  An upper bound: no FDM part reaches its bulk modulus.
# TPU95_PRINTED  Central estimate, ~1/3 of solid.  A printed bump is small
#                enough (r = 1.6-3.4mm) that the slicer fills it mostly with
#                perimeter walls, so it is fairly dense -- but FDM TPU still
#                loses substantially to interlayer adhesion and porosity, and
#                it is anisotropic in Z, which is the direction bumps are
#                loaded.  Use this unless there is a reason not to.
# TPU95_INFILLED Lower bound, for deliberately hollow or low-infill bumps.
#
# Reconciliation with the observed part: at the 4.2mm default pitch, solid
# TPU95 shears only 30um at 5N, which contradicts a part that visibly deforms.
# At the REAL 2-3mm pitch it shears 62-155um (3-8% strain), which does not.
# The earlier "10x too stiff" conclusion was an artefact of evaluating at a
# pitch outside the real design range -- see NOTES.md section 4.2.
# MuJoCo's contact constraint saturates near this stiffness whatever solref
# asks for -- requesting 1e8 and above goes unstable (>100% of analytic, huge
# jitter) before it ever gets stiffer.  Identified by measuring the series
# softening at three moduli spanning 15x: 1.005, 1.018 and 1.090 e6 N/m.  That
# consistency is what makes it a constant worth modelling rather than a bug.
#
# It is not unphysical.  A real bump pressing on a hard object DOES flatten
# locally, and 5N at 1e6 N/m is 5um of flattening, which is reasonable for a
# rubber surface.  So this stands in for exactly the local dimpling listed as
# "not modelled" in NOTES section 6 -- crudely, as a linear spring, but present.
K_CONTACT = 1.02e6

TPU95_SOLID = 43.8
TPU95_PRINTED = 15.0
TPU95_INFILLED = 3.0


def shore_a_to_modulus(shore_a: float) -> float:
    """Shore A hardness -> Young's modulus in MPa, via the Gent correlation.

    Shore 95A gives ~44 MPa.  Treat that as an order-of-magnitude anchor only:
    the correlation is known to degrade above ~90A, it describes SOLID
    elastomer rather than an FDM part, and printed TPU is anisotropic and
    porous.  It is here to make the gap against the fitted `e_eff_mpa` explicit
    rather than to be used directly.
    """
    return 0.0981 * (56 + 7.62336 * shore_a) / (0.137505 * (254 - 2.54 * shore_a))

MM = 1e-3


#: What is MEASURED and what is a GUESS, with the guesses' plausible spans.
#:
#: Shipped as data because a user cannot be expected to read NOTES.md section
#: 4.5 to find out that the single most important number in the model is
#: unmeasured. Choosing distributions and schedules over these is the RL
#: designer's job; stating the honest range is ours.
#:
#: e_eff_mpa is the one that matters. Printed TPU came out ~10x below solid
#: TPU95 and NOTES 4.1 says plainly that this is unexplained -- so the span is
#: wide because nobody knows, not because it varies that much. One afternoon
#: with a known mass and a dial indicator would collapse it.
ESTIMATE_RANGES = {
    "e_eff_mpa": (3.0, 44.0),     # ESTIMATE. Solid 95A by Gent is ~44; printed
                                  # measured ~10x lower, unexplained.
    "mu": (0.7, 1.2),             # ESTIMATE. Moves with print orientation and
                                  # surface finish; unmeasured on real prints.
    "poisson": (0.40, 0.49),      # ESTIMATE. Printed TPU with infill is
                                  # compressible, so below the 0.5 of a solid.
}

@dataclass
class BumpParams:
    """One lattice's geometry, material and solver settings. Lengths in mm."""

    # --- geometry: same fields, same defaults as OpenTac bump_sheet.BumpParams ---
    taxel_pitch_mm: tuple = (4.2, 4.2)   # OpenTac pitch_x / pitch_y between taxels
    bump_height_mm: float = 2.0
    bump_base_frac: float = 0.8          # of the bump pitch (= 2 x taxel pitch)
    bump_top_ratio: float = 0.7          # top/base diameter
    sheet_thickness_mm: float = 0.8

    # --- material: Polymaker PolyFlex TPU95 (Shore 95A) ---
    # ESTIMATED, not measured.  Named estimates with their basis live in
    # TPU95_* above; e_eff_mpa defaults to the central one.  Sweep the range.
    #
    # This matters less than it looks: Stage 2 found pitch changes the grip
    # result by ~3.5x while E_eff moves it weakly and non-monotonically, so the
    # uncertainty here does not propagate far into the conclusions that matter.
    e_eff_mpa: float = TPU95_PRINTED
    poisson: float = 0.45                # printed TPU with infill is compressible
    density: float = 1200.0              # kg/m^3, printed TPU
    mu: float = 1.0                      # TPU is grippy

    # --- numerics (see NOTES.md, "Why armature is not cheating") ---
    # DO NOT raise this without re-tuning the contact solref in mjcf.py.
    #
    # The obvious optimisation is tempting and does not work. armature is
    # k/omega_max^2, so the fictitious mass is (omega_true/omega_max)^2 times
    # the real bump -- 61x here, 260x at the 15 MPa default. Raising omega_max
    # shrinks that quadratically at NO cost in timestep, and on an impact test
    # it measurably helps: at dt=1e-4, omega_max=8000 cuts the fictitious mass
    # 61x -> 8.5x and the restitution error 11.2% -> 2.9%. There is even a
    # clean optimum at omega_max*dt ~ 0.8, past which the integrator can no
    # longer resolve the joint and degrades SILENTLY rather than blowing up
    # (e = 0.52 at omega*dt = 2.3, 0.08 at 4.0 -- no warning, no NaN).
    #
    # But it breaks Stage 0. Measured at omega_max=8000: static stiffness fell
    # to 56.6% of analytic from 99.7%, and joint-vs-contact agreement to 76.9%
    # from 100%. The contact solref (-3e7 -10000, see mjcf.py) was tuned
    # against the effective mass this armature produces; a lighter body makes
    # that same contact damping enormous in relative terms and the press goes
    # soft. So armature and solref are COUPLED, and 3000 is load-bearing for
    # the static calibration -- which is the one thing in this repo that is
    # actually validated against a closed form.
    #
    # Raising it is still the right move for dynamic work, but it is a
    # re-tuning job (solref, then re-run Stage 0 to 100% of analytic across
    # E_eff 3-30 MPa), not a one-line change.
    omega_max: float = 3000.0            # rad/s cap on bump natural frequency
    zeta: float = 0.7                    # joint damping ratio
    # Which mass zeta is measured against -- see `damping`. "stable" keeps the
    # integrated joint well damped and is what every validated result used;
    # "physical" damps the REAL bump correctly and is what a dynamic test
    # (impact, rebound, transient) needs.
    damping_mode: str = "stable"

    # ---------- derived geometry, all in metres ----------
    @property
    def bump_pitch(self):
        """Bumps sit on a 2x2 block of taxels, so their pitch is twice the taxel
        pitch.  This is `bump_centers_from_taxels` in OpenTac, reduced to a regular
        lattice: the real one also handles partial blocks and irregular edges."""
        return (2.0 * self.taxel_pitch_mm[0] * MM, 2.0 * self.taxel_pitch_mm[1] * MM)

    @property
    def r_base(self):
        px, py = self.bump_pitch
        return (self.bump_base_frac * px / 2.0, self.bump_base_frac * py / 2.0)

    @property
    def r_top(self):
        a, b = self.r_base
        return (self.bump_top_ratio * a, self.bump_top_ratio * b)

    @property
    def r_mid(self):
        a, b = self.r_base
        t = self.bump_top_ratio
        return (a * (1.0 + t) / 2.0, b * (1.0 + t) / 2.0)

    @property
    def height(self):
        return self.bump_height_mm * MM

    @property
    def volume(self):
        """Elliptical frustum, both semi-axes tapering by the same ratio."""
        a, b = self.r_base
        t = self.bump_top_ratio
        return math.pi * a * b * self.height * (1.0 + t + t * t) / 3.0

    @property
    def mass(self):
        return self.density * self.volume

    # ---------- lumped stiffnesses ----------
    @property
    def k_bump_axial(self):
        """The bump alone, N/m.  Exact for a linearly tapering section.

        k = E / integral(dz / A(z)).  With A(z) = A_base * (1-(1-t)z/H)^2 the
        integral evaluates to H / (A_base * t), so k = E * A_base * t / H.
        Sanity: t=1 gives the cylinder result EA/H, t->0 gives k->0.
        """
        a, b = self.r_base
        area = math.pi * a * b
        e = self.e_eff_mpa * 1e6
        return e * area * self.bump_top_ratio / self.height

    @property
    def k_sheet(self):
        """The sheet under one bump, N/m -- ESTIMATED, previously omitted.

        The 0.8mm sheet is not rigid: it compresses under every bump, in series
        with the bump itself.  Modelling it as a rigid plate made axial
        stiffness too high independent of any modulus error.

        Estimate: the sheet rests on the sensor/PCB, so it is supported over
        its whole underside and deforms in COMPRESSION rather than bending
        (bending would be far softer, and would apply if the sheet spanned a
        gap).  Treated as a short column of base area over the sheet thickness,
        E*A/t.  That comes out ~3.6x the bump's own stiffness, so it softens
        the stack by ~22% -- real, but not the missing order of magnitude.
        """
        a, b = self.r_base
        e = self.e_eff_mpa * 1e6
        return e * math.pi * a * b / (self.sheet_thickness_mm * MM)

    @property
    def k_axial(self):
        """Bump and sheet in series, N/m -- the JOINT stiffness in the model.

        Note this is the spring the solver carries and the one `base_forces`
        reads back through, so force readout is unaffected by the contact term
        below (series springs carry equal force).
        """
        return 1.0 / (1.0 / self.k_bump_axial + 1.0 / self.k_sheet)

    @property
    def k_axial_effective(self):
        """What an OBJECT actually feels: bump + sheet + contact, all in series.

        This, not `k_axial`, is the stiffness to compare against a real
        force-displacement measurement of a printed bump, because a real
        measurement includes the local flattening that K_CONTACT stands for.
        """
        return 1.0 / (1.0 / self.k_axial + 1.0 / K_CONTACT)

    @property
    def shear_modulus(self):
        return self.e_eff_mpa * 1e6 / (2.0 * (1.0 + self.poisson))

    def _k_shear(self, second_moment):
        """N/m for LATERAL translation of the bump top, top held parallel.

        This is the mode a flat object drives, and it is the important one for
        grasping.  Stage 0 found it the hard way: with only press + tip DoFs, a
        flat punch dragged across a bump produced full Coulomb friction and
        almost no response, because a rigid flat face covering the whole bump
        top physically PREVENTS tipping.  The bump has to shear over instead.

        Guided-cantilever bending (12EI/H^3) IN SERIES with shear (kappa*G*A/H).
        The shear term is not optional here: these bumps are stubby -- OpenTac's
        defaults give R/H = 1.68 -- and pure bending would put lateral stiffness
        at 235 kN/m, 6x the axial stiffness, which is nonsense for a rubber pad.
        Including shear brings it to ~17 kN/m, about 0.46x axial, which is the
        expected "easier to shear than to compress" behaviour.

        Euler-Bernoulli is frankly marginal at H/R = 0.6.  This is the single
        biggest reason to do the one-off single-bump FEM extraction described
        in NOTES.md rather than trusting these numbers.
        """
        e = self.e_eff_mpa * 1e6
        a, b = self.r_mid
        area = math.pi * a * b
        c_bend = self.height ** 3 / (12.0 * e * second_moment)
        c_shear = self.height / (0.9 * self.shear_modulus * area)
        return 1.0 / (c_bend + c_shear)

    def _k_tip(self, second_moment):
        """N*m/rad for rotation of the bump about its base, 4EI/H.

        The fixed-end value, chosen as the diagonal partner of the guided
        lateral stiffness above.  A real beam couples these two DoFs through an
        off-diagonal -6EI/H^2 term that MuJoCo cannot express as a joint spring;
        the consequence and the workaround are in sim.py::base_moments.
        """
        e = self.e_eff_mpa * 1e6
        return 4.0 * e * second_moment / self.height

    @property
    def i_xx(self):
        a, b = self.r_mid
        return math.pi * a * b ** 3 / 4.0

    @property
    def i_yy(self):
        a, b = self.r_mid
        return math.pi * a ** 3 * b / 4.0

    @property
    def k_shear_x(self):
        """Lateral stiffness for motion along +X (resisted by I_yy)."""
        return self._k_shear(self.i_yy)

    @property
    def k_shear_y(self):
        return self._k_shear(self.i_xx)

    @property
    def k_tip_x(self):
        """Hinge about +X, so the bump bends in the Y-Z plane: I = pi*a*b^3/4.

        Evaluated at the MID radius rather than the base, a crude stand-in for
        integrating the taper.  Elliptical because OpenTac sizes each bump from its
        own 2x2 block, so a 4x2mm taxel grid gets a bump twice as wide as deep
        and the two tip axes are genuinely different springs.
        """
        return self._k_tip(self.i_xx)

    @property
    def k_tip_y(self):
        return self._k_tip(self.i_yy)

    # ---------- solver conditioning ----------
    def armature(self, k):
        """Fictitious joint inertia that caps the bump's natural frequency.

        A real bump masses ~70 mg against a stiffness of ~1e5 N/m, which is a
        natural frequency near 1e5 rad/s and would need a ~1e-6 s timestep.
        Armature adds inertia to the joint's diagonal so omega = sqrt(k/m_eff)
        lands at omega_max instead.  It changes dynamics, NOT statics: every
        quasi-static quantity this prototype measures is untouched.
        """
        return k / (self.omega_max ** 2)

    @property
    def rot_inertia(self):
        """The bump's real rotational inertia about its base, kg*m^2.

        Estimated as a rod about its end, m*H^2/3. The bump is a taper rather
        than a rod so this is approximate -- but it is the REAL inertia to
        within a factor near one, where the armature is out by ~1e3.
        """
        return self.mass * self.height ** 2 / 3.0

    def damping(self, k, inertia=None):
        """Joint damping, N*s/m for a slide and N*m*s/rad for a hinge.

        `damping_mode` decides which mass zeta is measured against, and that
        matters far more than it looks:

        "stable" (default) -- c = 2*zeta*k/omega_max, i.e. zeta against the
            ARMATURE-inflated mass. Keeps the joint MuJoCo actually integrates
            near critically damped, which is what the armature trick needs to
            stay well behaved. But the REAL bump then sees something else
            entirely: at omega_max=3000 the intended zeta=0.7 lands as
            zeta=11.3 on the real 22 mg bump -- 16x overdamped.

        "physical" -- c = 2*zeta*sqrt(k*inertia) against the REAL inertia, so
            the bump is damped the way the material is meant to be whatever
            armature is set to.

        Measured, 50 g ball dropped 20 mm: "stable" gives a coefficient of
        restitution of 0.52 against a converged 0.98. Switching to "physical"
        alone recovers 0.87 at NO cost in timestep. The rest of the gap is the
        armature MASS, which only a higher omega_max (and a smaller timestep)
        removes.

        Default stays "stable" so every previously validated result is
        untouched. Anything measuring an impact, a rebound or a transient
        should ask for "physical". Past roughly zeta=0.2 on the integrated
        joint the bumps ring after separation, so this is not a knob to turn
        arbitrarily far.
        """
        if self.damping_mode == "physical":
            return 2.0 * self.zeta * math.sqrt(
                k * (self.mass if inertia is None else inertia))
        if self.damping_mode != "stable":
            raise ValueError(f"damping_mode must be 'stable' or 'physical', "
                             f"not {self.damping_mode!r}")
        return 2.0 * self.zeta * k / self.omega_max

    def summary(self):
        a, b = self.r_base
        return {
            "bump_pitch_mm": tuple(round(v / MM, 2) for v in self.bump_pitch),
            "r_base_mm": (round(a / MM, 3), round(b / MM, 3)),
            "r_top_mm": tuple(round(v / MM, 3) for v in self.r_top),
            "height_mm": round(self.height / MM, 3),
            "mass_mg": round(self.mass * 1e6, 2),
            "E_eff_MPa": self.e_eff_mpa,
            "k_bump_axial_N_per_m": round(self.k_bump_axial, 1),
            "k_sheet_N_per_m": round(self.k_sheet, 1),
            "k_axial_N_per_m": round(self.k_axial, 1),
            "k_axial_effective_N_per_m": round(self.k_axial_effective, 1),
            "k_shear_x_N_per_m": round(self.k_shear_x, 1),
            "shear_over_axial": round(self.k_shear_x / self.k_axial, 3),
            "k_tip_x_Nm_per_rad": round(self.k_tip_x, 5),
            "k_tip_y_Nm_per_rad": round(self.k_tip_y, 5),
            "axial_defl_at_1N_um": round(1.0 / self.k_axial_effective * 1e6, 3),
        }
