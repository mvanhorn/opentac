"""Bump geometry and the lumped-element stiffnesses derived from it; lengths in mm.

Geometry names mirror OpenTac-CAD's `pipeline/bump_sheet.py::BumpParams`, so a
design exported from OpenTac-CAD needs no translation layer.

A bump is not simulated as a deformable solid. It is a rigid frustum on a
compliant joint (press, lateral shear in two axes, tip about two axes), and
the stiffnesses of that joint follow from the slender-body formulas below.

The modulus of a printed bump in compression is not calibrated. K_CONTACT is a
solver contact-compliance estimate, not a material modulus.
"""

from dataclasses import dataclass
import math

#: Series contact compliance, N/m. MuJoCo's direct contact saturates near this
#: stiffness (measured 1.005, 1.018 and 1.090e6 N/m at three moduli spanning a
#: factor of 15); it stands in for the local flattening of a bump top.
K_CONTACT = 1.02e6

# Named estimates for PolyFlex TPU95, MPa. None is a compression measurement
# of a printed bump.
#   TPU95_REFERENCE  PolyFlex TPU95 technical data sheet V5.4, X-Y tensile, 100 % infill.
#   TPU95_SOLID      Gent correlation from Shore 95A; bulk elastomer, an upper bound.
#   TPU95_PRINTED    Central estimate for a printed bump, about one third of solid.
#   TPU95_INFILLED   Lower bound, for hollow or low-infill bumps.
TPU95_REFERENCE = 33.6
TPU95_SOLID = 43.8
TPU95_PRINTED = 15.0
TPU95_INFILLED = 3.0

#: The default effective modulus of BumpParams, MPa.
DEFAULT_E_EFF_MPA = TPU95_PRINTED

MM = 0.001


def shore_a_to_modulus(shore_a: float) -> float:
    """Shore A hardness to Young's modulus in MPa, by the Gent correlation.

    Shore 95A gives about 44 MPa. The correlation degrades above about 90A and
    describes solid elastomer, not a porous, anisotropic printed part; it
    serves as an order-of-magnitude reference for `e_eff_mpa`.
    """
    return 0.0981 * (56 + 7.62336 * shore_a) / (0.137505 * (254 - 2.54 * shore_a))


#: Plausible spans of the parameters that are estimated rather than measured.
#: The effective modulus is the one that matters most and is the least known.
ESTIMATE_RANGES = {
    "e_eff_mpa": (3.0, 44.0),   # solid TPU95 by Gent is about 44; printed bumps are softer
    "mu": (0.7, 1.2),           # depends on print orientation and surface finish
    "poisson": (0.40, 0.49),    # printed TPU with infill is compressible
}


@dataclass
class BumpParams:
    """One lattice's geometry, material and solver settings. Lengths in mm."""

    # geometry: the fields and defaults of OpenTac-CAD's bump_sheet.BumpParams
    taxel_pitch_mm: tuple = (4.2, 4.2)   # pitch between taxels, x and y
    bump_height_mm: float = 2.0
    bump_base_frac: float = 0.8          # base diameter as a fraction of the bump pitch (2 x taxel pitch)
    bump_top_ratio: float = 0.7          # top diameter over base diameter
    sheet_thickness_mm: float = 0.8
    # material: Polymaker PolyFlex TPU95 (Shore 95A), estimated
    e_eff_mpa: float = DEFAULT_E_EFF_MPA
    poisson: float = 0.45
    density: float = 1200.0              # kg/m^3
    mu: float = 1.0
    # numerics. omega_max caps each joint's natural frequency through its
    # armature; the direct contact solref of the MuJoCo scenes was tuned
    # against the effective mass this produces, so the two are coupled.
    omega_max: float = 3000.0            # rad/s
    zeta: float = 0.7                    # joint damping ratio
    damping_mode: str = "stable"         # see `damping`

    def __post_init__(self):
        positive = (self.e_eff_mpa, self.density, self.bump_height_mm,
                    self.sheet_thickness_mm, self.bump_base_frac, self.omega_max)
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Material, geometry and frequency parameters must be positive and finite")
        if len(self.taxel_pitch_mm) != 2 or any(not math.isfinite(v) or v <= 0 for v in self.taxel_pitch_mm):
            raise ValueError("taxel_pitch_mm must contain two positive finite pitches")
        if not math.isfinite(self.poisson) or not -1 < self.poisson < .5:
            raise ValueError("poisson must lie strictly between -1 and 0.5")
        if not math.isfinite(self.bump_top_ratio) or not 0 < self.bump_top_ratio <= 1:
            raise ValueError("bump_top_ratio must lie in (0, 1]")
        if any(not math.isfinite(v) or v < 0 for v in (self.mu, self.zeta)):
            raise ValueError("Friction and damping must be finite and nonnegative")

    # ---------- derived geometry, in metres ----------
    @property
    def bump_pitch(self):
        """Bumps sit on 2x2 blocks of taxels, so their pitch is twice the taxel pitch.

        This is OpenTac-CAD's `bump_centers_from_taxels` reduced to a regular
        lattice; a manifest gives the real centres, including partial blocks.
        """
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
        """The bump alone, N/m: k = E A_base t / H, exact for a linearly tapering section."""
        a, b = self.r_base
        area = math.pi * a * b
        e = self.e_eff_mpa * 1e6
        return e * area * self.bump_top_ratio / self.height

    @property
    def k_sheet(self):
        """The sheet under one bump in compression, N/m (estimated): E A_base / sheet thickness.

        The sheet rests on the sensor over its whole underside, so it is
        compressed rather than bent.
        """
        a, b = self.r_base
        e = self.e_eff_mpa * 1e6
        return e * math.pi * a * b / (self.sheet_thickness_mm * MM)

    @property
    def k_axial(self):
        """Bump and sheet in series, N/m: the joint stiffness of the model."""
        return 1.0 / (1.0 / self.k_bump_axial + 1.0 / self.k_sheet)

    @property
    def k_axial_effective(self):
        """What an object feels: bump, sheet and contact in series, N/m.

        This is the stiffness to compare with a force-displacement measurement
        of a printed bump, which includes the local flattening K_CONTACT stands for.
        """
        return 1.0 / (1.0 / self.k_axial + 1.0 / K_CONTACT)

    @property
    def shear_modulus(self):
        return self.e_eff_mpa * 1e6 / (2.0 * (1.0 + self.poisson))

    def _k_shear(self, second_moment):
        """N/m for lateral translation of the bump top with the top held parallel.

        Guided-cantilever bending (12EI/H^3) in series with shear (kappa G A / H).
        The shear term matters: the bumps are stubby (R/H of about 1.7 at the
        defaults), and bending alone would make them stiffer laterally than axially.
        """
        e = self.e_eff_mpa * 1e6
        a, b = self.r_mid
        area = math.pi * a * b
        c_bend = self.height ** 3 / (12.0 * e * second_moment)
        c_shear = self.height / (0.9 * self.shear_modulus * area)
        return 1.0 / (c_bend + c_shear)

    def _k_tip(self, second_moment):
        """N m/rad for rotation of the bump about its base, 4EI/H (fixed end)."""
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
        """Hinge about +X, so the bump bends in the Y-Z plane: I = pi a b^3 / 4, at the mid radius."""
        return self._k_tip(self.i_xx)

    @property
    def k_tip_y(self):
        return self._k_tip(self.i_yy)

    # ---------- solver conditioning ----------
    def armature(self, k):
        """Fictitious joint inertia that caps the bump's natural frequency at omega_max.

        It changes the dynamics, not the statics.
        """
        return k / self.omega_max ** 2

    @property
    def rot_inertia(self):
        """The bump's rotational inertia about its base, kg m^2 (a rod about its end, m H^2 / 3)."""
        return self.mass * self.height ** 2 / 3.0

    def damping(self, k, inertia=None):
        """Joint damping, N s/m for a slide and N m s/rad for a hinge.

        "stable" measures zeta against the armature-inflated mass, which keeps
        the integrated joint well damped; at the defaults the real bump is then
        strongly overdamped. "physical" measures zeta against the real mass or
        inertia, for impacts, rebounds and other transients.
        """
        if self.damping_mode == "physical":
            return 2.0 * self.zeta * math.sqrt(k * (self.mass if inertia is None else inertia))
        if self.damping_mode != "stable":
            raise ValueError(f"damping_mode must be 'stable' or 'physical', not {self.damping_mode!r}")
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
