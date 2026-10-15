"""The full MuJoCo scene for one real flat sensor: its own bumps, and a ball.

Ported from TactileSim's `scripts/stage3_sim_video.py` (class SensorLattice),
without the rendering. Every bump is an elliptical frustum at its real position
from the manifest, on the five compliant DoFs of `mjcf._bump_body_mesh`, and a
ball on position servos is pressed into them and twisted.

Why the bumps are meshes here and the sheet never is: MuJoCo collides the
CONVEX HULL of a mesh. A single frustum is convex, so its hull is itself. The
hull of the whole bump sheet is a featureless slab with every bump erased.

Why the ball is a servoed body and not a mocap body: MuJoCo takes contact
velocity from qvel, and a mocap body has no DoFs, so its velocity is always
zero. Normal force would still work while friction never mobilised, silently
deleting the shear channel the bumps exist for.
"""
import mujoco
import numpy as np

from .loader import TactileSensor
from .mjcf import build_sensor_mjcf

AXES = ("bz", "bsx", "bsy", "bx", "by")
Z, SX, SY, TX, TY = range(5)


class SensorScene:
    """The full lattice for one real board, driven by a servoed ball."""

    def __init__(self, sensor: TactileSensor, ball_r_m, centre_xy=None,
                 sheet_stl=None, timestep=1e-4):
        self.s = sensor
        self.p = sensor.p
        self.ball_r = ball_r_m
        self.xml = build_sensor_mjcf(sensor.bump_xy, sensor.p, ball_r_m,
                                     centre_xy or sensor.centre(),
                                     sheet_stl=sheet_stl, timestep=timestep)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.n = sensor.n_bumps

        self.qadr = [
            tuple(self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{ax}_{i}")]
                for ax in AXES)
            for i in range(self.n)
        ]
        self.iq = [self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ("ix", "iy", "iz", "rz")]
        self.z0 = self.data.qpos[self.iq[2]]

    # -- readout, from the joints (not from contacts) ---------------------

    def base_forces(self):
        q = self.data.qpos
        return np.array([-self.p.k_axial * q[self.qadr[i][Z]]
                         for i in range(self.n)])

    def base_moments(self):
        """Same guided-cantilever reconstruction lattice.Lattice uses.

        Reading the tip hinges alone reports NO shear under a flat-ish
        contact: MuJoCo's joint springs are diagonal, so the hinges stay near
        zero while the shear slides carry everything. The k*v*H/2 term puts
        the missing coupling back in the READOUT, leaving the forces the
        solver sees untouched.
        """
        q, p, h = self.data.qpos, self.p, self.p.height
        out = np.zeros((self.n, 2))
        for i in range(self.n):
            a = self.qadr[i]
            out[i, 1] = p.k_shear_x * q[a[SX]] * h / 2.0 + p.k_tip_y * q[a[TY]]
            out[i, 0] = -p.k_shear_y * q[a[SY]] * h / 2.0 + p.k_tip_x * q[a[TX]]
        return out

    def taxel_forces(self):
        return self.s.taxel_forces_from_wrench(self.base_forces(),
                                               self.base_moments())

    def yaw(self):
        """The ball's twist about the contact normal, radians."""
        return float(self.data.qpos[self.iq[3]])

    # -- driving ----------------------------------------------------------

    def _hold(self, z, yaw, steps):
        self.data.ctrl[0] = 0.0
        self.data.ctrl[1] = 0.0
        self.data.ctrl[2] = z
        self.data.ctrl[3] = yaw
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def press_to_force(self, target_n, max_mm=1.6, steps=90, settle=900):
        """Ramp the ball down until the bumps carry `target_n`.

        Ramped, never teleported. MuJoCo resolves friction against relative
        VELOCITY, so a position jump transmits normal force but zero shear --
        the same trap lattice.Lattice.move_indenter exists to avoid.
        Returns (z, total_N). CHECK THE FORCE: if the ball runs out of travel
        (`max_mm`) before reaching the target, the total is below it.
        """
        z = self.z0
        dz = max_mm * 1e-3 / steps
        for _ in range(steps):
            z -= dz
            self._hold(z, 0.0, 12)
            if self.base_forces().sum() >= target_n:
                break
        self._hold(z, 0.0, settle)
        return z, float(self.base_forces().sum())

    def twist_to(self, z, yaw_rad, steps=10):
        self._hold(z, yaw_rad, steps)
