"""Stage 2 -- two bump pads on a parallel jaw, and a slip test.

The milestone this whole prototype exists to reach: does a lattice of rigid
bumps on compliant joints actually GRIP better than a rigid pad?

Layout: both jaws squeeze along Z, the object is a cylinder whose axis lies
along Y, and the pull is along Y too -- along the cylinder's own axis. That is
deliberate. Pulled across its axis a cylinder ROLLS, and rolling is not slipping;
the measurement would be of the rolling resistance and would look like a wildly
compliance-dependent grip strength for the wrong reason. Pulled along its axis
it can only slide.

Gravity is off. The only tangential load is the applied pull, so the slip force
is read directly rather than inferred against the object's weight.
"""

import numpy as np
import mujoco

from .params import BumpParams, MM
from .mjcf import bump_bodies, bump_mesh, lattice_positions, IND_KP, IND_ARM, IND_KV


def _pad(p, positions, sign, tag, plate_xy):
    # The upper jaw's bumps point down: a mirrored mesh, not a rotated body,
    # so that both jaws keep the same joint axes.
    bodies = bump_bodies(positions, p, mesh="bumpmesh" if sign > 0 else "bumpmesh_down", tag=tag)
    t = p.sheet_thickness_mm * MM
    return f"""
      <geom name="{tag}plate" type="box" pos="0 0 {-sign * t / 2:.6f}"
            size="{plate_xy[0]:.6f} {plate_xy[1]:.6f} {t / 2:.6f}"
            rgba="0.2 0.2 0.25 1" contype="1" conaffinity="4"/>{bodies}"""


def build_gripper(p: BumpParams, pad_half_mm=15.0, obj_radius_mm=12.0,
                  obj_len_mm=15.0, bumps=True, obj="sphere", obj_offset_mm=(0.0, 0.0),
                  timestep=1e-4):
    """Two facing pads with a cylinder between them.

    `bumps=False` gives the RIGID-PLATE baseline: same pads, same materials,
    same object, no compliant elements. Without that line on the plot there is
    nothing to compare a slip force against.
    """
    if bumps:
        px, py = p.bump_pitch
        nx = max(1, int(2 * pad_half_mm * MM / px))
        ny = max(1, int(2 * pad_half_mm * MM / py))
        positions = lattice_positions(nx, ny, p)
        h = p.height
    else:
        positions, nx, ny, h = [], 0, 0, 0.0

    half = pad_half_mm * MM
    R = obj_radius_mm * MM
    # Object centred between the jaws; the upper jaw starts just touching, so
    # its slide joint reads squeeze depth directly.
    z_obj = h + R
    z_jaw = z_obj + R + h

    if obj == "sphere":
        obj_type, obj_size, obj_quat = "sphere", f"{R:.6f}", ""
    else:
        obj_type = "cylinder"
        obj_size = f"{R:.6f} {obj_len_mm * MM:.6f}"
        obj_quat = ' quat="0.7071068 0.7071068 0 0"'

    lower = _pad(p, positions, +1, "l", (half, half)) if bumps else _pad(
        p, [], +1, "l", (half, half))
    upper = _pad(p, positions, -1, "u", (half, half)) if bumps else _pad(
        p, [], -1, "u", (half, half))

    xml = f"""<mujoco model="bump_gripper">
  <compiler angle="radian"/>
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic"
          noslip_iterations="10" gravity="0 0 0"/>
  <default>
    <geom solref="-3e7 -10000" solimp="0.999 0.9999 0.00001" condim="3"/>
  </default>
  <asset>{bump_mesh(p)}{bump_mesh(p, name="bumpmesh_down", sign=-1)}</asset>
  <worldbody>
    <light pos="0 0 0.3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <body name="lower" pos="0 0 0">{lower}
    </body>
    <body name="upper" pos="0 0 {z_jaw:.6f}">
      <joint name="jaw" type="slide" axis="0 0 1" armature="{IND_ARM}"/>{upper}
    </body>
    <!-- Offsetting the object off the lattice centre is not cosmetic.  A
         sphere centred on a square lattice sits symmetrically on exactly 4
         bumps at EVERY pitch, so a pitch sweep silently becomes a comparison
         of moment arms at fixed engagement rather than of engagement itself.
         Averaging over a few offsets breaks that. -->
    <body name="obj" pos="{obj_offset_mm[0]*MM:.6f} {obj_offset_mm[1]*MM:.6f} {z_obj:.6f}">
      <!-- Tiny damping keeps the free body from drifting before the jaws meet.
           The pull is ramped slowly enough that velocity is ~0 right up to
           slip onset, so this contributes essentially nothing to the measured
           slip force. -->
      <joint type="free" damping="0.02"/>
      <geom name="objg" type="{obj_type}" size="{obj_size}"{obj_quat} mass="0.1"
            friction="{p.mu} 0.005 0.0001" rgba="0.85 0.75 0.3 1"
            contype="4" conaffinity="1"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="jaw" kp="{IND_KP}" kv="{IND_KV}"/>
  </actuator>
</mujoco>
"""
    return xml, positions, z_jaw


class Gripper:
    def __init__(self, p: BumpParams, bumps=True, **kw):
        self._kw = kw
        self.p = p
        self.bumps = bumps
        self.xml, self.positions, self.z_jaw = build_gripper(p, bumps=bumps, **kw)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.n = len(self.positions)
        self.obj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self.obj_qadr = self.model.jnt_qposadr[
            self.model.body_jntadr[self.obj]]
        self.lq = [[self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, f"l{ax}_{i}")]
            for ax in ("z", "sx", "sy", "x", "y")] for i in range(self.n)]

    # ---- readout ----------------------------------------------------------
    def pad_forces(self):
        """Base normal force on each bump of the LOWER pad, N."""
        if not self.n:
            return np.zeros(0)
        q = self.data.qpos
        return np.array([-self.p.k_axial * q[a[0]] for a in self.lq])

    def grip_force(self):
        """Normal force clamping the object, N.

        Read from the jaw actuator rather than by summing bump deflections, so
        the rigid-plate baseline -- which has no bumps to sum -- is measured the
        same way as the lattice.
        """
        return abs(float(self.data.actuator_force[0]))

    def obj_pos(self):
        return np.array(self.data.qpos[self.obj_qadr:self.obj_qadr + 3])

    def settle(self, steps=400):
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    # ---- driving ----------------------------------------------------------
    def close_to_force(self, target_n, tol=0.05, iters=16, steps=400):
        """Squeeze until the jaw carries `target_n`.  Safeguarded secant.

        Same trap as press_to_force: force climbs faster than linearly with jaw
        travel as more bumps engage, so an unbracketed step overshoots and
        cannot recover.  Returns (squeeze_mm, force) -- CHECK THE FORCE.
        """
        lo, hi = 0.0, None
        d, tot = 0.05, 0.0
        for _ in range(iters):
            d = min(max(d, 1e-4), 3.0)
            self.data.ctrl[0] = -d * 1e-3
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)
            self.settle(250)
            tot = self.grip_force()
            if abs(tot - target_n) < tol * target_n:
                return d, tot
            if tot < target_n:
                lo = max(lo, d)
            else:
                hi = d if hi is None else min(hi, d)
            nxt = d * 3.0 if tot < 1e-6 else d * (target_n / tot) ** 0.7
            if hi is not None and not (lo < nxt < hi):
                nxt = 0.5 * (lo + hi)
            elif hi is None and nxt <= lo:
                nxt = lo * 1.5 + 1e-3
            d = nxt
        return d, tot

    def inertial_floor(self, **kw):
        """Breakaway torque with friction switched off -- what the ramp reads
        from the object's own inertia and joint damping alone.

        Measured, not estimated.  Any twist result must be compared against
        this floor before it means anything.
        """
        p2 = BumpParams(**{**self.p.__dict__, "mu": 1e-4})
        g = Gripper(p2, bumps=self.bumps, **self._kw)
        g.close_to_force(15.0)
        return g.twist_until_slip(**kw)[0]

    def obj_yaw(self):
        """Object rotation about the grip axis, radians."""
        a = self.obj_qadr + 3
        w, x, y, z = self.data.qpos[a:a + 4]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def twist_until_slip(self, t_max=0.20, ramp_steps=30000, slip_deg=30.0):
        """Ramp a torque about the grip axis until the object breaks away.

        THIS, not an axial pull, is the test that can distinguish a lattice from
        a rigid plate.  Coulomb friction is AREA-INDEPENDENT: total slip force
        is mu*N however the contact is distributed, so a straight pull gives the
        same answer for both -- measured 31.0 N rigid against 28.6 N lattice,
        both at 2*mu*N, a correct result and a useless one.

        Torsional resistance is different: it is the friction force integrated
        against its MOMENT ARM, so it depends on how far the contact is spread.
        A rigid plate on a sphere touches at a point and resists almost no
        twist; a lattice engaging bumps across an area resists a lot.  Same
        Coulomb model, completely different answer.

        (The other real mechanism, that rubber friction falls with pressure so
        spreading load raises effective mu, cannot be shown here at all --
        MuJoCo's mu is a constant.  See NOTES.md.)

        `slip_deg` is 30, not 5.  Bumps shear ELASTICALLY by hundreds of microns
        before anything breaks loose, and at a ~3mm moment arm 300um is already
        6 degrees of object rotation.  A 5 degree threshold therefore fires
        inside the elastic range for soft bumps and outside it for stiff ones,
        which does not just add noise -- it INVERTS the ranking and makes stiff
        bumps look like better grippers.

        Ramp slowly.  The torque needed to inertially spin the object through
        the threshold goes as (ramp rate)^(2/3), so a slow ramp pushes that
        floor down; at these defaults it sits near 5 mN*m, well under the
        signal.  Compare any result against that floor before believing it.
        """
        yaw0 = self.obj_yaw()
        peak = 0.0
        for s in range(ramp_steps):
            t = t_max * (s + 1) / ramp_steps
            self.data.xfrc_applied[self.obj][5] = t
            mujoco.mj_step(self.model, self.data)
            if abs(self.obj_yaw() - yaw0) > np.radians(slip_deg):
                self.data.xfrc_applied[self.obj][5] = 0.0
                return t, True
            peak = t
        self.data.xfrc_applied[self.obj][5] = 0.0
        return peak, False

    def pull_until_slip(self, f_max=40.0, ramp_steps=30000, slip_mm=1.0):
        """Ramp a pull along the cylinder's axis until the object breaks away.

        Returns (slip_force_N, slipped).  Slip is declared on DISPLACEMENT, not
        on velocity: the bumps shear elastically by hundreds of microns before
        anything breaks loose, and a velocity threshold picks that up as slip.
        The 1mm default sits well clear of the elastic range.
        """
        y0 = self.obj_pos()[1]
        peak = 0.0
        for s in range(ramp_steps):
            f = f_max * (s + 1) / ramp_steps
            self.data.xfrc_applied[self.obj][1] = f
            mujoco.mj_step(self.model, self.data)
            if abs(self.obj_pos()[1] - y0) > slip_mm * 1e-3:
                self.data.xfrc_applied[self.obj][1] = 0.0
                return f, True
            peak = f
        self.data.xfrc_applied[self.obj][1] = 0.0
        return peak, False
