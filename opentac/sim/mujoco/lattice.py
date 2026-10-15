"""Run indentation sweeps and read per-bump base forces."""

import numpy as np
import mujoco

from .params import BumpParams
from .mjcf import build_mjcf, CLEARANCE

# joint order per bump: press, shear X, shear Y, tip about X, tip about Y
AXES = ("bz", "bsx", "bsy", "bx", "by")
Z, SX, SY, TX, TY = range(5)


class Lattice:
    def __init__(self, p: BumpParams, nx=5, ny=5, indenter="sphere", timestep=1e-4):
        self.p = p
        xml, self.positions, self.z0, self.drop = build_mjcf(p, nx, ny, indenter, timestep)
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.n = len(self.positions)
        # DoF addresses for each bump's three joints, resolved once.
        self.dof = []
        for i in range(self.n):
            self.dof.append(tuple(
                self.model.jnt_dofadr[mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{ax}_{i}")]
                for ax in AXES))
        self.iq = [self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("ix", "iy", "iz")]
        self.qadr = []
        for i in range(self.n):
            self.qadr.append(tuple(
                self.model.jnt_qposadr[mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{ax}_{i}")]
                for ax in AXES))

    def base_forces(self):
        """Normal force at each bump's base, in newtons, positive = compressed.

        Read from the JOINT deflection, not from contacts.  Two reasons: it is
        exactly the quantity the Velostat under that bump would see, and it is
        one number per bump regardless of how many contact points MuJoCo
        happened to generate on the bump's top face.
        """
        q = self.data.qpos
        return np.array([-self.p.k_axial * q[self.qadr[i][0]] for i in range(self.n)])

    def shear_forces(self):
        """(Fx, Fy) transmitted into each bump's base, N."""
        q = self.data.qpos
        return np.array([[self.p.k_shear_x * q[self.qadr[i][SX]],
                          self.p.k_shear_y * q[self.qadr[i][SY]]]
                         for i in range(self.n)])

    def base_moments(self):
        """(Mx, My) at each bump's base, N*m -- the shear channel.

        Reconstructed rather than read straight off the tip hinges, because a
        real beam couples lateral translation to base moment through an
        off-diagonal -6EI/H^2 term and MuJoCo joint springs are diagonal only.
        Under a flat object the hinges stay near zero while the shear joints
        carry everything, so reading the hinges alone reports NO shear -- the
        exact false negative Stage 0 hit.  A guided cantilever displaced by v
        puts a moment k*v*H/2 into its base, so that term is added back here.

        Putting the coupling in the READOUT rather than the dynamics keeps the
        forces the solver sees correct while making the sensed quantity right.
        A one-off single-bump FEM fit of the full 2x2 stiffness would let this
        move into the dynamics properly.
        """
        q, p, h = self.data.qpos, self.p, self.p.height
        out = []
        for i in range(self.n):
            a = self.qadr[i]
            my = p.k_shear_x * q[a[SX]] * h / 2.0 + p.k_tip_y * q[a[TY]]
            mx = -p.k_shear_y * q[a[SY]] * h / 2.0 + p.k_tip_x * q[a[TX]]
            out.append([mx, my])
        return np.array(out)

    def taxel_forces(self):
        """Normal force on each of the 4 taxels under every bump, N.

        Shape (n_bumps, 2, 2), indexed [bump][iy][ix] with ix/iy in {0,1} for
        the -/+ side.  This is the whole point of the bump sheet: statically
        distribute one bump's base wrench over its own 2x2 taxel block, so a
        sideways load shows up as a DIFFERENCE across four normal readings.
        Stage 4 turns these into Velostat conductances and a matrix scan; the
        mechanics prototype already produces them.
        """
        fz = self.base_forces()
        m = self.base_moments()
        ax = self.p.taxel_pitch_mm[0] * 1e-3 / 2.0
        ay = self.p.taxel_pitch_mm[1] * 1e-3 / 2.0
        out = np.zeros((self.n, 2, 2))
        for i in range(self.n):
            for iy, yv in enumerate((-ay, ay)):
                for ix, xv in enumerate((-ax, ax)):
                    out[i, iy, ix] = (fz[i] / 4.0
                                      + m[i][1] * xv / (4 * ax * ax)
                                      - m[i][0] * yv / (4 * ay * ay))
        return np.clip(out, 0.0, None)

    def settle(self, steps=400):
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def actual_depth_mm(self):
        """Indentation depth read back from the servo's real position.

        The servo deflects ~10um at 10N, so the commanded depth and the real
        one differ by a few percent at high load.  Everything reported uses
        this, not the command.
        """
        return -(CLEARANCE + self.data.qpos[self.iq[2]]) * 1e3

    def move_indenter(self, target, steps=2000):
        """Drive the indenter to `target` (x, y, z) CONTINUOUSLY, one small
        increment per solver step.

        Never teleport it.  MuJoCo resolves friction at the velocity level: it
        opposes relative tangential VELOCITY at the contact, and a jump in
        mocap_pos is a position change with no velocity attached.  Teleporting
        the indenter sideways therefore transmits exactly zero shear -- the
        contact simply re-forms at the new spot.  Stage 0 measured a tip moment
        of 0.005 mN*m against an expected 11 mN*m, a dead shear channel that
        would have looked like "bumps do not transmit shear" rather than a
        driving bug.  Normal force is unaffected (penetration is positional),
        which is exactly what makes this easy to miss.
        """
        start = np.array(self.data.ctrl[:3], dtype=float)
        target = np.asarray(target, dtype=float)
        for s in range(1, steps + 1):
            self.data.ctrl[:3] = start + (target - start) * (s / steps)
            mujoco.mj_step(self.model, self.data)

    def press_to(self, depth_mm, steps=2000, settle_steps=1000):
        """Ramp down to an indentation depth measured from first touch."""
        tgt = np.array(self.data.ctrl[:3], dtype=float)
        tgt[2] = -(CLEARANCE + depth_mm * 1e-3)
        self.move_indenter(tgt, steps)
        self.settle(settle_steps)

    def press_to_force(self, target_n, guess_mm=0.15, tol=0.03, iters=14,
                       steps=500, settle_steps=300):
        """Press until the lattice carries `target_n` newtons.

        Returns (depth_mm, total_N).  CHECK THE RETURNED FORCE -- on a hard
        config this can run out of iterations, and a silently unconverged point
        is indistinguishable from a real result in a summary table.

        Comparing lattices at equal DEPTH is meaningless: at fixed depth the
        object's surface alone decides which bumps it reaches, so every
        stiffness engages an identical set and the sweep comes out flat.
        Conformance is a question about equal LOAD -- a softer bump compresses
        further under the same force, letting the object descend onto its
        neighbours.  The first Stage 1 swept depth and "found" that stiffness
        has no effect, which was an artefact of the protocol.

        SAFEGUARDED secant, not plain secant.  Force rises much faster than
        linearly with depth as new bumps engage -- on a tilted plate each new
        column that lands adds stiffness -- so an unbracketed secant step
        overshoots and cannot recover.  Plain secant reported 25/25 bumps
        engaged at a "10 N" load whose peak single-bump force was 75 N.  Any
        step falling outside the known bracket falls back to bisection.
        """
        lo, hi = 0.0, None          # lo: force < target, hi: force > target
        d, tot = guess_mm, 0.0
        for _ in range(iters):
            d = min(max(d, 1e-4), 1.8)
            self.press_to(d, steps=steps, settle_steps=settle_steps)
            tot = float(self.base_forces().sum())
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

    def contact_tangential(self):
        """Magnitude of the tangential contact force on the indenter, N."""
        f = contact_force_total(self)
        return float(np.hypot(f[0], f[1]))

    def indent_sweep(self, depths_mm, settle_steps=400):
        """Press to each depth in turn, recording forces once settled.

        Depth is measured from FIRST TOUCH (indenter lowest point level with the
        undeformed bump tops), so 0.0 is the just-touching configuration.
        """
        out = []
        for d_mm in depths_mm:
            self.press_to(d_mm, steps=1500, settle_steps=settle_steps)
            f = self.base_forces()
            m = self.base_moments()
            out.append({
                "depth_mm": d_mm,
                "forces": f,
                "moments": m,
                "total_N": float(f.sum()),
                "engaged": int((f > 1e-3).sum()),
            })
        return out


def contact_force_total(lat: Lattice):
    """Sum of contact normal forces on the indenter -- an independent check that
    the joint-deflection readout above is not quietly wrong."""
    ind_geom = mujoco.mj_name2id(lat.model, mujoco.mjtObj.mjOBJ_GEOM, "ind")
    total = np.zeros(3)
    buf = np.zeros(6)
    for c in range(lat.data.ncon):
        con = lat.data.contact[c]
        if ind_geom not in (con.geom1, con.geom2):
            continue
        mujoco.mj_contactForce(lat.model, lat.data, c, buf)
        frame = con.frame.reshape(3, 3)
        f_world = frame.T @ buf[:3]
        total += f_world if con.geom1 == ind_geom else -f_world
    return total
