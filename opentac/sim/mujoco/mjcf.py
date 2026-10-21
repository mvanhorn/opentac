"""MJCF generation for bump lattices, the gripper and real sensors.

Every bump is a rigid elliptical frustum on a compliant joint: a slide along Z
(press), two lateral slides (shear) and two hinges at the base (tip about X
and Y). Tipping and shear turn a sideways load into a differential normal
load across the 2x2 taxel block underneath, which is the mechanism OpenTac's
bump sheet was designed around. One builder, `add_bump`, makes every bump of
every scene, at full double precision.

Geometry is built from parameters, never from the exported STL. MuJoCo
collides the convex hull of a mesh: a single frustum is convex, so its hull
is itself, while the hull of the whole bump sheet is a featureless slab with
every bump erased.
"""
import xml.etree.ElementTree as ET

import numpy as np

from .params import BumpParams, MM

CLEARANCE = 0.0005      # m, indenter standoff above undeformed bump tops
IND_KP = 1.0e6          # N/m, indenter position servo
IND_ARM = 0.11          # kg, caps the servo at ~3000 rad/s like the bumps
IND_KV = 600.0          # N*s/m, ~0.9 damping ratio against IND_ARM


def numbers(x):
    """Space-separated values at full double precision, for MJCF attributes."""
    return " ".join(format(float(v), ".17g") for v in np.asarray(x).ravel())


def frustum_vertices(rb, rt, h, segments=24):
    """Vertices of an elliptical frustum for a convex MuJoCo mesh, in metres; z = 0 at the base, h at the top."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    base = np.column_stack([rb[0] * np.cos(angles), rb[1] * np.sin(angles), np.zeros(segments)])
    top = np.column_stack([rt[0] * np.cos(angles), rt[1] * np.sin(angles), np.full(segments, h)])
    return np.vstack([base, top])


def bump_mesh(p: BumpParams, name="bumpmesh", sign=1, segments=24):
    """`<mesh>` asset of one bump; sign = -1 points it down (the upper jaw of the gripper)."""
    v = frustum_vertices(p.r_base, p.r_top, p.height, segments) * [1.0, 1.0, sign]
    return f'<mesh name="{name}" vertex="{numbers(v)}"/>'


#: Joint suffixes of a bump, in the order press, shear x, shear y, tip about x, tip about y.
JOINTS = ("z", "sx", "sy", "x", "y")


def add_bump(parent, idx, x, y, p: BumpParams, mesh="bumpmesh", tag="b"):
    """Add one frustum bump under `parent` on its five compliant joints; returns the body.

    Names are `{tag}ump_{idx}` for the body, `{tag}z_{idx}`, `{tag}sx_{idx}`,
    `{tag}sy_{idx}`, `{tag}x_{idx}` and `{tag}y_{idx}` for the joints, and
    `{tag}g_{idx}` for the geom. A mirrored bump (the upper jaw) takes a
    mirrored mesh rather than a rotated body: a body rotation would rotate
    the joint axes with it and invert the shear coordinates of that jaw only.
    """
    body = ET.SubElement(parent, "body", name=f"{tag}ump_{idx}", pos=numbers([x, y, 0]))
    for suffix, kind, axis, stiffness, inertia in (
        ("z", "slide", "0 0 1", p.k_axial, None),
        ("sx", "slide", "1 0 0", p.k_shear_x, None),
        ("sy", "slide", "0 1 0", p.k_shear_y, None),
        ("x", "hinge", "1 0 0", p.k_tip_x, p.rot_inertia),
        ("y", "hinge", "0 1 0", p.k_tip_y, p.rot_inertia),
    ):
        ET.SubElement(body, "joint", name=f"{tag}{suffix}_{idx}", type=kind, axis=axis,
                      stiffness=repr(float(stiffness)), damping=repr(float(p.damping(stiffness, inertia))),
                      armature=repr(float(p.armature(stiffness))))
    ET.SubElement(body, "geom", name=f"{tag}g_{idx}", type="mesh", mesh=mesh, mass=repr(float(p.mass)),
                  friction=numbers([p.mu, .005, .0001]), rgba="0.30 0.62 0.90 1",
                  contype="1", conaffinity="4")
    return body


def bump_bodies(positions, p: BumpParams, mesh="bumpmesh", tag="b"):
    """MJCF text of the bumps at `positions` (metres), for the string-built scenes."""
    holder = ET.Element("holder")
    for i, (x, y) in enumerate(positions):
        add_bump(holder, i, x, y, p, mesh=mesh, tag=tag)
    return "".join("\n    " + ET.tostring(b, encoding="unicode") for b in holder)


def lattice_positions(nx, ny, p: BumpParams):
    """Bump centres on a regular lattice, centred on the origin.

    OpenTac's `bump_centers_from_taxels` also handles partial 2x2 blocks and
    irregular board edges (a partial block still gets a full-size bump at the
    block's lattice centre).  `build_sensor_mjcf` takes real centres read from
    a design instead; the rest of the model is the same.
    """
    px, py = p.bump_pitch
    xs = [(i - (nx - 1) / 2.0) * px for i in range(nx)]
    ys = [(j - (ny - 1) / 2.0) * py for j in range(ny)]
    return [(x, y) for y in ys for x in xs]


INDENTERS = {
    # name: (geom type, size, drop from body origin to lowest point, quat)
    #
    # Sized against the bump pitch, which is 8.4mm -- twice the taxel pitch,
    # because OpenTac puts one bump per 2x2 taxel block.  That pitch is coarse: a
    # 15mm ball at 1mm indentation only spans +/-5.5mm, so it never reaches a
    # neighbouring bump no matter how soft the bumps are.  Anything meant to
    # show conformance has to be much larger or much flatter than that.
    "sphere": ("sphere", "0.004", 0.004, ""),
    "ball40": ("sphere", "0.040", 0.040, ""),
    "punch":  ("cylinder", "0.005 0.003", 0.003, ""),
    "edge":   ("box", "0.012 0.0008 0.003", 0.003, ""),
    "flat":   ("box", "0.030 0.030 0.003", 0.003, ""),
    # The conformance test.  A plate tilted 1 degree spans 0.59mm of height
    # across a 5x5 lattice -- comfortably inside a 2mm bump's travel, so
    # whether all 25 bumps meet it is purely a question of compliance and not
    # of geometry.  5 degrees would need 2.9mm and no bump could reach.
    "tilt1":  ("box", "0.030 0.030 0.003", 0.003523,
               ' quat="0.9999619 0 0.0087265 0"'),
    "cylinder_side": ("cylinder", "0.025 0.020", 0.025,
                      ' quat="0.7071068 0.7071068 0 0"'),
}


def build_mjcf(p: BumpParams, nx=5, ny=5, indenter="sphere", timestep=1e-4):
    positions = lattice_positions(nx, ny, p)
    bodies = bump_bodies(positions, p)

    gtype, gsize, drop, quat = INDENTERS[indenter]
    # Sitting the indenter exactly on the bump tops makes first contact
    # ambiguous; start it CLEARANCE clear and ramp down from there.
    z0 = p.height + drop + CLEARANCE

    plate_x = max(x for x, _ in positions) + p.r_base[0] + 0.002
    plate_y = max(y for _, y in positions) + p.r_base[1] + 0.002

    xml = f"""<mujoco model="bump_lattice">
  <compiler angle="radian"/>
  <!-- implicitfast integrates joint spring/damper implicitly, which is what
       lets these stiff bumps run at 1e-4 rather than 1e-6. -->
  <!-- noslip_iterations: MuJoCo solves friction as a soft constraint, which
       creeps under sustained tangential load.  Without the noslip pass a
       sliding bump saturates near 0.13*Fn instead of mu*Fn, an effective mu
       of 0.13 at mu = 1.0.  With the elliptic cone and the noslip pass,
       mu_eff = 1.03 at mu = 1.0.  Below mu = 0.5 the effective friction
       floors near 0.13. -->
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic"
          noslip_iterations="10" gravity="0 0 -9.81"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <asset>{bump_mesh(p)}</asset>

  <default>
    <!-- Collision masks, not MuJoCo's automatic parent-child filter.  A bump
         body is a child of the pad body, but the pad/bump pair is not
         filtered automatically.  The bump geom's base sits flush on the plate
         at z=0, so without the masks a compressed bump sinks into the plate
         and the plate carries the load to ground, bypassing the spring (the
         measured stiffness is then half the analytic value).  Bumps (1/4)
         and plate (1/4) therefore do not collide; the object (4/1) collides
         with both.  This also disables bump-bump contact, which at an 8.4mm
         pitch and 4.7mm diameter would need a ~40deg tilt to occur. -->
    <!-- Near-rigid contact, on purpose.  All compliance belongs in the bump
         joint; any softness left in the contact model acts as a series spring
         and reduces the measured bump stiffness.  The default (timeconst,
         dampratio) solref scales contact stiffness with the effective mass,
         which the armature inflates by orders of magnitude.  The direct
         (negative) form, stiffness and damping in N/m and N*s/m, removes the
         mass dependence.  The trade-off is that a rigid contact cannot dimple
         locally. -->
    <geom solref="-3e7 -10000" solimp="0.999 0.9999 0.00001" condim="3"/>
  </default>

  <worldbody>
    <light pos="0 0 0.3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <body name="pad" pos="0 0 0">
      <geom name="plate" type="box" pos="0 0 {-p.sheet_thickness_mm*MM/2:.6f}"
            size="{plate_x:.6f} {plate_y:.6f} {p.sheet_thickness_mm*MM/2:.6f}"
            rgba="0.2 0.2 0.25 1" contype="1" conaffinity="4"/>{bodies}
    </body>
    <!-- A servoed body, deliberately not a mocap body.  MuJoCo takes contact
         velocity from qvel, and a mocap body has no DoFs, so its velocity is
         always zero however fast mocap_pos changes.  Normal force is still
         produced (penetration is positional), but friction never mobilises,
         which removes the shear channel the bumps exist for.  Three slide
         joints under a stiff position servo provide real velocities; armature
         conditions them as it does the bumps.  Position error under load is
         ~10um at 10N, so depth is read back from qpos rather than taken from
         the command. -->
    <!-- Orientation goes on the geom, never on the body: a body quat rotates
         the body frame, in which the three slide joints are defined, so a
         tilted body would make the "z" servo push sideways. -->
    <body name="indenter" pos="0 0 {z0:.6f}">
      <joint name="ix" type="slide" axis="1 0 0" armature="{IND_ARM}"/>
      <joint name="iy" type="slide" axis="0 1 0" armature="{IND_ARM}"/>
      <joint name="iz" type="slide" axis="0 0 1" armature="{IND_ARM}"/>
      <geom name="ind" type="{gtype}" size="{gsize}" mass="0.05"{quat}
            friction="{p.mu} 0.005 0.0001" rgba="0.9 0.4 0.2 1"
            contype="4" conaffinity="1"/>
    </body>
  </worldbody>

  <actuator>
    <position joint="ix" kp="{IND_KP}" kv="{IND_KV}"/>
    <position joint="iy" kp="{IND_KP}" kv="{IND_KV}"/>
    <position joint="iz" kp="{IND_KP}" kv="{IND_KV}"/>
  </actuator>
</mujoco>
"""
    return xml, positions, z0, drop


# --- A scene built from a real OpenTac design ----------------------------------
# Real bump centres take the place of lattice_positions(). Everything else
# (joint stiffnesses, collision masks, the direct solref, the servoed rather
# than mocap driver) is the same as in build_mjcf, because all of it was tuned
# against the analytic stiffness and none of it depends on the layout.

# The default of the bump sheet's BumpParams.recess_depth_mm (opentac.cad). It
# is not a field of the simulation's BumpParams, which holds only what the
# mechanics needs: the recess is sheet geometry, and no stiffness depends on
# it. It is used here only to locate the sheet's top plane, so it is an
# argument with the sheet's default rather than a model parameter. A sheet
# printed with a different recess depth must pass its own value.
OTC_RECESS_DEPTH_MM = 1.0

BALL_KP = 2.0e6         # N/m, ball position servo (stiffer: bigger contact)
BALL_KV = 1200.0
BALL_ARM = 0.11
BALL_RKP = 800.0        # N*m/rad, twist servo
BALL_RKV = 8.0
BALL_RARM = 2.0e-3


def sheet_visual_stl(stl_path, p: BumpParams, out_path,
                     recess_depth_mm=OTC_RECESS_DEPTH_MM):
    """Strip the bumps off OpenTac's bump-sheet STL, keeping the real base.

    The exported sheet is the true part: the board's own outline, the skirt
    that stiffens its rim, the recess the sensor drops into, and the slits
    that cut every bump free of its neighbours. All of that is worth showing,
    and none of it is reproducible from parameters.

    Its bumps are not wanted here: this scene already has a physics bump at
    every one of those positions, and drawing both means the moving frustum
    sinks into a static grey shell of itself. So slice at the sheet's top
    plane and keep what is below.

    Visual only. This mesh must never become a collision geom: MuJoCo collides
    the convex hull, and the hull of a sheet is a featureless slab with every
    bump erased, although it still looks correct in the viewer. A visual geom
    is not collided, so the true triangles are drawn.

    Returns the z offset (mm) applied, so the caller can place it.
    """
    import numpy as _np
    import trimesh

    mesh = trimesh.load(stl_path, file_type="stl")
    # bump_sheet.py's own layer table: z=0 is the part surface, the recess
    # ceiling is at recess_depth, and the sheet top (where bumps start) is
    # another sheet_thickness above that.
    top = recess_depth_mm + p.sheet_thickness_mm       # sheet top plane, mm

    # Drop every face with a vertex above the plane, rather than cutting at it.
    # trimesh's slice_plane imports shapely unconditionally, and a true cut is
    # not needed: the bumps are unioned into the sheet, so its top face is
    # already closed across each bump footprint, and every triangle belonging
    # to a frustum has at least one vertex above the plane. What is left is the
    # slab with its skirt and slits, which is the part worth showing.
    v = _np.asarray(mesh.vertices, dtype=float)
    f = _np.asarray(mesh.faces)
    keep = v[f][:, :, 2].max(axis=1) <= top + 1e-6
    if not keep.any():
        raise ValueError(f"slicing {stl_path} at z={top} left nothing")
    out = trimesh.Trimesh(vertices=v, faces=f[keep], process=False)
    out.remove_unreferenced_vertices()
    # Put the sheet's top at z=0, which is where the bumps' bases sit here.
    out.apply_translation((0.0, 0.0, -top))
    out.export(out_path, file_type="stl")
    return top


def build_sensor_mjcf(bump_xy, p: BumpParams, ball_r_m, centre_xy,
                      sheet_stl=None, timestep=1e-4, segments=24,
                      ball_mode="servo", ball_mass=0.05, drop_h_m=0.0):
    """A scene for one real sensor: its own bumps, and a ball above them.

    `bump_xy` is in metres, already in the sensor's own frame -- straight from
    a manifest. The ball gets three slide DoFs and a twist hinge, all under
    position servos, because a mocap body has no DoFs and therefore no
    velocity: normal force would still be produced while friction never
    mobilised, which would remove the shear channel the bumps exist for.

    The twist hinge is declared last. Joints compose in order, so a hinge
    before the slides would rotate the slide axes with it and the "z" servo
    would push sideways once the ball turned.
    """
    import numpy as _np
    xy = _np.asarray(bump_xy, dtype=float)
    rb, h = p.r_base, p.height

    bodies = bump_bodies(xy, p)

    lo = xy.min(axis=0) - _np.array(rb) - 0.0015
    hi = xy.max(axis=0) + _np.array(rb) + 0.0015
    cx_p, cy_p = 0.5 * (lo + hi)
    sx_p, sy_p = 0.5 * (hi - lo)
    plate_t = p.sheet_thickness_mm * MM

    cx, cy = centre_xy
    z0 = h + ball_r_m + CLEARANCE + drop_h_m

    # "servo" drives the ball through position actuators -- deliberately not a
    # mocap body, which has no DoFs and therefore no velocity, so friction
    # would never mobilise. "free" gives it a free joint and lets gravity do
    # the work: the only way to get a real impact, since a position servo
    # would simply hold its commanded height through the collision.
    if ball_mode == "free":
        ball_joints = '<freejoint name="ballfree"/>'
        actuators = ""
    elif ball_mode == "servo":
        ball_joints = (
            f'<joint name="ix" type="slide" axis="1 0 0" armature="{BALL_ARM}"/>'
            f'<joint name="iy" type="slide" axis="0 1 0" armature="{BALL_ARM}"/>'
            f'<joint name="iz" type="slide" axis="0 0 1" armature="{BALL_ARM}"/>'
            f'<joint name="rz" type="hinge" axis="0 0 1" armature="{BALL_RARM}"/>')
        actuators = (
            f'<position joint="ix" kp="{BALL_KP}" kv="{BALL_KV}"/>'
            f'<position joint="iy" kp="{BALL_KP}" kv="{BALL_KV}"/>'
            f'<position joint="iz" kp="{BALL_KP}" kv="{BALL_KV}"/>'
            f'<position joint="rz" kp="{BALL_RKP}" kv="{BALL_RKV}"/>')
    else:
        raise ValueError(f"ball_mode must be 'servo' or 'free', not {ball_mode!r}")

    # The real sheet when one is given, a plain slab otherwise. The slab is the
    # collider in both cases: the sheet is visual-only (see sheet_visual_stl),
    # because MuJoCo would collide its convex hull -- a featureless slab.
    plate_geom = (f'<geom name="plate" type="box" '
                  f'pos="{cx_p:.6f} {cy_p:.6f} {-plate_t/2:.6f}" '
                  f'size="{sx_p:.6f} {sy_p:.6f} {plate_t/2:.6f}" '
                  f'material="gridmat" contype="1" conaffinity="4"/>')
    sheet_asset = ""
    if sheet_stl:
        # The stand-in slab turns invisible but keeps colliding, so the
        # contact behaviour is byte-identical with or without the sheet.
        plate_geom = (plate_geom.replace('material="gridmat"',
                                         'rgba="0 0 0 0"')
                      + '\n      '
                      + '<geom name="sheetvis" type="mesh" mesh="sheet" '
                        'contype="0" conaffinity="0" mass="0" '
                        'material="sheetmat"/>')
        sheet_asset = (f'<mesh name="sheet" file="{sheet_stl}" '
                       f'scale="0.001 0.001 0.001"/>'
                       + '\n    '
                       + '<material name="sheetmat" rgba="0.40 0.43 0.49 1" '
                         'specular="0.25" shininess="0.25"/>')

    return f"""<mujoco model="otc_sensor">
  <compiler angle="radian"/>
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic"
          noslip_iterations="10" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1600" offheight="1200"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55"/>
  </visual>
  <asset>
    {bump_mesh(p, segments=segments)}
    {sheet_asset}
    <texture name="grid" type="2d" builtin="checker" rgb1="0.16 0.17 0.20"
             rgb2="0.20 0.21 0.25" width="300" height="300"/>
    <material name="gridmat" texture="grid" texrepeat="6 6" reflectance="0.05"/>
    <!-- Translucent so the bumps compressing under the contact stay visible.
         The deformation is the thing worth watching and an opaque ball hides
         exactly the region where it happens. -->
    <material name="ballmat" rgba="0.97 0.52 0.26 0.22" specular="0.3" shininess="0.3"/>
  </asset>

  <default>
    <geom solref="-3e7 -10000" solimp="0.999 0.9999 0.00001" condim="3"/>
  </default>

  <worldbody>
    <light pos="{cx:.4f} {cy - 0.05:.4f} 0.16" dir="0 0.3 -1" directional="false"
           diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3" castshadow="true"/>
    <light pos="{cx + 0.06:.4f} {cy + 0.06:.4f} 0.12" dir="-0.4 -0.4 -1"
           directional="false" diffuse="0.35 0.35 0.4" castshadow="false"/>

    <!-- targetbody aiming, so the framing does not depend on hand-tuned
         xyaxes that break the moment the pad size or ball radius changes. -->
    <camera name="iso" mode="targetbody" target="pad"
            pos="{cx + 0.049:.4f} {cy - 0.056:.4f} 0.026"/>
    <camera name="close" mode="targetbody" target="pad"
            pos="{cx + 0.030:.4f} {cy - 0.034:.4f} 0.015"/>
    <camera name="side" mode="targetbody" target="pad"
            pos="{cx:.4f} {cy - 0.058:.4f} 0.007"/>

    <body name="pad" pos="0 0 0">
      {plate_geom}{bodies}
    </body>

    <body name="ball" pos="{cx:.6f} {cy:.6f} {z0:.6f}">
      {ball_joints}
      <geom name="ball" type="sphere" size="{ball_r_m:.6f}" mass="{ball_mass}"
            material="ballmat" friction="{p.mu} 0.005 0.0001"
            contype="4" conaffinity="1"/>
      <!-- A stripe, so the twist is visible on the ball itself rather than
           only in the numbers. Visual only: no contact, no mass. -->
      <geom name="mark" type="box" contype="0" conaffinity="0" mass="0"
            pos="0 {ball_r_m*0.999:.6f} 0"
            size="{ball_r_m*0.06:.6f} {ball_r_m*0.02:.6f} {ball_r_m*0.30:.6f}"
            rgba="0.15 0.15 0.18 1"/>
      <geom name="mark2" type="box" contype="0" conaffinity="0" mass="0"
            pos="{ball_r_m*0.999:.6f} 0 0"
            size="{ball_r_m*0.02:.6f} {ball_r_m*0.06:.6f} {ball_r_m*0.30:.6f}"
            rgba="0.15 0.15 0.18 1"/>
    </body>
  </worldbody>

  <actuator>
    {actuators}
  </actuator>
</mujoco>
"""
