"""MJCF generation for a bump lattice.

Every bump is a rigid geom on a 3-DoF compliant joint: slide along Z (press)
plus two hinges at the base (tip about X and Y).  Tipping is what turns a
sideways load into a differential normal load across the 2x2 taxel block
underneath, which is the mechanism OpenTac's bump sheet was designed around.

GEOMETRY IS BUILT FROM PARAMETERS, NEVER FROM THE EXPORTED STL.  MuJoCo takes
the convex hull of a mesh, and the hull of the whole bump sheet is a
featureless slab -- every bump silently erased, and it still looks fine in the
viewer.  See NOTES.md, "Never feed it the sheet STL".
"""

from .params import BumpParams, MM

CLEARANCE = 0.0005      # m, indenter standoff above undeformed bump tops
IND_KP = 1.0e6          # N/m, indenter position servo
IND_ARM = 0.11          # kg, caps the servo at ~3000 rad/s like the bumps
IND_KV = 600.0          # N*s/m, ~0.9 damping ratio against IND_ARM


def _bump_body(idx, x, y, p: BumpParams, sign=1, tag="b"):
    """One bump.  `sign` = -1 builds it pointing DOWN, for the upper jaw.

    The mirror is done by offsetting the GEOM, never by putting a quat on the
    body: a body quat rotates the joint axes with it (see the indenter comment
    below), which would silently invert the shear DoFs on one jaw only.
    """
    kz, ktx, kty = p.k_axial, p.k_tip_x, p.k_tip_y
    ksx, ksy = p.k_shear_x, p.k_shear_y
    rt_x, rt_y = p.r_top
    # Circular cylinder at the mean top radius stands in for the elliptical
    # frustum.  The taper is structural and we have already lumped structure
    # into the joint stiffnesses, so the geom only has to get the CONTACT face
    # and the lever arm right.  Fidelity upgrade path is in NOTES.md.
    r = 0.5 * (rt_x + rt_y)
    h = p.height
    return f"""
    <body name="{tag}ump_{idx}" pos="{x:.6f} {y:.6f} 0">
      <joint name="{tag}z_{idx}" type="slide" axis="0 0 1"
             stiffness="{kz:.6g}" damping="{p.damping(kz):.6g}" armature="{p.armature(kz):.6g}"/>
      <!-- Lateral shear.  A flat object cannot tip a flat-topped bump, it
           shears it over; without these two joints the shear channel is dead.
           The rigid body translating means its BASE slides across the sheet
           too, which a real bump's base does not do -- a ~0.3mm geometric
           fiction on an 8.4mm pitch.  The base LOADS, which is all the sensing
           layer reads, are unaffected. -->
      <joint name="{tag}sx_{idx}" type="slide" axis="1 0 0"
             stiffness="{ksx:.6g}" damping="{p.damping(ksx):.6g}" armature="{p.armature(ksx):.6g}"/>
      <joint name="{tag}sy_{idx}" type="slide" axis="0 1 0"
             stiffness="{ksy:.6g}" damping="{p.damping(ksy):.6g}" armature="{p.armature(ksy):.6g}"/>
      <joint name="{tag}x_{idx}" type="hinge" axis="1 0 0"
             stiffness="{ktx:.6g}" damping="{p.damping(ktx, p.rot_inertia):.6g}" armature="{p.armature(ktx):.6g}"/>
      <joint name="{tag}y_{idx}" type="hinge" axis="0 1 0"
             stiffness="{kty:.6g}" damping="{p.damping(kty, p.rot_inertia):.6g}" armature="{p.armature(kty):.6g}"/>
      <geom name="{tag}g_{idx}" type="cylinder" size="{r:.6f} {h/2:.6f}" pos="0 0 {sign*h/2:.6f}"
            mass="{p.mass:.8f}" friction="{p.mu} 0.005 0.0001" rgba="0.25 0.55 0.85 1"
            contype="1" conaffinity="4"/>
    </body>"""


def lattice_positions(nx, ny, p: BumpParams):
    """Bump centres on a regular lattice, centred on the origin.

    OpenTac's `bump_centers_from_taxels` also handles partial 2x2 blocks and
    irregular board edges (invariant 2: a partial block still gets a FULL-size
    bump at the block's lattice centre).  Stage 3 swaps this function out for
    real centres read from a design; the rest of the model does not change.
    """
    px, py = p.bump_pitch
    xs = [(i - (nx - 1) / 2.0) * px for i in range(nx)]
    ys = [(j - (ny - 1) / 2.0) * py for j in range(ny)]
    return [(x, y) for y in ys for x in xs]


INDENTERS = {
    # name: (geom type, size, drop from body origin to lowest point, quat)
    #
    # Sized against the BUMP pitch, which is 8.4mm -- twice the taxel pitch,
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
    bodies = "".join(_bump_body(i, x, y, p) for i, (x, y) in enumerate(positions))

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
       sliding bump saturated at ~0.13*Fn instead of mu*Fn, i.e. an effective
       mu of 0.13 when mu was set to 1.0 -- friction quietly ~8x too low, which
       for a grasping study is the whole result.  Verified: mu_eff = 1.03 at
       mu = 1.0 with elliptic cone + noslip.  Low mu (<0.5) still floors out
       around 0.13; see NOTES.md, "Friction needs validating". -->
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic"
          noslip_iterations="10" gravity="0 0 -9.81"/>
  <visual><global offwidth="1280" offheight="960"/></visual>

  <default>
    <!-- DIRECT solref (negative pair = stiffness, damping in N/m and N*s/m).
         The default (timeconst, dampratio) form scales contact stiffness with
         the effective mass, and armature has inflated that mass by ~1e3x, so
         the contact would have come out ~40x SOFTER than the bump joint and we
         would have been measuring the contact model instead of the bump. -->
    <!-- Collision masks, NOT MuJoCo's automatic parent-child filter.  A bump
         body is a child of the pad body, so the pad/bump pair looks like it
         should be filtered automatically -- it is not.  The bump geom's base
         sits flush on the plate at z=0, so the instant the joint compressed,
         the bump sank into the plate and the PLATE carried the load straight
         to ground, bypassing the spring.  Measured stiffness came out at
         exactly half of analytic with ~120 N of equal-and-opposite contact
         force cancelling invisibly inside the pad.  Bumps (1/4) and plate
         (1/4) therefore cannot see each other; the object (4/1) sees both.
         This also switches off bump-bump contact, which at an 8.4mm pitch and
         4.7mm diameter needs a ~40deg tilt to occur. -->
    <!-- Near-rigid contact, on purpose.  ALL compliance is supposed to live
         in the bump joint; any softness left in the contact model shows up as
         a series spring and silently reduces the measured bump stiffness.
         The default (timeconst, dampratio) solref scales contact stiffness
         with effective mass, and armature has inflated that by ~1e3x, so the
         default came out ~40x softer than the joint: measured stiffness was
         55% of analytic and drifted with depth.  The direct (negative) form
         removes the mass dependence.  Tuned to 100.0% of analytic with zero
         residual jitter at dt=1e-4, holding across E_eff = 3 and 30 MPa.
         The tradeoff is that a rigid contact cannot dimple locally -- see
         NOTES.md, "Where the lumped model breaks". -->
    <geom solref="-3e7 -10000" solimp="0.999 0.9999 0.00001" condim="3"/>
  </default>

  <worldbody>
    <light pos="0 0 0.3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <body name="pad" pos="0 0 0">
      <geom name="plate" type="box" pos="0 0 {-p.sheet_thickness_mm*MM/2:.6f}"
            size="{plate_x:.6f} {plate_y:.6f} {p.sheet_thickness_mm*MM/2:.6f}"
            rgba="0.2 0.2 0.25 1" contype="1" conaffinity="4"/>{bodies}
    </body>
    <!-- A SERVOED body, deliberately not a mocap body.  MuJoCo takes contact
         velocity from qvel, and a mocap body has no DoFs, so its velocity is
         always zero no matter how fast mocap_pos is changed.  Normal force
         still works (penetration is positional) but friction NEVER mobilises:
         a block on a mocap slab dragged 4mm moves 0.000mm.  That silently
         removes the entire shear channel, which is the one thing the bumps
         exist for.  Three slide joints under a stiff position servo restore
         real velocities; armature conditions them the same way it does the
         bumps.  Position error under load is ~10um at 10N, so depth is read
         back from qpos rather than trusted from the command. -->
    <!-- Orientation goes on the GEOM, never on the body: a body quat rotates
         the body FRAME, and the three slide joints are defined in that frame,
         so tilting the body 90deg made the "z" servo push sideways.  The
         cylinder indenter silently never touched anything (0.02N at 1.8mm of
         commanded depth) while the 1deg plate was too small an error to
         notice. -->
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


# --- Stage 3: a scene built from a REAL OpenTac design -------------------------
# lattice_positions() above is the placeholder this replaces. Everything else
# -- joint stiffnesses, collision masks, the direct solref, the servoed
# (never mocap) driver -- is reused unchanged, because all of it was tuned
# against the analytic stiffness and none of it depends on the layout.

# OpenTac's BumpParams.recess_depth_mm default. NOT a field on TactileSim's
# BumpParams, which mirrors only what the MECHANICS needs -- the recess is
# pure sheet geometry and no stiffness depends on it. It matters here only
# to locate the sheet's top plane, so it is an argument with OpenTac's default
# rather than an invented parameter. A user who changed it when printing
# has to say so.
OTC_RECESS_DEPTH_MM = 1.0

BALL_KP = 2.0e6         # N/m, ball position servo (stiffer: bigger contact)
BALL_KV = 1200.0
BALL_ARM = 0.11
BALL_RKP = 800.0        # N*m/rad, twist servo
BALL_RKV = 8.0
BALL_RARM = 2.0e-3


def _frustum_vertices(rb, rt, h, segments=24):
    """Vertex soup for one elliptical frustum, z = 0 at the base.

    Safe as a mesh where the bump SHEET would not be: MuJoCo collides the
    convex hull, and a single frustum IS convex, so the hull is the shape
    itself. The hull of the whole sheet is a featureless slab with every bump
    erased -- see NOTES.md, "Never feed it the sheet STL".

    This is also the fidelity upgrade mjcf.py's cylinder comment points at:
    the real bumps are ELLIPTICAL where a 2x2 block is not square, and a
    cylinder at the mean top radius cannot express that.
    """
    import numpy as _np
    a = _np.linspace(0.0, 2.0 * _np.pi, segments, endpoint=False)
    base = _np.column_stack([rb[0] * _np.cos(a), rb[1] * _np.sin(a),
                             _np.zeros(segments)])
    top = _np.column_stack([rt[0] * _np.cos(a), rt[1] * _np.sin(a),
                            _np.full(segments, h)])
    v = _np.vstack([base, top])
    return " ".join(f"{x:.6f}" for x in v.reshape(-1))


def _bump_body_mesh(idx, x, y, p: BumpParams, mesh="bumpmesh"):
    """Same five compliant DoFs as _bump_body, with the real frustum geom."""
    kz, ktx, kty = p.k_axial, p.k_tip_x, p.k_tip_y
    ksx, ksy = p.k_shear_x, p.k_shear_y
    return f"""
    <body name="bump_{idx}" pos="{x:.6f} {y:.6f} 0">
      <joint name="bz_{idx}" type="slide" axis="0 0 1"
             stiffness="{kz:.6g}" damping="{p.damping(kz):.6g}" armature="{p.armature(kz):.6g}"/>
      <joint name="bsx_{idx}" type="slide" axis="1 0 0"
             stiffness="{ksx:.6g}" damping="{p.damping(ksx):.6g}" armature="{p.armature(ksx):.6g}"/>
      <joint name="bsy_{idx}" type="slide" axis="0 1 0"
             stiffness="{ksy:.6g}" damping="{p.damping(ksy):.6g}" armature="{p.armature(ksy):.6g}"/>
      <joint name="bx_{idx}" type="hinge" axis="1 0 0"
             stiffness="{ktx:.6g}" damping="{p.damping(ktx, p.rot_inertia):.6g}" armature="{p.armature(ktx):.6g}"/>
      <joint name="by_{idx}" type="hinge" axis="0 1 0"
             stiffness="{kty:.6g}" damping="{p.damping(kty, p.rot_inertia):.6g}" armature="{p.armature(kty):.6g}"/>
      <geom name="bg_{idx}" type="mesh" mesh="{mesh}" mass="{p.mass:.8f}"
            friction="{p.mu} 0.005 0.0001" rgba="0.30 0.62 0.90 1"
            contype="1" conaffinity="4"/>
    </body>"""


def sheet_visual_stl(stl_path, p: BumpParams, out_path,
                     recess_depth_mm=OTC_RECESS_DEPTH_MM):
    """Strip the bumps off OpenTac's bump-sheet STL, keeping the real base.

    The exported sheet is the true part: the board's own outline, the skirt
    that stiffens its rim, the recess the sensor drops into, and the slits
    that cut every bump free of its neighbours (bump_sheet invariant 3). All
    of that is worth showing, and none of it is reproducible from parameters.

    Its BUMPS are not wanted here: this scene already has a physics bump at
    every one of those positions, and drawing both means the moving frustum
    sinks into a static grey shell of itself. So slice at the sheet's top
    plane and keep what is below.

    VISUAL ONLY. This mesh must never become a collision geom: MuJoCo collides
    the convex hull, and the hull of a sheet is a featureless slab with every
    bump erased -- and it still looks right in the viewer. See NOTES.md,
    "Never feed it the sheet STL". A visual geom is not collided, so the true
    triangles are what get drawn.

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
    # not needed: the bumps are UNIONED into the sheet, so its top face is
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
    # Put the sheet's TOP at z=0, which is where the bumps' bases sit here.
    out.apply_translation((0.0, 0.0, -top))
    out.export(out_path, file_type="stl")
    return top


def build_sensor_mjcf(bump_xy, p: BumpParams, ball_r_m, centre_xy,
                      sheet_stl=None, timestep=1e-4, segments=24,
                      ball_mode="servo", ball_mass=0.05, drop_h_m=0.0):
    """A scene for one real sensor: its own bumps, and a ball above them.

    `bump_xy` is in METRES, already in the sensor's own frame -- straight from
    a manifest. The ball gets three slide DoFs and a twist hinge, all under
    position servos, because a mocap body has no DoFs and therefore no
    velocity: normal force would still work while friction NEVER mobilised,
    silently deleting the shear channel the bumps exist for.

    The twist hinge is declared LAST. Joints compose in order, so a hinge
    before the slides would rotate the slide axes with it and the "z" servo
    would push sideways once the ball turned.
    """
    import numpy as _np
    xy = _np.asarray(bump_xy, dtype=float)
    rb, rt, h = p.r_base, p.r_top, p.height

    bodies = "".join(_bump_body_mesh(i, x, y, p) for i, (x, y) in enumerate(xy))

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
    # the work: the only way to get a real IMPACT, since a position servo
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

    # The real sheet if we have it, a plain slab if not. The slab stays the
    # COLLIDER either way: the sheet is visual-only (see sheet_visual_stl),
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
    <mesh name="bumpmesh" vertex="{_frustum_vertices(rb, rt, h, segments)}"/>
    {sheet_asset}
    <texture name="grid" type="2d" builtin="checker" rgb1="0.16 0.17 0.20"
             rgb2="0.20 0.21 0.25" width="300" height="300"/>
    <material name="gridmat" texture="grid" texrepeat="6 6" reflectance="0.05"/>
    <!-- Translucent so the bumps compressing UNDER the contact stay visible.
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
