"""The loader reads a real OpenTac board correctly, and says so when it cannot.

The failures these tests guard against produce plausible output rather than
errors: a contact split into two blobs at opposite ends of the image, thirty
bumps reported in contact when five are, a sensor placed on a surface in
entirely the wrong place. None of them crashes anything, which is why this
layer is tested rather than judged from a picture.

Fixtures are real manifests exported by OpenTac-CAD, not hand-written:

  flat_square.json  48 taxels, 12 bumps, every 2x2 block full
  flat_disc.json    41 taxels, 15 bumps, 9 blocks cut short by the curved edge

The disc is the important one. `bump_centers_from_taxels` gives a partial
block a full-size bump at the block's lattice centre, so those nine cells have
fewer than four taxels and asymmetric offsets, which an idealised "four taxels
at +/-pitch/2" model cannot express.
"""
import copy
import os

import numpy as np
import pytest

from opentac.sim.mujoco import BumpParams, ManifestError, TactileSensor
from opentac.sim.mujoco import surrogate as sg
from opentac.sim.mujoco.loader import MM, NO_BUMP

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def load(name, **kw):
    return TactileSensor.from_manifest(os.path.join(FIX, name), **kw)


@pytest.fixture(scope="module")
def square():
    return load("flat_square.json")


@pytest.fixture(scope="module")
def disc():
    return load("flat_disc.json")


@pytest.fixture(params=["flat_square.json", "flat_disc.json"])
def any_board(request):
    return load(request.param)


# --- geometry ------------------------------------------------------------

def test_counts_match_the_manifest(any_board):
    s = any_board
    man = s.manifest
    assert s.n_taxels == len(man["pixels"])
    assert s.n_bumps == len(man["shear"]["bumps"])
    assert len(s.taxel_bump) == s.n_taxels


def test_positions_are_converted_to_metres(square):
    """The manifest is mm and this package is SI. A 1000x error raises nothing:
    a sensor that size still simulates, but never touches anything."""
    man = square.manifest
    px = np.asarray([[p[0], p[1]] for p in man["pixels"]], dtype=float)
    assert np.allclose(square.xy, px * MM)
    # Sanity: a real pad is centimetres, so metres puts it well under 1.
    assert np.abs(square.xy).max() < 0.1
    assert 1e-3 < square.pitch[0] < 1e-2


def test_pitch_comes_from_pixel_spacing_not_from_px_py(square):
    """px/py are pad-scan order; the manifest's own _readme says so.

    split_row_pin_order reverses the bottom half of the rows, so py can even
    be negative. Deriving pitch from it would mis-size every bump.
    """
    man = square.manifest
    dx, dy = square.pitch
    assert dx > 0 and dy > 0
    # Whatever px/py say, the derived pitch must match real neighbour spacing.
    by_rc = {(int(r), int(c)): i for i, (_, _, r, c) in enumerate(man["pixels"])}
    seen = []
    for (r, c), i in by_rc.items():
        j = by_rc.get((r, c + 1))
        if j is not None:
            seen.append(abs(square.xy[j, 0] - square.xy[i, 0]))
    assert np.isclose(np.median(seen), dx)


def test_centre_lands_on_a_bump(any_board):
    """The taxel centroid falls between four bumps on a regular grid, which is
    the worst possible contact point: an 8mm ball there reads exactly zero."""
    cx, cy = any_board.centre()
    d = np.hypot(any_board.bump_xy[:, 0] - cx, any_board.bump_xy[:, 1] - cy)
    assert d.min() < 1e-12, "centre() must sit exactly on a bump"


def test_offsets_are_relative_to_the_owning_bump(any_board):
    s = any_board
    d = s.offsets()
    has = s.taxel_bump != NO_BUMP
    assert np.allclose(d[has], s.xy[has] - s.bump_xy[s.taxel_bump[has]])
    assert np.allclose(d[~has], 0.0)
    # No taxel should be further from its bump than half a bump pitch.
    reach = np.hypot(d[has, 0], d[has, 1])
    assert reach.max() < max(s.pitch)


# --- partial blocks ------------------------------------------------------

def test_the_disc_has_partial_blocks(disc):
    """Guard the fixture itself: if the exporter ever stops producing them,
    every partial-block test below would pass vacuously."""
    sizes = disc.group_sizes()
    assert (sizes < 4).sum() >= 5, f"expected cut-short blocks, got {sizes}"
    assert sizes.min() >= 1
    assert sizes.sum() == (disc.taxel_bump != NO_BUMP).sum()


def test_partial_blocks_spread_load_over_only_their_own_taxels(disc):
    """Assuming four taxels at +/-pitch/2 would put load on taxels the outline
    removed, which do not exist on this board."""
    surf = sg.plane_surface(disc.p.height - 0.3e-3)
    f = disc.taxel_forces(surf)
    base = disc.surrogate.base_forces(surf)

    for bi in np.nonzero(disc.group_sizes() < 4)[0]:
        members = np.nonzero(disc.taxel_bump == bi)[0]
        assert len(members) < 4
        if base[bi] > 1e-9:
            assert np.isclose(f[members].sum(), base[bi], rtol=1e-9)


# --- force conservation --------------------------------------------------

def _pressed(s, depth_mm=0.3):
    return sg.plane_surface(s.p.height - depth_mm * MM)


def test_taxel_forces_sum_to_the_bump_base_forces(any_board):
    s = any_board
    surf = _pressed(s)
    assert np.isclose(s.taxel_forces(surf).sum(),
                      s.surrogate.base_forces(surf).sum(), rtol=1e-9)


def test_a_moment_redistributes_load_without_creating_any(any_board):
    """A couple must move load inside a cell, never add to it -- including on
    a degenerate block whose offsets do not cancel."""
    s = any_board
    surf = _pressed(s)
    base = s.surrogate.base_forces(surf)
    rng = np.random.default_rng(0)
    m = rng.normal(0.0, 2e-4, (s.n_bumps, 2))

    f = s.taxel_forces(surf, moments=m)
    assert np.isclose(f.sum(), base.sum(), rtol=1e-6)
    assert (f >= 0).all(), "a taxel cannot pull"


def test_wrench_distribution_also_conserves(any_board):
    """taxel_forces_from_wrench is the path the full MuJoCo lattice uses."""
    s = any_board
    fz = np.linspace(0.05, 0.5, s.n_bumps)
    m = np.zeros((s.n_bumps, 2))
    f = s.taxel_forces_from_wrench(fz, m)
    assert np.isclose(f.sum(), fz.sum(), rtol=1e-9)
    for bi in range(s.n_bumps):
        members = np.nonzero(s.taxel_bump == bi)[0]
        assert np.isclose(f[members].sum(), fz[bi], rtol=1e-9)


# --- the shear channel ---------------------------------------------------

def test_decode_shear_recovers_an_applied_moment(any_board):
    """Shear is never measured, only decoded from four normal readings.

    A small moment keeps every taxel in compression, so no clipping -- this
    checks the arithmetic, not the saturation, which is tested separately.
    """
    s = any_board
    fz = np.full(s.n_bumps, 0.6)
    rng = np.random.default_rng(1)
    m = rng.normal(0.0, 3e-5, (s.n_bumps, 2))
    f = s.taxel_forces_from_wrench(fz, m)

    got = s.decode_shear(f)
    h = s.p.height
    want = np.column_stack([2.0 * m[:, 1] / h, -2.0 * m[:, 0] / h])
    full = s.group_sizes() == 4          # degenerate groups are exempt
    assert np.allclose(got[full], want[full], rtol=1e-6, atol=1e-9)


def test_no_shear_is_decoded_from_a_flat_press(any_board):
    """A uniform plane tips nothing, so the shear channel must read zero.

    The disc is the case that matters: its 9 cut-short blocks have taxels on
    one side only, so a decoder referenced to the bump centre would read shear
    from a pure normal press.
    """
    s = any_board
    f = s.taxel_forces(_pressed(s))
    assert np.abs(s.decode_shear(f)).max() < 1e-9


def test_checkering_measures_the_dipole_not_any_unevenness(square):
    """A monopole (all cells loaded evenly-but-heavily) is not checkering."""
    fz = np.full(square.n_bumps, 1.0)
    even = square.taxel_forces_from_wrench(fz, np.zeros((square.n_bumps, 2)))
    assert square.checkerness(even) < 1e-9

    # An exact expectation, not a threshold. The couple on a taxel is
    # My*dx/sum(dx^2); with four taxels at +/-a a moment of My = fz*a/2 puts
    # a fraction p = 0.75 of the cell's load on one edge. For an edge split
    # the dipole is fz*(2p-1)*a while the normaliser is the diagonal reach
    # a*sqrt(2), so checkerness must be exactly (2p-1)/sqrt(2).
    arm = square.pitch[0] / 2.0
    lop = square.taxel_forces_from_wrench(
        fz, np.tile([0.0, 1.0 * arm / 2.0], (square.n_bumps, 1)))
    assert square.checkerness(lop) == pytest.approx(0.5 / np.sqrt(2), rel=1e-9)

    # ...and a corner-loaded cell is the 1.0 case the docstring names.
    corner = np.zeros(square.n_taxels)
    first = np.nonzero(square.taxel_bump == 0)[0]
    corner[first[0]] = 1.0
    assert square.checkerness(corner) == pytest.approx(1.0, rel=1e-9)


# --- image ordering ------------------------------------------------------

def _components(img, thr):
    """Count 4-connected blobs of cells above `thr`."""
    on = img > thr
    seen = np.zeros_like(on, dtype=bool)
    n = 0
    for i in range(on.shape[0]):
        for j in range(on.shape[1]):
            if on[i, j] and not seen[i, j]:
                n += 1
                stack = [(i, j)]
                seen[i, j] = True
                while stack:
                    a, b = stack.pop()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        p, q = a + da, b + db
                        if (0 <= p < on.shape[0] and 0 <= q < on.shape[1]
                                and on[p, q] and not seen[p, q]):
                            seen[p, q] = True
                            stack.append((p, q))
    return n


def test_geometric_order_keeps_one_contact_in_one_piece(square):
    """`pixels`' row/col is pad order, and split_row_pin_order reverses the
    bottom half of the rows, so in pad order a single round contact plots as
    two blobs at opposite ends of the image, and a convolution over that, as
    a policy would apply, reads a scrambled picture.
    """
    cx, cy = square.centre()
    surf = sg.sphere_surface(0.25, square.p.height - 0.4e-3, cx, cy)
    f = square.taxel_forces(surf)
    assert (f > 1e-6).sum() >= 4, "need a real contact patch to test"

    geo = square.to_image(f)
    assert _components(geo, 1e-6) == 1, "geometric order must stay contiguous"


def test_scan_order_is_a_different_ordering(square):
    """Guards the premise of the test above: the two orderings differ,
    so 'geometric' is not silently returning the manifest's own indices."""
    assert not np.array_equal(square.geometric_rc(), square.rc)


def test_both_orderings_carry_the_same_values(square):
    s = square
    f = s.taxel_forces(_pressed(s))
    assert np.isclose(s.to_image(f, order="geometric").sum(), f.sum())
    assert np.isclose(s.to_image(f, order="scan").sum(), f.sum())
    with pytest.raises(ValueError):
        s.to_image(f, order="sideways")


def test_scan_image_matches_the_manifest_indices(square):
    """Whatever geometric order does, `order="scan"` has to reproduce exactly
    what the firmware streams, or hardware comparisons are meaningless."""
    s = square
    f = s.taxel_forces(_pressed(s))
    img = s.to_image(f, order="scan")
    for k, (r, c) in enumerate(s.rc):
        assert img[r, c] == f[k]


# --- refusals ------------------------------------------------------------

def test_a_curved_sensor_is_refused_not_silently_flattened(square):
    """A wrapped fingertip modelled as a flat pad gets every contact normal
    wrong, and the readings still look entirely reasonable."""
    man = copy.deepcopy(square.manifest)
    man["sim3d"] = {"taxels3d": {"pos": [[0, 0, 0]] * len(man["pixels"])},
                    "stack": {}}
    with pytest.raises(ManifestError, match="curved"):
        TactileSensor.from_manifest(man)


def test_a_manifest_without_the_shear_block_is_refused(square):
    man = copy.deepcopy(square.manifest)
    man["v"] = 4
    man.pop("shear", None)
    with pytest.raises(ManifestError, match="shear"):
        TactileSensor.from_manifest(man)


def test_an_old_version_number_with_a_shear_block_is_accepted(square):
    """The loader checks the block, not the version number.

    Version numbers travel differently from features: an export can carry
    "v": 3 with a complete shear block. Gating on v >= 5 would refuse a sensor
    this package can simulate.
    """
    man = copy.deepcopy(square.manifest)
    man["v"] = 3
    s = TactileSensor.from_manifest(man)
    assert s.v == 3
    assert s.n_bumps == square.n_bumps
    assert "v3" in s.describe()


def test_a_manifest_with_no_bumps_is_refused(square):
    man = copy.deepcopy(square.manifest)
    man["shear"] = {"bumps": [], "b": []}
    with pytest.raises(ManifestError, match="shear"):
        TactileSensor.from_manifest(man)


def test_a_desynchronised_shear_block_is_refused(square):
    """'shear.b' is specified as index-parallel to 'pixels', and consumers rely
    on that, so a mismatch is refused."""
    man = copy.deepcopy(square.manifest)
    man["shear"] = dict(man["shear"])
    man["shear"]["b"] = man["shear"]["b"][:-3]
    with pytest.raises(ManifestError, match="parallel"):
        TactileSensor.from_manifest(man)


# --- probing -------------------------------------------------------------

def test_probe_presses_to_a_force_not_a_depth(square):
    """Probes press to a force, not a depth: at a fixed depth the object's own
    curvature decides which bumps it reaches, and 0.6 mm of flat plate on a
    real board is 916 N."""
    for target in (2.0, 8.0):
        img = square.probe("plane", force_n=target)
        assert np.isclose(img.sum(), target, rtol=1e-3)


def test_probe_rejects_an_unknown_shape(square):
    with pytest.raises(ValueError, match="unknown probe shape"):
        square.probe("banana", force_n=1.0)


def test_a_twist_raises_the_shear_induced_checkering(square):
    """The claim the demo makes, as an assertion.

    Against the baseline-subtracted measure: a ball's curvature already makes
    a radial dipole in every off-centre cell, and the shear dipole is
    tangential, so the raw number dips before it climbs.
    """
    res = square.press_and_twist([0.0, 0.5, 1.0, 2.0], shape="sphere",
                                 force_n=8.0, radius_mm=250)
    ck = [r["checkerness_shear"] for r in res]
    assert ck[0] == pytest.approx(0.0, abs=1e-12)
    assert all(b >= a - 1e-12 for a, b in zip(ck, ck[1:])), ck
    assert ck[-1] > ck[0]
    # Normal force is held constant, so nothing here is a harder press.
    assert np.allclose([r["total_n"] for r in res], res[0]["total_n"], rtol=1e-6)


# --- randomisation hooks -------------------------------------------------

def test_changing_stiffness_changes_the_reading(square):
    """Assigning a new `p` changes the reading.

    The surrogate caches stiffness at construction, so `p` is a property that
    updates it. Otherwise `sensor.p = softer` would change nothing and report
    nothing, and an RL loop randomising per episode would train against one
    fixed stiffness while every number still looked reasonable.
    """
    import dataclasses

    surf = _pressed(square)
    stiff = square.taxel_forces(surf).sum()

    square.p = dataclasses.replace(square.p, e_eff_mpa=square.p.e_eff_mpa / 4.0)
    soft = square.taxel_forces(surf).sum()
    assert soft < stiff * 0.6, (stiff, soft)
    assert square.surrogate.p is square.p, "surrogate must follow the sensor"

    square.p = dataclasses.replace(square.p, e_eff_mpa=square.p.e_eff_mpa * 4.0)
    assert square.taxel_forces(surf).sum() == pytest.approx(stiff, rel=1e-9)


def test_set_params_returns_an_independent_sensor(square):
    """A vectorised env holds several stiffnesses at once; they must not alias."""
    soft = square.set_params(e_eff_mpa=3.0)
    assert soft is not square
    assert soft.p.e_eff_mpa == 3.0
    assert square.p.e_eff_mpa != 3.0
    assert soft.surrogate is not square.surrogate
    # Same board, though: geometry is not a parameter.
    assert np.allclose(soft.xy, square.xy)
    assert np.array_equal(soft.taxel_bump, square.taxel_bump)

    surf = _pressed(square)
    assert soft.taxel_forces(surf).sum() < square.taxel_forces(surf).sum()


def test_the_estimate_ranges_are_shipped_and_bracket_the_defaults():
    """The estimate ranges are published and contain the defaults."""
    from opentac.sim.mujoco.params import ESTIMATE_RANGES

    d = BumpParams()
    for field, (lo, hi) in ESTIMATE_RANGES.items():
        assert lo < hi
        assert lo <= getattr(d, field) <= hi, field


# --- damping mode --------------------------------------------------------

def test_damping_modes_measure_zeta_against_different_masses():
    """`armature` inflates the joint's mass by about 260x, and the default
    damping formula derives zeta from that mass, so the real bump experiences
    a much higher damping ratio than requested (11.3 where 0.7 is asked for).
    """
    import dataclasses
    import math

    from opentac.sim.mujoco.params import BumpParams as BP

    stable = BP(taxel_pitch_mm=(2.5, 2.5), e_eff_mpa=3.5)
    physical = dataclasses.replace(stable, damping_mode="physical")
    k = stable.k_axial

    c_stable = stable.damping(k)
    c_phys = physical.damping(k)
    assert c_phys == pytest.approx(2 * stable.zeta * math.sqrt(k * stable.mass))
    assert c_stable == pytest.approx(2 * stable.zeta * k / stable.omega_max)

    # The two disagree by exactly omega_true / omega_max, the factor by which
    # armature slows the joint down. This identity is why the discrepancy
    # grows with stiffness: 7.8x at 3.5 MPa, 16.1x at the 15 MPa default.
    omega_true = math.sqrt(k / stable.mass)
    assert c_stable / c_phys == pytest.approx(omega_true / stable.omega_max)
    assert c_stable > c_phys, "default over-damps the real bump"

    # zeta as the real bump experiences it, against the 0.7 asked for.
    zeta_real = c_stable / (2 * math.sqrt(k * stable.mass))
    assert zeta_real == pytest.approx(
        stable.zeta * omega_true / stable.omega_max)
    assert physical.damping(k) / (2 * math.sqrt(k * stable.mass)) == \
        pytest.approx(stable.zeta)


def test_damping_mode_does_not_touch_statics():
    """Damping cannot move an equilibrium: at rest F = k*x and mass and
    damping both drop out. Switching modes therefore leaves every quasi-static
    quantity, and with it every validated result, unchanged."""
    import dataclasses

    from opentac.sim.mujoco.params import BumpParams as BP

    a = BP(taxel_pitch_mm=(2.5, 2.5), e_eff_mpa=3.5)
    b = dataclasses.replace(a, damping_mode="physical")
    for attr in ("k_axial", "k_axial_effective", "k_shear_x", "k_tip_x",
                 "mass", "height", "r_base", "r_top"):
        assert getattr(a, attr) == getattr(b, attr), attr

    sq = load("flat_square.json", bump=a)
    sqp = load("flat_square.json", bump=b)
    surf = _pressed(sq)
    assert np.allclose(sq.taxel_forces(surf), sqp.taxel_forces(surf))


def test_hinges_are_damped_against_rotational_inertia():
    """A hinge's damping is N*m*s/rad, so it uses an inertia, not a mass.
    Passing the bump's mass there would be dimensionally wrong."""
    import dataclasses

    from opentac.sim.mujoco.params import BumpParams as BP

    p = dataclasses.replace(BP(taxel_pitch_mm=(2.5, 2.5), e_eff_mpa=3.5),
                            damping_mode="physical")
    assert p.rot_inertia == pytest.approx(p.mass * p.height ** 2 / 3.0)
    assert p.damping(p.k_tip_x, p.rot_inertia) != p.damping(p.k_tip_x)


def test_an_unknown_damping_mode_is_refused():
    import dataclasses

    from opentac.sim.mujoco.params import BumpParams as BP

    p = dataclasses.replace(BP(taxel_pitch_mm=(2.5, 2.5)), damping_mode="soft")
    with pytest.raises(ValueError, match="damping_mode"):
        p.damping(p.k_axial)
