# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SO-101 MuJoCo Tactile contributors.
# From TactileSim (tests/tactilesim/test_contracts.py, test_logic_audit.py, test_loader.py,
# test_readout.py); modified for OpenTac.
"""The bump model's core (opentac.sim.mujoco): force solve, input checks, moment split, readout checks.

Each test pins a case in which an unchecked model returns a plausible but
wrong result: a total force that is not the requested one, contact with an
object that is nowhere near the board, load that leaves a bump, or a pitch
that depends on the scan order.
"""
import copy
import glob
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from opentac import manifest
from opentac.manifest import ManifestError
from opentac.sim import Sensor
from opentac.sim.mujoco import BumpParams, TactileSensor
from opentac.sim.mujoco import surrogate as sg
from opentac.sim.readout import Readout

ROOT = Path(__file__).resolve().parents[2]
FIX = Path(__file__).parent / "fixtures"
SQUARE = FIX / "flat_square.json"
DISC = FIX / "flat_disc.json"
SITE_ROOT = Path(os.environ.get("OPENTAC_SITE_ROOT", ROOT.parent / "opentac.github.io")).resolve()


def square(**params):
    s = TactileSensor.from_manifest(str(SQUARE))
    return s.set_params(**params) if params else s


# --- force solve ------------------------------------------------------------

@pytest.mark.parametrize("height", [2., 4., 5.])
def test_force_solver_brackets_tall_bumps(height):
    """The search interval widens with the bump height: on 4 and 5 mm bumps a fixed -1 to 3 mm
    interval would give 337.65 N and 556.60 N instead of 5 N."""
    s = square(bump_height_mm=height)
    assert s.probe("plane", force_n=5., as_image=False).sum() == pytest.approx(5., rel=1e-7)


@pytest.mark.parametrize("force", [-1., np.nan, np.inf])
def test_force_solver_rejects_invalid_targets(force):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        square().press(force_n=force)


def test_force_target_outside_lumped_compression_domain_is_rejected():
    """1e6 N would compress the bumps past their height, where a clipped total (1766 N) is not a valid answer."""
    with pytest.raises(ValueError, match="flatten"):
        square().press("plane", force_n=1e6)


@pytest.mark.parametrize("shape", ["sphere", "cylinder"])
def test_remote_probe_has_no_fictitious_contact(shape):
    """Beyond its radius an object does not touch the board, at any depth."""
    s = square()
    with pytest.raises(ValueError, match="cannot be bracketed"):
        s.probe(shape, force_n=1., radius_mm=10., cx_mm=1000.)
    assert np.all(s.probe(shape, force_n=0., cx_mm=1000., as_image=False) == 0)


def test_surfaces_end_at_their_radius():
    x = np.array([0.0, 0.009, 0.011])
    np.testing.assert_array_equal(np.isinf(sg.sphere_surface(0.01, 0.0)(x, 0 * x)), [False, False, True])
    np.testing.assert_array_equal(np.isinf(sg.cylinder_surface(0.01, 0.0, "y")(x, 0 * x)), [False, False, True])


def test_load_on_unassigned_taxel_is_included_in_force_solve():
    """A taxel with no bump carries part of the object's load; the requested force is the total."""
    data = copy.deepcopy(square().manifest)
    data["shear"]["b"][0] = -1
    s = TactileSensor.from_manifest(data)
    x, y = s.xy[0] * 1000
    forces = s.probe("sphere", force_n=1., radius_mm=.2, cx_mm=x, cy_mm=y, as_image=False)
    assert forces[0] == pytest.approx(1., rel=1e-8)
    assert forces.sum() == pytest.approx(1., rel=1e-8)


# --- moment split and load conservation -----------------------------------

def test_wrench_clipping_preserves_each_bump_load():
    s = square()
    load = np.ones(s.n_bumps)
    moments = np.tile([s.p.height / np.sqrt(8), -s.p.height / np.sqrt(8)], (s.n_bumps, 1))
    f = s.taxel_forces_from_wrench(load, moments)
    np.testing.assert_allclose(np.bincount(s.taxel_bump, weights=f, minlength=s.n_bumps), load, atol=1e-12)
    assert np.all(f >= 0)


@pytest.mark.parametrize("path", ["surface", "wrench"])
@pytest.mark.parametrize("moments", [[[0., 0.]], np.nan, np.inf])
def test_both_force_paths_reject_invalid_moments(path, moments):
    s = square()
    if np.isscalar(moments):
        moments = np.full((s.n_bumps, 2), moments)
    with pytest.raises(ValueError, match="Moments"):
        if path == "surface":
            surface, _ = s.press("plane", force_n=5.)
            s.taxel_forces(surface, moments=moments)
        else:
            s.taxel_forces_from_wrench(np.ones(s.n_bumps), moments)


def test_tiny_surface_load_is_conserved_after_moment_clipping():
    s = square(e_eff_mpa=1e-6)
    surface = lambda x, y: np.full_like(x, np.nextafter(s.p.height, 0.))  # noqa: E731
    base = s.surrogate.base_forces(surface)
    forces = s.taxel_forces(surface, moments=np.tile([0., 1e-19], (s.n_bumps, 1)))
    assert (base > 0).all()
    np.testing.assert_allclose(np.bincount(s.taxel_bump, weights=forces, minlength=s.n_bumps), base, rtol=1e-12, atol=0.)


def test_three_taxel_groups_recover_both_moment_components():
    s = TactileSensor.from_manifest(str(DISC))
    moments = np.tile([1e-5, 2e-5], (s.n_bumps, 1))
    forces = s.taxel_forces_from_wrench(np.ones(s.n_bumps), moments)
    observable = s.group_sizes() >= 3
    expected = np.tile([2e-5 * 2 / s.p.height, -1e-5 * 2 / s.p.height], (observable.sum(), 1))
    np.testing.assert_allclose(s.decode_shear(forces)[observable], expected, atol=1e-12)
    assert np.all(forces > 0)
    np.testing.assert_allclose(np.bincount(s.taxel_bump, weights=forces), 1., atol=1e-12)


def test_rank_one_group_recovers_its_observable_moment():
    s = TactileSensor.from_manifest(str(DISC))
    centred = s._centred_offsets()
    k = int(np.flatnonzero(s.group_sizes() == 2)[0])
    members = np.flatnonzero(s.taxel_bump == k)
    expected_forces = np.zeros(s.n_taxels)
    expected_forces[members] = [.45, .55]
    mx = -float(expected_forces @ centred[:, 1])
    my = float(expected_forces @ centred[:, 0])
    moments = np.zeros((s.n_bumps, 2))
    moments[k] = [mx, my]
    base = np.zeros(s.n_bumps)
    base[k] = 1.
    np.testing.assert_allclose(s.taxel_forces_from_wrench(base, moments), expected_forces, atol=1e-12)


# --- manifest and parameter checks --------------------------------------------

def test_pitch_ignores_scan_order():
    s = square()
    shuffled = copy.deepcopy(s.manifest)
    rng = np.random.default_rng(0)
    rc = np.array([p[2:4] for p in shuffled["pixels"]])
    rows, cols = rng.permutation(rc[:, 0].max() + 1), rng.permutation(rc[:, 1].max() + 1)
    for p in shuffled["pixels"]:
        p[2], p[3] = int(rows[p[2]]), int(cols[p[3]])
    assert TactileSensor(shuffled).pitch == pytest.approx(s.pitch)


@pytest.mark.parametrize("index", [-2, 999, .5])
def test_invalid_bump_assignment_is_rejected(index):
    data = json.loads(SQUARE.read_text())
    data["shear"]["b"][0] = index
    with pytest.raises(ManifestError, match="shear.b"):
        TactileSensor.from_manifest(data)


def _edit(change):
    data = json.loads(SQUARE.read_text())
    change(data)
    return data


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(units="in"), "mm"),
    (lambda d: d["pixels"][1].__setitem__(slice(0, 2), d["pixels"][0][:2]), "unique"),
    (lambda d: d["pixels"][1].__setitem__(slice(2, 4), d["pixels"][0][2:4]), "unique"),
    (lambda d: d["pixels"][0].__setitem__(2, 0.5), "integers"),
    (lambda d: d["pixels"][0].__setitem__(2, 99), "outside the sensor dimensions"),
    (lambda d: d["pixels"][0].__setitem__(0, float("nan")), "finite"),
    (lambda d: d["shear"]["bumps"].__setitem__(1, d["shear"]["bumps"][0]), "unique"),
    (lambda d: d["shear"]["bumps"].append([100.0, 100.0]), "assigned taxel"),
    (lambda d: d.update(w=3.0, h=3.0), "overlap"),
    (lambda d: d.update(bump_params={"taxel_pitch_mm": (3.0, 3.0)}), "taxel_pitch_mm"),
])
def test_malformed_manifests_are_refused(change, message):
    with pytest.raises(ManifestError, match=message):
        TactileSensor.from_manifest(_edit(change))


def test_touching_footprints_are_accepted():
    """OpenTac-CAD allows a zero gap; manifest rounding can then make a footprint 1 um wider than the pitch."""
    pitch = square().pitch[0] * 1e3
    TactileSensor.from_manifest(_edit(lambda d: d.update(w=pitch + 0.001, h=pitch + 0.001)))
    with pytest.raises(ManifestError, match="overlap"):
        TactileSensor.from_manifest(_edit(lambda d: d.update(w=pitch + 0.02)))


def test_overlapping_bump_bases_are_rejected():
    with pytest.raises(ManifestError, match="overlap"):
        square().set_params(bump_base_frac=1.1)
    square().set_params(bump_base_frac=1.0)  # touching bases are accepted


def test_bump_params_from_the_manifest_are_read():
    s = TactileSensor.from_manifest(_edit(lambda d: d.update(bump_params={"bump_height_mm": 3.0})))
    assert s.p.bump_height_mm == 3.0 and s.p.taxel_pitch_mm == pytest.approx((2.5, 2.5))


@pytest.mark.parametrize("field, value", [
    ("e_eff_mpa", 0.0), ("e_eff_mpa", np.nan), ("density", -1.0), ("bump_height_mm", 0.0),
    ("sheet_thickness_mm", np.inf), ("bump_base_frac", 0.0), ("omega_max", 0.0),
    ("taxel_pitch_mm", (2.5,)), ("taxel_pitch_mm", (2.5, -1.0)), ("poisson", 0.5), ("poisson", -1.0),
    ("bump_top_ratio", 0.0), ("bump_top_ratio", 1.5), ("mu", -0.1), ("zeta", np.nan),
])
def test_bump_params_refuse_nonphysical_values(field, value):
    with pytest.raises(ValueError):
        BumpParams(**{field: value})


# --- every real manifest loads ------------------------------------------------

def _real_manifests():
    paths = [SQUARE, DISC]
    paths += sorted((ROOT / "examples" / "manifests").glob("*.json"))
    paths += sorted((ROOT / "tests" / "readout" / "fixtures" / "stm_test").glob("*.json"))
    return paths


@pytest.mark.parametrize("path", _real_manifests(), ids=lambda p: p.name)
def test_real_manifests_load_and_conserve_a_press(path):
    s = Sensor(manifest.load(path), model="mujoco")
    assert s.press("sphere", force_n=5.0, radius_mm=10.0).taxels.sum() == pytest.approx(5.0, rel=1e-7)


def test_the_published_example_designs_load_and_conserve_a_press():
    """The 31 ready-to-order designs of the website (docs/designs/files/*.zip)."""
    zips = sorted(glob.glob(str(SITE_ROOT / "docs" / "designs" / "files" / "*.zip")))
    if not zips:
        if "OPENTAC_SITE_ROOT" in os.environ:
            pytest.fail("OPENTAC_SITE_ROOT has no docs/designs/files/*.zip")
        pytest.skip("needs the website checkout; set OPENTAC_SITE_ROOT")
    for z in zips:
        with zipfile.ZipFile(z) as f:
            name = next(n for n in f.namelist() if n.endswith("manifest.json"))
            raw = json.loads(f.read(name))
        s = Sensor(raw, model="mujoco")
        total = s.press("sphere", force_n=5.0, radius_mm=10.0).taxels.sum()
        assert total == pytest.approx(5.0, rel=1e-7), Path(z).name


# --- the MuJoCo scenes ----------------------------------------------------------

def test_scene_bumps_are_frustums_at_full_precision():
    """Every scene builds its bumps with one frustum builder, with exact positions and masses."""
    mujoco = pytest.importorskip("mujoco")
    from opentac.sim.mujoco import Gripper, Lattice, SensorScene

    data = json.loads(SQUARE.read_text())
    scale = .1234567
    for pixel in data["pixels"]:
        pixel[0] *= scale
        pixel[1] *= scale
    data["shear"]["bumps"] = (np.asarray(data["shear"]["bumps"]) * scale).tolist()
    data["w"] *= scale
    data["h"] *= scale
    small = TactileSensor.from_manifest(data)
    scene = SensorScene(small, 0.002)
    for i, xy in enumerate(small.bump_xy):
        np.testing.assert_allclose(scene.model.body(f"bump_{i}").pos, [*xy, 0], atol=1e-16, rtol=0)
        np.testing.assert_allclose(scene.model.body(f"bump_{i}").mass, small.p.mass, rtol=1e-12)
    lattice = Lattice(BumpParams(taxel_pitch_mm=(2.5, 2.5)), nx=2, ny=2)
    gripper = Gripper(BumpParams(taxel_pitch_mm=(3.0, 3.0)), pad_half_mm=6.0)
    for model, names in ((scene.model, ["bg_0"]), (lattice.model, ["bg_0"]), (gripper.model, ["lg_0", "ug_0"])):
        for name in names:
            assert model.geom(name).type == mujoco.mjtGeom.mjGEOM_MESH, name
    # The upper jaw's bumps point down from the jaw: MuJoCo places each mesh
    # geom at its centroid, below the body for the mirrored mesh.
    assert gripper.model.geom("ug_0").pos[2] < 0 < gripper.model.geom("lg_0").pos[2]


# --- readout ----------------------------------------------------------------

@pytest.mark.parametrize("gain", [0., -1., np.nan, np.inf])
def test_readout_rejects_invalid_gain_in_both_directions(gain):
    r = Readout.from_profile("v3")
    with pytest.raises(ValueError, match="gain"):
        r.counts([.01], .002, gain=gain)
    with pytest.raises(ValueError, match="gain"):
        r.forces([100.], .002, gain=gain)


def test_readout_rejects_nonfinite_force_and_adc():
    r = Readout.from_profile("v3")
    with pytest.raises(ValueError, match="finite"):
        r.counts([np.nan], .002)
    with pytest.raises(ValueError, match="finite"):
        r.forces([np.nan], .002)


@pytest.mark.parametrize("parameter", ["b", "k", "adc_max"])
def test_readout_refuses_nonphysical_model_parameters(parameter):
    config = dict(name="test", material="", b=.5, k=2., s=.5, adc_max=4095., noise_counts=1.)
    config[parameter] = 0.
    with pytest.raises(ValueError, match="positive"):
        Readout(**config)


def test_profile_name_is_not_shadowed_by_a_local_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "v3").mkdir()
    assert Readout.from_profile("v3").name == "v3"
