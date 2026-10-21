"""opentac.sim.Sensor: one entry point, two models, Frames out."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from opentac import manifest
from opentac.frame import Frame
from opentac.manifest import ManifestError
from opentac.sim import FoamSensor, MujocoSensor, Sensor

FIX = Path(__file__).parent / "fixtures"
SQUARE = FIX / "flat_square.json"


def test_default_model_is_foam_even_with_a_shear_block():
    """OpenTac-CAD writes a `shear` block on every export, printed sheet or not,
    so the block cannot say whether bumps exist. The default stays foam."""
    m = manifest.load(SQUARE)
    assert m.get("shear")
    assert isinstance(Sensor(m), FoamSensor)
    assert isinstance(Sensor(m, model="mujoco"), MujocoSensor)
    with pytest.raises(ValueError, match="model"):
        Sensor(m, model="fem")


def test_foam_frame_is_in_manifest_order():
    m = manifest.load(SQUARE)
    s = Sensor(m)
    xy = m.xy[7]
    f = s.press("sphere", center_mm=xy, force_n=2.0, radius_mm=3.0)
    assert isinstance(f, Frame) and f.units == "N" and f.imu == {}
    assert f.mask.all() and len(f.taxels) == m.n
    assert int(np.argmax(f.taxels)) == 7
    img = f.image(m)
    assert np.nansum(img) == pytest.approx(f.taxels.sum())


def test_mujoco_needs_a_shear_block():
    raw = json.loads(SQUARE.read_text())
    raw.pop("shear")
    assert isinstance(Sensor(raw), FoamSensor)
    with pytest.raises(ManifestError, match="shear"):
        Sensor(raw, model="mujoco")


@pytest.mark.parametrize("model", ["foam", "mujoco"])
def test_curved_sensors_are_refused(model):
    """A wrapped fingertip simulated as a flat pad gets every contact normal
    wrong and still produces reasonable-looking readings."""
    raw = copy.deepcopy(json.loads(SQUARE.read_text()))
    raw["sim3d"] = {"taxels3d": {"pos": [[0, 0, 0]] * len(raw["pixels"])}}
    with pytest.raises(ManifestError, match="curved"):
        Sensor(raw, model=model)


def test_mujoco_press_conserves_force_and_keeps_order():
    m = manifest.load(SQUARE)
    s = Sensor(m, model="mujoco")
    f = s.press("sphere", force_n=6.0, radius_mm=30.0)
    assert f.units == "N" and len(f.taxels) == m.n
    assert f.taxels.sum() == pytest.approx(6.0, rel=1e-6)
    # same numbers as the underlying TactileSensor, in the same order
    np.testing.assert_allclose(f.taxels, s.tactile.probe("sphere", force_n=6.0, radius_mm=30.0, as_image=False))


def test_mujoco_params_keep_the_manifest_pitch():
    """A fresh BumpParams would silently use the 4.2 mm default pitch."""
    s = Sensor(SQUARE, model="mujoco", e_eff_mpa=3.5)
    assert s.tactile.p.e_eff_mpa == 3.5
    assert s.tactile.p.taxel_pitch_mm == pytest.approx((2.5, 2.5))


def test_mujoco_twist_decodes_shear():
    s = Sensor(SQUARE, model="mujoco")
    frames = s.twist([0.0, 0.5, 1.0, 2.0], force_n=8.0, radius_mm=250.0)
    assert len(frames) == 4
    base = s.decode_shear(frames[0])
    assert base.shape == (s.tactile.n_bumps, 2)
    grow = [np.abs(s.decode_shear(f) - base).sum() for f in frames]
    assert grow[0] == 0.0 and grow[1] > 0.0 and grow[-1] >= grow[1]
    # the normal force is held while the object turns
    assert all(f.taxels.sum() == pytest.approx(8.0, rel=1e-6) for f in frames)
    assert s.bump_xy_mm.shape == (s.tactile.n_bumps, 2)
    assert s.degenerate_axes().shape == (s.tactile.n_bumps, 2)


def test_readout_gives_counts_and_shear_needs_newtons():
    s = Sensor(SQUARE, model="mujoco", readout="v2")
    f = s.press("sphere", force_n=1.0, radius_mm=30.0)
    assert f.units == "counts" and f.taxels.max() > 0
    with pytest.raises(ValueError, match="newtons"):
        s.decode_shear(f)


def test_mujoco_rejects_an_angle():
    with pytest.raises(ValueError, match="angle"):
        Sensor(SQUARE, model="mujoco").press("cylinder", angle_deg=30.0)


# --- the full MuJoCo solver (needs the `sim` extra) -------------------------

def test_full_solver_press_reaches_the_force():
    pytest.importorskip("mujoco")
    s = Sensor(SQUARE, model="mujoco", e_eff_mpa=3.5)
    f = s.press("sphere", force_n=12.0, radius_mm=20.0, solver="mujoco")
    sur = s.press("sphere", force_n=12.0, radius_mm=20.0)
    # The ball stops at the first step past the target, so it can overshoot.
    assert 12.0 <= f.taxels.sum() < 12.0 * 1.1
    # Same contact: the same bumps carry load. (Within one bump the two solvers
    # split the load over its taxels differently, so compare per bump.)
    ts = s.tactile
    per_bump = []
    for frame in (f, sur):
        load = np.zeros(ts.n_bumps)
        np.add.at(load, ts.taxel_bump, frame.taxels)
        per_bump.append(load)
    assert set(np.nonzero(per_bump[0] > 5e-3)[0]) == set(np.nonzero(per_bump[1] > 5e-3)[0])


def test_full_solver_twist_loads_the_shear_channel():
    pytest.importorskip("mujoco")
    s = Sensor(SQUARE, model="mujoco", e_eff_mpa=3.5)
    frames = s.twist([0.0, 2.0, 4.0], force_n=12.0, radius_mm=20.0, solver="mujoco")
    base = s.decode_shear(frames[0])
    d = [np.abs(s.decode_shear(f) - base).max() for f in frames]
    assert d[0] == 0.0 and d[1] > 0.1 and d[2] > d[1]
    with pytest.raises(ValueError, match="sphere"):
        s.press("plane", solver="mujoco")


def test_lattice_and_gripper_build():
    """The regular-lattice studies the model was validated with build and run."""
    pytest.importorskip("mujoco")
    from opentac.sim.mujoco import BumpParams, Gripper, Lattice

    p = BumpParams(taxel_pitch_mm=(2.5, 2.5), e_eff_mpa=3.5)
    lat = Lattice(p, nx=3, ny=3, indenter="flat")
    depth, total = lat.press_to_force(5.0)
    assert total == pytest.approx(5.0, rel=0.05)
    # The configuration of the parallel-jaw grip study
    g = Gripper(BumpParams(taxel_pitch_mm=(3.0, 3.0)), obj="sphere", obj_radius_mm=20.0, pad_half_mm=12.0)
    squeeze, grip = g.close_to_force(15.0)
    assert grip == pytest.approx(15.0, rel=0.05)
    assert (g.pad_forces() > 0.01).sum() >= 1
