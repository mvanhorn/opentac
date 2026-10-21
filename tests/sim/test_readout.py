"""The readout is the inverse of itself, refuses the wrong units, and reports
when a simulated press is outside what the calibration covered.

The profiles are fitted from bench recordings that are not in this repository,
so nothing here re-derives a fit; these tests check the model's contract
instead.
"""
import os

import numpy as np
import pytest

from opentac.sim.mujoco import TactileSensor
from opentac.sim.readout import Readout, available_profiles

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
PITCH = (2e-3, 2e-3)


@pytest.fixture(params=["v2", "v3"])
def readout(request):
    return Readout.from_profile(request.param)


def test_shipped_profiles_are_listed_with_their_material():
    shipped = available_profiles()
    assert shipped["v2"] == "LINQSTAT"
    assert shipped["v3"] == "Yunjia Textiles"


@pytest.mark.parametrize("query", ["v3", "V3", "Yunjia Textiles", "yunjia"])
def test_profile_by_label_or_material(query):
    assert Readout.from_profile(query).name == "v3"


def test_unknown_profile_names_the_shipped_ones():
    with pytest.raises(FileNotFoundError, match="v3"):
        Readout.from_profile("no-such-velostat")


def test_profile_by_path_is_the_same_as_by_name(readout):
    from opentac.sim.readout import PROFILE_DIR
    by_path = Readout.from_profile(os.path.join(PROFILE_DIR, f"{readout.name}.json"))
    assert by_path == readout


def test_counts_and_forces_invert(readout):
    # stay above 3x noise and below the ADC ceiling, where the inverse is defined
    lo = readout.forces([10 * readout.noise_counts], PITCH)[0]
    hi = readout.saturation_force_n(PITCH) * 0.9
    f = np.geomspace(lo, hi, 25)
    back = readout.forces(readout.counts(f, PITCH), PITCH)
    np.testing.assert_allclose(back, f, rtol=1e-9)


def test_monotone_zero_at_rest_and_clipped(readout):
    f = np.linspace(0.0, 2.0 * readout.saturation_force_n(PITCH), 200)
    c = readout.counts(f, PITCH)
    assert c[0] == 0.0
    assert np.all(np.diff(c) >= 0)
    assert c.max() == readout.adc_max


def test_saturation_force_is_where_the_adc_tops_out(readout):
    fs = readout.saturation_force_n(PITCH)
    assert readout.counts([fs * 0.99], PITCH)[0] < readout.adc_max
    assert readout.counts([fs * 1.01], PITCH)[0] == readout.adc_max


def test_gain_scales_counts_and_is_undone_by_forces(readout):
    f = np.array([readout.saturation_force_n(PITCH) * 0.1])
    c1 = readout.counts(f, PITCH)
    c2 = readout.counts(f, PITCH, gain=1.5)
    np.testing.assert_allclose(c2, 1.5 * c1)
    np.testing.assert_allclose(readout.forces(c2, PITCH, gain=1.5), f)


def test_pitch_in_mm_is_refused(readout):
    # 2 "metres" of pitch: someone passed millimetres
    with pytest.raises(ValueError, match="metres"):
        readout.counts([1.0], (2.0, 2.0))


def test_larger_taxels_read_more_at_the_same_pressure(readout):
    """Same pressure, bigger interdigitated taxel -> more counts (s > 0)."""
    p_kpa = 5.0
    small, big = 1.5e-3, 4e-3
    c_small = readout.counts([p_kpa * small ** 2 * 1e3], small)[0]
    c_big = readout.counts([p_kpa * big ** 2 * 1e3], big)[0]
    assert c_big > c_small


def test_noise_is_seeded_and_never_negative(readout):
    f = np.zeros(1000)
    a = readout.counts(f, PITCH, rng=np.random.default_rng(0))
    b = readout.counts(f, PITCH, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 0.0 and a.max() > 0.0


def test_a_real_board_press_is_flagged_when_outside_the_calibration(readout):
    """A ball pressed on a real board, through a measured readout.

    A ball on bumps concentrates the load far beyond a flat press over a whole
    square. That is not a failure of either model, but it is what a user most
    needs to know, so it is flagged rather than extrapolated without notice.
    """
    s = TactileSensor.from_manifest(os.path.join(FIX, "flat_square.json"))
    f = s.probe("sphere", force_n=8.0, radius_mm=40)
    counts = readout.counts(f, s)
    assert counts.shape == f.shape
    assert np.isfinite(counts).all()
    assert readout.extrapolating(f, s).any()
    # and a gentle press stays inside it
    light = s.probe("sphere", force_n=0.05, radius_mm=40)
    assert not readout.extrapolating(light, s).any()
