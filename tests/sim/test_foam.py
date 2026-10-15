"""The ported foam model must give the same per-taxel forces as the original.

The reference numbers in fixtures/foam_reference.json were computed by the lab
script `opentac_indent_compare.sim_taxel_forces` (see make_reference.py), on
the inputs stored beside them.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from opentac import manifest
from opentac.sim import Sensor, foam

FIX = Path(__file__).parent / "fixtures"
REF = json.loads((FIX / "foam_reference.json").read_text())


@pytest.mark.parametrize("case", REF["cases"], ids=lambda c: f"{c['skin']}-{c['geom']['kind']}-{c['total_n']}N")
def test_port_matches_the_original(case):
    m = manifest.load(FIX / case["skin"])
    got = foam.taxel_forces(m.xy, Sensor(m).pitch_mm, case["geom"], case["centre_mm"], case["total_n"],
                            case["sigma_mm"], case["angle_deg"], case["k_n_per_mm3"])
    np.testing.assert_allclose(got, case["forces_n"], rtol=1e-12, atol=1e-15)


def test_reference_cases_are_not_trivial():
    """Guard the fixture: every case must actually press something."""
    for case in REF["cases"]:
        f = np.asarray(case["forces_n"])
        assert f.max() > 0.01 and (f > 1e-6).sum() >= 4, case["skin"]


def test_sensor_press_is_the_same_model():
    """`Sensor.press` is a thin wrapper: same inputs, same numbers."""
    case = REF["cases"][0]
    m = manifest.load(FIX / case["skin"])
    frame = Sensor(m).press(case["geom"], center_mm=case["centre_mm"], force_n=case["total_n"])
    assert frame.units == "N"
    assert len(frame.taxels) == m.n
    np.testing.assert_allclose(frame.taxels, case["forces_n"], rtol=1e-12, atol=1e-15)


def test_paper_parameters_are_the_defaults():
    s = Sensor(FIX / "flat_square.json")
    assert s.sigma_mm == 3.0
    assert s.k_n_per_mm3 == pytest.approx(0.00866, rel=1e-3)


def test_contact_grows_with_force():
    s = Sensor(FIX / "flat_square.json")
    lo = s.press("sphere", force_n=0.5, radius_mm=10.0).taxels
    hi = s.press("sphere", force_n=8.0, radius_mm=10.0).taxels
    assert (hi > 0.05 * hi.max()).sum() > (lo > 0.05 * lo.max()).sum()
    assert hi.sum() > lo.sum()


def test_shapes_by_name_match_geometry_dicts():
    s = Sensor(FIX / "flat_disc.json")
    a = s.press("bar", force_n=3.0, angle_deg=30.0, radius_mm=5.0, length_mm=40.0).taxels
    b = s.press({"kind": "bar", "R": 5.0, "L": 40.0}, force_n=3.0, angle_deg=30.0).taxels
    c = s.press("edge_R5_L40", force_n=3.0, angle_deg=30.0).taxels
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    d = s.press("disc", force_n=3.0, radius_mm=5.0).taxels
    np.testing.assert_array_equal(d, s.press("punch_D10", force_n=3.0).taxels)
    with pytest.raises(ValueError, match="unknown foam indenter"):
        s.press("banana")


def test_readout_turns_newtons_into_counts():
    m = manifest.load(FIX / "flat_square.json")
    n = Sensor(m).press("sphere", force_n=5.0)
    c = Sensor(m, readout="v3").press("sphere", force_n=5.0)
    assert c.units == "counts"
    from opentac.sim import Readout

    r = Readout.from_profile("v3")
    np.testing.assert_allclose(c.taxels, r.counts(n.taxels, (2.5e-3, 2.5e-3)))
    # and any object with counts(newtons) works, e.g. a per-skin calibration curve
    class Linear:
        def counts(self, newtons):
            return 1000.0 * np.asarray(newtons)
    np.testing.assert_allclose(Sensor(m, readout=Linear()).press("sphere", force_n=5.0).taxels,
                               1000.0 * n.taxels)


def test_each_taxel_reads_a_window_of_2_round_pitch_plus_1_cells():
    """A property of the original model, kept by the port and stated in docs/sim.md.

    Each taxel sums the pressure over a square of w = (2*round(pitch) + 1) * 0.5 mm.
    w equals the pitch at 2.5 mm, but is 0.5 mm wider at 2.0 and 4.0 mm, so
    neighbouring windows overlap and a frame can sum to more than the force.
    Checked with a punch far larger than the skin, which gives a uniform pressure
    p = F / (sampled area) everywhere the taxels are.
    """
    big = {"kind": "punch", "R": 500.0}
    for skin, w in (("flat_square.json", 2.5), ("grid_4mm.json", 4.5), ("grid_2mm.json", 2.5)):
        m = manifest.load(FIX / skin)
        pitch = Sensor(m).pitch_mm
        assert w == (2 * round(pitch) + 1) * foam.GRID_MM
        f = foam.taxel_forces(m.xy, pitch, big, m.xy.mean(0), 10.0)
        lo = m.xy.min(0) - 3 * pitch - 2 * foam.SIGMA_MM
        hi = m.xy.max(0) + 3 * pitch + 2 * foam.SIGMA_MM
        n_cells = len(np.arange(lo[0], hi[0] + foam.GRID_MM, foam.GRID_MM)) * \
            len(np.arange(lo[1], hi[1] + foam.GRID_MM, foam.GRID_MM))
        p = 10.0 / (n_cells * foam.GRID_MM ** 2)
        np.testing.assert_allclose(f, p * w * w, rtol=1e-6)
