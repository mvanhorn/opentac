"""The transfer gives the reference numbers.

The reference numbers in fixtures/transfer_reference.json were computed with
the analysis code of the OpenTac paper, on the inputs stored beside them.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from opentac import manifest
from opentac.frame import Frame
from opentac.transfer import Layout, Transfer, pixel_pitch_mm, placement_cop, transfer

FIX = Path(__file__).parent / "fixtures"
REF = json.loads((FIX / "transfer_reference.json").read_text())


def _layout(name):
    return Layout(manifest.load(FIX / name))


@pytest.mark.parametrize("skin", sorted(REF["pitch_um"]))
def test_pitch_matches_the_reference(skin):
    px, py = pixel_pitch_mm(manifest.load(FIX / skin))
    assert [round(px * 1000), round(py * 1000)] == REF["pitch_um"][skin]


@pytest.mark.parametrize("case", REF["cases"], ids=lambda c: f"{c['src']}->{c['dst']}-{c['kernel']}")
def test_matches_the_reference(case):
    la, lb = _layout(case["src"]), _layout(case["dst"])
    fa = np.asarray(case["fa_n"])
    place = case["placement"]
    if place.get("centered"):
        tr = Transfer.centered(la, lb, fa, angle_deg=place["angle_deg"], target=place.get("target"),
                               kernel=case["kernel"])
    else:
        tr = Transfer(la, lb, angle_deg=place["angle_deg"], shift_mm=place["shift_mm"], kernel=case["kernel"])
    np.testing.assert_allclose(tr.force(fa), case["fb_n"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(tr.kept, case["kept"], rtol=1e-12, atol=1e-15)
    assert tr.lost(fa) == pytest.approx(case["lost_n"], rel=1e-12, abs=1e-15)
    np.testing.assert_allclose(tr.shift_mm, case["shift_mm"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tr.about, case["about_mm"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(placement_cop(la.xy, fa)[0], case["placement_cop_mm"], rtol=1e-12)


def test_force_is_conserved_where_b_has_taxels():
    """What B reads plus what fell outside B is exactly what A read."""
    for case in REF["cases"]:
        assert sum(case["fb_n"]) + case["lost_n"] == pytest.approx(sum(case["fa_n"]), rel=1e-9)


def test_reference_cases_are_not_trivial():
    """Guards the fixture: the centred cases keep most of the force."""
    kept = [sum(c["fb_n"]) / sum(c["fa_n"]) for c in REF["cases"] if c["placement"].get("centered")]
    assert len(kept) >= 3 and min(kept) > 0.8, kept


def test_transfer_frame_matches_transfer_centered():
    case = next(c for c in REF["cases"] if c["placement"].get("centered") and "target" not in c["placement"])
    a, b = manifest.load(FIX / case["src"]), manifest.load(FIX / case["dst"])
    out = transfer(Frame(taxels=case["fa_n"], t=1.5, units="N"), a, b,
                   angle_deg=case["placement"]["angle_deg"], kernel=case["kernel"])
    assert out.units == "N" and out.t == 1.5 and len(out.taxels) == b.n
    np.testing.assert_allclose(out.taxels, case["fb_n"], rtol=1e-12, atol=1e-15)
    assert out.imu == {}


def test_transfer_without_centering_uses_a_mm_as_b_mm():
    case = REF["cases"][1]
    out = transfer(Frame(taxels=case["fa_n"], t=0.0, units="N"), FIX / case["src"], FIX / case["dst"],
                   center=False, angle_deg=case["placement"]["angle_deg"],
                   shift_mm=case["placement"]["shift_mm"], kernel=case["kernel"])
    np.testing.assert_allclose(out.taxels, case["fb_n"], rtol=1e-12, atol=1e-15)


def test_unread_taxels_count_as_zero():
    case = REF["cases"][3]
    fa = np.asarray(case["fa_n"], dtype=float)
    holed = fa.copy()
    holed[::5] = np.nan
    tr = Transfer(FIX / case["src"], FIX / case["dst"])
    np.testing.assert_allclose(tr.force(holed), tr.force(np.where(np.isnan(holed), 0.0, fa)))


def test_counts_are_refused():
    with pytest.raises(ValueError, match="newtons"):
        transfer(Frame(taxels=np.ones(48), t=0.0), FIX / "grid_4mm.json", FIX / "flat_square.json")


def test_an_empty_frame_transfers_to_zeros():
    out = transfer(Frame(taxels=np.zeros(48), t=0.0, units="N"), FIX / "grid_4mm.json", FIX / "flat_square.json")
    assert out.taxels.shape == (48,) and not out.taxels.any()


def test_non_square_pitch_is_refused():
    m = {"pixels": [[x * 2.0, y * 3.0, y, x] for y in range(3) for x in range(3)]}
    with pytest.raises(ValueError, match="square"):
        Layout(m)
