"""The paper's sim and transfer numbers, recomputed from the stored result rows.

The results file (results_pose.json, 29 MB) is not in this repository. Point
OPENTAC_PAPER_RESULTS at it to run this check; otherwise it is skipped:

    OPENTAC_PAPER_RESULTS=/path/to/results_pose.json uv run pytest tests/sim/test_paper_numbers.py

Each number is the median "shape F1" over every scored frame of that
predictor: how well the predicted contact patch overlaps the measured one, with
intensity taken out. The foam model function that produced the "sim" rows is
the one tests/sim/test_foam.py checks the port against.
"""
import json
import os

import numpy as np
import pytest

PATH = os.environ.get("OPENTAC_PAPER_RESULTS")
pytestmark = pytest.mark.skipif(not PATH or not os.path.exists(PATH),
                                reason="set OPENTAC_PAPER_RESULTS to results_pose.json")

# predictor: (median shape F1 as reported, number of scored rows)
REPORTED = {
    "ceiling": (0.80, 1857),
    "sim": (0.73, 954),
    "transfer": (0.75, 5290),
    "transfer (co-visible)": (0.79, 5286),
    "resize (manifest)": (0.71, 5291),
    "resize (no manifest)": (0.63, 5077),
}


@pytest.fixture(scope="module")
def results():
    with open(PATH, encoding="utf-8") as f:
        return json.load(f)


def test_foam_parameters_are_the_ones_shipped(results):
    from opentac.sim import foam

    assert results["fit_skin"] == "robotiq_long"
    assert results["sigma_mm"] == foam.SIGMA_MM
    assert results["stiffness_n_per_mm3"] == foam.K_N_PER_MM3


@pytest.mark.parametrize("predictor", sorted(REPORTED))
def test_median_shape_f1(results, predictor):
    v = [r["f1_shape"] for r in results["rows"]
         if r["predictor"] == predictor and r["placement"] == "pose" and r.get("f1_shape") is not None]
    want, n = REPORTED[predictor]
    assert len(v) == n
    assert round(float(np.median(v)), 2) == want


def test_sim_is_zero_shot(results):
    """The foam model was fitted on robotiq_long, so it is never scored there."""
    skins = {r["skin"] for r in results["rows"] if r["predictor"] == "sim"}
    assert skins == {"triangle", "agileX_2mm", "umi"}


def _median(results, predictor, field):
    return float(np.median([r[field] for r in results["rows"]
                            if r["predictor"] == predictor and r["placement"] == "pose"
                            and r.get(field) is not None]))


def test_sim_centroid_and_area(results):
    """docs/sim.md: the foam model places the contact centre closer than a
    repeat press (1.56 vs 2.29 mm) but predicts 1.44x the area."""
    assert round(_median(results, "sim", "centroid_mm"), 2) == 1.56
    assert round(_median(results, "ceiling", "centroid_mm"), 2) == 2.29
    assert round(_median(results, "sim", "area_ratio"), 2) == 1.44
