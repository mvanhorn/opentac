"""The cable tail is as long as the sensor needs, by default (pipeline/tail_length.py).

With no cable_length_mm the board is routed with the 12 mm minimum and the tail is
then lengthened, without a second routing pass, until the narrow tail reaches the
width of its wires and the printed case leaves the connector body clear. A typed
length is used as given, and so is the number every older design carries.
"""
import base64
import io
import json
import zipfile

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from opentac.cad.main import _recipe_params, app
from opentac.cad.pipeline.case import case_cable_check
from opentac.cad.pipeline.geometry import fit_tail_strip, tail_widening_length
from opentac.cad.pipeline.narrow_tail import MIN_STRAIGHT_MM
from opentac.cad.pipeline.tail_length import (
    AUTO_TAIL_MIN_MM, REASON_CASE, REASON_FIT, REASON_MIN, REASON_TYPED, round_up_length,
)
from opentac.cad.pipeline.types import Params

client = TestClient(app)
SMALL = dict(pitch_x_mm=2.0, pitch_y_mm=2.0, pixel_w_mm=1.8, pixel_h_mm=1.8, router="hug")
# 17 columns x 4 rows: the 60-pin board, pads well off the connector's centre.
# 12 x 3: the 30-pin board.
DESIGNS = {
    "60pin": dict(outline=[[0, 0], [34, 0], [34, 8], [0, 8]], **SMALL),
    "30pin": dict(outline=[[0, 0], [24, 0], [24, 6], [0, 6]], **SMALL),
}
_CACHE = {}


def _gen(name, **over):
    key = (name, json.dumps(over, sort_keys=True))
    if key not in _CACHE:
        r = client.post("/generate", json={**DESIGNS[name], **over})
        assert r.status_code == 200, r.text[:300]
        _CACHE[key] = r.json()
    return _CACHE[key]


def _cable_y(ed):
    return float(np.asarray(ed["outline"], dtype=float)[:2, 1].min())


def test_lengths_round_up_to_half_millimetres():
    assert round_up_length(12.0) == 12.0
    assert round_up_length(12.01) == 12.5
    assert round_up_length(17.5) == 17.5
    assert round_up_length(17.51) == 18.0


@pytest.mark.parametrize("name", sorted(DESIGNS))
def test_automatic_length_narrows_fully_and_clears_the_case(name):
    g = _gen(name)
    s, ed = g["stats"], g["edit_data"]
    length = s["tail_length_mm"]
    assert s["tail_length_auto"] is True
    assert s["tail_length_reason"] in (REASON_FIT, REASON_CASE, REASON_MIN)
    assert length >= AUTO_TAIL_MIN_MM and (2 * length) == int(2 * length)
    # the stored design carries the chosen number, so later steps use it
    assert ed["params"]["cable_length_mm"] == length and ed["params"]["cable_length_auto"] is True
    assert ed["layout"]["tail"]["length_mm"] == length
    # (a) the narrow tail is at its full fit width
    p = Params(**{k: v for k, v in ed["params"].items()})
    pads = {int(k): v for k, v in ed["layout"]["pad_to_net"].items()}
    edge = p.trace_w_mm / 2 + p.follow_main_padding_mm
    width, offset = fit_tail_strip(pads, ed["layout"]["connector_pos"], p, edge)
    t = ed["layout"]["tail"]
    assert t["mode"] == "fit"
    assert t["width_mm"] == pytest.approx(width) and t["fit_offset_mm"] == pytest.approx(offset)
    wide = t["connector_min_width_mm"]
    if width < wide - 1e-6:
        assert length >= tail_widening_length(width, wide, float("inf"), offset, t["step_fillet_mm"]) \
            + MIN_STRAIGHT_MM - 1e-6
    # (b) the printed case leaves the connector body clear
    check = case_cable_check(ed["final_outline"], _cable_y(ed))
    assert check["short_mm"] == 0 and check["warnings"] == []
    assert g["drc"]["violations"] == 0


@pytest.mark.parametrize("name", sorted(DESIGNS))
def test_routes_are_the_12mm_routes_with_a_longer_run_into_each_pad(name):
    auto, typed = _gen(name), _gen(name, cable_length_mm=12.0)
    assert auto["stats"]["active_pixels"] == typed["stats"]["active_pixels"]
    assert auto["edit_data"]["layout"]["placements_xy"] == typed["edit_data"]["layout"]["placements_xy"]
    d = auto["stats"]["tail_length_mm"] - 12.0
    n = np.asarray(auto["edit_data"]["layout"]["tail"]["n_out"], dtype=float)
    assert len(auto["edit_data"]["routes"]) == len(typed["edit_data"]["routes"])
    for ra, rt in zip(auto["edit_data"]["routes"], typed["edit_data"]["routes"]):
        assert (ra["net_name"], ra["layer"]) == (rt["net_name"], rt["layer"])
        Pa, Pt = np.asarray(ra["polyline_xy"]), np.asarray(rt["polyline_xy"])
        if np.linalg.norm(Pt[0] - np.asarray(rt["pad_xy"])) > 1.0:      # a link track: untouched
            assert np.allclose(Pa, Pt)
            continue
        assert np.allclose(Pa[0], Pt[0] + d * n)                         # the pad moved d along the tail
        rest = Pa[1:]
        assert (len(rest) == len(Pt) - 1 and np.allclose(rest, Pt[1:])) or \
               (len(rest) == len(Pt) and np.allclose(rest, Pt))          # and nothing else did
    assert auto["drc"]["violations"] == 0


def test_typed_length_is_used_as_given():
    g = _gen("60pin", cable_length_mm=15.0)
    assert g["stats"]["tail_length_mm"] == 15.0
    assert g["stats"]["tail_length_auto"] is False and g["stats"]["tail_length_reason"] == REASON_TYPED
    assert g["edit_data"]["params"]["cable_length_mm"] == 15.0
    assert g["edit_data"]["params"]["cable_length_auto"] is False


def test_redo_padding_export_and_case_keep_the_chosen_length():
    g = _gen("60pin")
    ed, routes, length = g["edit_data"], g["edit_data"]["routes"], g["stats"]["tail_length_mm"]
    # the page sends its form with Tail length blank: that keeps the board's own length
    r = client.post("/redo-padding", json={"edit_data": ed, "routes": routes,
                                           "params": {"cable_length_mm": None}})
    assert r.status_code == 200, r.text[:300]
    redo = r.json()
    assert redo["stats"]["tail_length_mm"] == length and redo["stats"]["tail_length_auto"] is True
    assert redo["edit_data"]["final_outline"] == ed["final_outline"]
    assert redo["drc"]["violations"] == 0
    r = client.post("/export-edited", json={"edit_data": redo["edit_data"], "routes": routes})
    assert r.status_code == 200 and r.json()["drc"]["violations"] == 0
    assert r.json()["stats"]["tail_length_mm"] == length
    r = client.post("/zip-with-case", json={"zip_b64": g["zip_b64"], "edit_data": ed, "name": "t"})
    assert r.status_code == 200
    out = r.json()
    assert out["case_error"] is None
    # (a short perimeter's rib-pattern note is not about the cable)
    assert [w for w in out["warnings"] if "cable" in w or "connector" in w] == []
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(out["zip_b64"])))
    stls = [n for n in z.namelist() if n.endswith(".stl")]
    assert len(stls) == 4
    for n in stls:
        m = trimesh.load(io.BytesIO(z.read(n)), file_type="stl")
        assert m.is_watertight and m.body_count == 1, n


def test_a_recipe_replay_chooses_the_length_again():
    g = _gen("60pin")
    body = _recipe_params(g["edit_data"]["params"], "hug", g["edit_data"]["layout"]["connector_pos"])
    assert body["cable_length_mm"] is None
    # a design from before the length was automatic keeps its number
    old = {k: v for k, v in g["edit_data"]["params"].items() if k != "cable_length_auto"}
    old["cable_length_mm"] = 12.0
    assert _recipe_params(old, "hug", 60)["cable_length_mm"] == 12.0
