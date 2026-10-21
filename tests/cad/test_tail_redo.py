"""Changing the tail of a routed board without routing it again.

A drawn tail extension that turns right where it starts (the page's diagonal
and sideways buttons) carries every wire on in its own lane, turning with the
path; and Redo padding with a Tail length typed in makes the board at that
length, or says to press Generate.
"""
import base64
import copy
import io
import json
import math
import shutil
import subprocess
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Polygon

from opentac.cad.main import app
from opentac.cad.pipeline.tail_length import TRIM_KEEP_MM

client = TestClient(app)
KICAD = shutil.which("kicad-cli")
SMALL = dict(pitch_x_mm=2.0, pitch_y_mm=2.0, pixel_w_mm=1.8, pixel_h_mm=1.8, router="hug")
# 17 columns x 4 rows: the 60-pin board, pads well off the connector's centre.
# 12 x 3: the 30-pin board.
DESIGNS = {
    "60pin": dict(outline=[[0, 0], [34, 0], [34, 8], [0, 8]], **SMALL),
    "30pin": dict(outline=[[0, 0], [24, 0], [24, 6], [0, 6]], **SMALL),
}
_CACHE = {}


def _gen(name, board_mode="expand", **over):
    key = (name, board_mode, json.dumps(over, sort_keys=True))
    if key not in _CACHE:
        r = client.post("/generate", json={**DESIGNS[name], "board_mode": board_mode, **over})
        assert r.status_code == 200, r.text[:300]
        _CACHE[key] = r.json()
    return _CACHE[key]


def _kicad_errors(zip_b64, tmp_path):
    zipfile.ZipFile(io.BytesIO(base64.b64decode(zip_b64))).extractall(tmp_path)
    out = tmp_path / "drc.json"
    subprocess.run([KICAD, "pcb", "drc", "--format", "json", "-o", str(out),
                    str(tmp_path / "board.kicad_pcb")], capture_output=True, check=False)
    report = json.loads(out.read_text())
    errors = [v["type"] + ": " + v["description"] for v in report["violations"] if v["severity"] == "error"]
    return errors, report["unconnected_items"]


def _with_extension(edit_data, angle_deg, length=20.0):
    """edit_data with a one-segment tail extension, as the page's direction
    buttons draw it: from the pad split point, `length` mm at `angle_deg`
    (0 right, -90 down)."""
    ed = copy.deepcopy(edit_data)
    t, p = ed["layout"]["tail"], ed["params"]
    start = np.asarray(t["pad_split_center"], dtype=float)
    a = math.radians(angle_deg)
    end = np.round((start + length * np.array([math.cos(a), math.sin(a)])) / 0.1) * 0.1
    ed["layout"]["tail_extension"] = {"enabled": True, "path": [start.tolist(), end.tolist()],
                                      "width_mm": max(0.0, t["width_mm"] - 0.7),
                                      "clearance_mm": p["trace_w_mm"] / 2 + p["board_edge_clear_mm"]}
    return ed


@pytest.mark.parametrize("name,board_mode,layers", [("60pin", "expand", 2), ("30pin", "expand", 2),
                                                     ("60pin", "fixed_keepout", 4)])
@pytest.mark.parametrize("angle", [-45, -135, -90])
def test_a_diagonal_extension_is_as_clean_as_a_straight_one(name, board_mode, layers, angle, tmp_path):
    """Each lane starts at its own pad. A lane offset square to the path's first
    segment would, on a diagonal, start beside its pad and need a jump across
    its neighbours to reach it."""
    g = _gen(name, board_mode, copper_layers=layers)
    ed = _with_extension(g["edit_data"], angle)
    r = client.post("/export-edited", json={"edit_data": ed, "routes": g["edit_data"]["routes"]})
    assert r.status_code == 200, r.text[:300]
    out = r.json()
    assert out["drc"]["violations"] == 0
    assert out["stats"]["active_pixels"] == g["stats"]["active_pixels"]
    # the redo-padding round trip the page makes, and the export after it
    # (the wires already carried down the path), keep it clean
    r = client.post("/redo-padding", json={"edit_data": ed, "routes": g["edit_data"]["routes"]})
    assert r.status_code == 200 and r.json()["drc"]["violations"] == 0
    redo = r.json()["edit_data"]
    r = client.post("/export-edited", json={"edit_data": redo, "routes": redo["routes"]})
    assert r.status_code == 200 and r.json()["drc"]["violations"] == 0
    if KICAD:
        errors, unconnected = _kicad_errors(out["zip_b64"], tmp_path)
        assert errors == [] and unconnected == []


def test_a_short_wire_run_moves_the_turn_down_the_tail():
    """On 4 layers a fixed outline's row feeds turn into their pads 1 mm before
    them: the lanes on the inside of a down-left turn cannot turn on those runs,
    so the drawn path starts that much further down the tail, straight."""
    g = _gen("60pin", "fixed_keepout", copper_layers=4)
    ed = _with_extension(g["edit_data"], -135)
    r = client.post("/redo-padding", json={"edit_data": ed, "routes": g["edit_data"]["routes"]})
    assert r.status_code == 200 and r.json()["drc"]["violations"] == 0
    res = r.json()["edit_data"]["layout"]["tail"]["tail_extension_resolved"]
    assert res["lead_in_mm"] > 0
    path, drawn = np.asarray(res["path"]), np.asarray(res["drawn_path"])
    n = np.asarray(ed["layout"]["tail"]["n_out"], dtype=float)
    assert np.allclose(path[0], drawn[0]) and np.allclose(path[1], drawn[0] + res["lead_in_mm"] * n)
    assert np.allclose(path[2:], drawn[1:] + res["lead_in_mm"] * n)       # the drawn path, moved down


@pytest.mark.parametrize("angle", [0, 180])
def test_a_sideways_extension_on_a_short_tail_is_clean(angle):
    """The inside of a right-angle turn after a short straight run: the
    corridor's offset centre line does not fold back on itself there, so the
    lanes stay on the board."""
    g = _gen("60pin", "fixed_keepout", copper_layers=4, cable_length_mm=6.0)
    ed = _with_extension(g["edit_data"], angle)
    r = client.post("/redo-padding", json={"edit_data": ed, "routes": g["edit_data"]["routes"]})
    assert r.status_code == 200 and r.json()["drc"]["violations"] == 0


def test_every_lane_carries_on_from_its_wire():
    """No wire jumps sideways where the extension starts: each runs down the
    tail into its lane's turn and then along the path, and none crosses
    another on its layer."""
    from shapely.geometry import LineString
    g = _gen("60pin")
    ed = _with_extension(g["edit_data"], -45)
    r = client.post("/redo-padding", json={"edit_data": ed, "routes": g["edit_data"]["routes"]})
    assert r.status_code == 200
    routes = r.json()["edit_data"]["routes"]
    n = np.asarray(g["edit_data"]["layout"]["tail"]["n_out"], dtype=float)
    path = np.asarray(ed["layout"]["tail_extension"]["path"], dtype=float)
    diag = (path[1] - path[0]) / np.linalg.norm(path[1] - path[0])
    by_layer = {}
    for rt in routes:
        P = np.asarray(rt["polyline_xy"], dtype=float)
        by_layer.setdefault(rt["layer"], []).append(LineString(P))
        if np.linalg.norm(P[0] - np.asarray(rt["pad_xy"])) > 1.0:
            continue                                     # a link track inside the sensor
        # from the moved pad: along the path, then (once) down the tail
        d = np.diff(P[:3], axis=0)
        d = d / np.linalg.norm(d, axis=1)[:, None]
        assert np.allclose(d[0], -diag, atol=1e-6)
        assert np.allclose(d[1], -n, atol=1e-6)
    for lines in by_layer.values():
        for i, a in enumerate(lines):
            assert not any(a.crosses(b) for b in lines[i + 1:])


# ---------------------------------------------------------------------------
# Redo padding with a Tail length typed in
# ---------------------------------------------------------------------------
def _redo(ed, routes, length):
    return client.post("/redo-padding", json={"edit_data": ed, "routes": routes,
                                              "params": {"cable_length_mm": length}})


def _same_routes(a, b):
    return len(a) == len(b) and all(
        (ra["net_name"], ra["layer"]) == (rb["net_name"], rb["layer"])
        and np.allclose(np.asarray(ra["polyline_xy"]), np.asarray(rb["polyline_xy"])) for ra, rb in zip(a, b))


@pytest.mark.parametrize("name", sorted(DESIGNS))
@pytest.mark.parametrize("board_mode", ["expand", "fixed_keepout"])
def test_a_longer_typed_length_moves_the_connector_out(name, board_mode):
    g = _gen(name, board_mode)
    ed, routes, length = g["edit_data"], g["edit_data"]["routes"], g["stats"]["tail_length_mm"]
    r = _redo(ed, routes, length + 3.0)
    assert r.status_code == 200, r.text[:300]
    out = r.json()
    s, t = out["stats"], out["edit_data"]["layout"]["tail"]
    assert s["tail_length_mm"] == length + 3.0 and s["tail_length_auto"] is False
    assert out["edit_data"]["params"]["cable_length_mm"] == length + 3.0
    assert out["edit_data"]["params"]["cable_length_auto"] is False
    assert t["mode"] == "fit" and t["width_mm"] == ed["layout"]["tail"]["width_mm"]
    assert s["active_pixels"] == g["stats"]["active_pixels"]
    assert out["drc"]["violations"] == 0
    # every pad moved 3 mm out along the tail; nothing else moved
    n = np.asarray(t["n_out"], dtype=float)
    for ra, rb in zip(out["edit_data"]["routes"], routes):
        Pa, Pb = np.asarray(ra["polyline_xy"]), np.asarray(rb["polyline_xy"])
        if np.linalg.norm(Pb[0] - np.asarray(rb["pad_xy"])) > 1.0:
            assert np.allclose(Pa, Pb)
            continue
        assert np.allclose(Pa[0], Pb[0] + 3.0 * n) and np.allclose(Pa[1:], Pb[1:])
    # the board grew by the tail alone
    o = Polygon(out["edit_data"]["final_outline"]).buffer(0)
    assert o.area > Polygon(ed["final_outline"]).buffer(0).area
    # and typed back, the board is the one generated
    r = _redo(out["edit_data"], out["edit_data"]["routes"], length)
    assert r.status_code == 200
    back = r.json()["edit_data"]
    assert _same_routes(back["routes"], routes)
    assert Polygon(back["final_outline"]).area == pytest.approx(Polygon(ed["final_outline"]).area, abs=1e-6)


@pytest.mark.parametrize("name", sorted(DESIGNS))
@pytest.mark.parametrize("board_mode", ["expand", "fixed_keepout"])
def test_a_shorter_typed_length_is_the_board_routed_for_it(name, board_mode, tmp_path):
    """Down to the 12 mm the board was routed with, the shortened board is the
    one /generate makes with that length typed."""
    typed = _gen(name, board_mode, cable_length_mm=12.0)
    g = _gen(name, board_mode)                 # routed at 12 mm, whatever its length
    ed, routes = g["edit_data"], g["edit_data"]["routes"]
    longer = _redo(ed, routes, 16.0).json()["edit_data"]
    r = _redo(longer, longer["routes"], 12.0)
    assert r.status_code == 200, r.text[:300]
    out = r.json()
    assert out["stats"]["tail_length_mm"] == 12.0 and out["drc"]["violations"] == 0
    assert _same_routes(out["edit_data"]["routes"], typed["edit_data"]["routes"])
    assert Polygon(out["edit_data"]["final_outline"]).area == pytest.approx(
        Polygon(typed["edit_data"]["final_outline"]).area, abs=1e-3)
    if KICAD:
        z = client.post("/export-edited", json={"edit_data": out["edit_data"], "routes": out["edit_data"]["routes"]})
        errors, unconnected = _kicad_errors(z.json()["zip_b64"], tmp_path)
        assert errors == [] and unconnected == []


def test_shortening_past_the_wires_straight_runs_says_to_press_generate():
    """On 4 layers a fixed outline's row feeds run down the tail and turn into
    their pads TRIM_KEEP_MM before them: no room to cut at all."""
    g = _gen("60pin", "fixed_keepout", copper_layers=4)
    ed, routes = g["edit_data"], g["edit_data"]["routes"]
    runs = [float(np.linalg.norm(np.asarray(r["polyline_xy"][1]) - np.asarray(r["polyline_xy"][0])))
            for r in routes if np.linalg.norm(np.asarray(r["polyline_xy"][0]) - np.asarray(r["pad_xy"])) <= 1.0]
    assert min(runs) == pytest.approx(TRIM_KEEP_MM)
    r = _redo(ed, routes, g["stats"]["tail_length_mm"] - 1.0)
    assert r.status_code == 400
    assert "routed again" in r.json()["detail"] and "Press Generate" in r.json()["detail"]
    # longer is always possible
    r = _redo(ed, routes, g["stats"]["tail_length_mm"] + 1.0)
    assert r.status_code == 200 and r.json()["drc"]["violations"] == 0


def test_a_typed_length_under_a_tail_extension_says_what_to_do():
    g = _gen("60pin")
    ed = _with_extension(g["edit_data"], -90)
    r = _redo(ed, g["edit_data"]["routes"], g["stats"]["tail_length_mm"] + 2.0)
    assert r.status_code == 400 and "press Generate" in r.json()["detail"]
    # the same length, or the field left blank, keeps the board's own
    for length in (g["stats"]["tail_length_mm"], None):
        r = _redo(ed, g["edit_data"]["routes"], length)
        assert r.status_code == 200 and r.json()["drc"]["violations"] == 0
