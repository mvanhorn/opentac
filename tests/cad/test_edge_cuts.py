"""The exported Edge.Cuts close into one outline with every line on it once.

The board's outline is exported cut open at the connector, and the connector
footprint's own Edge.Cuts draw the tab from the two open ends. If a board whose
wires run to the outermost connector pins had its tail padded past the tab's
sides down to the pad drills (post-route containment around the outermost
wires), the footprint's tab-side lines would lie inside the board, and a fab
would mill them as slits. KiCad's DRC passes such a board (it folds a slit into
the outline as a zero-width notch), so these tests read the exported KiCad file
itself and run export/edge_cuts_check.py on it.
"""
import base64
import io
import re
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import LineString, Polygon

from opentac.cad import main
from opentac.cad.export.edge_cuts_check import check_edge_cuts, kicad_pcb_edge_cuts
from opentac.cad.pipeline.geometry import drop_hairline_spikes, remove_collinear_closed

client = TestClient(main.app)
SQUARE_60 = [[-30, 0], [30, 0], [30, 60], [-30, 60]]
SQUARE_126 = [[-63, 0], [63, 0], [63, 126], [-63, 126]]
# Every pin of the connector in use: the tail is as wide as the connector's tab.
FULL_WIDTH = {
    "30pin_15x15": dict(outline=SQUARE_60, readout_board="30pin", target_cols=15, target_rows=15),
    "60pin_30x30": dict(outline=SQUARE_126, readout_board="60pin", target_cols=30, target_rows=30),
    "60pin_30x30_4layer": dict(outline=SQUARE_126, readout_board="60pin", target_cols=30, target_rows=30,
                               copper_layers=4),
}


def _generate(**req):
    r = client.post("/generate", json=dict(router="hug", **req))
    assert r.status_code == 200, r.text[:300]
    return r.json()


def _board_file(resp) -> str:
    return zipfile.ZipFile(io.BytesIO(base64.b64decode(resp["zip_b64"]))).read("board.kicad_pcb").decode()


def _tracks(pcb_text):
    num = r"(-?[\d.]+)"
    return [LineString([(float(m[0]), float(m[1])), (float(m[2]), float(m[3]))])
            for m in re.findall(r"\(segment \(start " + num + " " + num + r"\) \(end " + num + " " + num
                                + r"\) \(width [\d.]+\) \(layer \"(?:F|B|In1|In2)\.Cu\"\)", pcb_text)]


@pytest.mark.parametrize("name", sorted(FULL_WIDTH))
def test_full_width_tail_ends_at_the_connector(name):
    resp = _generate(**FULL_WIDTH[name])
    s = resp["stats"]
    pins = s["connector_pos"]
    assert len(resp["edit_data"]["layout"]["pad_to_net"]) == pins, "every connector pin in use"
    assert s["tail_width_mm"] == pytest.approx(s["connector_tab_tip_width_mm"])
    pcb = _board_file(resp)
    lines = kicad_pcb_edge_cuts(pcb)
    assert any(src == "connector" for *_, src in lines)
    assert check_edge_cuts(lines) == []
    assert resp["drc"]["violations"] == 0

    # Clipping the tail to the tab keeps every wire clear of the edge: the wires
    # to the outermost pins run one pin pitch from the tab's sides.
    tracks = _tracks(pcb)
    trace, clear = 0.2, 0.2
    edges = [LineString([a, b]) for a, b, _ in lines]
    assert min(t.distance(e) for t in tracks for e in edges) >= trace / 2 + clear - 1e-6
    tab = [LineString([a, b]) for a, b, src in lines if src == "connector"]
    assert min(t.distance(e) for t in tracks for e in tab) == pytest.approx(0.5, abs=1e-3)   # one pitch


def test_drc_reports_a_tail_padded_past_the_tab(monkeypatch):
    # With the tail-wire hold and the clip to the tab disabled, post-route
    # containment pads the tail's own wires too, and nothing clips the board to
    # the tab past the connector's opening; the DRC reports the result.
    monkeypatch.setattr(main, "tail_wire_hold", lambda layout: None)
    monkeypatch.setattr(main, "_clip_board_to_connector_tab", lambda *a, **k: False)
    resp = _generate(**FULL_WIDTH["30pin_15x15"])
    found = [d for d in resp["drc"]["details"] if d["layer"] == "Edge.Cuts"]
    assert found and resp["drc"]["violations"] >= len(found)
    assert check_edge_cuts(kicad_pcb_edge_cuts(_board_file(resp)))


# --- other ways a board can end up wider than the tab -----------------------
# Past the connector's opening the board is clipped to the tab
# (connector_geometry.clip_beside_tab), wherever the board was widened.

PAD = dict(outline=[[0, 0], [17.6, 0], [17.6, 34.6], [0, 34.6]], pitch_x_mm=4.2, pitch_y_mm=4.2,
           pixel_w_mm=4.0, pixel_h_mm=4.0)


def _export(edit_data, routes):
    r = client.post("/export-edited", json={"edit_data": edit_data, "routes": routes})
    assert r.status_code == 200, r.text[:300]
    return r.json()


def _redo(edit_data, **extra):
    r = client.post("/redo-padding", json={"edit_data": edit_data, "routes": edit_data["routes"], **extra})
    assert r.status_code == 200, r.text[:300]
    return r.json()["edit_data"]


def _assert_clean(resp, taxels=None):
    assert check_edge_cuts(kicad_pcb_edge_cuts(_board_file(resp))) == []
    assert resp["drc"]["violations"] == 0
    if taxels is not None:
        assert resp["stats"]["active_pixels"] == taxels


@pytest.mark.parametrize("board_mode", ["expand", "fixed_keepout"])
def test_typed_tail_wider_than_the_tab(board_mode):
    tab = 15.5                                       # the 30-pin connector this pad gets
    wide = _generate(**PAD, board_mode=board_mode, cable_length_mm=14.0, tail_width_mm=tab + 4)
    assert wide["stats"]["connector_tab_tip_width_mm"] == pytest.approx(tab)
    _assert_clean(wide)
    # the typed width is kept up to the connector's opening, which the export
    # puts at the pad row (cable length + CONNECTOR_BODY_ABOVE_MM)
    ed = wide["edit_data"]
    o = np.asarray(ed["outline"], dtype=float)
    t = ed["layout"]["tail"]
    mid = 0.5 * (o[0] + o[1])
    n, th = np.asarray(t["n_out"], dtype=float), np.asarray(t["t_hat"], dtype=float)
    board = Polygon(ed["final_outline"]).buffer(0)
    across = lambda depth: board.intersection(LineString([mid + depth * n - 60 * th, mid + depth * n + 60 * th])).length
    assert across(14.0 + 2.75 - 0.3) == pytest.approx(tab + 4, abs=0.01)
    assert across(14.0 + 2.75 + 0.3) == pytest.approx(tab, abs=0.01)     # past it: the tab


@pytest.mark.parametrize("board_mode", ["expand", "fixed_keepout"])
def test_bigger_connector_swap(board_mode):
    g = _generate(**PAD, board_mode=board_mode)
    ed = _redo(g["edit_data"], connector_pos_override=40)
    assert ed["layout"]["connector_pos"] == 40
    _assert_clean(_export(ed, ed["routes"]), g["stats"]["active_pixels"])


def test_bigger_connector_then_bent_tail_extension():
    g = _generate(**PAD)
    ed = _redo(g["edit_data"], connector_pos_override=40)
    te = dict(ed["layout"]["tail_extension"])
    s = np.asarray(te["path"][0], dtype=float)
    n = np.asarray(ed["layout"]["tail"]["n_out"], dtype=float)
    turn = np.array([n[0] - n[1], n[0] + n[1]]) / np.sqrt(2.0)          # 45 degrees
    te.update(enabled=True, path=[s.tolist(), (s + 6 * n).tolist(), (s + 6 * n + 10 * turn).tolist()])
    ed["layout"]["tail_extension"] = te
    ed = _redo(ed)
    _assert_clean(_export(ed, ed["routes"]), g["stats"]["active_pixels"])


def test_follow_padding_off():
    g = _generate(**PAD)
    ed = _redo(g["edit_data"], params={"follow_main_padding": False})
    assert ed["params"]["follow_main_padding"] is False
    _assert_clean(_export(ed, ed["routes"]), g["stats"]["active_pixels"])


def test_case_neck_is_not_the_connector():
    # A tail wider than the connector steps in to the tab at its end: the case
    # must not take the tab for the cable and seal the connector inside.
    from opentac.cad.pipeline.case import _tail_neck
    tail = [(-18, 0), (-18, -14.75), (-15.25, -14.75), (-15.25, -18), (15.25, -18), (15.25, -14.75),
            (18, -14.75), (18, 0)]
    outline = [(-20, 0), *tail, (20, 0), (20, 40), (-20, 40)]
    assert _tail_neck(outline, 0.0) is None


# --- the check itself --------------------------------------------------------

def _ring(points, src="board"):
    P = [np.asarray(p, dtype=float) for p in points]
    return [(P[i], P[(i + 1) % len(P)], src) for i in range(len(P))]


def _tab_board(slit=False):
    """A board cut open at y=0 between x=-5 and 5, and a 'connector' tab below
    it from those two open ends. slit: the board runs on 4 mm past the opening
    beside the tab and back up to the open end 0.02 mm from the tab's side,
    which a fab would mill as a slit."""
    def lines(points, src):
        return [(np.array(a, float), np.array(b, float), src) for a, b in zip(points[:-1], points[1:])]

    if slit:
        board = lines([(5, 0), (5.02, -4), (5.5, -4), (5.5, 20), (-5, 20), (-5, 0)], "board")
    else:
        board = lines([(5, 0), (5, 20), (-5, 20), (-5, 0)], "board")
    return board + lines([(-5, 0), (-5, -6), (5, -6), (5, 0)], "connector")


def test_a_clean_outline_passes():
    assert check_edge_cuts(_tab_board()) == []


def test_board_beside_the_tab_is_a_slit():
    kinds = {p["type"] for p in check_edge_cuts(_tab_board(slit=True))}
    assert kinds & {"overlap", "slit"}


def test_a_line_inside_the_board_is_found():
    lines = _ring([(0, 0), (10, 0), (10, 10), (0, 10)]) + [(np.array([5.0, 0.0]), np.array([5.0, 4.0]), "connector")]
    assert {p["type"] for p in check_edge_cuts(lines)} == {"open_end"}


def test_a_line_drawn_twice_is_found():
    lines = _ring([(0, 0), (10, 0), (10, 10), (0, 10)])
    kinds = {p["type"] for p in check_edge_cuts(lines + [lines[0]])}
    assert "overlap" in kinds


def test_a_hairline_spike_is_found_and_cleaned():
    # a polygon union's artefact: the edge overshoots 0.045 mm and comes back
    ring = [(0, 0), (10, 0), (10, 10), (-1, 10), (-0.5, 6.0), (0.0, 5.0), (-0.02, 5.04), (0.5, 4.0)]
    assert "spike" in {p["type"] for p in check_edge_cuts(_ring(ring))}
    clean = remove_collinear_closed(np.array(ring, dtype=float))
    assert check_edge_cuts(_ring(clean)) == []
    # a sharp corner the outline was drawn with is not a hairline spike
    drawn = np.array([(0, 0), (10, 0), (10, 0.5), (20, 0.3), (10, 1), (0, 1)], dtype=float)
    assert np.array_equal(drop_hairline_spikes(drawn), drawn)
