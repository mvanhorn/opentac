"""The exported Edge.Cuts close into one outline with every line on it once.

The board's outline is exported cut open at the connector, and the connector
footprint's own Edge.Cuts draw the tab from the two open ends. A board whose
wires run to the outermost connector pins used to have its tail padded past the
tab's sides (post-route containment round the outermost wires) down to the pad
drills: the footprint's tab-side lines then lay inside the board, and a fab
mills them as slits. KiCad's DRC passes such a board (it folds a slit into the
outline as a zero-width notch), so these tests read the exported KiCad file
itself and run export/edge_cuts_check.py on it.
"""
import base64
import io
import re
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import LineString

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
    # The old behaviour: post-route containment pads the tail's own wires too.
    monkeypatch.setattr(main, "tail_wire_hold", lambda layout: None)
    resp = _generate(**FULL_WIDTH["30pin_15x15"])
    found = [d for d in resp["drc"]["details"] if d["layer"] == "Edge.Cuts"]
    assert found and resp["drc"]["violations"] >= len(found)
    assert check_edge_cuts(kicad_pcb_edge_cuts(_board_file(resp)))


# --- the check itself --------------------------------------------------------

def _ring(points, src="board"):
    P = [np.asarray(p, dtype=float) for p in points]
    return [(P[i], P[(i + 1) % len(P)], src) for i in range(len(P))]


def _tab_board(slit=False):
    """A board cut open at y=0 between x=-5 and 5, and a 'connector' tab below
    it from those two open ends. slit: the board runs on 4 mm past the opening
    beside the tab, as the bug did, and back up to the open end 0.02 mm from
    the tab's side."""
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
