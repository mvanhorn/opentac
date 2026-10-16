"""The cable tail is cut down to its wires after routing (pipeline/narrow_tail.py).

Narrowing must never change the sensor: the board is routed with the
connector-width tail either way, so the taxels and every wire are identical.
The narrow tail flares back to the connector's tab, so the board's Edge.Cuts
still meet the connector footprint's own outline, and a drawn tail extension
takes the landing with it to the moved connector.
"""
import copy

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import LineString, Polygon

from opentac.cad.main import CONNECTOR_DIR, _layout_from_edit_data, app
from opentac.cad.export.connector_geometry import connector_edge_geometries, stitched_outline_segments

client = TestClient(app)
PAD = [[0, 0], [17.6, 0], [17.6, 34.6], [0, 34.6]]   # an 8x4 gripper pad at 4.2 mm pitch
PARAMS = dict(pitch_x_mm=4.2, pitch_y_mm=4.2, pixel_w_mm=4.0, pixel_h_mm=4.0, router="hug")


def _gen(board_mode, narrow):
    r = client.post("/generate", json=dict(outline=PAD, board_mode=board_mode, narrow_tail=narrow, **PARAMS))
    assert r.status_code == 200, r.text[:300]
    return r.json()


@pytest.mark.parametrize("board_mode", ["expand", "fixed_keepout"])
def test_narrowing_changes_nothing_but_the_tail(board_mode):
    narrow, wide = _gen(board_mode, True), _gen(board_mode, False)
    tn, tw = narrow["edit_data"]["layout"]["tail"], wide["edit_data"]["layout"]["tail"]
    assert tn["mode"] == "fit" and tn["width_mm"] < tw["width_mm"]
    assert narrow["stats"]["active_pixels"] == wide["stats"]["active_pixels"]
    assert narrow["edit_data"]["routes"] == wide["edit_data"]["routes"]
    assert narrow["drc"]["violations"] == 0


def _outline_is_closed(edit_data):
    """Board segments plus the connector's own Edge.Cuts form closed loops."""
    layout, outline_xy, params, _ = _layout_from_edit_data(edit_data)
    segs = list(stitched_outline_segments(layout, outline_xy, params, CONNECTOR_DIR))
    edges = connector_edge_geometries(layout, outline_xy, params, CONNECTOR_DIR)
    segs += list(edges["lines"]) + [(a, b) for a, _m, b in edges["arcs"]]
    ends = {}
    for a, b in segs:
        for p in (a, b):
            k = (round(float(p[0]), 3), round(float(p[1]), 3))
            ends[k] = ends.get(k, 0) + 1
    return all(n % 2 == 0 for n in ends.values()), min(
        float(np.hypot(*(np.asarray(b) - np.asarray(a)))) for a, b in segs)


@pytest.mark.parametrize("board_mode", ["expand", "fixed_keepout"])
def test_narrow_tail_meets_the_connector(board_mode):
    closed, shortest = _outline_is_closed(_gen(board_mode, True)["edit_data"])
    assert closed
    assert shortest > 0.001, "a sub-micron edge reads as a malformed outline in KiCad"


def _width_at(edit_data, routes, depth):
    """Board width across the tail, `depth` mm out from the cable edge."""
    r = client.post("/redo-padding", json={"edit_data": edit_data, "routes": routes})
    assert r.status_code == 200, r.text[:300]
    ed = r.json()["edit_data"]
    t = ed["layout"]["tail"]
    o = np.asarray(ed["sensor_outline"] if "sensor_outline" in ed else ed["outline"], dtype=float)
    mid = 0.5 * (o[0] + o[1])
    p = mid + depth * np.asarray(t["n_out"], dtype=float)
    d = np.asarray(t["t_hat"], dtype=float)
    cut = Polygon(ed["final_outline"]).buffer(0).intersection(LineString([p - 100 * d, p + 100 * d]))
    return cut.length, t


def test_extension_takes_the_landing_with_it():
    g = _gen("expand", True)
    ed, routes = g["edit_data"], g["edit_data"]["routes"]
    t = ed["layout"]["tail"]
    landing = float(t["length_mm"]) + 1.0          # inside the connector landing
    w0, _ = _width_at(ed, routes, landing)
    assert w0 == pytest.approx(t["connector_min_width_mm"], abs=0.05)

    ext = copy.deepcopy(ed)
    te = dict(ext["layout"]["tail_extension"])
    s = np.asarray(te["path"][0], dtype=float)
    n = np.asarray(t["n_out"], dtype=float)
    te.update(enabled=True, path=[s.tolist(), (s + 10 * n).tolist()])
    ext["layout"]["tail_extension"] = te
    w1, _ = _width_at(ext, routes, landing)
    assert w1 < t["connector_min_width_mm"] - 1.0, "the old landing stayed behind mid-cable"
    r = client.post("/export-edited", json={"edit_data": ext, "routes": routes})
    assert r.status_code == 200 and r.json()["drc"]["violations"] == 0

    off = copy.deepcopy(ext)
    off["layout"]["tail_extension"] = dict(te, enabled=False)
    w2, _ = _width_at(off, routes, landing)
    assert w2 == pytest.approx(w0, abs=0.05)


_MEMORY_PROBE = r'''
import resource, sys
import opentac.cad.main as main
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
grew = []
narrow = main.narrow_routed_tail
def watched(*a, **k):
    before = peak()
    out = narrow(*a, **k)
    grew.append(peak() - before)
    return out
main.narrow_routed_tail = watched
r = main.generate(main.GenerateRequest(
    outline=[[-30, 0], [30, 0], [30, 60], [-30, 60]], board_mode="expand", router="hug",
    pixel_w_mm=2.0, pixel_h_mm=2.0, pitch_x_mm=2.2, pitch_y_mm=2.2))
unit = 1 if sys.platform == "darwin" else 1024      # ru_maxrss: bytes vs KiB
print(r.stats["total_pixels"], max(grew) * unit / 2**20)
'''


def test_narrowing_a_dense_grown_tail_does_not_spike_memory():
    """A grown 26 x 26 board runs hundreds of wire pieces through the old tail.
    Buffering them as one MultiLineString cost ~300 MB in GEOS (macOS 302 MB,
    Linux 266 MB on a dense insole), enough to OOM-kill the 512 MB hosted server
    mid-request ("Failed to fetch" on the page). Per piece it is a few MB.
    Measured in a fresh process: ru_maxrss is a high-water mark, so a peak an
    earlier test already reached would hide the spike."""
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c", _MEMORY_PROBE], capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    taxels, grew_mb = out.stdout.split()[-2:]
    assert int(taxels) > 600
    assert float(grew_mb) < 100, f"narrow_routed_tail raised peak memory by {float(grew_mb):.0f} MB"
