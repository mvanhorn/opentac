"""The cable tail is cut down to its wires after routing (pipeline/narrow_tail.py).

Narrowing must never change the sensor: the board is routed with the
connector-width tail either way, so the taxels and every wire are identical.
The narrow tail is a strip over the connector pads in use, which sit off the
connector's centre whenever rows and columns differ in number; it steps out to
the connector's tab right before the connector, each side by its own amount, so
the board's Edge.Cuts still meet the connector footprint's own outline, and a
drawn tail extension takes the landing with it to the moved connector.
"""
import base64
import copy
import io
import json
import shutil
import subprocess
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import LineString, Polygon

from opentac.cad.main import CONNECTOR_DIR, _layout_from_edit_data, app
from opentac.cad.export.connector_geometry import connector_edge_geometries, stitched_outline_segments
from opentac.cad.pipeline.case import _tail_neck
from opentac.cad.pipeline.geometry import (
    TAIL_FLARE_MAX_SLOPE, TAIL_STEP_CORNER_MM, concave_fillet_infill, fit_tail_strip, make_tail_poly,
    tail_offset_mm, tail_side_edges, tail_step_fillet, tail_widening_length,
)
from opentac.cad.pipeline.narrow_tail import MIN_STRAIGHT_MM, strip_within_cable

client = TestClient(app)
PAD = [[0, 0], [17.6, 0], [17.6, 34.6], [0, 34.6]]   # an 8x4 gripper pad at 4.2 mm pitch
PARAMS = dict(pitch_x_mm=4.2, pitch_y_mm=4.2, pixel_w_mm=4.0, pixel_h_mm=4.0, router="hug")


def _gen(board_mode, narrow, **over):
    r = client.post("/generate", json=dict(outline=PAD, board_mode=board_mode, narrow_tail=narrow,
                                           **PARAMS, **over))
    assert r.status_code == 200, r.text[:300]
    return r.json()


@pytest.mark.parametrize("board_mode", ["expand", "fixed_keepout"])
def test_narrowing_changes_nothing_but_the_tail(board_mode):
    # One typed length for both: an automatic one is chosen per tail (the
    # printed case needs a different length round each), and the routes then
    # differ by the straight run into each pad.
    narrow, wide = _gen(board_mode, True, cable_length_mm=12.0), _gen(board_mode, False, cable_length_mm=12.0)
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
    """A grown 26 x 26 board runs hundreds of wire pieces through the wide tail.
    Buffered as one MultiLineString, they cost about 300 MB in GEOS on a dense
    board, enough to exhaust the 512 MB hosted server mid-request; buffered
    piece by piece, a few MB. Measured in a fresh process: ru_maxrss is a
    high-water mark, so a peak that an earlier test already reached would hide
    the spike."""
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c", _MEMORY_PROBE], capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    taxels, grew_mb = out.stdout.split()[-2:]
    assert int(taxels) > 600
    assert float(grew_mb) < 100, f"narrow_routed_tail raised peak memory by {float(grew_mb):.0f} MB"


# ---------------------------------------------------------------------------
# The strip sits over the pads in use, which are off the connector's centre
# whenever rows and columns differ in number (rows on one connector half,
# columns on the other, each packed against the centre).
# ---------------------------------------------------------------------------
PITCH = SimpleNamespace(pad_pitch_mm=0.5)


def _block(lo, hi):
    return {p: f"NET_{p}" for p in range(lo, hi + 1)}


@pytest.mark.parametrize("conn,pads,edge,width,offset", [
    (60, (23, 57), 1.0, 19.0, 4.75),    # 8 rows on 23-30, 27 columns on 31-57
    (60, (2, 36), 1.0, 19.0, -5.75),    # 29 rows on 2-30, 6 columns on 31-36
    (30, (10, 27), 1.0, 10.5, 1.5),     # 6 on 10-15, 12 on 16-27
    (60, (23, 57), 2.1, 21.1, 4.7),     # the high side stops at the tab (15.25)
    (30, (1, 26), 2.1, 15.1, -0.2),     # the low side stops at the tab (-7.75)
])
def test_strip_is_centred_on_the_pad_block(conn, pads, edge, width, offset):
    w, off = fit_tail_strip(_block(*pads), conn, PITCH, edge)
    assert w == pytest.approx(width, abs=1e-6) and off == pytest.approx(offset, abs=1e-6)
    half_tab = (conn + 1) * 0.5 / 2
    centre = (conn + 1) / 2
    assert off - w / 2 >= -half_tab - 1e-9 and off + w / 2 <= half_tab + 1e-9
    # every used pad is inside the strip, at least `edge` from its cut edge
    # unless the strip stops at the tab
    for p in range(pads[0], pads[1] + 1):
        x = (p - centre) * 0.5
        assert off - w / 2 <= x <= off + w / 2
        assert x - (off - w / 2) >= edge - 1e-9 or abs(off - w / 2 + half_tab) < 1e-9
        assert (off + w / 2) - x >= edge - 1e-9 or abs(off + w / 2 - half_tab) < 1e-9


def _cut(poly, depth, mid=np.zeros(2), t_hat=np.array([1.0, 0.0]), n_out=np.array([0.0, -1.0])):
    p = mid + depth * n_out
    g = poly.intersection(LineString([p - 100 * t_hat, p + 100 * t_hat]))
    xs = [float((np.asarray(q) - mid) @ t_hat) for part in getattr(g, "geoms", [g]) for q in part.coords]
    return min(xs), max(xs)


def test_step_widens_each_side_by_its_own_amount():
    w, W, off, cable, r = 12.0, 30.5, 5.0, 24.0, 1.0
    total = cable + 6.0
    poly = make_tail_poly(np.zeros(2), [1.0, 0.0], [0.0, -1.0], w, W, cable, total, off, step_fillet=r)
    assert poly.is_valid
    step = r + TAIL_STEP_CORNER_MM
    assert tail_widening_length(w, W, cable - 1.0, off, r) == pytest.approx(step)
    # the plain strip right up to the step, the full tab from the landing on
    for d in (1.0, cable - step - 1e-3):
        lo, hi = _cut(poly, d)
        assert (lo, hi) == (pytest.approx(off - w / 2), pytest.approx(off + w / 2))
    lo, hi = _cut(poly, cable + 1.0)
    assert (lo, hi) == (pytest.approx(-W / 2), pytest.approx(W / 2))
    # square to the tail: past the straight edge (the corner's radius before the
    # landing) the board is the tab, but for the corner's own rounding
    lo, hi = _cut(poly, cable - TAIL_STEP_CORNER_MM + 1e-3)
    assert -W / 2 < lo < -W / 2 + TAIL_STEP_CORNER_MM and W / 2 - TAIL_STEP_CORNER_MM < hi < W / 2
    # no inside corner sharper than the fillet: rounding them to it adds nothing
    # but slivers of the arcs' polygon steps (a square corner would add 0.21 mm2)
    assert _inside_corner_wedge(poly, r) < 0.01


def test_a_small_step_keeps_its_fillet_and_a_flush_side_stays_straight():
    W, r = 30.5, 1.0
    # the high side moves out 0.3 mm, less than fillet + corner: an S of the same
    # two arcs, so no inside corner sharper than the fillet, and shorter
    w, off = W - 0.3, -0.15                       # the low side on the tab's edge
    low, high = tail_side_edges(w, W, 23.0, off, r)
    assert len(low) == 1 and low[0] == (0.0, pytest.approx(-W / 2))          # flush: straight on
    assert high[0][1] == pytest.approx(off + w / 2) and high[-1] == (0.0, pytest.approx(W / 2))
    assert 0.0 < high[0][0] < r + TAIL_STEP_CORNER_MM
    poly = make_tail_poly(np.zeros(2), [1.0, 0.0], [0.0, -1.0], w, W, 24.0, 30.0, off, step_fillet=r)
    assert poly.is_valid and _inside_corner_wedge(poly, r) < 0.01


def _inside_corner_wedge(poly, r):
    """The largest piece rounding the polygon's inside corners to `r` adds."""
    infill = concave_fillet_infill(poly, 0.98 * r)
    return max([g.area for g in getattr(infill, "geoms", [infill])], default=0.0)


def test_typed_widths_and_older_designs_keep_the_flare():
    """A width the user typed is routed with the long flare (the router fans
    wires into it), and a tail narrowed before the step came back as it was."""
    w, W, off = 12.0, 30.5, 5.0
    assert tail_widening_length(w, W, 100.0, off) == pytest.approx(
        np.pi / 2 * (off - w / 2 + W / 2) / TAIL_FLARE_MAX_SLOPE)
    assert tail_step_fillet({"mode": "fit", "fit_offset_mm": 2.0}) is None
    assert tail_step_fillet({"mode": "manual"}) is None
    assert tail_step_fillet({"mode": "fit", "step_fillet_mm": 1.0}) == 1.0


def test_a_typed_width_narrower_than_the_pads_still_routes_through_its_flare():
    g = client.post("/generate", json=dict(outline=[[0, 0], [34, 0], [34, 8], [0, 8]], tail_width_mm=6.0,
                                           cable_length_mm=24.0, **SMALL))
    assert g.status_code == 200, g.text[:300]
    t = g.json()["edit_data"]["layout"]["tail"]
    assert t["mode"] == "manual" and t.get("step_fillet_mm") is None
    assert g.json()["drc"]["violations"] == 0


def test_short_cable_narrows_fully_or_not_at_all():
    w, off, W, r = 21.1, 4.7, 30.5, 1.0
    step = tail_widening_length(w, W, float("inf"), off, r)
    assert step <= r + TAIL_STEP_CORNER_MM
    # every tail of the automatic 12 mm minimum narrows to its wires
    for cable in (30.0, 12.0, step + MIN_STRAIGHT_MM):
        assert strip_within_cable(w, off, W, cable, r) == (w, off)
    # too short for the step and the straight run past the board: the connector's width
    assert strip_within_cable(w, off, W, step + MIN_STRAIGHT_MM - 0.1, r) is None


def test_designs_saved_before_the_offset_have_none():
    assert tail_offset_mm({"mode": "fit", "fit_width_mm": 20.0, "width_mm": 20.0}) == 0.0
    assert tail_offset_mm({"mode": "manual", "fit_offset_mm": 3.0}) == 0.0
    assert tail_offset_mm({"mode": "fit", "fit_offset_mm": -2.5}) == -2.5


# A short wide pad (17 columns x 4 rows) needs the 60-pin board, a 12 x 3 one the
# 30-pin board; both use pads well off the connector's centre.
OFF_CENTRE = {
    "60pin": dict(outline=[[0, 0], [34, 0], [34, 8], [0, 8]], cable_length_mm=24.0),
    "30pin": dict(outline=[[0, 0], [24, 0], [24, 6], [0, 6]], cable_length_mm=12.0),
}
SMALL = dict(pitch_x_mm=2.0, pitch_y_mm=2.0, pixel_w_mm=1.8, pixel_h_mm=1.8, router="hug")
_CACHE = {}


def _gen_off(name, board_mode="expand", narrow=True):
    key = (name, board_mode, narrow)
    if key not in _CACHE:
        r = client.post("/generate", json=dict(board_mode=board_mode, narrow_tail=narrow, **SMALL, **OFF_CENTRE[name]))
        assert r.status_code == 200, r.text[:300]
        _CACHE[key] = r.json()
    return _CACHE[key]


def _strip_of(edit_data):
    t = edit_data["layout"]["tail"]
    return t["fit_offset_mm"] - t["width_mm"] / 2, t["fit_offset_mm"] + t["width_mm"] / 2


@pytest.mark.parametrize("name", sorted(OFF_CENTRE))
@pytest.mark.parametrize("board_mode", ["expand", "fixed_keepout"])
def test_off_centre_block_keeps_every_taxel(name, board_mode):
    narrow, wide = _gen_off(name, board_mode, True), _gen_off(name, board_mode, False)
    assert narrow["stats"]["active_pixels"] == wide["stats"]["active_pixels"]
    assert narrow["edit_data"]["routes"] == wide["edit_data"]["routes"]
    assert narrow["drc"]["violations"] == 0
    closed, shortest = _outline_is_closed(narrow["edit_data"])
    assert closed and shortest > 0.001


@pytest.mark.parametrize("name,conn", [("60pin", 60), ("30pin", 30)])
def test_off_centre_strip_is_as_narrow_as_its_wires(name, conn):
    g = _gen_off(name)
    ed = g["edit_data"]
    assert g["stats"]["connector_pos"] == conn
    t = ed["layout"]["tail"]
    pads = sorted(int(p) for p in ed["layout"]["pad_to_net"])
    centre, pitch = (conn + 1) / 2, ed["params"]["pad_pitch_mm"]
    block = ((pads[0] - centre) * pitch, (pads[-1] - centre) * pitch)
    assert abs(block[0] + block[1]) > 2.0, "the test design must use an off-centre block"
    edge = ed["params"]["trace_w_mm"] / 2 + ed["params"]["follow_main_padding_mm"]
    half_tab = (conn + 1) * pitch / 2
    lo, hi = max(-half_tab, block[0] - edge), min(half_tab, block[1] + edge)
    assert t["mode"] == "fit"
    assert t["width_mm"] == pytest.approx(hi - lo, abs=1e-3)
    assert t["fit_offset_mm"] == pytest.approx((lo + hi) / 2, abs=1e-3)
    # the board itself: the narrowest cut through the straight cable is the strip
    o = np.asarray(ed["outline"], dtype=float)
    mid = 0.5 * (o[0] + o[1])
    poly = Polygon(ed["final_outline"]).buffer(0)
    step = tail_widening_length(t["width_mm"], t["connector_min_width_mm"], float("inf"), t["fit_offset_mm"],
                                t["step_fillet_mm"])
    cuts = [_cut(poly, d, mid, np.asarray(t["t_hat"]), np.asarray(t["n_out"]))
            for d in np.arange(1.0, t["length_mm"] - step, 0.25)]
    narrowest = min(cuts, key=lambda c: c[1] - c[0])
    assert narrowest[1] - narrowest[0] == pytest.approx(hi - lo, abs=0.05)
    assert narrowest[0] == pytest.approx(lo, abs=0.05) and narrowest[1] == pytest.approx(hi, abs=0.05)


def test_off_centre_tail_survives_redo_and_reopen():
    g = _gen_off("60pin")
    ed, routes = g["edit_data"], g["edit_data"]["routes"]
    r = client.post("/redo-padding", json={"edit_data": ed, "routes": routes})
    assert r.status_code == 200
    redo = r.json()
    assert redo["drc"]["violations"] == 0
    assert redo["edit_data"]["layout"]["tail"]["fit_offset_mm"] == ed["layout"]["tail"]["fit_offset_mm"]
    assert Polygon(redo["edit_data"]["final_outline"]).area == pytest.approx(Polygon(ed["final_outline"]).area, abs=1e-6)
    r = client.post("/export-edited", json={"edit_data": redo["edit_data"], "routes": routes})
    assert r.status_code == 200 and r.json()["drc"]["violations"] == 0
    assert r.json()["stats"]["active_pixels"] == g["stats"]["active_pixels"]


def test_extension_corridor_stays_over_the_off_centre_lanes():
    g = _gen_off("60pin")
    ed, routes = copy.deepcopy(g["edit_data"]), g["edit_data"]["routes"]
    t = ed["layout"]["tail"]
    te = dict(ed["layout"]["tail_extension"])
    s = np.asarray(te["path"][0], dtype=float)
    n = np.asarray(t["n_out"], dtype=float)
    te.update(enabled=True, path=[s.tolist(), (s + 30 * n).tolist()])
    ed["layout"]["tail_extension"] = te
    r = client.post("/redo-padding", json={"edit_data": ed, "routes": routes})
    assert r.status_code == 200 and r.json()["drc"]["violations"] == 0
    out = r.json()["edit_data"]
    o = np.asarray(out["outline"], dtype=float)
    mid = 0.5 * (o[0] + o[1])
    depth = float((s - mid) @ n) + 4.0                      # down the extension, above its step
    lo, hi = _cut(Polygon(out["final_outline"]).buffer(0), depth, mid, np.asarray(t["t_hat"]), n)
    assert (lo + hi) / 2 == pytest.approx(t["fit_offset_mm"], abs=0.05)
    r = client.post("/export-edited", json={"edit_data": out, "routes": routes})
    assert r.status_code == 200 and r.json()["drc"]["violations"] == 0
    assert r.json()["stats"]["active_pixels"] == g["stats"]["active_pixels"]


def test_case_slot_follows_an_off_centre_tail():
    ed = _gen_off("60pin")["edit_data"]
    cable_y = float(np.asarray(ed["outline"], dtype=float)[:2, 1].min())
    neck = _tail_neck(ed["final_outline"], cable_y)
    assert neck is not None
    lo, hi = _strip_of(ed)
    assert neck[1] == pytest.approx(lo, abs=0.3) and neck[2] == pytest.approx(hi, abs=0.3)
    r = client.post("/case", json={"edit_data": ed})
    assert r.status_code == 200 and r.json()["stats"]["cable_edge"] == "slot"


@pytest.mark.skipif(not shutil.which("kicad-cli"), reason="kicad-cli not installed")
@pytest.mark.parametrize("name", sorted(OFF_CENTRE))
def test_off_centre_outline_is_valid_in_kicad(name, tmp_path):
    """KiCad reads the stitched outline as one closed board, and no copper is
    closer to it than its default rule except the connector's own pads at its
    own tab (as on every export)."""
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(_gen_off(name)["zip_b64"])))
    pcb = tmp_path / "board.kicad_pcb"
    pcb.write_bytes(z.read(next(n for n in z.namelist() if n.endswith(".kicad_pcb"))))
    rep = tmp_path / "drc.json"
    subprocess.run(["kicad-cli", "pcb", "drc", "--format", "json", "--severity-all", "-o", str(rep), str(pcb)],
                   check=True, capture_output=True, timeout=300)
    violations = json.loads(rep.read_text())["violations"]
    assert not [v for v in violations if v["type"] == "invalid_outline"]
    edge = [v for v in violations if v["type"] == "copper_edge_clearance"
            and not any("J1" in i["description"] for i in v["items"])]
    assert not edge, edge[:3]
