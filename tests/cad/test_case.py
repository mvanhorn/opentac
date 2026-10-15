"""Two-piece case: the invariants that only show up across board shapes.

Every assertion here corresponds to a bug found by sweeping shapes rather than
by reasoning about one board. Keeping the sweep is the point: a single square
board passes all of this even when the geometry is badly wrong.
"""
import base64
import io
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import MultiPolygon, Polygon

from opentac.cad import main
from opentac.cad.pipeline.case import MIN_WEB_MM, CaseParams, _tail_neck, build_case

client = TestClient(main.app)

SHAPES = {
    "square": [[0, 0], [42, 0], [42, 42], [0, 42]],
    "lshape": [[0, 0], [50, 0], [50, 20], [25, 20], [25, 40], [0, 40]],
    "small": [[0, 0], [18, 0], [18, 18], [0, 18]],
    "wide": [[0, 0], [70, 0], [70, 25], [0, 25]],
}


def _board(outline):
    # readout_board "any": these fixtures were sized around the smallest connector
    # that fits (20-pos on the small board); the tests are about the case's tail
    # detection, not about which readout board the sensor plugs into.
    r = client.post("/generate", json={"outline": outline, "router": "hug", "readout_board": "any"})
    assert r.status_code == 200, r.text[:300]
    ed = r.json()["edit_data"]
    P = np.asarray(ed["layout"]["placements_xy"], dtype=float)
    removed = {int(i) for i in (ed.get("removed_pix") or [])}
    active = [i for i in range(len(P)) if i not in removed]
    cable_y = float(np.asarray(ed["outline"], dtype=float)[:2, 1].min())
    return ed, P, active, cable_y


def _build(outline, **kw):
    ed, P, active, cable_y = _board(outline)
    pr = ed["params"]
    return build_case(ed["final_outline"], cable_y, P, active,
                      pr["pitch_x_mm"], pr["pitch_y_mm"], CaseParams(**kw)), ed


def _pieces(g):
    return len(g.geoms) if isinstance(g, MultiPolygon) else 1


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("bumps", [False, True], ids=["flat", "bumped"])
def test_both_halves_are_printable_solids(shape, bumps):
    result, _ = _build(SHAPES[shape], bumps=bumps)
    for mesh, name in ((result.bottom, "bottom"), (result.top, "top")):
        assert mesh.is_watertight, f"{name} is not a closed solid"
        assert mesh.body_count == 1, f"{name} came out as {mesh.body_count} loose bodies"
        assert mesh.volume > 0


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("bumps", [False, True], ids=["flat", "bumped"])
def test_the_lid_stays_one_piece(shape, bumps):
    """An unbroken slot severs the lid; the rib gaps are what prevent it.

    Measured on the square board, a continuous slot left the inner panel hanging
    off a 1.4mm neck at the cable edge.
    """
    result, _ = _build(SHAPES[shape], bumps=bumps)
    assert result.stats["lid_pieces"] == 1, result.warnings


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_the_cable_edge_is_a_slot_not_an_open_side(shape):
    """The cable line is NOT where the tail is — the board is still full width
    there, so clipping at it opens a whole side of wall for a narrow tail."""
    result, _ = _build(SHAPES[shape])
    s = result.stats
    assert s["cable_edge"] == "slot", f"{shape} fell back to an open side: {result.warnings}"
    assert s["tail_slot_mm"] < 0.6 * s["case_w_mm"], (
        f"tail slot {s['tail_slot_mm']}mm is most of a {s['case_w_mm']}mm case — the neck "
        "detector has latched onto the shoulder rather than the tail")


def test_neck_detection_is_not_a_fraction_of_board_width():
    """A small board's tail is a LARGE fraction of it — 14.5 of 22.9mm — so any
    fixed fraction that catches a big board misses a small one."""
    found = {}
    for name, outline in SHAPES.items():
        ed, _, _, cable_y = _board(outline)
        neck = _tail_neck(ed["final_outline"], cable_y)
        assert neck is not None, f"no tail found on {name}"
        found[name] = neck[2] - neck[1]
    # The tail is sized by the connector, not the board: a board four times the
    # area does not get a four-times-wider tail.
    assert found["small"] < 1.5 * found["square"] * 1.5
    for name, w in found.items():
        assert 5.0 < w < 40.0, f"{name} tail width {w:.1f}mm is not plausible"


def test_scan_line_takes_the_widest_run_not_the_extent():
    """On an asymmetric outline the scan line hits two disjoint pieces; taking
    the bounds of both reported the L-shape's tail as 33mm instead of 19.5."""
    ed, _, _, cable_y = _board(SHAPES["lshape"])
    neck = _tail_neck(ed["final_outline"], cable_y)
    assert neck is not None
    assert (neck[2] - neck[1]) < 25.0, f"tail measured {neck[2] - neck[1]:.1f}mm"


def test_small_boards_keep_enough_bridges():
    """A fixed period does not scale down. An early version used the gap as a
    floor and so made the period LONGER on a small board, not shorter."""
    result, _ = _build(SHAPES["small"])
    s = result.stats
    assert s["rib_period_mm"] < CaseParams().rib_period_mm
    assert s["rib_gap_mm"] < CaseParams().rib_gap_mm, "gap must scale with the period"
    assert s["bridges"] >= CaseParams().min_bridges - 1


def test_narrow_gap_is_reported():
    result, _ = _build(SHAPES["square"], rib_gap_mm=0.5)
    assert any("bridge" in w or "tear" in w for w in result.warnings), result.warnings


def test_slits_are_cut_only_when_there_are_bumps():
    flat, _ = _build(SHAPES["square"], bumps=False)
    bumped, _ = _build(SHAPES["square"], bumps=True)
    assert flat.stats["slit"] is False
    assert bumped.stats["slit"] is True
    assert bumped.stats["bumps"] > 0
    # Cutting the tiles removes material.
    assert bumped.top.volume < flat.top.volume + 1e-6 + bumped.stats["bumps"] * 100


def test_the_web_between_slit_and_slot_survives_the_default_skirt():
    """Two concentric cuts near the pocket edge leave a ligament carrying the
    whole tile field. At a 3mm skirt it is 0.65mm — under two extrusion widths,
    which is why the case defaults wider than the bump sheet does."""
    result, _ = _build(SHAPES["square"], bumps=True)
    web = result.stats["slit_to_slot_mm"]
    assert web is not None
    assert web >= MIN_WEB_MM, f"web is {web}mm at default params"
    assert not any("between the bump slits" in w for w in result.warnings)


def test_narrow_skirt_warns_about_the_web():
    result, _ = _build(SHAPES["square"], bumps=True, skirt_width_mm=3.0)
    assert any("between the bump slits" in w for w in result.warnings), result.warnings


@pytest.mark.parametrize("bumps", [False, True], ids=["flat", "bumped"])
def test_endpoint_returns_two_stls(bumps):
    ed, _, _, _ = _board(SHAPES["square"])
    r = client.post("/case", json={"edit_data": ed, "params": {"bumps": bumps}})
    assert r.status_code == 200, r.text[:300]
    j = r.json()
    for key in ("bottom_stl_b64", "top_stl_b64"):
        raw = base64.b64decode(j[key])
        assert len(raw) > 1000, f"{key} is suspiciously small"
    assert j["stats"]["bottom_watertight"] and j["stats"]["top_watertight"]


def test_endpoint_rejects_edit_data_without_a_board():
    r = client.post("/case", json={"edit_data": {"params": {}}})
    assert r.status_code == 400
    assert "generate a board first" in r.json()["detail"].lower()


def test_bump_sheet_still_works():
    """The case is a separate endpoint precisely so this keeps working."""
    ed, _, _, _ = _board(SHAPES["square"])
    r = client.post("/bump-sheet", json={"edit_data": ed})
    assert r.status_code == 200, r.text[:300]
    assert r.json()["stats"]["bumps"] > 0


# ---------------------------------------------------------------------------
# Tail detection across board shapes and parameters.
#
# A 42x42 square passes every one of these even when the detector is badly
# wrong: it was the only board tested when the search window was 9mm deep, and
# a 65x65 broke it two different ways. The sweep is the test.
# ---------------------------------------------------------------------------

def _neck(outline, **extra):
    body = {"outline": outline, "router": "hug"}
    body.update(extra)
    r = client.post("/generate", json=body)
    assert r.status_code == 200, r.text[:300]
    ed = r.json()["edit_data"]
    cable_y = float(np.asarray(ed["outline"], dtype=float)[:2, 1].min())
    return _tail_neck(ed["final_outline"], cable_y)


@pytest.mark.parametrize("size", [30, 40, 50, 60, 65, 70, 80, 90])
def test_tail_found_across_board_sizes(size):
    neck = _neck([[0, 0], [size, 0], [size, size], [0, size]])
    assert neck is not None, f"{size}mm board fell back to an open cable edge"
    width = neck[2] - neck[1]
    assert 8.0 < width < 45.0, f"tail measured {width:.1f}mm"


@pytest.mark.parametrize("label,extra", [
    ("fixed_keepout", {"board_mode": "fixed_keepout"}),
    ("follow_padding_4mm", {"follow_main_padding_mm": 4.0}),
    ("target_grid_20x20", {"target_cols": 20, "target_rows": 20}),
    ("big_connector", {"connector_pos_override": 60}),
    ("no_smoothing", {"smooth_follow_padding": False}),
])
def test_tail_found_across_parameters(label, extra):
    """Two of these fell back when the scan only looked 9mm below the cable edge:
    at 4mm follow padding the neck is at -9.25, and a 20x20 grid settles at -7.75
    but needs samples to -9.75 to confirm it holds."""
    neck = _neck([[0, 0], [65, 0], [65, 65], [0, 65]], **extra)
    assert neck is not None, f"{label} fell back to an open cable edge"


def test_full_width_board_falls_back_and_explains_why():
    """With follow padding off the outline is full width down to the connector:
    the tail is INSIDE the board, not protruding, so an open edge is the only
    thing the shape allows. The message has to say that, because the fix is a
    different setting rather than a different tolerance."""
    from opentac.cad.pipeline.case import _no_tail_reason
    outline = [[0, 0], [65, 0], [65, 65], [0, 65]]
    r = client.post("/generate", json={"outline": outline, "router": "hug",
                                       "follow_main_padding": False}).json()
    ed = r["edit_data"]
    cable_y = float(np.asarray(ed["outline"], dtype=float)[:2, 1].min())
    assert _tail_neck(ed["final_outline"], cable_y) is None
    why = _no_tail_reason(ed["final_outline"], cable_y)
    assert "full width" in why and "follow-padding" in why, why


def test_slot_scales_with_the_connector():
    """The tail is sized by the connector, so a bigger one needs a wider slot."""
    small = _neck([[0, 0], [40, 0], [40, 40], [0, 40]])
    big = _neck([[0, 0], [40, 0], [40, 40], [0, 40]], connector_pos_override=60)
    assert big is not None and small is not None
    assert (big[2] - big[1]) > (small[2] - small[1]) + 5.0


# ---------------------------------------------------------------------------
# Tail length. The case eats skirt + tolerance + grip before any tail emerges,
# and the connector body is another 6mm that cannot bend, so a short cable can
# produce a case that seals the connector inside and still exports a clean STL.
# ---------------------------------------------------------------------------

def _case_with_cable(outline, cable_mm):
    r = client.post("/generate", json={"outline": outline, "router": "hug",
                                       "cable_length_mm": cable_mm})
    assert r.status_code == 200, r.text[:300]
    ed = r.json()["edit_data"]
    P = np.asarray(ed["layout"]["placements_xy"], dtype=float)
    removed = {int(i) for i in (ed.get("removed_pix") or [])}
    active = [i for i in range(len(P)) if i not in removed]
    cable_y = float(np.asarray(ed["outline"], dtype=float)[:2, 1].min())
    pr = ed["params"]
    return build_case(ed["final_outline"], cable_y, P, active,
                      pr["pitch_x_mm"], pr["pitch_y_mm"], CaseParams())


def _cable_warning(result):
    return [w for w in result.warnings if "extension" in w]


def test_default_cable_is_long_enough_on_typical_boards():
    for outline in (SHAPES["square"], SHAPES["small"]):
        result = _case_with_cable(outline, 12.0)
        assert not _cable_warning(result), result.warnings


def test_short_cable_is_reported():
    result = _case_with_cable([[0, 0], [65, 0], [65, 65], [0, 65]], 2.0)
    warn = _cable_warning(result)
    assert warn, result.warnings
    assert "PAST the end of the board" in warn[0]
    assert result.stats["tail_free_mm"] < 0
    assert result.stats["tail_short_by_mm"] > 0


def test_case_over_the_connector_is_reported():
    """Board end clear of the case, but the case still sits on the connector body."""
    result = _case_with_cable([[0, 0], [65, 0], [65, 65], [0, 65]], 4.0)
    warn = _cable_warning(result)
    assert warn, result.warnings
    assert result.stats["tail_flex_mm"] <= 0
    assert "connector cannot plug in" in warn[0]


def test_the_shortfall_we_report_actually_fixes_it():
    """The number in the message is only useful if adding it clears the warning.

    Free tail moves one-for-one with cable length -- the neck, and so the clip,
    does not move -- so the shortfall is directly the extension length needed.
    """
    outline = [[0, 0], [65, 0], [65, 65], [0, 65]]
    base = 12.0
    result = _case_with_cable(outline, base)
    short = result.stats["tail_short_by_mm"]
    assert short and short > 0, "expected the 65x65 default to be flagged"

    fixed = _case_with_cable(outline, base + short)
    assert not _cable_warning(fixed), (
        f"adding the reported {short}mm did not clear the warning: {fixed.warnings}")
    assert fixed.stats["tail_flex_mm"] >= CaseParams().min_tail_clear_mm - 0.15


def test_longer_cable_never_triggers_it():
    result = _case_with_cable([[0, 0], [65, 0], [65, 65], [0, 65]], 20.0)
    assert not _cable_warning(result), result.warnings
    assert result.stats["tail_short_by_mm"] is None


# ---------------------------------------------------------------------------
# A drawn tail extension is NOT in the board outline: apply_tail_extension moves
# the connector pads and re-routes to them, but final_outline_xy never changes.
# Anything measuring the outline is blind to it, so a user who adds a 15mm
# extension used to see the identical "cable too short" warning.
# ---------------------------------------------------------------------------
import copy


def _with_extension(edit_data, points):
    ed = copy.deepcopy(edit_data)
    te = dict(ed["layout"]["tail_extension"])
    start = te["path"][0]
    te.update({"enabled": True, "path": [start] + points})
    ed["layout"]["tail_extension"] = te
    return ed


def _case_stats(edit_data):
    r = client.post("/case", json={"edit_data": edit_data, "params": {}})
    assert r.status_code == 200, r.text[:300]
    j = r.json()
    return j["stats"], [w for w in j["warnings"] if "extension" in w or "sealed" in w]


def _short_board():
    r = client.post("/generate", json={"outline": [[0, 0], [65, 0], [65, 65], [0, 65]],
                                       "router": "hug"})
    assert r.status_code == 200, r.text[:300]
    return r.json()["edit_data"]


def test_extension_counts_towards_the_cable_check():
    ed = _short_board()
    base, warn = _case_stats(ed)
    assert warn, "expected the 65x65 default to be flagged"

    start = ed["layout"]["tail_extension"]["path"][0]
    ext, warn2 = _case_stats(_with_extension(ed, [[start[0], start[1] - 15.0]]))
    assert ext["tail_extension_mm"] == pytest.approx(15.0, abs=0.05)
    assert ext["tail_flex_mm"] > base["tail_flex_mm"] + 14.0
    assert not warn2, warn2


def test_drawing_the_reported_shortfall_clears_the_warning():
    """Same contract as the cable-length message: the number has to work."""
    ed = _short_board()
    base, warn = _case_stats(ed)
    short = base["tail_short_by_mm"]
    assert short and short > 0

    start = ed["layout"]["tail_extension"]["path"][0]
    fixed, warn2 = _case_stats(
        _with_extension(ed, [[start[0], start[1] - (short + 0.3)]]))
    assert not warn2, f"drawing the reported {short}mm did not clear it: {warn2}"


def test_extension_uses_arc_length_not_straight_line():
    """A user-drawn path can turn a corner; every millimetre of it is cable."""
    ed = _short_board()
    start = ed["layout"]["tail_extension"]["path"][0]
    bent, _ = _case_stats(_with_extension(ed, [[start[0], start[1] - 5.0],
                                               [start[0] + 5.0, start[1] - 5.0]]))
    assert bent["tail_extension_mm"] == pytest.approx(10.0, abs=0.05)


def test_disabled_extension_is_ignored():
    ed = _short_board()
    off = copy.deepcopy(ed)
    te = dict(off["layout"]["tail_extension"])
    start = te["path"][0]
    te.update({"enabled": False, "path": [start, [start[0], start[1] - 15.0]]})
    off["layout"]["tail_extension"] = te
    s, warn = _case_stats(off)
    assert s["tail_extension_mm"] == 0.0
    assert warn, "a disabled extension must not count"
