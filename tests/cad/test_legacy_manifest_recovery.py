"""A board exported without a design block is still usable.

Boards have been fabricated from manifests that carry geometry and nothing
else. A case and a bump sheet for such a board are a question of geometry,
which those files contain.

The property tested here is fidelity, not merely that the file loads.
Re-generating the board from what can be inferred gives bit-exact taxel
centres but a different cut boundary, so a case printed from it would fit a
board that differs slightly from the one manufactured. Every assertion below
is therefore against the stored geometry, not against a regenerated one.
"""
import copy
import json


import numpy as np
import pytest
from fastapi import HTTPException
from shapely.geometry import Point, Polygon

from opentac.cad.main import (
    generate, open_design, case, bump_sheet,
    GenerateRequest, OpenDesignRequest, CaseRequest, BumpSheetRequest,
)
from opentac.cad.export.legacy_manifest import RING_MAX_DEVIATION_MM

SQUARE = [[-20.5, 0.0], [20.5, 0.0], [20.5, 41.0], [-20.5, 41.0]]
LSHAPE = [[0, 0], [40, 0], [40, 18], [18, 18], [18, 40], [0, 40]]


def _v3(outline, **kw):
    """A current export, stripped back to a version 3 manifest (no design block)."""
    import base64, io, zipfile
    resp = generate(GenerateRequest(outline=outline, router="hug",
                                    board_mode="expand", **kw))
    man = json.loads(zipfile.ZipFile(
        io.BytesIO(base64.b64decode(resp.zip_b64))).read("manifest.json"))
    man.pop("design", None)
    man["v"] = 3
    return man


@pytest.fixture(scope="module")
def legacy():
    return _v3(SQUARE, pixel_w_mm=3, pixel_h_mm=3, pitch_x_mm=3.2, pitch_y_mm=3.2)


def test_it_opens_in_geometry_mode(legacy):
    op = open_design(OpenDesignRequest(file=legacy))
    assert op.mode == "geometry"
    assert op.edit_data and op.edit_data.get("geometry_only") is True
    assert op.svg and op.svg.startswith("<svg")
    # The caller must be told what it cannot do, not left to discover it.
    joined = " ".join(op.warnings).lower()
    assert "route" in joined and ("gerber" in joined or "export" in joined)
    # Nothing to re-generate from: offering params here would invite the
    # regeneration that this path avoids.
    assert op.params == {} and op.routes == []


def test_the_board_is_the_stored_one_not_a_regenerated_one(legacy):
    """Every dimension comes from the file, to within the ring cleanup."""
    op = open_design(OpenDesignRequest(file=legacy))
    ed = op.edit_data

    stored_px = np.asarray([p[:2] for p in legacy["pixels"]], float)
    got_px = np.asarray(ed["layout"]["placements_xy"], float)
    assert got_px.shape == stored_px.shape
    assert np.abs(got_px - stored_px).max() == 0.0, "taxel centres must be untouched"

    assert ed["outline"] == [[float(x), float(y)] for x, y in legacy["outline"]]

    a = Polygon(legacy["edge_cuts"])
    b = Polygon(ed["final_outline"])
    for lo, hi in zip(a.bounds, b.bounds):
        assert abs(lo - hi) <= RING_MAX_DEVIATION_MM, "board bounds moved"
    assert abs(a.area - b.area) / a.area < 1e-4, "board area changed"

    # Taxel size and pitch are read back, not defaulted.
    assert ed["params"]["pixel_w_mm"] == legacy["w"]
    assert ed["params"]["pitch_x_mm"] == pytest.approx(3.2, abs=1e-3)
    assert ed["params"]["gap_mm"] == legacy["footprint"]["gap"]
    for key in ("pixel_w_mm", "pixel_h_mm", "pitch_x_mm", "pitch_y_mm", "gap_mm"):
        assert key in ed["recovered_params"], f"{key} should be marked as recovered"


def test_it_carries_the_preview_box_the_editor_needs(legacy):
    """The route editor converts mm to SVG through edit_data["preview"].

    It does so on load() even when there is nothing to edit, so a recovered
    board needs this key. The box also matches the SVG's own viewBox;
    otherwise every overlay would land in the wrong place.
    """
    op = open_design(OpenDesignRequest(file=legacy))
    box = op.edit_data.get("preview")
    assert box, "no preview box: the route editor cannot place anything"
    for k in ("xmin", "ymin", "xmax", "ymax"):
        assert k in box
    board = np.asarray(op.edit_data["final_outline"], float)
    assert box["xmin"] < board[:, 0].min() and box["xmax"] > board[:, 0].max()
    assert box["ymin"] < board[:, 1].min() and box["ymax"] > board[:, 1].max()

    vb = op.svg.split('viewBox="')[1].split('"')[0].split()
    assert float(vb[2]) == pytest.approx(box["xmax"] - box["xmin"], abs=0.01), \
        "the preview box and the drawing disagree on width"
    assert float(vb[3]) == pytest.approx(box["ymax"] - box["ymin"], abs=0.01), \
        "the preview box and the drawing disagree on height"


def test_pitch_comes_from_the_taxels_not_from_px_py(legacy):
    """px/py are pad-ordered and py can be negative, as the manifest states.

    Read as the pitch, they would give a board whose row pads are split a case
    with the wrong taxel spacing.
    """
    poisoned = copy.deepcopy(legacy)
    poisoned["px"] = 99.0
    poisoned["py"] = -99.0
    ed = open_design(OpenDesignRequest(file=poisoned)).edit_data
    assert ed["params"]["pitch_x_mm"] == pytest.approx(3.2, abs=1e-3)
    assert ed["params"]["pitch_y_mm"] == pytest.approx(3.2, abs=1e-3)


@pytest.mark.parametrize("outline,kw", [
    (SQUARE, dict(pixel_w_mm=2, pixel_h_mm=2, pitch_x_mm=2, pitch_y_mm=2)),
    (SQUARE, dict(pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2)),
    (LSHAPE, dict(pixel_w_mm=3, pixel_h_mm=3, pitch_x_mm=3.2, pitch_y_mm=3.2)),
])
def test_a_recovered_board_builds_a_case_and_a_bump_sheet(outline, kw):
    """A recovered board yields a case and a bump sheet: physical parts for a
    board that already exists."""
    ed = open_design(OpenDesignRequest(file=_v3(outline, **kw))).edit_data

    c = case(CaseRequest(edit_data=ed))
    assert c.stats["bottom_watertight"] and c.stats["top_watertight"], \
        "a case that is not a closed solid will not slice"
    assert c.stats["taxels"] == len(ed["layout"]["placements_xy"])
    assert len(c.bottom_stl_b64) > 1000 and len(c.top_stl_b64) > 1000

    b = bump_sheet(BumpSheetRequest(edit_data=ed))
    assert len(b.stl_b64) > 1000


def test_the_case_is_built_on_the_stored_board():
    """A case for a fabricated board follows the fabricated edge.

    The test moves the stored boundary and checks that the case moves with it:
    a recovery that re-derived the outline instead of using the file's would
    produce the same case either way.
    """
    man = _v3(SQUARE, pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2)
    grown = copy.deepcopy(man)
    grown["edge_cuts"] = [[x * 1.25, y * 1.25] for x, y in man["edge_cuts"]]

    a = case(CaseRequest(edit_data=open_design(OpenDesignRequest(file=man)).edit_data))
    b = case(CaseRequest(edit_data=open_design(OpenDesignRequest(file=grown)).edit_data))
    assert b.stats["bottom_watertight"]
    assert a.stats != b.stats, "the case ignored the stored cut boundary"


def test_the_ring_cleanup_is_checked_not_trusted():
    """The cleanup does not move a real edge.

    shapely's hausdorff_distance can return NaN on such boundaries, and NaN
    fails every `>` comparison, so a guard written as a plain comparison would
    accept every cleanup unchecked. Here the stored ring is given a 2 mm spike,
    which the cleanup keeps.
    """
    man = _v3(SQUARE, pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2)
    ring = [list(p) for p in man["edge_cuts"]]
    i = len(ring) // 2
    spike = [ring[i][0], ring[i][1] + 2.0]
    ring.insert(i, spike)
    man["edge_cuts"] = ring

    ed = open_design(OpenDesignRequest(file=man)).edit_data
    poly = Polygon(ed["final_outline"])
    assert poly.exterior.distance(Point(*spike)) <= RING_MAX_DEVIATION_MM, \
        "a real 2mm feature was smoothed off the manufactured edge"


def test_a_current_export_is_untouched_by_this_path():
    """A current ZIP restores exactly and does not fall into geometry mode."""
    import base64, io, zipfile
    resp = generate(GenerateRequest(outline=SQUARE, router="hug", board_mode="expand",
                                    pixel_w_mm=4, pixel_h_mm=4,
                                    pitch_x_mm=4.2, pitch_y_mm=4.2))
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64)))
    assert open_design(OpenDesignRequest(file=json.loads(z.read("design.json")))).mode == "restore"
    assert open_design(OpenDesignRequest(file=json.loads(z.read("manifest.json")))).mode == "recipe"


def test_invalid_file_says_why():
    with pytest.raises(HTTPException) as e:
        open_design(OpenDesignRequest(file={"pixels": [[0, 0, 0, 0]], "outline": [[0, 0]]}))
    assert e.value.status_code == 400 and "outline" in e.value.detail.lower()
