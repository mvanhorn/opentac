"""An exported design must re-open and reproduce the same board.

manifest.json describes the board that was MADE. That is not the same thing as
describing how to make it: /generate takes about thirty-five inputs and the
manifest carried seven, so a re-import used to have to guess board_mode, the
router and every clearance, and would silently
produce a different board.

Two records now come out of an export (export/design.py), and this pins both:

  manifest.json["design"] — the recipe. Re-generating from it must reproduce
      the board EXACTLY: same cut boundary, same taxels, same DRC.
  design.json             — the save file. Restoring it through
      /export-edited must give back the same board, hand-dragged routes
      included, and must refuse rather than guess when it was written by a
      different pipeline version.

The reproduction check compares the re-generated manifest's `edge_cuts` and
`pixels` against the original's, not just the taxel count: the failure this
guards against is a board that comes back the right size with the wrong
wiring keep-out, which a count check cannot see.
"""
import base64, io, json, sys, os, zipfile


import numpy as np
import pytest
from fastapi import HTTPException

from opentac.cad.main import (
    generate, export_edited, open_design,
    GenerateRequest, ExportEditedRequest, OpenDesignRequest,
)
from opentac.cad.export.design import PIPELINE_VERSION, DESIGN_SCHEMA

SQUARE = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
TEE = [[-12, 0], [12, 0], [12, 34], [34, 34], [34, 52], [-34, 52], [-34, 34], [-12, 34]]


def _zip(resp):
    return zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64)))


def _manifest(resp):
    return json.loads(_zip(resp).read("manifest.json"))


def _design(resp):
    return json.loads(_zip(resp).read("design.json"))


def _assert_same_board(a, b, tag):
    """Same cuts, same taxel positions, same taxel copper, same scan matrix.

    All four, because each covers parameters the others cannot see. Comparing
    only edge_cuts and pixels passes even when via_dia_mm or gap_mm have been
    lost from the recipe — the board is the right shape with the wrong copper
    in it — and comparing only geometry misses a net/pad reassignment, which
    would ship a board whose firmware reads the wrong taxel.
    """
    ea = np.asarray(a["edge_cuts"], dtype=float)
    eb = np.asarray(b["edge_cuts"], dtype=float)
    assert ea.shape == eb.shape, \
        f"{tag}: cut boundary has {len(eb)} points, original had {len(ea)}"
    assert np.abs(ea - eb).max() < 1e-6, \
        f"{tag}: cut boundary moved by {np.abs(ea - eb).max():.4f}mm"

    pa = np.asarray([p[:2] for p in a["pixels"]], dtype=float)
    pb = np.asarray([p[:2] for p in b["pixels"]], dtype=float)
    assert pa.shape == pb.shape, f"{tag}: {len(pb)} taxels, original had {len(pa)}"
    assert np.abs(pa - pb).max() < 1e-6, \
        f"{tag}: taxels moved by {np.abs(pa - pb).max():.4f}mm"

    # The taxel's interdigitated copper. w/h alone are only its bounding box.
    for key in ("w", "h"):
        assert a[key] == b[key], f"{tag}: {key} is {b[key]}, was {a[key]}"
    assert a["footprint"] == b["footprint"], \
        f"{tag}: taxel copper changed: {b['footprint']} vs {a['footprint']}"

    # The scan matrix: which connector pad reads which taxel.
    for key in ("r", "c", "rp", "cp", "m"):
        assert a[key] == b[key], f"{tag}: scan field {key} changed"


CASES = [
    ("square/expand/hug", dict(outline=SQUARE, router="hug", board_mode="expand",
                               pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2)),
    ("square/fixed/hug", dict(outline=SQUARE, router="hug", board_mode="fixed_keepout",
                              pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2)),
    ("tee/expand/hug", dict(outline=TEE, router="hug", board_mode="expand",
                            pixel_w_mm=3, pixel_h_mm=3, pitch_x_mm=3.2, pitch_y_mm=3.2)),
    ("square/expand/legacy", dict(outline=SQUARE, router="legacy", board_mode="expand",
                                  pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2)),
    # Solved grid: the recipe stores the SOLVED taxel size, not target_cols/rows,
    # so a replay must not re-solve and land somewhere else.
    ("square/target-grid", dict(outline=SQUARE, router="hug", board_mode="expand",
                                target_cols=6, target_rows=6)),
    # Every value deliberately off its default, including a hand-upsized
    # connector. Without this, a recipe that silently fell back to defaults
    # would pass every case above — most designs are drawn near the defaults,
    # so "reproduced" and "regenerated from scratch" look identical.
    # test_the_recipe_is_load_bearing pins that down from the other side.
    # (Trace + clearance stays under the 0.5mm pin pitch: at 0.25 + 0.28 the
    # wires to neighbouring pins cannot clear each other, generate now refuses
    # it, and before that this board shipped 5 unconnected nets.)
    ("square/all-non-default", dict(
        outline=SQUARE, router="hug", board_mode="fixed_keepout",
        pixel_w_mm=3.4, pixel_h_mm=2.8, pitch_x_mm=3.9, pitch_y_mm=3.3,
        edge_keepout_mm=0.8, trace_w_mm=0.22, gap_mm=0.25, clearance_mm=0.26,
        center_clear_mm=0.3, edge_clear_mm=0.15, via_drill_mm=0.2, via_dia_mm=0.4,
        routing_margin_mm=2.1, cable_length_mm=17.0, tail_side_padding_mm=1.3,
        board_edge_clear_mm=0.35, connector_pos_override=40)),
]


@pytest.mark.parametrize("tag,req", CASES, ids=[c[0] for c in CASES])
def test_recipe_reproduces_the_board(tag, req):
    """manifest.json["design"] + manifest.json["outline"] -> the same board."""
    original = generate(GenerateRequest(**req))
    man = _manifest(original)

    design = man.get("design")
    assert design, f"{tag}: manifest has no design block"
    assert design["schema"] == DESIGN_SCHEMA
    assert design["router"] in ("legacy", "hug")
    assert design["params"]["board_mode"] in ("expand", "fixed_keepout")

    opened = open_design(OpenDesignRequest(file=man))
    assert opened.mode == "recipe", f"{tag}: a manifest can only be a recipe"

    replay = generate(GenerateRequest(outline=opened.outline, **opened.params))
    _assert_same_board(man, _manifest(replay), f"{tag} recipe replay")
    assert replay.drc["violations"] == original.drc["violations"], \
        f"{tag}: replay DRC {replay.drc['violations']} vs original {original.drc['violations']}"
    assert replay.stats["active_pixels"] == original.stats["active_pixels"]
    assert replay.stats["connector_pos"] == original.stats["connector_pos"]
    assert replay.stats["router"] == original.stats["router"]


@pytest.mark.parametrize("tag,req", CASES, ids=[c[0] for c in CASES])
def test_save_file_restores_the_board(tag, req):
    """design.json -> /export-edited gives the same board back."""
    original = generate(GenerateRequest(**req))
    saved = _design(original)
    assert saved["kind"] == "design"
    assert saved["pipeline"] == PIPELINE_VERSION
    assert saved["edit_data"]["params"]["board_mode"] == \
        _manifest(original)["design"]["params"]["board_mode"]
    assert saved["routes"], f"{tag}: design.json carries no routes"

    opened = open_design(OpenDesignRequest(file=saved))
    assert opened.mode == "restore", f"{tag}: a current design.json must restore exactly"
    assert not opened.warnings

    restored = export_edited(ExportEditedRequest(
        edit_data=opened.edit_data, routes=opened.routes))
    _assert_same_board(_manifest(original), _manifest(restored), f"{tag} restore")
    assert restored.drc["violations"] == original.drc["violations"]


def test_the_recipe_is_load_bearing():
    """The other direction: the outline ALONE must not reproduce the board.

    Every reproduction test above is worthless if the design block turns out
    to carry no information the defaults don't already supply. Re-generating
    the all-non-default board from its outline and nothing else has to give a
    visibly different board — otherwise those tests are green for the wrong
    reason and would stay green if the block were dropped entirely.
    """
    req = dict(CASES[-1][1])
    original = generate(GenerateRequest(**req))
    man = _manifest(original)
    opened = open_design(OpenDesignRequest(file=man))

    bare = _manifest(generate(GenerateRequest(outline=opened.outline)))
    with pytest.raises(AssertionError):
        _assert_same_board(man, bare, "outline-only")

    # ...and with the recipe it does reproduce, on the same two boards.
    _assert_same_board(man, _manifest(
        generate(GenerateRequest(outline=opened.outline, **opened.params))), "recipe")


def test_hand_edited_routes_survive_a_round_trip():
    """The point of design.json: a dragged route comes back dragged.

    /export-edited's ZIP has to carry the caller's LIVE routes, not
    edit_data["routes"] (the routes the board was generated from) — otherwise
    re-opening an edited export quietly reverts every edit.
    """
    original = generate(GenerateRequest(
        outline=SQUARE, router="hug", board_mode="expand",
        pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2))
    edit_data = original.edit_data
    routes = [dict(r) for r in edit_data["routes"]]

    # Bow one route sideways, the way a drag in the route editor does.
    target = next(i for i, r in enumerate(routes) if len(r["polyline_xy"]) > 3)
    moved = [list(p) for p in routes[target]["polyline_xy"]]
    mid = len(moved) // 2
    moved[mid] = [moved[mid][0] + 0.7, moved[mid][1]]
    routes[target] = {**routes[target], "polyline_xy": moved}

    edited = export_edited(ExportEditedRequest(edit_data=edit_data, routes=routes))
    reopened = open_design(OpenDesignRequest(file=_design(edited)))
    assert reopened.mode == "restore"
    got = reopened.routes[target]["polyline_xy"][mid]
    assert abs(got[0] - moved[mid][0]) < 1e-6 and abs(got[1] - moved[mid][1]) < 1e-6, \
        f"edited vertex came back as {got}, expected {moved[mid]}"
    # ...and the untouched routes are still untouched.
    assert len(reopened.routes) == len(routes)


def test_fill_region_round_trips():
    """A sensorize zone was dropped by every export before this.

    layout.sensor_outline_xy is set to the outline, not to the fill region, so
    the zone existed only in the browser: re-generating from a downloaded file
    would sensorize the whole board.
    """
    fill = [[-12, -12], [12, -12], [12, 12], [-12, 12]]
    original = generate(GenerateRequest(
        outline=SQUARE, fill_region=fill, router="hug", board_mode="expand",
        pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2))
    man = _manifest(original)

    assert man["design"]["fill_region"], "manifest design block dropped the fill region"
    assert original.edit_data["fill_region"], "edit_data dropped the fill region"

    opened = open_design(OpenDesignRequest(file=man))
    assert opened.fill_region and len(opened.fill_region) >= 3
    replay = generate(GenerateRequest(
        outline=opened.outline, fill_region=opened.fill_region, **opened.params))
    _assert_same_board(man, _manifest(replay), "fill-region replay")

    # And the zone genuinely constrained the pack — otherwise this test would
    # pass just as well if fill_region were ignored on both runs.
    unconstrained = generate(GenerateRequest(
        outline=SQUARE, router="hug", board_mode="expand",
        pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2))
    assert replay.stats["active_pixels"] < unconstrained.stats["active_pixels"]


def test_a_design_from_another_pipeline_falls_back_to_the_recipe():
    """A stored layout is only meaningful against the pipeline that built it."""
    original = generate(GenerateRequest(
        outline=SQUARE, router="hug", board_mode="expand",
        pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2))
    stale = {**_design(original), "pipeline": "1999.01.01"}

    opened = open_design(OpenDesignRequest(file=stale))
    assert opened.mode == "recipe", "a stale save file must not be replayed as-is"
    assert opened.edit_data is None and opened.routes is None
    assert any("1999.01.01" in w for w in opened.warnings), opened.warnings
    # It still has to be openable — degrading to the recipe is the whole point.
    replay = generate(GenerateRequest(outline=opened.outline, **opened.params))
    _assert_same_board(_manifest(original), _manifest(replay), "stale-save recipe")


def test_prefer_recipe_forces_a_regenerate():
    original = generate(GenerateRequest(
        outline=SQUARE, router="hug", board_mode="expand",
        pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2))
    opened = open_design(OpenDesignRequest(file=_design(original), prefer="recipe"))
    assert opened.mode == "recipe"
    assert opened.edit_data is None


def test_bad_files_say_why():
    original = generate(GenerateRequest(
        outline=SQUARE, router="hug", board_mode="expand",
        pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2))
    man = _manifest(original)

    # A manifest from before the design block is NOT an error: it opens in
    # geometry mode, which is what makes an already-fabricated board usable
    # for a case. See test_legacy_manifest_recovery.py for that behaviour.
    old = {k: v for k, v in man.items() if k != "design"}
    old["v"] = 3
    assert open_design(OpenDesignRequest(file=old)).mode == "geometry"

    # ...but one with no taxels at all describes no board.
    with pytest.raises(HTTPException) as e:
        open_design(OpenDesignRequest(file={**old, "pixels": []}))
    assert e.value.status_code == 400 and "taxel" in e.value.detail.lower()

    # A file from a NEWER app version: refuse rather than half-read it.
    with pytest.raises(HTTPException) as e:
        open_design(OpenDesignRequest(file={**man, "design": {**man["design"],
                                                              "schema": DESIGN_SCHEMA + 1}}))
    assert e.value.status_code == 400 and "newer version" in e.value.detail

    # Not one of ours at all.
    with pytest.raises(HTTPException) as e:
        open_design(OpenDesignRequest(file={"hello": "world"}))
    assert e.value.status_code == 400

    # A truncated design.json.
    with pytest.raises(HTTPException) as e:
        open_design(OpenDesignRequest(file={"kind": "design", "edit_data": {"outline": SQUARE}}))
    assert e.value.status_code == 400 and "params" in e.value.detail


def test_manifest_stays_readable_by_mcu_code():
    """The design block is additive: nothing an MCU reads may have changed."""
    man = _manifest(generate(GenerateRequest(
        outline=SQUARE, router="hug", board_mode="expand",
        pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2)))
    for key in ("units", "outline", "edge_cuts", "pixels", "w", "h", "footprint",
                "r", "c", "px", "py", "ox", "oy", "rp", "cp", "m"):
        assert key in man, f"manifest lost {key}"
    assert len(man["m"]) == man["r"]
    assert all(len(row) == man["c"] for row in man["m"])
