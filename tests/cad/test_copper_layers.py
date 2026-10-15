"""The copper-layers option (copper_layers = 2 | 4), end to end through the API
functions: a 4-layer board routes its rows on In1/In2 and exports those layers
everywhere (KiCad file, Gerbers on both exporter paths, preview, edit data,
manifest), keeps them through an edited export and a re-padding, and a
2-layer board is untouched by any of it."""

import base64
import io
import os
import sys
import zipfile


import numpy as np
import pytest
from fastapi import HTTPException

from opentac.cad import main
from opentac.cad.pipeline.drc import drc_check
from opentac.cad.pipeline.types import Route

T_SHAPE = [[-12, 0], [12, 0], [12, 34], [34, 34], [34, 52], [-34, 52], [-34, 34], [-12, 34]]


def _generate(layers, mode="expand", **kw):
    req = main.GenerateRequest(outline=T_SHAPE, router="hug", board_mode=mode, pixel_w_mm=3.0, pixel_h_mm=3.0,
                               pitch_x_mm=3.2, pitch_y_mm=3.2, copper_layers=layers, **kw)
    return main.generate(req)


def _zip(resp):
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64)))
    return z, z.read("board.kicad_pcb").decode()


@pytest.mark.parametrize("mode", ["expand", "fixed_keepout"])
def test_four_layers_route_rows_inside_and_export_the_inner_layers(mode):
    r = _generate(4, mode)
    z, pcb = _zip(r)
    assert r.stats["copper_layers"] == 4
    assert sum(r.stats["rows_on_inner_layers"].values()) > 0
    assert '(1 "In1.Cu" signal)' in pcb and '(2 "In2.Cu" signal)' in pcb
    assert pcb.count('(layer "In1.Cu")') > 0
    names = z.namelist()
    assert "gerber/board-In1_Cu.g1" in names and "gerber/board-In2_Cu.g2" in names
    assert r.drc["violations"] == 0
    assert 'class="in1-route"' in r.svg
    assert r.edit_data["params"]["copper_layers"] == 4
    assert "In1.Cu" in {x["layer"] for x in r.edit_data["routes"]}


def test_two_layers_has_no_inner_copper():
    r = _generate(2)
    z, pcb = _zip(r)
    assert r.stats["copper_layers"] == 2
    assert "In1.Cu" not in pcb
    assert not any("In1_Cu" in n or "In2_Cu" in n for n in z.namelist())
    assert {x["layer"] for x in r.edit_data["routes"]} <= {"F.Cu", "B.Cu"}


def test_copper_layers_must_be_two_or_four():
    with pytest.raises(HTTPException) as e:
        _generate(3)
    assert e.value.status_code == 400


def test_edited_export_and_repadding_keep_the_four_layer_stack():
    r = _generate(4, "fixed_keepout")
    e = main.export_edited(main.ExportEditedRequest(edit_data=r.edit_data, routes=r.edit_data["routes"]))
    _, pcb = _zip(e)
    assert e.stats["copper_layers"] == 4 and '(1 "In1.Cu" signal)' in pcb
    # asking a re-padding for 2 layers cannot drop the inner layers its wires are on
    p = main.redo_padding(main.RedoPaddingRequest(edit_data=r.edit_data, routes=r.edit_data["routes"],
                                                  params={"copper_layers": 2}))
    assert p.stats["copper_layers"] == 4


def test_python_gerber_fallback_writes_the_inner_layers(monkeypatch):
    # no kicad-cli (a hosted deployment): the direct Gerber writer must plot In1/In2 itself
    monkeypatch.setattr(main.shutil, "which", lambda name: None)
    r = _generate(4)
    z, _ = _zip(r)
    assert r.stats["gerber_exporter"] == "direct"
    in1 = z.read("gerber/board-In1_Cu.g1").decode()
    assert in1.count("D01*") > 0                     # route segments
    assert in1.count("D03*") >= r.stats["active_pixels"]   # every taxel via is flashed


def test_drc_checks_each_copper_layer_on_its_own():
    a = Route("ROW_0", 1, np.zeros(2), np.zeros(2), np.array([[0.0, -5.0], [0.0, 5.0]]))
    b = Route("ROW_1", 2, np.zeros(2), np.zeros(2), np.array([[-5.0, 0.0], [5.0, 0.0]]))
    assert drc_check([], [a, b], 0.2, 0.2)                       # crossing on B.Cu
    b.layer = "In1.Cu"
    assert not drc_check([], [a, b], 0.2, 0.2)                   # different copper layers


def test_a_near_miss_drops_the_taxel_beside_it_not_the_grid():
    # T at 3mm, 4 layers, fixed outline: the densest grid (127) has one column
    # wire 0.055mm too near the tail shoulder. Giving up the taxel beside it
    # keeps the grid (126); regrowing the band and re-packing used to ship 121,
    # fewer than the same board on 2 layers (123).
    four = _generate(4, "fixed_keepout")
    two = _generate(2, "fixed_keepout")
    assert four.drc["violations"] == 0
    assert four.stats["active_pixels"] > two.stats["active_pixels"]


# The 09x26 robot board (87x54mm, 3mm taxels) as fabricated: its outline and grid.
ROBOT_09X26 = [[-37.5, 0.0], [37.5, -0.0], [37.5, 28.37], [-37.5, 28.37], [-37.939, 28.347], [-38.373, 28.278], [-38.798, 28.164], [-39.208, 28.007], [-39.6, 27.807], [-39.969, 27.568], [-40.31, 27.291], [-40.621, 26.98], [-40.898, 26.639], [-41.137, 26.27], [-41.337, 25.878], [-41.494, 25.468], [-41.608, 25.043], [-41.677, 24.609], [-41.7, 24.17], [-41.7, 4.2], [-41.696, 4.017], [-41.651, 3.561], [-41.557, 3.113], [-41.414, 2.678], [-41.225, 2.261], [-40.992, 1.867], [-40.717, 1.5], [-40.404, 1.166], [-40.057, 0.868], [-39.679, 0.609], [-39.275, 0.394], [-38.85, 0.223], [-38.409, 0.1], [-37.957, 0.025]]


@pytest.mark.parametrize("layers", [2, 4])
def test_bent_tail_then_bigger_connector_exports(layers):
    # A bent tail extension, then a bigger connector on the routed board, then
    # the export: the shoulder's corner fill came back with a bare line in it
    # and the export crashed ('LineString' object has no attribute 'exterior').
    import math
    r = main.generate(main.GenerateRequest(outline=ROBOT_09X26, router="hug", board_mode="expand",
                                           pixel_w_mm=3.0, pixel_h_mm=3.0,
                                           pitch_x_mm=3.0, pitch_y_mm=3.0, copper_layers=layers))
    ed = r.edit_data
    t = ed["layout"]["tail"]
    c, n = t["pad_split_center"], t.get("n_out", [0.0, -1.0])
    a = math.radians(60)
    turn = [n[0] * math.cos(a) - n[1] * math.sin(a), n[0] * math.sin(a) + n[1] * math.cos(a)]
    p1 = [c[0] + 15 * n[0], c[1] + 15 * n[1]]
    ed["layout"]["tail_extension"] = {
        "enabled": True, "path": [c, p1, [p1[0] + 15 * turn[0], p1[1] + 15 * turn[1]]],
        "width_mm": max(0.0, t["width_mm"] - 0.7),
        "clearance_mm": ed["params"]["trace_w_mm"] / 2 + ed["params"]["board_edge_clear_mm"]}
    bigger = r.stats["connector_pos"] + 10
    resized = main.redo_padding(main.RedoPaddingRequest(edit_data=ed, routes=ed["routes"], connector_pos_override=bigger))
    out = main.export_edited(main.ExportEditedRequest(edit_data=resized.edit_data, routes=resized.edit_data["routes"]))
    assert out.stats["connector_pos"] == bigger
    assert out.drc["violations"] == 0


def test_four_layer_grow_keeps_the_drawn_edge_where_no_wire_runs():
    # Grow mode pads the drawn shape for wiring. On 4 layers the rows leave on
    # the inner layers, so nothing runs along the T's top edge: it stays where
    # it was drawn, while 2 layers still pad it (follow padding, 2mm).
    for layers, grows in ((4, False), (2, True)):
        r = _generate(layers, "expand")
        drawn_top = max(y for _, y in r.edit_data["outline"])
        board_top = max(y for _, y in r.edit_data["final_outline"])
        assert (board_top - drawn_top > 1.5) == grows, (layers, drawn_top, board_top)
        assert r.drc["violations"] == 0
