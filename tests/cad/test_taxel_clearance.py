"""A column wire on F.Cu must clear the taxels it passes: on F.Cu both of a
taxel's electrodes are combs, so a wire grazing a neighbouring taxel's frame
shorts a column to a row. The tool's DRC checks it (drc_check taxels=), and the
legacy router's smoothing keeps column shortcuts that far from every taxel."""



import numpy as np
from shapely.geometry import LineString
import pytest

from opentac.cad import main
from opentac.cad.pipeline.drc import TaxelCopper, drc_check
from opentac.cad.pipeline.types import Route


def _col(net, pts):
    return Route(net, 1, np.zeros(2), np.zeros(2), np.asarray(pts, dtype=float))


def test_drc_flags_a_column_wire_grazing_another_columns_taxel():
    # two 2mm taxels at pitch 2: COL_0's at x=0, COL_1's at x=2 (copper 0.1mm in)
    taxels = TaxelCopper([[0.0, 0.0], [2.0, 0.0]], ["COL_0", "COL_1"], set(), 2.0, 2.0, 0.1)
    # COL_0 runs down beside its own taxel and cuts COL_1's corner 0.05mm off its copper
    near = _col("COL_0", [[0.8, 3.0], [0.8, 1.2], [1.0, 1.15], [2.0, 1.15]])
    found = drc_check([near], [], 0.2, 0.2, taxels=taxels)
    assert [v.type for v in found] == ["taxel_clearance"] * len(found) and found
    assert all(v.net2 == "TAXEL:COL_1" for v in found)
    # landing on its own taxel is fine, and nothing is checked without taxels=
    own = _col("COL_0", [[0.0, 3.0], [0.0, 0.5]])
    assert not drc_check([own], [], 0.2, 0.2, taxels=taxels)
    assert not drc_check([near], [], 0.2, 0.2)


@pytest.mark.parametrize("router", ["legacy", "hug"])
def test_fixed_board_ships_no_column_wire_against_a_taxel(router):
    # The fabricated 17x09 board's outline and grid (20x35 mm, 2 mm taxels at
    # 2 mm pitch), 2 layers, fixed outline; the hug order falls back to the
    # legacy router here. Wire smoothing keeps its distance from the
    # bottom-left taxels, whose corners COL_0 and COL_1 would otherwise cut
    # (0.2 mm clearance is needed), so all 77 taxels ship with DRC 0.
    r = main.generate(main.GenerateRequest(
        outline=[[-10.0, 0.0], [10.0, 0.0], [10.0, 35.0], [-10.0, 35.0]], router=router,
        board_mode="fixed_keepout", pixel_w_mm=2.0, pixel_h_mm=2.0, pitch_x_mm=2.0, pitch_y_mm=2.0))
    assert r.drc["violations"] == 0
    assert r.stats["active_pixels"] == 77
    ed = r.edit_data
    taxels = TaxelCopper(ed["layout"]["placements_xy"], ed["layout"]["col_net_per_pixel"],
                         set(ed.get("removed_pix") or ()), 2.0, 2.0, ed["params"]["edge_clear_mm"])
    cols = [_col(x["net_name"], x["polyline_xy"]) for x in ed["routes"] if x["layer"] == "F.Cu"]
    assert not drc_check(cols, [], 0.2, 0.2, taxels=taxels)


@pytest.mark.parametrize("w,h,tw,gap", [(1.5, 1.5, 0.2, 0.2), (2.0, 2.0, 0.2, 0.2), (3.0, 3.0, 0.2, 0.2),
                                        (3.1, 3.1, 0.2, 0.2), (4.0, 4.0, 0.2, 0.2), (3.4, 2.8, 0.2, 0.2),
                                        (2.0, 2.08, 0.1, 0.1)])
def test_taxel_row_via_clears_every_column_finger(w, h, tw, gap):
    # The row via is wider than the row spine, so a column finger at its height
    # stops center_clear short of the via; footprints whose fingers already
    # clear it keep them at full length.
    from shapely.geometry import Point
    from opentac.cad.export.kicad_footprint import pixel_geometry
    cc, via = 0.2, 0.35
    g = pixel_geometry(w, h, tw, gap, cc, 0.1, via)
    ring = Point(g.via_x, g.via_y).buffer(via / 2, 256)
    for i in range(1, g.total_slots, 2):
        y = g.y0 + i * g.pitch
        finger = LineString([(g.xR, y), (g.col_end_x[i], y)]).buffer(tw / 2, 256)
        assert finger.distance(ring) >= cc - 1e-6
        if g.col_end_x[i] != g.x_end_right_min:
            # shortened only because the full-length finger was too close
            full = LineString([(g.xR, y), (g.x_end_right_min, y)]).buffer(tw / 2, 256)
            assert full.distance(ring) < cc
