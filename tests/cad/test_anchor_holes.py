"""Anchor cutouts: the clearance guarantee, measured on the emitted board.

The point of these tests is that DRC cannot cover this. DRC checks trace vs
trace and trace vs board edge; it has no idea the anchor cutouts exist, so it
reports clean whether the cutouts clear the copper or sit on top of it. The
first version of this feature passed DRC with 0.05mm of copper-to-cutout gap.
So the assertion here rebuilds every copper feature and measures.
"""
import base64
import io
import re
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from opentac.cad import main
from opentac.cad.export.kicad_footprint import pixel_geometry

client = TestClient(main.app)
SQUARE = [[0, 0], [45, 0], [45, 38], [0, 38]]
LSHAPE = [[0, 0], [50, 0], [50, 20], [25, 20], [25, 40], [0, 40]]

VARIANTS = [
    ("uniform", True), ("uniform", False),
    ("paired", True), ("paired", False),
]


def _generate(outline, **kw):
    r = client.post("/generate", json={"outline": outline, "router": "hug", **kw})
    assert r.status_code == 200, r.text[:400]
    return r.json()


def _pcb_text(d):
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(d["zip_b64"])))
    return z.read([n for n in z.namelist() if n.endswith(".kicad_pcb")][0]).decode("utf-8", "replace")


def _cutouts(pcb):
    return [(float(m.group(1)), -float(m.group(2)), float(m.group(3)) - float(m.group(1)))
            for m in re.finditer(
                r'\(gr_circle \(center ([-\d.]+) ([-\d.]+)\) \(end ([-\d.]+) [-\d.]+\) '
                r'\(layer "Edge\.Cuts"\)', pcb)]


def _copper(pcb, d):
    """Every copper feature on the board: emitted traces plus taxel combs."""
    ed = d["edit_data"]
    pr = ed["params"]
    P = np.array(ed["layout"]["placements_xy"], float)
    removed = {int(i) for i in (ed.get("removed_pix") or [])}
    g = pixel_geometry(pr["pixel_w_mm"], pr["pixel_h_mm"], pr["trace_w_mm"], pr["gap_mm"],
                       pr["center_clear_mm"], pr["edge_clear_mm"], pr["via_dia_mm"])
    tw = pr["trace_w_mm"]
    geoms = []
    for m in re.finditer(r'\(segment \(start ([-\d.]+) ([-\d.]+)\) \(end ([-\d.]+) ([-\d.]+)\) '
                         r'\(width ([-\d.]+)\)', pcb):
        x1, y1, x2, y2, w = (float(v) for v in m.groups())
        if (x1, y1) != (x2, y2):
            geoms.append(LineString([(x1, -y1), (x2, -y2)]).buffer(w / 2, quad_segs=8))
    for i in range(len(P)):
        if i in removed:
            continue
        cx, cy = P[i]
        for x0 in (g.xL, g.xR):
            geoms.append(LineString([(cx + x0, cy + g.spine_y_lo), (cx + x0, cy + g.spine_y_hi)])
                         .buffer(tw / 2, quad_segs=8))
        for k in range(g.total_slots):
            y = g.y0 + k * g.pitch
            a, b = (g.xL, g.x_end_left_max) if k % 2 == 0 else (g.xR, g.col_end_x[k])
            geoms.append(LineString([(cx + a, cy + y), (cx + b, cy + y)])
                         .buffer(tw / 2, quad_segs=8))
        geoms.append(Point(cx + g.via_x, cy + g.via_y).buffer(pr["via_dia_mm"] / 2, quad_segs=12))
    return unary_union(geoms)


@pytest.mark.parametrize("lattice,square", VARIANTS)
@pytest.mark.parametrize("outline", [SQUARE, LSHAPE], ids=["square", "lshape"])
def test_copper_clears_every_cutout(outline, lattice, square):
    clearance = 0.4
    d = _generate(outline, anchor_holes=True, anchor_hole_lattice=lattice,
                  anchor_hole_square_cells=square, anchor_hole_clearance_mm=clearance)
    pcb = _pcb_text(d)
    holes = _cutouts(pcb)
    assert holes, "anchor holes requested but none emitted"
    cu = _copper(pcb, d)
    worst = min(cu.distance(Point(cx, cy).buffer(r, quad_segs=32)) for cx, cy, r in holes)
    assert worst >= clearance - 1e-6, (
        f"{lattice}/square={square}: copper comes within {worst:.3f}mm of a cutout, "
        f"want {clearance}mm")


@pytest.mark.parametrize("lattice,square", VARIANTS)
def test_cutouts_lie_inside_the_board(lattice, square):
    """A cutout crossing the outline is a bite out of the edge, not a hole."""
    from shapely.geometry import Polygon
    d = _generate(SQUARE, anchor_holes=True, anchor_hole_lattice=lattice,
                  anchor_hole_square_cells=square)
    board = Polygon(np.array(d["edit_data"]["final_outline"], float))
    for cx, cy, r in _cutouts(_pcb_text(d)):
        assert board.contains(Point(cx, cy).buffer(r)), f"cutout at ({cx:.2f},{cy:.2f}) crosses the outline"


def test_bumps_land_on_the_cutouts():
    d = _generate(SQUARE, anchor_holes=True, anchor_hole_lattice="paired",
                  anchor_hole_square_cells=False)
    spec = d["edit_data"]["layout"]["routing"]["anchor_holes"]
    r = client.post("/bump-sheet", json={"edit_data": d["edit_data"]})
    assert r.status_code == 200, r.text[:300]
    assert r.json()["stats"]["bumps"] == len(spec["centers"])


def test_paired_keeps_more_taxels_than_uniform():
    """The whole reason the paired lattice exists."""
    counts = {}
    for lattice in ("uniform", "paired"):
        d = _generate(SQUARE, anchor_holes=True, anchor_hole_lattice=lattice,
                      anchor_hole_square_cells=False)
        counts[lattice] = d["stats"]["active_pixels"]
    assert counts["paired"] > counts["uniform"], counts


def test_off_by_default_emits_no_cutouts():
    assert not _cutouts(_pcb_text(_generate(SQUARE)))


@pytest.mark.parametrize("lattice", ["uniform", "paired"])
def test_cutouts_survive_a_re_export(lattice):
    """Regression: /export-edited used to silently drop every cutout.

    `_params_to_dict` is an explicit whitelist and the anchor fields were not
    in it, so re-exporting rebuilt Params with anchor_holes defaulting to
    False. The failure was invisible on screen -- the preview and the bump
    sheet read the cutouts from `layout.routing`, which DOES round-trip, so
    only the downloaded Edge.Cuts was missing them.
    """
    d = _generate(SQUARE, anchor_holes=True, anchor_hole_lattice=lattice)
    before = len(_cutouts(_pcb_text(d)))
    assert before

    stored = d["edit_data"]["params"]
    for key in ("anchor_holes", "anchor_hole_dia_mm", "anchor_hole_clearance_mm",
                "anchor_hole_lattice", "anchor_hole_square_cells"):
        assert key in stored, f"{key} missing from edit_data params — will not round-trip"

    r = client.post("/export-edited", json={"edit_data": d["edit_data"],
                                            "routes": d["edit_data"].get("routes") or []})
    assert r.status_code == 200, r.text[:300]
    assert len(_cutouts(_pcb_text(r.json()))) == before
