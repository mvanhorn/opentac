"""Velostat cut files: the laser DXF and the fold-over paper template.

The fold template only works if the two printed outlines coincide once the
sheet is folded, i.e. one is the exact mirror of the other across the crease.
An asymmetric board is the real test: on a square, a plain copy and a mirror
look the same.
"""
import base64
import io
import re
import zipfile

import ezdxf
import numpy as np
from shapely.geometry import Polygon


from opentac.cad.main import GenerateRequest, _layout_from_edit_data, generate
from opentac.cad.export.print_pdf import MM_TO_PT
from opentac.cad.export.velostat import velostat_dxf, velostat_fold_pdf, velostat_polygon

SQUARE = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
# Cable edge first (points 0 -> 1), a notch cut out of the upper right.
ELL = [[-20, -20], [20, -20], [20, 5], [0, 5], [0, 25], [-20, 25]]
HUGE = [[-170, -210], [170, -210], [170, 210], [-170, 210]]


def _generate(outline, **kw):
    body = {"outline": outline, "router": "hug", "board_mode": "expand",
            "pixel_w_mm": 4, "pixel_h_mm": 4, "pitch_x_mm": 4.2, "pitch_y_mm": 4.2}
    body.update(kw)
    return generate(GenerateRequest(**body))


def _polylines(pdf: bytes):
    """Closed polylines from the first page, in mm."""
    stream = re.findall(rb"stream\n(.*?)\nendstream", pdf, re.S)[0].decode("latin-1")
    out = []
    for m in re.finditer(r"([\d.]+) ([\d.]+) m\n((?:[\d.]+ [\d.]+ l\n)+)h S", stream):
        pts = [(float(m.group(1)), float(m.group(2)))]
        pts += [tuple(map(float, p)) for p in re.findall(r"([\d.]+) ([\d.]+) l", m.group(3))]
        out.append(np.array(pts) / MM_TO_PT)
    return out


def test_export_zip_carries_both_velostat_files():
    res = _generate(SQUARE)
    names = zipfile.ZipFile(io.BytesIO(base64.b64decode(res.zip_b64))).namelist()
    assert "velostat/velostat_cut.dxf" in names
    assert "velostat/velostat_fold_template.pdf" in names
    v = res.stats["velostat"]
    assert v["dxf_b64"] and v["fold_pdf_b64"]


def test_piece_is_the_drawn_outline_not_the_board():
    """The drawn shape: no tail, and inside the manufactured board."""
    res = _generate(ELL)
    layout, outline_xy, _p, _r = _layout_from_edit_data(res.edit_data)
    poly = velostat_polygon(layout, outline_xy)
    drawn = Polygon(outline_xy).buffer(0)
    assert poly.symmetric_difference(drawn).area < 1e-6
    final = Polygon(layout.final_outline_xy).buffer(0)
    assert poly.area < final.area            # the tail and routing margin are not covered
    assert poly.difference(final).area < 1e-6


def _uncovered_taxels(res):
    layout, outline_xy, params, removed = _layout_from_edit_data(res.edit_data)
    poly = velostat_polygon(layout, outline_xy).buffer(1e-3)
    hw, hh = params.pixel_w_mm / 2, params.pixel_h_mm / 2
    return [i for i, (x, y) in enumerate(layout.placements_xy) if i not in removed
            and not Polygon([(x - hw, y - hh), (x + hw, y - hh), (x + hw, y + hh),
                             (x - hw, y + hh)]).within(poly)]


def test_cable_on_an_inner_edge_covers_every_taxel():
    """With the cable on the stem's side of a T, the velostat covers every
    taxel, including the T's whole right arm: the board is not clipped at the
    cable-edge line."""
    tee = [[-12, 0], [12, 0], [12, 34], [34, 34], [34, 52], [-34, 52], [-34, 34], [-12, 34]]
    res = _generate(tee[1:] + tee[:1])     # cable edge = (12,0)->(12,34)
    assert _uncovered_taxels(res) == []


def test_dxf_round_trips_at_true_size():
    res = _generate(ELL)
    layout, outline_xy, _p, _r = _layout_from_edit_data(res.edit_data)
    poly = velostat_polygon(layout, outline_xy)
    doc = ezdxf.read(io.StringIO(velostat_dxf(poly)))
    assert doc.units == ezdxf.units.MM
    plines = list(doc.modelspace().query("LWPOLYLINE"))
    assert len(plines) == 1 + len(poly.interiors)
    assert all(p.closed for p in plines)
    got = Polygon([tuple(p[:2]) for p in plines[0].get_points()])
    minx, miny, maxx, maxy = poly.bounds
    assert abs(got.area - Polygon(poly.exterior).area) < 1e-3
    assert got.bounds[0] == 0 and got.bounds[1] == 0
    assert abs(got.bounds[2] - (maxx - minx)) < 1e-6
    assert abs(got.bounds[3] - (maxy - miny)) < 1e-6


def test_fold_halves_are_mirror_images_across_the_crease():
    res = _generate(ELL)
    layout, outline_xy, _p, _r = _layout_from_edit_data(res.edit_data)
    poly = velostat_polygon(layout, outline_xy)
    info = {}
    pdf = velostat_fold_pdf(poly, info=info)
    loops = _polylines(pdf)
    assert len(loops) == 2, "one outline per half"
    a, b = loops
    axis = 0 if info["fold"] == "vertical" else 1
    crease = (a[:, axis].max() + b[:, axis].min()) / 2.0 if a[:, axis].mean() < b[:, axis].mean() \
        else (b[:, axis].max() + a[:, axis].min()) / 2.0
    b_back = b.copy()
    b_back[:, axis] = 2 * crease - b_back[:, axis]
    # Same vertices, point for point, once folded back.
    assert np.abs(a - b_back).max() < 0.01
    # And at 1:1: each half is the real piece's size.
    size = a.max(axis=0) - a.min(axis=0)
    minx, miny, maxx, maxy = poly.bounds
    assert abs(size[0] - (maxx - minx)) < 0.01 and abs(size[1] - (maxy - miny)) < 0.01
    # An L is not symmetric, so a plain translated copy would NOT have passed.
    assert np.abs((b - b.mean(0)) - (a - a.mean(0))).max() > 1.0


def test_too_big_to_fold_still_ships_the_dxf():
    res = _generate(HUGE, pixel_w_mm=22, pixel_h_mm=22, pitch_x_mm=22.4, pitch_y_mm=22.4)
    v = res.stats["velostat"]
    assert v["dxf_b64"]
    assert v["fold_pdf_b64"] is None


def test_too_big_to_fold_prints_one_unmirrored_copy():
    """220x200 cannot go twice on A3 either way, but once fits A3 portrait."""
    poly = Polygon([(0, 0), (220, 0), (220, 200), (60, 200), (0, 140)])  # asymmetric
    info = {}
    pdf = velostat_fold_pdf(poly, info=info)
    assert pdf is not None
    assert info["fold"] is None and info["sheet"] == "A3 portrait"
    loops = _polylines(pdf)
    assert len(loops) == 1, "one copy, no mirror"
    size = loops[0].max(axis=0) - loops[0].min(axis=0)
    assert abs(size[0] - 220) < 0.01 and abs(size[1] - 200) < 0.01
    # Same orientation as the board, not flipped: the chamfer stays top-left.
    x0, y0 = loops[0].min(axis=0)
    assert any(abs(x - x0) < 0.01 and abs(y - (y0 + 140)) < 0.01 for x, y in loops[0])
    assert b"FOLD" not in pdf


def test_folding_on_a_bigger_sheet_beats_a_single_copy_on_a_smaller_one():
    """150x150 fits once on A4 but twice only on A3 (stacked, A3 portrait) -- folding wins."""
    info = {}
    pdf = velostat_fold_pdf(Polygon([(0, 0), (150, 0), (150, 150), (0, 150)]), info=info)
    assert info["fold"] == "horizontal" and info["sheet"] == "A3 portrait"
    assert len(_polylines(pdf)) == 2
