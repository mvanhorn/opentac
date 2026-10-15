"""The 1:1 printable cutting template.

The single property that matters is scale: a PDF that renders beautifully but
prints at 97% produces a cut-out that is wrong in a way nothing on screen
reveals. These pin the scale, the on-page scale check that lets a user catch a
scaling print dialog, and the tiling path for boards too big for one sheet.
"""
import base64
import io
import os
import re
import sys
import zipfile

import pytest


from opentac.cad.main import (GenerateRequest, _layout_from_edit_data,
                  connector_export_dir, generate)
from opentac.cad.export.print_pdf import MM_TO_PT, RULER_MM, generate_print_pdf

SQUARE = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
HUGE = [[-170, -210], [170, -210], [170, 210], [-170, 210]]


def _pdf_for(outline, **kw):
    body = {"outline": outline, "router": "hug", "board_mode": "expand",
            "pixel_w_mm": 4, "pixel_h_mm": 4, "pitch_x_mm": 4.2, "pitch_y_mm": 4.2}
    body.update(kw)
    edit_data = generate(GenerateRequest(**body)).edit_data
    layout, outline_xy, params, removed = _layout_from_edit_data(edit_data)
    cdir = connector_export_dir(
        layout, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "connectors"))
    return generate_print_pdf(layout, outline_xy, params, removed, cdir), edit_data


def _page_streams(raw):
    return re.findall(rb"stream\n(.*?)\nendstream", raw, re.S)


def test_pdf_is_structurally_valid():
    """Every xref offset must land exactly on its object header. Get this wrong
    and the file opens in some readers and not others."""
    raw, _ = _pdf_for(SQUARE)
    assert raw.startswith(b"%PDF-1.4")
    assert raw.rstrip().endswith(b"%%EOF")
    xref_at = int(re.search(rb"startxref\s+(\d+)", raw).group(1))
    assert raw[xref_at:xref_at + 4] == b"xref"
    body = raw[xref_at:].split(b"trailer")[0].decode("latin-1")
    rows = re.findall(r"(\d{10}) (\d{5}) ([nf])", body)
    assert len(rows) >= 5
    for i, (off, _gen, kind) in enumerate(rows):
        if kind == "n":
            assert raw[int(off):].startswith(f"{i} 0 obj".encode()), f"xref[{i}] misaligned"


def test_geometry_is_drawn_at_true_1_to_1():
    """1 mm of board must be exactly 72/25.4 PDF units. This is the whole point
    of the file, and it is one wrong constant away from being silently false."""
    raw, _ = _pdf_for(SQUARE)
    txt = _page_streams(raw)[0].decode("latin-1")
    lines = [tuple(map(float, m)) for m in
             re.findall(r"([\d.]+) ([\d.]+) m ([\d.]+) ([\d.]+) l S", txt)]
    horizontal = [(x2 - x1) / MM_TO_PT for x1, y1, x2, y2 in lines if abs(y2 - y1) < 1e-6]
    assert any(abs(h - RULER_MM) < 0.01 for h in horizontal), \
        f"no {RULER_MM}mm reference line found; got {sorted(set(round(h,2) for h in horizontal))}"


def test_every_page_carries_the_scale_check():
    """Including tiled sheets. The ruler was originally drawn in board space,
    which put it off the media box entirely on 2 of 4 tiles -- removing the only
    way a user can tell their printer scaled the sheet."""
    raw, _ = _pdf_for(HUGE, pixel_w_mm=22, pixel_h_mm=22, pitch_x_mm=22.4, pitch_y_mm=22.4)
    streams = _page_streams(raw)
    assert len(streams) > 1, "this board should have tiled"
    for i, st in enumerate(streams):
        txt = st.decode("latin-1")
        lines = [tuple(map(float, m)) for m in
                 re.findall(r"([\d.]+) ([\d.]+) m ([\d.]+) ([\d.]+) l S", txt)]
        horizontal = [(x2 - x1) / MM_TO_PT for x1, y1, x2, y2 in lines if abs(y2 - y1) < 1e-6]
        assert any(abs(h - RULER_MM) < 0.01 for h in horizontal), f"page {i + 1} has no ruler"


def test_small_board_fits_one_sheet():
    raw, _ = _pdf_for(SQUARE)
    assert len(_page_streams(raw)) == 1
    # A4 portrait, in PDF units
    assert f"{210.0 * MM_TO_PT:.3f}".encode() in raw


def test_pdf_is_in_the_export_zip():
    body = {"outline": SQUARE, "router": "hug", "board_mode": "expand",
            "pixel_w_mm": 4, "pixel_h_mm": 4, "pitch_x_mm": 4.2, "pitch_y_mm": 4.2}
    resp = generate(GenerateRequest(**body))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64))) as zf:
        names = zf.namelist()
        assert "printable_1to1.pdf" in names
        assert zf.read("printable_1to1.pdf").startswith(b"%PDF")


def test_the_cut_outline_is_closed():
    """Edge.Cuts deliberately leaves a gap where the connector footprint
    supplies its own edge. Correct for a board file; useless on paper, where an
    open loop is a shape you cannot cut out."""
    raw, _ = _pdf_for(SQUARE)
    st = _page_streams(raw)[0].decode("latin-1")
    subpaths = re.findall(r"(?:-?[\d.]+ -?[\d.]+ m)(?:\s+-?[\d.]+ -?[\d.]+ l)+\s+(h S|S)", st)
    assert subpaths, "no polyline drawn"
    assert any(k == "h S" for k in subpaths), "the board outline was left open"


def test_the_cut_outline_includes_the_connector():
    """The template is the board the fab cuts, so it must span the same extent
    as the Gerber Edge.Cuts in the same export. It used to stop at the
    connector's open edge, where the footprint draws the rest of the outline,
    and printed every board short by the connector body."""
    body = {"outline": SQUARE, "router": "hug", "board_mode": "expand",
            "pixel_w_mm": 4, "pixel_h_mm": 4, "pitch_x_mm": 4.2, "pitch_y_mm": 4.2}
    resp = generate(GenerateRequest(**body))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64))) as zf:
        pdf = zf.read("printable_1to1.pdf").decode("latin-1")
        gm1 = zf.read("gerber/board-Edge_Cuts.gm1").decode()
    pdf_w, pdf_h = map(float, re.search(r"Board ([\d.]+) x ([\d.]+) mm", pdf).groups())
    # Every endpoint, arcs included (KiCad writes the connector's corners as G02/G03).
    xy = [(int(x) / 1e6, int(y) / 1e6) for x, y in re.findall(r"X(-?\d+)Y(-?\d+)", gm1)]
    w = max(p[0] for p in xy) - min(p[0] for p in xy)
    h = max(p[1] for p in xy) - min(p[1] for p in xy)
    assert pdf_w == pytest.approx(w, abs=0.1)
    assert pdf_h == pytest.approx(h, abs=0.1)


def test_a_broken_pdf_never_costs_the_user_their_gerbers():
    """The PDF is a convenience; the Gerbers are the deliverable. Export must
    survive a PDF failure rather than returning a 500 for the whole board."""
    from opentac.cad import main as main_mod
    original = main_mod.generate_print_pdf
    main_mod.generate_print_pdf = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        resp = generate(GenerateRequest(
            outline=SQUARE, router="hug", board_mode="expand",
            pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2))
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64))) as zf:
            names = zf.namelist()
            assert "board.kicad_pcb" in names
            assert "gerber/board.drl" in names
            assert "printable_1to1.pdf" not in names
    finally:
        main_mod.generate_print_pdf = original
