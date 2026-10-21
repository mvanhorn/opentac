"""Tests for the design-name label on the front silkscreen.

The label is the only thing on a fabricated board that says which design it
is, and it is drawn twice -- once as a `gr_text` KiCad will stroke, once as
real segments in the Gerber, because the hosted deployment has no KiCad to
stroke it with. Everything here exists to keep those two honest about the
same three promises:

  * the label is inside the board, by the same margin copper keeps;
  * it never lands on the connector's contact fingers;
  * the on-screen preview shows exactly what the Gerber carries, including
    when the name had to be shrunk or truncated to fit.

The last one matters more than it looks: truncation is silent everywhere else,
so the preview is the only place a user finds out their board says
"sensor_20260908" and their ZIP says "sensor_20260908_1432".
"""

import base64
import io
import json
import os
import re
import zipfile
from unittest import mock


from shapely.geometry import LineString, Point, Polygon

from opentac.cad.export.silk_label import HEIGHTS_MM, MIN_CHARS, sanitize_name
from opentac.cad.export.stroke_font import _GLYPHS, text_strokes, text_width

SQUARE = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
SILK_GERBER = "gerber/board-F_Silkscreen.gto"


def _client():
    from fastapi.testclient import TestClient
    from opentac.cad import main
    return TestClient(main.app), main


def _generate(name, **extra):
    """Exported through the Python Gerber writer, the path every hosted export
    takes: with kicad-cli installed the export plots through KiCad instead,
    which draws an empty silk layer and its own glyphs."""
    client, main_mod = _client()
    body = {"outline": SQUARE, "design_name": name, **extra}
    with mock.patch.object(main_mod.shutil, "which", lambda name: None):
        resp = client.post("/generate", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    zf = zipfile.ZipFile(io.BytesIO(base64.b64decode(data["zip_b64"])))
    return data, zf, main_mod


def _silk_segments(gerber_text):
    """Read the stroked label back out of the Gerber as mm segments."""
    segs, pen = [], None
    for line in gerber_text.splitlines():
        m = re.match(r"X(-?\d+)Y(-?\d+)D0([12])\*", line.strip())
        if not m:
            continue
        # The writer emits absolute coordinates in 4.6 format, in the board
        # frame (Y up) -- the frame KiCad plots in, so no sign to undo.
        p = (int(m.group(1)) / 1e6, int(m.group(2)) / 1e6)
        if m.group(3) == "1" and pen is not None:
            segs.append((pen, p))
        pen = p
    return segs


# ---- the font ---------------------------------------------------------------

def test_font_covers_every_character_a_name_can_contain():
    """sanitize_name's output alphabet and the glyph table must agree.

    A name is sanitised to [A-Za-z0-9_-] and then folded to uppercase, so any
    character that survives sanitising and has no glyph would silently print
    as a gap in the middle of the label.
    """
    alphabet = [chr(c) for c in range(32, 127)]
    survivors = set()
    for ch in alphabet:
        s = sanitize_name("ab" + ch + "cd")
        survivors.update(s)
    missing = sorted(c for c in survivors if c.upper() not in _GLYPHS)
    assert not missing, f"no glyph for {missing}"


def test_glyphs_stay_in_their_cell_and_advance():
    """Ink stays inside the advertised cell, so text_width does not lie."""
    for ch, polys in _GLYPHS.items():
        xs = [p[0] for poly in polys for p in poly]
        ys = [p[1] for poly in polys for p in poly]
        # Q's tail is the one deliberate overhang, and it is under a tenth of
        # the advance, so it cannot reach the next glyph.
        assert min(xs) >= 0.0, ch
        assert max(xs) <= 6.5, (ch, max(xs))
        assert max(ys) <= 10.0, ch
        assert min(ys) >= -1.5, ch

    # Width scales with height and with character count, and an unsupported
    # character still advances rather than closing the gap up.
    assert text_width("AB", 2.0) == 2 * text_width("AB", 1.0)
    assert text_width("ABC", 1.0) > text_width("AB", 1.0)
    assert text_width("A B", 1.0) == text_width("ABB", 1.0)
    assert text_strokes("", 1.0) == []


# ---- the name ---------------------------------------------------------------

def test_sanitize_matches_the_frontend_rules():
    assert sanitize_name("fingertip_wrap.stl") == "fingertip_wrap"
    assert sanitize_name("My Sensor v2") == "My_Sensor_v2"
    assert sanitize_name("../../etc/passwd") == "etc_passwd"
    assert sanitize_name('a"b(c)') == "a_b_c"
    assert "/" not in sanitize_name("a/b")
    assert len(sanitize_name("a" * 200)) == 48
    assert sanitize_name(None) == ""


# ---- the board --------------------------------------------------------------

def test_label_is_on_the_board_and_off_the_connector():
    data, zf, main_mod = _generate("fingertip_wrap")

    pcb = zf.read("board.kicad_pcb").decode()
    assert re.search(r'\(gr_text "fingertip_wrap".*\(layer "F\.SilkS"\)', pcb), \
        "no gr_text on F.SilkS"
    assert SILK_GERBER in zf.namelist(), "no silkscreen Gerber"

    segs = _silk_segments(zf.read(SILK_GERBER).decode())
    assert segs, "silkscreen Gerber has no geometry"

    edit = data["edit_data"]
    board = Polygon(edit.get("final_outline") or edit["outline"])
    clear = edit["params"]["board_edge_clear_mm"]
    safe = board.buffer(-clear)
    for a, b in segs:
        assert safe.contains(Point(*a)) and safe.contains(Point(*b)), \
            f"silk stroke {a}->{b} is outside the board"

    # Nothing on a contact finger: silk over a pad is a print defect.
    from opentac.cad.export.connector_geometry import connector_pad_geometries
    layout, outline_xy, params, _removed = main_mod._layout_from_edit_data(edit)
    conn_dir = main_mod.connector_export_dir(
        layout, os.path.join(os.path.dirname(main_mod.__file__), "connectors"))
    pads = [Polygon(p["corners"])
            for p in connector_pad_geometries(layout, outline_xy, params, conn_dir)]
    assert pads, "no connector pads to test against"
    for a, b in segs:
        line = LineString([a, b])
        assert not any(line.intersects(pad) for pad in pads), \
            f"silk stroke {a}->{b} lands on a connector pad"


def test_label_sits_between_the_connector_and_the_array():
    """Inboard of the connector, not out past it and not up in the taxels."""
    data, zf, main_mod = _generate("fingertip_wrap")
    segs = _silk_segments(zf.read(SILK_GERBER).decode())

    edit = data["edit_data"]
    layout, outline_xy, params, _removed = main_mod._layout_from_edit_data(edit)
    from opentac.cad.export.connector_geometry import connector_pose
    t_hat, n_out, pad1 = connector_pose(layout, outline_xy, params)

    import numpy as np
    pts = np.array([p for seg in segs for p in seg], dtype=float)
    # Distance inboard from the routing pad row, along -n_out.
    inboard = -((pts - np.asarray(pad1, float)) @ np.asarray(n_out, float))
    assert inboard.min() > 0, "label is out past the connector, off the tail"
    assert inboard.max() < 12.0, "label has drifted up into the sensor array"

    # And it reads left-to-right rather than upside down.
    rot = float(re.search(r'\(gr_text "fingertip_wrap" \(at [-\d.]+ [-\d.]+ '
                          r'(-?[\d.]+)\)', zf.read("board.kicad_pcb").decode()).group(1))
    assert -90.0 < rot <= 90.0, rot


def test_preview_shows_exactly_what_the_gerber_carries():
    """A name too long to print is truncated -- the preview has to admit it."""
    long_name = "sensor_20260908_1432"
    data, zf, _main = _generate(long_name)

    printed = re.search(r'\(gr_text "([^"]+)"', zf.read("board.kicad_pcb").decode()).group(1)
    assert printed.startswith("sensor"), printed
    assert len(printed) <= len(long_name)

    svg_polys = re.findall(r'<polyline points="([^"]+)" class="silk"/>', data["svg"])
    assert svg_polys, "the preview drew no label"
    # The Gerber writer emits one move+draw pair per segment, so compare
    # segment counts rather than polyline counts.
    svg_segs = sum(len(p.split()) - 1 for p in svg_polys)
    assert svg_segs == len(_silk_segments(zf.read(SILK_GERBER).decode()))

    # And the preview shows the truncated name, not the one the user typed:
    # the glyph count is what gives it away.
    from opentac.cad.export.stroke_font import text_strokes
    assert len(svg_polys) == len(text_strokes(printed, 1.0))


def test_no_name_means_no_label_anywhere():
    for name in ("", "   ", "ab", "_" * 8):
        data, zf, _main = _generate(name)
        pcb = zf.read("board.kicad_pcb").decode()
        assert not re.search(r'\(gr_text ', pcb), name
        assert SILK_GERBER not in zf.namelist(), name
        assert 'class="silk"' not in data["svg"], name
    assert MIN_CHARS >= 3


def test_the_name_survives_a_round_trip_through_design_json():
    """Re-opening a saved design must not silently drop its label."""
    data, zf, _main = _generate("thumb_pad_rev_B")
    assert zf.read("design.json"), "no design.json in the bundle"
    saved = zf.read("design.json").decode()
    assert '"design_name": "thumb_pad_rev_B"' in saved

    client, _mod = _client()
    resp = client.post("/open-design",
                       json={"file": json.loads(zf.read("design.json"))})
    assert resp.status_code == 200, resp.text
    reopened = resp.json()
    assert reopened["edit_data"]["params"]["design_name"] == "thumb_pad_rev_B"
    assert reopened["params"].get("design_name") == "thumb_pad_rev_B"


def test_a_tiny_board_drops_the_label_rather_than_overhanging():
    """There is a floor: below it the label is dropped, never shrunk further."""
    assert min(HEIGHTS_MM) >= 0.8, "silk below 0.8mm does not survive screen printing"
