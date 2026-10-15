"""The back silkscreen has to describe the board that was actually exported.

Three things can go wrong here and none of them are visible by looking at a
board file:

  1. The QR stops describing the board. It is derived from manifest.json, so a
     manifest change that the codec does not follow produces a mark that
     decodes cleanly to the WRONG sensor -- worse than no mark at all. The
     round-trip below rebuilds every taxel position out of the payload and
     compares it against the manifest it came from.

  2. Silk lands over the board edge. Clipping the line CENTRELINE to the
     outline and then widening it puts the rect corners over the cut line
     wherever the edge is sloped or curved; on a 40 mm square that was 19 of 21
     grid rects. The fab clips it and nobody finds out until the boards arrive.

  3. The mark takes the whole export down with it. A 22.4 mm pitch used to
     raise out of the encoder (the pitch field was 10 bits) and turned into a
     400 on /generate -- a legend costing the user their Gerbers. main.py takes
     the same stance for the print PDF, and for the same reason.
"""
import json
import math
import os
import sys

import pytest
from shapely.geometry import Polygon, box


from opentac.cad.export.silk_backmark import (
    DIM_CELL_MARGIN_MM, DIM_STROKE_W_MM, DIM_TEXT_H_MM, EDGE_CLEAR_MM,
    GRID_MINOR_MM, MODULE_MM, PRECOMP_MM, backmark)
from opentac.cad.export.taxel_core import (
    ABSENT, B41, TaxelCoreError, b41decode, decode, manifest_from_scan, taxels)
from opentac.cad.main import generate, GenerateRequest

SQUARE = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
U_NOTCH = [[-34, 0], [-12, 0], [-12, 34], [12, 34], [12, 0], [34, 0], [34, 56], [-34, 56]]

CASES = [
    ("square/expand", SQUARE, "expand", 2.0, 2.2),
    ("square/fixed", SQUARE, "fixed_keepout", 2.0, 2.2),
    ("u-notch/expand", U_NOTCH, "expand", 4.0, 4.2),
    # 22.4 mm pitch: the geometry that used to raise out of the bit-packer.
    ("coarse pitch", [[-30, -30], [30, -30], [30, 30], [-30, 30]], "expand", 22.0, 22.4),
]


def _manifest(outline, mode, pw, pitch):
    resp = generate(GenerateRequest(
        outline=outline, router="hug", board_mode=mode,
        pixel_w_mm=pw, pixel_h_mm=pw, pitch_x_mm=pitch, pitch_y_mm=pitch))
    import base64
    import io
    import zipfile
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64)))
    return z.read("manifest.json").decode(), z


@pytest.mark.parametrize("tag,outline,mode,pw,pitch", CASES)
def test_qr_rebuilds_the_manifest_it_came_from(tag, outline, mode, pw, pitch):
    manifest_json, _ = _manifest(outline, mode, pw, pitch)
    mark = backmark(manifest_json)
    assert not mark["warnings"], "%s: %s" % (tag, mark["warnings"])
    assert mark["qr"], "%s: no QR produced" % tag

    core = decode(b41decode(mark["qr"]["payload"]))
    man = json.loads(manifest_json)

    rebuilt = {(r, c): (x, y) for x, y, r, c, _, _ in taxels(core)}
    ref = {(r, c): (x, y) for x, y, r, c in man["pixels"]}
    assert set(rebuilt) == set(ref), "%s: taxel set differs from the manifest" % tag

    worst = max(max(abs(rebuilt[k][0] - ref[k][0]), abs(rebuilt[k][1] - ref[k][1]))
                for k in ref)
    assert worst <= 0.01, "%s: taxel positions off by %.4f mm" % (tag, worst)

    # the pin map is the other half of what a readout needs
    assert core["rp"] == [p for p in man["rp"]], "%s: row pads differ" % tag
    assert core["cp"] == [p for p in man["cp"]], "%s: col pads differ" % tag


@pytest.mark.parametrize("tag,outline,mode,pw,pitch", CASES)
def test_the_pitch_label_sits_inside_one_grid_cell(tag, outline, mode, pw, pitch):
    """The grid lines are the dimension; the label only names it.

    That only works if the text sits INSIDE a cell -- a number floating across a
    line does not say which spacing it belongs to, and on a board with two line
    weights that is the whole question. Sitting inside one cell also means the
    label can never cross a grid line, so no keep-out is needed.
    """
    manifest_json, _ = _manifest(outline, mode, pw, pitch)
    mark = backmark(manifest_json)
    assert mark["strokes"], "%s: no pitch label" % tag

    pts = [p for poly in mark["strokes"] for p in poly]
    pad = DIM_STROKE_W_MM / 2.0
    x0, x1 = min(p[0] for p in pts) - pad, max(p[0] for p in pts) + pad
    y0, y1 = min(p[1] for p in pts) - pad, max(p[1] for p in pts) + pad

    # one cell, on the same lattice the grid is drawn on
    ci, cj = math.floor(x0 / GRID_MINOR_MM), math.floor(y0 / GRID_MINOR_MM)
    cell = (ci * GRID_MINOR_MM, cj * GRID_MINOR_MM,
            (ci + 1) * GRID_MINOR_MM, (cj + 1) * GRID_MINOR_MM)
    assert cell[0] <= x0 and x1 <= cell[2] and cell[1] <= y0 and y1 <= cell[3], \
        "%s: label spans x %.2f..%.2f y %.2f..%.2f, crossing cell %r" \
        % (tag, x0, x1, y0, y1, cell)

    # and clear of the walls, so print spread cannot merge it into a line
    clear = min(x0 - cell[0], cell[2] - x1, y0 - cell[1], cell[3] - y1)
    assert clear >= DIM_CELL_MARGIN_MM - pad - 1e-9, \
        "%s: label is only %.3f mm from a cell wall" % (tag, clear)


def test_the_pitch_label_is_readable_after_fabrication():
    """Text is chiral, so it is mirrored like the QR and unlike the grid.

    Drawn unmirrored it would look right in KiCad's top view and read backwards
    on the board, which is the only side anyone reads it from.
    """
    manifest_json, _ = _manifest(SQUARE, "expand", 2.0, 2.2)
    mark = backmark(manifest_json)
    got = [p for poly in mark["strokes"] for p in poly]
    assert got, "no label strokes"

    from opentac.cad.export.stroke_font import text_strokes
    ref = [p for poly in text_strokes("%dMM" % round(GRID_MINOR_MM), DIM_TEXT_H_MM)
           for p in poly]
    assert len(got) == len(ref)

    ref_w = max(p[0] for p in ref) - min(p[0] for p in ref)
    got_w = max(p[0] for p in got) - min(p[0] for p in got)
    assert abs(ref_w - got_w) < 1e-6, "glyph run width changed under mirroring"

    first_ref = ref[0][0] - min(p[0] for p in ref)
    first_got = got[0][0] - min(p[0] for p in got)
    assert abs(first_got - (ref_w - first_ref)) < 1e-6, \
        "pitch text is not mirrored; it would read backwards on the board"


@pytest.mark.parametrize("tag,outline,mode,pw,pitch", CASES)
def test_nothing_is_printed_over_the_board_edge(tag, outline, mode, pw, pitch):
    manifest_json, _ = _manifest(outline, mode, pw, pitch)
    mark = backmark(manifest_json)
    man = json.loads(manifest_json)

    poly = Polygon(man["edge_cuts"])
    if not poly.is_valid:
        poly = poly.buffer(0)
    inner = poly.buffer(-EDGE_CLEAR_MM)

    outside = [r for r in mark["rects"] if not inner.buffer(1e-9).contains(box(*r))]
    assert not outside, (
        "%s: %d of %d back-silk rects come within %.2f mm of the cut line, worst %r"
        % (tag, len(outside), len(mark["rects"]), EDGE_CLEAR_MM, outside[0]))

    # the stroked label too, allowing for its pen width
    pad = DIM_STROKE_W_MM / 2.0
    for poly in mark["strokes"]:
        for x, y in poly:
            assert inner.buffer(1e-9).contains(box(x - pad, y - pad, x + pad, y + pad)), \
                "%s: dimension label runs within %.2f mm of the cut line" % (tag, EDGE_CLEAR_MM)


@pytest.mark.parametrize("tag,outline,mode,pw,pitch", CASES)
def test_the_gerber_and_the_kicad_file_agree(tag, outline, mode, pw, pitch):
    """Both exporters read the same geometry, so both must carry the same count.

    kicad-cli plots B.Silkscreen from the board file when it is installed and
    the pure-Python path draws it directly when it is not. Those are different
    code paths to the same artwork, and a board that only carries the mark on
    one of them is the failure this catches.
    """
    manifest_json, z = _manifest(outline, mode, pw, pitch)
    mark = backmark(manifest_json)
    n_rects = len(mark["rects"])
    n_seg = sum(len(poly) - 1 for poly in mark["strokes"])
    assert n_rects > 0 and n_seg > 0

    pcb = z.read("board.kicad_pcb").decode()
    # every back-silk primitive: filled rects for the QR, grid and origin,
    # plus one stroked segment per polyline in the dimension label
    assert pcb.count('(layer "B.SilkS")') == n_rects + n_seg, \
        "%s: kicad_pcb carries %d back-silk primitives, artwork has %d rects + %d segments" \
        % (tag, pcb.count('(layer "B.SilkS")'), n_rects, n_seg)

    names = z.namelist()
    assert "gerber/board-B_Silkscreen.gbo" in names, \
        "%s: no back-silkscreen Gerber in the export" % tag
    gbo = z.read("gerber/board-B_Silkscreen.gbo").decode()
    assert gbo.count("G36*") == n_rects, \
        "%s: Gerber region count does not match the artwork" % tag

    # the Gerber carries the strokes as aperture draws, not regions
    assert gbo.count("D01*") >= n_seg, \
        "%s: Gerber is missing dimension-label segments" % tag


def test_a_broken_mark_never_costs_the_board():
    """backmark() degrades. It does not raise, whatever it is handed."""
    for junk in (None, "", "not json", "{}", '{"m": []}',
                 json.dumps({"m": ["11"], "pixels": [], "r": 1, "c": 2}),
                 json.dumps({"m": ["1"], "pixels": [[0, 0, 0, 0]], "r": 99, "c": 99,
                             "w": 1, "h": 1, "rp": [1], "cp": [2]})):
        out = backmark(junk)
        assert out["rects"] == [] or isinstance(out["rects"], list)
        assert "warnings" in out


def test_artwork_clears_the_fab_silkscreen_limits():
    """JLCPCB flex: line >= 0.15 mm, hollow internal gap > 0.25 mm.

    A knocked-out QR is a hollow feature, so the knockout is the binding rule.
    Lines under the minimum are silently WIDENED by the fab, which would undo
    the pre-compensation without any warning -- so the ink has to clear it by
    design, not by luck.
    """
    thinnest_ink = MODULE_MM - 2 * PRECOMP_MM
    smallest_knockout = MODULE_MM + 2 * PRECOMP_MM
    assert thinnest_ink >= 0.15, \
        "ink feature %.3f mm is under the fab minimum and would be auto-widened" % thinnest_ink
    assert smallest_knockout > 0.25, \
        "knockout %.3f mm is under the hollow-font gap limit" % smallest_knockout


# --------------------------------------------------------------------------
# the other end: a scanned string back to a manifest
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tag,outline,mode,pw,pitch", CASES)
def test_a_scan_rebuilds_a_usable_manifest(tag, outline, mode, pw, pitch):
    """What someone gets after pointing a phone at the board.

    Manifest-shaped on purpose: it has to drop into whatever already reads
    manifest.json, not into a second format nobody consumes.
    """
    manifest_json, _ = _manifest(outline, mode, pw, pitch)
    orig = json.loads(manifest_json)
    rec = manifest_from_scan(backmark(manifest_json)["qr"]["payload"])

    for key in ("v", "units", "r", "c", "footprint", "rp", "cp", "m"):
        assert rec[key] == orig[key], "%s: %s differs after a scan" % (tag, key)
    for key in ("w", "h", "px", "py", "ox", "oy"):
        assert abs(rec[key] - orig[key]) <= 0.01, \
            "%s: %s off by %.4f" % (tag, key, abs(rec[key] - orig[key]))

    a = {(r, c): (x, y) for x, y, r, c in orig["pixels"]}
    b = {(r, c): (x, y) for x, y, r, c in rec["pixels"]}
    assert set(a) == set(b), "%s: taxel set differs after a scan" % tag
    worst = max(max(abs(b[k][0] - a[k][0]), abs(b[k][1] - a[k][1])) for k in a)
    assert worst <= 0.01, "%s: taxels off by %.4f mm after a scan" % (tag, worst)

    # what it cannot carry is named, not faked
    for key in ABSENT:
        assert key not in rec, "%s: %s claims to be recovered but cannot be" % (tag, key)
    assert rec["recovered"]["absent"] == list(ABSENT)


QR_ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

# The four QR-alphanumeric characters that get mangled SILENTLY -- each turns a
# payload into a different valid payload rather than an error.
SILENT_HAZARDS = {
    " ": "trimmed by scanners and clipboards",
    "%": "eaten by URL percent-decoding",
    "+": "URL-decoded back into a space",
    "*": "expanded by a shell if the CLI argument is unquoted",
}


def test_the_alphabet_excludes_every_silently_mangled_character():
    """base41, not RFC 9285's base45, and the four dropped characters are the point.

    Density is set by needing N**3 >= 65536 to pack two bytes into three
    characters, which holds down to 41. QR's alphanumeric set has 45, so there
    are exactly four characters of headroom -- and exactly four that fail
    silently. Dropping them is free.
    """
    assert len(B41) == 41 and len(set(B41)) == 41
    assert set(B41) <= set(QR_ALNUM), "alphabet leaves QR alphanumeric mode"

    for ch, why in SILENT_HAZARDS.items():
        assert ch not in B41, "%r is still in the alphabet (%s)" % (ch, why)

    # and the density really is unchanged
    assert len(B41) ** 3 >= 65536 and len(B41) ** 2 >= 256


def test_payloads_survive_the_transports_that_used_to_corrupt_them():
    """Whitespace trimming, URL round-tripping and shell globbing are all no-ops now."""
    import glob
    import urllib.parse

    for tag, outline, mode, pw, pitch in CASES:
        manifest_json, _ = _manifest(outline, mode, pw, pitch)
        payload = backmark(manifest_json)["qr"]["payload"]
        reference = manifest_from_scan(payload)

        assert not (set(payload) & set(SILENT_HAZARDS)), \
            "%s: payload contains a silently-mangled character" % tag

        # a scanner or clipboard that trims
        assert manifest_from_scan("  \n" + payload + "\t\r\n ") == reference, \
            "%s: trimming changed the result" % tag
        assert manifest_from_scan("".join(payload.split())) == reference, \
            "%s: internal whitespace strip changed the result" % tag

        # a URL query round-trip, both quoting styles
        assert urllib.parse.unquote_plus(payload) == payload, \
            "%s: payload is altered by URL decoding" % tag
        assert urllib.parse.parse_qs("p=" + payload)["p"][0] == payload, \
            "%s: payload does not survive a query string" % tag

        # an unquoted shell argument: no glob metacharacters at all
        assert glob.escape(payload) == payload, \
            "%s: payload contains shell glob metacharacters" % tag


def test_a_mangled_scan_fails_loudly():
    """Every single-character substitution must be rejected.

    A payload that decodes to a plausible-but-wrong pin mapping is worse than
    one that does not decode: someone wires a sensor up from it.
    """
    import random

    manifest_json, _ = _manifest(SQUARE, "expand", 2.0, 2.2)
    payload = backmark(manifest_json)["qr"]["payload"]
    rng = random.Random(0)

    for _ in range(200):
        i = rng.randrange(len(payload))
        ch = rng.choice([c for c in B41 if c != payload[i]])
        with pytest.raises(TaxelCoreError):
            manifest_from_scan(payload[:i] + ch + payload[i + 1:])
