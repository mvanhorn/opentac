"""The manifest must describe the board that is actually manufactured.

manifest.json is the one file a user is expected to keep: it has to carry the
taxel placements, the taxel copper geometry, and the real cut boundary, so the
same sensor can be rebuilt from it alone.

The trap this pins down: `outline` is the shape the user DREW, which is not
the board that gets made. The manufactured edge adds the grown board, the
cable tail and the connector envelope (a drawn 40x40mm square exports cuts of
46.6 x 62.8mm), so `edge_cuts` is a separate field taken from the exporter's
own stitching call rather than re-derived here.
"""
import base64, io, json, re, sys, os

import numpy as np
import pytest
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from opentac.cad.main import generate, GenerateRequest
from opentac.cad.export.coverlay import taxel_opening
from opentac.cad.export.kicad_footprint import build_pixel_footprint_text, make_footprint_name
from opentac.cad.pipeline.bump_sheet import bump_centers_from_taxels

SQUARE = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
U_NOTCH = [[-34, 0], [-12, 0], [-12, 34], [12, 34], [12, 0], [34, 0], [34, 56], [-34, 56]]
LSHAPE = [[0, 0], [50, 0], [50, 20], [25, 20], [25, 40], [0, 40]]
SQ_45x38 = [[0, 0], [45, 0], [45, 38], [0, 38]]


def _board_edge_points(pcb_text):
    """Board-outline Edge.Cuts only (top-level gr_line). The connector
    footprint carries its own Edge.Cuts as fp_line, which is not part of the
    board ring."""
    pts, seen = [], set()
    for m in re.finditer(
            r'\(gr_line\s*\(start ([-\d.]+) ([-\d.]+)\)\s*\(end ([-\d.]+) ([-\d.]+)\)', pcb_text):
        if 'Edge.Cuts' in pcb_text[m.start():m.start() + 300]:
            # every vertex, ends included: the ring is open at the connector,
            # so one corner is only ever a segment's end
            for x, y in ((m.group(1), m.group(2)), (m.group(3), m.group(4))):
                key = (round(float(x), 3), round(float(y), 3))
                if key not in seen:
                    seen.add(key)
                    pts.append((float(x), float(y)))
    return np.array(pts)


def test_manifest_carries_the_real_board():
    for outline, mode in ((SQUARE, "expand"), (SQUARE, "fixed_keepout"),
                          (U_NOTCH, "expand"), (U_NOTCH, "fixed_keepout")):
        resp = generate(GenerateRequest(
            outline=outline, router="hug", board_mode=mode,
            pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2))
        z = zipfile = __import__("zipfile").ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64)))
        man = json.loads(z.read("manifest.json"))
        tag = f"{len(outline)}-gon/{mode}"

        ec = np.asarray(man.get("edge_cuts") or [], dtype=float)
        assert len(ec) >= 3, f"{tag}: manifest has no edge_cuts"

        # 1. edge_cuts IS the exported board outline (KiCad negates Y). A 1um
        #    tolerance, since KiCad writes 3 decimals and the manifest 4.
        kp = _board_edge_points(z.read("board.kicad_pcb").decode())
        mine = np.column_stack([ec[:, 0], -ec[:, 1]])
        assert len(kp) == len(mine), \
            f"{tag}: {len(mine)} manifest cut points vs {len(kp)} exported"
        d1, _ = cKDTree(kp).query(mine)
        d2, _ = cKDTree(mine).query(kp)
        worst = max(d1.max(), d2.max())
        assert worst < 0.001, f"{tag}: edge_cuts differ from the export by {worst:.4f}mm"

        # 1b. the ring closes straight across the connector's open edge, which
        #     is horizontal in the manifest frame: the closing edge is the only
        #     one that is not an exported segment, and it is not a diagonal
        #     chord across the tail (a corner used to be dropped there)
        segs = [LineString([(float(m.group(1)), -float(m.group(2))), (float(m.group(3)), -float(m.group(4)))])
                for m in re.finditer(r'\(gr_line\s*\(start ([-\d.]+) ([-\d.]+)\)\s*\(end ([-\d.]+) ([-\d.]+)\)',
                                     z.read("board.kicad_pcb").decode())]
        board = MultiLineString(segs)
        extra = [(ec[i], ec[(i + 1) % len(ec)]) for i in range(len(ec))
                 if board.distance(Point(*((ec[i] + ec[(i + 1) % len(ec)]) / 2))) > 0.002]
        assert len(extra) <= 1, f"{tag}: {len(extra)} manifest edges are not board edges"
        for a, b in extra:
            assert abs(a[1] - b[1]) < 0.002, f"{tag}: edge_cuts closes diagonally {a} -> {b}"

        # 2. the footprint block rebuilds the exported .kicad_mod exactly (its
        #    mask opening follows from the taxel size and the pitch)
        fp = man.get("footprint")
        assert fp, f"{tag}: manifest has no footprint parameters"
        mod_name = [n for n in z.namelist() if n.endswith(".kicad_mod")][0]
        rebuilt, _ = build_pixel_footprint_text(
            make_footprint_name("tactile_pixel", man["w"], man["h"], fp["trace_w"],
                                fp["gap"], fp["center_clear"], fp["edge_clear"],
                                fp["via_drill"], fp["via_dia"]),
            man["w"], man["h"], fp["trace_w"], fp["gap"], fp["center_clear"],
            fp["edge_clear"], fp["via_drill"], fp["via_dia"],
            opening=taxel_opening(man["w"], man["h"], fp["edge_clear"], man["px"], man["py"]))
        assert rebuilt.strip() == z.read(mod_name).decode().strip(), \
            f"{tag}: footprint params do not rebuild the exported footprint"

        # 3. every taxel sits inside the cut boundary
        poly = Polygon(ec).buffer(0.05)
        outside = [p for p in man["pixels"] if not poly.contains(Point(p[0], p[1]))]
        assert not outside, f"{tag}: {len(outside)} taxels outside edge_cuts"
    print("test_manifest_carries_the_real_board: OK (4 shape/mode combinations)")


def _manifest(**kw):
    resp = generate(GenerateRequest(outline=SQUARE, router="hug", board_mode="expand",
                                    pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2,
                                    pitch_y_mm=4.2, **kw))
    z = __import__("zipfile").ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64)))
    return json.loads(z.read("manifest.json")), resp.edit_data


def test_manifest_says_which_taxels_share_a_shear_bump():
    """The shear grouping has to be readable from the manifest alone.

    Without it a consumer would have to re-derive the 2x2 blocks, and the only
    indices in the file are PAD order — which is deliberately not geometric
    order, so `index // 2` pairs taxels that are nowhere near each other. This
    checks the published grouping is the one the bump sheet actually builds.
    """
    man, edit_data = _manifest()
    sh = man.get("shear")
    assert sh, "manifest has no shear block"
    assert sh["src"] == "taxel_2x2"
    assert len(sh["b"]) == len(man["pixels"]), "shear.b must be parallel to pixels"

    # The centres are the ones /bump-sheet builds for the same board.
    placements = np.asarray(edit_data["layout"]["placements_xy"], dtype=float)
    removed = set(edit_data.get("removed_pix") or [])
    active = [i for i in range(len(placements)) if i not in removed]
    want, _ = bump_centers_from_taxels(placements, active, 4.2, 4.2)
    got = np.asarray(sh["bumps"], dtype=float)
    assert len(got) == len(want), f"{len(got)} manifest bumps vs {len(want)} built"
    d, _ = cKDTree(want).query(got)
    assert d.max() < 1e-3, f"bump centres differ by {d.max():.4f}mm"

    # Groups are 2x2 blocks: at most four taxels, and each taxel within half a
    # pitch of its own bump in each axis (i.e. actually under it).
    groups = {}
    for p, b in zip(man["pixels"], sh["b"]):
        groups.setdefault(b, []).append(p)
    # This particular board is even x even, so nothing is orphaned. An odd
    # line count would orphan one — that rule has its own test.
    assert -1 not in groups, "even x even array should leave no taxel bumpless"
    assert max(len(g) for g in groups.values()) <= 4, "a bump covers one 2x2 block"
    assert 4 in {len(g) for g in groups.values()}, "expected some complete blocks"
    for b, members in groups.items():
        cx, cy = sh["bumps"][b]
        for x, y, _ri, _ci in members:
            assert abs(x - cx) <= 4.2 * 0.5 + 1e-6 and abs(y - cy) <= 4.2 * 0.5 + 1e-6


@pytest.mark.parametrize("outline", [LSHAPE, SQ_45x38])
def test_no_taxel_is_left_bumpless(outline):
    """Invariant 2 of the bump sheet, checked from the manifest.

    An odd line count does NOT orphan anything on the taxel path: the leftover
    line forms its own partial block and keeps a full-size bump. The L-shape
    here comes out 11x9 — odd in both axes — and still covers every taxel.
    """
    resp = generate(GenerateRequest(outline=outline, router="hug"))
    z = __import__("zipfile").ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64)))
    man = json.loads(z.read("manifest.json"))
    sh = man["shear"]
    assert sh["src"] == "taxel_2x2"
    P = np.asarray([[p[0], p[1]] for p in man["pixels"]], dtype=float)
    n_cols = len(np.unique(np.round(P[:, 0], 2)))
    n_rows = len(np.unique(np.round(P[:, 1], 2)))
    bumpless = [p for p, b in zip(man["pixels"], sh["b"]) if b < 0]
    assert not bumpless, \
        f"{n_cols}x{n_rows} grid left {len(bumpless)} taxels bumpless"


if __name__ == "__main__":
    test_manifest_carries_the_real_board()
    test_manifest_says_which_taxels_share_a_shear_bump()
    for _o in (LSHAPE, SQ_45x38):
        test_no_taxel_is_left_bumpless(_o)
