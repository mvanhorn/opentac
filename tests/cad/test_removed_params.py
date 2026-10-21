"""Designs that carry the removed anchor-hole parameters still open.

edit_data and design.json written by earlier versions can carry the five
anchor_hole* params and the internal pack_pattern_* keys. Loading drops them; a
design that had anchor holes switched on is refused, because rebuilding it
without its cutouts would produce a different board.
"""
import base64
import io
import zipfile

import pytest

from opentac.cad.main import ExportEditedRequest, GenerateRequest, export_edited, generate
from opentac.cad.pipeline.types import REMOVED_PARAM_KEYS, drop_removed_params

SQUARE = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
LEGACY = {
    "anchor_holes": False, "anchor_hole_dia_mm": 1.5, "anchor_hole_clearance_mm": 0.4,
    "anchor_hole_lattice": "uniform", "anchor_hole_square_cells": True,
    "pack_pattern_x": None, "pack_pattern_y": None,
}


def test_legacy_keys_are_dropped():
    assert set(LEGACY) == REMOVED_PARAM_KEYS
    assert drop_removed_params({"pitch_x_mm": 4.2, **LEGACY}) == {"pitch_x_mm": 4.2}


def test_a_design_that_used_anchor_holes_is_refused():
    with pytest.raises(ValueError, match="anchor holes"):
        drop_removed_params({**LEGACY, "anchor_holes": True})


def _files(zip_b64):
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(zip_b64)))
    return {n: z.read(n) for n in z.namelist() if n.endswith((".kicad_pcb", ".gbl", ".gtl", ".gm1"))}


def test_an_old_saved_design_re_exports_the_same_board():
    gen = generate(GenerateRequest(outline=SQUARE, router="hug"))
    ed = gen.edit_data
    fresh = export_edited(ExportEditedRequest(edit_data=ed, routes=ed["routes"]))

    old = dict(ed, params={**ed["params"], **LEGACY})
    reopened = export_edited(ExportEditedRequest(edit_data=old, routes=ed["routes"]))
    assert _files(reopened.zip_b64).keys() == _files(fresh.zip_b64).keys()
    # uuids and creation timestamps change on every export; the board must not
    volatile = (b"uuid", b"tstamp", b"CreationDate", b"G04 Created by")
    strip = lambda b: [l for l in b.split(b"\n") if not any(v in l for v in volatile)]
    for name, data in _files(fresh.zip_b64).items():
        assert strip(_files(reopened.zip_b64)[name]) == strip(data), name
