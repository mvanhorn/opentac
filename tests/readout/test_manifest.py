"""opentac.manifest: loading, validation (versions 3-6), and recovery from the QR code."""
import copy
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from opentac import manifest
from opentac.cad.export import taxel_core
from opentac.device import device_config

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((ROOT / "examples" / "manifests").glob("*.json"))
# The manifests the readout firmware was developed and tested against. Only on
# the lab's machines; skipped elsewhere.
STM_TEST = Path.home() / "Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Arduino/stm_test"
STM_MANIFESTS = sorted(STM_TEST.glob("manifest_*.json"))


def raw(name):
    return json.loads((ROOT / "examples" / "manifests" / name).read_text())


def test_examples_load_and_cover_versions():
    versions = set()
    for path in EXAMPLES:
        m = manifest.load(path, scan=True)
        assert m.n == len(m.raw["pixels"]) and m.xy.shape == (m.n, 2)
        assert m.source_bytes == path.read_bytes()
        assert m.name == path.name
        versions.add(m.version)
    assert versions == {3, 5, 6}


def test_version_4_reads_like_its_neighbours():
    # No v4 file is in circulation; v4 only added `design`, so a v3 file stamped 4 must read.
    d = raw("square_9x9_30pin.json")
    d["v"] = 4
    assert manifest.load(d, scan=True).version == 4


def test_versions_outside_3_to_6():
    d = raw("square_9x9_30pin.json")
    d["v"] = 2
    with pytest.raises(manifest.ManifestError, match="older"):
        manifest.load(d)
    d["v"] = 7
    assert any("newer" in w for w in manifest.load(d).warnings)
    del d["v"]
    m = manifest.load(d)
    assert m.version == 3 and any("version" in w for w in m.warnings)


@pytest.mark.skipif(not STM_MANIFESTS, reason="stm_test manifests are only on the lab machines")
def test_every_stm_test_manifest_loads():
    for path in STM_MANIFESTS:
        m = manifest.load(path, scan=True)
        assert m.version in (3, 5) and m.n > 0, path.name
    assert len(STM_MANIFESTS) >= 30


def test_zip_and_dict_and_manifest_sources(tmp_path):
    path = ROOT / "examples" / "manifests" / "square_9x9_60pin.json"
    z = tmp_path / "design.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("design/manifest.json", path.read_bytes())
        f.writestr("design/board.kicad_pcb", "(kicad_pcb)")
    from_zip = manifest.load(z)
    assert from_zip.source_bytes == path.read_bytes()
    assert from_zip.config_id == manifest.load(path).config_id
    assert manifest.load(from_zip).config_id == from_zip.config_id
    d = manifest.load(json.loads(path.read_text()))
    assert np.allclose(d.xy, from_zip.xy)


@pytest.mark.parametrize("break_it, message", [
    (lambda d: d.pop("pixels"), "no 'pixels'"),
    (lambda d: d.pop("rp"), "no 'rp'"),
    (lambda d: d["m"].pop(), "m must hold"),
    (lambda d: d.__setitem__("m", [d["m"][0][:-1]] + d["m"][1:]), r"m\[0\] must be"),
    (lambda d: d["pixels"].pop(), "m marks 81 taxels but pixels has 80"),
    (lambda d: d["pixels"].reverse(), "pixels must follow m's order"),
    (lambda d: d["rp"].pop(), "rp has 8 pads"),
    (lambda d: d["cp"].__setitem__(0, d["rp"][0]), "used by more than one"),
    (lambda d: d["shear"]["b"].pop(), "shear.b has 80 entries"),
    (lambda d: d["pixels"][3].__setitem__(0, float("nan")), "finite"),
    (lambda d: d.__setitem__("w", -1), "'w' must be a positive"),
])
def test_validation_says_what_is_wrong(break_it, message):
    d = copy.deepcopy(raw("square_9x9_30pin.json"))
    break_it(d)
    with pytest.raises(manifest.ManifestError, match=message):
        manifest.load(d, scan=True)


def test_scan_fields_only_required_for_scanning():
    d = raw("square_9x9_30pin.json")
    for key in manifest.SCAN_KEYS:
        d.pop(key)
    assert manifest.load(d).n == 81          # drawable
    with pytest.raises(manifest.ManifestError, match="cannot scan"):
        manifest.load(d, scan=True)


def test_not_a_manifest(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text("{not json")
    with pytest.raises(manifest.ManifestError, match="not JSON"):
        manifest.load(bad)
    with pytest.raises(manifest.ManifestError, match="no such file"):
        manifest.load(tmp_path / "missing.json")


def test_drawn_layout_uses_pixels_true():
    d = raw("pinky_tip_30pin.json")
    view = manifest.drawn(d)
    assert view["pixels"] == d["pixels_true"] and view is not d
    assert manifest.drawn(raw("square_9x9_30pin.json"))["pixels"] == raw("square_9x9_30pin.json")["pixels"]


# Only boards OpenTac CAD lays out carry the QR, and CAD puts the column pads
# right after the row pads, which the QR format relies on (taxel_core: "cp0 ==
# rp0 + r"). pinky_tip_30pin.json came from another tool with the columns on the
# lower pads, so it is left out: its QR would decode with the wrong column pads.
@pytest.mark.parametrize("name, board", [("square_9x9_60pin.json", "READOUT60"),
                                         ("finger_8x6_60pin.json", "READOUT60"),
                                         ("square_9x9_30pin.json", "NODE16")])
def test_qr_round_trip(name, board):
    original = manifest.load(ROOT / "examples" / "manifests" / name)
    payload = taxel_core.b41encode(taxel_core.encode(original.raw))
    back = manifest.from_qr(payload)
    for key in ("r", "c", "rp", "cp", "m"):
        assert back.raw[key] == original.raw[key], key
    assert (back.row == original.row).all() and (back.col == original.col).all()
    assert np.abs(back.xy - original.xy).max() <= 0.005 + 1e-9
    assert set(manifest.ABSENT) & set(back.raw) == set()
    # Scanners hand back trimmed or lowercased text; both still decode.
    assert manifest.from_qr(" " + payload.lower() + "\n").raw["m"] == original.raw["m"]
    # A recovered skin can be set up on its board, with the same lines and pitch.
    blob, spec = device_config(back, board)
    _, spec0 = device_config(original, board)
    for key in ("rows", "cols", "row_order", "col_order", "row_geom", "col_geom", "active"):
        assert spec[key] == spec0[key], key
    assert abs(spec["pitch_x_um"] - spec0["pitch_x_um"]) <= 10
    assert spec["shear_groups"] is None           # the QR carries no shear grouping


def test_qr_garbage():
    with pytest.raises(manifest.ManifestError, match="QR"):
        manifest.from_qr("HELLO WORLD!")
    with pytest.raises(manifest.ManifestError, match="QR"):
        manifest.from_qr("")
