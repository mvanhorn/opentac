"""The printed case in the Download ZIP (/zip-with-case).

What matters to the user: the ZIP carries a printable case by default, it is
built from their case settings, and a case that cannot be built never costs
them the board files.
"""
import base64
import io
import zipfile

import pytest
import trimesh
from fastapi.testclient import TestClient

from opentac.cad import main
from opentac.cad.export import case_zip

client = TestClient(main.app)

SQUARE = [[0, 0], [42, 0], [42, 42], [0, 42]]
STLS = ["case/demo_case_bottom.stl", "case/demo_case_top_bumps.stl",
        "case/demo_case_top_flat.stl", "case/demo_case_top_hatched.stl"]


@pytest.fixture(scope="module")
def board():
    r = client.post("/generate", json={"outline": SQUARE, "router": "hug", "readout_board": "any"})
    assert r.status_code == 200, r.text[:300]
    j = r.json()
    return j["zip_b64"], j["edit_data"]


def _zip(b64):
    return zipfile.ZipFile(io.BytesIO(base64.b64decode(b64)))


def _post(board, **extra):
    zip_b64, edit_data = board
    body = {"zip_b64": zip_b64, "edit_data": edit_data, "name": "demo", **extra}
    r = client.post("/zip-with-case", json=body)
    assert r.status_code == 200, r.text[:300]
    return r.json()


def _mesh(zf, name):
    return trimesh.load(io.BytesIO(zf.read(name)), file_type="stl")


def test_zip_gets_the_bottom_and_all_three_lids(board):
    j = _post(board)
    assert j["case_error"] is None
    assert sorted(j["case_files"]) == sorted(STLS)
    zf = _zip(j["zip_b64"])
    names = zf.namelist()
    for n in STLS + ["case/README.txt"]:
        assert n in names, n
    for n in STLS:
        mesh = _mesh(zf, n)
        assert len(mesh.faces) > 0, f"{n} is empty"
        assert mesh.is_watertight, f"{n} is not a closed solid"
        assert mesh.body_count == 1, f"{n} came out as {mesh.body_count} loose bodies"
        assert mesh.volume > 0


def test_board_files_are_untouched(board):
    src = _zip(board[0])
    out = _zip(_post(board)["zip_b64"])
    for n in src.namelist():
        assert out.read(n) == src.read(n), f"{n} changed"


def test_readme_lists_the_files_and_marks_the_bumped_lid_by_default(board):
    readme = _zip(_post(board)["zip_b64"]).read("case/README.txt").decode()
    for n in STLS:
        assert n.split("/")[1] in readme
    primary = [line for line in readme.splitlines() if "primary lid" in line]
    assert len(primary) == 1 and "demo_case_top_bumps.stl" in primary[0]


def test_case_settings_are_used(board):
    lid, bump_h, proud = 1.2, 3.0, 0.6
    j = _post(board, primary="flat",
              params={"lid_thickness_mm": lid, "plate_thickness_mm": 0.8, "recess_depth_mm": 0.4,
                      # the lid choice must not leak into the files: every lid is built
                      "bumps": False, "hatch": False},
              bump_params={"bump_height_mm": bump_h})
    assert j["case_error"] is None
    zf = _zip(j["zip_b64"])
    top_z = lambda n: float(_mesh(zf, n).bounds[1][2])  # noqa: E731
    assert top_z("case/demo_case_top_flat.stl") == pytest.approx(lid, abs=1e-6)
    assert top_z("case/demo_case_top_bumps.stl") == pytest.approx(lid + bump_h, abs=1e-6)
    assert top_z("case/demo_case_bottom.stl") == pytest.approx(0.8 + 0.4 + lid + proud, abs=1e-6)
    readme = zf.read("case/README.txt").decode()
    primary = [line for line in readme.splitlines() if "primary lid" in line]
    assert len(primary) == 1 and "demo_case_top_flat.stl" in primary[0]
    assert "lid 1.2" in readme and "height 3" in readme


def test_case_failure_still_returns_the_board_zip(board, monkeypatch):
    def boom(*a, **k):
        raise ValueError("The rib came out empty — check skirt width against the board size.")
    monkeypatch.setattr(case_zip, "build_case", boom)
    j = _post(board)
    assert j["case_files"] == []
    assert "rib came out empty" in j["case_error"]
    src, out = _zip(board[0]), _zip(j["zip_b64"])
    assert sorted(out.namelist()) == sorted(src.namelist())
    assert not any(n.startswith("case/") for n in out.namelist())


def test_unexpected_case_error_is_soft_too(board, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("engine failed")
    monkeypatch.setattr(case_zip, "build_case", boom)
    j = _post(board)
    assert j["case_files"] == [] and "engine failed" in j["case_error"]
    assert "board.kicad_pcb" in _zip(j["zip_b64"]).namelist()


def test_edit_data_without_a_board_is_soft(board):
    j = client.post("/zip-with-case", json={"zip_b64": board[0], "edit_data": {}}).json()
    assert j["case_files"] == [] and "generate a board first" in j["case_error"]
    assert "board.kicad_pcb" in _zip(j["zip_b64"]).namelist()


def test_one_lid_failing_keeps_the_rest(board, monkeypatch):
    real = case_zip.build_case

    def no_hatch(*a, **k):
        if a[6].hatch:
            raise ValueError("hatch broke")
        return real(*a, **k)
    monkeypatch.setattr(case_zip, "build_case", no_hatch)
    j = _post(board)
    assert sorted(j["case_files"]) == sorted(s for s in STLS if "hatched" not in s)
    assert "hatched lid was left out: hatch broke" in j["case_error"]
    assert "hatch broke" in _zip(j["zip_b64"]).read("case/README.txt").decode()


def test_an_old_case_in_the_zip_is_replaced(board):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        src = _zip(board[0])
        for n in src.namelist():
            zf.writestr(n, src.read(n))
        zf.writestr("case/old_case_top.stl", b"stale")
    j = _post((base64.b64encode(buf.getvalue()).decode(), board[1]))
    assert "case/old_case_top.stl" not in _zip(j["zip_b64"]).namelist()


def test_not_a_zip_is_rejected(board):
    r = client.post("/zip-with-case", json={"zip_b64": base64.b64encode(b"nope").decode(),
                                           "edit_data": board[1]})
    assert r.status_code == 400
