"""Manifest -> DeviceConfig bytes, and the board wiring tables they go through."""
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import zlib
from contextlib import redirect_stderr
from pathlib import Path

import pytest

from opentac.device import board, boards, device_config, fits
from opentac.manifest import ManifestError

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "manifests"
FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = json.loads((FIXTURES / "configs_from_opentac_device.json").read_text())
# Four of the manifests the readout firmware was developed against, with the
# bytes the original tool built from them.
STM_FIXTURES = FIXTURES / "stm_test"
STM_GOLDEN = json.loads((FIXTURES / "stm_test_configs_from_opentac_device.json").read_text())
# The full stm_test folder and the original tool live only on the lab's
# machines. Point OPENTAC_STM_TEST at that folder to compare all of it;
# otherwise that check is skipped, on every machine alike.
STM_TEST = Path(os.environ.get("OPENTAC_STM_TEST", "/nonexistent"))
ORIGINAL = STM_TEST / "tactile_imu_fast_cfg" / "opentac_device.py"


def _check_golden(path, golden):
    for board_name, want in golden.items():
        if want is None:
            with pytest.raises(ManifestError):
                device_config(path, board_name)
        else:
            blob, _ = device_config(path, board_name)
            assert blob.hex() == want, board_name


@pytest.mark.parametrize("name", [n for n in GOLDEN if not n.startswith("_")])
def test_config_bytes_match_the_original_tool(name):
    """Same manifest, same board -> the same bytes opentac_device.py built."""
    _check_golden(EXAMPLES / name, GOLDEN[name])


@pytest.mark.parametrize("name", [n for n in STM_GOLDEN if not n.startswith("_")])
def test_stm_test_config_bytes_match_the_original_tool(name):
    _check_golden(STM_FIXTURES / name, STM_GOLDEN[name])


@pytest.mark.skipif(not ORIGINAL.exists(), reason="set OPENTAC_STM_TEST to the lab's stm_test folder")
def test_every_stm_test_manifest_matches_the_original_tool():
    spec = importlib.util.spec_from_file_location("opentac_device_original", ORIGINAL)
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    same = refused = 0
    for path in sorted(STM_TEST.glob("manifest_*.json")):
        raw = path.read_bytes()
        for name, pad_line in orig.BOARD_MAPS.items():
            try:
                with redirect_stderr(io.StringIO()):
                    want = orig.build_config(orig.extract(json.loads(raw), pad_line),
                                             zlib.crc32(raw) & 0xFFFFFFFF)
            except orig.ManifestError:
                want = None
            if want is None:
                with pytest.raises(ManifestError):
                    device_config(path, name)
                refused += 1
            else:
                assert device_config(path, name)[0] == want, (path.name, name)
                same += 1
    assert same >= 28 and same + refused >= 60


def test_board_tables_are_self_consistent():
    """Each row: pad, board pin (the cable mirrors), net, line; nets agree with lines."""
    pins = {"READOUT60": 60, "NODE16": 30, "READOUT32": 70}
    for name, b in boards().items():
        rows = b.raw["pads"]
        lines = [r[3] for r in rows]
        assert len(set(lines)) == len(lines), name
        for pad, pin, net, line in rows:
            assert pin == pins[name] + 1 - pad, (name, pad)
            kind, k = net[0], int(net[1:])
            if kind == "r":
                assert line == k - 1, (name, pad, net)          # r1 is the '595's first output
            else:
                assert kind == "c" and line == 32 + k - 1, (name, pad, net)   # c1 is mux S1


def test_tables_match_the_measured_maps():
    # READOUT32, measured 2026-09-10: line = 67 - pad over pads 4..67.
    assert board("readout32").pad_line == {p: 67 - p for p in range(4, 68)}
    # NODE16, measured on pads 7..24 and extended: steps 15 - pad, then mux 30 - pad.
    assert board("node16").pad_line == {p: (15 - p if p <= 15 else 32 + 30 - p) for p in range(1, 31)}
    # READOUT60, from its netlist with the same mirrored cable: line = 62 - pad.
    assert board("readout60").pad_line == {p: 62 - p for p in range(1, 61)}
    assert not board("readout60").hardware_checked


def test_which_board_a_skin_fits():
    assert {k for k, v in fits(EXAMPLES / "square_9x9_60pin.json").items() if v is None} == {"READOUT60"}
    assert {k for k, v in fits(EXAMPLES / "square_9x9_30pin.json").items() if v is None} == {"NODE16"}
    with pytest.raises(ManifestError, match="not on the NODE16 connector"):
        device_config(EXAMPLES / "square_9x9_60pin.json", "NODE16")


def test_60pin_skin_lands_rows_on_the_mux_and_columns_on_the_595s():
    _, spec = device_config(EXAMPLES / "square_9x9_60pin.json", "READOUT60")
    rp = json.loads((EXAMPLES / "square_9x9_60pin.json").read_text())["rp"]
    assert rp == list(range(22, 31))
    assert spec["row_order"] == [32 + 30 - p for p in rp]          # mux channels 8..0
    assert spec["col_order"] == list(range(31, 22, -1))              # steps 31..23


@pytest.mark.skipif(shutil.which("c++") is None, reason="needs a C++ compiler")
def test_firmware_parses_and_xamples_these_configs(tmp_path):
    """The firmware's own parser, scan plan and xampling engine, on configs built here.

    firmware/tests/xampling_host_test.cpp compiles ot_devcfg.h and ot_xampling.h
    for the host and runs them against config files.
    """
    fw = ROOT / "firmware"
    cases = {"READOUT60": ["square_9x9_60pin.json", "finger_8x6_60pin.json"],
             "NODE16": ["square_9x9_30pin.json", "pinky_tip_30pin.json"]}
    for flag, names in cases.items():
        exe = tmp_path / f"xtest_{flag}"
        subprocess.run(["c++", "-std=c++17", "-O2", "-DOTCFG_NO_FLASH", f"-DOT_BOARD_{flag}",
                        f"-I{fw / 'opentac_readout'}", str(fw / "tests" / "xampling_host_test.cpp"),
                        "-o", str(exe)], check=True)
        files = []
        for name in names:
            blob, _ = device_config(EXAMPLES / name, flag)
            path = tmp_path / f"{name}.bin"
            path.write_bytes(blob)
            files.append(str(path))
        out = subprocess.run([str(exe), *files], capture_output=True, text=True)
        assert out.returncode == 0 and "all passed" in out.stdout, out.stdout[-2000:]
        assert len(re.findall(r"taxels, bump radius", out.stdout)) == len(names)
