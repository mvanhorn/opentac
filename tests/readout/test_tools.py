"""calibrate, flash command lines, the viewer (headless) and the CLI's offline commands."""
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from opentac.calibrate import Calibration, CalibrationError, design_key
from opentac.cli import main
from opentac.device import Sensor
from opentac.device.flash import BINARIES, flash_command
from opentac.device.link import UsbLink
from opentac.frame import Frame

from .fakeboard import FakeBoard

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "manifests"


@pytest.fixture(autouse=True)
def opentac_home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENTAC_HOME", str(tmp_path / "home"))


def calibration_file(tmp_path, model="power"):
    entry = {"status": "ok", "model": model, "power": {"a_N": 0.001, "b": 0.5},
             "linear": {"a_N": 0.0002, "b": 1.0}, "deadband_counts": 20.0, "sat_counts": 3800,
             "counts_fitted_max": 3000, "mux_settle_us": 8}
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"curves": {"p4.2_gap0.2": {"v2": entry},
                                           "p2.0_gap0.2": {"v1": {"status": "insufficient",
                                                                   "reason": "saturated"}}}}))
    return path


def test_counts_to_newtons(tmp_path):
    m = EXAMPLES / "square_9x9_60pin.json"
    assert design_key(m) == ("p4.2_gap0.2", 4.2)
    force = Calibration(calibration_file(tmp_path)).skin(m, "v2")
    counts = np.full(81, 120.0)
    counts[0] = np.nan                                   # not read this frame
    frame = Frame(taxels=counts, t=0.0)
    n = force.taxels(frame, zero=np.full(81, 20.0))
    assert np.isnan(n[0]) and n[1] == pytest.approx(0.001 * (120 - 20 - 20) ** 0.5)
    assert force.total(frame, np.full(81, 20.0)) == pytest.approx(80 * n[1])
    assert force.saturated(np.full(81, 3900.0)).all()
    linear = Calibration(calibration_file(tmp_path)).curve("p4.2_gap0.2", "v2", model="linear")
    assert linear.force(1020.0) == pytest.approx(0.2)


def test_calibration_says_what_is_missing(tmp_path):
    cal = Calibration(calibration_file(tmp_path))
    with pytest.raises(CalibrationError, match="no curve for velostat 'v9'"):
        cal.skin(EXAMPLES / "square_9x9_60pin.json", "v9")
    with pytest.raises(CalibrationError, match="saturated"):
        cal.curve("p2.0_gap0.2", "v1")
    with pytest.raises(CalibrationError, match="no calibration file"):
        Calibration(tmp_path / "missing.json")


def test_flash_command_lines():
    b = Path("NODE16.bin")
    assert flash_command("dfu-util", "dfu-util", b) == [
        "dfu-util", "-a", "0", "-d", "0483:df11", "-s", "0x08000000:leave", "-D", "NODE16.bin"]
    assert flash_command("cubeprog", "STM32_Programmer_CLI", b) == [
        "STM32_Programmer_CLI", "-c", "port=USB1", "-d", "NODE16.bin", "0x08000000", "-v",
        "-s", "0x08000000"]
    assert set(BINARIES) == {"readout60", "node16", "readout32"}


def test_viewer_snapshot(tmp_path):
    fake = FakeBoard("READOUT60")
    s = Sensor(_link=UsbLink(link=fake))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.setup(EXAMPLES / "square_9x9_60pin.json")
    from opentac.view import ShearField, view
    png = tmp_path / "snap.png"
    assert view([s], shear=True, snapshot=str(png), log=lambda *_: None) == 0
    assert png.stat().st_size > 10_000
    field = ShearField(s.manifest.raw)
    sx, sy, total = field.vectors(np.full(81, 1000.0))
    assert np.allclose(sx[field.valid[:, 0]], 0) and (total > 0).all()   # even load, no shear


def test_cli_manifest_check(capsys):
    assert main(["manifest", "check", str(EXAMPLES / "square_9x9_60pin.json")]) == 0
    out = capsys.readouterr().out
    assert "READOUT60  60-pin readout board: fits" in out and "NODE16     no:" in out


def test_cli_manifest_from_qr(tmp_path, capsys):
    from opentac.cad.export import taxel_core
    raw = json.loads((EXAMPLES / "square_9x9_30pin.json").read_text())
    payload = taxel_core.b41encode(taxel_core.encode(raw))
    out = tmp_path / "recovered.json"
    assert main(["manifest", "from-qr", payload, "-o", str(out)]) == 0
    assert json.loads(out.read_text())["m"] == raw["m"]
    assert main(["manifest", "from-qr", "not a qr!"]) == 1
    assert "cannot read the QR" in capsys.readouterr().err


def test_cli_setup_dry_run(tmp_path, capsys):
    out = tmp_path / "c.bin"
    assert main(["setup", str(EXAMPLES / "finger_8x6_60pin.json"), "--dry-run",
                 "--board", "readout60", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "rows    -> mux channels 7..0" in text and "columns -> shift-register steps 31..26" in text
    golden = json.loads((Path(__file__).parent / "fixtures" / "configs_from_opentac_device.json").read_text())
    assert out.read_bytes().hex() == golden["finger_8x6_60pin.json"]["READOUT60"]
    assert main(["setup", str(EXAMPLES / "finger_8x6_60pin.json"), "--dry-run", "--board", "node16"]) == 1


def test_cli_with_a_board_attached(tmp_path, monkeypatch, capsys):
    """setup, info and record through the CLI, the serial port replaced by a FakeBoard."""
    import opentac.device.link as link

    fake = FakeBoard("READOUT60")
    monkeypatch.setattr(link, "find_any", lambda: ("board", "/dev/fake"))
    monkeypatch.setattr(link, "find", lambda kind: "/dev/fake")

    def fake_open(self, port):
        self.link, self.port, self._sent, self._buf = fake, port, "", b""
    monkeypatch.setattr(link.Transport, "_open", fake_open)

    assert main(["info"]) == 0
    assert "no skin" in capsys.readouterr().out
    assert not [c for c in fake.commands if c.startswith("p ")]      # info changes nothing
    assert main(["setup", str(EXAMPLES / "square_9x9_60pin.json")]) == 0
    out = capsys.readouterr().out
    assert "READOUT60 on /dev/fake now holds square_9x9_60pin.json" in out
    assert "not yet been checked by pressing" in out
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "9 x 9, 81 taxels" in out and "GAIN_SW" in out and "(10000 ohm feedback)" in out
    assert main(["record", str(tmp_path / "takes"), "--seconds", "0.05", "--set", "GAIN_SW=2"]) == 0
    take = next((tmp_path / "takes").iterdir())
    info = json.loads((take / "take.json").read_text())
    assert info["settings"]["gain_sw"] == 2 and round(info["settings"]["feedback_ohm"]) == 991
