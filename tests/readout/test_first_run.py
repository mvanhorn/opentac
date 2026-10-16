"""What a first-time user meets with no board, a blank board, or a locked port.

No hardware: port listing, the serial port, dfu-util and the FakeBoard stand in.
"""
import errno
import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import serial
from serial.tools import list_ports

import opentac
from opentac import manifest
from opentac.cli import build_parser, main
from opentac.device import config, flash as flashmod, link
from opentac.device.protocol import DeviceError
from opentac.frame import Frame
from opentac.manifest import ManifestError

from .fakeboard import FakeBoard

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "manifests"


@pytest.fixture(autouse=True)
def opentac_home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENTAC_HOME", str(tmp_path / "home"))


def ports(*specs):
    """Fake list_ports.comports() entries: (device, vid)."""
    return lambda: [SimpleNamespace(device=d, vid=v) for d, v in specs]


@pytest.fixture
def no_usb(monkeypatch):
    """Nothing on USB, whatever is plugged into the machine running the tests."""
    monkeypatch.setattr(list_ports, "comports", ports())


def dfu(monkeypatch, state, tool=("dfu-util", "/usr/bin/dfu-util")):
    monkeypatch.setattr(flashmod, "find_tool", lambda: tool)
    monkeypatch.setattr(flashmod, "dfu_state", lambda name, path: state)


# --- flashing a blank board -------------------------------------------------------

def test_no_board_and_no_dfu_says_a_new_board_needs_boot0(no_usb, monkeypatch):
    dfu(monkeypatch, None)
    with pytest.raises(DeviceError) as exc:
        flashmod.flash()
    text = str(exc.value)
    assert "no board is waiting in its bootloader" in text
    assert "A new board has no firmware yet" in text and "BOOT0" in text
    assert "opentac flash --board readout60" in text and "opentac flash --board node16" in text
    with pytest.raises(DeviceError, match="run this command again"):
        flashmod.flash(binary="firmware/prebuilt/NODE16_bus.bin")


def test_cli_flash_with_nothing_attached(no_usb, monkeypatch, capsys):
    dfu(monkeypatch, None)
    assert main(["flash"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: no OpenTac board answered on USB")
    assert "hold its BOOT0 button" in err


def test_board_in_dfu_without_board_asks_for_board(no_usb, monkeypatch):
    dfu(monkeypatch, "ready")
    with pytest.raises(DeviceError, match="add --board readout60"):
        flashmod.flash(log=lambda *_: None)


def test_board_in_dfu_with_board_alias(no_usb, monkeypatch):
    dfu(monkeypatch, "ready")
    lines = []
    assert flashmod.flash(board="30pin", dry_run=True, log=lines.append) == 0
    assert lines[-2] == "would run:"
    assert lines[-1].split()[-1].endswith("NODE16.bin") and "0483:df11" in lines[-1]


def test_dry_run_with_board_needs_no_board_attached(no_usb, monkeypatch):
    dfu(monkeypatch, None)
    lines = []
    assert flashmod.flash(board="60-pin", dry_run=True, log=lines.append) == 0
    assert "no board answers on USB" in lines[-2] and "BOOT0" in lines[-2]
    assert lines[-1].strip().endswith("READOUT60.bin")
    # without --board a dry run still has to say what is missing
    with pytest.raises(DeviceError, match="A new board has no firmware"):
        flashmod.flash(dry_run=True, log=lines.append)


def test_dry_run_without_a_flashing_tool(no_usb, monkeypatch):
    monkeypatch.setattr(flashmod, "find_tool", lambda: None)
    lines = []
    assert flashmod.flash(board="readout60", dry_run=True, log=lines.append) == 0
    assert "brew install dfu-util" in lines[0] and "sudo apt install dfu-util" in lines[0]
    assert "STM32CubeProgrammer" in lines[0]
    assert lines[-1].strip().startswith("dfu-util -a 0")
    with pytest.raises(DeviceError, match="neither dfu-util nor STM32CubeProgrammer"):
        flashmod.flash(board="readout60")


def test_dfu_permission_on_linux_names_the_udev_rule(no_usb, monkeypatch):
    dfu(monkeypatch, "denied")
    with pytest.raises(DeviceError) as exc:
        flashmod.flash(board="readout60")
    text = str(exc.value)
    assert flashmod.UDEV_RULE in text and "udevadm control --reload-rules" in text
    assert 'ATTRS{idVendor}=="0483", ATTRS{idProduct}=="df11"' in flashmod.UDEV_RULE


def test_dfu_state_reads_dfu_util(monkeypatch):
    def run(stdout="", stderr=""):
        return lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout, stderr)
    monkeypatch.setattr(subprocess, "run", run(
        'Found DFU: [0483:df11] ver=0200, devnum=7, cfg=1, intf=0, alt=0, name="@Internal Flash"'))
    assert flashmod.dfu_state("dfu-util", "dfu-util") == "ready"
    monkeypatch.setattr(subprocess, "run", run(stderr=(
        "dfu-util: Cannot open DFU device 0483:df11 found on devnum 7 (LIBUSB_ERROR_ACCESS)")))
    assert flashmod.dfu_state("dfu-util", "dfu-util") == "denied"
    monkeypatch.setattr(subprocess, "run", run())
    assert flashmod.dfu_state("dfu-util", "dfu-util") is None


def test_flash_reads_the_board_name_from_running_firmware(monkeypatch):
    fake = FakeBoard("READOUT60")
    monkeypatch.setattr(link, "find", lambda kind: "/dev/fake")

    def fake_open(self, port):
        self.link, self.port, self._sent, self._buf = fake, port, "", b""
    monkeypatch.setattr(link.Transport, "_open", fake_open)
    dfu(monkeypatch, None)
    lines = []
    assert flashmod.flash(dry_run=True, log=lines.append) == 0
    assert lines[0] == "the board says it is the 60-pin board (READOUT60)"
    assert lines[1] == "would send B to /dev/fake (reboot into DFU), then run:"
    assert lines[2].endswith("READOUT60.bin")
    with pytest.raises(DeviceError, match=r"the 60-pin board \(READOUT60\), not the 30-pin pair"):
        flashmod.flash(board="30pin", dry_run=True, log=lines.append)


def test_prebuilt_firmware_matches_its_checksums():
    import hashlib
    sums = (flashmod.PREBUILT / "SHA256SUMS").read_text().split("\n")
    for line in filter(None, sums):
        digest, name = line.split()
        assert hashlib.sha256((flashmod.PREBUILT / name).read_bytes()).hexdigest() == digest, name
    for name in flashmod.BINARIES.values():
        assert (flashmod.PREBUILT / name).exists()


def test_prebuilt_is_found_inside_an_installed_package(tmp_path, monkeypatch):
    pkg = tmp_path / "site-packages" / "opentac" / "device"
    (pkg / "firmware").mkdir(parents=True)
    monkeypatch.setattr(flashmod, "__file__", str(pkg / "flash.py"))
    assert flashmod._prebuilt_dir() == pkg / "firmware"
    # and the wheel puts it there
    import tomllib
    wheel = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["hatch"]["build"][
        "targets"]["wheel"]
    assert wheel["force-include"]["firmware/prebuilt"] == "opentac/device/firmware"


# --- ports: permissions and which ports are probed ---------------------------------

def refuse(exc):
    def open_(*a, **k):
        raise exc
    return open_


def test_permission_denied_is_reported(monkeypatch):
    monkeypatch.setattr(link.sys, "platform", "linux")
    monkeypatch.setattr(link, "candidate_ports", lambda: ["/dev/ttyACM0"])
    monkeypatch.setattr(serial, "Serial", refuse(serial.SerialException(
        errno.EACCES, "could not open port /dev/ttyACM0: [Errno 13] Permission denied: "
                      "'/dev/ttyACM0'")))
    assert link.probe("/dev/ttyACM0") == "denied"
    with pytest.raises(link.NoDeviceError) as exc:
        link.find("board")
    assert str(exc.value) == ("found /dev/ttyACM0 but cannot open it (permission denied): "
                              "sudo usermod -aG dialout $USER, then log out and back in")
    assert exc.value.denied == ["/dev/ttyACM0"]
    with pytest.raises(link.NoDeviceError, match="permission denied"):
        link.find_any()
    # opening the port directly says the same
    with pytest.raises(DeviceError,
                       match=r"cannot open /dev/ttyACM0 \(permission denied\): sudo usermod"):
        link.UsbLink("/dev/ttyACM0")
    # and flash does not mistake it for a blank board
    monkeypatch.setattr(list_ports, "comports", ports())
    dfu(monkeypatch, None)
    with pytest.raises(link.NoDeviceError, match="dialout"):
        flashmod.flash()


def test_port_held_by_another_program_on_windows(monkeypatch):
    monkeypatch.setattr(link.sys, "platform", "win32")
    monkeypatch.setattr(link, "candidate_ports", lambda: ["COM5"])
    monkeypatch.setattr(serial, "Serial", refuse(serial.SerialException(
        "could not open port 'COM5': PermissionError(13, 'Access is denied.', None, 5)")))
    with pytest.raises(link.NoDeviceError, match="another program has it open"):
        link.find("board")


def test_nothing_on_usb_mentions_flashing(no_usb):
    with pytest.raises(link.NoDeviceError, match="A new board has no firmware yet"):
        link.find("board")


def test_only_st_and_teensy_ports_are_probed(monkeypatch):
    monkeypatch.setattr(list_ports, "comports", ports(
        ("/dev/ttyACM0", 0x0483), ("/dev/ttyACM1", 0x2341), ("/dev/ttyUSB0", 0x0403),
        ("/dev/ttyACM2", 0x16C0), ("/dev/ttyACM3", None)))
    assert link.candidate_ports() == ["/dev/ttyACM0", "/dev/ttyACM2"]
    # a port with no vendor id is tried only when nothing matched by vendor id
    monkeypatch.setattr(list_ports, "comports", ports(
        ("/dev/ttyACM1", 0x2341), ("/dev/ttyACM3", None), ("/dev/ttyS0", None)))
    assert link.candidate_ports() == ["/dev/ttyACM3"]
    # macOS lists a port as tty. and cu.; only cu. is used
    monkeypatch.setattr(list_ports, "comports", ports(
        ("/dev/tty.usbmodem101", 0x0483), ("/dev/cu.usbmodem101", 0x0483)))
    assert link.candidate_ports() == ["/dev/cu.usbmodem101"]


# --- board names ---------------------------------------------------------------------

@pytest.mark.parametrize("name, firmware", [
    ("readout60", "READOUT60"), ("READOUT60", "READOUT60"), ("60pin", "READOUT60"),
    ("60-pin", "READOUT60"), ("node16", "NODE16"), ("30pin", "NODE16"), ("30-PIN", "NODE16"),
    ("readout32", "READOUT32"), ("70pin", "READOUT32")])
def test_board_aliases(name, firmware):
    assert config.board(name).name == firmware
    assert config.canonical(name) == firmware


def test_unknown_board_lists_the_names():
    with pytest.raises(ManifestError, match="readout60 or 60pin"):
        config.board("30 pin")
    assert flashmod.binary_for("60pin").name == "READOUT60.bin"


def test_cli_board_choices_take_aliases(capsys):
    parser = build_parser()
    assert parser.parse_args(["flash", "--board", "60-PIN"]).board == "60-pin"
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["flash", "--board", "60"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err and "30pin" in err and "usage:" in err


def test_setup_dry_run_takes_an_alias_or_infers_the_board(capsys):
    assert main(["setup", str(EXAMPLES / "square_9x9_30pin.json"), "--dry-run",
                 "--board", "30pin"]) == 0
    assert "81 taxels on the 30-pin pair (NODE16)" in capsys.readouterr().out
    assert main(["setup", str(EXAMPLES / "finger_8x6_60pin.json"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "board: the 60-pin board (READOUT60), from the manifest's pads" in out
    assert "45 taxels on the 60-pin board (READOUT60)" in out


def test_setup_dry_run_that_cannot_tell_the_board(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:      # fits no board: argparse-style error
        main(["setup", str(ROOT / "tests/sim/fixtures/flat_square.json"), "--dry-run"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "usage: opentac setup" in err and "pass --board readout60 or --board node16" in err


def test_manifest_check_names_boards_like_the_docs(capsys):
    assert main(["manifest", "check", str(EXAMPLES / "pinky_tip_30pin.json")]) == 0
    out = capsys.readouterr().out
    assert "  fits the 30-pin pair (NODE16)" in out
    assert "  does not fit the 60-pin board (READOUT60): " in out
    assert "READOUT32" not in out and "70-pin" not in out


def test_manifest_check_names_the_70pin_board_only_when_it_alone_fits(monkeypatch, capsys):
    import opentac.device
    monkeypatch.setattr(opentac.device, "fits", lambda m: {
        "NODE16": "pad 40 is not on the NODE16 connector", "READOUT32": None,
        "READOUT60": "both lines on the mux"})
    assert main(["manifest", "check", str(EXAMPLES / "square_9x9_60pin.json")]) == 0
    out = capsys.readouterr().out
    assert "  fits the earlier 70-pin board (READOUT32) only" in out
    assert "does not fit the 60-pin board (READOUT60): both lines on the mux" in out


# --- manifests and the Python API -----------------------------------------------------

def test_manifest_check_on_design_json_says_what_to_pass(tmp_path, capsys):
    path = tmp_path / "design.json"
    path.write_text(json.dumps({"pipeline": "x", "edit_data": {"params": {}, "outline": []}}))
    assert main(["manifest", "check", str(path)]) == 1
    err = capsys.readouterr().err
    assert "design.json, OpenTac-CAD's save file" in err and "pass the design .zip" in err
    import zipfile
    gerbers = tmp_path / "gerbers.zip"
    with zipfile.ZipFile(gerbers, "w") as z:
        z.writestr("board-F_Cu.gtl", "")
    assert main(["manifest", "check", str(gerbers)]) == 1
    assert "has no manifest.json: pass the design .zip" in capsys.readouterr().err


def test_manifest_repr_is_short():
    m = manifest.load(EXAMPLES / "square_9x9_60pin.json")
    assert repr(m) == ("Manifest(square_9x9_60pin.json: 60-pin board (READOUT60), 9 x 9, "
                       "81 taxels, pitch 4.20 x 4.20 mm)")
    sim = manifest.load(ROOT / "tests/sim/fixtures/flat_square.json")
    assert repr(sim).startswith("Manifest(flat_square.json: ") and "48 taxels" in repr(sim)


def test_frame_declares_node_and_t_us():
    f = Frame(taxels=np.ones(3), t=0.0)
    assert f.node is None and f.t_us is None


def test_opentac_lists_its_lazy_names():
    names = dir(opentac)
    for name in ("Sensor", "Bus", "Frame", "load_manifest"):
        assert name in names and name in opentac.__doc__
    assert "opentac.sim.Sensor" in opentac.__doc__
    from opentac.device import Sensor
    assert "opentac.sim.Sensor" in Sensor.__doc__


# --- CLI argument errors and the examples ---------------------------------------------

@pytest.mark.parametrize("argv, words", [
    (["view", "--set", "GAIN_SW"], "argument --set: takes NAME=VALUE"),
    (["info", "--node", "two"], "argument --node: takes a node number"),
])
def test_cli_argument_errors_show_usage(argv, words, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "usage: opentac" in err and words in err


@pytest.mark.parametrize("script", ["read_sensor.py", "record_take.py", "xampling.py"])
def test_examples_without_a_board_print_one_line(script, monkeypatch):
    import opentac.device

    def no_board(*a, **k):
        raise link.not_found("readout board", {})
    monkeypatch.setattr(opentac.device, "Sensor", no_board)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(ROOT / "examples" / script), run_name="__main__")
    message = str(exc.value.code)
    assert message.startswith("error: no OpenTac readout board found on USB")
    assert "\n" not in message and "opentac flash" in message
