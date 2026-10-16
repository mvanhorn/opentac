"""Put firmware on a readout board over USB, with no BOOT0 button.

The running firmware reboots into the STM32's own USB bootloader (DFU) when it
gets the line `B`. Then dfu-util, or STMicroelectronics' STM32CubeProgrammer,
writes the binary from the start of flash. Both write only the pages the
binary covers, so the skin config near the top of flash survives (checked with
dfu-util on hardware; README 2026-09-10).

A new board has no firmware, so nothing answers `B`: it goes into the
bootloader only if BOOT0 is held while it is plugged in, and since a board in
the bootloader cannot say which board it is, `board` must then be given.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from opentac.device.config import canonical, label
from opentac.device.link import NoDeviceError
from opentac.device.protocol import DeviceError, parse_caps


def _prebuilt_dir() -> Path:
    """The prebuilt binaries: opentac/device/firmware/ in an installed wheel (pyproject
    copies firmware/prebuilt/ there), else firmware/prebuilt/ in a checkout of the repo."""
    packaged = Path(__file__).resolve().with_name("firmware")
    if packaged.is_dir():
        return packaged
    return Path(__file__).resolve().parents[2] / "firmware" / "prebuilt"


PREBUILT = _prebuilt_dir()
BINARIES = {"readout60": "READOUT60.bin", "node16": "NODE16.bin", "readout32": "READOUT32.bin"}
DFU_ID = "0483:df11"
FLASH_START = "0x08000000"

INSTALL = ("Install one: macOS `brew install dfu-util`; Ubuntu `sudo apt install dfu-util`; "
           "Windows: STM32CubeProgrammer from st.com. See docs/use/flash.md.")

#: Lets a logged-in Linux user open the STM32 bootloader (docs/use/flash.md, Linux).
UDEV_RULE = 'SUBSYSTEMS=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="df11", MODE="0666"'
UDEV_FILE = "/etc/udev/rules.d/50-opentac-dfu.rules"

_NEW_BOARD = (
    "no OpenTac board answered on USB, and no board is waiting in its bootloader (DFU).\n"
    "A new board has no firmware yet, so it cannot restart into the bootloader by itself. "
    "Unplug it, hold its BOOT0 button while you plug it back in, release the button, then {run}\n"
    "If the board already runs OpenTac firmware, check the USB cable, and close any other "
    "program (a viewer) that holds its port.")
#: What `opentac flash` says when nothing answers: most often a board new from the factory.
NEW_BOARD = _NEW_BOARD.format(run="run\n"
                              "  opentac flash --board readout60    (the 60-pin board)\n"
                              "  opentac flash --board node16       (the 30-pin pair)")
NEW_BOARD_BIN = _NEW_BOARD.format(run="run this command again.")

DFU_DENIED = (
    f"a board is waiting in the bootloader (USB {DFU_ID}), but this user may not open it. "
    f"On Linux, allow it with a udev rule, then unplug and replug the board:\n"
    f"  echo '{UDEV_RULE}' | sudo tee {UDEV_FILE}\n"
    f"  sudo udevadm control --reload-rules && sudo udevadm trigger\n"
    f"(docs/use/flash.md, \"On Linux\")")

CUBE_PATHS = [
    "/Applications/STMicroelectronics/STM32Cube/STM32CubeProgrammer/"
    "STM32CubeProgrammer.app/Contents/MacOs/bin/STM32_Programmer_CLI",
    "C:/Program Files/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI.exe",
    str(Path.home() / "STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/STM32_Programmer_CLI"),
]


def find_tool() -> tuple[str, str] | None:
    """('dfu-util', path) or ('cubeprog', path), whichever is installed (dfu-util first)."""
    path = shutil.which("dfu-util")
    if path:
        return "dfu-util", path
    path = shutil.which("STM32_Programmer_CLI")
    if path:
        return "cubeprog", path
    for candidate in CUBE_PATHS:
        if Path(candidate).exists():
            return "cubeprog", candidate
    return None


def flash_command(tool: str, path: str, binary: Path) -> list[str]:
    """The command line that writes `binary` to a board already in DFU."""
    if tool == "dfu-util":
        return [path, "-a", "0", "-d", DFU_ID, "-s", f"{FLASH_START}:leave", "-D", str(binary)]
    if tool == "cubeprog":
        return [path, "-c", "port=USB1", "-d", str(binary), FLASH_START, "-v", "-s", FLASH_START]
    raise ValueError(tool)


def dfu_state(tool: str, path: str) -> str | None:
    """'ready' when a board waits in the bootloader, 'denied' when one does but this
    user may not open it (Linux without the udev rule), else None."""
    if tool == "dfu-util":
        got = subprocess.run([path, "-l"], capture_output=True, text=True)
        if DFU_ID in got.stdout:
            return "ready"
        # dfu-util lists a device it cannot open on stderr:
        # "Cannot open DFU device 0483:df11 found on devnum 7 (LIBUSB_ERROR_ACCESS)"
        if f"Cannot open DFU device {DFU_ID}" in got.stderr:
            return "denied"
        return None
    out = subprocess.run([path, "-l", "usb"], capture_output=True, text=True).stdout
    return "ready" if "USB1" in out else None


def in_dfu(tool: str, path: str) -> bool:
    """Whether a board is waiting in the bootloader."""
    return dfu_state(tool, path) == "ready"


def binary_for(board: str) -> Path:
    """The prebuilt binary for a board, by any of its names (readout60, 60pin, ...)."""
    key = canonical(board).lower()
    if key not in BINARIES:
        raise DeviceError(f"no firmware for board {board!r}; choose one of {', '.join(BINARIES)}")
    path = PREBUILT / BINARIES[key]
    if not path.exists():
        raise DeviceError(f"{path} is missing. In a clone of the repository, restore it with "
                          f"git (or rebuild it: firmware/build.sh --prebuilt); or pass --bin "
                          f"with a binary from firmware/prebuilt/ on GitHub")
    return path


def board_port(port=None) -> str:
    """The board's serial port: one that answers `?`, else the only STM32 USB port.

    A daisy-chain node's firmware answers only `B` on USB, so it is found by
    its USB vendor id instead.
    """
    from serial.tools import list_ports

    from opentac.device.link import find
    if port:
        return port
    try:
        return find("board")
    except DeviceError:
        stm = sorted(p.device for p in list_ports.comports()
                     if p.vid == 0x0483 and not p.device.startswith("/dev/tty."))
        if len(stm) == 1:
            return stm[0]
        raise


def running_board(port=None) -> tuple[str | None, object]:
    """(board name the firmware reports, open link), or (None, link) if it does not say."""
    from opentac.device.link import UsbLink
    link = UsbLink(board_port(port), timeout=2.0)
    try:
        caps = [ln for ln in link.command("?", until="R,END") if ln.startswith("R,CAPS,")]
        return (parse_caps(caps[0])["board"] if caps else None), link
    except DeviceError:
        return None, link


def flash(board: str | None = None, port=None, binary=None, force=False, dry_run=False,
          log=print, wait_s: float = 10.0) -> int:
    """Reboot the attached board into DFU and write its firmware. Returns the tool's exit code.

    `board` takes any name the board goes by: readout60 / 60pin / 60-pin, node16 /
    30pin / 30-pin, readout32 / 70pin. A board with no firmware (new from the
    factory) must be put in its bootloader by hand (BOOT0) and named here.
    """
    if board is not None:
        board = canonical(board).lower()
    tool = find_tool()
    if tool is None:
        if not dry_run:
            raise DeviceError("neither dfu-util nor STM32CubeProgrammer is installed. " + INSTALL)
        log("note: neither dfu-util nor STM32CubeProgrammer is installed. " + INSTALL)
        name, path, state = "dfu-util", "dfu-util", None
    else:
        name, path = tool
        state = dfu_state(name, path)
    if state == "denied":
        raise DeviceError(DFU_DENIED)
    link, absent = None, False
    if state == "ready":
        log("a board is waiting in the bootloader (DFU)")
        if board is None and binary is None:
            raise DeviceError("a board is waiting in the bootloader (DFU), and a board there "
                              "cannot say which board it is: add --board readout60 (the 60-pin "
                              "board) or --board node16 (the 30-pin pair)")
    else:
        try:
            reported, link = running_board(port)
        except NoDeviceError as exc:
            if exc.denied or exc.busy:
                raise
            bus = [p for p, k in exc.found.items() if k == "bus"]
            if bus:
                raise DeviceError(f"only a bus controller answers (on {bus[0]}); flash each "
                                  f"readout board over its own USB cable "
                                  f"(docs/use/daisy-chain.md)") from None
            if not (dry_run and (board or binary)):
                raise DeviceError(NEW_BOARD_BIN if binary else NEW_BOARD) from None
            absent, reported = True, None
        if reported is not None:
            log(f"the board says it is the {label(reported)}")
            if board is None:
                board = reported.lower()
            elif board != reported.lower() and not force:
                link.close()
                raise DeviceError(f"the board says it is the {label(reported)}, not the "
                                  f"{label(board.upper())}; flashing the wrong board's firmware "
                                  f"drives the wrong pins. Pass --force if you are sure.")
    if binary is None:
        if board is None:
            if link is not None:
                link.close()
            raise DeviceError("the board's firmware does not say which board it is: add "
                              "--board readout60 (the 60-pin board) or --board node16 (the "
                              "30-pin pair)")
        binary = binary_for(board)
    binary = Path(binary)
    named = [k for k in BINARIES if k in binary.name.lower()]
    if board and named and board not in named and not force:
        if link is not None:
            link.close()
        raise DeviceError(f"{binary.name} is firmware for the {label(named[0].upper())}, but the "
                          f"board is the {label(board.upper())}. Pass --force if you are sure.")
    cmd = flash_command(name, path, binary)
    if dry_run:
        if link is not None:
            log(f"would send B to {link.port} (reboot into DFU), then run:")
            link.close()
        elif absent:
            log("no board answers on USB and none is in the bootloader. With the board in its "
                "bootloader (hold BOOT0 while plugging it in), this would run:")
        else:
            log("would run:")
        log("  " + " ".join(cmd))
        return 0

    if link is not None:
        log(f"rebooting {link.port} into the bootloader...")
        try:
            link.command("B", match=lambda ln: ln == "R,OK,B")
        except DeviceError:
            pass                     # the reply can be lost as USB goes away
        link.close()
        deadline = time.time() + wait_s
        while (state := dfu_state(name, path)) != "ready":
            if state == "denied":
                raise DeviceError(DFU_DENIED)
            if time.time() > deadline:
                raise DeviceError("the board did not show up in DFU. Unplug it, hold BOOT0 "
                                  "while plugging it back in, and run this again with --board.")
            time.sleep(0.5)
    log("writing " + binary.name + ": " + " ".join(cmd))
    code = subprocess.run(cmd).returncode
    if name == "dfu-util" and code != 0:
        # dfu-util reports "Error during download get_status" after the leave
        # request even when the write went through; the board has already reset.
        log("dfu-util exited with an error after the write; if it printed "
            "'Submitting leave request' the firmware is on the board")
    return code
