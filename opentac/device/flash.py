"""Put firmware on a readout board over USB, with no BOOT0 button.

The running firmware reboots into the STM32's own USB bootloader (DFU) when it
gets the line `B`. Then dfu-util, or STMicroelectronics' STM32CubeProgrammer,
writes the binary from the start of flash. Both write only the pages the
binary covers, so the skin config near the top of flash survives (checked with
dfu-util on hardware; README 2026-09-10).
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from opentac.device.protocol import DeviceError, parse_caps

PREBUILT = Path(__file__).resolve().parents[2] / "firmware" / "prebuilt"
BINARIES = {"readout60": "READOUT60.bin", "node16": "NODE16.bin", "readout32": "READOUT32.bin"}
DFU_ID = "0483:df11"
FLASH_START = "0x08000000"

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


def in_dfu(tool: str, path: str) -> bool:
    """Whether a board is waiting in the bootloader."""
    if tool == "dfu-util":
        out = subprocess.run([path, "-l"], capture_output=True, text=True).stdout
        return DFU_ID in out
    out = subprocess.run([path, "-l", "usb"], capture_output=True, text=True).stdout
    return "USB1" in out


def binary_for(board: str) -> Path:
    key = board.lower()
    if key not in BINARIES:
        raise DeviceError(f"no firmware for board {board!r}; choose one of {', '.join(BINARIES)}")
    path = PREBUILT / BINARIES[key]
    if not path.exists():
        raise DeviceError(f"{path} is missing: run firmware/build.sh --prebuilt, or pass --bin")
    return path


def running_board(port=None) -> tuple[str | None, object]:
    """(board name the firmware reports, open link), or (None, link) if it does not say."""
    from opentac.device.link import UsbLink
    link = UsbLink(port)
    try:
        caps = [ln for ln in link.command("?", until="R,END") if ln.startswith("R,CAPS,")]
        return (parse_caps(caps[0])["board"] if caps else None), link
    except DeviceError:
        return None, link


def flash(board: str | None = None, port=None, binary=None, force=False, dry_run=False,
          log=print, wait_s: float = 10.0) -> int:
    """Reboot the attached board into DFU and write its firmware. Returns the tool's exit code."""
    tool = find_tool()
    if tool is None:
        raise DeviceError("neither dfu-util nor STM32CubeProgrammer is installed: see "
                          "docs/use/flash.md (macOS: brew install dfu-util)")
    name, path = tool
    link = None
    if not in_dfu(name, path):
        reported, link = running_board(port)
        if reported is not None:
            log(f"board reports {reported}")
            if board is None:
                board = reported
            elif board.lower() != reported.lower() and not force:
                link.close()
                raise DeviceError(f"the board says it is {reported}, not {board}; flashing the "
                                  f"wrong board's firmware drives the wrong pins. Pass --force "
                                  f"if you are sure.")
    elif dry_run:
        log("a board is already waiting in DFU")
    if binary is None:
        if board is None:
            if link is not None:
                link.close()
            raise DeviceError("cannot tell which board this is: pass --board readout60|node16|readout32")
        binary = binary_for(board)
    binary = Path(binary)
    cmd = flash_command(name, path, binary)
    if dry_run:
        log(f"would send B to {getattr(link, 'port', 'the board')} (reboot into DFU), then run:")
        log("  " + " ".join(cmd))
        if link is not None:
            link.close()
        return 0

    if link is not None:
        log(f"rebooting {link.port} into the bootloader...")
        try:
            link.command("B", match=lambda ln: ln == "R,OK,B")
        except DeviceError:
            pass                     # the reply can be lost as USB goes away
        link.close()
        deadline = time.time() + wait_s
        while not in_dfu(name, path):
            if time.time() > deadline:
                raise DeviceError("the board did not show up in DFU. Unplug it, hold BOOT0 "
                                  "while plugging it back in, and run this again.")
            time.sleep(0.5)
    log("writing " + binary.name + ": " + " ".join(cmd))
    code = subprocess.run(cmd).returncode
    if name == "dfu-util" and code != 0:
        # dfu-util reports "Error during download get_status" after the leave
        # request even when the write went through; the board has already reset.
        log("dfu-util exited with an error after the write; if it printed "
            "'Submitting leave request' the firmware is on the board")
    return code
