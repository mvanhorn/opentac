"""Serial links: one board on its own USB port, or the bus controller and its nodes.

Both speak the same command lines and get the same replies back; only the
plumbing differs. A bus node runs the single-board firmware built with
-DOT_BUS_NODE, and the Teensy controller forwards `@<n> <line>` to node n and
prefixes what the node writes with `<n>|` (frames with `<n>,<t_us>|`).
"""
from __future__ import annotations

import errno
import re
import sys
import time
from collections import deque

from opentac.device.protocol import DeviceError, explain

BAUD = 115200            # ignored on USB CDC, but pyserial wants a number
#: USB vendor ids: STMicroelectronics (the readout boards' CDC) and PJRC (Teensy).
VENDOR_IDS = {0x0483, 0x16C0}
#: Port names a USB CDC device gets (macOS, Linux), for ports that report no vendor id.
CDC_NAMES = ("usbmodem", "ttyACM")

LINUX_SERIAL_FIX = "sudo usermod -aG dialout $USER, then log out and back in"


class NoDeviceError(DeviceError):
    """No OpenTac board (or bus controller) of the kind asked for answered on USB.

    `found` is port -> what answered there (see probe). `denied` lists ports the
    operating system would not open (Linux: the user is not in the dialout
    group); `busy` lists ports another program holds.
    """

    def __init__(self, message, found=None):
        super().__init__(message)
        self.found = dict(found or {})
        self.denied = [p for p, k in self.found.items() if k == "denied"]
        self.busy = [p for p, k in self.found.items() if k == "busy"]


def candidate_ports() -> list[str]:
    """USB serial ports that could be an OpenTac board or bus controller.

    A port whose USB vendor id is known must be ST's (the readout boards) or
    PJRC's (the Teensy bus controller): probing writes `?` to it, which other
    devices on a robot workstation should never see. Ports that report no
    vendor id (some virtual machines and containers) are tried by name, and
    only when no port matched by vendor id.
    """
    from serial.tools import list_ports
    listed = list(list_ports.comports())
    names = {q.device for q in listed}
    by_vid, unknown = set(), set()
    for p in listed:
        name = p.device
        # macOS lists each port twice; the cu. one does not wait for carrier.
        if name.startswith("/dev/tty.") and name.replace("/dev/tty.", "/dev/cu.") in names:
            continue
        if p.vid is not None:
            if p.vid in VENDOR_IDS:
                by_vid.add(name)
        elif any(tag in name for tag in CDC_NAMES):
            unknown.add(name)
    return sorted(by_vid or unknown)


def _refused(exc) -> str | None:
    """Why a port would not open: 'denied' (permissions), 'busy' (another program has it)."""
    code, text = getattr(exc, "errno", None), str(exc)
    if sys.platform == "win32":
        # Windows says "Access is denied" when another program has the port open.
        return "busy" if ("Access is denied" in text or "PermissionError" in text) else None
    if code in (errno.EACCES, errno.EPERM) or "Permission denied" in text:
        return "denied"
    if code == errno.EBUSY or "Resource busy" in text:
        return "busy"
    return None


def probe(port: str, wait_s: float = 1.0) -> str | None:
    """What answers on `port`: 'board', 'bus', or None (not OpenTac, or no answer).

    'denied' when the operating system will not let this user open the port
    (Linux without the dialout group), 'busy' when another program holds it.
    """
    import serial
    try:
        link = serial.Serial(port, BAUD, timeout=0.1)
    except (OSError, serial.SerialException) as exc:
        return _refused(exc)
    try:
        time.sleep(0.2)
        link.reset_input_buffer()
        link.write(b"?\n")
        buf, deadline = b"", time.time() + wait_s
        while time.time() < deadline:
            buf += link.read(max(1, link.in_waiting))
            if b"R,MCAPS," in buf:
                return "bus"
            if b"R,CAPS," in buf:
                return "board"
    finally:
        link.close()
    return None


def survey() -> dict[str, str | None]:
    """Port -> what answers there (see probe), for every candidate port."""
    return {p: probe(p) for p in candidate_ports()}


def not_found(what: str, found: dict, hint: str = "") -> NoDeviceError:
    """The error for "no OpenTac <what> on USB", naming ports that would not open."""
    denied = [p for p, k in found.items() if k == "denied"]
    busy = [p for p, k in found.items() if k == "busy"]
    if denied:
        them = "it" if len(denied) == 1 else "them"
        msg = (f"found {', '.join(denied)} but cannot open {them} (permission denied): "
               f"{LINUX_SERIAL_FIX}")
    elif busy:
        msg = (f"found {', '.join(busy)} but another program has it open (a viewer, a serial "
               f"monitor, another opentac command): close it and try again")
    elif hint:
        msg = f"no OpenTac {what} found on USB.{hint}"
    else:
        msg = (f"no OpenTac {what} found on USB. Is it plugged in (with a data cable), and is "
               f"no other program (a viewer) holding its port?")
        if "board" in what:
            msg += (" A new board has no firmware yet: put it on with `opentac flash` "
                    "(https://opentac.github.io/use/flash/).")
    return NoDeviceError(msg, found)


def find(kind: str) -> str:
    """The one port of `kind` ('board' or 'bus'); a clear error if none or several."""
    found = survey()
    match = [p for p, k in found.items() if k == kind]
    if len(match) == 1:
        return match[0]
    what = "readout board" if kind == "board" else "bus controller"
    if not match:
        other = [p for p, k in found.items() if k in ("board", "bus")]
        hint = ""
        if kind == "board" and other:
            hint = f" A bus controller answers on {other[0]}: pass node=N (CLI: --node N)."
        raise not_found(what, found, hint)
    raise DeviceError(f"several {what}s found: {', '.join(match)}. Pass port= (CLI: --port).")


def find_any() -> tuple[str, str]:
    """('board', port) for a single board on USB, else ('bus', port) for a bus controller."""
    found = survey()
    for kind in ("board", "bus"):
        match = [p for p, k in found.items() if k == kind]
        if len(match) == 1:
            return kind, match[0]
        if len(match) > 1:
            raise DeviceError(f"several OpenTac {kind}s found: {', '.join(match)}. "
                              f"Pass port= (CLI: --port).")
    raise not_found("readout board or bus controller", found)


class Transport:
    """Command/reply over a serial link. Replies are matched, not counted."""

    timeout = 5.0

    def _open(self, port):
        import serial
        try:
            self.link = serial.Serial(port, BAUD, timeout=0.2)
        except (OSError, serial.SerialException) as exc:
            why = _refused(exc)
            if why == "denied":
                raise DeviceError(f"cannot open {port} (permission denied): "
                                  f"{LINUX_SERIAL_FIX}") from None
            if why == "busy":
                raise DeviceError(f"cannot open {port}: another program has it open (a viewer, "
                                  f"a serial monitor); close it and try again") from None
            raise DeviceError(f"cannot open {port}: {exc}") from None
        self.port = port
        self._sent = ""
        self._buf = b""
        time.sleep(0.3)          # let the CDC endpoint settle
        self.link.reset_input_buffer()

    def _read_chunked(self) -> str:
        """One line, reading whatever is waiting rather than a byte at a time.

        pyserial's readline asks for one byte per call: at ~190 frames/s of
        ~1 KB lines that is four times the CPU of reading in chunks.
        """
        while b"\n" not in self._buf:
            chunk = self.link.read(max(1, self.link.in_waiting))
            if not chunk:
                return ""
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line.decode("ascii", "replace").strip()

    def readline(self, node=None) -> str:
        raise NotImplementedError

    def _read_replies(self, node, until=None, match=None) -> list[str]:
        replies, deadline = [], time.time() + self.timeout
        while time.time() < deadline:
            line = self.readline(node)
            if not line.startswith("R,"):
                continue
            if line.startswith("R,ERR,"):
                raise DeviceError(explain(line.split(",", 2)[2]))
            replies.append(line)
            if match is not None:
                if match(line):
                    return replies
                if line == "R,END":
                    raise DeviceError(f"no matching reply to {self._sent!r}")
                continue
            if until is None or line == until:
                return replies
        raise DeviceError(f"no reply to {self._sent!r} in {self.timeout:.0f} s")

    def reset_input_buffer(self):
        self._buf = b""
        self.link.reset_input_buffer()

    def close(self):
        self.link.close()


class UsbLink(Transport):
    """One board on its own USB port."""

    def __init__(self, port=None, timeout=5.0, link=None):
        self.timeout = timeout
        if link is None:
            self._open(port or find("board"))
        else:                       # a stand-in serial object, for tests
            self.link, self.port, self._sent, self._buf = link, "test", "", b""

    def readline(self, node=None) -> str:
        return self._read_chunked()

    def command(self, text, node=None, until=None, match=None):
        self._sent = text
        self.link.write((text + "\n").encode("ascii"))
        self.link.flush()
        return self._read_replies(node, until, match)


class BusLink(Transport):
    """The bus controller (Teensy 4.0) and the nodes behind it.

    Lines are sorted into a queue per node as they arrive, so a command to one
    node can wait for its reply while the others keep streaming. Node None is
    the controller itself. Not thread-safe: talk to the nodes first, then read
    frames from one thread.
    """

    NODE_LINE = re.compile(r"^(\d+)(?:,(\d+))?\|(.*)$")
    QUEUE_LINES = 4096

    def __init__(self, port=None, timeout=5.0, link=None):
        self.timeout = timeout
        if link is None:
            self._open(port or find("bus"))
        else:                       # a stand-in serial object, for tests
            self.link, self.port, self._sent, self._buf = link, "test", "", b""
        self._queues = {}
        self.last_t_us = {}
        self.events = deque(maxlen=256)     # J (joined) / L (lost) records

    def _queue(self, node):
        q = self._queues.get(node)
        if q is None:
            q = self._queues[node] = deque(maxlen=self.QUEUE_LINES)
        return q

    def _pump(self):
        line = self._read_chunked()
        if not line:
            return None
        m = self.NODE_LINE.match(line)
        if m:
            return int(m.group(1)), (int(m.group(2)) if m.group(2) else None), m.group(3)
        if line.startswith(("J,", "L,")):
            self.events.append(line)
        return None, None, line

    def readline(self, node=None) -> str:
        q = self._queue(node)
        deadline = time.time() + 0.2
        while not q:
            if time.time() >= deadline:
                return ""
            got = self._pump()
            if got is not None:
                self._queue(got[0]).append(got[1:])
        t_us, line = q.popleft()
        self.last_t_us[node] = t_us
        return line

    def readline_any(self):
        """The next line from anyone, as (node, t_us, text); None on a quiet port."""
        for node, q in self._queues.items():
            if q:
                return (node,) + q.popleft()
        return self._pump()

    def reset_input_buffer(self):
        super().reset_input_buffer()
        self._queues.clear()

    def command(self, text, node=None, until=None, match=None):
        self._sent = text if node is None else f"@{node} {text}"
        self.link.write((self._sent + "\n").encode("ascii"))
        self.link.flush()
        return self._read_replies(node, until, match)

    def list_nodes(self) -> list[dict]:
        """Every slot the controller knows, online or not."""
        nodes = []
        for line in self.command("N", node=None, until="R,END"):
            f = line.split(",")
            if f[1] == "NODE":
                nodes.append({"node": int(f[2]), "addr": int(f[3], 16), "uid": f[4],
                              "board": f[5], "online": f[6] == "1"})
        return nodes
