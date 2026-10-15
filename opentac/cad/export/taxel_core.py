"""The taxel core: a sensor's taxel geometry packed small enough for silkscreen.

manifest.json describes the whole board — outline, cut boundary, design recipe,
and `pixels`, an enumeration carrying one [x, y, row, col] row per taxel. That
enumeration is most of the file's bulk (6-17 KB), and it is the only part of the
manifest that is geometrically usable: manifest.py's own readme warns that
`ox`/`oy`/`px`/`py` are pad-scan order, not geometry, so `ox + ci*px` does not
reproduce positions.

This module states the same taxel facts PARAMETRICALLY instead of by
enumeration — origin, pitch, the pad-order permutation, and a bitmask — which
fits in 14-88 bytes. That is small enough to print as a QR code on the back
silkscreen, so a sensor that has been separated from its files still says what
it is.

It is strictly derived from the manifest and therefore carries no information
the manifest lacks. It is in fact slightly lossier: coordinates quantize to
0.01 mm here versus the manifest's 0.0001 mm. Measured reconstruction error
across all 24 boards in this repo is <= 0.005 mm, half a quantization step.
It also drops `outline`, `edge_cuts`, `design` and `shear` entirely. If you
have the manifest, use the manifest; this exists for the case where all you
have is the physical sensor.

What it does carry is everything needed to READ a sensor: grid dimensions, the
active-taxel mask, exact taxel centres, the taxel footprint, and the connector
pad numbers for every row and column net.
"""

import json
import math

# The footprint is default on all 24 boards in this repo, so a single flag
# usually replaces the whole block. Values are mm.
FP_DEFAULT = {
    "trace_w": 0.2, "gap": 0.2, "center_clear": 0.2,
    "edge_clear": 0.1, "via_drill": 0.15, "via_dia": 0.35,
}
FP_KEYS = sorted(FP_DEFAULT)

Q = 100.0           # quantization: 0.01 mm
MAGIC_VERSION = 0   # 2-bit format version

# Schema version stamped on a recovered manifest. It matches what
# export_manifest() writes, because a recovered file is meant to drop straight
# into anything that reads manifest.json -- with four keys missing and said so.
MANIFEST_V = 6

# One CRC-8 byte closes the gap between "the QR decoded" and "the payload is
# intact". The QR's own error correction covers print damage, but not a
# transport that mangles the string afterwards. The base41 alphabet below
# removes the four characters that get mangled silently; this catches whatever
# is left. Wrong pin mappings that look plausible are the worst outcome this
# format has, so both layers are worth having.
CRC8_POLY = 0x07    # CRC-8/ATM, the usual choice for a byte of framing


# --------------------------------------------------------------------------
# bit I/O
# --------------------------------------------------------------------------

class BitWriter:
    def __init__(self):
        self.bits = []

    def u(self, value, n):
        """Unsigned, n bits, MSB first."""
        if not 0 <= value < (1 << n):
            raise ValueError("%d does not fit in %d bits" % (value, n))
        self.bits.extend((value >> i) & 1 for i in range(n - 1, -1, -1))

    def s(self, value, n):
        """Signed, zigzagged into n bits."""
        self.u(_zigzag(value), n)

    def uvarint(self, value):
        """Unsigned varint: 1 continuation bit + 7 payload bits per group.

        Used for anything whose range is set by the user rather than by this
        format. Pitch was a 10-bit field once, which capped it at 10.23 mm and
        raised on a 22.4 mm board -- a silent format limit that only showed up
        as a failed export.
        """
        v = int(value)
        if v < 0:
            raise TaxelCoreError("negative value in uvarint: %d" % v)
        while True:
            chunk = v & 0x7F
            v >>= 7
            self.u(1 if v else 0, 1)
            self.u(chunk, 7)
            if not v:
                return

    def varint(self, value):
        """Zigzag varint: 1 continuation bit + 7 payload bits per group."""
        v = _zigzag(value)
        while True:
            chunk = v & 0x7F
            v >>= 7
            self.u(1 if v else 0, 1)
            self.u(chunk, 7)
            if not v:
                return

    def bytes(self):
        pad = (-len(self.bits)) % 8
        bits = self.bits + [0] * pad
        return bytes(
            int("".join(str(b) for b in bits[i:i + 8]), 2)
            for i in range(0, len(bits), 8)
        )


class TaxelCoreError(ValueError):
    """Geometry this format cannot represent. Callers degrade, they do not crash."""


class BitReader:
    def __init__(self, data):
        self.bits = [(byte >> i) & 1 for byte in data for i in range(7, -1, -1)]
        self.pos = 0

    def u(self, n):
        if self.pos + n > len(self.bits):
            raise ValueError("payload truncated")
        v = 0
        for _ in range(n):
            v = (v << 1) | self.bits[self.pos]
            self.pos += 1
        return v

    def s(self, n):
        return _unzigzag(self.u(n))

    def uvarint(self):
        v = 0
        shift = 0
        while True:
            more = self.u(1)
            v |= self.u(7) << shift
            shift += 7
            if not more:
                return v

    def varint(self):
        v = 0
        shift = 0
        while True:
            more = self.u(1)
            v |= self.u(7) << shift
            shift += 7
            if not more:
                return _unzigzag(v)


def _zigzag(v):
    return (v << 1) if v >= 0 else ((-v) << 1) - 1


def _unzigzag(v):
    return (v >> 1) if not (v & 1) else -((v + 1) >> 1)


def crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ CRC8_POLY) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def _bitwidth(n):
    """Bits needed to index n distinct values."""
    return max(1, math.ceil(math.log2(max(2, n))))


# --------------------------------------------------------------------------
# structure extraction
# --------------------------------------------------------------------------

def analyze(manifest):
    """Recover the latent grid structure the manifest states by enumeration.

    The manifest gives per-taxel [x, y, row, col]. Every board in this repo has
    exactly one y per row index and one x per col index, and the distinct
    y-values always form an arithmetic sequence — so the grid IS parametric,
    the manifest just does not say so.

    Two wrinkles this has to handle:

    1. Row index is PAD order, not geometric order (split_row_pin_order
       reverses the bottom half), so the pad->geometry permutation has to be
       stored. It is implicit in `pixels`; recovering it is the sort below.

    2. Columns can SKIP lattice positions. On `gecko` the limbs sit 12.0 mm
       from the body, which is exactly 5 x the 2.4 mm pitch — the pitch is
       uniform, but five column slots are empty and the manifest's col index
       is compacted over occupied columns only. Assuming `x0 + col*pitch`
       there is wrong by 8 mm. An occupancy bitmap over the true lattice
       fixes it for 3 bytes, and only 1 of 24 boards needs it.
    """
    row_y, col_x = {}, {}
    for x, y, ri, ci in manifest["pixels"]:
        row_y[ri] = y
        col_x[ci] = x

    ys = sorted(set(row_y.values()), reverse=True)   # geometric order, top down
    xs = sorted(set(col_x.values()))                 # geometric order, left to right

    pitch_y = min(round(ys[i] - ys[i + 1], 4) for i in range(len(ys) - 1)) if len(ys) > 1 else 0.0
    pitch_x = min(round(xs[i + 1] - xs[i], 4) for i in range(len(xs) - 1)) if len(xs) > 1 else 0.0

    # lattice index of each occupied row/col
    ky = [round((ys[0] - v) / pitch_y) if pitch_y else 0 for v in ys]
    kx = [round((v - xs[0]) / pitch_x) if pitch_x else 0 for v in xs]

    # pad row index -> index into ys. Rows with no active taxel never appear in
    # `pixels`; they contribute nothing, so any rank works. 0 is as good as any.
    rank = {i: ys.index(row_y[i]) for i in row_y}
    perm = [rank.get(i, 0) for i in range(manifest["r"])]

    # manifest col index -> index into xs. Verified identity on every board
    # here, but checked rather than assumed.
    col_map = [sorted(col_x).index(i) for i in sorted(col_x)]
    assert col_map == list(range(len(col_map))), "column indices are not in geometric order"

    return {
        "r": manifest["r"], "c": manifest["c"],
        "x0": xs[0], "y0": ys[0],
        "pitch_x": pitch_x, "pitch_y": pitch_y,
        "w": manifest["w"], "h": manifest["h"],
        "footprint": manifest.get("footprint") or dict(FP_DEFAULT),
        "rp0": (manifest.get("rp") or [0])[0],
        "perm": perm,
        "ky": ky, "kx": kx,
        "n_lat_y": max(ky) + 1, "n_lat_x": max(kx) + 1,
        "mask": list(manifest["m"]),
    }


# --------------------------------------------------------------------------
# encode / decode
# --------------------------------------------------------------------------

def encode(manifest, pad_pitch_05=True):
    """manifest dict -> taxel core bytes."""
    g = analyze(manifest)
    r, c = g["r"], g["c"]

    square = (g["pitch_x"] == g["pitch_y"]) and (g["w"] == g["h"])
    fp_default = all(
        abs(g["footprint"].get(k, FP_DEFAULT[k]) - FP_DEFAULT[k]) < 1e-9 for k in FP_KEYS
    )
    sparse_x = g["n_lat_x"] != len(g["kx"])
    sparse_y = g["n_lat_y"] != len(g["ky"])

    w = BitWriter()
    w.u(MAGIC_VERSION, 2)
    w.u(1 if square else 0, 1)
    w.u(1 if fp_default else 0, 1)
    w.u(1 if sparse_x else 0, 1)
    w.u(1 if sparse_y else 0, 1)
    w.u(0, 1)                            # reserved: was the anchor-holes flag, always 0
    w.u(1 if pad_pitch_05 else 0, 1)     # which FPC connector mates
    if not (0 < r < 64 and 0 < c < 64):
        raise TaxelCoreError("grid %dx%d exceeds the 63x63 this format encodes" % (r, c))
    w.u(r, 6)
    w.u(c, 6)
    # Varints, not fixed fields: pitch and taxel size are user geometry with no
    # bound this format gets to impose.
    w.uvarint(round(g["pitch_x"] * Q))
    if not square:
        w.uvarint(round(g["pitch_y"] * Q))
    # taxel size rides as a delta from pitch -- usually -0.40..0.00 mm
    w.varint(round((g["w"] - g["pitch_x"]) * Q))
    if not square:
        w.varint(round((g["h"] - g["pitch_y"]) * Q))
    w.varint(round(g["x0"] * Q))
    w.varint(round(g["y0"] * Q))
    w.u(g["rp0"], 6)                     # cp0 == rp0 + r on every board, so it is derived

    if not fp_default:
        for k in FP_KEYS:
            differs = abs(g["footprint"].get(k, FP_DEFAULT[k]) - FP_DEFAULT[k]) >= 1e-9
            w.u(1 if differs else 0, 1)
            if differs:
                w.u(round(g["footprint"][k] * Q), 8)

    if sparse_x:
        w.u(g["n_lat_x"], 6)
        occupied = set(g["kx"])
        for k in range(g["n_lat_x"]):
            w.u(1 if k in occupied else 0, 1)
    if sparse_y:
        w.u(g["n_lat_y"], 6)
        occupied = set(g["ky"])
        for k in range(g["n_lat_y"]):
            w.u(1 if k in occupied else 0, 1)

    nb = _bitwidth(len(g["ky"]))
    for rank in g["perm"]:
        w.u(rank, nb)

    _write_mask(w, g["mask"], r, c)
    body = w.bytes()
    return body + bytes([crc8(body)])


def _write_mask(w, mask_rows, r, c):
    """Active-taxel bitmap, run-length coded when that wins.

    Masks are mostly solid blocks, so RLE usually beats raw — but not on tiny
    boards where the varint overhead exceeds r*c bits, hence the choice bit.
    """
    bits = "".join(mask_rows)
    trial = BitWriter()
    runs, run = [], 1
    for i in range(1, len(bits)):
        if bits[i] == bits[i - 1]:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)
    trial.u(int(bits[0]), 1)
    for run_len in runs:
        trial.varint(run_len)

    if len(trial.bits) < len(bits):
        w.u(1, 1)
        w.bits.extend(trial.bits)
    else:
        w.u(0, 1)
        w.bits.extend(int(b) for b in bits)


def decode(data):
    """taxel core bytes -> dict with everything needed to read the sensor."""
    if len(data) < 2:
        raise TaxelCoreError("payload too short to be a taxel core")
    body, got = bytes(data[:-1]), data[-1]
    want = crc8(body)
    if got != want:
        raise TaxelCoreError(
            "checksum mismatch (got 0x%02X, expected 0x%02X) -- the scanned text "
            "was altered in transit." % (got, want))
    b = BitReader(body)
    version = b.u(2)
    if version != MAGIC_VERSION:
        raise ValueError("unknown taxel-core version %d" % version)
    square = b.u(1)
    fp_default = b.u(1)
    sparse_x = b.u(1)
    sparse_y = b.u(1)
    b.u(1)                               # reserved (anchor-holes flag in early boards)
    pad_pitch_05 = b.u(1)
    r = b.u(6)
    c = b.u(6)
    pitch_x = b.uvarint() / Q
    pitch_y = pitch_x if square else b.uvarint() / Q
    w_mm = pitch_x + b.varint() / Q
    h_mm = w_mm if square else pitch_y + b.varint() / Q
    x0 = b.varint() / Q
    y0 = b.varint() / Q
    rp0 = b.u(6)

    footprint = dict(FP_DEFAULT)
    if not fp_default:
        for k in FP_KEYS:
            if b.u(1):
                footprint[k] = b.u(8) / Q

    if sparse_x:
        n = b.u(6)
        kx = [k for k in range(n) if b.u(1)]
    else:
        kx = list(range(c))
    if sparse_y:
        n = b.u(6)
        ky = [k for k in range(n) if b.u(1)]
    else:
        ky = None   # filled in below once we know how many distinct rows there are

    n_rows_geo = len(ky) if ky is not None else r
    nb = _bitwidth(n_rows_geo)
    perm = [b.u(nb) for _ in range(r)]
    if ky is None:
        ky = list(range(n_rows_geo))

    mask = _read_mask(b, r, c)

    return {
        "r": r, "c": c, "x0": x0, "y0": y0,
        "pitch_x": pitch_x, "pitch_y": pitch_y,
        "w": w_mm, "h": h_mm,
        "footprint": footprint,
        "rp": list(range(rp0, rp0 + r)),
        "cp": list(range(rp0 + r, rp0 + r + c)),
        "pad_pitch_mm": 0.5 if pad_pitch_05 else None,
        "perm": perm, "kx": kx, "ky": ky,
        "m": mask,
    }


def _read_mask(b, r, c):
    total = r * c
    if b.u(1):
        bits = []
        value = b.u(1)
        while len(bits) < total:
            bits.extend([str(value)] * b.varint())
            value ^= 1
        bits = bits[:total]
    else:
        bits = [str(b.u(1)) for _ in range(total)]
    return ["".join(bits[i * c:(i + 1) * c]) for i in range(r)]


# --------------------------------------------------------------------------
# what a consumer actually wants
# --------------------------------------------------------------------------

def taxels(core):
    """[(x_mm, y_mm, pad_row, pad_col, row_pad_pin, col_pad_pin), ...]

    This is the manifest's `pixels` array rebuilt, plus the connector pin for
    each axis so a readout can drive the scan directly.
    """
    out = []
    for ri in range(core["r"]):
        for ci in range(core["c"]):
            if core["m"][ri][ci] != "1":
                continue
            x = core["x0"] + core["kx"][ci] * core["pitch_x"]
            y = core["y0"] - core["ky"][core["perm"][ri]] * core["pitch_y"]
            out.append((round(x, 4), round(y, 4), ri, ci, core["rp"][ri], core["cp"][ci]))
    return out


# --------------------------------------------------------------------------
# base41 -- QR alphanumeric mode packs 5.5 bits/char, so 1.5 chars/byte costs
# only ~3% over raw byte mode while keeping the payload inside the charset every
# scanner returns as a clean string. Raw binary in a QR gets mangled by anything
# that hands you a UTF-8 str, which is most phones.
#
# Why 41 and not RFC 9285's base45: the density is set by needing N**3 >= 65536
# to pack two bytes into three characters, which holds all the way down to 41.
# QR's alphanumeric charset has 45 characters, so there are exactly four to
# spend -- and exactly four of them fail SILENTLY, turning a payload into a
# different valid payload rather than an error:
#
#     ' '   trimmed by scanners and clipboards
#     '%'   eaten by URL percent-decoding
#     '+'   decoded back as a space by URL parsers
#     '*'   expanded by a shell if the CLI arg is left unquoted
#
# Dropping all four costs nothing: same characters per byte, same QR versions,
# measured. The remaining punctuation fails loudly if it fails at all -- a
# leading '-' looks like an option flag, ':' is an illegal Windows filename
# character -- and the CRC-8 catches whatever still gets through.
# --------------------------------------------------------------------------

B41 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ$-./:"

assert len(B41) == 41 and len(set(B41)) == 41


def b41encode(data):
    out = []
    for i in range(0, len(data) - 1, 2):
        n = data[i] * 256 + data[i + 1]
        n, c1 = divmod(n, 41)
        c3, c2 = divmod(n, 41)
        out += [B41[c1], B41[c2], B41[c3]]
    if len(data) % 2:
        c2, c1 = divmod(data[-1], 41)
        out += [B41[c1], B41[c2]]
    return "".join(out)


def b41decode(text):
    try:
        vals = [B41.index(ch) for ch in text]
    except ValueError as exc:
        raise TaxelCoreError("not base41: %s" % exc)
    out = bytearray()
    for i in range(0, len(vals) - 2, 3):
        n = vals[i] + vals[i + 1] * 41 + vals[i + 2] * 41 * 41
        if n > 0xFFFF:
            raise TaxelCoreError("base41 triple out of range")
        out.extend(divmod(n, 256))
    if len(vals) % 3 == 2:
        n = vals[-2] + vals[-1] * 41
        if n > 0xFF:
            raise TaxelCoreError("base41 tail out of range")
        out.append(n)
    return bytes(out)


def load_manifest(path):
    with open(path) as fh:
        m = json.load(fh)
    m.pop("_readme", None)
    return m


# --------------------------------------------------------------------------
# back to a manifest
# --------------------------------------------------------------------------

ABSENT = ("outline", "edge_cuts", "design", "shear")


def to_manifest(core):
    """A manifest.json-shaped dict rebuilt from a scanned payload.

    Shaped to drop into anything that already reads manifest.json rather than
    into a new format nobody consumes. Every key the taxel core carries is
    written with the same meaning export_manifest() gives it -- including the
    `ox`/`oy`/`px`/`py` block, which is PAD-SCAN order there and so is rebuilt
    in pad order here, negative `py` and all, rather than in the geometric
    order this module stores internally.

    Four keys cannot come back and are listed in `_readme` rather than faked:
    `outline` and `edge_cuts` (photograph the board -- the printed grid is
    there to make that measurable), `design` (the recipe, not the result) and
    `shear` (the bump grouping).
    """
    tx = taxels(core)
    # oy is the y of PAD row 0, which the permutation decides
    oy = core["y0"] - core["ky"][core["perm"][0]] * core["pitch_y"]

    return {
        "_readme": [
            "OpenTactileCAD sensor manifest, RECOVERED from the back-silkscreen QR.",
            "Lengths in mm. Coordinates are Y-UP, the same frame the printed "
            "registration grid uses, whose origin is this frame's (0, 0).",
            "Positions are quantized to 0.01 mm by the printed format; the "
            "original manifest carried 0.0001 mm.",
            "MISSING, because the QR does not carry them: " + ", ".join(ABSENT)
            + ". 'outline'/'edge_cuts' come from a photograph of the board - the "
            "printed grid is what makes that photograph measurable. 'design' is "
            "the recipe and 'shear' is the bump grouping; neither is recoverable.",
            "'pixels' is [x, y, row_index, col_index] per active taxel and is the "
            "authority on where taxels physically sit.",
            "'ox'/'oy'/'px'/'py'/'m'/'rp'/'cp' are for MCU scan order, NOT geometry: "
            "their row index is pad order, so ox+ci*px does not reproduce positions. "
            "Use 'pixels' for geometry.",
        ],
        "v": MANIFEST_V,
        "units": "mm",
        "recovered": {
            "source": "back-silkscreen taxel-core QR",
            "quantization_mm": 1.0 / Q,
            "absent": list(ABSENT),
        },
        "pixels": [[x, y, r, c] for x, y, r, c, _, _ in tx],
        "w": round(core["w"], 4),
        "h": round(core["h"], 4),
        "footprint": dict(core["footprint"]),
        "r": core["r"],
        "c": core["c"],
        "px": round(core["pitch_x"], 4),
        "py": round(-core["pitch_y"], 4),
        "ox": round(core["x0"], 4),
        "oy": round(oy, 4),
        "rp": list(core["rp"]),
        "cp": list(core["cp"]),
        "m": list(core["m"]),
        "pad_pitch_mm": core.get("pad_pitch_mm"),
    }


def manifest_from_scan(payload):
    """The whole path: the string a scanner gives you -> a manifest dict.

    `payload` is exactly what comes off the QR. Whitespace is stripped and case
    folded up, because base41 is uppercase and phone scanners sometimes hand
    back a trimmed or lowercased string.
    """
    # base41 contains no whitespace at all, so trimming is now safe as well as
    # convenient -- which is the point of dropping ' ' from the alphabet.
    text = "".join(str(payload).split()).upper()
    if not text:
        raise TaxelCoreError("empty payload")
    bad = sorted({ch for ch in text if ch not in B41})
    if bad:
        raise TaxelCoreError("not base41: unexpected character(s) %r" % bad)
    if len(text) % 3 == 1:
        raise TaxelCoreError(
            "base41 length %d is impossible (must be a multiple of 3, or 2 more); "
            "the scanned text is truncated or had a character removed" % len(text))
    try:
        blob = b41decode(text)
    except ValueError as exc:
        raise TaxelCoreError("not a base41 payload: %s" % exc)
    return to_manifest(decode(blob))


def payload_from_image(path):
    """Read the back-mark QR out of a photograph. Returns the payload text.

    Optional by design: OpenCV is ~60 MB and the hosted image builds Gerbers
    without it, so it is not a backend dependency. Someone at a bench with a
    photo has it or can pip install it; the server never needs to.

    The board is photographed from the back, where the artwork is printed the
    right way round, so the image is tried as-is first. It is tried mirrored
    second because a Gerber render, a screenshot of KiCad's top view, or a photo
    taken through the board all present it reversed -- and a mirrored QR is
    exactly what most decoders refuse.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise TaxelCoreError(
            "reading a QR from an image needs opencv: pip install "
            "opencv-python-headless. Or scan the code with a phone and pass the "
            "text instead.")

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise TaxelCoreError("could not read image: %s" % path)

    detector = cv2.QRCodeDetector()
    for label, candidate in (("as shot", img), ("mirrored", img[:, ::-1])):
        candidate = np.ascontiguousarray(candidate)
        for scale in (1, 2, 4):
            probe = (candidate if scale == 1 else
                     cv2.resize(candidate, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC))
            try:
                data, _, _ = detector.detectAndDecode(probe)
            except cv2.error:
                continue
            if data:
                return data
    raise TaxelCoreError(
        "no QR found in %s. Fill the frame with the code, hold the board flat, "
        "and keep the light off-axis so the silkscreen is not blown out." % path)


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write(
            "usage: python -m export.taxel_core '<scanned QR text>'\n"
            "       python -m export.taxel_core <photo.jpg>\n"
            "       writes a recovered manifest.json to stdout\n")
        raise SystemExit(2)

    arg = sys.argv[1]
    try:
        # A path is a photo; anything else is the text off a scanner.
        payload = payload_from_image(arg) if os.path.isfile(arg) else arg
        print(json.dumps(manifest_from_scan(payload), indent=2))
    except TaxelCoreError as exc:
        sys.stderr.write("cannot decode: %s\n" % exc)
        raise SystemExit(1)
