"""Read OpenTac sensor manifests: the one reader every other part of the package uses.

A manifest (manifest.json, written by OpenTac CAD) describes one sensor: where each
taxel is, which connector pads its row and column use, and which taxels share a
shear bump. `load()` takes a manifest.json, a design export .zip, or a dict, and
checks that the fields it carries agree with each other. `from_qr()` rebuilds one
from the text of the QR code on the back of a sensor.

Versions 3 to 6 are read. Each added fields and changed none: 4 the `design`
recipe, 5 the `shear` grouping, 6 the `sim3d` block.
"""
from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KNOWN_VERSIONS = (3, 4, 5, 6)
SCAN_KEYS = ("r", "c", "m", "rp", "cp")


class ManifestError(ValueError):
    """The file is not a manifest this package can use."""


@dataclass
class Manifest:
    raw: dict = field(repr=False)
    version: int
    xy: np.ndarray          # (N, 2) taxel centres, mm, Y up, cable edge at the bottom
    row: np.ndarray         # (N,) scan row index of each taxel (pad order, not geometric)
    col: np.ndarray         # (N,) scan column index of each taxel
    w: float                # taxel width, mm
    h: float                # taxel height, mm
    source_bytes: bytes = field(default=b"", repr=False)   # the file as read
    name: str = ""
    warnings: list = field(default_factory=list, repr=False)

    @property
    def n(self) -> int:
        return len(self.xy)

    @property
    def config_id(self) -> int:
        """The id a readout board stores for this skin: CRC-32 of the file's bytes."""
        import zlib
        return zlib.crc32(self.source_bytes) & 0xFFFFFFFF

    @property
    def shear_groups(self) -> np.ndarray | None:
        """Bump index per taxel (-1 = no bump), or None for a pressure-only skin."""
        shear = self.raw.get("shear") or {}
        return np.asarray(shear["b"], dtype=int) if "b" in shear else None

    def get(self, key, default=None):
        return self.raw.get(key, default)

    def grid_index(self) -> tuple[np.ndarray, np.ndarray]:
        """(row, col) of each taxel on the geometric grid: row 0 at the top,
        col 0 at the left. Derived from positions, because the scan indices
        follow connector pads, not geometry."""
        xs = np.unique(np.round(self.xy[:, 0], 3))
        ys = np.unique(np.round(self.xy[:, 1], 3))[::-1]
        col = np.searchsorted(xs, np.round(self.xy[:, 0], 3))
        row = np.searchsorted(-ys, -np.round(self.xy[:, 1], 3))
        return row, col

    def save(self, path) -> Path:
        """Write the manifest exactly as it was read (same bytes, same config id)."""
        path = Path(path)
        path.write_bytes(self.source_bytes)
        return path


def _read_raw(source) -> tuple[dict, bytes, str]:
    if isinstance(source, Manifest):
        return source.raw, source.source_bytes, source.name
    if isinstance(source, dict):
        return source, (json.dumps(source, indent=2) + "\n").encode("utf-8"), ""
    path = Path(source)
    if not path.exists():
        raise ManifestError(f"{path}: no such file")
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as z:
                names = [n for n in z.namelist() if n.endswith("manifest.json")]
                if not names:
                    raise ManifestError(f"{path.name} has no manifest.json")
                data = z.read(sorted(names, key=len)[0])
        except zipfile.BadZipFile:
            raise ManifestError(f"{path.name} is not a zip file") from None
    else:
        data = path.read_bytes()
    try:
        return json.loads(data), data, path.name
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ManifestError(f"{path.name} is not JSON: {exc}") from None


def validate(raw: dict, scan: bool = False) -> list[str]:
    """Check a manifest dict. Raises ManifestError on the first real problem.

    Returns soft warnings. With scan=True the fields a readout board needs
    (r, c, m, rp, cp) must be present; otherwise they are checked only if there.
    """
    warnings = []
    if not isinstance(raw, dict):
        raise ManifestError("not a manifest: the JSON is not an object")
    if "pixels" not in raw:
        raise ManifestError("not a manifest: no 'pixels'")
    v = raw.get("v")
    if v is None:
        warnings.append("no version field 'v'; read as version 3")
    elif not isinstance(v, int) or v < min(KNOWN_VERSIONS):
        raise ManifestError(f"manifest version {v!r} is older than this package reads (3 to 6)")
    elif v > max(KNOWN_VERSIONS):
        warnings.append(f"manifest version {v} is newer than this package knows (3 to 6); "
                        f"reading the fields it knows")

    pixels = raw["pixels"]
    if not isinstance(pixels, list) or not pixels:
        raise ManifestError("'pixels' must be a non-empty list of [x, y, row, col]")
    for k, p in enumerate(pixels):
        if not isinstance(p, (list, tuple)) or len(p) < 4:
            raise ManifestError(f"pixels[{k}] must be [x, y, row, col], got {p!r}")
        if not all(isinstance(a, (int, float)) and math.isfinite(a) for a in p[:2]):
            raise ManifestError(f"pixels[{k}] has a position that is not a finite number: {p!r}")
        if not all(isinstance(a, int) and a >= 0 for a in p[2:4]):
            raise ManifestError(f"pixels[{k}] row/col must be whole numbers >= 0: {p!r}")
    for key in ("w", "h"):
        if key not in raw:
            warnings.append(f"no taxel size '{key}'")
        elif not (isinstance(raw[key], (int, float)) and raw[key] > 0):
            raise ManifestError(f"'{key}' must be a positive number, got {raw[key]!r}")

    missing = [k for k in SCAN_KEYS if k not in raw]
    if missing and (scan or len(missing) < len(SCAN_KEYS)):
        raise ManifestError(f"manifest has no {', '.join(repr(k) for k in missing)}: "
                            f"a readout board cannot scan it")
    if not missing:
        rows, cols, m, rp, cp = (raw[k] for k in SCAN_KEYS)
        if not (isinstance(rows, int) and isinstance(cols, int) and rows >= 1 and cols >= 1):
            raise ManifestError(f"r and c must be whole numbers >= 1, got r={rows!r}, c={cols!r}")
        if not isinstance(m, list) or len(m) != rows:
            raise ManifestError(f"m must hold r = {rows} strings, it has "
                                f"{len(m) if isinstance(m, list) else m!r}")
        for i, line in enumerate(m):
            if not isinstance(line, str) or len(line) != cols or set(line) - {"0", "1"}:
                raise ManifestError(f"m[{i}] must be {cols} characters of 0/1, got {line!r}")
        active = [(r, c) for r in range(rows) for c in range(cols) if m[r][c] == "1"]
        if len(active) != len(pixels):
            raise ManifestError(f"m marks {len(active)} taxels but pixels has {len(pixels)}")
        for k, (p, rc) in enumerate(zip(pixels, active)):
            if (p[2], p[3]) != rc:
                raise ManifestError(
                    f"pixels[{k}] is (row {p[2]}, col {p[3]}) but m, read row by row, puts "
                    f"taxel {k} at (row {rc[0]}, col {rc[1]}): pixels must follow m's order")
        if len(rp) != rows:
            raise ManifestError(f"rp has {len(rp)} pads, r says {rows}")
        if len(cp) != cols:
            raise ManifestError(f"cp has {len(cp)} pads, c says {cols}")
        pads = list(rp) + list(cp)
        if not all(isinstance(p, int) and p >= 0 for p in pads):
            raise ManifestError("rp and cp must hold whole pad numbers")
        if len(set(pads)) != len(pads):
            dup = sorted({p for p in pads if pads.count(p) > 1})
            raise ManifestError(f"pad(s) {dup} used by more than one row/column")

    shear = raw.get("shear")
    if shear is not None:
        b = shear.get("b") if isinstance(shear, dict) else None
        if b is not None:
            if len(b) != len(pixels):
                raise ManifestError(f"shear.b has {len(b)} entries, pixels has {len(pixels)}")
            n_bumps = len(shear.get("bumps", []))
            bad = [g for g in b if not (g == -1 or (isinstance(g, int) and g >= 0
                                                    and (not n_bumps or g < n_bumps)))]
            if bad:
                raise ManifestError(f"shear.b holds bump indices outside -1..{n_bumps - 1}: {bad[:5]}")
    if raw.get("recovered"):
        warnings.append("recovered from a QR code: no outline, edge_cuts, design or shear")
    return warnings


def load(source, scan: bool = False) -> Manifest:
    """A manifest from a manifest.json path, a design export .zip, a dict, or a Manifest.

    Raises ManifestError with the reason if the fields disagree. scan=True also
    requires the fields a readout board needs.
    """
    raw, data, name = _read_raw(source)
    warnings = validate(raw, scan=scan)
    px = np.asarray([p[:4] for p in raw["pixels"]], dtype=float)
    return Manifest(
        raw=raw,
        version=int(raw.get("v", 3) or 3),
        xy=px[:, :2],
        row=px[:, 2].astype(int),
        col=px[:, 3].astype(int),
        w=float(raw.get("w", 0.0)),
        h=float(raw.get("h", 0.0)),
        source_bytes=data,
        name=name,
        warnings=warnings,
    )


def drawn(manifest) -> dict:
    """The layout to draw: a simplified skin's true positions, else the manifest itself.

    A hand-laid skin snapped onto one grid for the readout board keeps where its
    taxels really sit in `pixels_true` (index-parallel to `pixels`, so frame
    order is unchanged) and its real outline in `outline_true`. The board scans
    the grid; drawings and shear directions, which are geometry, use the truth.
    """
    raw = load(manifest).raw
    true = raw.get("pixels_true")
    if not true:
        return raw
    if [p[2:4] for p in true] != [p[2:4] for p in raw["pixels"]]:
        raise ManifestError("pixels_true is not index-parallel to pixels")
    view = dict(raw, pixels=true)
    if "pixel_rot_deg_true" in raw:
        view["pixel_rot_deg"] = raw["pixel_rot_deg_true"]
    if raw.get("outline_true"):
        view["outline"] = raw["outline_true"]
    shear = raw.get("shear")
    if shear and shear.get("bumps"):
        xy = np.array([p[:2] for p in true], dtype=float)
        b = np.asarray(shear["b"])
        view["shear"] = dict(shear, bumps=[xy[b == g].mean(axis=0).tolist() if (b == g).any() else c
                                           for g, c in enumerate(shear["bumps"])])
    return view


def from_qr(payload: str) -> Manifest:
    """A manifest rebuilt from the text of a sensor's back-silkscreen QR code.

    The QR carries the taxel grid, positions, pads and footprint, which is all a
    readout board and the viewer need. It does not carry the outline, the cut
    edge, the design recipe or the shear grouping (see `ABSENT`).
    """
    from opentac.cad.export.taxel_core import TaxelCoreError, manifest_from_scan
    try:
        raw = manifest_from_scan(payload)
    except TaxelCoreError as exc:
        raise ManifestError(f"cannot read the QR text: {exc}") from None
    m = load(raw, scan=True)
    m.name = "recovered from QR"
    return m


def __getattr__(name):
    # The keys a QR-recovered manifest lacks, from the module that writes it.
    if name == "ABSENT":
        from opentac.cad.export.taxel_core import ABSENT
        return ABSENT
    raise AttributeError(name)
