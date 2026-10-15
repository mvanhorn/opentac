"""Read OpenTac sensor manifests: the one reader every other part of the package uses.

A manifest (manifest.json, written by OpenTac CAD) describes one sensor: where each
taxel is, which connector pads its row and column use, and which taxels share a
shear bump. `load()` takes a manifest.json, a design export .zip, or a dict.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


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

    @property
    def n(self) -> int:
        return len(self.xy)

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


def _read_raw(source) -> dict:
    if isinstance(source, dict):
        return source
    path = Path(source)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith("manifest.json")]
            if not names:
                raise ManifestError(f"{path.name} has no manifest.json")
            return json.loads(z.read(sorted(names, key=len)[0]))
    return json.loads(path.read_text(encoding="utf-8"))


def load(source) -> Manifest:
    """A manifest from a manifest.json path, a design export .zip, or a dict."""
    raw = _read_raw(source)
    if "pixels" not in raw:
        raise ManifestError("not a manifest: no 'pixels'")
    px = np.asarray(raw["pixels"], dtype=float)
    if px.ndim != 2 or px.shape[1] < 4:
        raise ManifestError("'pixels' must be a list of [x, y, row, col]")
    return Manifest(
        raw=raw,
        version=int(raw.get("v", 0) or 0),
        xy=px[:, :2],
        row=px[:, 2].astype(int),
        col=px[:, 3].astype(int),
        w=float(raw.get("w", 0.0)),
        h=float(raw.get("h", 0.0)),
    )
