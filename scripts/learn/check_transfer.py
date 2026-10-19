#!/usr/bin/env python3
"""Check docs/learn/js/transfer.js against opentac.transfer.

    uv run python scripts/learn/check_transfer.py

Builds the transfer matrix M (A taxels -> B taxels) with the Python Transfer
class (the package's own, opentac/transfer.py) and with the page's JS port, for every ordered pair of the three
recorded skins and a few placements (rotation, shift), and requires them to
agree to 1e-9. Needs node. Run build_data.py first.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SITE_ROOT = Path(os.environ.get("OPENTAC_SITE_ROOT", REPO.parent / "opentac.github.io")).resolve()
DATA = SITE_ROOT / "docs" / "learn" / "data"
JS = SITE_ROOT / "docs" / "learn" / "js" / "transfer.js"

NODE_DRIVER = r"""
const T = require(process.argv[2]);
const job = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const out = job.cases.map(c => {
  const a = new T.Layout(job.skins[c.a].xy, job.skins[c.a].pitch);
  const b = new T.Layout(job.skins[c.b].xy, job.skins[c.b].pitch);
  const tr = new T.Transfer(a, b, c.angle, c.shift, c.about);
  const M = [];
  tr.rows.forEach((row, i) => row.forEach(([j, w]) => M.push([j, i, w])));
  return M;
});
process.stdout.write(JSON.stringify(out));
"""


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    sys.path.insert(0, str(REPO))
    from opentac.transfer import Layout, Transfer  # noqa: E402

    skins = json.loads((DATA / "skins.json").read_text())
    names = ["leap_palm", "foot", "agileX_2mm"]

    def layout(name):
        return Layout.from_xy(skins[name]["xy"], skins[name]["pitch"])

    cases = []
    for a in names:
        for b in names:
            ca = np.mean(skins[a]["xy"], axis=0)
            cb = np.mean(skins[b]["xy"], axis=0)
            for angle, extra in ((0.0, (0, 0)), (37.0, (3.3, -5.1)), (-118.5, (-12.0, 20.0))):
                cases.append({"a": a, "b": b, "angle": angle, "about": ca.tolist(),
                              "shift": (cb - ca + np.array(extra)).tolist()})
    with tempfile.TemporaryDirectory() as tmp:
        driver = Path(tmp) / "driver.js"
        driver.write_text(NODE_DRIVER)
        job = Path(tmp) / "job.json"
        job.write_text(json.dumps({"skins": {n: skins[n] for n in names}, "cases": cases}))
        js = json.loads(subprocess.run(["node", str(driver), str(JS), str(job)], check=True,
                                       capture_output=True, text=True).stdout)
    worst = 0.0
    for c, entries in zip(cases, js):
        la, lb = layout(c["a"]), layout(c["b"])
        M = Transfer(la, lb, angle_deg=c["angle"], shift_mm=c["shift"], about=c["about"]).M
        Mj = np.zeros_like(M)
        for j, i, w in entries:
            Mj[j, i] += w
        err = float(np.abs(M - Mj).max())
        worst = max(worst, err)
        if err > 1e-9:
            print(f"MISMATCH {c['a']} -> {c['b']} angle {c['angle']}: max |dM| {err:.3g}")
            sys.exit(1)
    print(f"OK: {len(cases)} placements, JS transfer matrix == opentac.transfer.Transfer (max |dM| {worst:.1e})")


if __name__ == "__main__":
    main()
