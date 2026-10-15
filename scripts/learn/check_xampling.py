#!/usr/bin/env python3
"""Check docs/learn/js/xampling.js against the firmware's own ot_xampling.h.

    uv run python scripts/learn/check_xampling.py [--sources DIR]

Needs node and a C++ compiler. For every recorded take on the page and every
budget the page offers, it replays the take through

  - the firmware engine: ot_xampling.h compiled on the host (xamp_replay.cpp),
    with the device config the board would be provisioned with
    (opentac_device.extract + build_config, READOUT32 connector map), and
  - the page's JS port, on the same frame schedule,

and requires the taxels read and the read tallies to be identical every frame.
Run build_data.py first.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "docs" / "learn" / "data"
JS = REPO / "docs" / "learn" / "js" / "xampling.js"
DEFAULT_SOURCES = Path.home() / "Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology"
BUDGETS = (16, 32, 64, 128, 256, 512)

NODE_DRIVER = r"""
const X = require(process.argv[2]);
const job = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const grid = new X.Grid(job.scan);
const idx = X.schedule(job.t_ms, job.fps);
const res = X.replay(grid, job.settings, job.z_base, job.z_noise, idx.length,
                     k => job.raw[idx[k]], job.pattern);
const lines = res.map(r => {
  const t = []; for (let i = 0; i < r.sampled.length; i++) if (r.sampled[i]) t.push(i);
  return `${r.fromPriority} ${r.fromPattern} ${r.fromGrowth} |` + t.map(v => ' ' + v).join('');
});
process.stdout.write(JSON.stringify({idx, lines}));
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path,
                    default=Path(os.environ.get("OPENTAC_SOURCES", DEFAULT_SOURCES)))
    args = ap.parse_args()
    stm = args.sources / "Arduino" / "stm_test"
    fw = stm / "tactile_imu_fast_cfg"
    sys.path.insert(0, str(fw))
    import opentac_device as device  # noqa: E402

    meta = json.loads((DATA / "meta.json").read_text())
    skins = json.loads((DATA / "skins.json").read_text())
    xm = meta["xampling"]
    d = xm["defaults"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        exe = tmp / "xamp_replay"
        subprocess.run(["c++", "-std=c++17", "-O2", "-DOTCFG_NO_FLASH", f"-I{fw}",
                        str(Path(__file__).with_name("xamp_replay.cpp")), "-o", str(exe)], check=True)
        driver = tmp / "driver.js"
        driver.write_text(NODE_DRIVER)

        frames_checked = 0
        for entry in meta["takes"]:
            take = json.loads((DATA / "takes" / f"{entry['take']}.json").read_text())
            skin = skins[take["skin"]]
            manifest = json.loads((stm / skin["manifest"]).read_text())
            cfg = tmp / "config.bin"
            cfg.write_bytes(device.build_config(device.extract(manifest, device.readout_32x32_line), 1))
            for mode, m in [(mode, m) for mode in (1, 2) for m in BUDGETS]:
                job = {"scan": skin["scan"], "t_ms": take["t_ms"], "fps": xm["fps"][str(m)],
                       "settings": {"mode": mode, "m": m, "threshold": d["XAMP_THRESH"],
                                    "searchMin": d["XAMP_SEARCH_MIN"], "bumpFill": False},
                       "z_base": take["z_base"], "z_noise": take["z_noise"],
                       "raw": take["raw"], "pattern": xm["pattern"]}
                jf = tmp / "job.json"
                jf.write_text(json.dumps(job))
                js = json.loads(subprocess.run(["node", str(driver), str(JS), str(jf)],
                                               check=True, capture_output=True, text=True).stdout)
                n = len(take["z_base"])
                lines = [f"{n} {len(js['idx'])} {mode} {m} {d['XAMP_SEARCH_MIN']} {d['XAMP_THRESH']}",
                         " ".join(map(str, take["z_base"])), " ".join(map(str, take["z_noise"]))]
                lines += [" ".join(map(str, take["raw"][i])) for i in js["idx"]]
                ff = tmp / "frames.txt"
                ff.write_text("\n".join(lines) + "\n")
                c = subprocess.run([str(exe), str(cfg), str(ff)], check=True,
                                   capture_output=True, text=True).stdout.strip().split("\n")
                bad = [k for k, (a, b) in enumerate(zip(c, js["lines"])) if a.strip() != b.strip()]
                if len(c) != len(js["lines"]) or bad:
                    k = bad[0] if bad else -1
                    print(f"MISMATCH {entry['take']} mode={mode} M={m} frame {k}:\n  C : {c[k][:200]}\n  JS: {js['lines'][k][:200]}")
                    sys.exit(1)
                frames_checked += len(c)
            print(f"  {entry['take']} ({take['skin']}): identical in fixed and grow mode for M = {', '.join(map(str, BUDGETS))}")
        print(f"OK: {frames_checked} xampling frames, JS port == ot_xampling.h")


if __name__ == "__main__":
    main()
