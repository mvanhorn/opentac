#!/usr/bin/env python3
"""Build the data files behind docs/learn/ (the "How it works" page).

    uv run python scripts/learn/build_data.py [--sources DIR]

Everything the page draws comes from here, and everything here comes from a
real source, read-only:

  <sources>/Arduino/stm_test/manifest_*.json            sensor manifests
  <sources>/Arduino/stm_test/recordings/transfer/*/     the 3-skin transfer takes
  <sources>/Arduino/stm_test/tactile_imu_fast_cfg/      firmware: opentac_device.py
        (manifest -> device config), ot_api.h (xampling defaults),
        xampling_m_vs_fps.csv (measured frame rate vs budget)
  <sources>/GitHub/OpenTactileCAD/figure_tactile_transfer/transfer_data.py
        (zeroing, press finding and pose, as used for the transfer figure)
  <sources>/GitHub/TactileSim/tactilesim/profiles/v2.json, v3.json
        (fitted film readout profiles)
  this repo: opentac.manifest, opentac.cad.export.kicad_footprint (taxel
        copper), opentac.cad.pipeline.bump_sheet (bump defaults)

<sources> defaults to $OPENTAC_SOURCES, else the MIT OneDrive folder.

Writes docs/learn/data/meta.json, skins.json and takes/<take>.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from opentac.manifest import load as load_manifest  # noqa: E402
from opentac.cad.export.kicad_footprint import pixel_geometry  # noqa: E402
from opentac.cad.pipeline.bump_sheet import BumpParams  # noqa: E402

DEFAULT_SOURCES = Path.home() / "Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology"
OUT = REPO / "docs" / "learn" / "data"

# Skins the page uses. The first three are the transfer-demo skins (recorded);
# umi and circle add geometry for the bump studies.
SKINS = {
    "leap_palm": ("manifest_leap_palm.json", "LEAP palm"),
    "foot": ("manifest_foot.json", "Quadruped foot"),
    "agileX_2mm": ("manifest_agileX_2mm.json", "AgileX jaw"),
    "umi": ("manifest_umi.json", "UMI finger"),
    "circle": ("manifest_circle.json", "Circle (test design)"),
}
RECORDED = ("leap_palm", "foot", "agileX_2mm")

# Excerpt of each take: the press shown, plus a little air before it.
LEAD_S = 0.25        # before the press onset (transfer_data: total > 15% of max)
TAIL_S = 0.10        # after the press ends ...
PAST_PEAK_S = 0.25   # ... or this long after its peak, whichever is first
ZERO_FRAMES = 64     # firmware XAMP_ZERO_FRAMES default
BASELINE_FRAMES = 16 # firmware BASELINE_FRAMES

NO_SHEAR = 0xFFFF
NONE = 0xFFFF
GRID_SIDE = 32
MAX_GROUP_RADIUS = 2


def r3(a):
    return np.round(np.asarray(a, dtype=float), 3).tolist()


# ----- taxel copper ----------------------------------------------------------

def comb(w, h, fp):
    """Interdigitated taxel copper as line segments, taxel-centred, Y up.

    The same numbers kicad_footprint.build_pixel_footprint_text() writes:
    pad 1 is the row electrode (with the via to the back layer), pad 2 the
    column electrode. KiCad is Y-down, the manifest Y-up, so y is negated.
    """
    tw, gap = fp["trace_w"], fp["gap"]
    g = pixel_geometry(w, h, tw, gap, fp["center_clear"], fp["edge_clear"], fp["via_dia"])
    row = [[g.xL, -g.spine_y_lo, g.xL, -g.spine_y_hi]]
    col = [[g.xR, -g.spine_y_lo, g.xR, -g.spine_y_hi]]
    for i in range(g.total_slots):
        y = -(g.y0 + i * g.pitch)
        if i % 2 == 0:
            row.append([g.xL, y, g.x_end_left_max, y])
        else:
            col.append([g.xR, y, g.col_end_x[i], y])
    return {"trace": tw, "gap": gap, "w": w, "h": h,
            "via": r3([g.via_x, -g.via_y, fp["via_dia"], fp["via_drill"]]),
            "row": r3(row), "col": r3(col), "slots": g.total_slots}


# ----- device config, scan plan and xampling grid ----------------------------
#
# Ports of buildScanPlan() (ot_devcfg.h) and xampBuildGrid() (ot_xampling.h).
# The config itself comes from opentac_device.extract(), the code that
# provisions a real board, with the READOUT32 connector map.

def scan_tables(manifest_raw, device):
    spec = device.extract(manifest_raw, device.readout_32x32_line)
    rows, cols = spec["rows"], spec["cols"]
    row_order, col_order = spec["row_order"], spec["col_order"]
    active = spec["active"]
    shear = spec["shear_groups"] or [NO_SHEAR] * len(active)

    def lines(r, c):
        a, b = row_order[r], col_order[c]
        a_mux, b_mux = bool(a & 0x20), bool(b & 0x20)
        assert a_mux != b_mux
        step = (b if a_mux else a) & 0x1F
        mux = (a if a_mux else b) & 0x1F
        return step, mux

    # buildScanPlan: entries grouped by shift-register step, row-major within.
    by_step = {}
    for taxel, (r, c) in enumerate(active):
        step, mux = lines(r, c)
        by_step.setdefault(step, []).append(taxel)
    entry_out = [t for s in sorted(by_step) for t in by_step[s]]

    # xampBuildGrid: taxel -> grid position from row_geom/col_geom.
    taxel_pos = []
    for r, c in active:
        gr, gc = spec["row_geom"][r], spec["col_geom"][c]
        assert gr < GRID_SIDE and gc < GRID_SIDE
        taxel_pos.append((gr << 5) | gc)
    grid_entry = [NONE] * (GRID_SIDE * GRID_SIDE)
    for e, t in enumerate(entry_out):
        assert grid_entry[taxel_pos[t]] == NONE
        grid_entry[taxel_pos[t]] = e
    entry_pos = [taxel_pos[t] for t in entry_out]

    def count_near(e, group, radius):
        r0, c0 = entry_pos[e] >> 5, entry_pos[e] & 31
        n = 0
        for r in range(r0 - radius, r0 + radius + 1):
            for c in range(c0 - radius, c0 + radius + 1):
                if 0 <= r < GRID_SIDE and 0 <= c < GRID_SIDE:
                    o = grid_entry[(r << 5) | c]
                    if o != NONE and shear[entry_out[o]] == group:
                        n += 1
        return n

    group_radius = 0
    if spec["shear_groups"]:
        group_radius = 1
        members = {}
        for g in shear:
            members[g] = members.get(g, 0) + 1
        for e in range(len(entry_out)):
            g = shear[entry_out[e]]
            if g == NO_SHEAR:
                continue
            if count_near(e, g, group_radius) == members[g]:
                continue
            if count_near(e, g, MAX_GROUP_RADIUS) != members[g]:
                raise SystemExit("shear group too spread out for xampling")
            group_radius = MAX_GROUP_RADIUS
    return {"entryOut": entry_out, "entryPos": entry_pos,
            "groupRadius": group_radius, "shear": shear}


# ----- firmware facts -----------------------------------------------------------

def firmware_defaults(api_h: Path):
    """XAMP_* defaults from the firmware's own parameter table."""
    text = api_h.read_text()
    out = {}
    for name in ("XAMP_M", "XAMP_THRESH", "XAMP_SEARCH_MIN", "XAMP_ZERO_FRAMES",
                 "XAMP_BUMP_FILL", "XAMP_CONFIRM_US"):
        m = re.search(r'"%s",\s*(\d+),\s*(\d+),\s*(\d+)\}' % name, text)
        out[name] = int(m.group(3))
    return out


def pattern_tables(xampling_h: Path):
    """The quadtree search order (XAMP_PATTERN_ROW / _COL), read from the header."""
    text = xampling_h.read_text()
    out = {}
    for key, name in (("row", "XAMP_PATTERN_ROW"), ("col", "XAMP_PATTERN_COL")):
        m = re.search(name + r"\[XAMP_PATTERN_LENGTH\] = \{(.*?)\};", text, re.S)
        out[key] = [int(v) for v in re.findall(r"\d+", m.group(1))]
        assert len(out[key]) == 1024
    return out


def fps_table(csv_path: Path):
    """Mean frame rate per budget over the measurement rounds (32x32 skin)."""
    rows = list(csv.DictReader(open(csv_path)))
    acc = {}
    for r in rows:
        key = "raster" if r["mode"] == "raster" else int(r["M"])
        acc.setdefault(key, []).append(float(r["fps"]))
    return {str(k): round(float(np.mean(v)), 1) for k, v in acc.items()}


# ----- takes ------------------------------------------------------------------

OBJECT_LABEL = {"tape roll": "tape roll", "square": "square block", "allenkey": "allen key"}


def zero_capture(raw, quiet, before):
    """The firmware's `Z`, run on recorded hands-off frames: baseline = integer
    mean of the first 16 frames, noise = each taxel's highest reading minus it."""
    idx = None
    for start in range(before - ZERO_FRAMES, -1, -1):        # latest run first
        if quiet[start:start + ZERO_FRAMES].all():
            idx = np.arange(start, start + ZERO_FRAMES)
            break
    if idx is None:                                           # quietest frames anywhere
        idx = np.sort(np.argsort(raw.sum(1))[:ZERO_FRAMES])
    block = raw[idx].astype(np.int64)
    base = block[:BASELINE_FRAMES].sum(0) // BASELINE_FRAMES
    noise = np.clip(block.max(0) - base, 0, None)
    return base.astype(int), noise.astype(int), int(idx[0])


def build_take(path: Path, td):
    t = td.Take(str(path))
    meta = t.meta
    skin = meta["manifest"]["file"][len("manifest_"):-len(".json")]
    a, b = t.best_press()          # the figure's rule: the hardest press in the take
    k_peak = a + int(np.argmax(t.total[a:b]))
    peak = t.total[k_peak]
    k_half = a + int(np.flatnonzero(t.total[a:k_peak + 1] >= 0.5 * peak)[0])
    centre, angle = t.pose(k_half)

    t0 = t.t[a] - LEAD_S
    t1 = min(t.t[b - 1] + TAIL_S, t.t[k_peak] + PAST_PEAK_S)
    sel = np.flatnonzero((t.t >= t0) & (t.t <= t1))
    lo, hi = int(sel[0]), int(sel[-1]) + 1

    npz = np.load(path / "tactile.npz")
    raw = npz["raw"].astype(int)
    total = raw.sum(1)
    quiet = total <= np.percentile(total, td.QUIET_PCT)
    z_base, z_noise, z_start = zero_capture(raw, quiet, lo)

    fps = (len(t.t) - 1) / (t.t[-1] - t.t[0])
    return skin, {
        "take": path.name,
        "skin": skin,
        "object": OBJECT_LABEL.get(meta["description"], meta["description"]),
        "fps": round(float(fps), 1),
        "frames_in_take": int(len(t.t)),
        "first_frame": lo,
        "t_ms": np.round((t.t[lo:hi] - t.t[lo]) * 1000.0, 2).tolist(),
        "onset": int(a - lo),
        "half": int(k_half - lo),
        "peak": int(k_peak - lo),
        "end": int(min(b, hi) - lo),
        "pose": {"centre": r3(centre), "angle_deg": round(float(np.degrees(angle)), 2)},
        "zero": np.round(t.zero).astype(int).tolist(),
        "z_base": z_base.tolist(),
        "z_noise": z_noise.tolist(),
        "z_first_frame": z_start,
        "raw": raw[lo:hi].tolist(),
    }


# ----- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path,
                    default=Path(os.environ.get("OPENTAC_SOURCES", DEFAULT_SOURCES)))
    args = ap.parse_args()
    src = args.sources
    stm = src / "Arduino" / "stm_test"
    fw = stm / "tactile_imu_fast_cfg"
    sys.path.insert(0, str(fw))
    sys.path.insert(0, str(src / "GitHub" / "OpenTactileCAD" / "figure_tactile_transfer"))
    import opentac_device as device      # noqa: E402
    import transfer_data as td           # noqa: E402

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "takes").mkdir(exist_ok=True)

    # skins
    skins = {}
    for key, (fname, label) in SKINS.items():
        path = stm / fname
        man = load_manifest(path)
        raw = man.raw
        xs = np.unique(np.round(man.xy[:, 0], 3))
        ys = np.unique(np.round(man.xy[:, 1], 3))
        pitch = float(min(np.diff(xs).min(), np.diff(ys).min()))
        shear = raw.get("shear") or {}
        entry = {
            "id": key, "label": label, "manifest": fname,
            "n": man.n, "pitch": round(pitch, 3), "w": man.w, "h": man.h,
            "xy": r3(man.xy), "outline": r3(raw["outline"]),
            "bumps": r3(shear.get("bumps", [])), "b": shear.get("b", []),
            "footprint": raw["footprint"],
            "comb": comb(man.w, man.h, raw["footprint"]),
        }
        if key in RECORDED:
            entry["scan"] = scan_tables(raw, device)
        skins[key] = entry
    (OUT / "skins.json").write_text(json.dumps(skins, separators=(",", ":")))

    # takes
    index = []
    root = stm / "recordings" / "transfer"
    for path in sorted(p for p in root.iterdir() if (p / "take.json").exists()):
        skin, data = build_take(path, td)
        (OUT / "takes" / f"{path.name}.json").write_text(json.dumps(data, separators=(",", ":")))
        index.append({"take": path.name, "skin": skin, "object": data["object"],
                      "fps": data["fps"], "frames": len(data["t_ms"])})
        print(f"  {path.name}: {skin:11s} {data['object']:12s} {len(data['t_ms'])} frames")

    # meta
    bp = BumpParams()
    profiles = {}
    for name in ("v2", "v3"):
        d = json.loads((src / "GitHub" / "TactileSim" / "tactilesim" / "profiles" / f"{name}.json").read_text())
        m = d["model"]
        profiles[name] = {"material": d["material"], "b": m["b"], "k": m["k"], "s": m["s"],
                          "adc_max": m["adc_max"], "noise_counts": m["noise_counts"],
                          "calibrated_kpa": m["calibrated_kpa"]}
    meta = {
        "built_from": "scripts/learn/build_data.py",
        "bump": {"height": bp.bump_height_mm, "base_frac": bp.bump_base_frac,
                 "top_ratio": bp.bump_top_ratio, "sheet": bp.sheet_thickness_mm},
        "profiles": profiles,
        "xampling": {"defaults": firmware_defaults(fw / "ot_api.h"),
                     "fps": fps_table(fw / "xampling_m_vs_fps.csv"),
                     "fps_source": "xampling_m_vs_fps.csv, 32x32 skin, fixed budget, mean of 2 rounds",
                     "pattern": pattern_tables(fw / "ot_xampling.h")},
        "takes": index,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))

    total = sum(f.stat().st_size for f in OUT.rglob("*.json"))
    print(f"wrote {OUT.relative_to(REPO)}: {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
