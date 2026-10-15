"""Write the reference numbers that test_foam.py and test_transfer.py check against.

The numbers come from the ORIGINAL lab scripts the port was made from
(opentac_indent_compare.py, opentac_transfer.py and the two modules they
import, opentac_force.py and opentac_device.py). Those are not part of this
repository. This script copies them to a temporary folder, imports them from
there, runs them on fixed inputs, and writes:

    tests/sim/fixtures/foam_reference.json
    tests/sim/fixtures/transfer_reference.json

Each case stores its inputs too, so the tests replay exactly what was run.

    uv run python tests/sim/make_reference.py /path/to/tactile_imu_fast_cfg
"""
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FIX = HERE / "fixtures"
SOURCES = ("opentac_indent_compare.py", "opentac_transfer.py", "opentac_force.py", "opentac_device.py")
SKINS = ("flat_square.json", "flat_disc.json", "grid_4mm.json", "grid_2mm.json")
SIGMA, K = 3.0, 0.008659643233600654          # the paper's fitted foam parameters

FOAM_CASES = [
    # skin, geom, centre (None = taxel centroid), total N, sigma mm, angle deg, k N/mm^3
    ("flat_square.json", {"kind": "sphere", "R": 5.0}, None, 5.0, SIGMA, 0.0, K),
    ("flat_square.json", {"kind": "sphere", "R": 10.0}, [2.0, 3.0], 2.0, SIGMA, 0.0, K),
    ("flat_disc.json", {"kind": "punch", "R": 5.0}, None, 8.0, SIGMA, 0.0, K),
    ("flat_disc.json", {"kind": "bar", "R": 5.0, "L": 40.0}, None, 4.0, SIGMA, 30.0, K),
    ("grid_4mm.json", {"kind": "sphere", "R": 25.0}, None, 6.0, SIGMA, 0.0, K),
    ("grid_4mm.json", {"kind": "bar", "R": 5.0, "L": 40.0}, [12.0, 4.0], 1.0, 1.5, 0.0, 0.1),
    ("grid_2mm.json", {"kind": "sphere", "R": 3.0}, [5.0, 8.0], 1.0, SIGMA, 0.0, K),
    ("grid_2mm.json", {"kind": "sphere", "R": 10.0}, None, 8.0, 0.0, 0.0, K),
]

TRANSFER_CASES = [
    # source skin, target skin, placement, kernel
    ("grid_4mm.json", "flat_square.json", {"angle_deg": 0.0, "shift_mm": [0.0, 0.0]}, "box"),
    ("grid_4mm.json", "flat_square.json", {"angle_deg": 30.0, "shift_mm": [3.0, -2.0]}, "box"),
    ("grid_4mm.json", "grid_2mm.json", {"angle_deg": -15.0, "shift_mm": [1.0, 4.0]}, "tent"),
    ("grid_4mm.json", "grid_2mm.json", {"centered": True, "angle_deg": 0.0}, "box"),
    ("flat_square.json", "grid_2mm.json", {"centered": True, "angle_deg": 45.0}, "box"),
    ("grid_2mm.json", "grid_4mm.json", {"centered": True, "angle_deg": 10.0, "target": [10.0, 5.0]}, "box"),
    ("flat_disc.json", "flat_square.json", {"centered": True, "angle_deg": 90.0}, "tent"),
]


def blob(xy, centre, width_mm=6.0, peak_n=0.8):
    """A smooth contact on the source skin, N per taxel."""
    d2 = ((np.asarray(xy) - np.asarray(centre)) ** 2).sum(1)
    return peak_n * np.exp(-d2 / (2 * width_mm ** 2))


def load_originals(src_dir):
    tmp = Path(tempfile.mkdtemp(prefix="opentac_ref_"))
    for name in SOURCES:
        shutil.copy2(Path(src_dir) / name, tmp / name)
    sys.path.insert(0, str(tmp))
    mods = {n[:-3]: importlib.import_module(n[:-3]) for n in SOURCES}
    return mods, tmp


def main(src_dir):
    mods, tmp = load_originals(src_dir)
    ic, ot, of, od = (mods["opentac_indent_compare"], mods["opentac_transfer"],
                      mods["opentac_force"], mods["opentac_device"])
    manifests = {s: json.loads((FIX / s).read_text()) for s in SKINS}

    def layout(skin):
        # Layout only reads xy_mm and pitch_mm from a SkinForce; the pitch is
        # what opentac_force.design_key computes (derive_pitch_um / 1000).
        m = manifests[skin]
        sf = of.SkinForce.__new__(of.SkinForce)
        sf.xy_mm = np.array([p[:2] for p in m["pixels"]], float)
        px_um, py_um = od.derive_pitch_um(m)
        assert px_um == py_um, skin
        sf.pitch_mm = px_um / 1000.0
        sf.n = len(sf.xy_mm)
        return ot.Layout(sf)

    pitch = {s: list(od.derive_pitch_um(manifests[s])) for s in SKINS}

    foam = []
    for skin, geom, centre, total, sigma, angle, k in FOAM_CASES:
        lay = layout(skin)
        c = lay.xy.mean(0) if centre is None else np.asarray(centre, float)
        f = ic.sim_taxel_forces(lay, geom, c, total, sigma, angle_deg=angle, k_n_per_mm3=k)
        foam.append({"skin": skin, "geom": geom, "centre_mm": c.tolist(), "total_n": total,
                     "sigma_mm": sigma, "angle_deg": angle, "k_n_per_mm3": k,
                     "forces_n": f.tolist()})
        print(f"foam  {skin:18s} {geom['kind']:6s} {total:4.1f} N -> sum {f.sum():.4f} N, peak {f.max():.4f} N")

    transfer = []
    for src, dst, place, kernel in TRANSFER_CASES:
        la, lb = layout(src), layout(dst)
        fa = blob(la.xy, la.centroid + np.array([1.7, -0.9]))
        if place.get("centered"):
            tr = ot.Transfer.centered(la, lb, fa, angle_deg=place["angle_deg"],
                                      target=place.get("target"), kernel=kernel)
        else:
            tr = ot.Transfer(la, lb, angle_deg=place["angle_deg"], shift_mm=place["shift_mm"], kernel=kernel)
        fb = tr.force(fa)
        transfer.append({"src": src, "dst": dst, "placement": place, "kernel": kernel,
                         "fa_n": fa.tolist(), "fb_n": fb.tolist(), "lost_n": float(tr.lost(fa)),
                         "kept": tr.kept.tolist(), "shift_mm": tr.shift_mm.tolist(),
                         "about_mm": np.asarray(tr.about, float).tolist(),
                         "placement_cop_mm": ot.placement_cop(la.xy, fa)[0].tolist()})
        print(f"xfer  {src:18s} -> {dst:18s} {kernel}: A {fa.sum():.4f} N, B {fb.sum():.4f} N, "
              f"lost {float(tr.lost(fa)):.4f} N")

    note = ("Reference outputs of the original lab scripts, written by tests/sim/make_reference.py. "
            "Do not edit by hand.")
    (FIX / "foam_reference.json").write_text(json.dumps(
        {"_readme": note, "source": "opentac_indent_compare.sim_taxel_forces", "cases": foam}, indent=1))
    (FIX / "transfer_reference.json").write_text(json.dumps(
        {"_readme": note, "source": "opentac_transfer.Transfer / placement_cop; "
         "opentac_device.derive_pitch_um", "pitch_um": pitch, "cases": transfer}, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {FIX / 'foam_reference.json'} and {FIX / 'transfer_reference.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
