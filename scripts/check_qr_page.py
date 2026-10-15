#!/usr/bin/env python3
"""Check the website's QR decoder (docs/javascripts/qr.js) against opentac.manifest.from_qr.

    uv run python scripts/check_qr_page.py

Needs node. The page promises that its "Download manifest.json" is the same
file as `opentac manifest from-qr TEXT -o manifest.json`. For every example
design (docs/designs/files/*.zip) and every OpenTac-CAD manifest in
examples/manifests/, this encodes the taxel core and decodes it both ways, as
the bare payload and as the printed link in the forms a scanner or a person can
hand over (upper and lower case, with and without https://, '*' or '/', the
decoder-page address), and requires byte-identical output. It also feeds both
sides random one-character corruptions and requires them to refuse the same
ones, and compares the JS port of Python's float formatting and round() on
values chosen to hit their edge cases.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JS = REPO / "docs" / "javascripts" / "qr.js"
DESIGNS = REPO / "docs" / "designs" / "files"
MANIFESTS = REPO / "examples" / "manifests"
# Laid out by another tool with the column pads below the row pads, which the
# QR format cannot state (taxel_core: cp0 == rp0 + r); see tests/readout/test_manifest.py.
NOT_CAD = {"pinky_tip_30pin.json"}
MUTATIONS = 40   # per design

NODE_DRIVER = r"""
const Q = require(process.argv[2]);
const job = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const texts = job.texts.map(t => { try { return {ok: Q.fromText(t).json}; } catch (e) { return {err: e.message}; } });
const repr = job.floats.map(x => Q.floatRepr(x));
const rounds = job.rounds.map(x => Q.floatRepr(Q.pyRound(x, 4)));
process.stdout.write(JSON.stringify({texts, repr, rounds}));
"""


def sources():
    """(name, manifest dict) for every board that carries a QR."""
    from opentac.cad.export.taxel_core import load_manifest
    out = []
    for z in sorted(DESIGNS.glob("*.zip")):
        man = json.loads(zipfile.ZipFile(z).read("manifest.json"))
        man.pop("_readme", None)
        out.append((z.stem, man))
    for p in sorted(MANIFESTS.glob("*.json")):
        if p.name not in NOT_CAD:
            out.append((p.name, load_manifest(p)))
    return out


def texts_for(payload, rng):
    from opentac.cad.export.taxel_core import B41, qr_link
    link = qr_link(payload)
    forms = [payload, payload.lower(), " \n" + payload + "\t ", link, link.lower(),
             link[len("HTTPS://"):], "  " + link[len("HTTPS://"):].lower() + "\r\n",
             link.replace("*", "/"), "http://opentac.github.io/q/" + payload,
             "https://opentac.github.io/qr/#" + payload]
    for _ in range(MUTATIONS):
        i = rng.randrange(len(payload))
        forms.append(payload[:i] + rng.choice([c for c in B41 if c != payload[i]]) + payload[i + 1:])
    for i in (0, len(payload) // 2, len(payload) - 1):   # a dropped character
        forms.append(payload[:i] + payload[i + 1:])
    return forms


def python_side(text):
    from opentac import manifest
    try:
        return {"ok": manifest.from_qr(text).source_bytes.decode("utf-8")}
    except manifest.ManifestError as exc:
        return {"err": str(exc)}


def edge_floats(rng):
    vals = [0.0, -0.0, 1.0, -1.0, 0.5, 0.01, 0.1, 0.2, 0.35, 1e-4, 1.5e-4, 9.99e-5, 1e-5, -2.5e-6,
            1e15, 1e16, 1.5e16, 9.999999999999998e15, 1e21, 1e22, 123456789.125, 2.675, -0.0001,
            5e-324, 1.7976931348623157e308, 0.30000000000000004]
    vals += [rng.uniform(-60, 60) for _ in range(300)]
    vals += [rng.randint(-6000, 6000) / 100 + rng.choice((0, 1e-12, -1e-12)) for _ in range(300)]
    return vals


def round_cases(rng):
    # Exact ties at 4 decimals are odd multiples of 1/32 (x * 10**4 has to end in .5
    # exactly); Python rounds those to even, JS's toFixed rounds them up.
    vals = [k / 32 for k in range(-64, 65)] + [0.00005, 0.00015, -0.00005, 2.675, 1.00005]
    vals += [rng.randint(-600000, 600000) / 10000 + rng.choice((0, 0.00005, -0.00005)) for _ in range(400)]
    vals += [a / 100 + k * b / 100 for a, b, k in
             ((rng.randint(-3000, 3000), rng.randint(50, 600), rng.randint(0, 40)) for _ in range(400))]
    return vals


def main() -> int:
    rng = random.Random(1)
    from opentac.cad.export.taxel_core import b41encode, encode
    cases = []
    for name, man in sources():
        payload = b41encode(encode(man))
        cases += [(name, t) for t in texts_for(payload, rng)]
    floats, rounds = edge_floats(rng), round_cases(rng)

    with tempfile.TemporaryDirectory() as tmp:
        driver, job = Path(tmp) / "driver.js", Path(tmp) / "job.json"
        driver.write_text(NODE_DRIVER)
        job.write_text(json.dumps({"texts": [t for _, t in cases], "floats": floats, "rounds": rounds}))
        res = json.loads(subprocess.run(["node", str(driver), str(JS), str(job)], check=True,
                                        capture_output=True, text=True).stdout)

    bad = 0
    for x, got in zip(floats, res["repr"]):
        if got != repr(x):
            print(f"MISMATCH float repr {x!r}: JS {got}")
            bad += 1
    for x, got in zip(rounds, res["rounds"]):
        if got != repr(round(x, 4)):
            print(f"MISMATCH round({x!r}, 4) = {round(x, 4)!r}: JS {got}")
            bad += 1

    decoded = refused = 0
    designs = set()
    for (name, text), js in zip(cases, res["texts"]):
        py = python_side(text)
        if ("ok" in py) != ("ok" in js) or py.get("ok") != js.get("ok"):
            print(f"MISMATCH {name} {text!r}:\n  python: {str(py)[:200]}\n  js:     {str(js)[:200]}")
            bad += 1
        elif "ok" in py:
            decoded += 1
            designs.add(name)
        else:
            refused += 1
    if bad:
        print(f"FAILED: {bad} mismatches")
        return 1
    print(f"OK: {len(designs)} boards, {decoded} texts decoded to byte-identical manifest.json, "
          f"{refused} corrupted texts refused by both; {len(floats)} float reprs and "
          f"{len(rounds)} round() calls identical")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    raise SystemExit(main())
