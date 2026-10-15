"""Regenerate the ready-to-order example designs for the website.

    uv run python scripts/make_example_designs.py [--only slug,slug] [--workers N]

Reads examples/designs/sources.json (the 31 sensors fabricated for the paper),
generates each one for an OpenTac readout board with the current pipeline, and
writes, for the site:

    docs/designs/files/<slug>.zip   the export: KiCad, Gerbers, manifest, design, PDFs
    docs/designs/files/<slug>.svg   the board preview
    docs/designs/designs.json       what the examples page lists
    docs/design/examples.md         the examples page

A design whose fabricated grid has more than 30 rows or columns does not fit the
60-pin readout board; its pitch is raised in small steps (taxel size scaled with
it) until it fits, and the page says so.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "examples" / "designs" / "sources.json"
FILES = ROOT / "docs" / "designs" / "files"
INDEX = ROOT / "docs" / "designs" / "designs.json"
PAGE = ROOT / "docs" / "design" / "examples.md"

GROUP_ORDER = [("robot", "Robot hands and grippers"), ("legged robot", "Legged robots"),
               ("calibration", "Calibration set")]


def build(spec: dict) -> dict:
    from opentac.cad.main import GenerateRequest, generate
    from opentac.cad.pipeline.layout import ConnectorCapacityError

    params = dict(spec["params"])
    scale = 1.0
    for _ in range(12):
        try:
            resp = generate(GenerateRequest(
                outline=spec["outline"], router="hug", board_mode=spec.get("board_mode", "expand"),
                readout_board="auto", copper_layers=2, design_name=spec["slug"][:24], **params))
            break
        except Exception as exc:  # FastAPI wraps pipeline errors in HTTPException
            detail = str(getattr(exc, "detail", exc))
            if "readout board reads at most" not in detail and not isinstance(exc, ConnectorCapacityError):
                return {"slug": spec["slug"], "error": detail[:300]}
            scale *= 1.03
            for k in ("pixel_w_mm", "pixel_h_mm", "pitch_x_mm", "pitch_y_mm"):
                params[k] = round(spec["params"][k] * scale, 3)
    else:
        return {"slug": spec["slug"], "error": "does not fit the 60-pin readout board"}

    zip_bytes = base64.b64decode(resp.zip_b64)
    (FILES / f"{spec['slug']}.zip").write_bytes(zip_bytes)
    (FILES / f"{spec['slug']}.svg").write_text(resp.svg, encoding="utf-8")
    man = json.loads(zipfile.ZipFile(io.BytesIO(zip_bytes)).read("manifest.json"))
    s = resp.stats
    return {
        "slug": spec["slug"], "title": spec["title"], "group": spec["group"],
        "fabricated_as": spec["fabricated_as"],
        "taxels": int(s["active_pixels"]), "rows": int(man["r"]), "cols": int(man["c"]),
        "readout_board": {30: "30-pin", 60: "60-pin"}.get(int(s["connector_pos"]), f"{s['connector_pos']}-pos"),
        "pitch_mm": [params["pitch_x_mm"], params["pitch_y_mm"]],
        "taxel_mm": [params["pixel_w_mm"], params["pixel_h_mm"]],
        "pitch_raised": scale > 1.0,
        "drc_violations": int(resp.drc.get("violations", 0)),
        "zip_kb": round(len(zip_bytes) / 1024),
    }


def write_page(rows: list[dict]) -> None:
    lines = [
        "# Example designs",
        "",
        "These are the sensors we fabricated for the paper, regenerated with the current",
        "OpenTac CAD for the OpenTac readout boards. Each download is a complete export:",
        "KiCad board, Gerbers, manifest, the saved design (open it in OpenTac CAD to edit),",
        "a printable 1:1 PDF and the piezoresistive-layer cut files. Order it as described",
        "in [Order the sensor](../build/order-sensor.md).",
        "",
        "All are 2-layer boards made in grow mode. Open a design in OpenTac CAD to switch to",
        "4 layers, change the outline or the taxel size.",
        "",
    ]
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    for key, heading in GROUP_ORDER:
        items = by_group.get(key, [])
        if not items:
            continue
        lines += [f"## {heading}", "",
                  "| Design | Taxels | Grid | Pitch | Readout board | Download |",
                  "|---|---:|---|---|---|---|"]
        for r in items:
            pitch = f"{r['pitch_mm'][0]:g} mm" if r["pitch_mm"][0] == r["pitch_mm"][1] \
                else f"{r['pitch_mm'][0]:g} x {r['pitch_mm'][1]:g} mm"
            if r["pitch_raised"]:
                pitch += " *"
            lines.append(
                f"| [{r['title']}](../designs/files/{r['slug']}.svg) | {r['taxels']} | "
                f"{r['rows']} x {r['cols']} | {pitch} | {r['readout_board']} | "
                f"[zip, {r['zip_kb']} KB](../designs/files/{r['slug']}.zip) |")
        lines.append("")
    if any(r["pitch_raised"] for r in rows):
        lines += ["\\* Fabricated with more than 30 rows or columns, for an earlier 70-pin readout",
                  "board. The pitch here is raised just enough to fit the 60-pin board.", ""]
    PAGE.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    specs = json.loads(SOURCES.read_text())["designs"]
    if args.only:
        keep = set(args.only.split(","))
        specs = [s for s in specs if s["slug"] in keep]
    FILES.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(build, s) for s in specs]):
            r = fut.result()
            results.append(r)
            print(r["slug"], "ERROR " + r["error"] if "error" in r else
                  f"{r['taxels']} taxels {r['rows']}x{r['cols']} {r['readout_board']} drc {r['drc_violations']}",
                  flush=True)
    order = {s["slug"]: i for i, s in enumerate(json.loads(SOURCES.read_text())["designs"])}
    ok = sorted((r for r in results if "error" not in r), key=lambda r: order[r["slug"]])
    if not args.only:
        INDEX.write_text(json.dumps(ok, indent=1), encoding="utf-8")
        write_page(ok)
    return 1 if any("error" in r for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
