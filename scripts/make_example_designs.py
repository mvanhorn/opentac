"""Regenerate the ready-to-order example designs for the website.

    uv run python scripts/make_example_designs.py [--only slug,slug] [--workers N]
    uv run python scripts/make_example_designs.py --page-only

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

--page-only rewrites only the examples page from designs.json and the SVGs
already in docs/designs/files/: nothing is generated, no ZIP or SVG changes.
The page shows each SVG as a lazy-loaded thumbnail (31 SVGs, about 2 MB, about
0.5 MB gzipped as GitHub Pages serves them).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
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


CASE_BLOCKS = re.compile(r"Lengthen the cable by at least ([\d.]+)mm")


def generate_fitting(spec: dict, params: dict):
    """Generate for the 60-pin board at most, raising the pitch in 3% steps if the grid is too big."""
    from opentac.cad.main import GenerateRequest, generate
    from opentac.cad.pipeline.layout import ConnectorCapacityError

    scaled = dict(params)
    scale = 1.0
    for _ in range(12):
        try:
            resp = generate(GenerateRequest(
                outline=spec["outline"], router="hug", board_mode=spec.get("board_mode", "expand"),
                readout_board="auto", copper_layers=2, design_name=spec["slug"][:24], **scaled))
            return resp, scaled, scale
        except Exception as exc:  # FastAPI wraps pipeline errors in HTTPException
            detail = str(getattr(exc, "detail", exc))
            if "readout board reads at most" not in detail and not isinstance(exc, ConnectorCapacityError):
                raise RuntimeError(detail[:300]) from exc
            scale *= 1.03
            for k in ("pixel_w_mm", "pixel_h_mm", "pitch_x_mm", "pitch_y_mm"):
                scaled[k] = round(params[k] * scale, 3)
    raise RuntimeError("does not fit the 60-pin readout board")


def build(spec: dict) -> dict:
    """Generate one design and add its printed case, as OpenTac-CAD's Download ZIP does.

    The designs were fabricated before the printed case existed, with a 12 mm cable
    tail. They are generated with the automatic tail length (no cable_length_mm),
    which is long enough for the narrow tail and for the printed case to leave the
    connector clear; a design whose case still warns about the cable, or has a part
    that is not a closed solid, is reported instead of written.
    """
    from opentac.cad.main import _case_inputs
    from opentac.cad.export.case_zip import build_case_files, zip_with_case

    params = {k: v for k, v in spec["params"].items() if k != "cable_length_mm"}
    try:
        resp, used, scale = generate_fitting(spec, params)
    except RuntimeError as exc:
        return {"slug": spec["slug"], "error": str(exc)}
    board, cp = _case_inputs(resp.edit_data, None, None)
    case = build_case_files(*board["args"], cp, primary="bumps", stem=spec["slug"],
                            tail_extension=board["tail_extension"])
    cable = [w for w in case.warnings
             if CASE_BLOCKS.search(w) or w.startswith("No cable tail was found")]
    if cable:
        return {"slug": spec["slug"], "error": f"printed case still blocks the connector: {cable}"}
    if case.left_out or any("not a closed solid" in w for w in case.warnings):
        return {"slug": spec["slug"], "error": f"printed case: {case.left_out or case.warnings}"}

    zip_bytes = base64.b64decode(zip_with_case(resp.zip_b64, case.files))
    (FILES / f"{spec['slug']}.zip").write_bytes(zip_bytes)
    (FILES / f"{spec['slug']}.svg").write_text(resp.svg, encoding="utf-8")
    man = json.loads(zipfile.ZipFile(io.BytesIO(zip_bytes)).read("manifest.json"))
    s = resp.stats
    return {
        "slug": spec["slug"], "title": spec["title"], "group": spec["group"],
        "fabricated_as": spec["fabricated_as"],
        "taxels": int(s["active_pixels"]), "rows": int(man["r"]), "cols": int(man["c"]),
        "readout_board": {30: "30-pin", 60: "60-pin"}.get(int(s["connector_pos"]), f"{s['connector_pos']}-pos"),
        "pitch_mm": [used["pitch_x_mm"], used["pitch_y_mm"]],
        "taxel_mm": [used["pixel_w_mm"], used["pixel_h_mm"]],
        "pitch_raised": scale > 1.0,
        "cable_length_mm": s["tail_length_mm"],
        "cable_length_reason": s.get("tail_length_reason"),
        "drc_violations": int(resp.drc.get("violations", 0)),
        "zip_kb": round(len(zip_bytes) / 1024),
        "case_warnings": case.warnings,
    }


THUMB_BOX = (240, 180)  # width/height attributes: the preview fitted in this box


def thumb_size(slug: str) -> tuple[int, int]:
    """The preview's aspect ratio, from the SVG's own width/height, fitted in THUMB_BOX."""
    head = (FILES / f"{slug}.svg").read_text(encoding="utf-8")[:600]
    w, h = (float(re.search(rf'<svg[^>]*\b{k}="([\d.]+)"', head).group(1)) for k in ("width", "height"))
    k = min(THUMB_BOX[0] / w, THUMB_BOX[1] / h)
    return round(w * k), round(h * k)


def design_card(r: dict) -> list[str]:
    files = f"../designs/files/{r['slug']}"
    grid = f"{r['rows']} × {r['cols']}"
    pitch = f"{r['pitch_mm'][0]:g} mm" if r["pitch_mm"][0] == r["pitch_mm"][1] \
        else f"{r['pitch_mm'][0]:g} × {r['pitch_mm'][1]:g} mm"
    star = "\\*" if r["pitch_raised"] else ""  # see the note under the last group
    w, h = thumb_size(r["slug"])
    alt = f"{r['title']}: board preview, {r['taxels']} taxels, {grid} grid"
    return [
        '<figure class="ot-design" markdown="span">',
        f'[![{alt}]({files}.svg){{ width="{w}" height="{h}" loading="lazy" }}]({files}.svg)',
        f'<figcaption markdown="span">**{r["title"]}**<br>{r["taxels"]} taxels, {grid} grid<br>'
        # a narrow card wraps between pitch and board, never inside "60-pin"
        f'<span class="ot-nowrap">{pitch} pitch{star},</span> '
        f'<span class="ot-nowrap">{r["readout_board"]} board</span><br>'
        f'[Download ZIP ({r["zip_kb"]} KB)]({files}.zip)</figcaption>',
        "</figure>",
    ]


def write_page(rows: list[dict]) -> None:
    lines = [
        "---",
        "title: Ready-to-order designs",
        "description: The sensors fabricated for the OpenTac paper, regenerated for the OpenTac "
        "readout boards, as complete OpenTac-CAD exports to download and order.",
        "---",
        "",
        "<!-- Generated by scripts/make_example_designs.py (--page-only rewrites just this page). "
        "Edit the script, not this file. -->",
        "",
        "# Ready-to-order designs",
        "",
        "These are the sensors we fabricated for the paper, regenerated with the current",
        "OpenTac-CAD for the OpenTac readout boards. Each download is the same complete export",
        "OpenTac-CAD's **Download ZIP** gives: KiCad board, Gerbers, manifest, the saved design",
        "(open it in OpenTac-CAD to edit), a printable 1:1 PDF, the piezoresistive-layer cut",
        "files and the [printed case](../build/index.md#the-printed-case) in `case/` (bottom,",
        "bumped lid, and flat and hatched lids). Order the board as described in",
        "[Order the sensor](../build/order-sensor.md), then build it with the",
        "[assembly guide](../build/assembly.md).",
        "",
        "All are 2-layer boards made in grow mode. Open a design in OpenTac-CAD to switch to",
        "4 layers, change the outline or the taxel size. In the previews, taxels are pale blue,",
        "columns red (top copper) and rows blue (bottom copper); click one to open it full size.",
        "",
    ]
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    for key, heading in GROUP_ORDER:
        items = by_group.get(key, [])
        if not items:
            continue
        lines += [f"## {heading}", "", '<div class="ot-designs" markdown>']
        for r in items:
            lines += design_card(r)
        lines += ["</div>", ""]
    if any(r["pitch_raised"] for r in rows):
        lines += ["\\* Fabricated with more than 30 rows or columns, for an earlier 70-pin readout",
                  "board. The pitch here is raised just enough to fit the 60-pin board.", ""]
    PAGE.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--page-only", action="store_true",
                    help="rewrite docs/design/examples.md from designs.json; generate nothing")
    args = ap.parse_args()
    if args.page_only:
        write_page(json.loads(INDEX.read_text(encoding="utf-8")))
        print(f"wrote {PAGE.relative_to(ROOT)}")
        return 0
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
