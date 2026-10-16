"""Run the OpenTac-CAD pipeline in process and report what came out.

No HTTP server. main.generate() is the same entry point the web app's
/generate route uses, so this shares the web app's behaviour exactly — the
board-mode policy, the fixed-outline keep-out retry loop, the post-route
smoothing and the DRC fallback all live in that handler, not in pipeline/.
Calling pipeline/ directly would mean reimplementing them.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np


def _load_backend():
    from opentac.cad import main
    return main


def generate_board(outline, out_dir: str | os.PathLike, *,
                   board_mode: str = "expand", router: str | None = None,
                   **params):
    """Generate a board and write its artefacts to out_dir.

    Defaults to grow mode ("expand"), as SKILL.md recommends: the board grows
    just enough to fit the wiring and is shrink-wrapped back toward the
    copper, keeps every taxel the shape holds, and is the fast mode. Pass
    board_mode="fixed_keepout" only when the drawn outline is a hard limit
    (the board must fit an enclosure or a part with no margin).

    router defaults to "hug" in both board modes, which is what the site
    itself sends (the frontend's checkbox ships checked and is never gated on
    board mode). "legacy" is the original router, kept only so the two can be
    compared; it is not a mode to design in. On a concave outline the gap is
    not subtle — measured on a C-shape under fixed_keepout, legacy returned 6
    taxels and 25 DRC violations, 15 of them crossings, where hug returned 26
    taxels and no crossing at all.
    """
    main = _load_backend()
    outline = np.asarray(outline, float)
    if router is None:
        router = "hug"

    t0 = time.time()
    resp = main.generate(main.GenerateRequest(
        outline=outline.tolist(), board_mode=board_mode, router=router,
        **params))
    elapsed = time.time() - t0

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64))) as zf:
        zf.extractall(out)
    (out / "preview.svg").write_text(resp.svg)

    return {
        "report": summarize(resp, elapsed),
        "stats": resp.stats,
        "drc": resp.drc,
        "out_dir": str(out),
        # Held so follow-ups can use redo-padding instead of a full re-route.
        "edit_data": resp.edit_data,
        "elapsed_s": round(elapsed, 1),
    }


def summarize(resp, elapsed: float | None = None) -> str:
    """Plain-language verdict. A board is good when no copper leaves the
    board edge and no two traces cross — everything else is detail."""
    s, drc = resp.stats, resp.drc
    kinds: dict[str, int] = {}
    for d in drc.get("details", []):
        kinds[d["type"]] = kinds.get(d["type"], 0) + 1

    # The verdict is the final DRC only. It is run on the finished artefact
    # with the board polygon and edge clearance passed in (main.py's
    # _build_outputs), so "copper outside the edge" and "traces crossing" are
    # both inside this one number. stats' fixed_keepout_worst_escape_mm is a
    # mid-retry-loop measurement and routinely non-zero on a clean board.
    # Rows and columns from the manifest: stats' col_routes/row_routes count
    # routed segments, which overcounts a line split around a notch.
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64))) as zf:
        man = json.loads(zf.read("manifest.json"))
    lines = [
        f"{s['total_pixels']} taxels, {man['c']} columns x "
        f"{man['r']} rows, {s['connector_pos']}-position connector, "
        f"{s.get('router', 'legacy')} router.",
    ]
    dropped = s.get("max_pack_pixels", 0) - s["total_pixels"]
    if dropped > 0:
        lines.append(
            f"{dropped} more taxels fit the outline geometrically but were "
            "dropped to leave room for wiring.")

    if drc["violations"] == 0:
        lines.append("Clean: no crossings, no copper outside the board edge.")
    else:
        detail = ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items()))
        lines.append(f"PROBLEM: {drc['violations']} DRC violations ({detail}). "
                     "Diagnose the cause before regenerating — see SKILL.md.")
    if elapsed:
        lines.append(f"({elapsed:.0f}s)")
    return " ".join(lines)


def files_written(out_dir: str | os.PathLike) -> dict:
    """What ended up on disk, and what each part is for."""
    out = Path(out_dir)
    manifest = {}
    mf = out / "manifest.json"
    if mf.is_file():
        manifest = json.loads(mf.read_text())
    return {
        "preview.svg": "Open this first — the board as it will be built.",
        "board.kicad_pcb": "Open in KiCad to inspect or edit.",
        "gerber/": "Send this folder to the fab house. This is the order.",
        "manifest.json": "Machine-readable summary of what was built.",
        "_manifest": manifest,
    }
