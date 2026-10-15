"""The OpenTac-CAD tool surface offered to an LLM, and its dispatch.

In process: this calls `main.generate()` directly, the same entry point the
/generate route uses, so a described board goes through the identical
board-mode policy, fixed-outline retry loop and post-route smoothing as a
drawn one.

Provider-neutral by construction — plain JSON Schema in, plain dicts out — so
swapping Gemini for another model touches chat/gemini.py only.

Three tools, matching the three stages SKILL.md prescribes: build a valid
outline, predict cheaply what it will produce, then spend the seconds. The
model chooses parameters; it never draws copper, and it cannot hand the
pipeline an outline that skipped the flat-cable-edge and winding invariants,
because only build_sensor_outline can mint one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .context import SKILL_DIR  # noqa: E402

HELPERS_DIR = SKILL_DIR / "helpers"
if str(HELPERS_DIR) not in sys.path:
    sys.path.insert(0, str(HELPERS_DIR))

import outline as O            # noqa: E402  (the skill's own helper)
import preflight as P          # noqa: E402  (the skill's own helper)

CABLE_SIDES = ("bottom", "top", "left", "right")

TOOL_SPECS = [
    {
        "name": "build_sensor_outline",
        "description": (
            "Build and validate the sensor's board outline, with a real flat "
            "cable edge cut on the side the connector exits. Always call this "
            "before generating, and call it again whenever the shape, the "
            "size, or the connector side changes. Returns an outline_id to "
            "pass to the other tools."),
        "parameters": {
            "type": "object",
            "properties": {
                "shape": {
                    "type": "string",
                    "enum": ["rectangle", "circle", "ellipse",
                             "regular_polygon", "polygon"],
                    "description": (
                        "Use 'polygon' with explicit points for anything "
                        "organic or irregular (an insole, a hand, an L)."),
                },
                "width_mm": {
                    "type": "number",
                    "description": "Width for rectangle/ellipse; across-flats "
                                   "size for regular_polygon.",
                },
                "height_mm": {"type": "number",
                              "description": "Height for rectangle/ellipse."},
                "diameter_mm": {"type": "number",
                                "description": "Diameter for circle."},
                "n_sides": {"type": "integer",
                            "description": "Sides for regular_polygon: 3 "
                                           "triangle, 6 hexagon."},
                "points": {
                    "type": "array",
                    "description": (
                        "shape='polygon' only: [x, y] control points in mm, "
                        "counter-clockwise. Spend points where the outline "
                        "turns rather than spreading them evenly. A simple "
                        "oval or rounded rectangle needs only a handful; "
                        "something with several distinct curves - an insole, "
                        "a hand, a fingertip - is usually worth around 20, "
                        "since too few splines into a smooth but wrong shape "
                        "(an 11-point insole lost its arch entirely). State "
                        "the real dimensions you are working from, and say so "
                        "if you are recalling them rather than certain."),
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                "smooth": {
                    "type": "boolean",
                    "description": (
                        "shape='polygon' only. Defaults to TRUE, which splines "
                        "the faceting out of hand-placed control points - what "
                        "an organic shape needs. Pass false only for a polygon "
                        "with genuinely straight edges (an L-bracket, a face "
                        "unrolled from an STL), where splining would round the "
                        "corners off."),
                },
                "cable_side": {
                    "type": "string",
                    "enum": list(CABLE_SIDES),
                    "description": (
                        "Which side the flat cable exits. This also sets the "
                        "grid orientation, so ask the user rather than "
                        "guessing. Default bottom."),
                },
                "cable_edge_width_mm": {
                    "type": "number",
                    "description": (
                        "Width of the flat edge to cut for the connector, "
                        "default 20. It must be at least as wide as the "
                        "connector the grid needs; preflight reports that "
                        "number as min_cable_edge_mm."),
                },
            },
            "required": ["shape"],
        },
    },
    {
        "name": "preflight_sensor",
        "description": (
            "Predict what a generate would produce - taxel count, rows, "
            "columns, channels, connector width needed - and flag blockers "
            "and warnings. Costs milliseconds against a slow generate, so "
            "always run it first and tell the user the predicted taxel count "
            "before spending the time."),
        "parameters": {
            "type": "object",
            "properties": {
                "outline_id": {"type": "string"},
                "pixel_w_mm": {"type": "number",
                               "description": "Taxel width. Never below 1.5."},
                "pixel_h_mm": {"type": "number",
                               "description": "Taxel height. Never below 1.5."},
                "pitch_x_mm": {"type": "number",
                               "description": "Column spacing; must exceed "
                                              "pixel_w_mm."},
                "pitch_y_mm": {"type": "number",
                               "description": "Row spacing; must exceed "
                                              "pixel_h_mm."},
                "board_mode": {
                    "type": "string",
                    "enum": ["fixed_keepout", "expand"],
                    "description": "The mode you intend to generate with.",
                },
            },
            "required": ["outline_id", "pixel_w_mm", "pixel_h_mm",
                         "pitch_x_mm", "pitch_y_mm"],
        },
    },
    {
        "name": "generate_sensor_board",
        "description": (
            "Run the real OpenTac-CAD pipeline: taxels, routing, "
            "connector, DRC. Takes 10-40 seconds. Report the taxel count, "
            "board size and DRC result it returns - not the numbers that were "
            "asked for. The board is then shown in the user's editor "
            "automatically, so do not tell them to download anything."),
        "parameters": {
            "type": "object",
            "properties": {
                "outline_id": {"type": "string"},
                "design_name": {
                    "type": "string",
                    "description": "Short slug naming this design, e.g. "
                                   "'square40_p2'.",
                },
                "pixel_w_mm": {"type": "number"},
                "pixel_h_mm": {"type": "number"},
                "pitch_x_mm": {"type": "number"},
                "pitch_y_mm": {"type": "number"},
                "gap_mm": {
                    "type": "number",
                    "description": "Interdigitated electrode gap inside one "
                                   "taxel, default 0.2. This is not the "
                                   "spacing between taxels - that is pitch "
                                   "minus taxel size.",
                },
                "trace_w_mm": {"type": "number",
                               "description": "Trace width, default 0.2."},
                "clearance_mm": {"type": "number",
                                 "description": "Copper clearance, default "
                                                "0.2."},
                "cable_length_mm": {"type": "number",
                                    "description": "Tail length, default 12."},
                "board_mode": {
                    "type": "string",
                    "enum": ["fixed_keepout", "expand"],
                    "description": (
                        "fixed_keepout: the drawn outline IS the board - use "
                        "it whenever the shape has to fit something physical. "
                        "expand: grow the board to fit the wiring, for when "
                        "the shape is a starting point rather than a "
                        "constraint."),
                },
                "readout_board": {
                    "type": "string",
                    "enum": ["auto", "30pin", "60pin"],
                    "description": "Which OpenTac readout board the sensor "
                                   "plugs into. auto (default) picks the "
                                   "30-pin board (up to 15x15 taxels) when "
                                   "the grid fits, else the 60-pin board "
                                   "(up to 30x30).",
                },
                "target_cols": {"type": "integer",
                                "description": "Grow mode only: solve taxel "
                                               "size from a column count."},
                "target_rows": {"type": "integer",
                                "description": "Grow mode only; pairs with "
                                               "target_cols."},
            },
            "required": ["outline_id", "design_name"],
        },
    },
]

# Whitelisted so a hallucinated argument fails loudly here instead of being
# silently dropped by GenerateRequest, which ignores unknown fields.
GEN_PARAMS = {
    "pixel_w_mm", "pixel_h_mm", "pitch_x_mm", "pitch_y_mm", "gap_mm",
    "trace_w_mm", "clearance_mm", "cable_length_mm", "readout_board",
    "target_cols", "target_rows",
}

# Diagnostic detail the model does not need is dropped, and
# stats["print_pdf"]["b64"] is a multi-megabyte blob that must never reach a
# prompt.
STAT_KEYS = (
    "total_pixels", "active_pixels", "max_pack_pixels", "dropped_by_keepout",
    "col_routes", "row_routes", "connector_pos", "board_mode", "router",
    "hug_fallback", "fixed_keepout_max_mm", "tail_length_mm",
    "connector_export_warning",
)


# --- shape construction -----------------------------------------------------
def _rect(w: float, h: float) -> np.ndarray:
    return np.array([(-w / 2, 0.0), (w / 2, 0.0), (w / 2, h), (-w / 2, h)])


def _ellipse(w: float, h: float, n: int = 240) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([0.5 * w * np.cos(t), 0.5 * h * np.sin(t) + h / 2])


def _regular(n_sides: int, across_mm: float) -> np.ndarray:
    r = across_mm / 2
    t = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) - np.pi / 2
    return np.column_stack([r * np.cos(t), r * np.sin(t) + r])


def _build_shape(a: dict):
    shape = a["shape"]
    if shape == "rectangle":
        return _rect(float(a["width_mm"]), float(a["height_mm"])), False
    if shape == "circle":
        d = float(a.get("diameter_mm") or a.get("width_mm"))
        return _ellipse(d, d), True
    if shape == "ellipse":
        return _ellipse(float(a["width_mm"]), float(a["height_mm"])), True
    if shape == "regular_polygon":
        n = int(a["n_sides"])
        if n < 3:
            raise ValueError("n_sides must be at least 3.")
        return _regular(n, float(a.get("width_mm") or a["diameter_mm"])), False
    if shape == "polygon":
        pts = np.asarray(a["points"], float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError("points must be at least 3 [x, y] pairs in mm.")
        # Smoothing defaults ON. Anything reaching "polygon" is a shape with no
        # primitive of its own, which in practice means organic, and a curve
        # should not depend on the model remembering an optional flag - the
        # same request produced a splined insole one run and a faceted one the
        # next. Straight-edged shapes can still opt out, and resample_closed
        # falls back to linear whenever the spline leaves the source outline's
        # envelope, so an L or an unrolled STL face survives even if nobody
        # does.
        return pts, bool(a.get("smooth", True))
    raise ValueError(f"unknown shape {shape!r}")


class Session:
    """One conversation's worth of state: the outlines and the boards."""

    def __init__(self) -> None:
        self.outlines: dict[str, list] = {}
        self.designs: list[dict] = []
        self._n = 0

    # -- tools ------------------------------------------------------------
    def build_sensor_outline(self, **a) -> dict:
        """Every SKILL.md outline invariant is enforced here rather than
        trusted to the model: counter-clockwise winding, an explicitly *cut*
        flat first edge, and splining only for genuinely curved shapes."""
        side = a.get("cable_side", "bottom")
        if side not in CABLE_SIDES:
            raise ValueError(f"cable_side must be one of {CABLE_SIDES}")

        pts, curved = _build_shape(a)
        if curved and a["shape"] == "polygon":
            pts = O.resample_closed(pts, n=240, smooth=True)
        pts = O.orient_ccw(pts)
        ol = O.with_flat_cable_edge(
            pts, side=side, width_mm=float(a.get("cable_edge_width_mm", 20.0)))

        problems = O.validate(ol)
        self._n += 1
        oid = f"outline_{self._n}"
        self.outlines[oid] = [[round(float(x), 4), round(float(y), 4)]
                              for x, y in ol]
        minx, miny = ol.min(axis=0)
        maxx, maxy = ol.max(axis=0)
        out = {
            "outline_id": oid,
            "n_vertices": int(len(ol)),
            "bbox_w_mm": round(float(maxx - minx), 2),
            "bbox_h_mm": round(float(maxy - miny), 2),
            "cable_side": side,
            "cable_edge_len_mm": round(float(np.hypot(*(ol[1] - ol[0]))), 2),
        }
        if problems:
            out["problems"] = problems
            out["usable"] = False
        return out

    def preflight_sensor(self, outline_id: str, **a) -> dict:
        ol = np.asarray(self._outline(outline_id), float)
        pf = P.preflight(ol, float(a["pixel_w_mm"]), float(a["pixel_h_mm"]),
                         float(a["pitch_x_mm"]), float(a["pitch_y_mm"]),
                         router="hug")   # the only router to design with
        need = O.min_cable_edge_mm(pf["predicted"]["rows"],
                                   pf["predicted"]["cols"])
        pf["min_cable_edge_mm"] = None if need is None else round(need, 1)
        pf["cable_edge_len_mm"] = round(float(np.hypot(*(ol[1] - ol[0]))), 2)
        return pf

    def generate_sensor_board(self, outline_id: str, design_name: str,
                              **a) -> dict:
        # router is deliberately not a parameter the model can set: hug is the
        # only router to design with, in either board mode.
        unknown = set(a) - GEN_PARAMS - {"board_mode"}
        if unknown:
            raise ValueError(
                f"unsupported parameter(s): {sorted(unknown)}. Supported: "
                f"{sorted(GEN_PARAMS | {'board_mode'})}")

        from opentac.cad import main   # late: avoids a circular import at module load

        outline = self._outline(outline_id)
        board_mode = a.pop("board_mode", "fixed_keepout")
        params = {k: v for k, v in a.items() if k in GEN_PARAMS}
        resp = main.generate(main.GenerateRequest(
            outline=outline, board_mode=board_mode, router="hug", **params))

        stats = {k: resp.stats[k] for k in STAT_KEYS if k in resp.stats}
        drc = resp.drc
        kinds: dict[str, int] = {}
        for d in drc.get("details", []):
            kinds[d["type"]] = kinds.get(d["type"], 0) + 1

        self.designs.append({
            "name": design_name,
            "outline": outline,
            # The site loads this straight into its editor.
            "params": {**params, "board_mode": board_mode, "router": "hug"},
            "svg": resp.svg,
            "stats": stats,
        })
        return {
            "report": _summarize(stats, drc, kinds),
            "stats": stats,
            "drc": {
                "violations": drc["violations"],
                "by_type": kinds,
                "sample": drc.get("details", [])[:5],
                # An incomplete net is the opposite of a violation: hug
                # refused to emit copper rather than route it unsafely, so
                # those taxels are simply unconnected.
                "incomplete_nets": drc.get("routing_warning_nets", []),
                "unconnected_taxels": len(drc.get("routing_problem_pixels",
                                                  [])),
            },
            "shown_to_user": True,
        }

    # -- plumbing ---------------------------------------------------------
    def _outline(self, outline_id: str) -> list:
        if outline_id not in self.outlines:
            raise ValueError(
                f"no such outline_id {outline_id!r}. Known: "
                f"{sorted(self.outlines) or 'none yet'}. Call "
                "build_sensor_outline first.")
        return self.outlines[outline_id]

    def dispatch(self, name: str, args: dict) -> dict:
        """Run one tool call. Failures come back as data, so the model can
        read the error and correct itself rather than the turn dying."""
        if name not in {s["name"] for s in TOOL_SPECS}:
            return {"error": f"unknown tool {name!r}"}
        try:
            return getattr(self, name)(**args)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


def _summarize(stats: dict, drc: dict, kinds: dict) -> str:
    """Plain-language verdict. A board is good when no copper leaves the board
    edge and no two traces cross - everything else is detail."""
    lines = [
        f"{stats.get('total_pixels', 0)} taxels, "
        f"{stats.get('col_routes', 0)} columns x {stats.get('row_routes', 0)} "
        f"rows, {stats.get('connector_pos', 0)}-position connector, "
        f"{stats.get('router', 'hug')} router.",
    ]
    dropped = stats.get("max_pack_pixels", 0) - stats.get("total_pixels", 0)
    if dropped > 0:
        lines.append(f"{dropped} more taxels fit the outline geometrically "
                     "but were dropped to leave room for wiring.")
    if drc["violations"] == 0:
        lines.append("Clean: no crossings, no copper outside the board edge.")
    else:
        detail = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
        lines.append(f"PROBLEM: {drc['violations']} DRC violations"
                     f"{' (' + detail + ')' if detail else ''}. Diagnose the "
                     "cause before regenerating.")
    return " ".join(lines)


def log_call(name: str, args: dict, result: dict) -> None:
    """Tool traffic on the terminal, for debugging.

    say() rather than print(): this echoes model-written arguments, and the
    Windows console cannot encode much of what comes back. Imported inside the
    function because chat/__init__.py imports this module — at call time the
    package is fully loaded, at import time it is not.
    """
    from . import say
    shown = {k: v for k, v in args.items() if k != "points"}
    if "points" in args:
        shown["points"] = f"<{len(args['points'])} pts>"
    say(f"  -> TOOL {name}({json.dumps(shown)})")
    text = json.dumps({k: v for k, v in result.items() if k != "svg"})
    say(f"  <- {text[:400]}{' ...' if len(text) > 400 else ''}")
