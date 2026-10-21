"""Cheap checks to run before a generate, so problems surface in milliseconds
rather than after a route (about 1-6 s in grow mode and 10-45 s in fixed mode
on a laptop).

Everything here is arithmetic or a line-polygon test. None of it routes.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Polygon

MIN_TAXEL_MM = 1.5  # Neither taxel dimension may go below this.


def preflight(outline, pixel_w_mm: float, pixel_h_mm: float,
              pitch_x_mm: float, pitch_y_mm: float,
              router: str = "hug") -> dict:
    """Predict what a generate would produce, and flag what will go wrong.

    Returns {"blockers": [...], "warnings": [...], "predicted": {...}}.
    Blockers must be resolved (usually by asking the user); warnings are
    things to say out loud before spending the time.

    router only changes how the re-entrant-band check below is worded: the
    hug router joins a split row/column into one net with a link trace
    hugging the concavity, so the crossing risk that "legacy" carries does
    not apply. Leave it at "hug" — that is what generate_board() uses in both
    board modes. Pass router="legacy" only when deliberately comparing
    against the legacy router.
    """
    outline = np.asarray(outline, float)
    poly = Polygon(outline)
    blockers: list[str] = []
    warnings: list[str] = []

    # --- taxel size floor -------------------------------------------------
    for name, v in (("width", pixel_w_mm), ("height", pixel_h_mm)):
        if v < MIN_TAXEL_MM:
            blockers.append(
                f"Taxel {name} is {v:.2f}mm. Neither taxel dimension should "
                f"go below {MIN_TAXEL_MM}mm.")
    if pitch_x_mm < pixel_w_mm or pitch_y_mm < pixel_h_mm:
        blockers.append("Pitch is smaller than the taxel — taxels would overlap.")

    # --- predicted grid ---------------------------------------------------
    minx, miny, maxx, maxy = poly.bounds
    cols = max(1, int((maxx - minx) // pitch_x_mm))
    rows = max(1, int((maxy - miny) // pitch_y_mm))
    cells = cols * rows
    fill = poly.area / ((maxx - minx) * (maxy - miny))
    approx_taxels = int(round(cells * fill))

    if cells != approx_taxels:
        warnings.append(
            f"A {cols}x{rows} grid is {cells} cells, but this shape only "
            f"fills {fill:.0%} of its bounding box, so roughly "
            f"{approx_taxels} taxels will land inside it. Wiring keep-out "
            "will take a further share.")

    # --- connector capacity ----------------------------------------------
    from outline import min_cable_edge_mm
    need = min_cable_edge_mm(rows, cols)
    if need is None:
        blockers.append(
            f"{rows} rows and {cols} columns "
            "do not fit an OpenTac readout board: the 60-pin board reads at most "
            "30 rows and 30 columns, the 30-pin board 15 x 15. Use larger taxels "
            "or sense a smaller area.")
    else:
        edge_len = float(np.hypot(*(outline[1] - outline[0])))
        if edge_len + 1e-6 < need:
            blockers.append(
                f"The cable edge is {edge_len:.1f}mm wide but this grid needs "
                f"a {need:.1f}mm connector. Widen the flat edge, move the "
                "cable to a wider side, or use larger taxels.")

    # --- split columns / rows --------------------------------------------
    for axis, label, n in (("x", "column", cols), ("y", "row", rows)):
        band = _split_band(poly, axis)
        if band:
            lo, hi = band
            if router == "hug":
                warnings.append(
                    f"This outline is re-entrant along {axis} between {lo:.0f}mm "
                    f"and {hi:.0f}mm: a {label} there enters the shape, leaves "
                    "it and comes back. The layout stage still splits such a "
                    f"{label} into two hops, but the hug router joins them back "
                    "into one net with a link trace that hugs the concavity — "
                    "no crossing risk. Expect it to take some board area out "
                    "of the notch; the notch will not stay perfectly open.")
            else:
                warnings.append(
                    f"This outline is re-entrant along {axis} between {lo:.0f}mm "
                    f"and {hi:.0f}mm: a {label} there enters the shape, leaves it "
                    "and comes back. The layout stage splits such a "
                    f"{label} into two separate nets, and the legacy router can "
                    "route them across each other. If DRC reports crossings, "
                    "this is why — use router=\"hug\", which works in both "
                    "board modes and joins the split back into one net.")

    return {
        "blockers": blockers,
        "warnings": warnings,
        "predicted": {"cols": cols, "rows": rows, "grid_cells": cells,
                      "approx_taxels": approx_taxels,
                      "board_w_mm": round(maxx - minx, 1),
                      "board_h_mm": round(maxy - miny, 1),
                      "channels": rows + cols},
    }


def _split_band(poly: Polygon, axis: str, samples: int = 120):
    """Range over which a scan line crosses the shape more than once."""
    minx, miny, maxx, maxy = poly.bounds
    hits = []
    lo, hi = (minx, maxx) if axis == "x" else (miny, maxy)
    for v in np.linspace(lo + 0.5, hi - 0.5, samples):
        line = (LineString([(v, miny - 1), (v, maxy + 1)]) if axis == "x"
                else LineString([(minx - 1, v), (maxx + 1, v)]))
        seg = poly.intersection(line)
        n = 0 if seg.is_empty else (1 if seg.geom_type == "LineString"
                                    else len(seg.geoms))
        if n > 1:
            hits.append(v)
    return (min(hits), max(hits)) if hits else None
