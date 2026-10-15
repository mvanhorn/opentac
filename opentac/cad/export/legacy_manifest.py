"""Recovering a design from a manifest written before the design block existed.

Boards exported before 2026-09-04 carry a v3 manifest: outline, edge_cuts,
taxel centres, the taxel footprint and the scan matrix. That describes the
board that was MADE and nothing about how to make it, so it cannot be
re-generated -- and for an already-fabricated board, it should not be.

Re-generating was tried and rejected. On a stored 41x41mm calibration square
the taxel centres came back bit-exact, but the cut boundary did not: 452
stored points against 492 fresh ones, because the outline stitching has
changed since. A case built from the regenerated board would fit a board that
is very slightly not the one on the bench. The stored `edge_cuts` IS the
boundary that was manufactured, so the recovery below uses it directly and
regenerates nothing.

What that buys and what it costs:

  works   the case and the bump sheet. Both are geometry: /case reads only
          final_outline, the cable edge, the taxel centres and the two
          pitches, and /bump-sheet is the same shape of question. Every one
          of those is in a v3 manifest, exactly as fabricated.
  lost    routing. Trace polylines were never in the manifest, so route
          editing and a Gerber/KiCad re-export are not recoverable -- and
          faking them by re-routing would produce copper that does not match
          the board the user is holding. The caller is told this rather than
          handed a plausible-looking board.

Measured over 29 real fabricated designs: bump sheet 29/29, case 29/29
(24 directly, the other 5 after the ring cleanup below).
"""

import numpy as np
from shapely.geometry import Point, Polygon

from opentac.cad.pipeline.types import Params

# The stored ring can carry sub-micron segments that the case builder's mesh
# boolean chokes on ("Not all meshes are volumes!"): measured on the 20x20
# p2.0 calibration square, whose 452-point boundary has a 0.000985mm segment.
# Simplifying at 1um drops it to 401 points and the case builds watertight.
#
# 1um is three orders of magnitude below PCB fab tolerance (~100um), so this
# cannot move a manufactured edge in any way that matters -- and the result is
# checked against the original below rather than trusted.
RING_SIMPLIFY_MM = 0.001

# How far the cleaned ring is allowed to depart from the stored one before the
# cleanup is abandoned and the raw ring used instead. simplify() is normally
# well behaved at this tolerance, but it is a shape operation on a boundary
# that came out of a file, and a silent shape change here would reach a
# printed case.
RING_MAX_DEVIATION_MM = 0.01


def _max_vertex_distance(points, curve) -> float:
    """Farthest any of `points` sits from `curve`. NaN if it cannot be measured."""
    try:
        worst = 0.0
        for x, y in np.asarray(points, dtype=float):
            d = curve.distance(Point(float(x), float(y)))
            if not np.isfinite(d):
                return float("nan")
            worst = max(worst, float(d))
        return worst
    except Exception:
        return float("nan")


def _clean_ring(points):
    """The stored cut boundary, with degenerate micro-segments removed.

    Returns (ring, note). Falls back to the stored ring, unchanged, whenever
    the cleanup cannot be shown to be faithful.
    """
    raw = np.asarray(points, dtype=float)
    if len(raw) < 4:
        return [[float(x), float(y)] for x, y in raw], None
    try:
        poly = Polygon(raw)
        if not poly.is_valid:
            return [[float(x), float(y)] for x, y in raw], "stored boundary is not a valid ring"
        simple = poly.simplify(RING_SIMPLIFY_MM)
        ring = np.asarray(simple.exterior.coords[:-1], dtype=float)
        if len(ring) < 4:
            return [[float(x), float(y)] for x, y in raw], None
        # Deviation, measured BOTH ways: no stored vertex may sit far from the
        # new boundary and no new vertex far from the stored one. One way alone
        # would miss a simplification that cuts a corner off entirely.
        #
        # Vertex-to-curve distances rather than shapely's hausdorff_distance:
        # that returned NaN on some of these real boundaries, and NaN fails
        # every `>` comparison, so the guard below would have waved the cleanup
        # through without ever checking it.
        dev = max(_max_vertex_distance(raw, simple.exterior),
                  _max_vertex_distance(ring, poly.exterior))
        if not np.isfinite(dev) or dev > RING_MAX_DEVIATION_MM:
            return ([[float(x), float(y)] for x, y in raw],
                    f"boundary cleanup skipped: it would have moved the edge {dev:.4f}mm")
        if len(ring) == len(raw):
            return [[float(x), float(y)] for x, y in raw], None
        return ([[float(x), float(y)] for x, y in ring],
                f"cut boundary cleaned: {len(raw)} points to {len(ring)}, "
                f"max edge movement {dev * 1000:.2f}um")
    except Exception:
        return [[float(x), float(y)] for x, y in raw], "boundary cleanup failed; using it as stored"


def _pitch(values, fallback):
    """Taxel pitch along one axis, from the taxel centres themselves.

    NOT from the manifest's px/py: those are indexed by pad order, which is
    deliberately not geometric order, so py can even be negative. The manifest
    says so in its own _readme. The centres are the authority.
    """
    uniq = np.unique(np.round(np.asarray(values, dtype=float), 4))
    if len(uniq) < 2:
        return float(fallback)
    return round(float(np.diff(uniq).min()), 4)


# Margin around the board in the preview box, matching _build_edit_data in
# main.py and generate_svg in export/svg_preview.py. All three have to agree:
# the route editor converts mm to SVG units with edit_data["preview"] and the
# SVG's own viewBox, so a box that disagrees with the drawing puts every
# overlay in the wrong place.
PREVIEW_MARGIN_MM = 2.0


def _preview_box(final_outline) -> dict:
    pts = np.asarray(final_outline, dtype=float)
    return {"xmin": float(pts[:, 0].min()) - PREVIEW_MARGIN_MM,
            "ymin": float(pts[:, 1].min()) - PREVIEW_MARGIN_MM,
            "xmax": float(pts[:, 0].max()) + PREVIEW_MARGIN_MM,
            "ymax": float(pts[:, 1].max()) + PREVIEW_MARGIN_MM}


def is_legacy_manifest(data: dict) -> bool:
    """A manifest that describes a board but carries no design block."""
    return bool(data) and "design" not in data and data.get("pixels") is not None


def recover(manifest: dict) -> tuple[dict, list[str]]:
    """Build a geometry-only edit_data from a v3 manifest. Returns (edit_data, notes).

    The result is deliberately NOT a full edit_data: it has no nets, no pads
    and no routes, so /export-edited and the route editor cannot run on it.
    /case and /bump-sheet can, which is the point.
    """
    notes: list[str] = []
    pixels = manifest.get("pixels") or []
    if len(pixels) < 1:
        raise ValueError("This manifest lists no taxels, so there is no board to recover.")
    outline = manifest.get("outline") or []
    if len(outline) < 3:
        raise ValueError("This manifest has no usable outline.")

    centres = [[float(p[0]), float(p[1])] for p in pixels]
    w = float(manifest.get("w") or 0) or 1.0
    h = float(manifest.get("h") or 0) or 1.0
    pitch_x = _pitch([c[0] for c in centres], w)
    pitch_y = _pitch([c[1] for c in centres], h)

    cuts = manifest.get("edge_cuts") or []
    if len(cuts) >= 4:
        final_outline, note = _clean_ring(cuts)
        if note:
            notes.append(note)
    else:
        final_outline = [[float(x), float(y)] for x, y in np.asarray(outline, dtype=float)]
        notes.append("This manifest has no edge_cuts, so the drawn outline is being used as "
                     "the board edge. A case built from it will be smaller than the real board.")

    # Every Params field, so downstream code that indexes params cannot
    # KeyError. Only the ones listed in `recovered` came from the file; the
    # rest are defaults and are irrelevant to the case and the bump sheet,
    # which read pitch and taxel size and nothing else.
    fp = manifest.get("footprint") or {}
    params = {k: v for k, v in vars(Params()).items()}
    params.update({"pixel_w_mm": w, "pixel_h_mm": h,
                   "pitch_x_mm": pitch_x, "pitch_y_mm": pitch_y})
    recovered = ["pixel_w_mm", "pixel_h_mm", "pitch_x_mm", "pitch_y_mm"]
    for key, field in (("trace_w", "trace_w_mm"), ("gap", "gap_mm"),
                       ("center_clear", "center_clear_mm"), ("edge_clear", "edge_clear_mm"),
                       ("via_drill", "via_drill_mm"), ("via_dia", "via_dia_mm")):
        if key in fp:
            params[field] = float(fp[key])
            recovered.append(field)
    if not fp:
        notes.append("This manifest predates the footprint block, so the taxel copper "
                     "parameters could not be recovered. The board shape and taxel "
                     "positions are unaffected.")

    edit_data = {
        "version": 1,
        "geometry_only": True,
        "recovered_from": f"manifest v{manifest.get('v', '?')}",
        "recovered_params": sorted(recovered),
        "outline": [[float(x), float(y)] for x, y in np.asarray(outline, dtype=float)],
        "sensor_outline": [[float(x), float(y)] for x, y in np.asarray(outline, dtype=float)],
        "final_outline": final_outline,
        "fill_region": None,
        "params": params,
        # Required by the route editor even with no routes to edit: load()
        # resets the tail draft, which converts mm to SVG through this box.
        "preview": _preview_box(final_outline),
        # The manifest lists ACTIVE taxels only -- a taxel the router dropped
        # never appears -- so nothing here is removed.
        "removed_pix": [],
        "layout": {
            "placements_xy": centres,
            "routing": {},
            "tail": {},
        },
        "routes": [],
    }
    return edit_data, notes


def geometry_svg(edit_data: dict) -> str:
    """A preview drawn from geometry alone: board edge, drawn outline, taxels.

    Deliberately shows no traces. The routing is not in the file, and drawing
    a plausible guess over a fabricated board would be worse than drawing
    nothing -- the whole reason this path exists is that the stored board and
    a re-routed one are not the same board. Class names match
    export/svg_preview.py so it inherits the app's visual language.
    """
    board = np.asarray(edit_data["final_outline"], dtype=float)
    drawn = np.asarray(edit_data["outline"], dtype=float)
    centres = np.asarray(edit_data["layout"]["placements_xy"], dtype=float)
    hw = float(edit_data["params"]["pixel_w_mm"]) / 2
    hh = float(edit_data["params"]["pixel_h_mm"]) / 2

    box = edit_data.get("preview") or _preview_box(board)
    xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
    width, height = xmax - xmin, ymax - ymin
    if width <= 0 or height <= 0:
        raise ValueError("The recovered board has no area.")

    def tx(x): return x - xmin
    def ty(y): return ymax - y          # SVG is Y-down, the board frame is Y-up

    svg_w = 800.0
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'viewBox="0 0 {width:.2f} {height:.2f}" '
           f'width="{svg_w:.0f}" height="{svg_w * height / width:.0f}">',
           '<style>'
           '.outline{fill:none;stroke:#333;stroke-width:0.3}'
           '.sensor-outline{fill:none;stroke:#6688aa;stroke-width:0.16;stroke-dasharray:0.7,0.35}'
           '.pixel{fill:#ddeeff;stroke:#6688aa;stroke-width:0.1}'
           '</style>']

    def ring(pts, cls):
        d = " ".join(f"{tx(x):.3f},{ty(y):.3f}" for x, y in pts)
        return f'<polygon class="{cls}" points="{d}"/>'

    out.append(ring(board, "outline"))
    out.append(ring(drawn, "sensor-outline"))
    for cx, cy in centres:
        out.append(f'<rect class="pixel" x="{tx(cx) - hw:.3f}" y="{ty(cy) - hh:.3f}" '
                   f'width="{2 * hw:.3f}" height="{2 * hh:.3f}"/>')
    out.append("</svg>")
    return "".join(out)
