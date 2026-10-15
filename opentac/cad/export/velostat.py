"""Velostat cut files: a laser-cutter DXF and a fold-over paper template.

What gets cut. The outline the user drew (the manifest's "outline"), not the
manufactured board: it contains every taxel and never the cable tail. See
`velostat_polygon` for why the board-minus-tail version was dropped. Anchor-hole
cutouts are punched through too: the bump-sheet posts pass through the board
there, and a Velostat sheet without the holes would block every post.

The fold template. Velostat is thin, static-clingy and curls, and a pen line on
it is hard to follow with scissors. The PDF prints the shape twice, mirrored
across a dashed crease. Fold the paper on the crease with the Velostat inside:
the two outlines land exactly on top of each other, so the film is clamped flat
between two sheets that both show the cut line, and you cut paper and film in
one pass from whichever side you like.

Mirroring is what makes that work, not a detail. Folding reflects one half onto
the other. A plain copy would land flipped relative to its partner, and on any
shape that is not symmetric the two outlines would not line up at all.
"""

from __future__ import annotations

import io

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .print_pdf import (INK, FAINT, MARGIN_MM, PAGES, MM_TO_PT, _Content,
                        _build_pdf, _draw_scale_check, _page_furniture)

CREASE_GAP_MM = 4.0      # shape <-> crease, each side; room to cut without nicking the fold
FOOTER_MM = 16.0         # matches print_pdf: the scale ruler and its caption
HEADER_MM = 10.0


def velostat_polygon(layout, outline_xy, inset_mm: float = 0.0) -> Polygon:
    """The Velostat piece, in board coordinates (mm): the outline the user drew.

    Not the manufactured board. That one carries the cable tail and connector,
    and cutting the tail back off is fragile: clipping at the cable-edge line
    threw away whole sensor arms whenever the cable sat on an inner edge (14 of
    128 boards in a sweep over every edge of the site's presets). The drawn
    outline has no tail by construction and every taxel is packed inside it,
    whichever edge the cable is on. It ends a few mm inside the board edge,
    over the routing margin, where there is nothing to sense.

    `inset_mm` > 0 shrinks it further. The anchor holes are not shrunk with it:
    they are for a post that is the same size whatever the outer margin is.
    """
    body = Polygon(np.asarray(outline_xy, dtype=float)).buffer(0)
    body = max(getattr(body, "geoms", [body]), key=lambda g: g.area)
    if inset_mm > 0:
        shrunk = body.buffer(-inset_mm, join_style="mitre")
        if not shrunk.is_empty:
            body = max(getattr(shrunk, "geoms", [shrunk]), key=lambda g: g.area)

    anchors = (getattr(layout, "routing", None) or {}).get("anchor_holes") or {}
    centers = anchors.get("centers") or []
    dia = float(anchors.get("dia_mm", 0.0))
    if centers and dia > 0:
        holes = [Point(c).buffer(dia / 2.0, quad_segs=16) for c in centers]
        body = body.difference(unary_union(holes))
        if not isinstance(body, Polygon):
            body = max(body.geoms, key=lambda g: g.area)
    return body.buffer(0)


def _rings(poly: Polygon):
    """Exterior first, then holes; each an open list of (x, y) without the repeat."""
    rings = [poly.exterior] + list(poly.interiors)
    return [list(r.coords)[:-1] for r in rings]


def velostat_dxf(poly: Polygon) -> str:
    """One closed LWPOLYLINE per ring, in millimetres, on layer VELOSTAT_CUT.

    Holes are separate closed loops rather than a hatch or region: a laser
    cutter wants cut paths, and every cutter's import treats a closed polyline
    as exactly one.
    """
    import ezdxf
    from ezdxf import units

    doc = ezdxf.new("R2010")
    doc.units = units.MM
    doc.header["$MEASUREMENT"] = 1   # metric
    doc.layers.add("VELOSTAT_CUT", color=1)
    msp = doc.modelspace()
    # Shift so the piece sits at the origin: board coordinates can be anywhere,
    # and a cutter that imports at absolute position would put it off the bed.
    minx, miny, _, _ = poly.bounds
    for ring in _rings(poly):
        msp.add_lwpolyline([(x - minx, y - miny) for x, y in ring],
                           close=True, dxfattribs={"layer": "VELOSTAT_CUT"})
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue()


def _mirror_rings(rings, axis: str, about: float):
    if axis == "x":      # mirror across the vertical line x = about
        return [[(2 * about - x, y) for x, y in r] for r in rings]
    return [[(x, 2 * about - y) for x, y in r] for r in rings]


def velostat_fold_pdf(poly: Polygon, title="OpenTactileCAD Velostat fold template",
                      info=None) -> bytes | None:
    """Shape + its mirror image across a crease, at 1:1.

    Tries a vertical crease (halves side by side) and a horizontal one (halves
    stacked) on each sheet size in print_pdf.PAGES and takes the first fit.
    Every fold layout on every sheet is tried before giving up on folding, so
    an A3 fold wins over a single copy on A4.

    If the doubled shape fits no sheet, the page carries the shape ONCE, with
    no mirror and no crease: tape the Velostat under the sheet and cut along
    the line. `info["fold"]` is None in that case. Returns None only when even
    one copy fits no single sheet; the DXF still covers that, and a tiled
    template would have to be cut as separate pieces anyway.
    """
    rings = _rings(poly)
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny

    candidates = [(fold, page) for fold in ("vertical", "horizontal") for page in PAGES]
    candidates.sort(key=lambda fp: PAGES.index(fp[1]))   # smallest sheet first
    candidates += [(None, page) for page in PAGES]        # single copy, last resort

    for fold, (name, pw, ph) in candidates:
        avail_w = pw - 2 * MARGIN_MM
        avail_h = ph - 2 * MARGIN_MM - FOOTER_MM - HEADER_MM
        need_w = 2 * (w + CREASE_GAP_MM) if fold == "vertical" else w
        need_h = 2 * (h + CREASE_GAP_MM) if fold == "horizontal" else h
        if need_w > avail_w or need_h > avail_h:
            continue
        # Page-space mm: centre the layout, crease (if any) through the middle.
        area_x0 = MARGIN_MM
        area_y0 = MARGIN_MM + FOOTER_MM
        cx = area_x0 + avail_w / 2.0
        cy = area_y0 + avail_h / 2.0
        c = _Content(0.0, 0.0)
        crease = None
        if fold == "vertical":
            # Original on the left, its right edge CREASE_GAP_MM from the crease.
            dx = cx - CREASE_GAP_MM - maxx
            dy = cy - (miny + h / 2.0)
            first = [[(x + dx, y + dy) for x, y in r] for r in rings]
            halves = [first, _mirror_rings(first, "x", cx)]
            crease = ((cx, area_y0), (cx, area_y0 + avail_h))
        elif fold == "horizontal":
            # Original on top, its bottom edge CREASE_GAP_MM above the crease.
            dx = cx - (minx + w / 2.0)
            dy = cy + CREASE_GAP_MM - miny
            first = [[(x + dx, y + dy) for x, y in r] for r in rings]
            halves = [first, _mirror_rings(first, "y", cy)]
            crease = ((area_x0, cy), (area_x0 + avail_w, cy))
        else:
            dx = cx - (minx + w / 2.0)
            dy = cy - (miny + h / 2.0)
            halves = [[[(x + dx, y + dy) for x, y in r] for r in rings]]

        _page_furniture(
            c, title if crease else "OpenTactileCAD Velostat cutting template",
            f"Piece {w:.1f} x {h:.1f} mm  |  1:1 - print at 100% (Actual size)  |  sheet: {name}",
            pw, ph)
        c.text(MARGIN_MM, ph - MARGIN_MM - 8.5,
               ("Fold along the dashed line, printed side out, with the Velostat inside. "
                "Both outlines line up; cut through paper and film together.") if crease else
               ("Too large to fold on one sheet. Tape the Velostat flat under this sheet "
                "and cut through paper and film together along the line."),
               7.5, INK, page_space=True)

        if crease:
            c.stroke(FAINT, 0.25, dash=(3.0, 2.0))
            c.line(*crease)
            label_at = ((crease[0][0] + 1.0, crease[1][1] - 4.0) if fold == "vertical"
                        else (crease[0][0], crease[0][1] + 1.0))
            c.text(label_at[0], label_at[1], "FOLD", 7.0, FAINT)

        c.stroke(INK, 0.35)
        for half in halves:
            for ring in half:
                c.polyline(ring, close=True)
        _draw_scale_check(c, MARGIN_MM, MARGIN_MM)
        if info is not None:
            info.update(piece_w_mm=round(w, 1), piece_h_mm=round(h, 1),
                        sheet=name, fold=fold, pages=1)
        return _build_pdf([(pw, ph, c.render())])
    if info is not None:
        info.update(piece_w_mm=round(w, 1), piece_h_mm=round(h, 1),
                    sheet=None, fold=None, pages=0)
    return None
