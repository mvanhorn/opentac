"""KiCad .kicad_mod footprint generation for tactile pixels."""

import math
import re
from opentac.cad.pipeline.types import PixelGeometry
from opentac.cad.export.coverlay import COVERLAY_MARGIN_MM, TaxelOpening, copper_half_extent


def fmt(v: float) -> str:
    """Format number for KiCad — remove trailing zeros."""
    s = f"{v:.6f}"
    s = s.rstrip('0').rstrip('.')
    return s if s else "0"


def name_num(v: float, decimals: int) -> str:
    """Format number for footprint name."""
    s = f"{v:.{decimals}f}"
    s = s.rstrip('0').rstrip('.')
    if not s:
        s = "0"
    s = s.replace('.', 'p')
    if s.startswith('-'):
        s = 'm' + s[1:]
    return s


def pixel_geometry(w, h, tracew, gap, center_clear, edge_clear, via_dia) -> PixelGeometry:
    """Compute pixel footprint geometry."""
    ix1 = -w / 2 + edge_clear
    iy1 = -h / 2 + edge_clear
    ix2 = w / 2 - edge_clear
    iy2 = h / 2 - edge_clear

    inner_w = ix2 - ix1
    inner_h = iy2 - iy1
    if inner_w <= tracew or inner_h <= tracew:
        raise ValueError("edgeClearMm too large for given pixel size/trace width")

    xL = ix1 + tracew / 2
    xR = ix2 - tracew / 2
    ymid = (iy1 + iy2) / 2

    pitch = tracew + gap
    total_slots = max(1, int((inner_h + gap) / pitch))
    span = total_slots * tracew + (total_slots - 1) * gap
    y0 = (iy1 + iy2 - span) / 2 + tracew / 2

    spine_y_lo = iy1 + tracew / 2
    spine_y_hi = iy2 - tracew / 2

    x_end_left_max = xR - (center_clear + tracew)
    x_end_right_min = xL + (center_clear + tracew)

    via_x = ix1 + via_dia / 2
    via_y = ymid

    # A column finger stops center_clear short of the row spine, but the row
    # via is wider than the spine and protrudes toward the fingers, so a
    # column finger at the via's height would pass it closer than center_clear.
    # Such a finger alone stops center_clear short of the via ring (+1 um for
    # the 6-decimal output); a finger that already clears it is unchanged.
    reach = via_dia / 2 + tracew / 2 + center_clear + 0.001
    col_end_x = []
    for i in range(total_slots):
        dy = abs(y0 + i * pitch - via_y)
        x_end = x_end_right_min
        if i % 2 == 1 and dy < reach:
            x_end = min(xR, max(x_end, via_x + math.sqrt(reach * reach - dy * dy)))
        col_end_x.append(x_end)

    return PixelGeometry(
        xL=xL, xR=xR, ymid=ymid,
        pitch=pitch, y0=y0,
        spine_y_lo=spine_y_lo, spine_y_hi=spine_y_hi,
        total_slots=total_slots,
        x_end_left_max=x_end_left_max,
        x_end_right_min=x_end_right_min,
        via_x=via_x, via_y=via_y,
        col_end_x=col_end_x,
    )


def make_footprint_name(prefix, w, h, tracew, gap, center_clear, edge_clear, via_drill, via_dia, decimals=3):
    """Generate footprint name."""
    return (
        f"{prefix}"
        f"_w{name_num(w, decimals)}"
        f"_h{name_num(h, decimals)}"
        f"_tw{name_num(tracew, decimals)}"
        f"_g{name_num(gap, decimals)}"
        f"_cc{name_num(center_clear, decimals)}"
        f"_ec{name_num(edge_clear, decimals)}"
        f"_vd{name_num(via_drill, decimals)}"
        f"_v{name_num(via_dia, decimals)}"
    )


def lone_taxel_opening(w, h, edge_clear) -> TaxelOpening:
    """The mask opening of a taxel with no neighbours: the electrode copper
    plus COVERLAY_MARGIN_MM all round (a board's own comes from
    coverlay.taxel_opening, which knows the pitch)."""
    hx, hy = copper_half_extent(w, h, edge_clear)
    m = COVERLAY_MARGIN_MM
    return TaxelOpening(margin_x=m, margin_y=m, half_w=round(hx + m, 6), half_h=round(hy + m, 6))


def mask_opening_line(opening: TaxelOpening, indent: str = "  ") -> str:
    """The taxel's one F.Mask opening, a filled rectangle in the footprint.
    The electrode pads and the via carry no mask layer (export/coverlay.py)."""
    x1, y1, x2, y2 = opening.rect()
    return (f'{indent}(fp_rect (start {fmt(x1)} {fmt(y1)}) (end {fmt(x2)} {fmt(y2)}) '
            f'(stroke (width 0) (type solid)) (fill solid) (layer "F.Mask"))')


# The taxel's one mask opening exposes both electrodes (two nets) by design.
PIXEL_ATTR = '(attr smd allow_soldermask_bridges)'


def build_pixel_footprint_text(
    name, w, h, tracew, gap, center_clear, edge_clear, via_drill, via_dia, opening=None
) -> tuple[str, int]:
    """Build KiCad footprint S-expression text. Returns (text, total_slots).

    `opening` is the board's coverlay.TaxelOpening; without one the footprint
    gets lone_taxel_opening()."""
    g = pixel_geometry(w, h, tracew, gap, center_clear, edge_clear, via_dia)
    if opening is None:
        opening = lone_taxel_opening(w, h, edge_clear)

    x1 = -w / 2
    y1 = -h / 2
    x2 = w / 2
    y2 = h / 2

    def gr_poly_square(cx, cy, sz=0.04):
        r = sz / 2
        return (f"(gr_poly (pts (xy {fmt(cx-r)} {fmt(cy-r)}) (xy {fmt(cx+r)} {fmt(cy-r)}) "
                f"(xy {fmt(cx+r)} {fmt(cy+r)}) (xy {fmt(cx-r)} {fmt(cy+r)})) (width 0) (fill yes))")

    def gr_line(x1, y1, x2, y2, width):
        return f"(gr_line (start {fmt(x1)} {fmt(y1)}) (end {fmt(x2)} {fmt(y2)}) (width {fmt(width)}))"

    # Pad 1 primitives (relative to anchor at (xL, ymid))
    prim1 = [gr_poly_square(0, 0)]
    prim1.append(gr_line(0, g.spine_y_lo - g.ymid, 0, g.spine_y_hi - g.ymid, tracew))
    for i in range(g.total_slots):
        if i % 2 == 0:
            y = g.y0 + i * g.pitch
            prim1.append(gr_line(0, y - g.ymid, g.x_end_left_max - g.xL, y - g.ymid, tracew))

    # Pad 2 primitives (relative to anchor at (xR, ymid))
    prim2 = [gr_poly_square(0, 0)]
    prim2.append(gr_line(0, g.spine_y_lo - g.ymid, 0, g.spine_y_hi - g.ymid, tracew))
    for i in range(g.total_slots):
        if i % 2 == 1:
            y = g.y0 + i * g.pitch
            prim2.append(gr_line(0, y - g.ymid, g.col_end_x[i] - g.xR, y - g.ymid, tracew))

    def pad_custom(padnum, atx, aty, primitives):
        header = (f'(pad "{padnum}" smd custom (at {fmt(atx)} {fmt(aty)}) (size 0.05 0.05) '
                  f'(layers "F.Cu") (options (clearance outline) (anchor circle)) (primitives\n')
        body = '\n'.join(f'  {p}' for p in primitives)
        return header + body + '\n))'

    def pad_via(padnum, x, y, drill, dia):
        return (f'(pad "{padnum}" thru_hole circle (at {fmt(x)} {fmt(y)}) '
                f'(size {fmt(dia)} {fmt(dia)}) (drill {fmt(drill)}) (layers "*.Cu"))')

    lines = [
        f'(footprint "{name}"',
        '  (layer "F.Cu")',
        f'  {PIXEL_ATTR}',
        f'  (fp_rect (start {fmt(x1)} {fmt(y1)}) (end {fmt(x2)} {fmt(y2)}) '
        f'(stroke (width 0.1) (type default)) (fill no) (layer "User.2"))',
        mask_opening_line(opening),
        f'  {pad_custom(1, g.xL, g.ymid, prim1)}',
        f'  {pad_custom(2, g.xR, g.ymid, prim2)}',
        f'  {pad_via(1, g.via_x, g.via_y, via_drill, via_dia)}',
        ')',
    ]

    return '\n'.join(lines), g.total_slots
