"""A 1:1 printable PDF of the board outline, for cutting and test-fitting.

Why this exists: the Gerbers are for a fab house, and nothing in the ZIP let
you check a shape against the real part before paying for a board. Printing
this, cutting it out and holding it against the gripper takes a minute and
catches a wrong dimension that would otherwise arrive as 20 useless PCBs.

Why a hand-rolled writer instead of reportlab: PDF vector line art is a small,
stable format -- paths, a stroke colour, one Type1 font -- and the one thing
this file must get exactly right is scale. Owning the transform end to end is
worth more here than a dependency, and this project has already shipped one
outage from an unpinned transitive import that only failed in production.

Everything is emitted in PDF user space, where 1 unit = 1/72 inch, so the
millimetre -> unit factor below IS the 1:1 guarantee.
"""

from __future__ import annotations

import datetime

MM_TO_PT = 72.0 / 25.4

# Page sizes in mm, tried in this order; the first that fits wins.
PAGES = [
    ("A4 portrait", 210.0, 297.0),
    ("A4 landscape", 297.0, 210.0),
    ("A3 portrait", 297.0, 420.0),
    ("A3 landscape", 420.0, 297.0),
]
TILE_PAGE = ("A4 portrait", 210.0, 297.0)   # used when nothing single-page fits
MARGIN_MM = 12.0
RULER_MM = 100.0                            # printed scale check


def _esc(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


class _Content:
    """Accumulates one page's content stream, in millimetres."""

    def __init__(self, origin_x_mm: float, origin_y_mm: float):
        # Where board (0,0) sits on the page, in mm from the page's lower-left.
        self.ox = origin_x_mm
        self.oy = origin_y_mm
        self.parts: list[str] = []

    def _pt(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        return ((x_mm + self.ox) * MM_TO_PT, (y_mm + self.oy) * MM_TO_PT)

    def stroke(self, rgb, width_mm=0.25, dash=None):
        r, g, b = rgb
        self.parts.append(f"{r:.3f} {g:.3f} {b:.3f} RG")
        self.parts.append(f"{width_mm * MM_TO_PT:.3f} w")
        if dash:
            on, off = dash
            self.parts.append(f"[{on * MM_TO_PT:.2f} {off * MM_TO_PT:.2f}] 0 d")
        else:
            self.parts.append("[] 0 d")

    def fill_color(self, rgb):
        r, g, b = rgb
        self.parts.append(f"{r:.3f} {g:.3f} {b:.3f} rg")

    def line_page(self, a, b):
        """A line positioned on the PAGE, unaffected by where the board sits.

        Page furniture -- the scale ruler, tiling guides -- must not ride on the
        board origin: on a tiled sheet that origin is far off-page, and drawing
        the ruler in board space put it outside the media box entirely on two
        of four sheets, silently removing the only check that the print came
        out at true size.
        """
        self.parts.append(f"{a[0] * MM_TO_PT:.3f} {a[1] * MM_TO_PT:.3f} m "
                          f"{b[0] * MM_TO_PT:.3f} {b[1] * MM_TO_PT:.3f} l S")

    def line(self, a, b):
        x1, y1 = self._pt(a[0], a[1])
        x2, y2 = self._pt(b[0], b[1])
        self.parts.append(f"{x1:.3f} {y1:.3f} m {x2:.3f} {y2:.3f} l S")

    def polyline(self, pts, close=False):
        if len(pts) < 2:
            return
        x, y = self._pt(pts[0][0], pts[0][1])
        self.parts.append(f"{x:.3f} {y:.3f} m")
        for p in pts[1:]:
            x, y = self._pt(p[0], p[1])
            self.parts.append(f"{x:.3f} {y:.3f} l")
        self.parts.append("h S" if close else "S")

    def rect(self, x_mm, y_mm, w_mm, h_mm, fill=False):
        x, y = self._pt(x_mm, y_mm)
        self.parts.append(
            f"{x:.3f} {y:.3f} {w_mm * MM_TO_PT:.3f} {h_mm * MM_TO_PT:.3f} re "
            + ("f" if fill else "S"))

    def text(self, x_mm, y_mm, s, size_pt=8.0, rgb=(0, 0, 0), page_space=False):
        """page_space=True places text relative to the PAGE, not the board."""
        if page_space:
            x, y = x_mm * MM_TO_PT, y_mm * MM_TO_PT
        else:
            x, y = self._pt(x_mm, y_mm)
        r, g, b = rgb
        self.parts.append(
            f"BT {r:.3f} {g:.3f} {b:.3f} rg /F1 {size_pt:.2f} Tf "
            f"{x:.3f} {y:.3f} Td ({_esc(s)}) Tj ET")

    def render(self) -> str:
        return "\n".join(self.parts)


def _build_pdf(pages) -> bytes:
    """pages: list of (width_mm, height_mm, content_stream_str)."""
    objects: list[bytes] = []          # 1-indexed on output

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    font_id = None
    page_ids: list[int] = []
    content_ids: list[int] = []
    # Reserve: 1 = catalog, 2 = pages tree. Fill them in after the children.
    objects.append(b"")                # placeholder 1
    objects.append(b"")                # placeholder 2
    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
                  b"/Encoding /WinAnsiEncoding >>")

    for _w, _h, stream in pages:
        data = stream.encode("latin-1", "replace")
        content_ids.append(add(b"<< /Length " + str(len(data)).encode() + b" >>\nstream\n"
                               + data + b"\nendstream"))
    for (w_mm, h_mm, _s), cid in zip(pages, content_ids):
        page_ids.append(add(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 "
            f"{w_mm * MM_TO_PT:.3f} {h_mm * MM_TO_PT:.3f}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {cid} 0 R >>".encode()))

    kids = " ".join(f"{i} 0 R" for i in page_ids)
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0] * (len(objects) + 1)
    for i, body in enumerate(objects, start=1):
        offsets[i] = len(out)
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for i in range(1, len(objects) + 1):
        out += f"{offsets[i]:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF\n").encode()
    return bytes(out)


INK = (0.10, 0.10, 0.10)
TAXEL = (0.25, 0.45, 0.80)
FAINT = (0.62, 0.62, 0.62)
ALERT = (0.75, 0.20, 0.20)


def _segments_to_polylines(segments, tol=1e-6):
    """Chain loose (a, b) edge segments into ordered polylines.

    Edge.Cuts come back as an unordered soup of segments; drawing them
    individually works but leaves visible seams at every joint when a printer
    renders thin strokes, and it makes the cut line harder to follow by eye.
    """
    remaining = [(tuple(map(float, a)), tuple(map(float, b))) for a, b in segments]
    chains = []
    while remaining:
        a, b = remaining.pop(0)
        chain = [a, b]
        grew = True
        while grew:
            grew = False
            for i, (p, q) in enumerate(remaining):
                if abs(p[0] - chain[-1][0]) < tol and abs(p[1] - chain[-1][1]) < tol:
                    chain.append(q); remaining.pop(i); grew = True; break
                if abs(q[0] - chain[-1][0]) < tol and abs(q[1] - chain[-1][1]) < tol:
                    chain.append(p); remaining.pop(i); grew = True; break
                if abs(q[0] - chain[0][0]) < tol and abs(q[1] - chain[0][1]) < tol:
                    chain.insert(0, p); remaining.pop(i); grew = True; break
                if abs(p[0] - chain[0][0]) < tol and abs(p[1] - chain[0][1]) < tol:
                    chain.insert(0, q); remaining.pop(i); grew = True; break
        chains.append(chain)
    return chains


def _draw_scale_check(c, x_mm, y_mm):
    """A printed 100 mm ruler.

    Not decoration: every print dialog defaults to "fit to page" or "shrink to
    printable area", which silently scales the sheet by a few percent. A board
    that measures 97 mm across this ruler was printed at 97%, and the cut-out
    is wrong in a way nothing on screen would reveal.
    """
    c.stroke(INK, 0.3)
    c.line_page((x_mm, y_mm), (x_mm + RULER_MM, y_mm))
    for i in range(0, int(RULER_MM) + 1, 10):
        tall = 3.0 if i % 50 == 0 else 2.0
        c.line_page((x_mm + i, y_mm), (x_mm + i, y_mm + tall))
    c.text(x_mm, y_mm + 4.5,
           f"Scale check: this line must measure exactly {RULER_MM:.0f} mm. "
           "If it does not, reprint at 100% / Actual size.", 7.5, INK, page_space=True)


def _page_furniture(c, title, subtitle, page_w, page_h, page_label=None):
    c.text(MARGIN_MM, page_h - MARGIN_MM + 2.0, title, 10.5, INK, page_space=True)
    c.text(MARGIN_MM, page_h - MARGIN_MM - 3.5, subtitle, 7.5, FAINT, page_space=True)
    if page_label:
        c.text(page_w - MARGIN_MM - 40.0, page_h - MARGIN_MM + 2.0,
               page_label, 8.5, INK, page_space=True)


def _draw_board(c, chains, taxels, pixel_w, pixel_h, show_taxels=True):
    if show_taxels:
        c.stroke(TAXEL, 0.12)
        for (x, y) in taxels:
            c.rect(x - pixel_w / 2.0, y - pixel_h / 2.0, pixel_w, pixel_h)
    # Cut line last so it sits on top of the taxel grid.
    #
    # Every chain is closed. With the connector's own edges joined in (see
    # generate_print_pdf) the outline is already one loop; this is the safety
    # net for a footprint whose edges do not quite meet. An open loop is a
    # shape you cannot cut out -- the tail once printed as two dangling stubs
    # with a hole between them.
    c.stroke(INK, 0.35)
    for chain in chains:
        closed = (abs(chain[0][0] - chain[-1][0]) < 1e-6
                  and abs(chain[0][1] - chain[-1][1]) < 1e-6)
        c.polyline(chain[:-1] if closed else chain, close=True)


def generate_print_pdf(layout, outline_xy, params, removed, connector_dir,
                       title="OpenTac CAD sensor", info=None) -> bytes:
    """Build the 1:1 cutting template.

    Draws the board's REAL manufactured boundary (Edge.Cuts, connector body
    included), not the outline the user drew -- the two differ by the tail
    and connector, and cutting out the drawn shape would test the wrong thing.

    Pass a dict as `info` to receive what was actually drawn -- board_w_mm,
    board_h_mm, pages, sheet. Callers that report size to the user must read it
    from here rather than measuring the outline themselves, or the number on
    screen disagrees with the number printed on the sheet.
    """
    from .connector_geometry import connector_edge_geometries, stitched_outline_segments
    from .gerber import _arc_to_segments
    import numpy as np

    segments = stitched_outline_segments(layout, outline_xy, params, connector_dir)
    if segments:
        # The board stops at the connector's open edge and the footprint
        # draws the rest of Edge.Cuts, as in the Gerber and KiCad exporters.
        # Without these the template ended at that edge and printed every
        # board short by the connector body: 2.25 mm on a 20-pin, 6 mm on a 70.
        edges = connector_edge_geometries(layout, outline_xy, params, connector_dir)
        segments = list(segments) + list(edges["lines"])
        for start, mid, end in edges["arcs"]:
            pts = _arc_to_segments(start, mid, end, n_segs=12)
            pts[0], pts[-1] = tuple(start), tuple(end)   # exact, so the chain joins
            segments += list(zip(pts, pts[1:]))
    else:
        pts = np.asarray(getattr(layout, "final_outline_xy", None)
                         if getattr(layout, "final_outline_xy", None) is not None
                         else outline_xy, dtype=float)
        segments = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    # 0.01 mm, the tolerance connector_open_edge_endpoints joins footprint edges at.
    chains = _segments_to_polylines(segments, tol=0.01)

    P = np.asarray(layout.placements_xy, dtype=float)
    taxels = [tuple(P[i]) for i in range(len(P)) if i not in (removed or set())]

    xs = [p[0] for ch in chains for p in ch]
    ys = [p[1] for ch in chains for p in ch]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    board_w, board_h = maxx - minx, maxy - miny

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    subtitle = (f"Board {board_w:.1f} x {board_h:.1f} mm  |  {len(taxels)} sensors  |  "
                f"1:1 - print at 100% (Actual size), not Fit to page  |  {stamp}")

    # Room for the scale check and its caption along the bottom.
    footer_mm = 16.0
    for name, pw, ph in PAGES:
        avail_w = pw - 2 * MARGIN_MM
        avail_h = ph - 2 * MARGIN_MM - footer_mm
        if board_w <= avail_w and board_h <= avail_h:
            ox = MARGIN_MM + (avail_w - board_w) / 2.0 - minx
            oy = MARGIN_MM + footer_mm + (avail_h - board_h) / 2.0 - miny
            c = _Content(ox, oy)
            _page_furniture(c, title, f"{subtitle}  |  sheet: {name}", pw, ph)
            _draw_board(c, chains, taxels, params.pixel_w_mm, params.pixel_h_mm)
            _draw_scale_check(c, MARGIN_MM, MARGIN_MM)
            if info is not None:
                info.update(board_w_mm=round(board_w, 1), board_h_mm=round(board_h, 1),
                            pages=1, sheet=name)
            return _build_pdf([(pw, ph, c.render())])

    # Too big for any single sheet: tile it, with an overlap so the pieces can
    # be aligned and taped rather than butted edge to edge and hoped for.
    name, pw, ph = TILE_PAGE
    overlap = 10.0
    avail_w = pw - 2 * MARGIN_MM
    avail_h = ph - 2 * MARGIN_MM - footer_mm
    step_w, step_h = avail_w - overlap, avail_h - overlap
    n_cols = max(1, int(-(-board_w // step_w)))
    n_rows = max(1, int(-(-board_h // step_h)))
    pages = []
    for r in range(n_rows):
        for col in range(n_cols):
            ox = MARGIN_MM - minx - col * step_w
            oy = MARGIN_MM + footer_mm - miny - (n_rows - 1 - r) * step_h
            c = _Content(ox, oy)
            label = f"Sheet {r * n_cols + col + 1} of {n_rows * n_cols}  (row {r + 1}, col {col + 1})"
            _page_furniture(c, title, f"{subtitle}  |  sheet: {name}, tiled", pw, ph, label)
            # Clip to this sheet's window so neighbouring sheets do not each
            # draw the whole board and disagree about where the edge falls.
            c.parts.append("q")
            c.parts.append(f"{MARGIN_MM * MM_TO_PT:.3f} {(MARGIN_MM + footer_mm) * MM_TO_PT:.3f} "
                           f"{avail_w * MM_TO_PT:.3f} {avail_h * MM_TO_PT:.3f} re W n")
            _draw_board(c, chains, taxels, params.pixel_w_mm, params.pixel_h_mm)
            c.parts.append("Q")
            # Overlap guides: trim to the dashed line and butt the next sheet.
            c.stroke(FAINT, 0.2, dash=(2.0, 2.0))
            if col < n_cols - 1:
                x = MARGIN_MM + avail_w - overlap
                c.line_page((x, MARGIN_MM + footer_mm), (x, MARGIN_MM + footer_mm + avail_h))
            if r > 0:
                y = MARGIN_MM + footer_mm + avail_h - overlap
                c.line_page((MARGIN_MM, y), (MARGIN_MM + avail_w, y))
            _draw_scale_check(c, MARGIN_MM, MARGIN_MM)
            pages.append((pw, ph, c.render()))
    if info is not None:
        info.update(board_w_mm=round(board_w, 1), board_h_mm=round(board_h, 1),
                    pages=len(pages), sheet=f"{name}, tiled {n_cols}x{n_rows}")
    return _build_pdf(pages)
