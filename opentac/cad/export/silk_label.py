"""Where the design's name goes on the front silkscreen, and how big.

Without a printed name, a folder of five flex sensors is five near-identical
amber rectangles, and the only thing that tells them apart is the ZIP
filename -- which does not survive being unzipped, let alone being
fabricated. So the design name is printed on the board.

It goes on the tail, just inboard of the connector: the one place on every
board that is guaranteed to exist, guaranteed to be flat copper-and-mask
rather than taxels, and the place a person looks when plugging the board in.
Silkscreen sits above the soldermask, so printing over the fan-in traces
underneath it is fine; printing over the connector's own contact fingers is
not, which is what the clearance below is for.

The label is sized to fit and then, only if it still does not, truncated. It
is never allowed off the board: `board_poly` shrunk by the edge clearance is
the final arbiter, so a short tail or a narrow connector loses the label
rather than producing one that the router bit would cut through.

Callers: `export/kicad_pcb.py` (a gr_text on F.SilkS), `export/gerber.py`
(stroked into board-F_Silkscreen.gto by `stroke_font`), and
`export/svg_preview.py` so what is on screen is what is on the board.
"""

import re

import numpy as np
from shapely.geometry import Polygon

from opentac.cad.export.connector_geometry import connector_geometries, connector_pose
from opentac.cad.export.stroke_font import text_width

# Matches web/design-name.js. Kept in step with it deliberately: the
# name the user reads in the download field has to be the name on the board.
MAX_STEM = 48

# Cap heights tried, largest first. 0.8mm is a common fab minimum for
# screen-printed silk; going below it buys a longer label that cannot be read.
HEIGHTS_MM = (1.2, 1.0, 0.9, 0.8)
STROKE_W_MM = 0.15

# Gap between the connector's copper/outline and the label. Silk on a pad is a
# print defect (the fab clips it, or it lands on a contact finger).
CONNECTOR_CLEAR_MM = 0.6

# Nothing shorter than this is worth printing -- "sen" identifies no board.
MIN_CHARS = 4


def sanitize_name(raw: str) -> str:
    """The frontend's sanitiser, re-applied server-side.

    `design_name` arrives over HTTP and ends up in a `.kicad_pcb` string and a
    Gerber, so it is never trusted: anything outside `[A-Za-z0-9_-]` collapses
    to an underscore here exactly as it does in the browser.
    """
    s = re.sub(r"\.[A-Za-z0-9]{1,6}$", "", str(raw or ""))
    s = re.sub(r"[^A-Za-z0-9\-_]+", "_", s)
    s = re.sub(r"_{2,}", "_", s)
    s = s.strip("_-")
    return s[:MAX_STEM]


def _local_frame(model, layout, outline_xy, params):
    """(t_hat, n_out, origin, inner_y, center_x) for the connector.

    Local +X runs along the pad row, local +Y runs from the connector back
    into the board -- see `connector_geometry.local_to_world`, which maps
    local +Y to -n_out. `inner_y` is the board-side extreme of everything the
    connector occupies; `center_x` the middle of its pad row.
    """
    if model is not None:
        t_hat = model["t_hat"]
        n_out = model["n_out"]
        origin = model["pad1_world"]
        pts = []
        for pad in model["pads"]:
            pts.extend(np.asarray(pad["corners"], dtype=float))
        for a, b in model["edge_lines"]:
            pts.extend([np.asarray(a, dtype=float), np.asarray(b, dtype=float)])
        for a, m, b in model["edge_arcs"]:
            pts.extend([np.asarray(a, dtype=float), np.asarray(m, dtype=float),
                        np.asarray(b, dtype=float)])
        if pts:
            rel = np.asarray(pts, dtype=float) - origin
            lx = rel @ t_hat
            ly = -(rel @ n_out)
            return t_hat, n_out, origin, float(np.max(ly)), float((lx.min() + lx.max()) / 2)

    # No footprint on disk (an unsupported connector size, so nothing was
    # exported for it either). The routing pad row is still a real, known
    # place: pad 1 sits at the origin and the row runs one pitch per position.
    t_hat, n_out, origin = connector_pose(layout, outline_xy, params)
    span = max(0, int(layout.connector_pos) - 1) * float(params.pad_pitch_mm)
    return np.asarray(t_hat, float), np.asarray(n_out, float), np.asarray(origin, float), 0.0, span / 2


def _read_direction(t_hat):
    """Text direction, flipped so the label reads left-to-right.

    A connector on the top edge of a board has a `t_hat` pointing the other
    way from one on the bottom edge; following it blindly prints half the
    boards upside down. Restricting the angle to (-90, 90] is what KiCad's
    own text does.
    """
    d = np.asarray(t_hat, dtype=float)
    if d[0] < -1e-9 or (abs(d[0]) <= 1e-9 and d[1] < 0):
        d = -d
    return d


def silk_label(layout, outline_xy, params, connector_dir: str) -> dict | None:
    """Placement for the design-name label, or None if it cannot be printed.

    Returns the world-space centre, the reading direction, the chosen cap
    height, and the text actually used (which may be shorter than the name).
    """
    text = sanitize_name(getattr(params, "design_name", "") or "")
    if len(text) < MIN_CHARS:
        return None

    board = getattr(layout, "board_poly", None)
    if board is None or board.is_empty:
        return None
    # The label has to clear the cut edge by the same margin copper does, plus
    # half its own stroke width -- the stroke straddles its centreline.
    inset = float(params.board_edge_clear_mm) + STROKE_W_MM / 2
    safe = board.buffer(-inset)
    if safe.is_empty:
        return None

    model = connector_geometries(layout, outline_xy, params, connector_dir)
    t_hat, n_out, origin, inner_y, center_x = _local_frame(model, layout, outline_xy, params)
    d = _read_direction(t_hat)
    up = np.array([-d[1], d[0]], dtype=float)

    def place(s: str, h: float):
        """Centre and bounding box for `s` at cap height `h`, or None if off-board."""
        # Local +Y is inboard, and local_to_world maps it to -n_out.
        ly = inner_y + CONNECTOR_CLEAR_MM + h / 2
        center = origin + center_x * t_hat - ly * n_out
        w = text_width(s, h)
        # Descenders (the underscore) hang below the baseline, so the box is
        # measured a little taller than the cap height.
        half = np.array([w / 2, h * 0.75])
        corners = [center + sx * half[0] * d + sy * half[1] * up
                   for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        box = Polygon([tuple(c) for c in corners])
        return (center, box) if safe.contains(box) else None

    for h in HEIGHTS_MM:
        got = place(text, h)
        if got:
            return _label(text, got[0], d, h)

    # Still too wide at the smallest readable height: trim rather than drop it.
    h = HEIGHTS_MM[-1]
    for n in range(len(text) - 1, MIN_CHARS - 1, -1):
        got = place(text[:n], h)
        if got:
            return _label(text[:n], got[0], d, h)
    return None


def _label(text, center, d, height):
    return {
        "text": text,
        "center": np.asarray(center, dtype=float),
        "dir": np.asarray(d, dtype=float),
        "up": np.array([-d[1], d[0]], dtype=float),
        "height_mm": float(height),
        "thickness_mm": STROKE_W_MM,
        # KiCad's board space negates Y, so its angle does too -- the same
        # convention every other rotation in export/kicad_pcb.py uses.
        "rot_deg": float(np.degrees(np.arctan2(-d[1], d[0]))) + 0.0,
    }


def label_strokes(label: dict) -> list[list[tuple[float, float]]]:
    """The label as world-space polylines, for Gerber and for the SVG preview."""
    from opentac.cad.export.stroke_font import text_strokes

    h = label["height_mm"]
    w = text_width(label["text"], h)
    d, up, center = label["dir"], label["up"], label["center"]
    origin = center - (w / 2) * d - (h / 2) * up
    out = []
    for poly in text_strokes(label["text"], h):
        out.append([tuple(origin + x * d + y * up) for x, y in poly])
    return out
