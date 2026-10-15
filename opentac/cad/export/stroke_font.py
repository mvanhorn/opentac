"""A minimal single-stroke font, so the Gerber writer can draw text.

Gerber has no text primitive worth using: an aperture-based format draws
strokes, and a fab's viewer will not render a font for you. KiCad solves this
by shipping its own stroke font and plotting glyphs as line segments, which is
exactly what `export/gerber.py` has to do for the F.Silkscreen layer.

Doing that needs a glyph table, and this is the smallest one that covers the
job. The design name is already sanitised to `[A-Za-z0-9_-]` before it gets
here (see `silk_label.sanitize_name`), and silkscreen labels are conventionally
uppercase, so folding case leaves 38 glyphs instead of 64.

Glyphs are polylines on a 6-wide, 10-tall integer grid with the baseline at
y=0, normalised on the way out so that one unit of `height_mm` is the cap
height. The advance is 8 grid units -- a 6-wide glyph and a 2-wide gap -- and
runs a little loose next to KiCad's own metrics, deliberately: `silk_label`
measures with `text_width` to decide whether a name fits across the tail, and
erring wide costs a smaller label rather than one that runs off the board.
"""

GRID_H = 10.0     # cap height in grid units
GLYPH_W = 6.0     # widest a glyph gets, grid units
ADVANCE = 8.0     # pen advance per character, grid units

# Each entry is a list of polylines; each polyline a list of (x, y) grid points.
_GLYPHS = {
    "A": [[(0, 0), (3, 10), (6, 0)], [(1.2, 4), (4.8, 4)]],
    "B": [[(0, 0), (0, 10), (4, 10), (6, 8.5), (6, 6.5), (4, 5), (0, 5)],
          [(4, 5), (6, 3.5), (6, 1.5), (4, 0), (0, 0)]],
    "C": [[(6, 8), (4, 10), (2, 10), (0, 8), (0, 2), (2, 0), (4, 0), (6, 2)]],
    "D": [[(0, 0), (0, 10), (3, 10), (6, 7), (6, 3), (3, 0), (0, 0)]],
    "E": [[(6, 10), (0, 10), (0, 0), (6, 0)], [(0, 5), (4, 5)]],
    "F": [[(6, 10), (0, 10), (0, 0)], [(0, 5), (4, 5)]],
    "G": [[(6, 8), (4, 10), (2, 10), (0, 8), (0, 2), (2, 0), (4, 0), (6, 2), (6, 5), (3, 5)]],
    "H": [[(0, 10), (0, 0)], [(6, 10), (6, 0)], [(0, 5), (6, 5)]],
    "I": [[(1, 10), (5, 10)], [(3, 10), (3, 0)], [(1, 0), (5, 0)]],
    "J": [[(2, 10), (6, 10)], [(4, 10), (4, 2), (2, 0), (0, 2)]],
    "K": [[(0, 10), (0, 0)], [(6, 10), (0, 4)], [(2, 6), (6, 0)]],
    "L": [[(0, 10), (0, 0), (6, 0)]],
    "M": [[(0, 0), (0, 10), (3, 5), (6, 10), (6, 0)]],
    "N": [[(0, 0), (0, 10), (6, 0), (6, 10)]],
    "O": [[(2, 10), (4, 10), (6, 8), (6, 2), (4, 0), (2, 0), (0, 2), (0, 8), (2, 10)]],
    "P": [[(0, 0), (0, 10), (4, 10), (6, 8.5), (6, 6.5), (4, 5), (0, 5)]],
    "Q": [[(2, 10), (4, 10), (6, 8), (6, 2), (4, 0), (2, 0), (0, 2), (0, 8), (2, 10)],
          [(3.5, 2.5), (6.5, -0.5)]],
    "R": [[(0, 0), (0, 10), (4, 10), (6, 8.5), (6, 6.5), (4, 5), (0, 5)], [(3, 5), (6, 0)]],
    "S": [[(6, 8), (4, 10), (2, 10), (0, 8.5), (0, 6.5), (2, 5), (4, 5),
           (6, 3.5), (6, 1.5), (4, 0), (2, 0), (0, 2)]],
    "T": [[(0, 10), (6, 10)], [(3, 10), (3, 0)]],
    "U": [[(0, 10), (0, 2), (2, 0), (4, 0), (6, 2), (6, 10)]],
    "V": [[(0, 10), (3, 0), (6, 10)]],
    "W": [[(0, 10), (1.5, 0), (3, 7), (4.5, 0), (6, 10)]],
    "X": [[(0, 10), (6, 0)], [(0, 0), (6, 10)]],
    "Y": [[(0, 10), (3, 5), (6, 10)], [(3, 5), (3, 0)]],
    "Z": [[(0, 10), (6, 10), (0, 0), (6, 0)]],
    # A slashed zero: these names mix letters and digits ("sensor_20260908"),
    # and O/0 at 1mm on a screen-printed flex is otherwise a coin toss.
    "0": [[(2, 10), (4, 10), (6, 8), (6, 2), (4, 0), (2, 0), (0, 2), (0, 8), (2, 10)],
          [(1, 2), (5, 8)]],
    "1": [[(1, 8), (3, 10), (3, 0)], [(1, 0), (5, 0)]],
    "2": [[(0, 8), (2, 10), (4, 10), (6, 8), (6, 6), (0, 0), (6, 0)]],
    "3": [[(0, 9), (2, 10), (4, 10), (6, 8.5), (6, 6.5), (4, 5), (2.5, 5)],
          [(4, 5), (6, 3.5), (6, 1.5), (4, 0), (2, 0), (0, 1)]],
    "4": [[(4, 0), (4, 10), (0, 3), (6, 3)]],
    "5": [[(6, 10), (0, 10), (0, 5.5), (4, 5.5), (6, 4), (6, 2), (4, 0), (2, 0), (0, 1)]],
    "6": [[(5, 10), (2, 10), (0, 7), (0, 2), (2, 0), (4, 0), (6, 2), (6, 4),
           (4, 6), (2, 6), (0, 4)]],
    "7": [[(0, 10), (6, 10), (2, 0)]],
    "8": [[(2, 5), (0, 6.5), (0, 8.5), (2, 10), (4, 10), (6, 8.5), (6, 6.5), (4, 5),
           (2, 5), (0, 3.5), (0, 1.5), (2, 0), (4, 0), (6, 1.5), (6, 3.5), (4, 5)]],
    "9": [[(1, 0), (4, 0), (6, 3), (6, 8), (4, 10), (2, 10), (0, 8), (0, 6),
           (2, 4), (4, 4), (6, 6)]],
    "_": [[(0, -1.5), (6, -1.5)]],
    "-": [[(1, 5), (5, 5)]],
}


def supported(ch: str) -> bool:
    return ch.upper() in _GLYPHS


def text_width(text: str, height_mm: float) -> float:
    """Inked width of `text` at the given cap height, in mm.

    Left edge of the first glyph to right edge of the last -- the trailing
    inter-glyph gap is not part of it, so a label centred on this width is
    optically centred rather than sitting a fraction to the left.

    Every character advances, drawable or not, so a stray unsupported one
    leaves a gap rather than closing it up.
    """
    if not text:
        return 0.0
    grid = (len(text) - 1) * ADVANCE + GLYPH_W
    return grid / GRID_H * height_mm


def text_strokes(text: str, height_mm: float) -> list[list[tuple[float, float]]]:
    """Polylines for `text`, in mm, with the origin at its left baseline.

    +X runs along the text, +Y is up. The caller rotates and translates.
    """
    s = height_mm / GRID_H
    out = []
    pen = 0.0
    for ch in text:
        for poly in _GLYPHS.get(ch.upper(), ()):
            out.append([((pen + x) * s, y * s) for x, y in poly])
        pen += ADVANCE
    return out
