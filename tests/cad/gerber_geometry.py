"""Read Gerber (RS-274X) and Excellon files back into geometry, for tests.

Two writers produce this project's manufacturing files: KiCad (kicad-cli plots
the generated .kicad_pcb) and export/gerber.py + export/excellon.py, the
pure-Python fallback the hosted service uses. Comparing their text says little
-- KiCad flashes aperture macros where the fallback draws strokes and regions
-- so the parity tests compare what the files draw: each layer as a shapely
geometry in the Gerber frame (mm, Y up), each drill file as a list of hits.

Covers what either writer emits: %FS/%MO, circle/rectangle/obround/polygon
apertures, aperture macros (primitives 1, 4, 5, 20, 21, arithmetic on $n),
D01/D02/D03 with modal coordinates, G01/G02/G03 under G75, G36/G37 regions
with several contours, and %LPD/%LPC polarity. Not a general Gerber reader:
step-and-repeat, block apertures and image transforms (%LM/%LR/%LS) raise.
"""

import math
import re

from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

# Arcs (strokes and region edges) are cut into chords no longer than this
# angle; at the 0.05-5 mm radii these files use the chord error is < 1 um.
ARC_STEP_RAD = math.radians(2.0)
CIRCLE_QUAD_SEGS = 32          # shapely buffer resolution for round shapes


class GerberError(ValueError):
    pass


def _statements(text):
    """(is_extended, body) for every statement, in order.

    Extended commands (%...%) can span lines and hold several '*'-terminated
    words (an aperture macro does); they are returned whole, without the %.
    """
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \r\n\t":
            i += 1
        elif c == "%":
            j = text.index("%", i + 1)
            out.append((True, text[i + 1:j]))
            i = j + 1
        else:
            j = text.index("*", i)
            out.append((False, text[i:j].strip()))
            i = j + 1
    return out


def _eval(expr, env):
    """Evaluate a macro expression: numbers, $n, + - x / and parentheses."""
    expr = expr.strip()
    if not expr:
        return 0.0
    s = re.sub(r"\$(\d+)", lambda m: repr(env.get(int(m.group(1)), 0.0)), expr)
    s = s.replace("x", "*").replace("X", "*")
    if not re.fullmatch(r"[0-9eE.+\-*/() ]+", s):
        raise GerberError(f"unsupported macro expression {expr!r}")
    return float(eval(s, {"__builtins__": {}}, {}))  # noqa: S307 -- charset checked above


def _rot(geom, deg):
    return affinity.rotate(geom, deg, origin=(0, 0)) if deg else geom


def _macro_geometry(body_words, args):
    """Shape of one flash of a macro aperture, centred on the flash point."""
    env = {i + 1: v for i, v in enumerate(args)}
    shapes = []                     # (exposure_on, geometry)
    for word in body_words:
        word = word.strip()
        if not word or re.match(r"0(\s|$)", word):
            continue              # empty, or the comment primitive
        m = re.fullmatch(r"\$(\d+)\s*=\s*(.+)", word, re.S)
        if m:
            env[int(m.group(1))] = _eval(m.group(2), env)
            continue
        parts = [p.strip() for p in word.replace("\n", "").split(",")]
        code = int(parts[0])
        vals = [_eval(p, env) for p in parts[1:]]
        if code == 1:          # circle: exposure, diameter, x, y[, rotation]
            exp, d, x, y = vals[:4]
            rot = vals[4] if len(vals) > 4 else 0.0
            g = _rot(Point(x, y).buffer(d / 2, quad_segs=CIRCLE_QUAD_SEGS), rot)
        elif code == 4:        # outline: exposure, n, (n+1) points, rotation
            exp, npts = vals[0], int(vals[1])
            xy = vals[2:2 + 2 * (npts + 1)]
            rot = vals[2 + 2 * (npts + 1)]
            pts = list(zip(xy[0::2], xy[1::2]))
            g = _rot(Polygon(pts).buffer(0), rot)
        elif code == 5:        # regular polygon: exposure, n, x, y, diameter, rotation
            exp, nv, x, y, d, rot = vals[:6]
            nv = int(nv)
            pts = [(x + d / 2 * math.cos(2 * math.pi * k / nv),
                    y + d / 2 * math.sin(2 * math.pi * k / nv)) for k in range(nv)]
            g = _rot(Polygon(pts), rot)
        elif code == 20:       # vector line: exposure, width, x1, y1, x2, y2, rotation
            exp, w, x1, y1, x2, y2, rot = vals[:7]
            g = _rot(LineString([(x1, y1), (x2, y2)]).buffer(w / 2, cap_style="flat"), rot)
        elif code == 21:       # centre line: exposure, width, height, x, y, rotation
            exp, w, h, x, y, rot = vals[:6]
            g = _rot(Polygon([(x - w / 2, y - h / 2), (x + w / 2, y - h / 2),
                              (x + w / 2, y + h / 2), (x - w / 2, y + h / 2)]), rot)
        else:
            raise GerberError(f"unsupported macro primitive {code}")
        shapes.append((exp >= 0.5, g))
    geom = Polygon()
    for on, g in shapes:
        geom = geom.union(g) if on else geom.difference(g)
    return geom


def _standard_aperture(kind, mods):
    if kind == "C":
        d = mods[0]
        g = Point(0, 0).buffer(d / 2, quad_segs=CIRCLE_QUAD_SEGS) if d > 0 else Polygon()
        hole = mods[1] if len(mods) > 1 else 0.0
    elif kind == "R":
        w, h = mods[0], mods[1]
        g = Polygon([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)])
        hole = mods[2] if len(mods) > 2 else 0.0
    elif kind == "O":
        w, h = mods[0], mods[1]
        r = min(w, h) / 2
        if w >= h:
            g = LineString([(-(w / 2 - r), 0), (w / 2 - r, 0)]).buffer(r, quad_segs=CIRCLE_QUAD_SEGS)
        else:
            g = LineString([(0, -(h / 2 - r)), (0, h / 2 - r)]).buffer(r, quad_segs=CIRCLE_QUAD_SEGS)
        hole = mods[2] if len(mods) > 2 else 0.0
    elif kind == "P":
        d, nv = mods[0], int(mods[1])
        rot = mods[2] if len(mods) > 2 else 0.0
        g = _rot(Polygon([(d / 2 * math.cos(2 * math.pi * k / nv),
                           d / 2 * math.sin(2 * math.pi * k / nv)) for k in range(nv)]), rot)
        hole = mods[3] if len(mods) > 3 else 0.0
    else:
        raise GerberError(f"unsupported aperture type {kind}")
    if hole:
        g = g.difference(Point(0, 0).buffer(hole / 2, quad_segs=CIRCLE_QUAD_SEGS))
    return g


def _arc_points(start, end, center, clockwise):
    """Points along a G75 (multi-quadrant) arc, start and end included."""
    (sx, sy), (ex, ey), (cx, cy) = start, end, center
    r = math.hypot(sx - cx, sy - cy)
    a0 = math.atan2(sy - cy, sx - cx)
    a1 = math.atan2(ey - cy, ex - cx)
    if clockwise:
        sweep = a0 - a1
        if sweep <= 1e-12:
            sweep += 2 * math.pi
        sweep = -sweep
    else:
        sweep = a1 - a0
        if sweep <= 1e-12:
            sweep += 2 * math.pi
    n = max(2, int(math.ceil(abs(sweep) / ARC_STEP_RAD)))
    pts = [(cx + r * math.cos(a0 + sweep * k / n), cy + r * math.sin(a0 + sweep * k / n))
           for k in range(n + 1)]
    pts[0], pts[-1] = (sx, sy), (ex, ey)
    return pts


def _stroke(points, shape, aperture_round, width):
    """The area an aperture sweeps along a polyline."""
    if width <= 0:
        return Polygon()
    if aperture_round:
        if len(points) == 1 or all(p == points[0] for p in points):
            return Point(points[0]).buffer(width / 2, quad_segs=CIRCLE_QUAD_SEGS)
        return LineString(points).buffer(width / 2, quad_segs=CIRCLE_QUAD_SEGS)
    # A non-round aperture swept along straight pieces: the hull of the
    # aperture at both ends of every piece.
    pieces = []
    for a, b in zip(points, points[1:] or points):
        pieces.append(unary_union([affinity.translate(shape, *a),
                                   affinity.translate(shape, *b)]).convex_hull)
    return unary_union(pieces)


def gerber_geometry(text):
    """Everything a Gerber file draws, as one shapely geometry (mm, Y up)."""
    scale = None
    unit = 1.0
    macros = {}
    apertures = {}            # D-code -> (shape, round?, width)
    cur_ap = None
    x = y = 0.0
    interp = "G01"
    in_region = False
    region_contours, contour = [], []
    dark = True
    layers = []               # [(dark, [geoms])] in drawing order

    def emit(g):
        if g.is_empty:
            return
        if not layers or layers[-1][0] != dark:
            layers.append((dark, []))
        layers[-1][1].append(g)

    def close_contour():
        nonlocal contour
        if len(contour) >= 3:
            region_contours.append(contour)
        contour = []

    for ext, body in _statements(text):
        if ext:
            words = [w for w in body.split("*")]
            head = words[0].strip()
            if head.startswith("FS"):
                m = re.match(r"FSLAX(\d)(\d)Y(\d)(\d)", head)
                if not m:
                    raise GerberError(f"unsupported format {head}")
                scale = 10 ** int(m.group(2))
            elif head.startswith("MO"):
                unit = 25.4 if head == "MOIN" else 1.0
            elif head.startswith("AM"):
                macros[head[2:]] = words[1:]
            elif head.startswith("AD"):
                m = re.match(r"ADD(\d+)([A-Za-z_.$][\w.$]*),?(.*)", head)
                if not m:
                    raise GerberError(f"bad aperture {head}")
                code, kind, rest = int(m.group(1)), m.group(2), m.group(3)
                mods = [float(v) * unit for v in rest.split("X")] if rest else []
                if kind in ("C", "R", "O", "P"):
                    if kind == "P":
                        mods[1] /= unit          # vertex count and rotation are not lengths
                        if len(mods) > 2:
                            mods[2] /= unit
                    shape = _standard_aperture(kind, mods)
                    apertures[code] = (shape, kind == "C", mods[0] if kind == "C" else 0.0)
                elif kind in macros:
                    args = [float(v) for v in rest.split("X")] if rest else []
                    shape = _macro_geometry(macros[kind], args)
                    if unit != 1.0:
                        shape = affinity.scale(shape, unit, unit, origin=(0, 0))
                    apertures[code] = (shape, False, 0.0)
                else:
                    raise GerberError(f"unknown aperture macro {kind}")
            elif head.startswith("LP"):
                dark = head == "LPD"
            elif head.startswith(("LM", "LR", "LS", "SR", "AB")):
                identity = ("LMN", "LR0", "LS1", "SR", "SRX1Y1I0J0")
                if head not in identity:
                    raise GerberError(f"unsupported command {head}")
            # TF/TA/TO/TD/IP/... attributes and legacy no-ops: nothing drawn
            continue

        stmt = body
        if not stmt or stmt.startswith("G04") or stmt.startswith("M02"):
            continue
        # G codes at the start of a statement
        while True:
            m = re.match(r"G0?(\d+)", stmt)
            if not m:
                break
            g = int(m.group(1))
            stmt = stmt[m.end():]
            if g in (1, 2, 3):
                interp = f"G0{g}"
            elif g == 36:
                in_region, region_contours, contour = True, [], []
            elif g == 37:
                close_contour()
                polys = [Polygon(c).buffer(0) for c in region_contours]
                emit(unary_union(polys) if polys else Polygon())
                in_region = False
            elif g in (74,):
                raise GerberError("single-quadrant arcs (G74) are not supported")
            elif g in (75, 70, 71, 90, 91):
                pass
            else:
                raise GerberError(f"unsupported G{g}")
        if not stmt:
            continue
        m = re.fullmatch(r"D(\d+)", stmt)
        if m and int(m.group(1)) >= 10:
            cur_ap = int(m.group(1))
            continue
        m = re.fullmatch(r"(?:X(-?\d+))?(?:Y(-?\d+))?(?:I(-?\d+))?(?:J(-?\d+))?(?:D0?([123]))?", stmt)
        if not m:
            raise GerberError(f"unparsed statement {stmt!r}")
        if scale is None:
            raise GerberError("coordinates before %FS")
        nx = int(m.group(1)) / scale * unit if m.group(1) is not None else x
        ny = int(m.group(2)) / scale * unit if m.group(2) is not None else y
        i = int(m.group(3)) / scale * unit if m.group(3) is not None else 0.0
        j = int(m.group(4)) / scale * unit if m.group(4) is not None else 0.0
        d = int(m.group(5)) if m.group(5) else 1
        if d == 2:
            if in_region:
                close_contour()
                contour = [(nx, ny)]
        elif d == 1:
            if interp == "G01":
                seg = [(x, y), (nx, ny)]
            else:
                seg = _arc_points((x, y), (nx, ny), (x + i, y + j), clockwise=interp == "G02")
            if in_region:
                if not contour:
                    contour = [(x, y)]
                contour.extend(seg[1:])
            else:
                shape, rnd, w = apertures[cur_ap]
                emit(_stroke(seg, shape, rnd, w))
        elif d == 3:
            shape = apertures[cur_ap][0]
            emit(affinity.translate(shape, nx, ny))
        x, y = nx, ny

    geom = Polygon()
    for is_dark, geoms in layers:
        u = unary_union(geoms)
        geom = geom.union(u) if is_dark else geom.difference(u)
    return geom


def excellon_hits(text):
    """[(diameter_mm, x_mm, y_mm)] for every hole in an Excellon drill file.

    Decimal coordinates only (what both writers emit); slots raise.
    """
    tools, hits = {}, []
    unit = 1.0
    cur = None
    in_header = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line == "M48":
            in_header = True
            continue
        if line in ("%", "M95"):
            in_header = False
            continue
        if line.startswith("METRIC"):
            unit = 1.0
            continue
        if line.startswith("INCH"):
            unit = 25.4
            continue
        m = re.fullmatch(r"T(\d+)C([\d.]+).*", line)
        if m:
            tools[int(m.group(1))] = float(m.group(2)) * unit
            continue
        m = re.fullmatch(r"T(\d+)", line)
        if m and not in_header:
            cur = int(m.group(1))
            continue
        if "G85" in line:
            raise GerberError("drill slots are not supported")
        m = re.fullmatch(r"X(-?[\d.]+)Y(-?[\d.]+)", line)
        if m:
            if "." not in m.group(1) or "." not in m.group(2):
                raise GerberError(f"non-decimal drill coordinate {line}")
            hits.append((tools[cur], float(m.group(1)) * unit, float(m.group(2)) * unit))
    return hits


def pieces(geom):
    """The polygons of a geometry, as a list."""
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def layer_difference(text_a, text_b):
    """How far apart two Gerber files' drawings are.

    {"area_a", "area_b", "xor", "rel", "max_piece"}: the symmetric difference's
    area, that area over the larger drawing's, and its largest connected piece
    (mm^2). Two empty drawings are equal.
    """
    a, b = gerber_geometry(text_a), gerber_geometry(text_b)
    x = a.symmetric_difference(b)
    biggest = max((p.area for p in pieces(x)), default=0.0)
    denom = max(a.area, b.area)
    return {"area_a": a.area, "area_b": b.area, "xor": x.area,
            "rel": x.area / denom if denom > 0 else 0.0, "max_piece": biggest,
            "bounds_a": a.bounds if not a.is_empty else None,
            "bounds_b": b.bounds if not b.is_empty else None}


def drill_difference(text_a, text_b):
    """{"count_a", "count_b", "same_tools", "max_offset"} for two drill files.

    Holes are matched per diameter (to 1 um) by nearest neighbour both ways;
    max_offset is the worst distance (mm), inf when a diameter's counts differ.
    """
    from scipy.spatial import cKDTree
    import numpy as np

    ha, hb = excellon_hits(text_a), excellon_hits(text_b)
    da = sorted({round(h[0], 3) for h in ha})
    db = sorted({round(h[0], 3) for h in hb})
    worst = 0.0
    for dia in set(da) | set(db):
        pa = np.array([h[1:] for h in ha if round(h[0], 3) == dia]).reshape(-1, 2)
        pb = np.array([h[1:] for h in hb if round(h[0], 3) == dia]).reshape(-1, 2)
        if len(pa) != len(pb):
            worst = math.inf
            continue
        if len(pa):
            worst = max(worst, float(cKDTree(pb).query(pa)[0].max()),
                        float(cKDTree(pa).query(pb)[0].max()))
    return {"count_a": len(ha), "count_b": len(hb), "same_tools": da == db, "max_offset": worst}
