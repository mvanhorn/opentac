"""Check the board's Edge.Cuts as exported: one closed outline, every line on it once.

The exported Edge.Cuts are two sets of lines: the board's own outline, cut open
at the connector (connector_geometry.stitched_outline_segments), and the
connector footprint's Edge.Cuts, which draw the connector's tab from the two
open ends of that cut. A fab mills every one of those lines. If the two sets do
not join into one simple outline, the board it gets is not the board that was
drawn: a footprint line lying inside the board is milled as a slit, a line drawn
twice (or two lines a hair apart) as a slit or a sliver. KiCad's DRC does not
see these: it folds a slit into the board outline as a zero-width notch and
reports a valid outline, and every line still lies on that outline, so a
distance-to-outline test passes as well.

edge_cuts_problems() takes the lines exactly as the exporters write them and
reports:

- "open_end" / "branch": a line end that is not joined to exactly one other
  line (a slit's dead end, or three lines meeting);
- "overlap": two lines running along each other (closer than GAP_MM, parallel,
  overlapping), drawn twice;
- "loops": the lines close into more than one outline;
- "self_intersection": the outline crosses itself;
- "slit": two stretches of the outline closer than GAP_MM that are more than
  LOOP_MM apart along it -- a slit or a sliver narrower than any mill;
- "spike": the outline turns back on itself at a vertex (a zero-width spike or
  notch too short for the slit test).
"""
from __future__ import annotations

import math
import re

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon
from shapely.validation import explain_validity

from opentac.cad.export.connector_geometry import connector_edge_geometries, stitched_outline_segments

GAP_MM = 0.1        # two stretches of the outline closer than this ...
LOOP_MM = 1.0       # ... and further apart than this along it: a slit or a sliver
SNAP_MM = 1e-3      # line ends closer than this are one point
SPIKE_TURN_DEG = 170.0
PIECE_MM = 0.2      # the slit test measures the outline in pieces no longer than this
ARC_STEPS = 8       # chords per footprint arc (KiCad approximates arcs the same way)
MAX_REPORTS = 12


def _arc_points(start, mid, end, steps=ARC_STEPS):
    """Points along the circular arc start -> mid -> end (a straight line if
    the three are collinear)."""
    a, m, b = (np.asarray(p, dtype=float) for p in (start, mid, end))
    d = 2.0 * (a[0] * (m[1] - b[1]) + m[0] * (b[1] - a[1]) + b[0] * (a[1] - m[1]))
    if abs(d) < 1e-12:
        return [a, b]
    ux = ((a @ a) * (m[1] - b[1]) + (m @ m) * (b[1] - a[1]) + (b @ b) * (a[1] - m[1])) / d
    uy = ((a @ a) * (b[0] - m[0]) + (m @ m) * (a[0] - b[0]) + (b @ b) * (m[0] - a[0])) / d
    c = np.array([ux, uy])
    r = float(np.linalg.norm(a - c))
    t0, tm, t1 = (math.atan2(p[1] - c[1], p[0] - c[0]) for p in (a, m, b))

    def ccw(x, y):
        return (y - x) % (2 * math.pi)

    sweep = ccw(t0, t1)
    if ccw(t0, tm) > sweep:          # the arc runs clockwise through mid
        sweep -= 2 * math.pi
    pts = [a]
    for i in range(1, steps):
        t = t0 + sweep * i / steps
        pts.append(c + r * np.array([math.cos(t), math.sin(t)]))
    pts.append(b)
    return pts


def exported_edge_cuts(layout, outline_xy, params, connector_dir):
    """Every Edge.Cuts line the KiCad and Gerber exporters write, as
    (start, end, source) with source "board" or "connector"; footprint arcs as
    ARC_STEPS chords. The board's lines are stitched_outline_segments (the
    plain final outline when there is no connector to stitch to), exactly as
    export/kicad_pcb.py writes them."""
    out = []
    segments = stitched_outline_segments(layout, outline_xy, params, connector_dir)
    if segments is None:
        final_xy = getattr(layout, "final_outline_xy", None)
        if final_xy is None:
            return out
        final_xy = np.asarray(final_xy, dtype=float)
        segments = [(final_xy[i], final_xy[(i + 1) % len(final_xy)]) for i in range(len(final_xy))]
    for a, b in segments:
        out.append((np.asarray(a, dtype=float), np.asarray(b, dtype=float), "board"))
    edges = connector_edge_geometries(layout, outline_xy, params, connector_dir)
    for a, b in edges["lines"]:
        out.append((np.asarray(a, dtype=float), np.asarray(b, dtype=float), "connector"))
    for start, mid, end in edges["arcs"]:
        pts = _arc_points(start, mid, end)
        for p, q in zip(pts[:-1], pts[1:]):
            out.append((p, q, "connector"))
    return out


def _problem(kind, where, text, distance=0.0, required=0.0, sources=("board", "board")):
    return {"type": kind, "where": [round(float(where[0]), 4), round(float(where[1]), 4)],
            "text": text, "distance": float(distance), "min_required": float(required),
            "sources": tuple(sources)}


def _nodes(segs):
    """Node index of each line end, ends closer than SNAP_MM merged."""
    pts = np.array([p for a, b, _ in segs for p in (a, b)], dtype=float)
    parent = list(range(len(pts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(pts).query_pairs(SNAP_MM):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = {}
    ids = np.array([roots.setdefault(find(i), len(roots)) for i in range(len(pts))])
    return ids.reshape(-1, 2), pts


def _overlaps(segs):
    """Pairs of lines running along each other: within GAP_MM, parallel, and
    sharing more than SNAP_MM of their length."""
    found = []
    lines = [LineString([a, b]) for a, b, _ in segs]
    tree = shapely.STRtree(lines)
    left, right = tree.query(lines, predicate="dwithin", distance=GAP_MM)
    for i, j in zip(left.tolist(), right.tolist()):
        if j <= i:
            continue
        a1, b1, s1 = segs[i]
        a2, b2, s2 = segs[j]
        d1, d2 = b1 - a1, b2 - a2
        l1, l2 = float(np.linalg.norm(d1)), float(np.linalg.norm(d2))
        if l1 < SNAP_MM or l2 < SNAP_MM:
            continue
        u1 = d1 / l1
        if abs(u1[0] * d2[1] - u1[1] * d2[0]) / l2 > math.sin(math.radians(2.0)):
            continue                                    # not parallel
        # how far line 2 lies off line 1, and how much of it runs alongside
        off = max(abs(u1[0] * (p - a1)[1] - u1[1] * (p - a1)[0]) for p in (a2, b2))
        if off > GAP_MM:
            continue
        t = sorted([float((a2 - a1) @ u1), float((b2 - a1) @ u1)])
        shared = min(l1, t[1]) - max(0.0, t[0])
        if shared > SNAP_MM:
            mid = a1 + u1 * (max(0.0, t[0]) + shared / 2)
            if any(np.linalg.norm(mid - np.asarray(f["where"])) < 2.0 for f in found):
                continue
            found.append(_problem(
                "overlap", mid,
                f"two Edge.Cuts lines run along each other for {shared:.2f} mm, {off:.3f} mm apart "
                f"({s1} and {s2}): the outline is drawn twice there",
                off, GAP_MM, (s1, s2)))
    return found


def _loop(segs, ends):
    """(node ids round the loop through the first node, whether that loop
    takes in every line). Every node has two lines when this is called."""
    adj = {}
    for k, (i, j) in enumerate(ends):
        if i == j:
            continue
        adj.setdefault(int(i), []).append((k, int(j)))
        adj.setdefault(int(j), []).append((k, int(i)))
    start = next(iter(adj))
    ring_nodes, used = [start], set()
    node, prev_k = start, None
    while True:
        nxt = [(k, n) for k, n in adj[node] if k != prev_k and k not in used]
        if not nxt:
            break
        k, n = nxt[0]
        used.add(k)
        if n == start:
            break
        ring_nodes.append(n)
        node, prev_k = n, k
    n_lines = sum(1 for i, j in ends if i != j)
    return ring_nodes, len(used) == n_lines


def _slits(ring):
    """Stretches of the ring closer than GAP_MM but more than LOOP_MM apart
    along it (slits, slivers, long spikes), one report per place."""
    P = np.asarray(ring, dtype=float)
    Q = np.roll(P, -1, axis=0)
    L = np.linalg.norm(Q - P, axis=1)
    total = float(L.sum())
    a_pts, b_pts, starts = [], [], []
    pos = 0.0
    for p, q, l in zip(P, Q, L):
        n = max(1, int(math.ceil(l / PIECE_MM)))
        for k in range(n):
            a_pts.append(p + (q - p) * (k / n))
            b_pts.append(p + (q - p) * ((k + 1) / n))
            starts.append(pos + l * k / n)
        pos += l
    a_pts, b_pts, starts = np.array(a_pts), np.array(b_pts), np.array(starts)
    pieces = shapely.linestrings(np.stack([a_pts, b_pts], axis=1))
    tree = shapely.STRtree(pieces)
    left, right = tree.query(pieces, predicate="dwithin", distance=GAP_MM)
    keep = right > left
    left, right = left[keep], right[keep]
    if len(left) == 0:
        return []
    gaps = shapely.distance(pieces[left], pieces[right])
    near = shapely.get_coordinates(shapely.shortest_line(pieces[left], pieces[right])).reshape(-1, 2, 2)
    pos_l = starts[left] + np.linalg.norm(near[:, 0] - a_pts[left], axis=1)
    pos_r = starts[right] + np.linalg.norm(near[:, 1] - a_pts[right], axis=1)
    along = np.abs(pos_l - pos_r)
    along = np.minimum(along, total - along)
    bad = np.flatnonzero(along > LOOP_MM)
    found, places = [], []
    for k in bad[np.argsort(gaps[bad])]:
        where = 0.5 * (near[k, 0] + near[k, 1])
        if any(np.linalg.norm(where - w) < 2.0 for w in places):
            continue
        places.append(where)
        found.append(_problem(
            "slit", where,
            f"the outline comes within {gaps[k]:.3f} mm of itself {along[k]:.1f} mm further along it: "
            "a slit or a sliver no mill can cut",
            gaps[k], GAP_MM))
        if len(found) >= MAX_REPORTS:
            break
    return found


def _spikes(ring):
    P = np.asarray(ring, dtype=float)
    found = []
    for i in range(len(P)):
        a, b, c = P[i - 1], P[i], P[(i + 1) % len(P)]
        u, v = b - a, c - b
        lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        if lu < SNAP_MM or lv < SNAP_MM:
            continue
        turn = math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v) / (lu * lv)))))
        if turn > SPIKE_TURN_DEG:
            found.append(_problem(
                "spike", b, f"the outline turns back on itself ({turn:.0f} degrees): a zero-width spike or notch",
                0.0, GAP_MM))
            if len(found) >= MAX_REPORTS:
                break
    return found


def edge_cuts_problems(layout, outline_xy, params, connector_dir) -> list[dict]:
    """Everything wrong with the exported Edge.Cuts (see the module docstring);
    empty when they form one simple closed outline with every line on it once.
    Each problem: {"type", "where" (board mm), "text", "distance",
    "min_required", "sources"}."""
    return check_edge_cuts(exported_edge_cuts(layout, outline_xy, params, connector_dir))


def kicad_pcb_edge_cuts(pcb_text: str):
    """The Edge.Cuts lines of a .kicad_pcb as written by export/kicad_pcb.py,
    as (start, end, source) in KiCad's coordinates: board gr_lines ("board"),
    and the connector footprint's fp_lines and fp_arcs ("connector"), which
    that exporter writes in board coordinates (the footprint sits at 0 0 0)."""
    num = r"(-?[\d.]+)"
    out = []
    for m in re.finditer(r"\(gr_line \(start " + num + " " + num + r"\) \(end " + num + " " + num
                         + r'\)[^\n]*?\(layer "Edge\.Cuts"\)', pcb_text):
        v = [float(g) for g in m.groups()]
        out.append((np.array(v[:2]), np.array(v[2:]), "board"))
    for m in re.finditer(r"\(fp_line \(start " + num + " " + num + r"\) \(end " + num + " " + num
                         + r'\)[^\n]*?\(layer "Edge\.Cuts"\)', pcb_text):
        v = [float(g) for g in m.groups()]
        out.append((np.array(v[:2]), np.array(v[2:]), "connector"))
    for m in re.finditer(r"\(fp_arc \(start " + num + " " + num + r"\) \(mid " + num + " " + num
                         + r"\) \(end " + num + " " + num + r'\)[^\n]*?\(layer "Edge\.Cuts"\)', pcb_text):
        v = [float(g) for g in m.groups()]
        pts = _arc_points(v[0:2], v[2:4], v[4:6])
        out += [(p, q, "connector") for p, q in zip(pts[:-1], pts[1:])]
    return out


def check_edge_cuts(lines) -> list[dict]:
    """edge_cuts_problems for any set of Edge.Cuts lines (start, end, source)."""
    segs = [(np.asarray(a, dtype=float), np.asarray(b, dtype=float), s) for a, b, s in lines]
    segs = [(a, b, s) for a, b, s in segs if float(np.linalg.norm(b - a)) > 1e-9]
    if len(segs) < 3:
        return []
    problems = _overlaps(segs)

    ends, pts = _nodes(segs)
    degree = np.bincount(ends[ends[:, 0] != ends[:, 1]].ravel(), minlength=int(ends.max()) + 1)
    first = {}
    for k, (i, j) in enumerate(ends):
        first.setdefault(int(i), (pts[2 * k], segs[k][2]))
        first.setdefault(int(j), (pts[2 * k + 1], segs[k][2]))
    for node in np.flatnonzero((degree != 2) & (degree > 0)):
        where, src = first[int(node)]
        if degree[node] == 1:
            problems.append(_problem("open_end", where, f"an Edge.Cuts line ({src}) ends here without joining "
                                     "another: the outline is open, or a line lies inside the board",
                                     sources=(src, src)))
        else:
            problems.append(_problem("branch", where, f"{degree[node]} Edge.Cuts lines meet here: one of them "
                                     "lies inside or outside the board instead of on its edge",
                                     sources=(src, src)))
        if len(problems) >= MAX_REPORTS:
            return problems
    if any(p["type"] in ("open_end", "branch") for p in problems):
        return problems

    ring_nodes, whole = _loop(segs, ends)
    node_pt = {}
    for k, (i, j) in enumerate(ends):
        node_pt.setdefault(int(i), pts[2 * k])
        node_pt.setdefault(int(j), pts[2 * k + 1])
    if not whole:
        problems.append(_problem("loops", node_pt[ring_nodes[0]],
                                 "the Edge.Cuts lines close into more than one outline"))
        return problems
    ring = [node_pt[n] for n in ring_nodes]
    poly = Polygon(ring)
    if not poly.is_valid:
        why = explain_validity(poly)
        where = ring[0]
        if "[" in why:
            try:
                where = [float(v) for v in why[why.index("[") + 1:why.index("]")].split()]
            except ValueError:
                pass
        problems.append(_problem("self_intersection", where, f"the outline crosses or touches itself ({why})"))
    problems += _slits(ring)
    problems += _spikes(ring)
    return problems


def edge_cuts_violations(layout, outline_xy, params, connector_dir):
    """edge_cuts_problems as pipeline DRC violations (layer Edge.Cuts), so the
    tool reports them with the copper checks."""
    from opentac.cad.pipeline.types import DrcViolation
    names = {"board": "board edge", "connector": "connector tab"}
    try:
        problems = edge_cuts_problems(layout, outline_xy, params, connector_dir)
    except Exception as exc:                # a check that cannot run is not a pass
        problems = [_problem("check_failed", (0.0, 0.0), f"the Edge.Cuts check could not run: {exc}")]
    out = []
    for p in problems:
        where = np.array(p["where"], dtype=float)
        out.append(DrcViolation(
            layer="Edge.Cuts", net1=names.get(p["sources"][0], p["sources"][0]),
            net2=names.get(p["sources"][1], p["sources"][1]), type=p["type"],
            seg1=np.array([where, where]), seg2=np.array([where, where]),
            distance=p["distance"], min_required=p["min_required"]))
    return out
