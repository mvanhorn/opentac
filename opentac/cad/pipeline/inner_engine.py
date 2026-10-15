"""Inner-layer wiring engine for 4-layer boards (PROTOTYPE).

One mechanism routes every wire on an inner copper layer: a sweep over the
via lattice, one via row at a time, top down.

Geometry. Every taxel has one through-hole via, so vias sit on a lattice of
via columns X and via rows Y. A wire may run vertically only in a LANE (a
position between two via columns, at least `a` from both) and horizontally
only along a via row (a stub from a via to the lane beside it) or on a TRACK
(a position between two via rows, at least `a` from both). Copper must stay
inside the allowed region F (under the taxels, inside the board).

Sweep. Going down, the wires that cross a via row form a left-to-right
ORDER. Between two rows a wire may step sideways to a new lane; steps keep
the order, so the layer stays planar by construction. At each row:

  * wires whose region ends under them leave (escapes: towards the fan);
  * new wires start: from any via of their row, in a lane of the gap on
    either side of it (the stub stays inside that gap);
  * every wire gets a lane for this row. That is solved exactly: wires in
    order, each with its candidate lanes, minimum total distance to each
    wire's target, subject to spacing (one trace pitch, or touching if
    neighbours share a net) and to how far a wire can step;
  * the steps are put on tracks in the gap above. A set of wires stepping
    the same way needs as many tracks as their spans overlap, so a step is
    limited to (tracks - 1) trace pitches per row. That bound, not a tuned
    rate, is what makes a bundle drift across over several rows.

Only geometry and capacity enter. Targets are what the wires are for: an
escape heads for its pad (wires in order onto consecutive pads), a bridge
end stays inside its petal until it meets its partner under the notch.

Everything the engine returns is checked again by `verify`.
"""

import math

import numpy as np
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.ops import unary_union

TOL = 1e-6


def cap(width, a, lane):
    """tracks (or lanes) that fit in a gap of `width` between two via lines"""
    return int(math.floor((width - 2 * a) / lane + 1e-9)) + 1 if width >= 2 * a - 1e-9 else 0


class Lattice:
    def __init__(self, X, Yr, a, lane, pitch=None):
        """pitch: via column spacing; the half gap outside each outermost via
        column is under the taxels too, and a strip one taxel wide has no
        other lanes at all."""
        X = np.asarray(sorted(X), dtype=float)
        if pitch:
            X = np.concatenate([[X[0] - pitch], X, [X[-1] + pitch]])
        self.X = X
        self.Yd = sorted((float(y) for y in Yr), reverse=True)
        self.a, self.lane = a, lane
        # lanes anchored on either via of each gap (a wire hugging the right
        # via of a gap wants the lattice that starts there)
        xs = [np.arange(X[k] + a, X[k + 1] - a + 1e-9, lane) for k in range(len(X) - 1)]
        xs += [np.arange(X[k + 1] - a, X[k] + a - 1e-9, -lane) for k in range(len(X) - 1)]
        self.lanes = np.unique(np.round(np.concatenate(xs), 6)) if xs else np.zeros(0)

    def gap_lanes(self, tap):
        """lanes in the two gaps beside the via column at x = tap"""
        k = int(np.argmin(np.abs(self.X - tap)))
        lo = self.X[k - 1] if k > 0 else -math.inf
        hi = self.X[k + 1] if k + 1 < len(self.X) else math.inf
        return [float(x) for x in self.lanes if lo < x < hi]

    def tracks_up(self, y_row, y_above):
        """track heights in the gap above via row y_row, lowest first"""
        return [y_row + self.a + t * self.lane for t in range(cap(y_above - y_row, self.a, self.lane))]

    def tracks_down(self, y_row, y_below):
        """track heights in the gap below via row y_row, highest first"""
        return [y_row - self.a - t * self.lane for t in range(cap(y_row - y_below, self.a, self.lane))]


def intervals(F, y):
    cut = F.intersection(LineString([(-1e5, y), (1e5, y)]))
    out = []
    for c in getattr(cut, "geoms", [cut]):
        if c.geom_type == "LineString" and not c.is_empty:
            xs = [p[0] for p in c.coords]
            out.append((min(xs), max(xs)))
    return sorted(out)


def which(iv, x, m=1e-3):
    for k, (u, v) in enumerate(iv):
        if u + m <= x <= v - m:
            return k
    return None


# --------------------------------------------------------------------------
# one row: lanes for ordered wires, exactly
# --------------------------------------------------------------------------

def place_row(elems, lane):
    """elems: ordered list of (net, [(x, lo, hi, cost), ...]). Picks one
    candidate per element: lo of each element at least hi of the previous +
    `lane` (two neighbours of one net need only keep their lanes in order).
    `lane` may be a function of the previous wire's x.
    Minimum total cost. Returns the chosen candidates, or None."""
    if not elems:
        return []
    gap = lane if callable(lane) else (lambda x, _l=lane: _l)   # spacing after a wire at x
    prev = None            # states of the previous element: (hi, x, total)
    backs = []             # per element: per candidate, index of the best predecessor
    totals = []
    prev_net = None
    for net, cands in elems:
        if not cands:
            return None
        if prev is None:
            tot = [c[3] for c in cands]
            back = [None] * len(cands)
        else:
            same = net == prev_net
            keyv = [(p[1] if same else p[0] + gap(p[1])) for p in prev]
            idx = sorted(range(len(prev)), key=lambda i: keyv[i])
            ks, runs, run = [], [], (math.inf, None)
            for i in idx:
                if prev[i][2] < run[0]:
                    run = (prev[i][2], i)
                ks.append(keyv[i])
                runs.append(run)
            tot, back = [], []
            for c in cands:
                # one net: strictly left of it (two wires in one lane could never
                # be separated again); another net: a trace pitch clear of it
                need = c[0] - 1e-6 if same else c[1]
                j = int(np.searchsorted(np.asarray(ks), need + 1e-9, side="right"))
                if j == 0 or runs[j - 1][1] is None:
                    tot.append(math.inf)
                    back.append(None)
                else:
                    tot.append(runs[j - 1][0] + c[3])
                    back.append(runs[j - 1][1])
        cur = [(c[2], c[0], t) for c, t in zip(cands, tot)]
        if not any(math.isfinite(t) for t in tot):
            return None
        backs.append(back)
        totals.append(tot)
        prev, prev_net = cur, net
    k = min(range(len(prev)), key=lambda i: prev[i][2])
    picks = []
    for e in range(len(elems) - 1, -1, -1):
        picks.append(elems[e][1][k])
        k = backs[e][k]
    return picks[::-1]


def jog_tracks(jogs, heights, F):
    """Tracks for sideways steps in one gap. jogs: [(key, x0, x1)];
    heights: usable track heights, lowest first. Steps the same way whose
    spans come within a trace pitch need distinct tracks, the one in front
    (furthest along the move) highest: each then steps under wires that have
    already moved away. Returns {key: y} or None if they do not fit."""
    out = {}
    for sgn in (-1, +1):                        # leftward, rightward
        grp = [(k, x0, x1) for k, x0, x1 in jogs if (x1 - x0) * sgn > 0]
        # the wire in front (furthest along the move) must step on a HIGHER
        # track than the ones behind it: it is out of their way before they
        # step under it. So the one at the back goes first, lowest.
        grp.sort(key=lambda j: sgn * j[1])
        placed = []
        for k, x0, x1 in grp:
            lo, hi = min(x0, x1), max(x0, x1)
            t = 0
            for (plo, phi, pt) in placed:
                if plo < hi + heights_gap(heights) and lo < phi + heights_gap(heights):
                    t = max(t, pt + 1)
            # the lowest track at or above that whose run stays in the region
            # (a step out of a wide row into a neck runs high, under the
            # wide row)
            while t < len(heights) and not F.contains(LineString([(x0, heights[t]), (x1, heights[t])])):
                t += 1
            if t >= len(heights):
                return None
            placed.append((lo, hi, t))
            out[k] = heights[t]
    return out


class Gap:
    """The gap above a via row, seen by the wires stepping into that row.
    `heights` lowest first; `ivs[i]` the stretches of the region at
    heights[i]. A step from x0 to x1 can run on track i if one stretch there
    holds both ends."""

    def __init__(self, F, heights):
        self.heights = list(heights)
        self.ivs = [intervals(F, h) for h in self.heights]

    def can_step(self, x0, x1):
        lo, hi = min(x0, x1), max(x0, x1)
        return any(any(u - 1e-6 <= lo and hi <= v + 1e-6 for u, v in iv) for iv in self.ivs)

    def can_step_all(self, x0, x1):
        """the step fits on EVERY track: then any assignment of tracks works,
        however many wires step together"""
        lo, hi = min(x0, x1), max(x0, x1)
        return bool(self.ivs) and all(any(u - 1e-6 <= lo and hi <= v + 1e-6 for u, v in iv) for iv in self.ivs)

    def reach(self):
        """the longest step that can never need more tracks than there are"""
        return max(0, len(self.heights) - 1)


def heights_gap(heights):
    return (heights[1] - heights[0]) if len(heights) > 1 else 0.0


# --------------------------------------------------------------------------
# verification, shared by every caller
# --------------------------------------------------------------------------

def verify(paths, net_of, via_xy, via_net, F, a, lane, others=(), tails=None, reasons=None,
           region_slack=1e-3):
    """paths: {key: polyline}. A key is bad if its copper comes within a
    trace pitch of another net's wire (incl. `others`: [(net, geom)]),
    within `a` of another net's via, or leaves F. tails: {key: n} — the last
    n points run outside F by design (an escape's drop to the fan).
    Returns the set of bad keys."""
    from shapely import multipoints
    from shapely.geometry import Point
    def geometry(p):
        return Point(p[0]) if len(p) == 1 or np.all(np.asarray(p) == p[0]) else LineString(p)
    lines = {k: geometry(p) for k, p in paths.items()}
    bad = set()
    Fb = F.buffer(region_slack) if region_slack else F
    via_xy = np.asarray(via_xy, dtype=float)
    via_net = np.asarray(via_net, dtype=object)
    foreign = {n: multipoints(via_xy[via_net != n]) for n in set(net_of.values())}
    keys = list(lines)
    for i, k1 in enumerate(keys):
        n1 = net_of[k1]
        pts = paths[k1]
        inside = pts[: len(pts) - (tails or {}).get(k1, 0) + 1] if tails and tails.get(k1) else pts
        note = (lambda k, m: reasons.setdefault(k, []).append(m)) if reasons is not None else (lambda k, m: None)
        if not Fb.covers(geometry(inside)):
            bad.add(k1)
            note(k1, "leaves the region")
        for k2 in keys[i + 1:]:
            if net_of[k2] != n1 and lines[k1].distance(lines[k2]) < lane - 1e-6:
                bad |= {k1, k2}
                note(k1, f"within {lines[k1].distance(lines[k2]):.3f} of {k2}")
        for n2, g in others:
            if n2 != n1 and lines[k1].distance(g) < lane - 1e-6:
                bad.add(k1)
                note(k1, f"within {lines[k1].distance(g):.3f} of {n2} (placed)")
        if not foreign[n1].is_empty and lines[k1].distance(foreign[n1]) < a - 1e-6:
            bad.add(k1)
            note(k1, f"within {lines[k1].distance(foreign[n1]):.3f} of a via")
    return bad


def _chamfered(pts, c):
    """pts with every corner cut by a 45-degree-ish chord: c back along each
    side, at most half of either side (two corners on one short side meet
    in the middle: a square step becomes one diagonal). Ends stay put."""
    p = [tuple(map(float, q)) for q in pts]
    clean = [p[0]]
    for q in p[1:]:
        if math.dist(q, clean[-1]) > 1e-9:
            clean.append(q)
    p = [clean[0]]
    for i in range(1, len(clean) - 1):          # drop collinear points
        a, b, d = np.subtract(clean[i], p[-1]), np.subtract(clean[i + 1], clean[i]), None
        if abs(a[0] * b[1] - a[1] * b[0]) > 1e-9 * max(1.0, np.hypot(*a) * np.hypot(*b)):
            p.append(clean[i])
    p.append(clean[-1])
    if c <= 0 or len(p) < 3:
        return p
    out = [p[0]]
    for i in range(1, len(p) - 1):
        a, b, d = np.asarray(p[i - 1]), np.asarray(p[i]), np.asarray(p[i + 1])
        la, ld = np.linalg.norm(b - a), np.linalg.norm(d - b)
        k = min(c, la / 2, ld / 2)
        if k <= 1e-6:
            out.append(tuple(b))
            continue
        u, v = b + (a - b) / la * k, b + (d - b) / ld * k
        for q in (tuple(u), tuple(v)):
            if math.dist(q, out[-1]) > 1e-9:
                out.append(q)
    if math.dist(p[-1], out[-1]) > 1e-9:
        out.append(p[-1])
    return out


def chamfer_paths(paths, via_xy, via_net, F, a, lane, others=(),
                  sizes=(1e9, 1.0, 0.6, 0.4, 0.25, 0.15, 0.1)):
    """Smooth the sweep's staircases: each wire, in turn, takes the largest
    corner cut from `sizes` that still keeps a lane from every other wire
    (as they stand by then), `a` from other nets' vias, and stays inside F.
    A wire none of the cuts fits keeps its square corners. Taps and exits
    do not move, so nothing about which rows route changes."""
    Fb = F.buffer(1e-3)
    cur = {k: [tuple(q) for q in v] for k, v in paths.items()}
    lines = {k: LineString(v) for k, v in cur.items()}
    via_xy = np.asarray(via_xy, dtype=float)
    # Wires in a bundle sit a lane apart round every bend: an outer wire's
    # cut only fits once the wire inside it is cut. So go round again until
    # nothing changes; each pass lets the next wire out follow.
    vias_of = {}
    for k in cur:
        mask = np.array([n != k for n in via_net])
        vias_of[k] = MultiPoint(via_xy[mask]) if mask.any() else None
    best = {k: None for k in cur}            # index into sizes of the cut taken
    for _ in range(12):
        changed = False
        for k in list(cur):
            for i, c in enumerate(sizes):
                if best[k] is not None and i >= best[k]:
                    break
                cand = _chamfered(paths[k], c)
                if cand == cur[k]:
                    continue
                L = LineString(cand)
                if (Fb.contains(L)
                        and all(L.distance(lines[o]) >= lane - 1e-6 for o in lines if o != k)
                        and all(L.distance(g) >= lane - 1e-6 for n2, g in others if n2 != k)
                        and (vias_of[k] is None or L.distance(vias_of[k]) >= a - 1e-6)):
                    cur[k], lines[k], best[k] = cand, L, i
                    changed = True
                    break
        if not changed:
            break
    return cur


# --------------------------------------------------------------------------
# escapes: each row to the connector
# --------------------------------------------------------------------------

def sweep_escapes(rows, lat, F, exit_y, exit_ok, pad_xs):
    """rows: [{"net", "y", "vias"}] (vias: x of every via of the row's net).
    exit_y(x): bottom of F under x. exit_ok(x): a drop at x can fan out.
    pad_xs: x of the pads these rows may land on.

    Wires head for the pads, in order onto consecutive pads. At the bottom of
    F a wire leaves where it is (straight down to the fan) and stays put as
    an obstacle for the wires that go on.

    Lookahead is a funnel computed bottom up before the sweep: for each row,
    the lanes from which a wire can still reach a usable exit, given where F
    is and how far a wire can step per row. Every lane a wire takes is in the
    funnel, so wires gather before a neck because the geometry says they
    must.
    Returns ({net: polyline tap -> ... -> (x, exit)}, {net: reason})."""
    a, lane = lat.a, lat.lane
    pads = sorted(pad_xs)
    Yd = lat.Yd
    n_rows = len(Yd)
    iv = [intervals(F, y) for y in Yd]
    S = [[float(x) for x in lat.lanes if which(iv[r], float(x)) is not None] for r in range(n_rows)]
    gaps = [Gap(F, []) if r == 0 else Gap(F, lat.tracks_up(Yd[r], Yd[r - 1])) for r in range(n_rows)]
    safe = [[] if r == 0 else _open_tracks(F, iv[r], gaps[r].heights) for r in range(n_rows)]
    reach = [gaps[r].reach() * lane for r in range(n_rows)]
    # The funnel only counts steps that fit on every one of the tracks that
    # run under the whole next row: any assignment of those tracks then works
    # however many wires step together, so the funnel never asks for more
    # than a gap can carry. A bundle heading into a one-taxel waist therefore
    # gathers over the rows of the wide lobe above it, not all in the last
    # gap.
    safe_gap = [Gap(F, safe[r]) for r in range(n_rows)]
    safe_reach = [safe_gap[r].reach() * lane for r in range(n_rows)]

    def leaves_after(x, r):
        """F ends under x before the next row: a wire here exits at this row"""
        ex = exit_y(x)
        return ex is not None and (r + 1 >= n_rows or ex > Yd[r + 1] - 1e-6)

    funnel = [set() for _ in range(n_rows)]
    for r in range(n_rows - 1, -1, -1):
        nxt = sorted(funnel[r + 1]) if r + 1 < n_rows else []
        for x in S[r]:
            if leaves_after(x, r):
                if exit_ok(x):
                    funnel[r].add(x)
                continue
            j = int(np.searchsorted(nxt, x - safe_reach[r + 1] - 1e-9))
            while j < len(nxt) and nxt[j] <= x + safe_reach[r + 1] + 1e-9:
                if abs(nxt[j] - x) < 1e-9 or safe_gap[r + 1].can_step_all(x, nxt[j]):
                    funnel[r].add(x)
                    break
                j += 1

    def targets(ws):
        """wires in order onto consecutive pads (the window nearest them)"""
        n = len(ws)
        if n == 0 or n > len(pads):
            return [None] * n
        cx = [w["x"] if w["x"] is not None else w["cx"] for w in ws]
        best = min(range(len(pads) - n + 1), key=lambda s0: sum(abs(cx[i] - pads[s0 + i]) for i in range(n)))
        return [pads[best + i] for i in range(n)]

    by_y = {}
    for r in rows:
        by_y.setdefault(round(r["y"], 4), []).append(r)
    order, why = [], {}
    for ri, y in enumerate(Yd):
        Sr = S[ri]
        # wires whose region ended above this row leave (they were in the
        # funnel, so their exit is usable)
        for w in order:
            if not w.get("out") and which(iv[ri], w["x"]) is None and leaves_after(w["x"], ri - 1):
                w["out"], w["end"] = True, exit_y(w["x"])

        def passing_cands(w, rch):
            if w.get("out"):
                return [(w["x"], w["x"], w["x"], 0.0)]
            return [(x, x, x, 0.0) for x in Sr if x in funnel[ri] and abs(x - w["x"]) <= rch + 1e-9
                    and (abs(x - w["x"]) < 1e-9 or gaps[ri].can_step(w["x"], x))]

        # a wire that has fallen out of the funnel (it could not keep up) has
        # nowhere to go: it alone is dropped, the rest carry on
        stuck = [w for w in order if not passing_cands(w, reach[ri])]
        for w in stuck:
            why[w["net"]] = "it could not move over in time for a narrowing below"
        order = [w for w in order if w not in stuck]

        news = by_y.get(round(y, 4), [])
        for k_new in range(max(1, len(news))):
            # several row pieces can start at one height (a U's two arms):
            # each joins in turn; after the first, the wires already here stay
            # put (this gap's tracks are spoken for)
            new = news[k_new:k_new + 1]
            first = k_new == 0
            opts = []
            if new:
                r_new = new[0]
                for t in sorted({float(v) for v in r_new["vias"]}):
                    for x in lat.gap_lanes(t):
                        if x in funnel[ri] and x in Sr:
                            opts.append((x, min(t, x), max(t, x), t))

            def plan(rch, seek=True):
                """best placement of this row (new wire anywhere in the order).
                seek: head for the pads; otherwise move as little as possible
                (the funnel still gets every wire to its narrowings in time)."""
                found = []
                spots = range(len(order) + 1) if (new and opts) else [None]
                for k in spots:
                    ws = list(order) if k is None else order[:k] + [{"net": new[0]["net"], "x": None,
                                                                   "cx": float(np.median([o[0] for o in opts]))}] + order[k:]
                    tg = targets(ws)
                    elems = []
                    for w, tgt in zip(ws, tg):
                        if w["x"] is None:
                            cs = [(x, lo_, hi_, (abs(x - tgt) if tgt is not None else 0.0) + 1e-3 * abs(x - t))
                                  for x, lo_, hi_, t in opts]
                        else:
                            cs = []
                            for x, lo_, hi_, _ in passing_cands(w, rch):
                                to_t = abs(x - tgt) if (tgt is not None and not w.get("out")) else 0.0
                                moved = abs(x - w["x"])
                                cs.append((x, lo_, hi_, (to_t + 1e-3 * moved) if seek else (moved + 1e-3 * to_t)))
                        elems.append((w["net"], cs))
                    pick = place_row(elems, lane)
                    if pick is not None:
                        found.append((sum(c[3] for c in pick), k if k is not None else -1, ws, pick))
                found.sort(key=lambda f: (f[0], f[1]))
                return [(c, ws, pick) for c, _, ws, pick in found]

            result = None
            # every placement, cheapest first; the first whose steps fit the gap
            for rch, seek in (((reach[ri], True), (reach[ri], False), (safe_reach[ri], False)) if first
                              else ((0.0, True),)):
                for _, ws, pick in plan(rch, seek):
                    jogs = [(id(w), w["x"], c[0]) for w, c in zip(ws, pick)
                            if w["x"] is not None and not w.get("out") and abs(c[0] - w["x"]) > 1e-9]
                    tr = jog_tracks(jogs, gaps[ri].heights, F) if jogs else {}
                    if tr is not None:
                        result = (ws, pick, tr)
                        break
                if result is not None:
                    break
            if result is None:
                # no placement keeps everyone: the new row cannot join here, and
                # if the passing wires still do not fit, the outermost one yields
                # (to the next layer) until they do
                if new:
                    why[new[0]["net"]] = "no room for it on this layer where its row is"
                new = []
                opts = []
                stay = plan(0.0, False)
                while not stay and order:
                    live_w = [w for w in order if not w.get("out")]
                    if not live_w:
                        break
                    cx = float(np.mean(pads)) if pads else 0.0
                    drop = max((live_w[0], live_w[-1]), key=lambda w: abs(w["x"] - cx))
                    why[drop["net"]] = "no room for it on this layer further down"
                    order = [w for w in order if w is not drop]
                    stay = plan(0.0, False)
                if not stay:
                    for w in order:
                        why[w["net"]] = "no room for it on this layer further down"
                    order = []
                    continue
                result = (stay[0][1], stay[0][2], {})
            ws, pick, tr = result
            for w, c in zip(ws, pick):
                if w["x"] is None:
                    tap = next(t for x, lo_, hi_, t in opts if abs(x - c[0]) < 1e-9 and abs(lo_ - c[1]) < 1e-9
                               and abs(hi_ - c[2]) < 1e-9)
                    w["pts"] = [(tap, y), (c[0], y)]
                    w["x"] = c[0]
                elif not w.get("out") and abs(c[0] - w["x"]) > 1e-9:
                    h = tr[id(w)]
                    w["pts"] += [(w["x"], h), (c[0], h)]
                    w["x"] = c[0]
            order = ws
    paths = {}
    for w in order:
        ex = w["end"] if w.get("out") else exit_y(w["x"])
        if ex is None or not exit_ok(w["x"]):
            why.setdefault(w["net"], "it reaches the bottom of the taxels where no fan can start")
            continue
        paths[w["net"]] = w["pts"] + [(w["x"], ex)]
    for r in rows:
        if r["net"] not in paths:
            why.setdefault(r["net"], "no lane beside any of its vias leads to an exit")
    return paths, why


def _open_tracks(F, iv, heights):
    """the tracks (lowest first) that run under every stretch of taxels of
    this row: a step there stays inside F wherever the row's lanes are"""
    out = []
    for h in heights:
        ih = intervals(F, h)
        if all(any(p - 1e-6 <= u and v <= q + 1e-6 for p, q in ih) for u, v in iv):
            out.append(h)
        else:
            break
    return out


# --------------------------------------------------------------------------
# bridges: a row's pieces joined across the notches between them
# --------------------------------------------------------------------------

def sweep_bridges(links, lat, F, via_xy, via_dx, pitch):
    """links: [{"id", "net", "y", "pixA", "pixB"}] (A: left piece, B: right
    piece). Each link is two half-wires: one leaves A to the right, one
    leaves B to the left; going down, the two meet under the notch on a
    track and the link is done.

    Pieces stacked over each other form a PETAL; a petal's wires keep to its
    own strip (halfway to the petal beside it). The order inside a petal is
    fixed by the nesting all petals must agree on: a link wraps every petal
    it skips; the narrower U goes inside; on a tie the higher row does.
    Returns ({link id: polyline}, {link id: reason})."""
    a, lane = lat.a, lat.lane
    ends = []
    for L in links:
        for side, pix in ((+1, L["pixA"]), (-1, L["pixB"])):
            xs = np.sort(np.unique(np.round(via_xy[pix, 0], 6)))
            ends.append({"L": L, "side": side, "net": L["net"], "y": L["y"], "vias": [float(v) for v in xs],
                         "c0": float(xs.min() - via_dx - pitch / 2), "c1": float(xs.max() - via_dx + pitch / 2)})
    # Strips. Wires on either side of a notch must stay apart until they
    # cross under its root, so the field is cut into strips at the notch
    # roots. Notches come from the links' own gaps: gaps stacked over each
    # other (rows a pitch apart, x overlapping) are one notch, its lowest gap
    # the root. A gap that spans two notches belongs to a link that skips the
    # petal between them; it is not a notch of its own.
    gaps = []
    for L in links:
        x0 = float(np.max(via_xy[L["pixA"], 0])) - via_dx + pitch / 2       # right edge of piece A's cells
        x1 = float(np.min(via_xy[L["pixB"], 0])) - via_dx - pitch / 2       # left edge of piece B's cells
        gaps.append((L["y"], min(x0, x1), max(x0, x1), L["id"]))
    notches = []                                  # each: list of gaps, lowest first
    skip_gap = set()
    for y, g0, g1, lid in sorted(gaps):           # lowest rows first: roots come first
        hits = [k for k, nt in enumerate(notches)
                if abs(nt[-1][0] - y) <= pitch * 1.05 + 1e-6 or abs(nt[-1][0] - y) < 1e-6
                if nt[-1][1] < g1 - 1e-6 and g0 < nt[-1][2] - 1e-6]
        if len(hits) == 1:
            notches[hits[0]].append((y, g0, g1, lid))
        elif len(hits) == 0:
            notches.append([(y, g0, g1, lid)])
        else:
            skip_gap.add(lid)
    roots = sorted(0.5 * (nt[0][1] + nt[0][2]) for nt in notches)

    def strip_of(x):
        return int(np.searchsorted(np.asarray(roots), x))

    nst = len(roots) + 1
    for e in ends:
        # the side of its piece the end leaves from
        e["petal"] = strip_of(max(e["vias"]) if e["side"] > 0 else min(e["vias"]))
    petals = [[e for e in ends if e["petal"] == k] for k in range(nst)]
    lo = [-math.inf] + roots
    hi = roots + [math.inf]
    ext = [((min(e["c0"] for e in g) if g else 0.0), (max(e["c1"] for e in g) if g else 0.0)) for g in petals]
    pet = {(e["L"]["id"], e["side"]): e["petal"] for e in ends}

    # Nesting rank per link (the same at both ends; smallest = innermost U).
    # One hard rule comes from the geometry: a piece in the middle of a row's
    # chain with a single via sends both its wires (left and right) out of
    # that one via, so nothing can pass between them — every wire coming down
    # past it must nest INSIDE its links.
    # Within that, the preference: a link wraps every petal it skips, the
    # narrower U goes inside, and on a tie the higher row does.
    import heapq
    pref = {}
    for L in links:
        skipped = pet[(L["id"], -1)] - pet[(L["id"], +1)] - 1
        span = float(np.min(via_xy[L["pixB"], 0]) - np.max(via_xy[L["pixA"], 0]))
        pref[L["id"]] = (skipped, round(span, 2), -L["y"])
    inside = {L["id"]: set() for L in links}           # outer link -> links that must be inside it
    for g in petals:
        # pieces in the middle of a row's chain: one end leaves left, one right
        by_piece = {}
        for e in g:
            pix = e["L"]["pixA"] if e["side"] > 0 else e["L"]["pixB"]
            by_piece.setdefault((e["net"], tuple(sorted(int(i) for i in pix))), []).append(e)
        for pe in by_piece.values():
            if len(pe) < 2 or len(set(pe[0]["vias"])) != 1:
                continue
            for p_end in pe:
                for f in g:
                    if f["side"] == p_end["side"] and f["y"] > p_end["y"] + 1e-3 and f["net"] != p_end["net"]:
                        inside[p_end["L"]["id"]].add(f["L"]["id"])
    def attempt(inside):
        """one sweep with the nesting constraints `inside`; returns
        (polylines, reasons, first conflict or None)"""
        n_in = {k: 0 for k in inside}
        outs = {k: [] for k in inside}
        for o, ins in inside.items():
            for i in ins:
                outs[i].append(o)
                n_in[o] += 1
        heap = [(pref[k], k) for k, n in n_in.items() if n == 0]
        heapq.heapify(heap)
        rank_of = {}
        while heap:
            _, k = heapq.heappop(heap)
            rank_of[k] = len(rank_of)
            for o in outs[k]:
                n_in[o] -= 1
                if n_in[o] == 0:
                    heapq.heappush(heap, (pref[o], o))
        for k in pref:                      # a cycle: no order satisfies every via; fall back to the preference
            if k not in rank_of:
                rank_of[k] = len(rank_of) + 1000
        for e in ends:
            e["rank"] = (rank_of[e["L"]["id"]],)

        def key(e):            # left to right in a petal: left-going inner first, then right-going inner last
            return (0, e["rank"]) if e["side"] < 0 else (1, tuple(-v for v in e["rank"]))

        # each end's final slot in its petal's nesting: on its own side of the
        # petal, the innermost link at the edge (where it meets its partner), the
        # next one a lane further in, and so on. A wire aims there from the start,
        # so a wire from above leaves room for an inner one that starts lower.
        # An outer wire must also stay outside every inner wire's way in: an
        # inner end starts beside one of its own vias, at the latest the one
        # nearest its edge, so the outer wire keeps that via's keepout plus a lane
        # for each wire nested between them.
        tgt = {}
        for pi, g in enumerate(petals):
            for side in (-1, +1):
                ranked = sorted((e for e in g if e["side"] == side), key=lambda e: e["rank"])
                if side < 0:
                    edge = lo[pi] if math.isfinite(lo[pi]) else ext[pi][0]
                else:
                    edge = hi[pi] if math.isfinite(hi[pi]) else ext[pi][1]
                for k, e in enumerate(ranked):
                    t = edge - side * (k + 0.5) * lane
                    for kj, j in enumerate(ranked[:k]):
                        way_in = (max(j["vias"]) if side > 0 else min(j["vias"])) - side * (a + (k - kj - 1) * lane)
                        t = min(t, way_in) if side > 0 else max(t, way_in)
                    tgt[id(e)] = t

        def target(e):
            return tgt[id(e)]

        order = [[] for _ in petals]
        wire_of, out, why = {}, {}, {}
        conflict = None
        Yd = lat.Yd
        merges = {}                          # gap below row index -> tracks used by meetings

        def plan_row(y, iv, reach, gap):
            """lanes for every petal's wires on this row, without changing
            anything. Returns ({petal: (wires, picks)}, {petal: reason}, jogs)."""
            plans, failed, jogs = {}, {}, []
            for pi in range(len(petals)):
                new = [e for e in petals[pi] if abs(e["y"] - y) < 1e-3]
                if not new and not order[pi]:
                    continue
                S = [float(x) for x in lat.lanes if lo[pi] + lane / 2 <= x <= hi[pi] - lane / 2
                     and which(iv, float(x)) is not None]
                ws = order[pi] + [{"e": e, "net": e["net"], "x": None, "pts": []} for e in new]
                ws.sort(key=lambda w: key(w["e"]))
                elems = []
                for w in ws:
                    tg = target(w["e"])
                    if w["x"] is None:
                        cs = [(x, min(t, x), max(t, x), abs(x - tg) + 1e-3 * abs(x - t))
                              for t in w["e"]["vias"] for x in lat.gap_lanes(t) if x in S]
                    else:
                        k = which(iv, w["x"])
                        u, v = iv[k] if k is not None else (math.inf, -math.inf)
                        cs = [(x, x, x, abs(x - tg) + 1e-3 * abs(x - w["x"])) for x in S
                              if u + 1e-3 <= x <= v - 1e-3 and abs(x - w["x"]) <= reach + 1e-9
                              and (abs(x - w["x"]) < 1e-9 or gap.can_step(w["x"], x))]
                    elems.append((w["net"], cs))
                pick = place_row(elems, lane)
                if pick is None:
                    failed[pi] = ("its petal's wires do not fit through this row", ws)
                    continue
                plans[pi] = (ws, pick)
                for w, c in zip(ws, pick):
                    if w["x"] is not None and abs(c[0] - w["x"]) > 1e-9:
                        jogs.append((id(w), w["x"], c[0]))
            return plans, failed, jogs

        for ri, y in enumerate(Yd):
            iv = intervals(F, y)
            if ri > 0:
                # the top of the gap above is taken by the meetings placed there
                free = cap(Yd[ri - 1] - y, a, lane) - merges.get(ri - 1, 0)
                gap = Gap(F, lat.tracks_up(y, Yd[ri - 1])[:max(0, free)])
                safe = _open_tracks(F, iv, gap.heights)
            else:
                gap, safe = Gap(F, []), []
            # A wire may step as far as the gap has tracks (steps the same way
            # need as many tracks as their spans overlap). If the tracks cannot
            # then be assigned, the row is redone with the reach of the tracks
            # that run under the whole row, which always can be.
            result = None
            for reach in (gap.reach() * lane, max(0, len(safe) - 1) * lane):
                plans, failed, jogs = plan_row(y, iv, reach, gap)
                tr = jog_tracks(jogs, gap.heights, F) if jogs else {}
                if tr is not None:
                    result = (plans, failed, tr)
                    break
            if result is None:          # cannot happen with no steps; keep wires put
                plans, failed, _ = plan_row(y, iv, 0.0, gap)
                result = (plans, failed, {})
            plans, failed, tr = result
            for pi, (reason, ws_f) in failed.items():
                if conflict is None:
                    conflict = (pi, y, ws_f)
                for w in order[pi]:
                    why[w["e"]["L"]["id"]] = reason
                for e in petals[pi]:
                    if abs(e["y"] - y) < 1e-3:
                        why[e["L"]["id"]] = reason
                order[pi] = []
            for pi, (ws, pick) in plans.items():
                for w, c in zip(ws, pick):
                    if w["x"] is None:
                        e = w["e"]
                        tap = next(t for t in e["vias"]
                                   if abs(min(t, c[0]) - c[1]) < 1e-9 and abs(max(t, c[0]) - c[2]) < 1e-9)
                        w["pts"] = [(tap, y), (c[0], y)]
                        w["x"] = c[0]
                        wire_of[(e["L"]["id"], e["side"])] = w
                    elif abs(c[0] - w["x"]) > 1e-9:
                        h = tr[id(w)]
                        w["pts"] += [(w["x"], h), (c[0], h)]
                        w["x"] = c[0]
                order[pi] = ws
            if ri + 1 >= len(Yd):
                break
            # meetings in the gap below this row, innermost first
            downs = lat.tracks_down(y, Yd[ri + 1])
            used, progress = 0, True
            while progress and used < len(downs):
                progress = False
                for pi in range(len(petals)):
                    if not order[pi] or order[pi][-1]["e"]["side"] < 0:
                        continue
                    w = order[pi][-1]
                    pw = wire_of.get((w["e"]["L"]["id"], -1))
                    if pw is None:
                        continue
                    qi = pw["e"]["petal"]
                    if qi <= pi or any(order[k] for k in range(pi + 1, qi)) or not order[qi] or order[qi][0] is not pw:
                        continue
                    yt = downs[used]
                    if not F.contains(LineString([(w["x"], yt), (pw["x"], yt)])):
                        continue
                    out[w["e"]["L"]["id"]] = (w["pts"] + [(w["x"], yt), (pw["x"], yt)]
                                              + (pw["pts"] + [(pw["x"], yt)])[::-1][1:])
                    order[pi].pop()
                    order[qi].pop(0)
                    used += 1
                    progress = True
                    if used >= len(downs):
                        break
            merges[ri] = used
        for L in links:
            if L["id"] not in out:
                why.setdefault(L["id"], "its two ends never met under the notch")
        return out, why, conflict

    # Learn from conflicts: a petal row that cannot be placed is one whose new
    # links were nested outside wires that cannot get past their entry. Nest
    # them inside those wires instead and sweep again (a sweep is cheap); stop
    # when nothing new can be learned. Keep the best sweep.
    learned = {k: set(v) for k, v in inside.items()}
    best = None
    for _ in range(len(links) + 1):
        out, why, conflict = attempt(learned)
        if best is None or len(out) > len(best[0]):
            best = (out, why)
        if conflict is None:
            break
        _, _, ws_f = conflict
        added = False
        for n in (w for w in ws_f if w["x"] is None):
            for f in (w for w in ws_f if w["x"] is not None):
                if f["e"]["side"] == n["e"]["side"] and f["net"] != n["net"]:
                    o, i = f["e"]["L"]["id"], n["e"]["L"]["id"]
                    if i not in learned[o] and o not in learned[i]:
                        learned[o].add(i)
                        added = True
        if not added:
            break
    return best


# --------------------------------------------------------------------------
# field exit -> connector pad: a nested comb along the field's outline
# --------------------------------------------------------------------------

def _ring_cross(ring, x, y, up):
    """where a vertical ray from (x, y) first crosses `ring` (up or down)"""
    ray = LineString([(x, y), (x, y + (1e4 if up else -1e4))])
    hit = ray.intersection(ring)
    pts = []
    for g in getattr(hit, "geoms", [hit]):
        pts += [tuple(c) for c in getattr(g, "coords", [])]
    pts = [p for p in pts if (p[1] > y + 1e-9 if up else p[1] < y - 1e-9)]
    if not pts:
        return None
    return min(pts, key=lambda p: abs(p[1] - y))


def _hugs_or_beyond(body, a, b, gap):
    """the run a-b (horizontal) hugs the corridor: everywhere along it the
    corridor is at most `gap` straight above, or not above it at all (the
    run is out past the corridor's footprint). A run with open space
    between it and the corridor would be a shortcut across a bay."""
    import shapely
    n = max(2, int(math.ceil(abs(b[0] - a[0]) / (gap / 2))))
    xs = a[0] + (b[0] - a[0]) * np.arange(n + 1) / n
    y0 = np.full(n + 1, a[1] + 1e-6)
    ray = lambda h: shapely.linestrings(np.stack([np.column_stack([xs, y0]),
                                                  np.column_stack([xs, y0 + h])], axis=1))
    shapely.prepare(body)
    far = shapely.intersects(body, ray(1e4))
    near = shapely.intersects(body, ray(gap))
    return not np.any(far & ~near)


def _leave(ring, body, u0, pad, sd, entry=0.0, gap=None):
    """Where a wire riding `ring` from arc length u0 in direction sd (+1 with
    the ring's order, -1 against it) leaves for `pad`, as (arc length
    ridden, leave point, elbow or None), or None.

    The hug router's fan-comb (route_hug._pad_entry), followed along the
    ride: the wire leaves at the first point q from which the L to its pad
    (across to the pad's x at q's height, then straight down) stays outside
    the corridor `body`. Under a flat stretch that is the vertical drop over
    the pad; past the corridor's footprint it is the run out sideways then
    down; and a wire never climbs back over an overhang to reach a pad that
    an earlier stretch already looks down on. The last `entry` above the
    pad is exempt: that is where the wire goes in through the connector's
    keep-out."""
    from shapely.geometry import Point
    L = ring.length
    coords = list(ring.coords)[:-1]
    us = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(np.asarray(coords), axis=0), axis=1))].tolist()
    ray = LineString([tuple(pad), (pad[0], pad[1] + 1e4)]).intersection(ring)
    for g in getattr(ray, "geoms", [ray]):
        us += [ring.project(Point(c)) for c in getattr(g, "coords", [])]
    # rides past half the ring lose to the other way round anyway
    cand = sorted({t for t in (round(((u - u0) * sd) % L, 9) for u in us) if t <= L / 2} | {0.0})
    for t in cand:
        q = ring.interpolate((u0 + sd * t) % L).coords[0]
        if q[1] < pad[1] - 1e-9:
            continue
        elbow = (pad[0], q[1])
        stop = (pad[0], pad[1] + entry)
        if q[1] < stop[1] - 1e-9:
            continue
        pts = [q] + ([elbow] if abs(elbow[0] - q[0]) > 1e-9 else []) + [stop]
        if pts[-2] == pts[-1]:
            pts.pop()
        if (len(pts) == 1 and not body.contains(Point(q))) or (
                len(pts) > 1 and not body.intersects(LineString(pts))):
            run = abs(elbow[0] - q[0]) > 1e-9
            if run and gap is not None and not _hugs_or_beyond(body, q, elbow, gap):
                continue
            return t, q, (elbow if run else None)
    return None


def bus_to_pads(items, lane, floor_y):
    """items: [(key, exit_xy, pad_xy)] — wires that leave the field straight
    above the connector tail, for pads below them, in the same left-to-right
    order as their exits.

    Each wire drops straight down from its exit, takes one 45-degree jog
    over to its pad's x, and drops onto the pad. The jogs all end at or
    above floor_y (clear of the connector's pins). Wires jogging the same way
    are parallel 45-degree lines, which stand a lane apart only if their
    starts are staggered: a wire further along the jog direction starts its
    jog higher, by just enough. Returns ({key: polyline exit -> pad}, failed
    keys); a wire whose jog would have to start above its exit fails."""
    stag = lane * math.sqrt(2) * 1.02
    its = sorted(items, key=lambda it: it[1][0])
    jog_y = {}
    for sign, seq in ((+1, its), (-1, its[::-1])):
        prev = None
        for key, ex, pad in seq:
            d = (pad[0] - ex[0]) * sign
            if d <= 1e-9:
                continue
            y = floor_y + d
            if prev is not None:
                y = max(y, prev[1] - abs(ex[0] - prev[0]) + stag)
            jog_y[key] = y
            prev = (ex[0], y)
    out, failed = {}, set()
    for key, ex, pad in its:
        if key not in jog_y:
            out[key] = [tuple(ex), tuple(pad)]
            continue
        y = jog_y[key]
        if y > ex[1] + 1e-9:
            failed.add(key)
            continue
        d = abs(pad[0] - ex[0])
        pts = [tuple(ex), (ex[0], y), (pad[0], y - d), tuple(pad)]
        out[key] = [p for i, p in enumerate(pts) if i == 0 or math.dist(p, pts[i - 1]) > 1e-6]
    return out, failed


def comb_to_pads(items, field_poly, base, lane, cut_x, keepout=None, entry=0.0):
    """items: [(key, exit_xy, pad_xy)] — wires that leave the field at
    exit_xy (on its outline) for the connector pad pad_xy below it.

    The hug router's corridor, drawn with exact geometry. The corridor
    starts as the field grown by `base`. Wires are laid one at a time: each
    drops from its exit to the corridor's outline, rides it, and leaves for
    its pad (_leave), then the corridor grows by the new wire buffered by
    `lane`. Every later wire rides around all the earlier ones at exactly a
    lane, so drops, runs and staircases nest without any rule per case.

    Order: each wire rides the way that is shorter around the bare corridor
    (cut_x only breaks exact ties). Among wires riding the same way the one
    whose exit lies furthest downstream goes first: an upstream wire passes
    over that exit, so it has to ride outside the downstream one, and it
    leaves for its pad before the downstream wire does (pads were handed
    out in exit order). Returns ({key: polyline exit -> pad}, failed keys).
    """
    from shapely.geometry import Point
    from shapely.geometry.polygon import orient
    from shapely.ops import substring
    # buffer() draws each arc as chords 2*pi/(4*16) wide, which cut inside
    # the true circle by r*(1 - cos(pi/64)); stretch every offset by that so
    # the drawn copper keeps the full distance
    # the touch allowance `tol` lets a wire run along the outline; the extra
    # 2*tol on every offset keeps that allowance from eating into the lane
    sag = 1.0 / math.cos(math.pi / 64)
    tol = 1e-4
    # The corridor is redrawn with points at least `thin` apart from the
    # true outline (simplify): every buffered wire would otherwise copy the
    # rounded joins of the outline it rode, and the point count compounds
    # wire after wire (17 nested wires: ~1600 points each, 2 um apart).
    # Simplifying may move the outline by up to `thin` either way, so every
    # offset is grown by `thin` to keep the full clearance.
    thin = 0.001
    out, failed = {}, set()
    seed = field_poly.buffer(base * sag + 2 * tol + thin, quad_segs=16)
    anchor = field_poly.representative_point()

    def solid(G):
        """corridor pieces without holes; the one holding the field first
        (the one wires ride); the keep-out may stand apart from it"""
        parts = [Polygon(g.exterior.coords).simplify(thin, preserve_topology=True)
                 for g in getattr(G, "geoms", [G])]
        parts.sort(key=lambda g: not g.covers(anchor))
        return [orient(g, 1.0) for g in parts]

    R = solid(seed.union(keepout) if keepout is not None else seed)
    ring0 = LineString(R[0].exterior.coords)
    inside = lambda P: unary_union(P).buffer(-tol)
    hug_gap = 1.5 * lane
    L0 = ring0.length
    groups = {-1: [], +1: []}
    for it in items:
        key, ex, pad = it
        p_ex = _ring_cross(ring0, ex[0], ex[1], up=False)
        if p_ex is None:
            failed.add(key)
            continue
        u0 = ring0.project(Point(p_ex))
        cost = {}
        for d in (+1, -1):
            lv = _leave(ring0, inside(R), u0, pad, d, entry, hug_gap)
            if lv is not None:
                cost[d] = lv[0] + math.dist(lv[1], pad)
        if not cost:
            failed.add(key)
            continue
        if len(cost) == 2 and abs(cost[1] - cost[-1]) < 1e-6:
            # both ways leave at once: ride the way the pad lies
            toward = pad[0] - p_ex[0] if abs(pad[0] - p_ex[0]) > 1e-6 else pad[0] - cut_x
            sd = +1 if (ring0.interpolate((u0 + 1e-3) % L0).x - p_ex[0]) * toward > 0 else -1
        else:
            sd = min(cost, key=cost.get)
        groups[sd].append((u0, it))
    order = []
    for sd, grp in groups.items():
        if grp:
            ref = grp[0][0]
            rel = lambda u: ((u - ref + L0 / 2) % L0) - L0 / 2
            grp.sort(key=lambda g: -sd * rel(g[0]))    # most downstream first
            order += [(sd, it) for _, it in grp]
    for sd, (key, ex, pad) in order:
        ring = LineString(R[0].exterior.coords)
        L = ring.length
        p_ex = _ring_cross(ring, ex[0], ex[1], up=False)
        if p_ex is None:
            failed.add(key)
            continue
        u0 = ring.project(Point(p_ex))
        body = inside(R)
        opts = [(lv[0], d, lv) for d in (sd, -sd) for lv in [_leave(ring, body, u0, pad, d, entry, hug_gap)] if lv is not None]
        if not opts:
            failed.add(key)
            continue
        # the group's way if it can get there within half the ring
        t, sd_w, (_, p_out, elbow) = min(opts, key=lambda o: (o[1] != sd, o[0]))
        if sd_w > 0:
            arc = _arc(ring, u0, u0 + t, L, substring)
        else:
            u1 = (u0 - t) % L
            arc = _arc(ring, u1, u1 + t, L, substring)[::-1]
        path = [tuple(ex), tuple(p_ex)] + arc + [tuple(p_out)] + ([elbow] if elbow else []) + [tuple(pad)]
        if keepout is not None and min(q[1] for q in path[:-1]) < pad[1] + entry - 1e-6:
            # it would pass under the pad row: around the connector, not into it
            failed.add(key)
            continue
        clean = [path[0]]
        for q in path[1:]:
            if math.dist(q, clean[-1]) > 1e-6:
                clean.append(q)
        out[key] = clean
        # the corridor grows by everything but the final drop onto the pad:
        # wires outside this one leave before they reach it, and the drops
        # stand a pad pitch apart, so their tubes would fuse into a wall
        # under the field that later wires could not get past
        body_part = clean[:-1] if len(clean) > 2 else clean
        R = solid(unary_union(R + [LineString(body_part).buffer(lane * sag + 2 * tol + thin, quad_segs=16)]))
    return out, failed


def _arc(ring, a, b, L, substring):
    """ring points from arc length a to b (b may run past L: wraps)"""
    if b <= L + 1e-9:
        return [tuple(c) for c in substring(ring, a, min(b, L)).coords]
    return [tuple(c) for c in substring(ring, a, L).coords] + [tuple(c) for c in substring(ring, 0, b - L).coords]
