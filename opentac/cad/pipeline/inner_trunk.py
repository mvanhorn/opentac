"""Inner-layer row feeds for 4-layer boards: one steered bundle from the
connector.

The wires run on the via lattice: down lanes (x positions in the gaps between
two via columns, a via keep-out clear of both, a trace pitch apart — about
four per gap) and across band lanes (the same between two via rows). A lane
never comes near a via, so a wire may go wherever the field is, in straight
axis-aligned legs.

One bundle carries the rows to the connector, in row order:

  - rows join from the top down — the farthest first — each on the bundle's
    outer side (one side, the same for every row), so the wires, and the pads
    they are handed, come out in row order;
  - a row joins at any of its vias beside that outer side: a crown bump or a
    short row joins wherever it has one;
  - the bundle wants to lie packed against one lane, the trunk. Between two
    via rows a wire may jog sideways along a band lane toward its place, or
    away from a lane the field ends under, so the bundle steers round narrow
    sections and gathers at the trunk;
  - jogs are staircased — wires moving left jog leftmost first, each one band
    lane lower, and the mirror image for wires moving right — so no two wires
    ever cross, and none passes another;
  - each wire leaves through the field's bottom edge; the exits, left to
    right, are handed the connector pads in that order and fanned down.

Nothing is searched: the trunk lane and the side are chosen by building the
bundle for each and keeping the one that places the most rows (then the
fewest jogs, then the nearest the pads), and every leg is checked. A row
that cannot join is refused; given a second layer (In2, round the links
already there), route_feeds builds a second bundle there for the refused
rows. Only what fits on neither stays on B.Cu.
"""

import os
from bisect import bisect_left

import numpy as np
from shapely.geometry import LineString
from shapely.ops import unary_union

from .inner_hug import _M, _Layer, _Space, _bottom
from .route_inner import _cluster
from .types import Route


def _lanes(X, a, s):
    """per gap g (between via columns X[g] and X[g+1]): the lane x positions
    a via keep-out clear of both columns, a trace pitch apart, centred"""
    out = []
    for g in range(len(X) - 1):
        w = X[g + 1] - X[g] - 2 * (a + _M)
        if w < 0:
            out.append([])
            continue
        n = int(np.floor(w / (s + _M) + 1e-9)) + 1
        off = (w - (n - 1) * (s + _M)) / 2
        out.append([X[g] + a + _M + off + i * (s + _M) for i in range(n)])
    return out


def _fans(exits, starts, y_top, s, deg=None):
    """Fans from the wires' exits (x, y) — left to right, at the field's
    bottom edge — down to their pads' starts (x, y), in the same order: each
    drops straight down, slants across, drops onto its pad. Slants all run at
    one angle; the wires shifting right start slanting rightmost first, those
    shifting left leftmost first, each a little lower than the last, so a
    neighbour is always a pitch clear: never crossing, never squeezed.
    Returns [polyline exit -> start]."""
    n = len(exits)
    if not n:
        return []
    dx = [st[0] - ex[0] for ex, st in zip(exits, starts)]
    right = sorted((i for i in range(n) if dx[i] > 1e-9), key=lambda i: -exits[i][0])
    left = sorted((i for i in range(n) if dx[i] < -1e-9), key=lambda i: exits[i][0])
    rank = {i: k for k, i in enumerate(right)}
    rank.update({i: k for k, i in enumerate(left)})
    # Neighbours slanting side by side at angle t (from vertical) need
    # (s + M) / cos t across; they start (x_j - x_i) apart, and the one ahead
    # gains tan t for every unit it starts higher. So the stagger per wire is
    # what makes up the difference — next to nothing on a steep slant, a
    # full pitch on a flat run. Pick the angle (8 degrees .. flat) leaving
    # the most room above the pads, unless deg names one.
    xs_ = sorted(e[0] for e in exits)
    sep = min([b - a for a, b in zip(xs_[:-1], xs_[1:])] + [s + _M])
    dmax = max([abs(d) for d in dx] + [1e-9])
    groups = max(len(right), len(left), 1)
    floor = max(st[1] for st in starts)
    best = None
    for dg in ([deg] if deg is not None else range(8, 91, 2)):
        t = np.radians(dg)
        if dg == 90:
            delta, cot = s + _M + 1e-3, 0.0
        else:
            delta, cot = max(0.0, (s + _M) / np.cos(t) - sep) / np.tan(t) + 1e-3, 1.0 / np.tan(t)
        need = delta * groups + dmax * cot
        spare = (y_top - floor) - need
        if best is None or spare > best[0] + 1e-9:
            best = (spare, delta, cot)
    _, delta, cot = best
    out = []
    for i in range(n):
        (ex, ey), (sx, sy) = exits[i], starts[i]
        y_s = y_top - rank.get(i, 0) * delta
        y_t = y_s - abs(dx[i]) * cot
        out.append([(ex, ey), (ex, y_s), (sx, y_t), (sx, sy)])
    return out


def _widen(V):
    """the lattice lines V with one more line a median pitch past each end"""
    p = float(np.median(np.diff(V))) if len(V) > 1 else 2.0
    return np.concatenate([[V[0] - p], V, [V[-1] + p]])


class _Grid:
    """The lattice one layer's wires run on, and where the field lets them.

    Slot k: the k-th lane from the left (x SX[k], in gap SG[k] between via
    columns X[g] and X[g + 1]). Band b: between via rows Y[b] and Y[b + 1],
    with band lanes hl[b]. The field's cover along every slot's vertical and
    every horizontal used is measured once, as intervals, so the checks of a
    build are lookups. obstacles: [(net, polyline)] already on the layer
    (In2's links) — then every leg is also checked a pitch clear of them."""

    def __init__(self, sp, obstacles=()):
        self.sp = sp
        # one more column / row of lattice past each side (with no vias):
        # lanes beside the outermost via columns, and the band under the
        # lowest via row, wherever the field reaches there
        self.X = X = _widen(_cluster(sp.vxy[:, 0], 0.05))
        self.Y = Y = _widen(_cluster(sp.vxy[:, 1], 0.05))
        lanes = _lanes(X, sp.a, sp.s)
        self.SX = [x for xs in lanes for x in xs]
        self.SG = [g for g, xs in enumerate(lanes) for _ in xs]
        self.gap_slots, k = [], 0
        for xs in lanes:
            self.gap_slots.append(list(range(k, k + len(xs))))
            k += len(xs)
        self.hl = _lanes(Y, sp.a, sp.s)
        self.via_at, self.vias_in_row = {}, {}
        for i, (x, y) in enumerate(sp.vxy):
            c, r = int(np.argmin(np.abs(X - x))), int(np.argmin(np.abs(Y - y)))
            self.via_at[(c, r)] = i
            self.vias_in_row.setdefault((sp.vnet[i], r), []).append((c, i))
        b = sp.region_b.bounds
        self._b = b
        self._v = [self._iv(LineString([(x, b[1] - 1.0), (x, b[3] + 1.0)]), 1) for x in self.SX]
        self._h = {}
        self.obstacles = [(n, LineString(p)) for n, p in obstacles]
        self._obst, self._taps = {}, {}

    def _iv(self, line, axis):
        hit = self.sp.region_b.intersection(line)
        out = []
        for q in getattr(hit, "geoms", [hit]):
            if q.is_empty or q.geom_type != "LineString":
                continue
            c = np.asarray(q.coords)[:, axis]
            out.append((float(c.min()), float(c.max())))
        return sorted(out)

    @staticmethod
    def _cov(iv, a, b):
        lo, hi = min(a, b), max(a, b)
        return any(p <= lo + 1e-9 and q >= hi - 1e-9 for p, q in iv)

    def vcov(self, k, y0, y1):
        return self._cov(self._v[k], y0, y1)

    def hcov(self, y, x0, x1):
        key = round(float(y), 6)
        if key not in self._h:
            b = self._b
            self._h[key] = self._iv(LineString([(b[0] - 1.0, y), (b[2] + 1.0, y)]), 0)
        return self._cov(self._h[key], x0, x1)

    def exit_y(self, k, y):
        """where slot k's vertical, going down from y, leaves the field through
        its bottom edge — None if the field resumes below (that is no exit)"""
        iv = self._v[k]
        for j, (p, q) in enumerate(iv):
            if p <= y + 1e-9 and q >= y - 1e-9:
                return p if j == 0 else None
        return None

    def runs(self, r):
        """[(first slot, last slot)]: the stretches of field across via row r
        (consecutive slots joined by field along the row)"""
        if not hasattr(self, "_runs"):
            self._runs = {}
        if r not in self._runs:
            y, out = self.Y[r], []
            for k in range(len(self.SX)):
                if not self.vcov(k, y, y):
                    continue
                if out and out[-1][1] == k - 1 and self.hcov(y, self.SX[k - 1], self.SX[k]):
                    out[-1][1] = k
                else:
                    out.append([k, k])
            self._runs[r] = [tuple(q) for q in out]
        return self._runs[r]

    def taps(self, net, r):
        """[(slot, via, x lo, x hi)]: the lanes beside `net`'s vias on via
        row r that a stub along the row reaches inside the field (and clear
        of obstacles), with the x span the stub covers"""
        key = (net, r)
        if key not in self._taps:
            sp, X, SX, y = self.sp, self.X, self.SX, self.Y[r]
            eps = sp.a + _M
            out = []
            for c, i in self.vias_in_row.get(key, ()):
                for g in (c - 1, c):
                    if g < 0 or g >= len(self.gap_slots):
                        continue
                    for k in self.gap_slots[g]:
                        stub = [tuple(sp.vxy[i]), (SX[k], y), (SX[k], y - eps)]
                        if self.hcov(y, X[c], SX[k]) and self.vcov(k, y, y - eps) and self.clear(net, stub):
                            out.append((k, i, min(X[c], SX[k]), max(X[c], SX[k])))
            self._taps[key] = out
        return self._taps[key]

    def clear(self, net, pts):
        """a pitch from every obstacle of another net (no obstacles: True)"""
        if not self.obstacles:
            return True
        if net not in self._obst:
            others = [w for n, w in self.obstacles if n != net]
            self._obst[net] = unary_union(others) if others else None
        o = self._obst[net]
        return o is None or LineString(pts).distance(o) >= self.sp.s - 1e-6


def _steer(G, rows_up, t, d, y_exit_max, first_inner=True, window=(-np.inf, np.inf), clamp=False):
    """One bundle with its trunk at slot t, growing outward to the right
    (d = +1) or to the left (d = -1). rows_up: [(net, via row)] top down.
    Wire j (the j-th to join) wants slot t + d*j; with clamp, t is clamped,
    row by row, into the stretch of field the bundle is passing (trunk_at). A wire may leave the field
    only through its bottom edge, no higher than y_exit_max, between the
    window's x — where a fan can still reach the pads. Returns
    ({net: [tap via, ..., exit]}, {net: why}, jogs, rows joined inside)."""
    sp, X, Y, SX, SG = G.sp, G.X, G.Y, G.SX, G.SG
    by_row = {}
    for net, r in rows_up:
        by_row.setdefault(r, []).append(net)
    W = []                  # inner -> outer: [net, slot, points, exited]
    why, jogs, inner = {}, 0, 0
    eps = sp.a + _M         # a wire's first leg must clear its via row

    def exit_at(k, y0):
        ye = G.exit_y(k, y0)
        ok = ye is not None and ye <= y_exit_max and window[0] - 1e-9 <= SX[k] <= window[1] + 1e-9
        return ye if ok else None

    def joins(net, r):
        """[(position, slot, tap via)] where `net` can join at via row r: a
        lane beside one of its vias, reachable by a stub along the via row
        that crosses no wire; position p = how many wires lie inward of it
        (len(W): outward of them all)"""
        wx = sorted(SX[w[1]] for w in W)
        out = []
        for k, i, lo, hi in G.taps(net, r):
            j = bisect_left(wx, lo - 1e-9)
            if j < len(wx) and wx[j] <= hi + 1e-9:
                continue            # the stub would cross a wire (or the lane is taken)
            below = bisect_left(wx, SX[k]) if d > 0 else len(wx) - bisect_left(wx, SX[k])
            out.append((below, k, i))
        return out

    def trunk_at(r):
        """the trunk slot at via row r: t, clamped into the stretch of field
        the bundle is in there — a neck or a petal it has to pass — so the
        bundle gathers where the field lets it and returns toward t below"""
        runs = G.runs(r) if clamp and 0 <= r < len(Y) else []
        if not runs:
            return t
        live = [w[1] for w in W if not w[3]]
        ref = live[0] if live else t
        run = next((q for q in runs if q[0] <= ref <= q[1]), None) or \
            min(runs, key=lambda q: min(abs(ref - q[0]), abs(ref - q[1])))
        return min(max(t, run[0]), run[1])

    for r in range(max(by_row), -1, -1):
        y = Y[r]
        tr = trunk_at(r)
        # 1) this via row's rows join: on the bundle's outer side, the one
        # reaching it nearest first. A row with no via beside the outer side
        # joins inside the bundle instead, between the two wires its via
        # lies between — the nearest the outer side it can.
        pending = list(by_row.get(r, []))
        while pending:
            best = None
            for net in pending:
                c = joins(net, r)
                if not c:
                    continue
                if not W:           # the first wire: nearest the trunk, on the side asked
                    key = lambda pk: ((d * (pk[1] - tr) > 0) == first_inner, abs(pk[1] - tr))
                else:
                    key = lambda pk: (-pk[0], abs(pk[1] - (tr + d * pk[0])), d * pk[1])
                pk = min(c, key=key)
                rank = (pk[0] < len(W), d * pk[1] if pk[0] == len(W) else -pk[0], net)
                if best is None or rank < best[0]:
                    best = (rank, net, pk)
            if best is None:
                for net in pending:
                    why[net] = "no via of its row beside the bundle"
                break
            _, net, (pos, k, i) = best
            pending.remove(net)
            inner += pos < len(W)
            W.insert(pos, [net, k, [tuple(sp.vxy[i]), (SX[k], y)], False])

        live = [j for j, w in enumerate(W) if not w[3]]
        if r == 0:
            for j in live:
                k = W[j][1]
                ye = exit_at(k, y)
                if ye is not None and G.clear(W[j][0], [(SX[k], y), (SX[k], ye)]):
                    W[j][2].append((SX[k], ye))
                    W[j][3] = True
            break
        y_lo = Y[r - 1]

        def down(k, y0):
            """going down slot k from y0: "on" to the next via row, the exit
            y where it leaves through the field's bottom edge, or None"""
            return "on" if G.vcov(k, y0, y_lo) else exit_at(k, y0)

        # 2) jogs in the band below: toward lower slots, left to right, then
        # toward higher slots, right to left; each mover one band lane lower
        # than the last in its pass (the two passes may share band lanes).
        # A wire not in its place moves toward it; one whose lane the field
        # ends under (not at its bottom edge) moves wherever it can.
        ys = sorted(G.hl[r - 1], reverse=True)
        slot = {j: W[j][1] for j in range(len(W))}
        fate = {j: down(slot[j], y) for j in live}
        fate = {j: (f if f is None or G.clear(W[j][0], [(SX[slot[j]], y), (SX[slot[j]], y_lo if f == "on" else f)])
                    else None) for j, f in fate.items()}
        tb = trunk_at(r - 1)
        goal = {j: tb + d * j for j in live}
        new, lane_of, after = dict(slot), {}, {}
        by_x = sorted(range(len(W)), key=lambda j: slot[j])

        def jog(j, k0, k1, yl):
            if not (G.vcov(k0, y, yl) and G.hcov(yl, SX[k0], SX[k1])):
                return None
            f = down(k1, yl)
            if f is None:
                return None
            pts = [(SX[k0], y), (SX[k0], yl), (SX[k1], yl), (SX[k1], y_lo if f == "on" else f)]
            return f if G.clear(W[j][0], pts) else None

        def take(j, cands, li):
            for k1 in cands:
                f = jog(j, slot[j], k1, ys[li])
                if f is not None:
                    new[j], lane_of[j], after[j] = k1, ys[li], f
                    return True
            return False

        li = 0
        for pos, j in enumerate(by_x):
            if j not in fate or li >= len(ys):
                continue
            k, forced = slot[j], fate[j] is None
            if not (goal[j] < k or (forced and goal[j] == k)):
                continue
            lb = new[by_x[pos - 1]] + 1 if pos > 0 else 0
            if take(j, range(max(lb, min(goal[j], k - 1)), k), li) or \
                    (forced and take(j, range(k - 1, lb - 1, -1), li)):
                li += 1
        li = 0
        for pos in range(len(by_x) - 1, -1, -1):
            j = by_x[pos]
            if j not in fate or j in lane_of or li >= len(ys):
                continue
            k, forced = slot[j], fate[j] is None
            if not (goal[j] > k or forced):
                continue
            ub = new[by_x[pos + 1]] - 1 if pos + 1 < len(by_x) else len(SX) - 1
            if (goal[j] > k and take(j, range(min(ub, goal[j]), k, -1), li)) or \
                    (forced and take(j, range(k + 1, ub + 1), li)):
                li += 1
        drop = []
        for j in live:
            k = slot[j]
            if j in lane_of:
                W[j][2] += [(SX[k], lane_of[j]), (SX[new[j]], lane_of[j])]
                W[j][1] = new[j]
                jogs += 1
                if after[j] != "on":
                    W[j][2].append((SX[new[j]], after[j]))
                    W[j][3] = True
            elif fate[j] is None:
                drop.append(j)
            elif fate[j] != "on":                  # 3) it leaves through the bottom edge
                W[j][2].append((SX[k], fate[j]))
                W[j][3] = True
        for j in sorted(drop, reverse=True):
            why[W[j][0]] = "the field ends under it and the bundle cannot shift it"
            del W[j]

    placed = {}
    for net, k, pts, done in W:
        if done:
            placed[net] = pts
        else:
            why[net] = "it does not reach the field's bottom edge"
    for net, _ in rows_up:
        if net not in placed and net not in why:
            why[net] = "no via of its row beside the bundle's outer side"
    return placed, why, jogs, inner


def route_feeds(layout, params, plan, layers=("In1.Cu",), preplaced=None):
    """Returns ({layer: [Route]}, failed_nets, info) — the route_rows_inner
    contract. Each Route runs pad -> tap via. Mutates layout.pad_to_net:
    a layer's rows get their pads in the order their wires leave the field.
    A second layer, when given, takes a bundle of the rows the first refused
    (round what preplaced already holds there)."""
    rows = {net: ch for net, ch in plan.chains.items() if ch.face == "row"}
    info = {"rows": len(rows), "rows_failed": [], "fail_reasons": {}, "rows_bcu_fallback": 0,
            "per_layer": {L: 0 for L in layers}, "direct": 0, "channel_rows": 0, "arrival": 0,
            "fallback_ranks": [], "wire_model": {L: "trunk" for L in layers}, "trunk": []}
    routes = {L: [] for L in layers}
    if not rows:
        return routes, set(), info
    sp = _Space(layout, params, plan.cut_xy)
    pad_xy = {int(ch.pad_num): np.asarray(ch.hops[0].a_xy, dtype=float) for ch in rows.values()}
    stubs = getattr(plan, "pad_stubs", {}) or {}
    pxs = [float(q[0]) for q in pad_xy.values()]
    pad_mid = float(np.mean(pxs))
    y_exit_max = sp.region.bounds[1] + 2.5 * float(params.pitch_y_mm)
    width = sp.region.bounds[2] - sp.region.bounds[0]
    xs, yb = _bottom(sp, min(pxs) - width, max(pxs) + width)
    # fixed board: the fans stay on the board as built (outline, tail and the
    # fill at the tail's shoulders — what the escape check measures against),
    # inside the edge clearance
    board_in = None
    if not params.auto_expand_board:
        board = layout.board_poly
        if board is None:
            parts = [q for q in (layout.outline_poly,
                                 layout.cable_poly.buffer(0.01, join_style=2) if layout.cable_poly is not None else None)
                     if q is not None and not q.is_empty]
            board = unary_union(parts) if parts else None
        if board is not None and not board.is_empty:
            board_in = board.buffer(-(params.trace_w_mm / 2 + params.board_edge_clear_mm) + 1e-3)
    lift = sp.a + _M
    # where a wire may leave the field: with OT_TRUNK_WINDOW=1 on a fixed
    # board, only where a fan can still reach the pads — a straight line from
    # the field's bottom there to the nearest pad stays on the board, no
    # flatter than the fans' steepest slant (60 degrees from vertical);
    # otherwise anywhere along the bottom edge
    window = (-np.inf, np.inf)
    if os.environ.get("OT_TRUNK_WINDOW", "0") == "1" and board_in is not None and np.isfinite(yb).any():
        from shapely.prepared import prep
        bp = prep(board_in)
        y_pad = max(float(q[1]) for q in pad_xy.values()) + 1.0
        p_lo, p_hi = min(pxs), max(pxs)
        fin = np.isfinite(yb)
        reach = np.zeros(len(xs), dtype=bool)
        for k in np.flatnonzero(fin)[::4]:
            x0, y0 = float(xs[k]), float(yb[k]) - lift
            xp = min(max(x0, p_lo), p_hi)
            reach[k] = (y0 > y_pad and abs(x0 - xp) <= (y0 - y_pad) * np.tan(np.radians(60))
                        and bp.covers(LineString([(x0, y0), (xp, y_pad)])))
        ok = np.flatnonzero(reach)
        if len(ok):
            # the run of reachable samples nearest the pads' middle
            runs = np.split(ok, np.flatnonzero(np.diff(ok) > 4) + 1)
            run = min(runs, key=lambda q: min(abs(xs[q[0]] - pad_mid), abs(xs[q[-1]] - pad_mid))
                      if not xs[q[0]] <= pad_mid <= xs[q[-1]] else -1.0)
            window = (float(xs[run[0]]) - 0.2, float(xs[run[-1]]) + 0.2)
        else:
            window = (pad_mid, pad_mid)

    def finish(pre, placed, why, pads):
        """fans: the pads to the exits, both left to right; a wire whose fan
        or path fails a check is dropped and the rest matched again.
        pads: {net: pad} — the set handed out (in x order) to the exits"""
        placed = dict(placed)
        wires = {}
        for _round in range(len(placed) + 1):
            outs = sorted(placed, key=lambda n: placed[n][-1][0])
            pool = sorted((pads[n] for n in outs), key=lambda p: pad_xy[p][0])
            layer = _Layer(sp, pre)
            wires, bad = {}, set()
            exits = [placed[n][-1] for n in outs]
            starts = [(float(pad_xy[p][0]), float(pad_xy[p][1]) + (0.0 if p in stubs else 1.0)) for p in pool]
            fans = []
            if exits:
                lo = min(min(e[0] for e in exits), min(st[0] for st in starts))
                hi = max(max(e[0] for e in exits), max(st[0] for st in starts))
                m = (xs >= lo - 1e-9) & (xs <= hi + 1e-9) & np.isfinite(yb)
                y_top = (float(np.min(yb[m])) if m.any() else min(e[1] for e in exits)) - lift

                # the angle leaving the most room first; steeper or flatter
                # ones where its fans would leave the board
                def n_bad(fs):
                    return sum(1 for (pnum, fan) in zip(pool, fs)
                               if fan[2][1] < fan[3][1] - 1e-9 or (board_in is not None and not board_in.covers(
                                   LineString([tuple(pad_xy[pnum])] + [tuple(q) for q in fan[::-1]]))))
                for deg in (None, 90, 75, 60, 45, 30, 15):
                    fs = _fans(exits, starts, y_top, sp.s, deg)
                    nb = n_bad(fs)
                    if not fans or nb < fans[0]:
                        fans = (nb, fs)
                    if nb == 0:
                        break
                fans = fans[1]
            for net, pnum, fan in zip(outs, pool, fans):
                up = placed[net][::-1]                      # exit -> tap via
                down = [tuple(pad_xy[pnum])] + [tuple(q) for q in fan[::-1]]
                path = down + [tuple(q) for q in up[1:]]
                reason = None
                if fan[2][1] < fan[3][1] - 1e-9:
                    reason = "no room above the pads for its fan"
                elif board_in is not None and not board_in.covers(LineString(down)):
                    reason = "its fan leaves the board"
                else:
                    chk = layer.ok(net, [])
                    r_in = chk(LineString(up))
                    r_fan = chk(LineString(down[:-1] if len(down) > 2 else down), in_region=False)
                    reason = ("its wire " + r_in) if r_in else (("its fan " + r_fan) if r_fan else None)
                if reason:
                    bad.add(net)
                    why[net] = reason
                    continue
                wires[net] = (pnum, path)
                layer.lay(net, path)
            if not bad:
                break
            for n in bad:
                placed.pop(n, None)
        return wires, why

    own = {n: int(ch.pad_num) for n, ch in rows.items()}
    todo, why_all, shipped, row_y = set(rows), {}, {}, {}
    for L in layers:
        if not todo:
            break
        pre = (preplaced or {}).get(L, [])
        G = _Grid(sp, pre)
        if len(G.SX) == 0:
            why_all.update({n: "no gap between via columns for a trunk" for n in todo})
            continue
        rows_y = {}
        for i, n in enumerate(sp.vnet):
            if n in todo:
                rows_y.setdefault(n, set()).add(int(np.argmin(np.abs(G.Y - sp.vxy[i, 1]))))
        # one via row per row net (the lowest, if a row were ever spread)
        rows_up = sorted(((n, min(rs)) for n, rs in rows_y.items()), key=lambda q: (-G.Y[q[1]], q[0]))
        row_y.update({n: float(G.Y[r]) for n, r in rows_up})

        # every trunk lane (the inner edge of each gap) and both sides; the
        # builds placing the most are finished (fans included); most rows
        # placed wins, then (a later layer) the first layer's direction,
        # then the fewest rows joined inside the bundle (out of order), then
        # the exits nearest the pads (in pitches), then the fewest jogs.
        # A later layer's bundle grows the way the first one did, so its rows
        # come out in the same direction (the pads below stay in row order).
        d_first = next(iter(shipped.values()))[0][0] if shipped else None
        built = []
        for d in (1, -1):
            for g, ks in enumerate(G.gap_slots):
                if not ks:
                    continue
                t = ks[0] if d > 0 else ks[-1]
                for first_inner, clamp in ((True, False), (False, False), (True, True), (False, True)):
                    placed, why, jogs, inner = _steer(G, rows_up, t, d, y_exit_max, first_inner, window, clamp)
                    if not placed:
                        continue
                    ex = sorted(p[-1][0] for p in placed.values())
                    px = sorted(float(pad_xy[own[n]][0]) for n in placed)
                    near = -round(float(np.mean(np.abs(np.subtract(ex, px)))) / float(params.pitch_x_mm))
                    built.append(((len(placed), d == d_first, -inner, near, -jogs), (d, t, first_inner, clamp), placed, why))
        built.sort(key=lambda b: b[0], reverse=True)
        # finished: the best 16 builds with a fixed trunk and the best 8 with
        # a clamped one (each family keeps its own share, so neither crowds
        # out the other's builds that fan out cleanly)
        best, seen = None, set()
        for clamp, quota in ((False, 16), (True, 8)):
            done = 0
            for score, how, placed, why in built:
                if how[3] != clamp:
                    continue
                if done >= quota or (best is not None and score[0] < best[0][0]):
                    break
                sig = tuple(sorted((n, tuple(map(tuple, np.round(p, 4)))) for n, p in placed.items()))
                if sig in seen:
                    continue
                seen.add(sig)
                done += 1
                wires, why = finish(pre, placed, dict(why), own)
                key = (len(wires),) + tuple(score[1:])
                if best is None or key > best[0]:
                    best = (key, how, wires, why, placed)
        if best is None:
            why_all.update({n: "no row can join a bundle" for n in todo})
            continue
        _, how, wires, why, placed = best
        shipped[L] = [how, wires, {n: placed[n] for n in wires}, pre]
        todo -= set(wires)
        why_all.update({n: v for n, v in why.items() if n in todo})
        info["trunk"].append([[G.SX[how[1]], sp.region.bounds[1]], [G.SX[how[1]], sp.region.bounds[3]]])

    # One pad order across the layers: the first layer's wires in the order
    # they leave the field, each other layer's row slotted in where its via
    # row falls in that order; the pads, left to right, in that sequence.
    # Kept only if every layer still places all its wires on those pads.
    if len(shipped) > 1:
        Ls = list(shipped)
        d0 = shipped[Ls[0]][0][0]
        seq = sorted(shipped[Ls[0]][1], key=lambda n: shipped[Ls[0]][2][n][-1][0])
        later = sorted((n for L in Ls[1:] for n in shipped[L][1]), key=lambda n: -d0 * row_y[n])
        for n in later:
            beyond = (lambda m: row_y[m] < row_y[n]) if d0 > 0 else (lambda m: row_y[m] > row_y[n])
            seq.insert(next((i for i, m in enumerate(seq) if beyond(m)), len(seq)), n)
        pads = dict(zip(seq, sorted((own[n] for n in seq), key=lambda p: pad_xy[p][0])))
        redo = {}
        for L in Ls:
            how, wires, placed, pre = shipped[L]
            w2, _ = finish(pre, placed, {}, pads)
            if len(w2) < len(wires):
                redo = None
                break
            redo[L] = w2
        if redo:
            for L in Ls:
                shipped[L][1] = redo[L]

    for L, (how, wires, placed, pre) in shipped.items():
        for net, (pnum, path) in wires.items():
            stub = stubs.get(int(pnum))
            poly = [tuple(q) for q in stub[:-1]] + path if stub else path
            routes[L].append(Route(net_name=net, pad_num=pnum,
                                   anchor_xy=np.asarray(path[-1], dtype=float),
                                   pad_xy=np.asarray(stub[0] if stub else pad_xy[pnum], dtype=float),
                                   polyline_xy=np.asarray(poly, dtype=float)))
            layout.pad_to_net[pnum] = net
    failed = set(todo)
    info.update(rows_failed=sorted(failed), fail_reasons={n: why_all.get(n, "no route") for n in failed},
                rows_bcu_fallback=len(failed), per_layer={L: len(routes[L]) for L in layers})
    return routes, failed, info


# ==========================================================================
# In2: links across the gaps, as nested U's on the lattice
# ==========================================================================

def route_links(layout, params, plan):
    """Returns (routes, nets_done, info) — the route_row_links_in2 contract.

    A row a notch cuts in two is joined by a U: from the end via of one piece
    down a lane (in the gap beside it), across a lane between two via rows
    below the notch's tip, and up a lane to the other piece's end via. Round
    one notch the U's nest like boxes: the row nearest the tip goes first and
    takes the innermost lanes and the highest crossing; each further row goes
    one lane further out and one crossing lower. Straight lines only, in a
    calculated order, every one checked. (A notch opening downward: the same,
    upside down.) A link that finds no free lanes is refused, and its row
    keeps the hug router's B.Cu link."""
    sp = _Space(layout, params, plan.cut_xy)
    X = _cluster(sp.vxy[:, 0], 0.05)
    Y = _cluster(sp.vxy[:, 1], 0.05)
    via_at = {}
    for i, (x, y) in enumerate(sp.vxy):
        via_at[(int(np.argmin(np.abs(X - x))), int(np.argmin(np.abs(Y - y))))] = i
    vlanes = _lanes(X, sp.a, sp.s)            # x positions per gap between via columns
    hlanes = _lanes(Y, sp.a, sp.s)            # y positions per band between via rows
    col_of = lambda i: int(np.argmin(np.abs(X - sp.vxy[i, 0])))
    row_of = lambda i: int(np.argmin(np.abs(Y - sp.vxy[i, 1])))

    links = []
    for net, ch in plan.chains.items():
        if ch.face != "row" or len(ch.segments) < 2:
            continue
        for k in range(1, len(ch.segments)):
            A, B = np.asarray(ch.segments[k - 1]), np.asarray(ch.segments[k])
            if np.mean(sp.vxy[A, 0]) > np.mean(sp.vxy[B, 0]):
                A, B = B, A
            iA = int(A[np.argmax(sp.vxy[A, 0])])
            iB = int(B[np.argmin(sp.vxy[B, 0])])
            cA, cB, r = col_of(iA), col_of(iB), row_of(iA)
            # where the field runs unbroken from piece to piece: the notch's tip
            def whole(rr):
                return all((c, rr) in via_at for c in range(cA, cB + 1))
            down = next((r - rr for rr in range(r - 1, -1, -1) if whole(rr)), None)
            up = next((rr - r for rr in range(r + 1, len(Y)) if whole(rr)), None)
            if down is None and up is None:
                links.append({"net": net, "key": f"{net}:link{k}", "bad": "its pieces never meet below or above"})
                continue
            go_down = up is None or (down is not None and down <= up)
            d = down if go_down else up
            links.append({"net": net, "key": f"{net}:link{k}", "iA": iA, "iB": iB, "cA": cA, "cB": cB,
                          "r": r, "down": go_down, "tip": r - d if go_down else r + d, "depth": d,
                          "pad_xy": ch.hops[0].a_xy, "pad_num": ch.pad_num})
    if not links:
        return [], set(), {"links": 0}
    why = {L["key"]: L["bad"] for L in links if "bad" in L}
    todo = sorted((L for L in links if "bad" not in L), key=lambda L: (L["depth"], L["cB"] - L["cA"], L["key"]))
    layer = _Layer(sp)
    paths = {}
    for L in todo:
        net, r, yr = L["net"], L["r"], Y[L["r"]]
        va, vb = tuple(sp.vxy[L["iA"]]), tuple(sp.vxy[L["iB"]])
        # crossings: from the band just past the tip outward, as far as the
        # field goes; risers: from beside each end via outward, as far as the
        # field goes (nearest first, so the U's nest; the checks keep each
        # one inside the field)
        if L["down"]:
            bands = [b for b in range(L["tip"], -1, -1) if b < len(hlanes)]
            ys = [y for b in bands for y in sorted(hlanes[b], reverse=True)]
        else:
            bands = [b for b in range(L["tip"] - 1, len(hlanes)) if b >= 0]
            ys = [y for b in bands for y in sorted(hlanes[b])]
        xa = [x for g in range(L["cA"], -1, -1) if g < len(vlanes) for x in sorted(vlanes[g], reverse=True)]
        xb = [x for g in range(L["cB"] - 1, len(vlanes)) if g >= 0 for x in sorted(vlanes[g])]
        chk = layer.ok(net, [])
        done = None
        for yb in ys:
            ga = next((x for x in xa if chk(LineString([va, (x, yr), (x, yb)])) is None), None)
            if ga is None:
                continue
            gb = next((x for x in xb if x > ga + sp.s and chk(LineString([vb, (x, yr), (x, yb)])) is None), None)
            if gb is None:
                continue
            path = [va, (ga, yr), (ga, yb), (gb, yb), (gb, yr), vb]
            if chk(LineString(path)) is None:
                done = path
                break
        if done is None:
            why[L["key"]] = "no free lanes round its notch"
            continue
        paths[L["key"]] = (net, done)
        layer.lay(net, done)

    by_net = {}
    for L in links:
        by_net.setdefault(L["net"], []).append(L["key"])
    nets_done = {n for n, ks in by_net.items() if all(k in paths for k in ks)}
    info_links = {L["key"]: L for L in links}
    routes = [Route(net_name=net, pad_num=info_links[k]["pad_num"],
                    anchor_xy=np.asarray(p[-1], dtype=float),
                    pad_xy=np.asarray(info_links[k]["pad_xy"], dtype=float),
                    polyline_xy=np.asarray(p, dtype=float))
              for k, (net, p) in paths.items() if net in nets_done]
    info = {"links": len(links), "unplaced_links": len(links) - len(paths), "wire_model": ["lanes"],
            "why": {k.split(":")[0]: v for k, v in why.items()},
            "nets_in2": len(nets_done), "nets_fallback_bcu": len(by_net) - len(nets_done)}
    return routes, nets_done, info
