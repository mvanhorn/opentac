"""Inner-layer wiring for 4-layer boards: the straight-line model (PROTOTYPE).

Same job and the same sweep as inner_engine.py (rows top down, wires keep
their left-to-right order, every row's choices made jointly and exactly),
with a different wire model. inner_engine draws a wire as lanes between via
columns joined by sideways steps on tracks between via rows, so a bundle
crossing the field diagonally comes out as a staircase. Here a wire runs on
straight lines:

  VERTICAL  down a lane between two via columns;
  DIAGONAL  along the via lattice's diagonal, one via column per via row:
            the steepest line on which several wires fit side by side
            (flatter lattice lines hold one wire per channel and splay a
            bundle; a 2.08 x 2.0 mm grid takes 7 wires per lane, 4 per
            diagonal channel, 1 per knight's line).

Going down from one via row to the next, a wire keeps its line or switches
to the neighbouring kind, and the BEND sits where the two lines meet --
anywhere in the gap, so a bundle turning together nests its bends by
itself and a wire right beside a via can still turn towards it. Two more
moves complete the set: a JOG (vertical, a 45-degree shift, vertical, inside
one gap; the wire in front of the move jogs higher, so a bundle can shift
over together) and the START of a new wire from its via (45 degrees into a
lane, or one straight line at any angle to its first crossing).

Each move is checked exactly (segments `a` from other nets' vias, inside
F, a lane from the wires beside it); what a wire pays: B0 per bend plus the
pull towards its pad (escapes) or its nesting slot (bridges). Everything
returned is checked again by inner_engine.verify in route_inner.
"""
import heapq
import math

import numpy as np
from shapely.geometry import LineString

EPS = 1e-6


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def pt_seg(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _orient(px, py, qx, qy, rx, ry):
    return (qx - px) * (ry - py) - (qy - py) * (rx - px)


def seg_seg(a0x, a0y, a1x, a1y, b0x, b0y, b1x, b1y):
    """distance between two segments (0 where they cross)"""
    d = np.minimum(np.minimum(pt_seg(a0x, a0y, b0x, b0y, b1x, b1y), pt_seg(a1x, a1y, b0x, b0y, b1x, b1y)),
                   np.minimum(pt_seg(b0x, b0y, a0x, a0y, a1x, a1y), pt_seg(b1x, b1y, a0x, a0y, a1x, a1y)))
    cross = ((_orient(a0x, a0y, a1x, a1y, b0x, b0y) * _orient(a0x, a0y, a1x, a1y, b1x, b1y) < 0)
             & (_orient(b0x, b0y, b1x, b1y, a0x, a0y) * _orient(b0x, b0y, b1x, b1y, a1x, a1y) < 0))
    return np.where(cross, 0.0, d)


def intervals(F, y):
    cut = F.intersection(LineString([(-1e5, y), (1e5, y)]))
    out = []
    for c in getattr(cut, "geoms", [cut]):
        if c.geom_type == "LineString" and not c.is_empty:
            xs = [p[0] for p in c.coords]
            out.append((min(xs), max(xs)))
    return sorted(out)


def inside(iv, x, m=1e-3):
    """x (array) inside one of the intervals iv, m clear of its ends"""
    x = np.asarray(x, dtype=float)
    if not iv:
        return np.zeros(x.shape, bool)
    u = np.array([p for p, _ in iv]) + m
    v = np.array([q for _, q in iv]) - m
    k = np.searchsorted(u, x, side="right") - 1
    kk = np.clip(k, 0, len(iv) - 1)
    return (k >= 0) & (x <= v[kk] + 1e-12)


class Grid:
    """The via lattice: via rows Yd (top down), via columns X, the crossing
    positions on every row (rc: a grid of step q anchored `a` from both vias
    of each gap between via columns, inside F), and the move checks."""

    def __init__(self, X, Yr, via_xy, via_net, F, a, s, pitch, q=0.1, tol=0.05, jog=0.6):
        X = np.asarray(sorted(X), dtype=float)
        if pitch:
            X = np.concatenate([[X[0] - pitch], X, [X[-1] + pitch]])
        self.F, self.a, self.s, self.X, self.jog = F, a, s, X, jog
        self.Yd = [float(y) for y in sorted(Yr, reverse=True)]
        n = len(self.Yd)
        self.px = float(np.median(np.diff(X)))
        self.py = float(np.median(-np.diff(self.Yd))) if n > 1 else self.px
        xs = []
        for k in range(len(X) - 1):
            lo, hi = X[k] + a, X[k + 1] - a
            if hi >= lo - 1e-9:
                xs += [np.arange(lo, hi + 1e-9, q), np.arange(hi, lo - 1e-9, -q)]
        self.xs = np.unique(np.round(np.concatenate(xs), 6)) if xs else np.zeros(0)
        self.rc = [self.xs[inside(intervals(F, y), self.xs)] for y in self.Yd]
        via_xy = np.asarray(via_xy, dtype=float)
        nets = np.asarray(via_net, dtype=object)
        self.vx, self.vn = [], []
        for y in self.Yd:
            m = np.abs(via_xy[:, 1] - y) < tol
            o = np.argsort(via_xy[m, 0])
            self.vx.append(via_xy[m, 0][o])
            self.vn.append(nets[m][o])
        # F's outline, 1e-3 inside (the margin crossings keep too): a path
        # that starts inside F and crosses none of these edges stays inside
        Fs = F.buffer(-1e-3, join_style=2).simplify(1e-4)
        E = []
        for gg in getattr(Fs, "geoms", [Fs]):
            for ring in [gg.exterior] + list(gg.interiors):
                c = np.asarray(ring.coords)
                E.append(np.column_stack([c[:-1], c[1:]]))
        self.edges = np.concatenate(E) if E else np.zeros((0, 4))
        self._er = {}

    def lam(self, r):
        """lateral run of the diagonal per unit height, in the gap above row r"""
        return self.px / (self.Yd[r - 1] - self.Yd[r])

    def check(self, P, r, net=None):
        """P: (K, 4, 2) paths in the gap above row r. True where every
        segment keeps `a` from the vias of rows r-1 and r not on `net` and
        the path stays inside F."""
        P = np.asarray(P, float)
        K, npts, _ = P.shape
        ok = np.ones(K, bool)
        if not K:
            return ok
        for k in range(npts - 1):
            x0, y0, x1, y1 = P[:, k, 0], P[:, k, 1], P[:, k + 1, 0], P[:, k + 1, 1]
            lo, hi = np.minimum(x0, x1) - self.a, np.maximum(x0, x1) + self.a
            for rr in (r - 1, r):
                if rr < 0 or rr >= len(self.Yd) or not len(self.vx[rr]):
                    continue
                vx, vn, vy = self.vx[rr], self.vn[rr], self.Yd[rr]
                i0 = np.searchsorted(vx, lo, side="left")
                i1 = np.searchsorted(vx, hi, side="right")
                for j in range(int(np.max(i1 - i0, initial=0))):
                    idx = i0 + j
                    live = ok & (idx < i1)
                    if not live.any():
                        break
                    ii = np.clip(idx, 0, len(vx) - 1)
                    blk = live & (pt_seg(vx[ii], vy, x0, y0, x1, y1) < self.a - EPS)
                    if net is not None:
                        blk &= vn[ii] != net
                    ok &= ~blk
        if not ok.any() or not len(self.edges):
            return ok
        ylo, yhi = float(P[:, :, 1].min()), float(P[:, :, 1].max())
        key = (round(ylo, 4), round(yhi, 4))
        if key not in self._er:
            e = self.edges
            m = (np.minimum(e[:, 1], e[:, 3]) <= yhi + 1e-9) & (np.maximum(e[:, 1], e[:, 3]) >= ylo - 1e-9)
            self._er[key] = e[m]
        e = self._er[key]
        if not len(e):
            return ok
        ex0, ey0, ex1, ey1 = (e[None, :, i] for i in range(4))
        for k in range(npts - 1):
            idx = np.nonzero(ok)[0]
            if not len(idx):
                break
            x0, y0 = P[idx, k, 0][:, None], P[idx, k, 1][:, None]
            x1, y1 = P[idx, k + 1, 0][:, None], P[idx, k + 1, 1][:, None]
            deg = (np.abs(x1 - x0) + np.abs(y1 - y0))[:, 0] < 1e-12
            o1 = (x1 - x0) * (ey0 - y0) - (y1 - y0) * (ex0 - x0)
            o2 = (x1 - x0) * (ey1 - y0) - (y1 - y0) * (ex1 - x0)
            o3 = (ex1 - ex0) * (y0 - ey0) - (ey1 - ey0) * (x0 - ex0)
            o4 = (ex1 - ex0) * (y1 - ey0) - (ey1 - ey0) * (x1 - ex0)
            hit = ((o1 * o2 <= 0) & (o3 * o4 <= 0)).any(axis=1) & ~deg
            ok[idx[hit]] = False
        return ok


# --------------------------------------------------------------------------
# moves through the gap above row r. Paths are (K, 4, 2), padded by
# repeating their last point.
# --------------------------------------------------------------------------

def _P(*pts):
    """pts: up to 4 (x, y) pairs of arrays/scalars -> (K, 4, 2)"""
    pts = list(pts) + [pts[-1]] * (4 - len(pts))
    sizes = [np.size(c) for p in pts for c in p if np.ndim(c) > 0]
    K = (0 if min(sizes) == 0 else max(sizes)) if sizes else 1
    return np.stack([np.column_stack([np.broadcast_to(np.asarray(x, float), (K,)),
                                      np.broadcast_to(np.asarray(y, float), (K,))]) for x, y in pts], axis=1)


def _turn_on(x0, x, y0, y1, lam):
    """vertical down to where the diagonal through (x, y1) passes, then along it"""
    yb = np.minimum(y1 + np.abs(x - x0) / lam, y0)
    return _P((x0, y0), (x0, yb), (x, y1))


def _turn_off(x0, x, y0, y1, lam):
    """along the diagonal from (x0, y0) to the lane x, then down it (also the
    45-degree stub out of a via)"""
    return _P((x0, y0), (x, y0 - np.abs(x - x0) / lam), (x, y1))


def _jog(G, x0, x, y0, y1, lam):
    """vertical, a 45-degree shift, vertical. Wires jogging the same way side
    by side need the one in front (furthest along the move) to jog higher,
    or their shifts run closer than a lane: the jog's height rises with the
    wire's place in its gap between via columns, towards the side it moves."""
    x0 = np.broadcast_to(np.asarray(x0, float), np.shape(x))
    hh = np.abs(x - x0) / lam / 2
    k = np.clip(np.searchsorted(G.X, x0) - 1, 0, len(G.X) - 2)
    off = (x0 - G.X[k]) / (G.X[k + 1] - G.X[k]) - 0.5
    c = 0.5 * (y0 + y1) + np.sign(x - x0) * G.jog * off * (y0 - y1)
    c = np.clip(c, y1 + hh + 0.05, y0 - hh - 0.05)
    return _P((x0, y0), (x0, c + hh), (x, c - hh), (x, y1))


def moves(G, r, x0, m0, xs):
    """a wire at (x0, Yd[r-1]) heading m0 (0 vertical, +-1 diagonal, "stub"
    if x0 is its via and it starts here): its moves to the crossings xs of
    row r. Returns [(paths, x, m, bends), ...]."""
    y0, y1 = G.Yd[r - 1], G.Yd[r]
    lam = G.lam(r)
    xs = np.asarray(xs, float)
    out = []
    if m0 == "stub":
        d = np.abs(xs - x0)
        x = xs[(d >= G.a - 1e-9) & (d < G.px - 1e-9)]
        out.append((_turn_off(x0, x, y0, y1, lam), x, np.zeros(len(x)), np.ones(len(x))))
        x = xs[d < G.px - 1e-9]
        out.append((_P((x0, y0), (x, y1)), x, np.zeros(len(x)), np.full(len(x), 1.5)))
        return out
    if m0 == 0:
        out.append((_P((x0, y0), (x0, y1)), np.array([x0]), np.zeros(1), np.zeros(1)))
        for m1 in (-1, 1):
            # onto the diagonal; it has to turn back again, so it is charged
            # both bends now and never looks cheaper than a jog
            d = (xs - x0) * m1
            x = xs[(d > 1e-9) & (d <= G.px + 1e-9)]
            out.append((_turn_on(x0, x, y0, y1, lam), x, np.full(len(x), float(m1)), np.full(len(x), 2.0)))
        d = np.abs(xs - x0)
        x = xs[(d > 1e-9) & (d < G.px - 1e-9)]
        out.append((_jog(G, x0, x, y0, y1, lam), x, np.zeros(len(x)), np.full(len(x), 2.0)))
        return out
    xc = x0 + m0 * G.px
    out.append((_P((x0, y0), (xc, y1)), np.array([xc]), np.array([float(m0)]), np.zeros(1)))
    # off the diagonal into a lane (or straight down from the crossing)
    d = (xs - x0) * m0
    x = xs[(d > -1e-9) & (d < G.px - 1e-9)]
    if not np.any(np.abs(x - x0) < 1e-9):
        x = np.concatenate([[x0], x])
    out.append((_turn_off(x0, x, y0, y1, lam), x, np.zeros(len(x)), np.zeros(len(x))))
    return out


# --------------------------------------------------------------------------
# one gap: every wire's move, jointly
# --------------------------------------------------------------------------

def _ranges(P, s, nb=6):
    """per path, the x range it covers in each of nb horizontal slabs of the
    gap (each slab grown by s up and down): (K, nb, 2)"""
    y0, y1 = float(P[:, :, 1].max()), float(P[:, :, 1].min())
    edges = np.linspace(y1, y0, nb + 1)
    lo_b, hi_b = edges[:-1] - s, edges[1:] + s
    K = len(P)
    out = np.empty((K, nb, 2))
    out[:, :, 0], out[:, :, 1] = np.inf, -np.inf
    for k in range(P.shape[1] - 1):
        xa, ya, xb, yb = P[:, k, 0][:, None], P[:, k, 1][:, None], P[:, k + 1, 0][:, None], P[:, k + 1, 1][:, None]
        dy = yb - ya
        flat = np.abs(dy) < 1e-12
        t1 = np.where(flat, 0.0, (lo_b[None, :] - ya) / np.where(flat, 1.0, dy))
        t2 = np.where(flat, 1.0, (hi_b[None, :] - ya) / np.where(flat, 1.0, dy))
        ta, tb = np.clip(np.minimum(t1, t2), 0, 1), np.clip(np.maximum(t1, t2), 0, 1)
        hit = np.where(flat, (ya >= lo_b[None, :]) & (ya <= hi_b[None, :]), np.minimum(t1, t2) <= np.maximum(t1, t2))
        hit &= ta <= tb + 1e-12
        if not flat.all():
            hit &= ~(~flat & ((np.maximum(t1, t2) < 0) | (np.minimum(t1, t2) > 1)))
        xa_, xb_ = xa + (xb - xa) * ta, xa + (xb - xa) * tb
        out[:, :, 0] = np.where(hit, np.minimum(out[:, :, 0], np.minimum(xa_, xb_)), out[:, :, 0])
        out[:, :, 1] = np.where(hit, np.maximum(out[:, :, 1], np.maximum(xa_, xb_)), out[:, :, 1])
    return out


def _far(PA, PB, s, todo=None):
    """(KA, KB): paths i of PA and j of PB keep at least s apart (only the
    pairs in `todo` are tested; the rest come back False)"""
    res = np.zeros((len(PA), len(PB)), bool)
    if todo is None:
        todo = np.ones(res.shape, bool)
    ia, ib = np.nonzero(todo)
    if len(ia):
        A, B = PA[ia], PB[ib]
        d = np.full(len(ia), np.inf)
        # (a segment of no length beyond the first repeats a point the one
        # before already ends on; the first may be the whole path: a point)
        for i in range(3):
            if i and np.all(np.abs(A[:, i] - A[:, i + 1]).sum(axis=1) < 1e-12):
                continue
            for j in range(3):
                if j and np.all(np.abs(B[:, j] - B[:, j + 1]).sum(axis=1) < 1e-12):
                    continue
                d = np.minimum(d, seg_seg(A[:, i, 0], A[:, i, 1], A[:, i + 1, 0], A[:, i + 1, 1],
                                          B[:, j, 0], B[:, j, 1], B[:, j + 1, 0], B[:, j + 1, 1]))
        res[ia, ib] = d >= s - EPS
    return res


def _apart(A, B, s):
    """(KA, KB) for two neighbours, A on the left: their paths keep s apart.
    Cheap tests first: crossings closer than s can never be; paths a lane
    apart in every (overlapping) slab always are; the rest exactly."""
    for E in (A, B):
        if "rng" not in E:
            E["rng"] = _ranges(E["P"], s)
    ra, rb = A["rng"], B["rng"]
    gap = rb[None, :, :, 0] - ra[:, None, :, 1]          # (KA, KB, nb): B's min - A's max per slab
    sure = np.all(gap >= s - EPS, axis=2)
    never = (B["x"][None, :] - A["x"][:, None]) < s - EPS
    res = _far(A["P"], B["P"], s, todo=~sure & ~never)
    return res | sure


def compat(A, B, s):
    """(KA, KB): candidate i of A (on the left) and j of B fit side by side"""
    ka, kb = A["kind"], B["kind"]
    if ka == "wall" or kb == "wall":
        return _far(A["P"], B["P"], s)
    if ka == "bar" or kb == "bar":
        if ka == "bar" and kb == "bar":
            return np.ones((len(A["x"]), len(B["x"])), bool)
        if ka == "bar":
            return np.broadcast_to((B["P"][:, :, 0].min(axis=1) >= A.get("hi", A["x"][0]) + s - EPS)[None, :],
                                   (1, len(B["x"]))).copy()
        return np.broadcast_to((A["P"][:, :, 0].max(axis=1) <= B.get("lo", B["x"][0]) - s + EPS)[:, None],
                               (len(A["x"]), 1)).copy()
    xa, xb = A["x"][:, None], B["x"][None, :]
    if A["net"] == B["net"]:
        return xb >= xa - 1e-9
    return (xb > xa + 1e-9) & _apart(A, B, s)


def place_row(elems, s):
    """one candidate per element, every neighbour pair compatible, least
    total cost (exact). Every path spans the whole gap, so neighbours are
    all that need checking. -> chosen indices, or None"""
    if not elems:
        return []
    best = elems[0]["cost"].astype(float)
    backs = []
    for i in range(1, len(elems)):
        A, B = elems[i - 1], elems[i]
        tot = np.where(compat(A, B, s), best[:, None], np.inf)
        back = np.argmin(tot, axis=0)
        best = B["cost"] + tot[back, np.arange(len(B["x"]))]
        if not len(best) or not np.isfinite(best).any():
            return None
        backs.append(back)
    if not np.isfinite(best).any():
        return None
    k = int(np.argmin(best))
    picks = [k]
    for back in reversed(backs):
        k = int(back[k])
        picks.append(k)
    return picks[::-1]


def _cat(parts):
    return tuple(np.concatenate([p[i] for p in parts]) if parts else np.zeros((0, 4, 2) if i == 0 else 0)
                 for i in range(4))


# --------------------------------------------------------------------------
# escapes: each row to the connector
# --------------------------------------------------------------------------

def sweep_escapes(rows, G, exit_y, exit_ok, pad_xs, B0=0.5, w=1.0):
    """rows: [{"net", "y", "vias"}]. Same contract as inner_engine.sweep_escapes:
    ({net: polyline tap -> ... -> (x, exit)}, {net: reason}).

    The sweep stands on via row r with every wire's crossing known and
    places all their moves through the gap below, jointly: the wires
    passing on, the rows starting at r (from one of their vias) and the
    wires that leave (F ends under them: straight down). Wires head for
    their pads in order, as in the old sweep: a move costs B0 per bend plus
    w * how far its next crossing is from the wire's pad.

    The funnel (computed bottom up) is the set of lanes from which a wire
    going straight down, jogging as it needs, still reaches an exit; a wire
    may only cross a row in it, or on a diagonal that can still turn into
    it at the next row."""
    s = G.s
    pads = sorted(pad_xs)
    Yd = G.Yd
    n = len(Yd)
    ey = {}

    def exy(x):
        k = round(float(x), 6)
        if k not in ey:
            ey[k] = exit_y(k)
        return ey[k]

    def ends_under(x, r):
        ex = exy(x)
        return ex is not None and (r + 1 >= n or ex > Yd[r + 1] - 1e-6)

    def pairs(a_, b_, lo, hi):
        """index pairs (i, j): b_[j] (sorted) within [a_[i] + lo, a_[i] + hi)"""
        j0 = np.searchsorted(b_, a_ + lo - 1e-9)
        j1 = np.searchsorted(b_, a_ + hi - 1e-9)
        cnt = np.maximum(j1 - j0, 0)
        ii = np.repeat(np.arange(len(a_)), cnt)
        jj = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(j0, cnt)
        return ii, jj

    funnel = [None] * n
    for r in range(n - 1, -1, -1):
        c = G.rc[r]
        f = np.zeros(len(c), bool)
        e = np.array([ends_under(x, r) for x in c], bool)
        for i in np.nonzero(e)[0]:
            f[i] = bool(exit_ok(c[i]))
        if r + 1 < n:
            nxt = G.rc[r + 1][funnel[r + 1]]
            go = np.nonzero(~e)[0]
            ii, jj = pairs(c[go], nxt, -G.px + 2e-9, G.px)
            if len(ii):
                x0, x = c[go][ii], nxt[jj]
                y0, y1, lam = Yd[r], Yd[r + 1], G.lam(r + 1)
                P = np.where((np.abs(x - x0) < 1e-9)[:, None, None], _P((x0, y0), (x0, y1)),
                             _jog(G, x0, x, y0, y1, lam))
                ok = G.check(P, r + 1)
                f[go] = np.bincount(ii[ok], minlength=len(go)) > 0
        funnel[r] = f

    def in_funnel(r, x):
        c = G.rc[r]
        if not len(c):
            return np.zeros(len(x), bool)
        i = np.clip(np.searchsorted(c, x - 1e-6), 0, len(c) - 1)
        return (np.abs(c[i] - x) < 1e-6) & funnel[r][i]

    def can_turn_down(r, x, m, net):
        """diagonal wires crossing row r at x can still turn into a funnel
        lane at row r+1"""
        out = np.zeros(len(x), bool)
        if r + 1 >= n or not len(x):
            return out
        nxt = G.rc[r + 1][funnel[r + 1]]
        for sgn in (-1.0, 1.0):
            sel = np.nonzero(m == sgn)[0]
            if not len(sel):
                continue
            lo, hi = (0.0, G.px) if sgn > 0 else (-G.px + 2e-9, 2e-9)
            ii, jj = pairs(x[sel], nxt, lo, hi)
            if not len(ii):
                continue
            P = _turn_off(x[sel][ii], nxt[jj], Yd[r], Yd[r + 1], G.lam(r + 1))
            ok = G.check(P, r + 1, net)
            out[sel] = np.bincount(ii[ok], minlength=len(sel)) > 0
        return out

    def options(r, x0, m0, net):
        """a wire at (x0, row r) heading m0: its moves to row r+1 that keep a
        way out. -> (P, x, m, bends)"""
        xs = G.rc[r + 1]
        j0, j1 = np.searchsorted(xs, [x0 - G.px - 1e-9, x0 + G.px + 1e-9])
        P, x, m, b = _cat(moves(G, r + 1, x0, m0, xs[j0:j1]))
        ok = G.check(P, r + 1, net)
        v = m == 0
        ok &= np.where(v, in_funnel(r + 1, x), True)
        dg = np.nonzero(~v & ok)[0]
        if len(dg):
            ok[dg] = can_turn_down(r + 1, x[dg], m[dg], net)
        return P[ok], x[ok], m[ok], b[ok]

    def exit_path(r, x0, y0, net, tap=None, pad=None, keep=()):
        """leave the field from lane x0 below row r (from a tap: stub first).
        A wire whose pad is less than half a via column away angles its drop
        onto the pad's x, so the run to the connector below is straight --
        if the drop keeps a lane from the paths in `keep` ((K, 4, 2): the
        other wires where they stand, and drops already placed)."""
        if pad is not None and tap is None and 1e-6 < abs(pad - x0) < G.px / 2:
            ex = exy(pad)
            if ex is not None and exit_ok(pad) and ex < y0 - 1e-6:
                P = _P((x0, y0), (pad, ex))
                Pt = _P((x0, y0), (x0 + (pad - x0) * (y0 - ex - 2e-3) / (y0 - ex), ex + 2e-3))
                if G.check(Pt, r + 1, net)[0] and (not len(keep) or _far(P, keep, s).all()):
                    return P
        ex = exy(x0)
        if ex is None or not exit_ok(x0):
            return None
        lam = G.px / G.py
        pts = [(x0, y0)] if tap is None else [(tap, y0), (x0, y0 - abs(x0 - tap) / lam)]
        if ex > pts[-1][1] + 1e-9:
            return None
        P = _P(*pts, (x0, ex))
        # the exit is on F's edge by definition: test the drop short of it
        Pt = P.copy()
        Pt[0, len(pts):, 1] = ex + 2e-3
        return P if G.check(Pt, r + 1, net)[0] else None

    def targets(ws):
        n_ = len(ws)
        if n_ == 0 or n_ > len(pads):
            return [None] * n_
        cx = [q["x"] for q in ws]
        b = min(range(len(pads) - n_ + 1), key=lambda s0: sum(abs(cx[i] - pads[s0 + i]) for i in range(n_)))
        return [pads[b + i] for i in range(n_)]

    by_y = {}
    for rw in rows:
        by_y.setdefault(round(rw["y"], 4), []).append(rw)
    order, why = [], {}
    for r, y in enumerate(Yd):
        # wires that leave below this row: F ends under them
        pad_of = dict(zip((id(q) for q in order), targets(order)))
        drops = []
        for q in order:
            if not q.get("out") and q["m"] == 0 and ends_under(q["x"], r):
                # the others where they stand -- or, if they leave here too,
                # all the way down their own drop -- and the drops placed
                keep = [_P((o["x"], y), (o["x"], exy(o["x"]))) if (o["m"] == 0 and ends_under(o["x"], r))
                        else _P((o["x"], y)) for o in order if o is not q and not o.get("out")] + drops
                P = exit_path(r, q["x"], y, q["net"], pad=pad_of.get(id(q)),
                              keep=np.concatenate(keep) if keep else ())
                if P is not None:
                    drops.append(P)
                if P is None:
                    why[q["net"]] = "it reaches the bottom of the taxels where no fan can start"
                    q["dead"] = True
                else:
                    q["pts"].append(tuple(map(float, P[0, -1])))
                    q["out"], q["end"] = True, float(P[0, -1, 1])
                    q["span"] = (min(q["x"], float(P[0, -1, 0])), max(q["x"], float(P[0, -1, 0])))
                    q["x"] = float(P[0, -1, 0])
        if r + 1 >= n:
            # the last row: a wire that has not left by now never will
            for q in order:
                if not q.get("out"):
                    why[q["net"]] = "it reaches the bottom of the taxels where no fan can start"
                    q["dead"] = True
        order = [q for q in order if not q.get("dead")]
        if r + 1 < n:
            for q in order:
                if not q.get("out"):
                    q["opts"] = options(r, q["x"], q["m"], q["net"])
            stuck = [q for q in order if not q.get("out") and not len(q["opts"][1])]
            for q in stuck:
                why[q["net"]] = "it could not move over in time for a narrowing below"
            order = [q for q in order if q not in stuck]

        news = by_y.get(round(y, 4), [])
        for k_new in range(max(1, len(news))):
            # several row pieces can start at one height (a U's two arms):
            # each joins in turn; after the first, the wires already placed
            # in this gap keep their moves
            new = news[k_new:k_new + 1]
            first = k_new == 0
            # the new row's starts, grouped by where their via falls among
            # the wires already here
            starts = {}
            for t in sorted({float(v) for v in (new[0]["vias"] if new else [])}):
                k = sum(1 for q in order if q["x"] < t)
                cand = []
                if r + 1 < n:
                    xs = G.rc[r + 1][funnel[r + 1]]
                    j0, j1 = np.searchsorted(xs, [t - G.px, t + G.px])
                    P, x, m, b = _cat(moves(G, r + 1, t, "stub", xs[j0:j1]))
                    ok = G.check(P, r + 1, new[0]["net"])
                    if ok.any():
                        cand.append((P[ok], x[ok], m[ok], b[ok]))
                # into a lane F ends under: leave straight away
                for xl in G.rc[r][np.abs(G.rc[r] - t) < G.px]:
                    if abs(xl - t) >= G.a - 1e-9 and ends_under(xl, r):
                        P = exit_path(r, xl, y, new[0]["net"], tap=t)
                        if P is not None:
                            cand.append((P, np.array([xl]), np.zeros(1), np.ones(1)))
                if cand:
                    starts.setdefault(k, []).append(_cat(cand))

            def elem(q, tgt):
                if q.get("out"):
                    # it has left, but it still pays for how far its exit is
                    # from its pad: a row starting on the wrong side would
                    # push every wire that left off its pad
                    lo_, hi_ = q.get("span", (q["x"], q["x"]))
                    return {"kind": "bar", "net": q["net"], "x": np.array([q["x"]]), "lo": lo_, "hi": hi_,
                            "cost": np.array([w * abs(q["x"] - tgt) if tgt is not None else 0.0])}
                P, x, m, b = q["opts"]
                if not first and not q.get("new"):
                    k = q["pick"]
                    P, x, m, b = P[k:k + 1], x[k:k + 1], m[k:k + 1], b[k:k + 1]
                return {"kind": "pass", "net": q["net"], "P": P, "x": x, "m": m,
                        "cost": B0 * b + (w * np.abs(x - tgt) if tgt is not None else 0.0)}

            def plan(with_new):
                found = []
                for k in (sorted(starts) if (with_new and starts) else [None]):
                    ws = list(order)
                    if k is not None:
                        opts = _cat(starts[k])
                        ws = order[:k] + [{"net": new[0]["net"], "x": float(np.median(opts[1])), "m": "stub",
                                           "opts": opts, "new": True}] + order[k:]
                    tg = targets(ws)
                    els = [elem(q, t) for q, t in zip(ws, tg)]
                    pick = place_row(els, s)
                    if pick is not None:
                        found.append((sum(float(e["cost"][j]) for e, j in zip(els, pick)), ws, els, pick))
                found.sort(key=lambda f: f[0])
                return found

            res = plan(True)
            if not res:
                if new:
                    why[new[0]["net"]] = "no room for it on this layer where its row is"
                res = plan(False) if r + 1 < n else [(0, list(order), [], [])]
                while not res and order:
                    # the passing wires do not fit: the outermost one yields
                    # (to the next layer) until they do
                    lw = [q for q in order if not q.get("out")]
                    if not lw:
                        break
                    cx = float(np.mean(pads)) if pads else 0.0
                    drop = max((lw[0], lw[-1]), key=lambda q: abs(q["x"] - cx))
                    why[drop["net"]] = "no room for it on this layer further down"
                    order = [q for q in order if q is not drop]
                    res = plan(False)
                if not res:
                    for q in order:
                        why[q["net"]] = "no room for it on this layer further down"
                    order = []
                    continue
            _, ws, els, pick = res[0]
            for q, e, j in zip(ws, els, pick):
                if e["kind"] != "pass":
                    continue
                if first or q.get("new"):
                    q["pick"] = j
                if q.get("new"):
                    q["pts"] = [tuple(map(float, e["P"][j][0]))]
                    # from now on its moves are the ones it picked from
                    q["opts"] = tuple(o[j:j + 1] for o in q["opts"])
                    q["pick"] = 0
                q["move"] = (e["P"][j], float(e["x"][j]), float(e["m"][j]))
            for q in ws:
                q.pop("new", None)
            order = ws
        # every wire steps through the gap
        nxt = []
        for q in order:
            if q.get("out"):
                nxt.append(q)
                continue
            if "move" not in q:
                why.setdefault(q["net"], "no room for it on this layer further down")
                continue
            P, q["x"], q["m"] = q.pop("move")
            q.pop("pick", None)
            for p in P[1:]:
                p = (float(p[0]), float(p[1]))
                if math.dist(p, q["pts"][-1]) > 1e-9:
                    q["pts"].append(p)
            if r + 1 >= n or P[-1, 1] > Yd[r + 1] + 1e-6:
                # the move ended above the next row: it left the field
                q["out"], q["end"] = True, float(P[-1, 1])
            nxt.append(q)
        order = nxt
    paths = {}
    for q in order:
        if not q.get("out"):
            why.setdefault(q["net"], "it reaches the bottom of the taxels where no fan can start")
            continue
        paths[q["net"]] = straighten(q["pts"])
    for rw in rows:
        if rw["net"] not in paths:
            why.setdefault(rw["net"], "no lane beside any of its vias leads to an exit")
    return paths, why


# --------------------------------------------------------------------------
# bridges: a row's pieces joined across the notches between them
# --------------------------------------------------------------------------

def sweep_bridges(links, G, via_xy, via_dx, pitch, B0=0.5, w=1.0):
    """links: [{"id", "net", "y", "pixA", "pixB"}]. Same contract and the
    same nesting logic as inner_engine.sweep_bridges (petals, strips at the
    notch roots, nesting ranks, each end's slot, learning from conflicts);
    the wires are straight lines. Each link is two half-wires that come down
    from their pieces on lines and meet under the notch on a straight run
    in a gap (the innermost pair highest). Returns ({link id: polyline},
    {link id: reason})."""
    a, lane = G.a, G.s
    ends = []
    for L in links:
        for side, pix in ((+1, L["pixA"]), (-1, L["pixB"])):
            xs = np.sort(np.unique(np.round(via_xy[pix, 0], 6)))
            ends.append({"L": L, "side": side, "net": L["net"], "y": L["y"], "vias": [float(v) for v in xs],
                         "c0": float(xs.min() - via_dx - pitch / 2), "c1": float(xs.max() - via_dx + pitch / 2)})
    # strips, cut at the notch roots (as inner_engine.sweep_bridges)
    gaps = []
    for L in links:
        x0 = float(np.max(via_xy[L["pixA"], 0])) - via_dx + pitch / 2
        x1 = float(np.min(via_xy[L["pixB"], 0])) - via_dx - pitch / 2
        gaps.append((L["y"], min(x0, x1), max(x0, x1), L["id"]))
    notches = []
    for y, g0, g1, lid in sorted(gaps):
        hits = [k for k, nt in enumerate(notches)
                if abs(nt[-1][0] - y) <= pitch * 1.05 + 1e-6 or abs(nt[-1][0] - y) < 1e-6
                if nt[-1][1] < g1 - 1e-6 and g0 < nt[-1][2] - 1e-6]
        if len(hits) == 1:
            notches[hits[0]].append((y, g0, g1, lid))
        elif len(hits) == 0:
            notches.append([(y, g0, g1, lid)])
    roots = sorted(0.5 * (nt[0][1] + nt[0][2]) for nt in notches)

    def strip_of(x):
        return int(np.searchsorted(np.asarray(roots), x))

    nst = len(roots) + 1
    for e in ends:
        e["petal"] = strip_of(max(e["vias"]) if e["side"] > 0 else min(e["vias"]))
    petals = [[e for e in ends if e["petal"] == k] for k in range(nst)]
    lo = [-math.inf] + roots
    hi = roots + [math.inf]
    ext = [((min(e["c0"] for e in g) if g else 0.0), (max(e["c1"] for e in g) if g else 0.0)) for g in petals]
    pet = {(e["L"]["id"], e["side"]): e["petal"] for e in ends}
    pref = {}
    for L in links:
        skipped = pet[(L["id"], -1)] - pet[(L["id"], +1)] - 1
        span = float(np.min(via_xy[L["pixB"], 0]) - np.max(via_xy[L["pixA"], 0]))
        pref[L["id"]] = (skipped, round(span, 2), -L["y"])
    inside_of = {L["id"]: set() for L in links}
    for g in petals:
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
                        inside_of[p_end["L"]["id"]].add(f["L"]["id"])

    def attempt(nest):
        n_in = {k: 0 for k in nest}
        outs = {k: [] for k in nest}
        for o, ins in nest.items():
            for i in ins:
                outs[i].append(o)
                n_in[o] += 1
        heap = [(pref[k], k) for k, c in n_in.items() if c == 0]
        heapq.heapify(heap)
        rank_of = {}
        while heap:
            _, k = heapq.heappop(heap)
            rank_of[k] = len(rank_of)
            for o in outs[k]:
                n_in[o] -= 1
                if n_in[o] == 0:
                    heapq.heappush(heap, (pref[o], o))
        for k in pref:
            if k not in rank_of:
                rank_of[k] = len(rank_of) + 1000
        for e in ends:
            e["rank"] = (rank_of[e["L"]["id"]],)

        def key(e):
            return (0, e["rank"]) if e["side"] < 0 else (1, tuple(-v for v in e["rank"]))

        tgt = {}
        for pi, g in enumerate(petals):
            for side in (-1, +1):
                ranked = sorted((e for e in g if e["side"] == side), key=lambda e: e["rank"])
                edge = (lo[pi] if math.isfinite(lo[pi]) else ext[pi][0]) if side < 0 else \
                    (hi[pi] if math.isfinite(hi[pi]) else ext[pi][1])
                for k, e in enumerate(ranked):
                    t = edge - side * (k + 0.5) * lane
                    for kj, j in enumerate(ranked[:k]):
                        way_in = (max(j["vias"]) if side > 0 else min(j["vias"])) - side * (a + (k - kj - 1) * lane)
                        t = min(t, way_in) if side > 0 else max(t, way_in)
                    tgt[id(e)] = t

        Yd = G.Yd
        n = len(Yd)
        lam = G.px / G.py
        order = [[] for _ in petals]
        wire_of, out, why = {}, {}, {}
        conflict = None

        def start(w_, y):
            """where a wire stands on row y and its direction; a new end at
            its piece's via nearest its partner, heading for it at 45"""
            if w_["x"] is None:
                return (max(w_["e"]["vias"]) if w_["e"]["side"] > 0 else min(w_["e"]["vias"])), w_["e"]["side"]
            return w_["x"], w_["m"]

        for r, y in enumerate(Yd):
            for pi, g in enumerate(petals):
                for e in g:
                    if abs(e["y"] - y) < 1e-3:
                        wn = {"e": e, "net": e["net"], "x": None, "m": None, "pts": []}
                        order[pi].append(wn)
                        wire_of[(e["L"]["id"], e["side"])] = wn
                order[pi].sort(key=lambda w_: key(w_["e"]))
            # meetings in the gap below this row, innermost first: each pair
            # runs on down its lines to a track, and across
            y_lo = Yd[r + 1] + a if r + 1 < n else y - G.py / 2 + 0.05
            tracks = list(np.arange(y - a, y_lo - 1e-9, -lane))
            walls, met = {}, []
            used, progress = 0, True
            while progress and used < len(tracks):
                progress = False
                for pi in range(len(petals)):
                    if not order[pi] or order[pi][-1]["e"]["side"] < 0:
                        continue
                    wa = order[pi][-1]
                    wb = wire_of.get((wa["e"]["L"]["id"], -1))
                    if wb is None:
                        continue
                    qi = wb["e"]["petal"]
                    if qi <= pi or any(order[k] for k in range(pi + 1, qi)) or not order[qi] or order[qi][0] is not wb:
                        continue
                    (xa, ma), (xb, mb) = start(wa, y), start(wb, y)
                    found = None
                    for j in range(used, len(tracks)):
                        h = tracks[j]
                        ha, hb = xa + ma * lam * (y - h), xb + mb * lam * (y - h)
                        if hb - ha < 1e-6:
                            continue
                        P = _P((xa, y), (ha, h), (hb, h), (xb, y))
                        if not G.check(P, r + 1, wa["net"])[0] or any(not _far(P, Q, lane)[0, 0] for Q in met):
                            continue
                        found = (j, P)
                        break
                    if found is None:
                        continue
                    j, P = found
                    met.append(P)
                    pa = wa["pts"] if wa["x"] is not None else [(xa, y)]
                    pb = wb["pts"] if wb["x"] is not None else [(xb, y)]
                    out[wa["e"]["L"]["id"]] = pa + [tuple(P[0, 1]), tuple(P[0, 2])] + pb[::-1]
                    order[pi].pop()
                    order[qi].pop(0)
                    walls.setdefault(pi, []).insert(0, P)
                    walls.setdefault(("L", qi), []).append(P)
                    used = j + 1
                    progress = True
            if r + 1 >= n:
                break
            # every petal's wires through the gap below, jointly (a petal
            # keeps to its strip; the meetings beside it are walls)
            for pi in range(len(petals)):
                ws = order[pi]
                if not ws:
                    continue
                xs = G.rc[r + 1]
                xs = xs[(xs >= lo[pi] + lane / 2) & (xs <= hi[pi] - lane / 2)]
                els = [{"kind": "wall", "P": P, "x": np.array([P[0, 0, 0]]), "cost": np.zeros(1)}
                       for P in walls.get(("L", pi), [])]
                for w_ in ws:
                    tg = tgt[id(w_["e"])]
                    if w_["x"] is None:
                        parts = [(t, p) for t in w_["e"]["vias"] for p in moves(G, r + 1, t, "stub", xs)]
                        taps = np.concatenate([np.full(len(p[1]), t) for t, p in parts])
                        P, x, m, b = _cat([p for _, p in parts])
                        extra = 1e-3 * np.abs(x - taps)
                    else:
                        P, x, m, b = _cat(moves(G, r + 1, w_["x"], w_["m"], xs))
                        extra = 0.0
                    ok = G.check(P, r + 1, w_["net"])
                    els.append({"kind": "pass", "net": w_["net"], "P": P[ok], "x": x[ok], "m": m[ok],
                                "cost": (B0 * b + w * np.abs(x - tg) + extra)[ok]})
                els += [{"kind": "wall", "P": P, "x": np.array([P[0, -1, 0]]), "cost": np.zeros(1)}
                        for P in walls.get(pi, [])]
                pick = place_row(els, lane) if all(len(e["x"]) for e in els) else None
                if pick is None:
                    if conflict is None:
                        conflict = (pi, y, ws)
                    for w_ in ws:
                        why[w_["e"]["L"]["id"]] = "its petal's wires do not fit through this row"
                    order[pi] = []
                    continue
                mv = [(e, k) for e, k in zip(els, pick) if e["kind"] == "pass"]
                for w_, (e, k) in zip(ws, mv):
                    P = e["P"][k]
                    if w_["x"] is None:
                        w_["pts"] = [tuple(map(float, P[0]))]
                    for q in P[1:]:
                        q = (float(q[0]), float(q[1]))
                        if math.dist(q, w_["pts"][-1]) > 1e-9:
                            w_["pts"].append(q)
                    w_["x"], w_["m"] = float(e["x"][k]), float(e["m"][k])
        for L in links:
            if L["id"] not in out:
                why.setdefault(L["id"], "its two ends never met under the notch")
        return {k: straighten(v) for k, v in out.items()}, why, conflict

    learned = {k: set(v) for k, v in inside_of.items()}
    best = None
    for _ in range(len(links) + 1):
        out, why, conflict = attempt(learned)
        if best is None or len(out) > len(best[0]):
            best = (out, why)
        if conflict is None:
            break
        _, _, ws_f = conflict
        added = False
        for n_ in (w_ for w_ in ws_f if w_["x"] is None):
            for f in (w_ for w_ in ws_f if w_["x"] is not None):
                if f["e"]["side"] == n_["e"]["side"] and f["net"] != n_["net"]:
                    o, i = f["e"]["L"]["id"], n_["e"]["L"]["id"]
                    if i not in learned[o] and o not in learned[i]:
                        learned[o].add(i)
                        added = True
        if not added:
            break
    return best


def straighten(pts, eps=1e-6):
    """drop points that sit on the line through their neighbours"""
    p = [tuple(map(float, q)) for q in pts]
    out = [p[0]]
    for i in range(1, len(p) - 1):
        a, b, c = np.asarray(out[-1]), np.asarray(p[i]), np.asarray(p[i + 1])
        u, v = b - a, c - b
        if np.hypot(*u) < eps:
            continue
        if abs(u[0] * v[1] - u[1] * v[0]) > eps * max(1.0, np.hypot(*u) * np.hypot(*v)) or u @ v < 0:
            out.append(tuple(b))
    out.append(p[-1])
    return out
