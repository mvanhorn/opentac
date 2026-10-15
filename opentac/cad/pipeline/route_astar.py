"""A* grid router — free-flowing 45° routing (replaces the greedy dodge router).

For each net (routed one at a time, in a sensible order) we run A* on a grid over
the board, with 8-connectivity so paths move straight or at 45°. Obstacles are the
board exterior + edge clearance, other nets' pixels, other nets' connector pads, and
every previously-routed trace (expanded so the new trace keeps >= G clearance).
Because A* finds a shortest path through free space, traces fan out and stop
staircasing. If a net can't be routed, the nearest blocking pixel is removed and the
layer is retried — same fallback contract as the greedy router.

Public functions mirror route_columns_grid / route_rows_grid so main.py can swap them.
"""

import heapq
import numpy as np
import shapely
from shapely.geometry import Point

from .types import Route, Params
from .geometry import remove_collinear
from .route_columns import _build_col_info
from .route_rows import _build_row_info


def route_columns_astar(layout, outline_xy, params, external_skip_pix=None):
    return _route_layer(layout, outline_xy, params, external_skip_pix, "col")


def route_rows_astar(layout, outline_xy, params, external_skip_pix=None):
    return _route_layer(layout, outline_xy, params, external_skip_pix, "row")


def _route_layer(layout, outline_xy, params, external_skip_pix, which):
    G = params.trace_w_mm + params.clearance_mm
    skip = set(external_skip_pix) if external_skip_pix else set()
    max_attempts = len(layout.placements_xy) + 1

    grid = _Grid(layout, outline_xy, params)

    for _ in range(max_attempts):
        routes, fail_anchor = _route_once(layout, outline_xy, params, skip, which, grid, G)
        if fail_anchor is None:
            return routes, skip
        # remove nearest active OTHER pixel to the failed anchor, retry
        pix = _nearest_active_pixel(layout, skip, fail_anchor)
        if pix is None:
            return routes, skip
        skip.add(pix)
    return routes, skip


def _route_once(layout, outline_xy, params, skip, which, grid, G):
    if which == "col":
        nets, pad_nums, anchor_xy, pad_xy, pixel_idx = _build_col_info(layout, outline_xy, params, skip)
        order = np.argsort(np.abs(anchor_xy[:, 0] - pad_xy[:, 0])) if len(nets) else []
    else:
        nets, pad_nums, anchor_xy, pad_xy, pixel_idx = _build_row_info(layout, outline_xy, params, skip)
        order = np.argsort(anchor_xy[:, 1]) if len(nets) else []

    N = len(nets)
    if N == 0:
        return [], None

    pixel_h = params.pixel_h_mm
    active = set(range(len(layout.placements_xy))) - skip

    pad_pts, pad_nets = grid.pad_positions(layout, outline_xy, params)

    routes = [None] * N
    trace_mask = np.zeros(grid.shape, dtype=bool)

    for oi in range(N):
        i = int(order[oi])
        my_net = nets[i]
        own = pixel_idx[i]

        blocked = grid.base.copy()
        blocked |= grid.pixel_mask(layout, active, own, params)
        blocked |= grid.pad_mask(pad_pts, pad_nets, my_net, G)
        blocked |= trace_mask

        start = grid.to_ij(pad_xy[i, 0], pad_xy[i, 1])
        goal = grid.to_ij(anchor_xy[i, 0], anchor_xy[i, 1])
        blocked[start] = False
        blocked[goal] = False

        cells = _astar(blocked, start, goal)
        if cells is None:
            return routes[:oi], anchor_xy[i]  # signal failure at this anchor

        pts = [grid.to_xy(r, c) for (r, c) in cells]
        pts[0] = [pad_xy[i, 0], pad_xy[i, 1]]
        pts[-1] = [anchor_xy[i, 0], anchor_xy[i, 1]]
        if which == "col":
            pts.append([anchor_xy[i, 0], anchor_xy[i, 1] + pixel_h / 2])  # spine entry

        P = remove_collinear(np.array(pts, dtype=float))

        routes[i] = Route(net_name=my_net, pad_num=pad_nums[i],
                          anchor_xy=anchor_xy[i], pad_xy=pad_xy[i], polyline_xy=P)

        # mark this trace as an obstacle for later nets
        trace_mask |= grid.polyline_mask(P, G)

    return routes, None


# =========================================================================
# Grid
# =========================================================================

class _Grid:
    def __init__(self, layout, outline_xy, params):
        G = params.trace_w_mm + params.clearance_mm
        minx, miny, maxx, maxy = layout.board_poly.bounds
        w, h = maxx - minx, maxy - miny
        step = max(G / 2, max(w, h) / 600.0)
        pad = 2 * step
        minx -= pad; miny -= pad; maxx += pad; maxy += pad

        nx = int(np.ceil((maxx - minx) / step)) + 1
        ny = int(np.ceil((maxy - miny) / step)) + 1

        self.step = step
        self.minx = minx
        self.miny = miny
        self.nx = nx
        self.ny = ny
        self.shape = (ny, nx)

        xs = minx + np.arange(nx) * step
        ys = miny + np.arange(ny) * step
        self.X, self.Y = np.meshgrid(xs, ys)  # (ny, nx)

        # base: outside board OR too close to a (non-cable) outline edge
        inside = shapely.contains_xy(layout.board_poly, self.X.ravel(), self.Y.ravel())
        base = ~inside.reshape(ny, nx)
        ec = params.board_edge_clear_mm
        n = len(outline_xy)
        for ei in range(1, n):  # skip edge 0 (cable attach)
            a = outline_xy[ei]
            b = outline_xy[(ei + 1) % n]
            base |= self._seg_dist(a, b) < ec
        self.base = base

    def to_ij(self, x, y):
        c = int(round((x - self.minx) / self.step))
        r = int(round((y - self.miny) / self.step))
        c = min(max(c, 0), self.nx - 1)
        r = min(max(r, 0), self.ny - 1)
        return (r, c)

    def to_xy(self, r, c):
        return [self.minx + c * self.step, self.miny + r * self.step]

    def _seg_dist(self, a, b):
        ax, ay = a; bx, by = b
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:
            return np.hypot(self.X - ax, self.Y - ay)
        t = np.clip(((self.X - ax) * dx + (self.Y - ay) * dy) / L2, 0.0, 1.0)
        return np.hypot(self.X - (ax + t * dx), self.Y - (ay + t * dy))

    def pixel_mask(self, layout, active, own, params):
        Pxy = layout.placements_xy
        hw = params.pixel_w_mm / 2 + params.trace_w_mm / 2 + params.clearance_mm
        hh = params.pixel_h_mm / 2 + params.trace_w_mm / 2 + params.clearance_mm
        m = np.zeros(self.shape, dtype=bool)
        for p in active:
            if p in own:
                continue
            m |= (np.abs(self.X - Pxy[p, 0]) <= hw) & (np.abs(self.Y - Pxy[p, 1]) <= hh)
        return m

    def pad_mask(self, pad_pts, pad_nets, my_net, G):
        m = np.zeros(self.shape, dtype=bool)
        for (px, py), net in zip(pad_pts, pad_nets):
            if net == my_net:
                continue
            m |= np.hypot(self.X - px, self.Y - py) < G
        return m

    def polyline_mask(self, P, G):
        m = np.zeros(self.shape, dtype=bool)
        for s in range(len(P) - 1):
            m |= self._seg_dist(P[s], P[s + 1]) < G - 1e-9
        return m

    def pad_positions(self, layout, outline_xy, params):
        conn_pos = layout.connector_pos
        pp = params.pad_pitch_mm
        x_mid = np.mean(outline_xy[:2, 0])
        pad_x_left = x_mid - (conn_pos - 1) * pp / 2
        pad_y = layout.pad_y_mm
        pts, nets = [], []
        for pad_num, net_name in layout.pad_to_net.items():
            pts.append((pad_x_left + (pad_num - 1) * pp, pad_y))
            nets.append(net_name)
        return pts, nets


# =========================================================================
# A*  (8-connected, no corner cutting, mild straight-line preference)
# =========================================================================

_DIRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, 1.4142135623730951), (-1, 1, 1.4142135623730951),
         (1, -1, 1.4142135623730951), (1, 1, 1.4142135623730951)]
_TURN = 0.001  # tiny penalty to prefer straighter paths among equal-length routes


def _astar(blocked, start, goal):
    ny, nx = blocked.shape
    sr, sc = start
    gr, gc = goal
    if blocked[sr, sc] or blocked[gr, gc]:
        return None

    INF = float("inf")
    g = np.full((ny, nx), INF)
    g[sr, sc] = 0.0
    came = {}
    h = lambda r, c: ((r - gr) ** 2 + (c - gc) ** 2) ** 0.5
    pq = [(h(sr, sc), 0.0, sr, sc)]

    while pq:
        f, gcost, r, c = heapq.heappop(pq)
        if (r, c) == (gr, gc):
            break
        if gcost > g[r, c] + 1e-12:
            continue
        prev = came.get((r, c))
        pdir = (0, 0) if prev is None else (_sign(r - prev[0]), _sign(c - prev[1]))
        for dr, dc, cost in _DIRS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= ny or nc >= nx:
                continue
            if blocked[nr, nc]:
                continue
            if dr != 0 and dc != 0 and (blocked[r + dr, c] or blocked[r, c + dc]):
                continue  # don't cut through obstacle corners
            turn = 0.0 if (dr, dc) == pdir else _TURN
            ng = gcost + cost + turn
            if ng < g[nr, nc] - 1e-12:
                g[nr, nc] = ng
                came[(nr, nc)] = (r, c)
                heapq.heappush(pq, (ng + h(nr, nc), ng, nr, nc))

    if g[gr, gc] == INF:
        return None
    path = [(gr, gc)]
    cur = (gr, gc)
    while cur != (sr, sc):
        cur = came.get(cur)
        if cur is None:
            return None
        path.append(cur)
    path.reverse()
    return path


def _sign(v):
    return (v > 0) - (v < 0)


def _nearest_active_pixel(layout, skip, anchor):
    Pxy = layout.placements_xy
    active = [i for i in range(len(Pxy)) if i not in skip]
    if not active:
        return None
    d = np.sum((Pxy[active] - np.asarray(anchor)) ** 2, axis=1)
    return active[int(np.argmin(d))]
