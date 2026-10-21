"""Contracts of the inner-layer hug router (pipeline/inner_hug.py), checked
independently of how it builds its wires: every wire stays under the taxels
(a feed's fan below the field excepted), keeps a via keep-out from every
foreign via and a trace pitch from every other wire, starts and ends where it
must, and the same input always gives the same copper."""



import numpy as np
import pytest
from shapely.geometry import LineString, Point

from opentac.cad.pipeline import inner_hug
from opentac.cad.pipeline.layout import layout_with_connector
from opentac.cad.pipeline.route_hug import build_plan
from opentac.cad.pipeline.types import Params

# cable edge is the first edge (0 -> 1)
RECT = [[0, 0], [40, 0], [40, 30], [0, 30]]
T_SHAPE = [[-12, 0], [12, 0], [12, 34], [34, 34], [34, 52], [-34, 52], [-34, 34], [-12, 34]]
# two notches from the top: every row above the notch floors is cut in three
TWO_NOTCH = [[-30, 0], [30, 0], [30, 40], [18, 40], [18, 22], [6, 22], [6, 40],
             [-6, 40], [-6, 22], [-18, 22], [-18, 40], [-30, 40]]


def _setup(outline, w=2.0, pitch=2.2):
    params = Params(pixel_w_mm=w, pixel_h_mm=w, pitch_x_mm=pitch, pitch_y_mm=pitch)
    xy = np.asarray(outline, dtype=float)
    layout = layout_with_connector(xy, params, join_split_nets=True)
    plan = build_plan(layout, xy, params)
    return layout, params, plan


def _check(paths, nets, sp, inside_from=0):
    """paths: [(net, polyline)]; the part from index inside_from on must stay
    in the region"""
    lines = [(n, LineString(p)) for n, p in paths]
    for n, p in paths:
        line = LineString(p[inside_from:]) if len(p) - inside_from >= 2 else Point(p[-1])
        assert sp.region.buffer(1e-3).covers(line), f"{n} leaves the taxels"
        foreign = sp.foreign(n)
        assert LineString(p).distance(foreign) >= sp.a - 1e-6, f"{n} too near a via"
    for i, (n1, l1) in enumerate(lines):
        for n2, l2 in lines[i + 1:]:
            if n1 != n2:
                assert l1.distance(l2) >= sp.s - 1e-6, f"{n1} too near {n2}"


@pytest.mark.parametrize("outline", [RECT, T_SHAPE], ids=["rect", "T"])
def test_feeds_route_every_row_cleanly(outline):
    layout, params, plan = _setup(outline)
    pads_before = {p for p, n in layout.pad_to_net.items() if n.startswith("ROW")}
    routes, failed, info = inner_hug.route_feeds(layout, params, plan)
    rows = {n for n, ch in plan.chains.items() if ch.face == "row"}
    assert not failed, info["fail_reasons"]
    got = routes["In1.Cu"]
    assert {r.net_name for r in got} == rows
    # each row gets its own pad, from the row pads
    assert sorted(r.pad_num for r in got) == sorted(pads_before)
    sp = inner_hug._Space(layout, params, plan.cut_xy)
    for r in got:
        P = np.asarray(r.polyline_xy)
        assert np.allclose(P[0], r.pad_xy), "a feed starts at its pad"
        own = sp.vxy[np.asarray(layout.row_net_per_pixel) == r.net_name]
        assert np.min(np.linalg.norm(own - P[-1], axis=1)) < 1e-6, "a feed ends on its row's via"
    # the fan below the field is outside the region by design: check the rest
    paths = []
    for r in got:
        P = [tuple(q) for q in r.polyline_xy]
        k = next(i for i, q in enumerate(P) if sp.region.buffer(1e-3).covers(Point(q)))
        paths.append((r.net_name, P))
        assert sp.region.buffer(1e-3).covers(LineString(P[k:])) if len(P) - k >= 2 else True
    _check(paths, rows, sp, inside_from=len(paths[0][1]))   # clearances only


def test_links_nest_around_notches():
    layout, params, plan = _setup(TWO_NOTCH, w=3.0, pitch=3.2)
    routes, done, info = inner_hug.route_links(layout, params, plan)
    split = {n for n, ch in plan.chains.items() if ch.face == "row" and len(ch.segments) > 1}
    assert split, "the fixture must cut rows"
    assert done == split, info["why"]
    sp = inner_hug._Space(layout, params, plan.cut_xy)
    _check([(r.net_name, [tuple(q) for q in r.polyline_xy]) for r in routes], split, sp)
    # every link runs from a via of one piece to a via of another
    for r in routes:
        P = np.asarray(r.polyline_xy)
        own = sp.vxy[np.asarray(layout.row_net_per_pixel) == r.net_name]
        for end in (P[0], P[-1]):
            assert np.min(np.linalg.norm(own - end, axis=1)) < 1e-6


def test_same_input_same_copper():
    out = []
    for _ in range(2):
        layout, params, plan = _setup(T_SHAPE)
        routes, failed, _ = inner_hug.route_feeds(layout, params, plan)
        out.append(sorted((r.net_name, r.pad_num, np.round(r.polyline_xy, 6).tobytes()) for r in routes["In1.Cu"]))
    assert out[0] == out[1]


def test_slot_assignment_keeps_order_gap_and_windows():
    xs = np.arange(0.0, 10.0, 0.05)
    yb = np.zeros_like(xs)
    cand = [(float(x), 0.0) for x in xs]
    starts = [(2.0, -10.0), (2.5, -10.0), (3.0, -10.0)]
    got = inner_hug._assign_slots(starts, cand, 0.41, xs, yb, 0.5,
                                  window=[(-np.inf, np.inf), (4.0, np.inf), (-np.inf, np.inf)])
    slots, _ = got
    xs_ = [s[0] for s in slots]
    assert xs_ == sorted(xs_) and min(np.diff(xs_)) >= 0.41 - 1e-9
    assert xs_[1] >= 4.0


@pytest.mark.parametrize("outline", [RECT, T_SHAPE], ids=["rect", "T"])
def test_trunk_feeds_every_row_planar(outline):
    from opentac.cad.pipeline import inner_trunk
    layout, params, plan = _setup(outline)
    pads_before = {p for p, n in layout.pad_to_net.items() if n.startswith("ROW")}
    routes, failed, info = inner_trunk.route_feeds(layout, params, plan)
    rows = {n for n, ch in plan.chains.items() if ch.face == "row"}
    assert not failed, info["fail_reasons"]
    got = routes["In1.Cu"]
    assert {r.net_name for r in got} == rows
    assert sorted(r.pad_num for r in got) == sorted(pads_before)
    sp = inner_hug._Space(layout, params, plan.cut_xy)
    lines = [(r.net_name, LineString(r.polyline_xy)) for r in got]
    for n, line in lines:
        assert line.distance(sp.foreign(n)) >= sp.a - 1e-6, f"{n} too near a via"
    for i, (n1, l1) in enumerate(lines):
        for n2, l2 in lines[i + 1:]:
            assert l1.distance(l2) >= sp.s - 1e-6, f"{n1} too near {n2}"
    for r in got:
        P = np.asarray(r.polyline_xy)
        assert np.allclose(P[0], r.pad_xy)
        own = sp.vxy[np.asarray(layout.row_net_per_pixel) == r.net_name]
        assert np.min(np.linalg.norm(own - P[-1], axis=1)) < 1e-6


def test_trunk_fans_never_cross_or_crowd():
    from opentac.cad.pipeline.inner_trunk import _fans
    exits = [(x, 0.0) for x in (-2.0, -1.59, -1.18, 0.2, 0.61, 1.02)]
    starts = [(x, -14.0) for x in (-4.0, -3.5, -3.0, -2.5, -2.0, -1.5)]
    paths = _fans(exits, starts, -0.5, 0.4)
    lines = [LineString(p) for p in paths]
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            assert lines[i].distance(lines[j]) >= 0.4 - 1e-6


def test_lane_links_nest_around_notches():
    from opentac.cad.pipeline import inner_trunk
    layout, params, plan = _setup(TWO_NOTCH, w=3.0, pitch=3.2)
    routes, done, info = inner_trunk.route_links(layout, params, plan)
    split = {n for n, ch in plan.chains.items() if ch.face == "row" and len(ch.segments) > 1}
    assert done == split, info["why"]
    sp = inner_hug._Space(layout, params, plan.cut_xy)
    _check([(r.net_name, [tuple(q) for q in r.polyline_xy]) for r in routes], split, sp)
    for r in routes:            # a U: straight, axis-aligned legs only
        P = np.asarray(r.polyline_xy)[1:-1]
        assert all(abs(a[0] - b[0]) < 1e-9 or abs(a[1] - b[1]) < 1e-9 for a, b in zip(P[:-1], P[1:]))


# a bump off to the side of a stem: its rows lie nowhere above the stem, so
# the bundle has to steer over to them; a stem too narrow for every row
BUMP_T = [[-8, 0], [8, 0], [8, 20], [20, 20], [20, 30], [16, 30], [16, 37], [10, 37], [10, 30],
          [-20, 30], [-20, 20], [-8, 20]]
NARROW_T = [[-5, 0], [5, 0], [5, 30], [30, 30], [30, 46], [-30, 46], [-30, 30], [-5, 30]]


def _row_order(layout, routes):
    """the via-row heights of the shipped rows, read along the pads left to right"""
    ys = {}
    for i, n in enumerate(layout.row_net_per_pixel):
        ys.setdefault(n, []).append(layout.placements_xy[i, 1])
    got = sorted((r for rs in routes.values() for r in rs), key=lambda r: r.pad_xy[0])
    return [float(np.mean(ys[r.net_name])) for r in got]


def _monotone(seq):
    return all(a < b for a, b in zip(seq, seq[1:])) or all(a > b for a, b in zip(seq, seq[1:]))


def _clean(layout, params, plan, routes, preplaced=None):
    sp = inner_hug._Space(layout, params, plan.cut_xy)
    for L, rs in routes.items():
        lines = [(r.net_name, LineString(r.polyline_xy)) for r in rs]
        lines += [(n, LineString(p)) for n, p in (preplaced or {}).get(L, [])]
        for n, line in lines[:len(rs)]:
            assert line.distance(sp.foreign(n)) >= sp.a - 1e-6, f"{n} too near a via on {L}"
        for i, (n1, l1) in enumerate(lines[:len(rs)]):
            for n2, l2 in lines[i + 1:]:
                if n1 != n2:
                    assert l1.distance(l2) >= sp.s - 1e-6, f"{n1} too near {n2} on {L}"
        for r in rs:
            P = np.asarray(r.polyline_xy)
            own = sp.vxy[np.asarray(layout.row_net_per_pixel) == r.net_name]
            assert np.min(np.linalg.norm(own - P[-1], axis=1)) < 1e-6, "a feed ends on its row's via"
            # from where it enters the field: axis-aligned legs only
            k = next(i for i, q in enumerate(P) if sp.region_b.covers(Point(q)))
            legs = P[k:]
            assert all(abs(a[0] - b[0]) < 1e-9 or abs(a[1] - b[1]) < 1e-9 for a, b in zip(legs[:-1], legs[1:]))


def test_trunk_steers_to_rows_off_the_stem_in_order():
    from opentac.cad.pipeline import inner_trunk
    layout, params, plan = _setup(BUMP_T)
    rows = {n for n, ch in plan.chains.items() if ch.face == "row"}
    routes, failed, info = inner_trunk.route_feeds(layout, params, plan)
    assert not failed, info["fail_reasons"]
    assert {r.net_name for r in routes["In1.Cu"]} == rows
    assert _monotone(_row_order(layout, routes)), "pads not in row order"
    _clean(layout, params, plan, routes)


def test_trunk_puts_what_in1_cannot_take_on_in2_in_one_pad_order():
    from opentac.cad.pipeline import inner_trunk
    layout, params, plan = _setup(NARROW_T)
    links, _, _ = inner_trunk.route_links(layout, params, plan)
    pre = {"In2.Cu": [(r.net_name, r.polyline_xy) for r in links]}
    import copy
    one, failed1, _ = inner_trunk.route_feeds(copy.deepcopy(layout), params, plan, ("In1.Cu",), pre)
    assert failed1, "the fixture must overflow In1"
    routes, failed, info = inner_trunk.route_feeds(layout, params, plan, ("In1.Cu", "In2.Cu"), pre)
    assert not failed, info["fail_reasons"]
    assert routes["In2.Cu"] and len(routes["In1.Cu"]) == len(one["In1.Cu"])
    assert _monotone(_row_order(layout, routes)), "pads not in row order across the layers"
    _clean(layout, params, plan, routes, pre)
