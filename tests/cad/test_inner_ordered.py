"""Geometric contracts of the ordered inner router, independent of its solver."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from opentac.cad.pipeline.inner_engine import verify
from opentac.cad.pipeline.inner_ordered import allowed_region, bridges, solve, source_order
from opentac.cad.pipeline.route_inner import taxel_cells


def rectangle(nrows=4):
    xs = np.array([-4., 0., 4.])
    ys = np.arange(nrows, dtype=float) * 4 + 2
    vias = np.array([(x, y) for y in ys for x in xs])
    nets = [f"ROW_{i}" for i in range(nrows) for _ in xs]
    field = box(-5.275, 0, 6.725, ys[-1] + 2)
    board = box(-6, -8, 7, ys[-1] + 3)
    rows = [dict(net=f"ROW_{i}", y=y, vias=xs, pad_num=i + 1,
                 pad_xy=np.array([(i - (nrows - 1) / 2) * .5, -6.]))
            for i, y in enumerate(ys)]
    return xs, ys, vias, nets, field, board, rows


@pytest.mark.parametrize("nrows", [1, 3, 6])
def test_complete_paths_clear_vias_wires_and_board(nrows):
    xs, ys, vias, nets, field, board, rows = rectangle(nrows)
    ranks, refused = source_order(rows, xs, vias, field, .475, .4, 4.)
    assert not refused
    region = allowed_region(field, board, -6.01)
    paths, info = solve([dict(r, order=ranks[r['net']]) for r in rows],
                        xs, ys, region, .475, .4, 4.)
    assert set(paths) == {r['net'] for r in rows}, info
    assert set(info['pads'].values()) == set(range(1, nrows + 1))
    lines = {n: LineString(np.round(p, 6)) for n, p in paths.items()}
    for n, line in lines.items():
        assert region.covers(line)
        assert any(np.linalg.norm(np.array(paths[n][0]) - v) < 1e-6
                   for v, vn in zip(vias, nets) if vn == n)
        for v, vn in zip(vias, nets):
            if vn != n:
                assert line.distance(Point(v)) >= .475
        for other, geometry in lines.items():
            if other != n:
                assert line.distance(geometry) >= .4


def test_input_order_does_not_change_copper_or_pad_assignment():
    xs, ys, vias, nets, field, board, rows = rectangle()
    outcomes = []
    for perm in [rows, rows[::-1], rows[2:] + rows[:2]]:
        ranks, refused = source_order(perm, xs, vias, field, .475, .4, 4.)
        assert not refused
        outcomes.append(solve([dict(r, order=ranks[r['net']]) for r in perm],
                              xs, ys, allowed_region(field, board, -6.01), .475, .4, 4.))
    assert outcomes[0] == outcomes[1] == outcomes[2]


@pytest.mark.parametrize('fixture', ['inner_u_channel.json', 'inner_c_channel.json'])
def test_global_blocks_and_connector_shelves_on_real_inputs(fixture):
    data = json.loads((Path(__file__).parent / 'fixtures' / fixture).read_text())
    placements = np.asarray(data['placements'])
    vias = placements + [data['via_x'], 0.]
    xs, ys = np.unique(vias[:, 0]), np.unique(vias[:, 1])
    count, pitch = len(ys), data['pitch']
    board = Polygon(data['board']).buffer(-.3)
    field = taxel_cells(SimpleNamespace(placements_xy=placements),
                        SimpleNamespace(pitch_x_mm=pitch, pitch_y_mm=pitch)).intersection(board)
    region = allowed_region(field, board, data['pad_y'] - .001)
    outcomes = []
    # Net labels and input iteration order must not determine geometry.
    for labels in [[f'ROW_{i}' for i in range(count)], [f'N{count-i}' for i in range(count)]]:
        rows = [dict(net=labels[i], y=y, vias=vias[vias[:, 1] == y, 0],
                     pad_num=i+1, pad_xy=(data['pad_x'][i], data['pad_y']))
                for i, y in enumerate(ys)]
        nets = [labels[int(np.argmin(abs(ys-y)))] for x, y in vias]
        ranks, refused = source_order(rows[::-1], xs, vias, region, .475, .4, pitch)
        assert not refused
        paths, info = solve([dict(r, order=ranks[r['net']]) for r in rows],
                            xs, ys, region, .475, .4, pitch)
        assert len(paths) == count, info
        assert not verify(paths, {n: n for n in paths}, vias, nets, region,
                          .475, .4, region_slack=0)
        outcomes.append([paths[n] for n in labels])
    assert outcomes[0] == outcomes[1]


def test_tail_exception_does_not_allow_routing_along_sensor_sides():
    field = box(-2, 0, 2, 10)
    region = allowed_region(field, box(-5, -10, 5, 12), -8)
    assert region.covers(Point(4, -4))
    assert not region.covers(Point(4, 5))
    assert not region.covers(Point(0, 11))


def test_capacity_refusal_emits_no_partial_bundle():
    xs, ys, vias, nets, field, board, rows = rectangle(4)
    # A connector narrower than the sum of the four trace pitches cannot fit.
    region = field.union(box(-.3, -8, .3, .01))
    paths, info = solve([dict(r, order=i) for i, r in enumerate(rows)],
                        xs, ys, region, .475, .4, 4.)
    assert paths == {}
    assert 'reason' in info


def test_corridor_certificate_includes_holes_between_via_rows():
    xs, ys, vias, nets, field, board, rows = rectangle(3)
    region = allowed_region(field, board, -6.01).difference(box(-5.2, 7, -4.9, 8))
    row = dict(rows[-1], order=0)
    paths, info = solve([row], xs, ys, region, .475, .4, 4.)
    assert len(paths) == 1, info
    assert region.covers(LineString(paths[row['net']]))


@pytest.mark.parametrize("pts", [[(.5, .5)], [(.5, .5), (.5, .5)]])
def test_point_paths_are_checked_against_vias_and_placed_copper(pts):
    assert verify({'a': pts}, {'a': 'a'}, np.array([[.5, .5]]), ['b'],
                  box(0, 0, 1, 1), .375, .2) == {'a'}
    assert verify({'a': pts}, {'a': 'a'}, np.empty((0, 2)), [],
                  box(0, 0, 1, 1), .375, .2,
                  others=[('b', Point(.5, .5))]) == {'a'}


def test_strict_checker_refuses_small_region_escape_and_crossing():
    paths = {'a': [(1.0005, .2), (1.0005, .8)]}
    assert verify(paths, {'a': 'a'}, np.empty((0, 2)), [], box(0, 0, 1, 1),
                  .375, .2, region_slack=0) == {'a'}
    paths = {'a': [(0, 0), (1, 1)], 'b': [(0, 1), (1, 0)]}
    assert verify(paths, {n: n for n in paths}, np.empty((0, 2)), [],
                  box(-1, -1, 2, 2), .375, .2, region_slack=0) == {'a', 'b'}


def test_nested_bridges_connect_both_pieces_without_crossing():
    vias = np.array([(x, y) for y in [0., 4., 8., 12., 16.]
                     for x in [0., 4., 8., 12., 16.] if y <= 8 or x != 8])
    nets = [f"R{int(y)}" for x, y in vias]
    field = unary_union([box(x - 1.275, y - 2.005, x + 2.725, y + 2.005) for x, y in vias])
    links = [dict(id=k, net=f"R{int(y)}", y=y,
                  A=np.flatnonzero((vias[:, 1] == y) & (vias[:, 0] < 8)),
                  B=np.flatnonzero((vias[:, 1] == y) & (vias[:, 0] > 8)))
             for k, y in enumerate([12., 16.])]
    paths, refused = bridges(links, vias, field, .475, .4, 4.)
    assert not refused
    assert len(paths) == 2
    assert not verify(paths, {l['id']: l['net'] for l in links}, vias, nets,
                      field, .475, .4, region_slack=0)
