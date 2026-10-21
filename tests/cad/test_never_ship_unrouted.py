"""Fixed-outline mode never ships a board with nets known not to reach the
connector: when no attempt routes everything and pruning the failed taxels
does not either, generate refuses with a 400 instead of returning the board.
Rules that cannot reach the connector at all are refused up front."""



import pytest
from fastapi import HTTPException

from opentac.cad import main

T_SHAPE = [[-12, 0], [12, 0], [12, 34], [34, 34], [34, 52], [-34, 52], [-34, 34], [-12, 34]]


def _req(**kw):
    return main.GenerateRequest(outline=T_SHAPE, router="hug", pixel_w_mm=3.0, pixel_h_mm=3.0,
                                pitch_x_mm=3.2, pitch_y_mm=3.2, **kw)


def test_unconnected_nets_are_refused_not_shipped(monkeypatch):
    # A router that always reports one taxel's nets as failed: every attempt
    # and every prune round is left with an unconnected net. The board is
    # refused, although the prune rounds end with most taxels still present.
    real = main.route_for

    def refusing(params):
        route = real(params)

        def run(layout, outline_xy, p, *a, **k):
            col, row, plan, failed = route(layout, outline_xy, p, *a, **k)
            return col, row, plan, set(failed) | {0}
        return run
    monkeypatch.setattr(main, "route_for", refusing)
    with pytest.raises(HTTPException) as e:
        main.generate(_req(board_mode="fixed_keepout"))
    assert e.value.status_code == 400


def test_the_same_board_routes_without_the_fake_failure():
    r = main.generate(_req(board_mode="fixed_keepout"))
    assert r.drc["violations"] == 0 and r.stats["active_pixels"] > 100


@pytest.mark.parametrize("mode", ["expand", "fixed_keepout"])
def test_rules_wider_than_the_pin_pitch_are_refused(mode):
    # 0.25 + 0.28 > 0.5mm: the wires to neighbouring connector pins cannot
    # clear each other (a 40mm square kept 2 taxels in grow mode).
    with pytest.raises(HTTPException) as e:
        main.generate(_req(board_mode=mode, trace_w_mm=0.25, clearance_mm=0.28))
    assert e.value.status_code == 400
    assert "pin pitch" in e.value.detail
