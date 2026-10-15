"""Regression test: routing must succeed on boards whose cable-attach edge
is shorter than the connector (any polygon with enough vertices, or a
DXF-imported circle).

Root cause (found by tracing pass-by-pass on a 16-gon vs a working 12-gon):
route_columns.py/route_rows.py's edge-proximity clearance check measured
distance against the raw, un-unioned outline_xy edges (excluding only the
single cable-attach edge index 0). When the connector is wider than that
edge's own natural length, board_poly's union legitimately swallows the
neighboring vertex into the board's interior (e.g. on a 16-gon at r=20 with
a 10.5mm connector, the vertex ending the 7.8mm cable edge ends up 1.35mm
inside the true boundary) — but the raw-edge check didn't know that, and
flagged routes correctly passing through the widened opening as "too close
to a board edge" that wasn't physically there anymore. This is the same
bug class fixed earlier for the export/preview code, independently present
in routing validation. Fixed by checking distance against
board_edge_check_boundary() (board_poly's real exterior, minus the cable
region) instead of the raw polygon edges.
"""

import sys
import os

import numpy as np
from opentac.cad.pipeline.types import Params
from opentac.cad.pipeline.geometry import rotate_for_cable
from opentac.cad.pipeline.layout import layout_with_connector
from opentac.cad.pipeline.route_columns import route_columns_grid
from opentac.cad.pipeline.route_rows import route_rows_grid


def _route(outline, params):
    outline = rotate_for_cable(outline)
    layout = layout_with_connector(outline, params)
    total = len(layout.placements_xy)
    col_routes, col_removed = route_columns_grid(layout, outline, params)
    row_routes, row_removed = route_rows_grid(layout, outline, params, col_removed)
    return total, len(row_removed)


def _regular_polygon(n, r=20.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def test_many_sided_polygons_route_successfully():
    """16/24/32-gons at r=20 with default 4mm-pixel params previously lost
    every single pixel (49/49, 51/51, 49/49 removed). Require the large
    majority to survive now.
    """
    params = Params()
    for n in [16, 24, 32]:
        total, removed = _route(_regular_polygon(n), params)
        surviving = total - removed
        assert surviving > total * 0.7, (
            f"{n}-gon: only {surviving}/{total} pixels survived — routing "
            f"regression on many-sided polygons"
        )
        print(f"test_many_sided_polygons_route_successfully: {n}-gon OK "
              f"({surviving}/{total} surviving)")


def test_few_sided_polygons_unaffected():
    """hexagon/octagon/12-gon always worked; confirm the fix didn't change
    that (same shapes, same params as the many-sided test above).
    """
    params = Params()
    for n in [6, 8, 12]:
        total, removed = _route(_regular_polygon(n), params)
        surviving = total - removed
        assert surviving > total * 0.7, (
            f"{n}-gon: only {surviving}/{total} pixels survived — regression "
            f"on a previously-working shape"
        )
        print(f"test_few_sided_polygons_unaffected: {n}-gon OK "
              f"({surviving}/{total} surviving)")


def test_finely_flattened_circle_routes_successfully():
    """The original bug report: a 64-vertex circle (matching DXF-import
    flattening) with a small pixel size lost all 92 pixels. Require the
    large majority to survive now.
    """
    params = Params(pixel_w_mm=3, pixel_h_mm=3, pitch_x_mm=3.2, pitch_y_mm=3.2)
    total, removed = _route(_regular_polygon(64), params)
    surviving = total - removed
    assert surviving > total * 0.7, (
        f"circle: only {surviving}/{total} pixels survived — routing "
        f"regression on finely-tessellated curved boundaries"
    )
    print(f"test_finely_flattened_circle_routes_successfully: OK "
          f"({surviving}/{total} surviving)")


if __name__ == "__main__":
    test_many_sided_polygons_route_successfully()
    test_few_sided_polygons_unaffected()
    test_finely_flattened_circle_routes_successfully()
    print("\n=== ALL ROUTING / CURVED BOUNDARY TESTS PASSED ===")
