"""Routing succeeds on boards whose cable-attach edge is shorter than the
connector (a polygon with many vertices, or a DXF-imported circle).

When the connector is wider than the cable edge, board_poly's union absorbs
the neighbouring vertex into the board's interior: on a 16-gon at r = 20 with
a 10.5 mm connector, the vertex ending the 7.8 mm cable edge lies 1.35 mm
inside the true boundary. The edge-proximity check in route_columns.py and
route_rows.py therefore measures distance against
board_edge_check_boundary() (board_poly's real exterior, minus the cable
region), not against the raw outline edges. Against the raw edges, routes
through the widened opening would be rejected as too close to an edge that
is not part of the board.
"""


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
    """16-, 24- and 32-gons at r = 20 with the default 4 mm pixels keep the
    large majority of their pixels.
    """
    params = Params()
    for n in [16, 24, 32]:
        total, removed = _route(_regular_polygon(n), params)
        surviving = total - removed
        assert surviving > total * 0.7, (
            f"{n}-gon: only {surviving}/{total} pixels survived routing "
            f"on a many-sided polygon"
        )


def test_few_sided_polygons_route_successfully():
    """Hexagons, octagons and 12-gons, with the parameters of the many-sided
    test above, keep the large majority of their pixels.
    """
    params = Params()
    for n in [6, 8, 12]:
        total, removed = _route(_regular_polygon(n), params)
        surviving = total - removed
        assert surviving > total * 0.7, (
            f"{n}-gon: only {surviving}/{total} pixels survived routing "
            f"on a few-sided polygon"
        )


def test_finely_flattened_circle_routes_successfully():
    """A 64-vertex circle (the flattening of a DXF import) with 3 mm pixels
    keeps the large majority of its pixels.
    """
    params = Params(pixel_w_mm=3, pixel_h_mm=3, pitch_x_mm=3.2, pitch_y_mm=3.2)
    total, removed = _route(_regular_polygon(64), params)
    surviving = total - removed
    assert surviving > total * 0.7, (
        f"circle: only {surviving}/{total} pixels survived routing "
        f"on a finely tessellated curved boundary"
    )
