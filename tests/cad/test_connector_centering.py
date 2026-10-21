"""Tests that connector pad assignment stays centred on the connector body.

The row/column boundary is the connector's mechanical centre line: the tail
corridor and the connector housing are both placed on it (see
apply_tail_extension), while the nets are packed outward from it (see
build_pad_to_net_map_centered). If the two segments hold unequal numbers of
assignable pads, the whole used-pad block — and the wire bundle feeding it —
slides off the housing by half the imbalance.

For example, a 16-row/16-column board on a 40-pos connector uses pads 5-20 and
21-36. With a 16/24 split it would use pads 1-16 and 17-32, and the connector
body would sit 2.00 mm off the tail centre.
"""


from opentac.cad.pipeline.layout import (
    CONNECTOR_FAMILY,
    build_pad_to_net_map_centered,
    connector_segment_pad_ranges,
    select_connector_for_nets,
    split_row_pin_order,
)
from opentac.cad.pipeline.tail_extension import _active_split_pad


class _StubLayout:
    """Carries just the two attributes _active_split_pad reads."""

    def __init__(self, pad_to_net, connector_pos):
        self.pad_to_net = pad_to_net
        self.connector_pos = connector_pos


def test_segments_are_equal_and_centred():
    for conn_pos in [*CONNECTOR_FAMILY, 32]:
        left, right, fixed_nc = connector_segment_pad_ranges(conn_pos)
        left_avail = [p for p in left if p not in fixed_nc]
        right_avail = [p for p in right if p not in fixed_nc]

        assert len(left_avail) == len(right_avail), (
            f"{conn_pos}-pos: uneven banks {len(left_avail)}/{len(right_avail)}")

        boundary = (left_avail[-1] + right_avail[0]) / 2.0
        centre = (conn_pos + 1) / 2.0
        assert abs(boundary - centre) < 1e-9, (
            f"{conn_pos}-pos: boundary at pad {boundary}, centre at {centre}")

        # Segments must partition the assignable pads with no gap or overlap.
        assert right_avail[0] == left_avail[-1] + 1
        assert left_avail + right_avail == [
            p for p in range(1, conn_pos + 1) if p not in fixed_nc]


def test_70_pos_keeps_its_balancing_no_connects():
    """70-pos has an even pad count but needs 3 dead pads per end to centre."""
    left, right, fixed_nc = connector_segment_pad_ranges(70)
    assert fixed_nc == [1, 2, 3, 68, 69, 70]
    assert left[0] == 4 and left[-1] == 35
    assert right[0] == 36 and right[-1] == 67


def test_assigned_pads_straddle_the_centre():
    """The split point between the assigned ROW and COL pads is the centre."""
    cases = [(16, 16), (13, 18), (20, 20), (8, 8), (24, 16), (25, 25), (32, 32),
             (10, 10), (15, 15), (2, 2), (18, 13)]
    for n_row, n_col in cases:
        conn_pos, swap = select_connector_for_nets(n_row, n_col)
        rows = [f"ROW_{i}" for i in range(n_row)]
        cols = [f"COL_{i}" for i in range(n_col)]
        pad_to_net = build_pad_to_net_map_centered(
            conn_pos, swap, split_row_pin_order(rows), cols)

        split = _active_split_pad(_StubLayout(pad_to_net, conn_pos))
        centre = (conn_pos + 1) / 2.0
        assert abs(split - centre) < 1e-9, (
            f"{n_row}r/{n_col}c on {conn_pos}-pos: split {split} vs centre {centre}")

        # Every net gets exactly one pad, and no pad is double-booked.
        assert len(pad_to_net) == n_row + n_col
        assert len(set(pad_to_net.values())) == n_row + n_col


def test_16x16_on_40_pos_is_centred():
    """A 16x16 board on a 40-pos connector uses pads 5-20 for rows and 21-36
    for columns."""
    conn_pos, swap = select_connector_for_nets(16, 16)
    assert conn_pos == 40 and not swap
    pad_to_net = build_pad_to_net_map_centered(
        conn_pos, swap,
        split_row_pin_order([f"ROW_{i}" for i in range(16)]),
        [f"COL_{i}" for i in range(16)])
    row_pads = sorted(p for p, n in pad_to_net.items() if n.startswith("ROW_"))
    col_pads = sorted(p for p, n in pad_to_net.items() if n.startswith("COL_"))
    assert (row_pads[0], row_pads[-1]) == (5, 20), row_pads
    assert (col_pads[0], col_pads[-1]) == (21, 36), col_pads
