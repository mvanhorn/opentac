"""A sensor made for an OpenTac readout board must land its lines on that board's
drive and sense sides.

On both boards one half of the FPC connector is wired to the drive side (74HC595
outputs) and the other half to the sense side (the analog mux). Read from the
boards' KiCad netlists (hardware/readout_60pin, hardware/readout_30pin/bottom):

    60-pin, J4 = Molex 505110-6091: pads 1-30 drive (r3..r32), 31-60 sense (c1..c30)
    30-pin, J4 = Molex 505110-3091: pads 1-15 sense (c1..c15), 16-30 drive (r1..r15)

The firmware reads either orientation (rows on the drive side or on the sense
side), and a flipped cable swaps the halves, so the requirement is: all of a
sensor's rows in one half, all of its columns in the other.
"""
import pytest

from opentac.cad.main import GenerateRequest, generate

BOARD_HALVES = {
    60: (set(range(1, 31)), set(range(31, 61))),
    30: (set(range(1, 16)), set(range(16, 31))),
}


def _square(w, h):
    return [[-w / 2, 0], [w / 2, 0], [w / 2, h], [-w / 2, h]]


CASES = [
    # (outline, pitch mm, readout_board, expected connector)
    (_square(20, 20), 2.2, "auto", 30),     # 9 x 9: fits the 30-pin board
    (_square(32, 32), 2.2, "auto", 30),     # about 14 x 14
    (_square(40, 40), 2.2, "auto", 60),     # more than 15 per side
    (_square(60, 60), 2.2, "60pin", 60),    # up to 30 x 30
    (_square(20, 20), 2.2, "60pin", 60),    # a small sensor on the big board
    (_square(24, 50), 2.2, "auto", 60),     # tall: rows exceed 15
]


@pytest.mark.parametrize("outline,pitch,board,want", CASES)
def test_rows_and_columns_land_on_opposite_halves(outline, pitch, board, want):
    resp = generate(GenerateRequest(
        outline=outline, router="hug", readout_board=board,
        pixel_w_mm=pitch - 0.2, pixel_h_mm=pitch - 0.2, pitch_x_mm=pitch, pitch_y_mm=pitch))
    layout = resp.edit_data["layout"]
    assert layout["connector_pos"] == want
    pad_to_net = {int(p): n for p, n in layout["pad_to_net"].items()}
    row_pads = {p for p, n in pad_to_net.items() if str(n).startswith("ROW_")}
    col_pads = {p for p, n in pad_to_net.items() if str(n).startswith("COL_")}
    a, b = BOARD_HALVES[want]
    assert (row_pads <= a and col_pads <= b) or (row_pads <= b and col_pads <= a), (
        f"rows on pads {sorted(row_pads)} and columns on {sorted(col_pads)} mix the "
        f"board's drive and sense halves")


def test_a_grid_too_big_for_the_30_pin_board_says_so():
    with pytest.raises(Exception, match="30-pin readout board reads at most 15 rows x 15 columns"):
        generate(GenerateRequest(outline=_square(40, 40), router="hug", readout_board="30pin",
                                 pixel_w_mm=2.0, pixel_h_mm=2.0, pitch_x_mm=2.2, pitch_y_mm=2.2))
