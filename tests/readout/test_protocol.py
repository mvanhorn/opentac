"""Parsing the board's lines: frames (F, X), replies, and the bus controller's records."""
from pathlib import Path

import numpy as np
import pytest

from opentac.device.link import BusLink
from opentac.device.protocol import (explain, parse_caps, parse_config, parse_frame, parse_param,
                                     parse_zero)
from opentac.frame import Frame

FIXTURE = Path(__file__).resolve().parents[2] / "firmware" / "tests" / "fixtures" / "bus_records.txt"
IMU = "-12,1003,32767,-2000000,0,123456,0.70711,0.00000,0.70711,0.00000,-179.99,0.00,45.50"


def test_raster_frame():
    f = parse_frame(f"F,99,7,0,3150,4,10,20,30,4095,{IMU}", 4, t=1.5)
    assert isinstance(f, Frame)
    assert f.taxels.tolist() == [10, 20, 30, 4095] and f.mask.all() and f.t == 1.5
    assert (f.index, f.generation, f.mode, f.dt_us, f.dropped) == (99, 7, 0, 3150, 4)
    assert f.imu["accel_mg"] == [-12, 1003, 32767] and f.imu["euler_deg"] == [-179.99, 0.0, 45.5]
    assert f.units == "counts" and not f.sparse


def test_xampling_frame():
    # 10 taxels; read 0, 3 and 9: mask bytes LSB first = 0x09, 0x02
    f = parse_frame(f"X,5,1,2,900,0,3,0902,11,12,13,{IMU}", 10)
    assert np.nonzero(f.mask)[0].tolist() == [0, 3, 9]
    assert f.taxels[[0, 3, 9]].tolist() == [11, 12, 13]
    assert np.isnan(f.taxels[[1, 2, 4, 5, 6, 7, 8]]).all()
    assert f.sparse and f.units == "counts above zero"


@pytest.mark.parametrize("line, active", [
    (f"F,1,0,0,10,0,1,2,3,{IMU}", 4),                 # one value short
    (f"X,1,0,1,10,0,2,0100,7,8,{IMU}", 10),            # mask marks 1, n says 2
    (f"X,1,0,1,10,0,1,01,7,{IMU}", 10),                # mask one byte short
    (f"X,1,0,1,10,0,1,0004,7,{IMU}", 10),              # marks taxel 10 of 10
])
def test_malformed_frames_refuse_rather_than_misplace(line, active):
    with pytest.raises(ValueError):
        parse_frame(line, active)


def test_other_lines_are_not_frames():
    for line in ("V,1,2,3,4", "M,0,0,1,1,1,81,1", "R,OK,Z", "# hello", ""):
        assert parse_frame(line, 81) is None


def test_replies():
    caps = parse_caps("R,CAPS,1,32,32,1024,00000007,0000001B,21,00000000,READOUT60")
    assert caps["board"] == "READOUT60" and caps["steps"] == 32 and caps["tactile_modes"] == 7
    assert parse_caps("R,CAPS,1,32,32,1024,00000007,0000001B,20")["board"] == "READOUT32"
    assert parse_param("R,PARAM,35,GAIN_SW,0,15,2") == {"id": 35, "name": "GAIN_SW", "lo": 0, "hi": 15, "value": 2}
    cfg = parse_config("R,CFG,2,9,9,81,4200,4200,81,04EE4CD5,1A2B3C4D,241")
    assert cfg["config_id"] == 0x04EE4CD5 and cfg["active"] == 81 and cfg["bytes"] == 241
    assert parse_config("R,NOCFG") is None
    assert parse_zero("R,OK,Z,64,310,402,16,27,0")["noise_max"] == 27
    assert "baseline" in explain("NOBASE")
    assert "flash program failed at 0x0801f000" in explain("FLASH,2,00000080,00000008,0801F000,00000468,128")


def test_bus_controller_records_parse_to_what_the_node_published():
    """firmware/tests/bus_host_test.cpp writes these with the controller's own formatter."""
    blocks, current = [], None
    for line in FIXTURE.read_text().splitlines():
        if line.startswith("# expect "):
            fields = dict(kv.split("=", 1) for kv in line[len("# expect "):].split(" "))
            current = {"expect": fields, "lines": []}
            blocks.append(current)
        elif line:
            current["lines"].append(line)
    assert len(blocks) >= 10
    for block in blocks:
        e = block["expect"]
        node, active = int(e["node"]), int(e["active"])
        values = [int(v) for v in e["values"].split(";")] if e["values"] else []
        parsed = [BusLink.NODE_LINE.match(ln) for ln in block["lines"]]
        assert all(parsed) and all(int(m.group(1)) == node and m.group(2) for m in parsed)
        assert sum(m.group(3).startswith("V,") for m in parsed) == int(e["vib"])
        frames = [f for f in (parse_frame(m.group(3), active) for m in parsed) if f is not None]
        assert len(frames) == (1 if active else 0)
        if not active:
            continue
        f = frames[0]
        assert f.sparse == (e["sparse"] == "1")
        want = np.array([np.nan if v < 0 else v for v in values], dtype=float)
        assert np.array_equal(f.taxels, want, equal_nan=True)
        assert (f.index, f.generation, f.dt_us, f.dropped) == (99, 7, 3150, 4)
        if e["imu"] == "1":
            assert f.imu["accel_mg"] == [-12, 1003, 32767] and f.imu["gyro_mdps"] == [-2000000, 0, 123456]
            assert abs(f.imu["quat_wxyz"][0] - 0.70711) < 1e-4 and f.imu["euler_deg"] == [-179.99, 0.0, 45.5]
        else:
            assert f.imu["quat_wxyz"] == [1.0, 0.0, 0.0, 0.0] and f.imu["accel_mg"] == [0, 0, 0]
