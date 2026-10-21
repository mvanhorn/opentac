"""Sensor and Bus against a stand-in board (tests/readout/fakeboard.py): no hardware."""
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from opentac.device import Bus, DeviceError, Sensor, cache_path
from opentac.device.link import UsbLink
from opentac.record import load_take, record

from .fakeboard import FakeBoard, FakeBus

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "manifests"
GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "reference_configs.json").read_text())


@pytest.fixture(autouse=True)
def opentac_home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENTAC_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def open_sensor(board, **kw):
    return Sensor(_link=UsbLink(link=board), **kw)


def setup_quietly(sensor, source):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sensor.setup(source)


def test_open_asserts_a_known_session():
    fake = FakeBoard("READOUT60")
    s = open_sensor(fake)
    assert s.board_name == "READOUT60" and s.board.name == "READOUT60"
    assert s.stored is None and s.manifest is None
    assert fake.values[2] == 0 and fake.values[6] == 8           # raster, MUX_SETTLE_US 8
    assert fake.values[8] == 0 and fake.values[9] == 0           # IMU off
    untouched = FakeBoard("READOUT60")
    open_sensor(untouched, raster=False, mux_settle_us=None, imu=None)
    assert not [c for c in untouched.commands if c.startswith("p ")]


def test_setup_stores_the_same_bytes_and_caches_the_manifest(opentac_home):
    fake = FakeBoard("READOUT60")
    s = open_sensor(fake)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stored = s.setup(EXAMPLES / "square_9x9_60pin.json")
    assert not [w for w in caught if "not yet been checked" in str(w.message)]
    assert fake.flash.hex() == GOLDEN["square_9x9_60pin.json"]["READOUT60"]
    assert stored["config_id"] == s.manifest.config_id and stored["active"] == 81
    cached = cache_path(stored["config_id"])
    assert cached.parent == opentac_home / "manifests"
    assert cached.read_bytes() == (EXAMPLES / "square_9x9_60pin.json").read_bytes()
    # The same skin again writes nothing to flash.
    setup_quietly(s, EXAMPLES / "square_9x9_60pin.json")
    assert fake.commits == 1


def test_setup_warns_for_a_board_not_checked_on_hardware(monkeypatch):
    """A board whose wiring has not been checked by pressing a skin warns at setup."""
    from dataclasses import replace

    from opentac.device import config as cfg

    unchecked = replace(cfg.board("readout60"), hardware_checked=False)
    monkeypatch.setattr(cfg, "board", lambda name: unchecked)
    s = open_sensor(FakeBoard("READOUT60"))
    with pytest.warns(UserWarning, match="not yet been checked"):
        s.setup(EXAMPLES / "square_9x9_60pin.json")


def test_a_new_session_finds_the_manifest_by_the_boards_id():
    fake = FakeBoard("NODE16")
    setup_quietly(open_sensor(fake), EXAMPLES / "square_9x9_30pin.json")
    again = open_sensor(fake)
    assert again.manifest is not None and again.manifest.n == 81
    assert again.manifest.config_id == again.stored["config_id"]


def test_wrong_board_is_refused_before_anything_is_sent():
    fake = FakeBoard("NODE16")
    s = open_sensor(fake)
    with pytest.raises(Exception, match="not on the NODE16 connector"):
        setup_quietly(s, EXAMPLES / "square_9x9_60pin.json")
    assert fake.flash is None and not [c for c in fake.commands if c.startswith("S")]


def test_read_and_frames():
    fake = FakeBoard("NODE16")
    s = open_sensor(fake)
    with pytest.raises(DeviceError, match="no skin"):
        s.read()
    setup_quietly(s, EXAMPLES / "pinky_tip_30pin.json")
    f = s.read()
    assert f.taxels.shape == (32,) and f.mask.all() and f.mode == 0
    frames = list(s.frames(limit=5))
    assert len(frames) == 5 and [g.index for g in frames] == sorted(g.index for g in frames)


def test_xampling_needs_a_zero_then_streams_sparse_frames():
    fake = FakeBoard("NODE16")
    s = open_sensor(fake)
    setup_quietly(s, EXAMPLES / "square_9x9_30pin.json")
    with pytest.raises(DeviceError, match="baseline"):
        s.xampling(budget=16)
    z = s.zero()
    assert z["frames"] == 64 and z["noise_max"] == 27
    assert s.xampling(budget=16, grow=False) == 16 and fake.values[2] == 1   # fixed
    f = s.read()
    assert f.sparse and int(f.mask.sum()) == 16 and np.isnan(f.taxels[~f.mask]).all()
    s.xampling(budget=16)                    # grow is the default
    assert fake.values[2] == 2
    s.raster()
    assert s.read().mask.all()


def test_gain_on_the_60pin_board():
    s = open_sensor(FakeBoard("READOUT60"))
    assert s.feedback_ohm() == pytest.approx(10000)
    assert s.set_feedback(1000) == pytest.approx(1 / (1 / 10000 + 1 / 1100))
    assert s.get("GAIN_SW") == 2
    assert s.feedback_ohm(0b0101) == pytest.approx(1 / (1 / 10000 + 1 / 4700 + 1 / 270))
    with pytest.raises(DeviceError, match="gain switch"):
        open_sensor(FakeBoard("READOUT60", gain_ok=False)).set("GAIN_SW", 3)
    with pytest.raises(DeviceError, match="no gain switch"):
        open_sensor(FakeBoard("NODE16")).set_feedback(1000)


def test_parameters_clamp_and_names():
    s = open_sensor(FakeBoard("NODE16"))
    assert s.set("MUX_SETTLE_US", 500) == 50          # the board clamps; set() returns what it stored
    assert s.set("LED_MODE", "manual") == 1
    with pytest.raises(DeviceError, match="no parameter"):
        s.set("NOPE", 1)


def test_bus_nodes_are_sensors():
    with Bus(link=FakeBus(nodes=3)) as bus:
        assert bus.version == 5 and len(bus) == 3 and bus[1].node == 1
        assert bus[0].board_name == "NODE16"
        setup_quietly(bus[0], EXAMPLES / "square_9x9_30pin.json")
        setup_quietly(bus[2], EXAMPLES / "pinky_tip_30pin.json")
        got = list(bus.frames(limit=6))
        nodes = {node for node, _ in got}
        assert nodes == {0, 2}
        for node, frame in got:
            assert frame.taxels.shape == ((81,) if node == 0 else (32,))
            assert frame.node == node and frame.t_us is not None
        with pytest.raises(DeviceError, match="no online node 7"):
            bus[7]


def test_record_a_take_and_read_it_back(tmp_path):
    fake = FakeBoard("NODE16")
    s = open_sensor(fake)
    setup_quietly(s, EXAMPLES / "pinky_tip_30pin.json")
    path = record(s, tmp_path / "takes", seconds=0.05, description="tape roll, edge-on", log=lambda *_: None)
    assert path.name.endswith("tape_roll_edge_on")
    take = load_take(path)
    assert take["info"]["format"] == "opentac_view_take/1"
    assert take["raw"].dtype == np.int16 and take["raw"].shape[1] == 32
    assert take["xy"].shape == (32, 2) and "sampled" not in take
    assert take["manifest"].config_id == s.stored["config_id"]
    assert take["info"]["board"]["name"] == "NODE16"
    assert take["info"]["settings"]["mux_settle_us"] == 8
    # Xampling: unread taxels are -1 in raw and False in `sampled`.
    s.zero()
    s.xampling(budget=8)
    path = record(s, tmp_path / "takes", seconds=0.05, log=lambda *_: None)
    take = load_take(path)
    assert take["sampled"].sum(axis=1).max() == 8
    assert (take["raw"][~take["sampled"]] == -1).all()
    assert take["info"]["settings"]["xamp"]["m"] == 8
