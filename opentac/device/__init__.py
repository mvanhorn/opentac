"""Read OpenTac sensors from their readout boards.

    from opentac.device import Sensor

    s = Sensor()                 # the board on USB
    s.setup("my_design.zip")     # once per skin
    frame = s.read()             # opentac.frame.Frame, taxels in manifest `pixels` order

Bus nodes (many boards behind one Teensy 4.0): `Sensor(node=2)`, or `Bus()` for
all of them. See https://opentac.github.io/use/sensor/ and
https://opentac.github.io/reference/protocol/.
"""
from opentac.device.config import Board, board, boards, build_config, device_config, extract, fits
from opentac.device.protocol import BoardFrame, DeviceError, parse_frame
from opentac.device.sensor import Bus, Sensor, cache_lookup, cache_path, cache_store

__all__ = ["Board", "BoardFrame", "Bus", "DeviceError", "Sensor", "board", "boards",
           "build_config", "cache_lookup", "cache_path", "cache_store", "device_config",
           "extract", "fits", "parse_frame"]
