"""OpenTac: design, build, read and simulate custom tactile sensors.

    import opentac
    sensor = opentac.Sensor()           # a readout board on USB
    frame = sensor.read()               # an opentac.Frame
    m = opentac.load_manifest("manifest.json")

Names here (imported on first use, so `import opentac` stays light for people
who only run the CAD tool):

    Sensor          a readout board on USB, or one node behind a bus controller
                    (opentac.device.Sensor). Not the simulator: that is
                    opentac.sim.Sensor, which takes a manifest instead of a port.
    Bus             every sensor behind a Teensy bus controller (opentac.device.Bus)
    Frame           one reading: taxels in manifest `pixels` order (opentac.frame.Frame)
    load_manifest   read a manifest.json or a design .zip (opentac.manifest.load)

Subpackages: opentac.device (readout boards), opentac.manifest, opentac.sim
(simulator), opentac.transfer, opentac.record, opentac.calibrate, opentac.cad
(the CAD tool). The command line is `opentac --help`.
"""

__version__ = "0.1.0.dev0"

__all__ = ["Bus", "Frame", "Sensor", "load_manifest", "__version__"]

_LAZY = ("Bus", "Frame", "Sensor", "load_manifest")


def __getattr__(name):
    if name in ("Sensor", "Bus"):
        from opentac import device
        return getattr(device, name)
    if name == "Frame":
        from opentac.frame import Frame
        return Frame
    if name == "load_manifest":
        from opentac.manifest import load
        return load
    raise AttributeError(f"module 'opentac' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
