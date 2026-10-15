"""OpenTac: design, build, read and simulate custom tactile sensors.

    import opentac
    sensor = opentac.Sensor()           # a readout board on USB
    frame = sensor.read()               # an opentac.Frame
    m = opentac.load_manifest("manifest.json")

Imports are lazy, so `import opentac` stays light for people who only run the CAD tool.
"""

__version__ = "0.1.0.dev0"

__all__ = ["Bus", "Frame", "Sensor", "load_manifest", "__version__"]


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
