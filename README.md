# OpenTac

Design, build, read and simulate flex-PCB tactile sensors for any robot surface.

OpenTac CAD turns a drawn outline, a DXF, or a surface of an STL part into a manufacturable
tactile sensor: a grid of taxels on a flex PCB, wired to a connector, with Gerbers, a printed
case and a manifest that describes the sensor. The same manifest configures the readout board,
the simulator and transfer between sensors.

**Website and docs:** https://opentac.github.io  ·  **Try it online:** https://opentac.github.io/cad/

## Quick start

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) (the launchers offer to
install it).

```bash
git clone https://github.com/opentac/opentac.github.io opentac
cd opentac
uv run opentac cad          # design a sensor in your browser, locally
```

Or double-click `Start_OpenTac.command` (macOS/Linux) or `Start_OpenTac.bat` (Windows).
The first run downloads what it needs, once.

With a readout board plugged in over USB:

```bash
uv run opentac flash                 # put the firmware on the board
uv run opentac setup my_design.zip   # tell the board which sensor is attached (once)
uv run opentac view                  # live heatmap
```

```python
import opentac
sensor = opentac.Sensor()     # finds the board; it already knows its sensor
frame = sensor.read()         # one value per taxel; frame.image(manifest) lays it out
```

## Design with an AI agent

The repo ships a skill at `.claude/skills/opentac/`, which Claude Code loads automatically when
you open this folder. Other agents: see [AGENTS.md](AGENTS.md).

## What is here

| Folder | Contents |
|---|---|
| `opentac/` | the Python package: CAD tool (`opentac.cad`), manifests, readout, viewer, simulation, transfer |
| `firmware/` | readout board firmware (60-pin and 30-pin boards) |
| `hardware/` | readout board KiCad files, fabrication files and BOMs |
| `examples/` | small manifests, example scripts, and the sources of the ready-to-order designs |
| `docs/` | the website |

## Readout boards

| Board | Reads up to | Size |
|---|---|---|
| 60-pin | 30 rows x 30 columns | 37 x 34.3 mm |
| 30-pin (two stacked boards) | 15 rows x 15 columns | 16 x 22.8 mm |

Both plug straight into USB, carry an IMU, and can be daisy-chained over I2C.

## Citing

See [CITATION.cff](CITATION.cff) and the [paper page](https://opentac.github.io/paper/).

## License

Code: MIT ([LICENSE](LICENSE)). Hardware designs: CERN-OHL-P-2.0 ([LICENSE-hardware](LICENSE-hardware)).
Documentation and media: CC BY 4.0 ([LICENSE-docs](LICENSE-docs)).
