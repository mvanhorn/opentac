# OpenTac

Design, build, read and simulate custom tactile sensors.

> **Status:** private, under construction. The code is being moved here from its
> development repos; see `RELEASE_PLAN.md` in the OpenTactileCAD working folder.

## Quickstart (target)

```bash
git clone https://github.com/opentac/opentac.github.io opentac && cd opentac
uv run opentac cad                    # design a sensor in the browser, locally
uv run opentac flash                  # put firmware on the readout board
uv run opentac setup my_design.zip    # tell the board which sensor is plugged in (once)
uv run opentac view                   # live heatmap
```

The only prerequisite is [uv](https://docs.astral.sh/uv/getting-started/installation/).

## Layout

| Folder | Contents |
|---|---|
| `opentac/` | the Python package: CAD tool, manifests, readout, viewer, sim, transfer |
| `firmware/` | readout board firmware (60-pin and 30-pin boards) |
| `hardware/` | readout board KiCad files, fabrication files and BOMs |
| `examples/` | small manifests, reference designs and short scripts |
| `docs/` | the website, opentac.github.io |

## License

To be decided before release.
