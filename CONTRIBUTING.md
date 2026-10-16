# Contributing to OpenTac

## Set up

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the repository and
run everything through `uv run` from the repository root. The first `uv run` creates `.venv/`
from `uv.lock`.

```bash
git clone --depth 1 https://github.com/opentac/opentac.github.io opentac
cd opentac
uv run opentac --version
```

## Where things live

| Path | What |
|---|---|
| `opentac/cad/` | OpenTac-CAD: the FastAPI backend (`main.py`), the layout and routing pipeline (`pipeline/`), the exporters (`export/`) and the browser frontend (`web/`) |
| `opentac/manifest/`, `opentac/frame.py` | the sensor manifest and the `Frame` every reader returns |
| `opentac/device/` | readout boards over USB and the I2C bus: protocol, config, flashing |
| `opentac/sim/`, `opentac/transfer.py` | OpenTac-Sim (foam and bump models) and OpenTac-Transfer |
| `opentac/cli.py`, `opentac/view.py`, `opentac/record.py` | the `opentac` command, live viewer and recording |
| `firmware/` | readout board firmware, with prebuilt binaries in `firmware/prebuilt/` |
| `hardware/` | readout board KiCad projects and fabrication files |
| `docs/` | the website (MkDocs); `mkdocs.yml` is its table of contents |
| `.claude/skills/opentac/` | the agent skill: how to design a sensor from a description |
| `examples/` | example manifests, scripts, STL parts and the sources of the ready-to-order designs |
| `tests/` | the test suite: `cad/`, `readout/` (manifests and boards), `sim/` (simulation and transfer), `hardware/` (fabrication files) |
| `scripts/` | maintenance scripts (see each file's docstring) |

## Run the tests

```bash
uv run --extra sim pytest -q
```

About 6 minutes on a laptop; most of it is the CAD tests, which generate and route real boards.
`uv run pytest tests/sim -q` runs one folder. On a fresh clone 11 tests skip, because they need
data that is not in the repository. Optional tests and what they need:

| Tests | Need | Without it |
|---|---|---|
| the MuJoCo tests in `tests/sim/test_sensor.py` (3) | `--extra sim` (MuJoCo) | skipped |
| design-chat tests in `tests/cad/test_chat_*.py` | `--extra demo` (google-genai) | skipped |
| `tests/test_qr_page.py` | `node` on the PATH | skipped |
| the firmware's config parser, in `tests/readout/test_config.py` | a C++ compiler (`c++`) | skipped |
| `tests/sim/test_paper_numbers.py` (9) | `OPENTAC_PAPER_RESULTS=/path/to/results_pose.json`, the paper's scored hand-press rows (29 MB, not in the repository) | skipped |
| the lab-manifest sweeps in `tests/readout/` (2) | `OPENTAC_STM_TEST=/path/to/stm_test`, the 30 manifests the firmware was developed against, and the original config tool | skipped; four of those manifests are in `tests/readout/fixtures/stm_test/` and always run |

CI (`.github/workflows/tests.yml`) installs `--extra demo --extra sim` and runs the whole suite on
every push.

To build the website as CI does, with no warnings allowed:

```bash
uv run --extra docs mkdocs build --strict -d /tmp/opentac-site
uv run --extra docs python scripts/preview_site.py    # the site, with OpenTac-CAD at /cad/
```

## Commit messages

One line that says what changed, in plain words, led by the area it touches. Add a body only when
the reason is not obvious. Examples from the history:

- `Readout boards: the Tag-Connect pads J2 are excluded from the position files, so JLCPCB's BOM step no longer stops on them`
- `Narrow tail: the cable is cut to its traces after routing and flares back to the connector`
- `Docs: 60-pin cable is A type (contacts down at both ends)`

Documentation is plain: what to do, what was measured, with numbers. If a page states a number,
say where it comes from.

## Report a problem

Open an issue with the bug report form. For a problem with a board or a sensor, include:

- the output of `uv run opentac info` with the board plugged in (or the error it prints);
- which board: 60-pin, 30-pin pair, or 30-pin boards on a Teensy bus;
- your operating system;
- the design ZIP from OpenTac-CAD, if the problem involves a sensor or its manifest;
- what you expected and what you saw: a screenshot of `uv run opentac view`, and a photo of the
  sensor's tail in the connector if readings are missing or mirrored.

For a design that will not generate or fails DRC, the design ZIP (or the `design.json` inside it)
is enough to reproduce it.
