# Firmware

The readout boards run one sketch, `opentac_readout/`, built once per board.
The daisy-chain controller, a Teensy 4.0, runs `bus_controller/`.

To put firmware on a board, see [Flash the firmware](https://opentac.github.io/use/flash/)
(`uv run opentac flash`). This page is about what is in this folder and how it
is built.

## Targets

| Target (`build.sh`) | Board | Build flags | Binary |
|---|---|---|---|
| `readout60` | 60-pin board | `-DOT_BOARD_READOUT60` | `READOUT60.bin` |
| `node16` | 30-pin pair | `-DOT_BOARD_NODE16` | `NODE16.bin` |
| `readout32` | earlier 70-pin board | `-DOT_BOARD_READOUT32` | `READOUT32.bin` |
| `node16_bus` | 30-pin pair as a daisy-chain node | `-DOT_BUS_NODE -DOT_BOARD_NODE16 -DI2C_TXRX_BUFFER_SIZE=160` | `NODE16_bus.bin` |
| `bus_controller` | Teensy 4.0 daisy-chain controller | | `bus_controller.hex` |

The board name in the binary (`READOUT60`, `NODE16`, `READOUT32`) is what the
board reports to `?`, and what `opentac flash` checks before writing. The
60-pin board does not run as a daisy-chain node: a 30 x 30 sensor does not fit
the node build's memory.

## Folder

| Path | Contents |
|---|---|
| `opentac_readout/` | The readout board sketch. `ot_api.h`: commands, parameters and capabilities. `ot_devcfg.h`: the sensor config a board stores. `ot_xampling.h`: Xampling. |
| `bus_controller/` | The Teensy 4.0 controller for the daisy chain. |
| `libraries/OpenTactileBus/` | The bus code that node and controller share (`ot_bus.h`). |
| `tests/` | Host tests, no hardware: `bus_host_test.cpp` (the bus) and `xampling_host_test.cpp` (Xampling against real configs; build line at the top of the file). |
| `prebuilt/` | The released binaries and `SHA256SUMS`. |

## Prebuilt binaries

`prebuilt/` holds one binary per target, built by `build.sh --prebuilt` from
the sources in this folder. `opentac flash` writes them. They are also packed
into the Python wheel (as `opentac/device/firmware/`), so a pip-installed
`opentac` can flash a board without a copy of this repository.

Check them against their checksums:

```sh
cd firmware/prebuilt
shasum -a 256 -c SHA256SUMS      # macOS
sha256sum -c SHA256SUMS          # Linux
```

## Build

`build.sh` compiles every target into `firmware/build/` and flashes nothing:

```sh
firmware/build.sh                 # every target
firmware/build.sh readout60       # only the named targets
firmware/build.sh --prebuilt      # every target, then copy to prebuilt/ and rewrite SHA256SUMS
```

It needs:

- arduino-cli 1.x.
- The STM32 core, `STMicroelectronics:stm32` version **2.7.1** (the commands
  are in [Flash the firmware, arduino-cli](https://opentac.github.io/use/flash/#3-arduino-cli)). The board
  is *Generic STM32G4 series*, part *Generic G431KBUx*, USB CDC.
- For the controller only, the Teensy core (`teensy:avr`). Without it the
  `bus_controller` target is skipped.

On its first run `build.sh` installs the two Arduino libraries it uses,
STM32duino LSM6DSV16BX 1.0.0 and Adafruit NeoPixel 1.15.5, into
`firmware/build/arduino/`, so your own sketchbook is neither used nor changed.
The `node16_bus` target first compiles and runs `tests/bus_host_test.cpp`
(needs a C++17 compiler) and stops if the records it writes no longer match
`tests/fixtures/bus_records.txt`.

The build is reproducible: the same sources give byte-for-byte the same
binaries on any machine, and `build.sh` stops if the link step would make them
differ (the long folder name under `build/work/` is there for that; see the
comment in the script). The current `prebuilt/` files were built with
arduino-cli 1.4.1, STM32 core 2.7.1 and Teensy core 1.62.0.

### Expected warnings

The 60-pin and 70-pin builds print

```
Low memory available, stability problems may occur.
Global variables use 27736 bytes (84%) of dynamic memory, leaving 5032 bytes for local variables.
```

This is expected. Those boards size their per-taxel tables (scan order, frame,
Xampling zero and noise) for a full 32 x 32 grid, and the tables are allocated
when the firmware is built, so the figure is the same whatever sensor is
plugged in. The 30-pin build, with 16 x 16 tables, uses 57%.
