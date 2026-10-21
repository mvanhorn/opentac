#!/usr/bin/env bash
# Build the OpenTac firmware. No hardware needed, nothing is flashed.
#
#   firmware/build.sh                 every target, into firmware/build/
#   firmware/build.sh readout60       only the named targets
#   firmware/build.sh --prebuilt      every target, then copy them to firmware/prebuilt/
#
# Targets:
#   readout60       60-pin readout board                -> build/READOUT60.bin
#   node16          30-pin board (NODE16)               -> build/NODE16.bin
#   readout32       70-pin readout board (older boards) -> build/READOUT32.bin
#   node16_bus      30-pin board as a daisy-chain node  -> build/NODE16_bus.bin
#   bus_controller  Teensy 4.0 daisy-chain controller   -> build/bus_controller.hex
#
# Needs arduino-cli and the STM32 core 2.7.1 (and the Teensy core for the
# controller; it is skipped if that core is missing). The two Arduino libraries
# are installed into firmware/build/arduino on the first run, so the build does
# not depend on what else is in your sketchbook.
#
# Board flags go in build.extra_flags, not compiler.cpp.extra_flags: the core's
# I2C driver (twi.c) is C, and the bus node needs I2C_TXRX_BUFFER_SIZE to reach
# it too, or it keeps a 32-byte buffer and cuts every frame chunk short.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/build"
STM_FQBN='STMicroelectronics:stm32:GenG4:pnum=GENERIC_G431KBUX,usb=CDCgen,xusb=FS,xserial=generic'
TEENSY_FQBN='teensy:avr:teensy40:usb=serial,speed=600,opt=o2std'
LIB_VERSIONS=("STM32duino LSM6DSV16BX@1.0.0" "Adafruit NeoPixel@1.15.5")

PREBUILT=0
TARGETS=()
for arg in "$@"; do
  case "$arg" in
    --prebuilt) PREBUILT=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) TARGETS+=("$arg") ;;
  esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(readout60 node16 readout32 node16_bus bus_controller)

command -v arduino-cli >/dev/null || { echo "arduino-cli not found: see https://opentac.github.io/use/flash/"; exit 1; }
mkdir -p "$OUT"

# A private sketchbook: only the libraries this firmware uses, at fixed versions.
export ARDUINO_DIRECTORIES_USER="$OUT/arduino"
USERLIBS="$ARDUINO_DIRECTORIES_USER/libraries"
if [ ! -d "$USERLIBS/STM32duino_LSM6DSV16BX" ] || [ ! -d "$USERLIBS/Adafruit_NeoPixel" ]; then
  echo "== installing Arduino libraries into $USERLIBS"
  arduino-cli lib update-index >/dev/null
  arduino-cli lib install "${LIB_VERSIONS[@]}"
fi
STM_LIBS=(--library "$USERLIBS/STM32duino_LSM6DSV16BX" --library "$USERLIBS/Adafruit_NeoPixel"
          --library "$HERE/libraries/OpenTactileBus")

# arduino-cli links through per-folder archives (objs.a) only when the quoted
# object paths add up to more than 30000 characters, and the two ways give
# differently laid out binaries. This folder name is long enough to choose the
# archives wherever the repo is cloned, so a build is byte-for-byte the same on
# every machine (and the same as the released binaries).
LONG="object-paths-longer-than-30000-characters-make-arduino-cli-link-through-archives-so-the-binary-is-reproducible"

stm() {   # stm <name> <flags>
  local name="$1" flags="$2" work="$OUT/work/$1/$LONG"
  echo "== $name ($flags)"
  arduino-cli compile --fqbn "$STM_FQBN" "${STM_LIBS[@]}" \
    --build-property "build.extra_flags=$flags" \
    --build-path "$work" "$HERE/opentac_readout" | grep -E "Sketch uses|Global variables"
  grep -q "^LOAD .*objs\.a" "$work/opentac_readout.ino.map" ||
    { echo "linked without archives: the binary will differ from the release"; exit 1; }
  cp "$work/opentac_readout.ino.bin" "$OUT/$name.bin"
}

for t in "${TARGETS[@]}"; do
  case "$t" in
    readout60) stm READOUT60 "-DOT_BOARD_READOUT60" ;;
    node16)    stm NODE16 "-DOT_BOARD_NODE16" ;;
    readout32) stm READOUT32 "-DOT_BOARD_READOUT32" ;;
    node16_bus)
      # The bus test checks ot_bus.h and regenerates the records the Python
      # parser is tested against; a change to the formatter must show up there.
      if command -v c++ >/dev/null; then
        echo "== bus host test (ot_bus.h)"
        c++ -std=c++17 -O2 -Wall -Wextra -I"$HERE/libraries/OpenTactileBus/src" \
          "$HERE/tests/bus_host_test.cpp" -o "$OUT/bus_host_test"
        "$OUT/bus_host_test" "$OUT/bus_records.txt" | tail -1
        cmp -s "$OUT/bus_records.txt" "$HERE/tests/fixtures/bus_records.txt" ||
          { echo "bus records changed: update firmware/tests/fixtures/bus_records.txt"; exit 1; }
      fi
      stm NODE16_bus "-DOT_BUS_NODE -DOT_BOARD_NODE16 -DI2C_TXRX_BUFFER_SIZE=160" ;;
    bus_controller)
      if ! arduino-cli core list | grep -q '^teensy:avr'; then
        echo "== bus_controller skipped: Teensy core not installed"; continue
      fi
      echo "== bus_controller (Teensy 4.0)"
      arduino-cli compile --fqbn "$TEENSY_FQBN" --library "$HERE/libraries/OpenTactileBus" \
        --build-path "$OUT/work/bus_controller" "$HERE/bus_controller" | grep -E "Memory|FLASH|RAM" | head -3
      cp "$OUT/work/bus_controller/bus_controller.ino.hex" "$OUT/bus_controller.hex" ;;
    *) echo "unknown target: $t"; exit 1 ;;
  esac
done

if [ "$PREBUILT" = 1 ]; then
  mkdir -p "$HERE/prebuilt"
  for f in READOUT60.bin NODE16.bin READOUT32.bin NODE16_bus.bin bus_controller.hex; do
    [ -f "$OUT/$f" ] && cp "$OUT/$f" "$HERE/prebuilt/$f" && chmod 644 "$HERE/prebuilt/$f"
  done
  (cd "$HERE/prebuilt" && shasum -a 256 *.bin *.hex > SHA256SUMS)
  echo "== copied to $HERE/prebuilt"
fi
echo "== done: $OUT"
