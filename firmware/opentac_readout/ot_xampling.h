// OpenTAC-Xampling -- binary compressive sampling over any provisioned skin.
//
// The same scheme as the hand-written Teensy sketches (binary_sampling_64x64,
// adaptiveFR_binary, binary_sampling_SHIFTING_SHEAR), derived instead from the
// DeviceConfig, so it runs on whatever skin the board was provisioned with:
//
//   1. Last frame's trigger points are re-checked first.
//   2. The sketches' coarse-to-fine quadtree pattern searches the skin in
//      PHYSICAL space (row_geom/col_geom), shifted by one taxel each frame so
//      contacts that fall between coarse lattice points are found.
//   3. Any sample at or above threshold grows: its whole shear bump and its
//      8 physical neighbours (each with its whole bump) are sampled, and
//      every new sample above threshold grows in turn.
//
// Two budgets, as in the originals:
//   FIXED  every sample counts toward M -- constant frame time.
//   GROW   M counts pattern samples only; growth is unbounded, so a light
//          touch costs little and a large contact is followed completely.
// In both, growth waits until SEARCH_MIN pattern samples have been taken, so
// a large contact from the last frame cannot use up a frame before it has
// looked anywhere else (the fixed-M 64x64 sketch could).
//
// ----- What changed from the Teensy sketches, and why -----------------------
//
// Sample order. The Teensy drove each row from its own pin, so any order cost
// the same. Here the rows are a '595 walking 1 that only moves forward, and
// every change of driven line costs a walk and ROW_SETTLE_US. So a frame is as
// few forward passes as the algorithm allows, each read exactly as the raster
// scan reads (scan-plan entries in order, skipping those not queued):
//   1. last frame's seeds, then one pass per growth wave (empty, and free,
//      when nothing is being touched);
//   2. the search: M pattern points in GROW, SEARCH_MIN in FIXED, then growth;
//   3. FIXED only: one pass filling the rest of M from the pattern.
// A pass crosses the driven lines at most once, as a raster frame does. Growth
// waits for the end of its pass rather than starting at each trigger; seeds
// still grow before the search, so the search skips held contacts as the
// sketches' isSampled check did. In FIXED, triggers found by the filling pass
// grow next frame, as seeds.
//
// Pattern. The 32x32 ctrMsrRows/ctrMsrCols tables, verbatim, shifted one taxel
// per frame mod 32 -- but diagonally, not along the row as the sketches did
// (see the end of xampFrame). Every OpenTAC skin fits in 32x32,
// so the physical grid is laid on the table's top-left corner; a position that
// falls off the skin or into a hole in its mask is skipped and does not count
// toward any budget.
//
// Shear bumps. The shear sketch filled a hard-coded (r>>1, c>>1) 2x2 block.
// Here the bump is the manifest's shear group, found by searching a small grid
// window around the taxel. Building the grid checks that every group fits in
// that window; a skin whose groups do not is refused for xampling rather than
// sampled with half-filled bumps. Bump fill is a setting (XAMP_BUMP_FILL, off
// by default): off, or on a pressure-only skin, growth takes the 8 neighbours
// alone.
//
// Values and threshold. Values are baseline-subtracted and clamped at zero,
// against a per-taxel baseline captured by the firmware (`Z`). A taxel
// triggers when its value reaches its OWN hands-off noise peak (also from `Z`)
// plus the margin XAMP_THRESH. One global threshold cannot fit a skin whose
// taxels range from ~3 counts of noise to occasional 600-count spikes.
// Measured on the 32x32: a 64-frame capture and a 30-count margin leave the
// 950 quiet taxels ~0.005 false triggers per frame.
//
// Everything here is hardware-free: the one hardware call is a read function
// passed in, so the host test drives the identical code.

#pragma once

#include <stdint.h>
#include <string.h>

#include "ot_devcfg.h"

#define XAMP_MODE_FIXED        1
#define XAMP_MODE_GROW         2

#define XAMP_GRID_SIDE         32u
#define XAMP_GRID_CELLS        (XAMP_GRID_SIDE * XAMP_GRID_SIDE)
#define XAMP_WORDS             ((OTCFG_MAX_TAXELS + 31u) / 32u)
#define XAMP_MAX_PRIORITY      50u
#define XAMP_MAX_GROUP_RADIUS  2u     // bumps up to 5x5 taxels
#define XAMP_NONE              0xFFFFu

// Grid position: grid row in the top five bits, grid column in the low five.
static inline uint16_t xampPos(uint8_t row, uint8_t col) {
  return (uint16_t)(((uint16_t)row << 5) | col);
}
static inline uint8_t xampPosRow(uint16_t pos) { return (uint8_t)(pos >> 5); }
static inline uint8_t xampPosCol(uint16_t pos) { return (uint8_t)(pos & 0x1Fu); }

// ----- Grid ------------------------------------------------------------------
//
// Samples are named by their ScanPlan entry, so reading a batch is the raster
// loop. Two tables relate entries to the physical grid, built once per config.

struct XampGrid {
  const OtDeviceConfig *config;
  const ScanPlan *plan;
  uint16_t active;
  uint8_t groupRadius;                        // 0 = no shear groups
  uint16_t gridEntry[XAMP_GRID_CELLS];        // grid position -> entry, or NONE
  uint16_t entryPos[OTCFG_MAX_TAXELS];        // entry -> grid position
};

static inline uint16_t xampShearOf(const XampGrid *grid, uint16_t entry) {
  return otCfgShearGroup(grid->config, grid->plan->entryOut[entry]);
}

// Entries in a (2r+1)^2 window around `entry` whose taxels share `group`.
static uint16_t xampCountGroupNear(const XampGrid *grid, uint16_t entry,
                                   uint16_t group, uint8_t radius) {
  const int16_t r0 = (int16_t)xampPosRow(grid->entryPos[entry]);
  const int16_t c0 = (int16_t)xampPosCol(grid->entryPos[entry]);
  uint16_t count = 0;
  for (int16_t r = r0 - radius; r <= r0 + radius; ++r) {
    if (r < 0 || r >= (int16_t)XAMP_GRID_SIDE) continue;
    for (int16_t c = c0 - radius; c <= c0 + radius; ++c) {
      if (c < 0 || c >= (int16_t)XAMP_GRID_SIDE) continue;
      const uint16_t other = grid->gridEntry[xampPos((uint8_t)r, (uint8_t)c)];
      if (other != XAMP_NONE && xampShearOf(grid, other) == group) ++count;
    }
  }
  return count;
}

// Flatten config + scan plan into the grid tables. False when this skin cannot
// be xampled -- a taxel with no grid position, or a shear group too spread out
// to be found by a window search. Raster is unaffected either way.
//
// The group check is O(taxels^2) flash reads, ~tens of ms at 1024 taxels, and
// runs only when a config is loaded.
static bool xampBuildGrid(const OtDeviceConfig *config, const ScanPlan *plan,
                          XampGrid *grid) {
  grid->config = config;
  grid->plan = plan;
  grid->active = 0;
  grid->groupRadius = 0;
  for (uint16_t i = 0; i < XAMP_GRID_CELLS; ++i) grid->gridEntry[i] = XAMP_NONE;
  if (!config->valid) return false;

  // Taxel index -> grid position first, parked in entryPos...
  uint16_t taxel = 0;
  for (uint8_t r = 0; r < config->rows; ++r) {
    for (uint8_t c = 0; c < config->cols; ++c) {
      if (!otCfgIsActive(config, r, c)) continue;
      const uint8_t gr = otCfgRowGridIndex(config, r);
      const uint8_t gc = otCfgColGridIndex(config, c);
      if (gr >= XAMP_GRID_SIDE || gc >= XAMP_GRID_SIDE) return false;
      grid->entryPos[taxel++] = xampPos(gr, gc);
    }
  }
  if (taxel != config->activeCount) return false;
  grid->active = taxel;

  // ...then grid position -> entry through the plan's entry -> taxel map, and
  // entryPos rebuilt from that, entry-indexed. No scratch array needed.
  for (uint16_t e = 0; e < taxel; ++e) {
    const uint16_t pos = grid->entryPos[plan->entryOut[e]];
    if (grid->gridEntry[pos] != XAMP_NONE) return false;   // unreachable: parse
    grid->gridEntry[pos] = e;                               // forbids duplicates
  }
  for (uint16_t pos = 0; pos < XAMP_GRID_CELLS; ++pos) {
    if (grid->gridEntry[pos] != XAMP_NONE) grid->entryPos[grid->gridEntry[pos]] = pos;
  }

  if (!otCfgHasShear(config)) return true;
  grid->groupRadius = 1;
  for (uint16_t e = 0; e < taxel; ++e) {
    const uint16_t group = xampShearOf(grid, e);
    if (group == OTCFG_NO_SHEAR) continue;
    uint16_t members = 0;
    for (uint16_t k = 0; k < taxel; ++k) {
      if (otCfgShearGroup(config, k) == group) ++members;
    }
    if (xampCountGroupNear(grid, e, group, grid->groupRadius) == members) continue;
    if (xampCountGroupNear(grid, e, group, XAMP_MAX_GROUP_RADIUS) != members) {
      return false;
    }
    grid->groupRadius = XAMP_MAX_GROUP_RADIUS;
  }
  return true;
}

// ----- Pattern ----------------------------------------------------------------
//
// ctrMsrRows / ctrMsrCols from adaptiveFR_binary.ino (identical in
// binary_sampling_SHIFTING_SHEAR), a permutation of all 1024 positions. Kept in
// flash: 2 KB of flash, no RAM.

#define XAMP_PATTERN_LENGTH 1024u

static const uint8_t XAMP_PATTERN_ROW[XAMP_PATTERN_LENGTH] = {
  15, 15, 15, 7, 23, 7, 23, 7, 23, 7, 23, 7, 23, 7, 23, 3, 11, 19, 27, 3, 11, 19, 27, 3, 11, 19, 27, 3, 11, 19, 27, 3,
  11, 19, 27, 3, 11, 19, 27, 3, 11, 19, 27, 3, 11, 19, 27, 3, 11, 19, 27, 3, 11, 19, 27, 3, 11, 19, 27, 3, 11, 19, 27, 1,
  5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1,
  5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1,
  5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1,
  5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1,
  5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1,
  5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0,
  2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 3,
  7, 11, 15, 19, 23, 27, 31, 7, 15, 23, 31, 3, 7, 11, 15, 19, 23, 27, 31, 1, 5, 9, 13, 15, 17, 21, 25, 29, 31, 3, 7, 11,
  15, 19, 23, 27, 31, 7, 15, 23, 31, 3, 7, 11, 15, 19, 23, 27, 31, 1, 3, 5, 9, 11, 13, 17, 19, 21, 25, 27, 29, 31, 3, 7,
  11, 15, 19, 23, 27, 31, 7, 15, 23, 31, 3, 7, 11, 15, 19, 23, 27, 31, 1, 5, 9, 13, 15, 17, 21, 25, 29, 31, 3, 7, 11, 15,
  19, 23, 27, 31, 7, 15, 23, 31, 3, 7, 11, 15, 19, 23, 27, 31, 1, 3, 5, 7, 9, 11, 13, 17, 19, 21, 23, 25, 27, 29, 31, 3,
  7, 11, 15, 19, 23, 27, 31, 7, 15, 23, 31, 3, 7, 11, 15, 19, 23, 27, 31, 1, 5, 9, 13, 15, 17, 21, 25, 29, 31, 3, 7, 11,
  15, 19, 23, 27, 31, 7, 15, 23, 31, 3, 7, 11, 15, 19, 23, 27, 31, 1, 3, 5, 9, 11, 13, 17, 19, 21, 25, 27, 29, 31, 3, 7,
  11, 15, 19, 23, 27, 31, 7, 15, 23, 31, 3, 7, 11, 15, 19, 23, 27, 31, 1, 5, 9, 13, 15, 17, 21, 25, 29, 31, 3, 7, 11, 15,
  19, 23, 27, 31, 7, 15, 23, 31, 3, 7, 11, 15, 19, 23, 27, 31, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
};
static const uint8_t XAMP_PATTERN_COL[XAMP_PATTERN_LENGTH] = {
  15, 7, 23, 7, 7, 23, 23, 3, 3, 11, 11, 19, 19, 27, 27, 3, 3, 3, 3, 11, 11, 11, 11, 19, 19, 19, 19, 27, 27, 27, 27, 1,
  1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9, 13, 13, 13, 13, 17, 17, 17, 17, 21, 21, 21, 21, 25, 25, 25, 25, 29, 29, 29, 29, 1,
  1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9, 13, 13, 13, 13, 13, 13, 13, 13, 17,
  17, 17, 17, 17, 17, 17, 17, 21, 21, 21, 21, 21, 21, 21, 21, 25, 25, 25, 25, 25, 25, 25, 25, 29, 29, 29, 29, 29, 29, 29, 29, 0,
  0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 8,
  8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14, 16,
  16, 16, 16, 16, 16, 16, 16, 18, 18, 18, 18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 20, 20, 20, 22, 22, 22, 22, 22, 22, 22, 22, 24,
  24, 24, 24, 24, 24, 24, 24, 26, 26, 26, 26, 26, 26, 26, 26, 28, 28, 28, 28, 28, 28, 28, 28, 30, 30, 30, 30, 30, 30, 30, 30, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8,
  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12,
  12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 16,
  16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 20,
  20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 28,
  28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9,
  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 13,
  13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 17,
  17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 21,
  21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 25,
  25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 29,
  29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 0,
  0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
  4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8,
  8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12,
  12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16,
  16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20,
  20, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24,
  24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28,
  28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
};
// ----- Frame ------------------------------------------------------------------

typedef uint16_t (*XampReadFn)(uint8_t step, uint8_t mux);
// Read the taxel just read again, after a further wait, without moving the
// shift register or mux. NULL = no confirmation.
typedef uint16_t (*XampRereadFn)(void);

struct XampSettings {
  uint8_t mode;          // XAMP_MODE_FIXED or XAMP_MODE_GROW
  uint16_t m;
  uint16_t threshold;    // margin, counts above each taxel's noise peak
  uint16_t searchMin;
  bool bumpFill;         // grow whole shear bumps, not just neighbours
};

struct XampState {
  // Carried between frames.
  uint16_t priority[XAMP_MAX_PRIORITY];   // entries that triggered, strongest first
  uint16_t priorityValue[XAMP_MAX_PRIORITY];
  uint8_t priorityCount;
  uint8_t shiftRow;
  uint8_t shiftCol;

  // This frame, all indexed by scan-plan entry. `sampled` is every entry read;
  // each one's baseline-subtracted value is in values[plan->entryOut[e]].
  uint32_t sampled[XAMP_WORDS];
  uint32_t pending[XAMP_WORDS];           // queued for the next batch
  uint32_t frontier[XAMP_WORDS];          // read, above threshold, not grown
  uint32_t taxelMask[XAMP_WORDS];         // sampled, by TAXEL (pixels order)
  uint16_t peak;                          // largest value read
  uint16_t count;                         // sampled + pending
  uint16_t limit;
  uint16_t fromPriority;
  uint16_t fromPattern;
  uint16_t fromGrowth;
  uint16_t batches;
  uint16_t confirms;                      // reads taken again this frame
};

// Forget everything carried between frames -- on a new config, since entries
// from another skin mean nothing.
static inline void xampReset(XampState *state) {
  state->priorityCount = 0;
  state->shiftRow = 0;
  state->shiftCol = 0;
  state->count = 0;
  memset(state->sampled, 0, sizeof(state->sampled));
}

static inline bool xampWasSampled(const XampState *st, uint16_t entry) {
  return (st->sampled[entry >> 5] >> (entry & 31u)) & 1u;
}

struct XampFrameContext {
  const XampGrid *grid;
  const XampSettings *settings;
  const uint16_t *baseline;
  const uint16_t *noise;       // per taxel, counts above baseline; NULL = 0
  uint16_t *values;
  XampState *state;
  XampReadFn read;
  XampRereadFn reread;
};

static bool xampQueue(XampFrameContext *ctx, uint16_t entry) {
  XampState *st = ctx->state;
  if (entry == XAMP_NONE || st->count >= st->limit) return false;
  const uint32_t bit = 1UL << (entry & 31u);
  const uint16_t word = entry >> 5;
  if ((st->sampled[word] | st->pending[word]) & bit) return false;
  st->pending[word] |= bit;
  ++st->count;
  return true;
}

// Trigger memory (hot start): up to 50 entries that triggered, strongest
// first. When more compete, the weakest is dropped. Only triggers from the seed
// and search passes are remembered, never growth: a contact is one hit that
// grows, and because seeds grow before the search runs, the search skips a
// held contact rather than re-finding it -- so a large contact does not fill
// the list.
static void xampRemember(XampState *st, uint16_t entry, uint16_t value) {
  uint8_t at = st->priorityCount;
  if (at == XAMP_MAX_PRIORITY) {
    if (value <= st->priorityValue[at - 1u]) return;
    --at;                                   // replace the weakest
  } else {
    ++st->priorityCount;
  }
  while (at > 0u && st->priorityValue[at - 1u] < value) {
    st->priority[at] = st->priority[at - 1u];
    st->priorityValue[at] = st->priorityValue[at - 1u];
    --at;
  }
  st->priority[at] = entry;
  st->priorityValue[at] = value;
}

// Remember the triggers of the pass just read (they are the whole frontier).
static void xampRememberPass(XampFrameContext *ctx) {
  XampState *st = ctx->state;
  for (uint16_t word = 0; word < XAMP_WORDS; ++word) {
    uint32_t bits = st->frontier[word];
    while (bits) {
      const uint16_t e = (uint16_t)((word << 5) | (uint16_t)__builtin_ctz(bits));
      bits &= bits - 1u;
      xampRemember(st, e, ctx->values[ctx->grid->plan->entryOut[e]]);
    }
  }
}

// Read every queued entry in plan order -- the raster loop, skipping. Triggers
// join the frontier.
static void xampReadBatch(XampFrameContext *ctx) {
  XampState *st = ctx->state;
  const ScanPlan *plan = ctx->grid->plan;
  const uint16_t margin = ctx->settings->threshold;
  const uint16_t *noise = ctx->noise;
  uint8_t step = 0;
  ++st->batches;
  for (uint16_t word = 0; word < XAMP_WORDS; ++word) {
    uint32_t bits = st->pending[word];
    if (bits == 0u) continue;
    st->sampled[word] |= bits;
    st->pending[word] = 0;
    while (bits) {
      const uint16_t entry = (uint16_t)((word << 5) | (uint16_t)__builtin_ctz(bits));
      bits &= bits - 1u;
      while (plan->stepStart[step + 1u] <= entry) ++step;
      const uint16_t taxel = plan->entryOut[entry];
      uint16_t raw = ctx->read(step, plan->entryMux[entry]);
      const uint16_t base = ctx->baseline[taxel];
      uint16_t value = (raw > base) ? (uint16_t)(raw - base) : 0u;
      const uint32_t trigger = (uint32_t)margin + (noise ? noise[taxel] : 0u);
      // Settle-and-confirm: a reading that would trigger is taken again after
      // a longer wait, and the second reading stands. Hands-off, sparse reads
      // at an 8 us mux settle spiked on 2.4% of quiet-taxel reads and on none
      // at 50 us, so the spikes are settling that has not finished; a press
      // stays high. It costs only on the reads that would trigger.
      if (value >= trigger && ctx->reread) {
        raw = ctx->reread();
        value = (raw > base) ? (uint16_t)(raw - base) : 0u;
        ++st->confirms;
      }
      ctx->values[taxel] = value;
      st->taxelMask[taxel >> 5] |= 1UL << (taxel & 31u);
      if (value > st->peak) st->peak = value;
      if (value >= trigger) {
        st->frontier[word] |= 1UL << (entry & 31u);
      }
    }
  }
}

// `entry` and every entry sharing its shear bump.
static void xampQueueWithBump(XampFrameContext *ctx, uint16_t entry) {
  const XampGrid *grid = ctx->grid;
  const uint16_t group = (grid->groupRadius && ctx->settings->bumpFill)
                             ? xampShearOf(grid, entry) : OTCFG_NO_SHEAR;
  if (group == OTCFG_NO_SHEAR) {
    xampQueue(ctx, entry);
    return;
  }
  const int16_t r0 = (int16_t)xampPosRow(grid->entryPos[entry]);
  const int16_t c0 = (int16_t)xampPosCol(grid->entryPos[entry]);
  const int16_t radius = (int16_t)grid->groupRadius;
  for (int16_t r = r0 - radius; r <= r0 + radius; ++r) {
    if (r < 0 || r >= (int16_t)XAMP_GRID_SIDE) continue;
    for (int16_t c = c0 - radius; c <= c0 + radius; ++c) {
      if (c < 0 || c >= (int16_t)XAMP_GRID_SIDE) continue;
      const uint16_t other = grid->gridEntry[xampPos((uint8_t)r, (uint8_t)c)];
      if (other != XAMP_NONE && xampShearOf(grid, other) == group) xampQueue(ctx, other);
    }
  }
}

// Breadth-first growth, one wave per batch: every trigger not yet grown queues
// its neighbours with their bumps (its own bump is among them), then the wave
// is read and its triggers form the next wave.
static void xampGrow(XampFrameContext *ctx) {
  XampState *st = ctx->state;
  const XampGrid *grid = ctx->grid;
  uint32_t wave[XAMP_WORDS];
  for (;;) {
    bool any = false;
    for (uint16_t w = 0; w < XAMP_WORDS; ++w) {
      wave[w] = st->frontier[w];
      st->frontier[w] = 0;
      any = any || wave[w] != 0u;
    }
    if (!any) return;

    const uint16_t before = st->count;
    for (uint16_t word = 0; word < XAMP_WORDS && st->count < st->limit; ++word) {
      uint32_t bits = wave[word];
      while (bits) {
        const uint16_t entry = (uint16_t)((word << 5) | (uint16_t)__builtin_ctz(bits));
        bits &= bits - 1u;
        const int16_t r0 = (int16_t)xampPosRow(grid->entryPos[entry]);
        const int16_t c0 = (int16_t)xampPosCol(grid->entryPos[entry]);
        for (int16_t r = r0 - 1; r <= r0 + 1; ++r) {
          if (r < 0 || r >= (int16_t)XAMP_GRID_SIDE) continue;
          for (int16_t c = c0 - 1; c <= c0 + 1; ++c) {
            if (c < 0 || c >= (int16_t)XAMP_GRID_SIDE) continue;
            const uint16_t near = grid->gridEntry[xampPos((uint8_t)r, (uint8_t)c)];
            if (near != XAMP_NONE) xampQueueWithBump(ctx, near);
          }
        }
      }
    }
    if (st->count == before) return;
    st->fromGrowth = (uint16_t)(st->fromGrowth + (st->count - before));
    xampReadBatch(ctx);
  }
}

// Pattern entry `index`, shifted as the sketches did: mod 32, whatever the
// skin's size. Off-skin positions map to NONE and are refused by xampQueue().
static inline uint16_t xampPatternEntry(const XampGrid *grid, const XampState *st,
                                        uint16_t index) {
  return grid->gridEntry[xampPos(
      (uint8_t)((XAMP_PATTERN_ROW[index] + st->shiftRow) & 0x1Fu),
      (uint8_t)((XAMP_PATTERN_COL[index] + st->shiftCol) & 0x1Fu))];
}

// One frame. On return state->sampled marks every entry read, and values[]
// holds their baseline-subtracted readings by taxel index. `noise` is each
// taxel's hands-off peak above baseline, or NULL to trigger on the margin alone.
static void xampFrame(const XampGrid *grid, const XampSettings *settings,
                      const uint16_t *baseline, const uint16_t *noise,
                      uint16_t *values, XampState *st, XampReadFn read,
                      XampRereadFn reread) {
  XampFrameContext ctx = {grid, settings, baseline, noise, values, st, read, reread};

  const bool fixed = settings->mode == XAMP_MODE_FIXED;
  uint16_t m = settings->m ? settings->m : 1u;
  if (m > grid->active) m = grid->active;
  st->limit = fixed ? m : grid->active;
  const uint16_t searchMin = settings->searchMin < st->limit ? settings->searchMin : st->limit;
  // GROW: M pattern samples (at least SEARCH_MIN); growth is on top of that.
  const uint16_t patternTarget = fixed ? st->limit : (m > searchMin ? m : searchMin);

  st->count = 0;
  st->peak = 0;
  st->fromPriority = st->fromPattern = st->fromGrowth = st->batches = st->confirms = 0;
  memset(st->sampled, 0, sizeof(st->sampled));
  memset(st->pending, 0, sizeof(st->pending));
  memset(st->frontier, 0, sizeof(st->frontier));
  memset(st->taxelMask, 0, sizeof(st->taxelMask));

  uint16_t previous[XAMP_MAX_PRIORITY];
  const uint8_t previousCount = st->priorityCount;
  memcpy(previous, st->priority, sizeof(previous[0]) * previousCount);
  st->priorityCount = 0;

  // Pass 1: last frame's seeds, grown before anything else, so every contact
  // they still touch is sampled before the search runs -- the sketches' order,
  // where a hit grew at once and later search points on it were skipped. In
  // FIXED they may not eat into the SEARCH_MIN reserve.
  const uint16_t limit = st->limit;
  if (fixed) st->limit = (uint16_t)(limit > searchMin ? limit - searchMin : 0u);
  for (uint8_t p = 0; p < previousCount; ++p) {
    if (xampQueue(&ctx, previous[p])) ++st->fromPriority;
  }
  xampReadBatch(&ctx);
  xampRememberPass(&ctx);
  xampGrow(&ctx);
  st->limit = limit;

  // Pass 2: the search -- M points in GROW, SEARCH_MIN in FIXED. A point
  // already sampled is skipped and costs nothing.
  const uint16_t firstSearch = fixed ? searchMin : patternTarget;
  uint16_t index = 0;
  while (st->fromPattern < firstSearch && st->count < st->limit &&
         index < XAMP_PATTERN_LENGTH) {
    if (xampQueue(&ctx, xampPatternEntry(grid, st, index++))) ++st->fromPattern;
  }
  xampReadBatch(&ctx);
  xampRememberPass(&ctx);
  xampGrow(&ctx);

  // Pass 3, FIXED only: the rest of M from the pattern. Its triggers grow next
  // frame, as seeds.
  if (fixed && st->count < st->limit) {
    while (st->count < st->limit && index < XAMP_PATTERN_LENGTH) {
      if (xampQueue(&ctx, xampPatternEntry(grid, st, index++))) ++st->fromPattern;
    }
    xampReadBatch(&ctx);
    xampRememberPass(&ctx);
    memset(st->frontier, 0, sizeof(st->frontier));
  }

  // Slide the whole pattern one taxel diagonally for next frame, with one extra
  // column step each time the row wraps, so all 1024 offsets still come round.
  // The sketches moved the row only when the column wrapped: the first M
  // pattern points sit on a coarse row lattice, so for 32 frames at a time the
  // same rows were searched and a press on any other waited for the next row
  // shift (simulated, 32x32 grow M=64: a 2x2 press waited 25 frames on
  // average, up to 158; diagonal, 2.8 and 14).
  st->shiftRow = (uint8_t)((st->shiftRow + 1u) & 0x1Fu);
  st->shiftCol = (uint8_t)((st->shiftCol + (st->shiftRow == 0u ? 2u : 1u)) & 0x1Fu);
}
