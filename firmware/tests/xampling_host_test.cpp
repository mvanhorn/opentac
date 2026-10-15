// Host test for ot_xampling.h, run against real provisioned configs.
//
//   python3 ../opentac_device.py provision ../../manifest_32x32.json --dry-run --out /tmp/c32.bin
//   c++ -std=c++17 -O2 -DOTCFG_NO_FLASH -I.. xampling_host_test.cpp -o xtest && ./xtest /tmp/c32.bin
//
// NODE16 configs need -DOT_BOARD_NODE16, the same as the firmware build.
//
// The sensor is simulated behind the same read(step, mux) call the firmware
// makes, so the engine under test is the code that runs on the board.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

#include "ot_devcfg.h"
#include "ot_xampling.h"

static int failures = 0;
#define CHECK(cond, ...)                                              \
  do {                                                                \
    if (!(cond)) {                                                    \
      ++failures;                                                     \
      std::printf("  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
      std::printf(__VA_ARGS__);                                       \
      std::printf("\n");                                              \
    }                                                                 \
  } while (0)

// ----- Simulated sensor ------------------------------------------------------

static const XampGrid *simGrid;
static uint16_t lineTaxel[OTCFG_SR_STEPS][OTCFG_MUX_CHANNELS];
static uint16_t simRaw[OTCFG_MAX_TAXELS];
static std::vector<uint16_t> readLog;   // taxel per read, this frame
static std::vector<uint8_t> stepLog;

static uint16_t simRead(uint8_t step, uint8_t mux) {
  const uint16_t taxel = lineTaxel[step][mux];
  if (taxel == XAMP_NONE) {
    std::printf("  FAIL read of an unused line: step %u mux %u\n", step, mux);
    ++failures;
    return 0;
  }
  readLog.push_back(taxel);
  stepLog.push_back(step);
  return simRaw[taxel];
}

static std::vector<uint16_t> taxelPos;      // taxel -> grid position
static std::vector<uint16_t> taxelEntry;    // taxel -> scan-plan entry
static ScanPlan plan;

static void setupSensor(const OtDeviceConfig *config, const XampGrid *grid) {
  simGrid = grid;
  for (auto &row : lineTaxel) for (auto &t : row) t = XAMP_NONE;
  taxelPos.assign(config->activeCount, XAMP_NONE);
  taxelEntry.assign(config->activeCount, XAMP_NONE);
  uint8_t step = 0;
  for (uint16_t e = 0; e < config->activeCount; ++e) {
    while (plan.stepStart[step + 1u] <= e) ++step;
    const uint16_t taxel = plan.entryOut[e];
    CHECK(taxelEntry[taxel] == XAMP_NONE, "taxel %u in two entries", taxel);
    taxelEntry[taxel] = e;
    taxelPos[taxel] = grid->entryPos[e];
    CHECK(grid->gridEntry[grid->entryPos[e]] == e, "entry %u does not round-trip", e);
    CHECK(lineTaxel[step][plan.entryMux[e]] == XAMP_NONE, "two taxels on one line pair");
    lineTaxel[step][plan.entryMux[e]] = taxel;
    // Cross-check the grid position against the config's own geometry.
    uint16_t k = 0;
    for (uint8_t r = 0; r < config->rows; ++r) {
      for (uint8_t c = 0; c < config->cols; ++c) {
        if (!otCfgIsActive(config, r, c)) continue;
        if (k++ == taxel) {
          CHECK(taxelPos[taxel] == xampPos(otCfgRowGridIndex(config, r), otCfgColGridIndex(config, c)),
                "taxel %u at the wrong grid position", taxel);
        }
      }
    }
  }
  uint16_t cells = 0;
  for (uint16_t pos = 0; pos < XAMP_GRID_CELLS; ++pos) cells += grid->gridEntry[pos] != XAMP_NONE;
  CHECK(cells == config->activeCount, "%u grid cells for %u taxels", cells, config->activeCount);
}

static uint16_t gridRows(const XampGrid *grid) {
  uint16_t rows = 0;
  for (uint16_t pos = 0; pos < XAMP_GRID_CELLS; ++pos)
    if (grid->gridEntry[pos] != XAMP_NONE && xampPosRow(pos) + 1u > rows) rows = xampPosRow(pos) + 1u;
  return rows;
}
static uint16_t gridCols(const XampGrid *grid) {
  uint16_t cols = 0;
  for (uint16_t pos = 0; pos < XAMP_GRID_CELLS; ++pos)
    if (grid->gridEntry[pos] != XAMP_NONE && xampPosCol(pos) + 1u > cols) cols = xampPosCol(pos) + 1u;
  return cols;
}

// Baseline noise floor plus Gaussian-ish contacts, in counts above baseline.
struct Contact { double row, col, radius, peak; };

static uint16_t baseline[OTCFG_MAX_TAXELS];
static const uint16_t *simNoise = nullptr;     // per-taxel noise; null = none
static uint16_t values[OTCFG_MAX_TAXELS];
static uint16_t truth[OTCFG_MAX_TAXELS];   // counts above baseline

static void paint(uint16_t active, const std::vector<Contact> &contacts) {
  for (uint16_t k = 0; k < active; ++k) {
    const double r = xampPosRow(taxelPos[k]), c = xampPosCol(taxelPos[k]);
    double v = 0;
    for (const Contact &ct : contacts) {
      const double d2 = (r - ct.row) * (r - ct.row) + (c - ct.col) * (c - ct.col);
      if (d2 <= ct.radius * ct.radius) v += ct.peak * (1.0 - 0.5 * d2 / (ct.radius * ct.radius));
    }
    if (v > 3000) v = 3000;
    truth[k] = (uint16_t)v;
    simRaw[k] = (uint16_t)(baseline[k] + truth[k]);
  }
}

struct FrameStats { uint16_t count; uint16_t resets; uint16_t stepChanges; };

static bool sampledTaxel(const XampState *st, uint16_t taxel) {
  return xampWasSampled(st, taxelEntry[taxel]);
}

static bool simBumpFill = true;
static XampRereadFn simReread = nullptr;

static FrameStats runFrame(const XampGrid *grid, XampState *st, uint8_t mode, uint16_t m,
                           uint16_t threshold, uint16_t searchMin) {
  readLog.clear();
  stepLog.clear();
  XampSettings s = {mode, m, threshold, searchMin, simBumpFill};
  xampFrame(grid, &s, baseline, simNoise, values, st, simRead, simReread);

  FrameStats fs = {st->count, 0, 0};
  for (size_t i = 1; i < stepLog.size(); ++i) {
    if (stepLog[i] != stepLog[i - 1]) ++fs.stepChanges;
    if (stepLog[i] < stepLog[i - 1]) ++fs.resets;
  }

  // Invariants every frame must hold.
  CHECK(readLog.size() == st->count, "%zu reads for count %u", readLog.size(), st->count);
  CHECK(st->fromPriority + st->fromPattern + st->fromGrowth == st->count,
        "origins %u+%u+%u != %u", st->fromPriority, st->fromPattern, st->fromGrowth, st->count);
  // The walk only rewinds between batches: each batch is one forward pass.
  CHECK(fs.resets < st->batches || st->count == 0, "%u rewinds in %u batches", fs.resets, st->batches);
  std::set<uint16_t> readSet(readLog.begin(), readLog.end());
  CHECK(readSet.size() == readLog.size(), "a taxel was read twice");
  uint16_t marked = 0;
  for (uint16_t k = 0; k < grid->active; ++k) {
    if (!sampledTaxel(st, k)) continue;
    ++marked;
    CHECK(readSet.count(k), "taxel %u marked sampled but not read", k);
    CHECK(values[k] == truth[k], "taxel %u value %u, truth %u", k, values[k], truth[k]);
  }
  CHECK(marked == st->count, "%u marked for count %u", marked, st->count);
  uint16_t peak = 0;
  for (uint16_t k = 0; k < grid->active; ++k) {
    CHECK((((st->taxelMask[k >> 5] >> (k & 31u)) & 1u) != 0u) == sampledTaxel(st, k), "taxel mask bit %u", k);
    if (sampledTaxel(st, k) && values[k] > peak) peak = values[k];
  }
  CHECK(peak == st->peak, "peak %u, recorded %u", peak, st->peak);
  return fs;
}

// ----- Tests ------------------------------------------------------------------

static void testPattern(const XampGrid *grid) {
  // The sketches' table must be a permutation of all 1024 positions, with the
  // level ends where the batching assumes.
  std::set<uint16_t> seen;
  for (uint16_t i = 0; i < XAMP_PATTERN_LENGTH; ++i) {
    CHECK(seen.insert(xampPos(XAMP_PATTERN_ROW[i], XAMP_PATTERN_COL[i])).second,
          "table repeats at %u", i);
  }
  CHECK(seen.size() == 1024u, "table covers %zu positions", seen.size());
  CHECK(XAMP_PATTERN_ROW[0] == 15 && XAMP_PATTERN_COL[0] == 15, "table starts at (15,15)");
  uint16_t onSkin = 0;
  for (uint16_t i = 0; i < XAMP_PATTERN_LENGTH; ++i) {
    onSkin += grid->gridEntry[xampPos(XAMP_PATTERN_ROW[i], XAMP_PATTERN_COL[i])] != XAMP_NONE;
  }
  std::printf("  pattern: %u of 1024 unshifted entries land on this skin\n", onSkin);
}

static void testNoContact(const XampGrid *grid, uint16_t active) {
  paint(active, {});
  const uint16_t ms[] = {1, 10, 30, 64, 128, 500, active};
  for (uint8_t mode : {XAMP_MODE_FIXED, XAMP_MODE_GROW}) {
    for (uint16_t m : ms) {
      if (m > active) continue;
      XampState st;
      xampReset(&st);
      FrameStats fs = runFrame(grid, &st, mode, m, 150, 30);
      const uint16_t want = mode == XAMP_MODE_FIXED ? m : (m > 30 ? m : (active < 30 ? active : 30));
      CHECK(fs.count == want, "mode %u M %u: %u samples, want %u", mode, m, fs.count, want);
      CHECK(st.fromGrowth == 0, "growth with nothing touching");
    }
  }
  // Full budget reads every taxel, each once, with one forward walk per level.
  XampState st;
  xampReset(&st);
  FrameStats fs = runFrame(grid, &st, XAMP_MODE_FIXED, active, 150, 30);
  CHECK(fs.count == active, "full budget sampled %u of %u", fs.count, active);
  CHECK(fs.resets < 4u, "%u walk resets for %u levels", fs.resets,
        4u);
  std::printf("  full frame: %u samples, %u batches, %u row changes, %u walk resets\n", fs.count,
              st.batches, fs.stepChanges, fs.resets);
}

static void checkGrowthComplete(const XampGrid *grid, const XampState *st, uint16_t active,
                                uint16_t threshold, const char *label) {
  // Every sampled taxel at/above threshold has its bump and its 8 neighbours'
  // bumps sampled.
  uint16_t missing = 0;
  for (uint16_t k = 0; k < active; ++k) {
    if (!sampledTaxel(st, k) || truth[k] < threshold) continue;
    const int r0 = xampPosRow(taxelPos[k]), c0 = xampPosCol(taxelPos[k]);
    for (int r = r0 - 1; r <= r0 + 1; ++r) {
      for (int c = c0 - 1; c <= c0 + 1; ++c) {
        if (r < 0 || c < 0 || r >= 32 || c >= 32) continue;
        const uint16_t e = grid->gridEntry[xampPos((uint8_t)r, (uint8_t)c)];
        if (e == XAMP_NONE) continue;
        const uint16_t n = plan.entryOut[e];
        if (!sampledTaxel(st, n)) ++missing;
        if (grid->groupRadius && simBumpFill) {
          const uint16_t g = otCfgShearGroup(grid->config, n);
          if (g == OTCFG_NO_SHEAR) continue;
          for (uint16_t j = 0; j < active; ++j) {
            if (otCfgShearGroup(grid->config, j) == g && !sampledTaxel(st, j)) ++missing;
          }
        }
      }
    }
  }
  CHECK(missing == 0, "%s: %u neighbour/bump samples missing", label, missing);
}

static void testContacts(const XampGrid *grid, uint16_t active) {
  const double R = gridRows(grid), C = gridCols(grid);
  const uint16_t thr = 150;

  // A large contact plus a small one, in GROW with a small M.
  paint(active, {{R * 0.3, C * 0.5, (R < C ? R : C) * 0.25 + 1.0, 1500},
                 {R * 0.8, C * 0.2, 1.0, 900}});
  uint16_t hot = 0;
  for (uint16_t k = 0; k < active; ++k) hot += truth[k] >= thr;

  XampState st;
  xampReset(&st);
  uint16_t foundBoth = 0xFFFF;
  for (uint16_t f = 0; f < 1100u && foundBoth == 0xFFFF; ++f) {
    runFrame(grid, &st, XAMP_MODE_GROW, 30, thr, 30);
    checkGrowthComplete(grid, &st, active, thr, "grow");
    uint16_t covered = 0;
    for (uint16_t k = 0; k < active; ++k) covered += truth[k] >= thr && sampledTaxel(&st, k);
    if (covered == hot) foundBoth = f;
  }
  CHECK(foundBoth != 0xFFFF, "GROW never covered every taxel above threshold");
  FrameStats fs = runFrame(grid, &st, XAMP_MODE_GROW, 30, thr, 30);
  std::printf("  grow M=30: %u taxels above thr, all covered by frame %u; then %u samples "
              "(%u priority, %u pattern, %u growth), %u row changes\n",
              hot, foundBoth, fs.count, st.fromPriority, st.fromPattern, st.fromGrowth,
              fs.stepChanges);
  CHECK(st.fromPriority > 0, "no priority samples after contacts were found");

  // FIXED: never exceeds M, and SEARCH_MIN stays searched even when last
  // frame's triggers alone would fill the budget.
  for (uint16_t m : {16, 40, 64, 128}) {
    if (m > active) continue;
    XampState fx;
    xampReset(&fx);
    for (int f = 0; f < 20; ++f) {
      fs = runFrame(grid, &fx, XAMP_MODE_FIXED, m, thr, 30);
      CHECK(fs.count <= m, "FIXED M %u sampled %u", m, fs.count);
      const uint16_t wantSearch = m < 30 ? m : 30;
      CHECK(fx.fromPattern >= wantSearch || fx.fromPattern + fx.fromPriority + fx.fromGrowth == m,
            "FIXED M %u frame %d: only %u pattern samples", m, f, fx.fromPattern);
      CHECK(fx.fromPriority <= (m > 30 ? m - 30 : 0), "priority %u overran the search reserve",
            fx.fromPriority);
    }
    std::printf("  fixed M=%u: %u samples (%u priority, %u pattern, %u growth)\n", m, fs.count,
                fx.fromPriority, fx.fromPattern, fx.fromGrowth);
  }

  // A single-taxel press between coarse lattice points is found by the shift.
  const double pr = R > 2 ? 2 : 0, pc = C > 2 ? 1 : 0;
  paint(active, {{pr, pc, 0.5, 800}});
  xampReset(&st);
  uint16_t foundAt = 0xFFFF;
  for (uint16_t f = 0; f < 1100u; ++f) {
    runFrame(grid, &st, XAMP_MODE_FIXED, 16, thr, 16);
    for (uint16_t k = 0; k < active; ++k) {
      if (truth[k] >= thr && sampledTaxel(&st, k)) { foundAt = f; break; }
    }
    if (foundAt != 0xFFFF) break;
  }
  if (grid->gridEntry[xampPos((uint8_t)pr, (uint8_t)pc)] != XAMP_NONE) {
    CHECK(foundAt != 0xFFFF, "single-taxel press never found at M=16");
    std::printf("  single-taxel press at (%g,%g) found at M=16 in frame %u\n", pr, pc, foundAt);
  }
}

// A taxel's own noise raises its trigger: a contact inside a noisy region is
// read by the search but does not grow, while the same contact elsewhere does.
static void testPerTaxelNoise(const XampGrid *grid, uint16_t active) {
  static uint16_t noise[OTCFG_MAX_TAXELS];
  const uint16_t centre = taxelPos[active / 2];   // on the skin, whatever its shape
  paint(active, {{(double)xampPosRow(centre), (double)xampPosCol(centre), 2.5, 600}});
  for (uint16_t k = 0; k < active; ++k) noise[k] = truth[k] ? 1000 : 5;
  simNoise = noise;
  XampState st;
  xampReset(&st);
  runFrame(grid, &st, XAMP_MODE_FIXED, active, 30, 30);
  CHECK(st.fromGrowth == 0 && st.priorityCount == 0,
        "noisy taxels triggered: %u growth, %u priority", st.fromGrowth, st.priorityCount);
  for (uint16_t k = 0; k < active; ++k) noise[k] = 5;
  xampReset(&st);
  runFrame(grid, &st, XAMP_MODE_GROW, 30, 30, 30);
  bool found = st.priorityCount > 0;
  for (int f = 0; f < 1100 && !found; ++f) {
    runFrame(grid, &st, XAMP_MODE_GROW, 30, 30, 30);
    found = st.priorityCount > 0;
  }
  CHECK(found, "quiet taxels never triggered on a 600-count contact");
  simNoise = nullptr;
}

// Hot start under load: many contacts held at a low threshold. Once a contact
// is found it must be read every frame after, however many search hits the
// earlier contacts have accumulated.
static void testManyContactsStayFound(const XampGrid *grid, uint16_t active) {
  const double R = gridRows(grid), C = gridCols(grid);
  if (R < 12 || C < 12) return;
  std::vector<Contact> contacts;
  for (int i = 0; i < 8; ++i) contacts.push_back({2.0 + (R - 4) * ((i * 37) % 8) / 8.0, 2.0 + (C - 4) * i / 8.0, 2.0, 1500});
  paint(active, contacts);
  XampState st;
  xampReset(&st);
  std::vector<int> foundAt(contacts.size(), -1);
  uint16_t lost = 0;
  for (int f = 0; f < 400; ++f) {
    runFrame(grid, &st, XAMP_MODE_GROW, 64, 100, 30);
    for (size_t i = 0; i < contacts.size(); ++i) {
      bool got = false;
      for (uint16_t k = 0; k < active; ++k) {
        const double r = xampPosRow(taxelPos[k]) - contacts[i].row, c = xampPosCol(taxelPos[k]) - contacts[i].col;
        if (r * r + c * c <= contacts[i].radius * contacts[i].radius && truth[k] >= 100 && sampledTaxel(&st, k)) got = true;
      }
      if (got && foundAt[i] < 0) foundAt[i] = f;
      if (!got && foundAt[i] >= 0) ++lost;
    }
  }
  for (size_t i = 0; i < contacts.size(); ++i) {
    bool onSkin = false;   // a contact in a hole of a shaped skin presses nothing
    for (uint16_t k = 0; k < active; ++k) {
      const double r = xampPosRow(taxelPos[k]) - contacts[i].row, c = xampPosCol(taxelPos[k]) - contacts[i].col;
      onSkin = onSkin || (r * r + c * c <= contacts[i].radius * contacts[i].radius && truth[k] >= 100);
    }
    if (onSkin) CHECK(foundAt[i] >= 0, "contact %zu never found", i);
  }
  CHECK(lost == 0, "found contacts went missing in %u frames (priority %u/50)", lost, st.priorityCount);
}

// XAMP_BUMP_FILL: the same frame from the same state, with and without. Off,
// growth reads only neighbours; on, it also reads their bumps, so more.
static void testBumpFillOff(const XampGrid *grid, uint16_t active) {
  if (!grid->groupRadius) return;
  // One taxel with all 8 neighbours on the skin, so they span several bumps.
  uint16_t chosen = XAMP_NONE;
  for (uint16_t k = 0; k < active && chosen == XAMP_NONE; ++k) {
    const int r0 = xampPosRow(taxelPos[(k + active / 2) % active]), c0 = xampPosCol(taxelPos[(k + active / 2) % active]);
    int n = 0;
    for (int r = r0 - 1; r <= r0 + 1; ++r)
      for (int c = c0 - 1; c <= c0 + 1; ++c)
        n += r >= 0 && c >= 0 && r < 32 && c < 32 && grid->gridEntry[xampPos((uint8_t)r, (uint8_t)c)] != XAMP_NONE;
    if (n == 9) chosen = (uint16_t)((k + active / 2) % active);
  }
  if (chosen == XAMP_NONE) return;
  const uint16_t centre = taxelPos[chosen];
  paint(active, {{(double)xampPosRow(centre), (double)xampPosCol(centre), 0.5, 1500}});
  CHECK(truth[chosen] >= 100, "test press missed its taxel");
  XampState st;
  xampReset(&st);
  for (int f = 0; f < 1100 && st.priorityCount == 0; ++f) runFrame(grid, &st, XAMP_MODE_GROW, 30, 100, 30);
  CHECK(st.priorityCount > 0, "press never found");
  XampState off = st, on = st;                  // next frame grows from the seed
  simBumpFill = false;
  runFrame(grid, &off, XAMP_MODE_GROW, 30, 100, 30);
  checkGrowthComplete(grid, &off, active, 100, "bump fill off");
  simBumpFill = true;
  runFrame(grid, &on, XAMP_MODE_GROW, 30, 100, 30);
  checkGrowthComplete(grid, &on, active, 100, "bump fill on");
  CHECK(off.fromGrowth <= 8u, "bump fill off grew %u samples from one taxel", off.fromGrowth);
  CHECK(on.fromGrowth > off.fromGrowth, "bump fill on grew %u, off %u", on.fromGrowth, off.fromGrowth);
}

// Settle-and-confirm: a spike that is gone on the second reading must not
// trigger, and a press that stays must.
static uint16_t spikeTaxel = XAMP_NONE;
static uint16_t lastReadTaxel = XAMP_NONE;
static uint16_t spikyRead(uint8_t step, uint8_t mux) {
  lastReadTaxel = lineTaxel[step][mux];
  return (lastReadTaxel == spikeTaxel) ? (uint16_t)(baseline[lastReadTaxel] + 900u) : simRead(step, mux);
}
static uint16_t settledReread() {
  return (lastReadTaxel == spikeTaxel) ? baseline[lastReadTaxel] : (uint16_t)(baseline[lastReadTaxel] + truth[lastReadTaxel]);
}
static void testConfirm(const XampGrid *grid, uint16_t active) {
  paint(active, {});
  XampSettings s = {XAMP_MODE_FIXED, active, 100, 30, true};
  XampState st;
  spikeTaxel = (uint16_t)(active / 3);
  xampReset(&st);
  xampFrame(grid, &s, baseline, nullptr, values, &st, spikyRead, nullptr);
  CHECK(st.fromGrowth > 0 || st.priorityCount > 0, "unconfirmed spike did not trigger (test is not testing)");
  xampReset(&st);
  xampFrame(grid, &s, baseline, nullptr, values, &st, spikyRead, settledReread);
  CHECK(st.priorityCount == 0 && st.confirms == 1, "confirmed spike: priority %u, confirms %u", st.priorityCount, st.confirms);
  CHECK(values[spikeTaxel] == 0, "confirmed spike stored %u", values[spikeTaxel]);
  spikeTaxel = XAMP_NONE;
  const uint16_t centre = taxelPos[active / 2];
  paint(active, {{(double)xampPosRow(centre), (double)xampPosCol(centre), 1.5, 1500}});
  xampReset(&st);
  xampFrame(grid, &s, baseline, nullptr, values, &st, spikyRead, settledReread);
  CHECK(st.priorityCount > 0, "a real press was confirmed away");
  paint(active, {});
}

static void testOrderIsStepSorted(const XampGrid *grid, uint16_t active) {
  // Pure search: reads within one level must never walk backwards more than
  // once per level.
  paint(active, {});
  XampState st;
  xampReset(&st);
  FrameStats fs = runFrame(grid, &st, XAMP_MODE_FIXED, active < 128 ? active : 128, 150, 30);
  CHECK(fs.resets < 4u, "%u walk resets", fs.resets);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::printf("usage: %s config.bin ...\n", argv[0]);
    return 2;
  }
  srand(1);
  for (int a = 1; a < argc; ++a) {
    FILE *f = std::fopen(argv[a], "rb");
    if (!f) { std::printf("cannot open %s\n", argv[a]); return 2; }
    static uint8_t blob[OTCFG_MAX_BYTES];
    const size_t n = std::fread(blob, 1, sizeof(blob), f);
    std::fclose(f);

    OtDeviceConfig config;
    std::printf("%s\n", argv[a]);
    if (!otCfgParse(blob, (uint32_t)n, &config)) {
      std::printf("  FAIL config does not parse\n");
      ++failures;
      continue;
    }
    static XampGrid grid;
    CHECK(buildScanPlan(&config, &plan), "scan plan refused");
    const bool ok = xampBuildGrid(&config, &plan, &grid);
    CHECK(ok, "grid refused");
    if (!ok) continue;
    std::printf("  %ux%u logical, %ux%u grid, %u taxels, bump radius %u\n",
                config.rows, config.cols, gridRows(&grid), gridCols(&grid), grid.active, grid.groupRadius);

    for (uint16_t k = 0; k < config.activeCount; ++k) baseline[k] = (uint16_t)(rand() % 600);
    setupSensor(&config, &grid);
    testPattern(&grid);
    testNoContact(&grid, config.activeCount);
    testContacts(&grid, config.activeCount);
    testOrderIsStepSorted(&grid, config.activeCount);
    testPerTaxelNoise(&grid, config.activeCount);
    testManyContactsStayFound(&grid, config.activeCount);
    testBumpFillOff(&grid, config.activeCount);
    testConfirm(&grid, config.activeCount);
  }
  std::printf(failures ? "\n%d FAILURES\n" : "\nall passed\n", failures);
  return failures ? 1 : 0;
}
