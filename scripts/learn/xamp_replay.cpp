// Replay recorded raster frames through the firmware's xampling engine.
//
// Built and run by check_xampling.py, which compiles this file against the
// firmware folder (-I .../tactile_imu_fast_cfg) so the engine is ot_xampling.h
// itself, reached through the same read(step, mux) callback the board uses.
//
//   xamp_replay config.bin frames.txt > out.txt
//
// frames.txt: "n_taxels n_frames mode m search_min thresh", then a baseline
// line, a noise line, and one line of raw counts per xampling frame (taxels in
// manifest `pixels` order). Output: one line per frame,
// "from_priority from_pattern from_growth | taxel taxel ...", taxels ascending.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ot_devcfg.h"
#include "ot_xampling.h"

static ScanPlan plan;
static XampGrid grid;
static uint16_t lineTaxel[OTCFG_SR_STEPS][OTCFG_MUX_CHANNELS];
static std::vector<uint16_t> raw;
static uint16_t baseline[OTCFG_MAX_TAXELS], noise[OTCFG_MAX_TAXELS], values[OTCFG_MAX_TAXELS];
static uint16_t lastTaxel;

static uint16_t replayRead(uint8_t step, uint8_t mux) {
  lastTaxel = lineTaxel[step][mux];
  if (lastTaxel == XAMP_NONE) { std::fprintf(stderr, "unused line\n"); std::exit(1); }
  return raw[lastTaxel];
}
static uint16_t replayReread() { return raw[lastTaxel]; }

int main(int argc, char **argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s config.bin frames.txt\n", argv[0]); return 2; }
  static uint8_t blob[OTCFG_MAX_BYTES];
  FILE *f = std::fopen(argv[1], "rb");
  if (!f) return 2;
  const size_t nb = std::fread(blob, 1, sizeof(blob), f);
  std::fclose(f);
  OtDeviceConfig config;
  if (!otCfgParse(blob, (uint32_t)nb, &config)) { std::fprintf(stderr, "config\n"); return 1; }
  if (!buildScanPlan(&config, &plan) || !xampBuildGrid(&config, &plan, &grid)) {
    std::fprintf(stderr, "plan/grid refused\n");
    return 1;
  }
  for (auto &row : lineTaxel) for (auto &t : row) t = XAMP_NONE;
  uint8_t step = 0;
  for (uint16_t e = 0; e < config.activeCount; ++e) {
    while (plan.stepStart[step + 1u] <= e) ++step;
    lineTaxel[step][plan.entryMux[e]] = plan.entryOut[e];
  }

  FILE *in = std::fopen(argv[2], "r");
  if (!in) return 2;
  int n, frames, mode, m, searchMin, thresh;
  if (std::fscanf(in, "%d %d %d %d %d %d", &n, &frames, &mode, &m, &searchMin, &thresh) != 6) return 3;
  if (n != config.activeCount) { std::fprintf(stderr, "taxel count\n"); return 3; }
  for (int k = 0; k < n; ++k) { int v; std::fscanf(in, "%d", &v); baseline[k] = (uint16_t)v; }
  for (int k = 0; k < n; ++k) { int v; std::fscanf(in, "%d", &v); noise[k] = (uint16_t)v; }
  raw.assign(n, 0);

  XampSettings s = {(uint8_t)mode, (uint16_t)m, (uint16_t)thresh, (uint16_t)searchMin, false};
  XampState st;
  xampReset(&st);
  for (int fr = 0; fr < frames; ++fr) {
    for (int k = 0; k < n; ++k) { int v; std::fscanf(in, "%d", &v); raw[k] = (uint16_t)v; }
    xampFrame(&grid, &s, baseline, noise, values, &st, replayRead, replayReread);
    std::printf("%u %u %u |", st.fromPriority, st.fromPattern, st.fromGrowth);
    for (int k = 0; k < n; ++k)
      if ((st.taxelMask[k >> 5] >> (k & 31)) & 1u) std::printf(" %d", k);
    std::printf("\n");
  }
  std::fclose(in);
  return 0;
}
