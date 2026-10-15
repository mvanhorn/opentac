// OpenTAC DeviceConfig -- the little the STM needs to know about the skin.
//
// One generic firmware plus one provisioned config equals a working sensor.
// Everything geometric (pixels, outline, bump centres, plotting) stays on the
// computer; what lands here is only: which taxels exist, how to electrically
// reach them, their spatial scale, and which of them share a shear bump.
//
// The config is written once into the top of flash and read IN PLACE from
// there -- flash is memory-mapped, so there is no RAM copy and no parser
// beyond bounds-checking a handful of fields.
//
// ----- Binary layout, little-endian, no padding, no struct casting ----------
//
//   off  size  field
//     0     4  magic 'O','T','C','1'
//     4     2  version                 (OTCFG_VERSION)
//     6     1  rows                    1..32, logical
//     7     1  cols                    1..32, logical
//     8     4  pitch_x_um              physical taxel pitch, micrometres
//    12     4  pitch_y_um
//    16     2  active_count            popcount(active_mask)
//    18     2  shear_count             0 (no shear) or == active_count
//    20     4  config_id               host's hash of the source manifest
//    24     4  payload_len             bytes between header and CRC
//    28        payload:
//                 u8  row_order[rows]     logical row -> connector line code
//                 u8  col_order[cols]     logical col -> connector line code
//                 u8  row_geom[rows]      logical row -> grid row, top to
//                                          bottom. 0xFF if the row holds no
//                                          taxels and so has no position.
//                 u8  col_geom[cols]      logical col -> grid col, left to
//                                          right. 0xFF as above.
//                 u8  active_mask[(rows*cols+7)/8]
//                                          bit (row*cols+col), LSB first
//                 u16 shear_group[shear_count]
//                                          parallel to the active taxels in
//                                          row-major order, i.e. to the
//                                          manifest's `pixels`. 0xFFFF = the
//                                          taxel has no bump above it.
//   28+payload_len
//           4  crc32 over bytes [0, 28+payload_len)
//
// Active taxels are enumerated row-major over set mask bits, which is exactly
// the order OpenTAC writes `pixels` and `shear.b` in -- so frame sample k,
// shear_group[k], pixels[k] and shear.b[k] all describe the same taxel.
//
// Logical row index is PAD order and is not geometric: OpenTAC's own manifest
// readme says so, and `pixels` confirms it -- the old example permutes rows as
// 0,1,2,5,4,3. So pitch alone does not place a taxel. row_geom/col_geom carry
// the permutation, and with them
//
//     x = col_geom[col] * pitch_x_um     (left to right)
//     y = row_geom[row] * pitch_y_um     (top to bottom)
//
// reproduces `pixels` exactly, up to the grid origin the host holds. That is
// what future xampling needs: sampling happens in physical space, which is not
// the space the shift register scans in. The scan stays in hardware order and
// the two spaces are related by these two small arrays.

#pragma once

#include <stdint.h>
#include <string.h>

#define OTCFG_MAGIC          0x3143544FUL   // 'O','T','C','1'
// 2: row_order/col_order are connector line codes, so either side of a taxel
//    may be the driven one. A v1 config is rejected rather than reinterpreted.
#define OTCFG_VERSION        2

#define OTCFG_MAX_ROWS       32   // logical limits; the hardware limits are
#define OTCFG_MAX_COLS       32   // OTCFG_SR_STEPS / OTCFG_MUX_CHANNELS below

// ----- Board ------------------------------------------------------------------
//
// One sketch, three boards, and a build is for exactly one of them
// (opentac_readout.ino says how to choose):
//   READOUT60  the 60-pin readout: four cascaded '595s and a 32:1 mux
//              (MUX0..MUX4), 30 of each on the connector, an I2C gain switch,
//              and its own '595 pin order (see the sketch's pinout).
//   NODE16     the 30-pin board: two '595s and a 16:1 mux (MUX0..MUX3 only;
//              PB5 is not part of it).
//   READOUT32  the 70-pin readout of the older boards: four '595s and a 32:1
//              mux.
// Pins, ADC, IMU and RGB are otherwise shared.
#if (defined(OT_BOARD_READOUT60) + defined(OT_BOARD_NODE16) + defined(OT_BOARD_READOUT32)) != 1
#error "Build for exactly one board: OT_BOARD_READOUT60, OT_BOARD_NODE16 or OT_BOARD_READOUT32 (see the top of opentac_readout.ino)"
#endif
#if defined(OT_BOARD_NODE16)
#define OTCFG_SR_STEPS       16
#define OTCFG_MUX_CHANNELS   16
#define OT_BOARD_NAME        "NODE16"
#elif defined(OT_BOARD_READOUT60)
#define OTCFG_SR_STEPS       32
#define OTCFG_MUX_CHANNELS   32
#define OT_BOARD_NAME        "READOUT60"
#else
#define OTCFG_SR_STEPS       32
#define OTCFG_MUX_CHANNELS   32
#define OT_BOARD_NAME        "READOUT32"
#endif
// A taxel is one (driven step, sensed mux channel) pair, and otCfgParse()
// requires every line to be distinct and to exist on this board -- so no
// config can hold more active taxels than steps x channels. That is 1024 on
// the readout, the same as rows x cols, and 256 on the node, which needs the
// RAM back for its bus buffers.
#define OTCFG_MAX_TAXELS     ((uint16_t)OTCFG_SR_STEPS * OTCFG_MUX_CHANNELS)
#define OTCFG_HEADER_BYTES   28u
#define OTCFG_CRC_BYTES      4u
#define OTCFG_NO_SHEAR       0xFFFFu
#define OTCFG_NO_POSITION    0xFFu

// Worst case: header + 4*32 + 128 + 2*1024 + crc.
#define OTCFG_MAX_BYTES      2336u

// Top two 2 KB pages of the G431KB's 128 KB. Two pages because the worst-case
// config does not fit in one, and configs are rare enough that the extra
// erase costs nothing that matters.
#define OTCFG_FLASH_PAGE     62u
#define OTCFG_FLASH_PAGES    2u
#define OTCFG_FLASH_ADDR     0x0801F000UL
#define OTCFG_FLASH_BYTES    4096UL

// ----- Little-endian byte-wise readers --------------------------------------
//
// Byte-wise rather than casting: the shear array's offset depends on rows,
// cols and the mask length, so it is not reliably even, and the host is a
// different compiler on a different architecture.

static inline uint16_t otCfgU16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static inline uint32_t otCfgU32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// CRC-32/ISO-HDLC, the zlib one. Bitwise: 2 KB at boot is microseconds and a
// 1 KB table is not worth the flash.
static uint32_t otCfgCrc32(const uint8_t *data, uint32_t length) {
  uint32_t crc = 0xFFFFFFFFUL;
  for (uint32_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ (0xEDB88320UL & (uint32_t)(-(int32_t)(crc & 1u)));
    }
  }
  return ~crc;
}

// ----- Connector line codes -------------------------------------------------
//
// Which side of a taxel is driven belongs to the READOUT BOARD's connector, not
// to the skin. On the G431 32x32 readout the manifest's COLUMNS land on the
// shift register and its ROWS on the mux -- the reverse of what the first
// version of this firmware assumed, which left half of a 31-row skin dead.
//
// So every logical row and column names a physical line:
//   0..31   shift-register step (driven)
//   32..63  mux channel (code - 32; sensed)
// A taxel is a resistor between its row line and its column line and reads the
// same either way round, but exactly one of the two must be driven and the
// other sensed. Two driven lines or two sensed lines cannot be read at all.

#define OTCFG_LINE_MUX       0x20u
#define OTCFG_LINE_CODES     64u

static inline bool otCfgLineIsMux(uint8_t code) {
  return (code & OTCFG_LINE_MUX) != 0u;
}

static inline uint8_t otCfgLineIndex(uint8_t code) { return (uint8_t)(code & 0x1Fu); }

// ----- Parsed view ----------------------------------------------------------

struct OtDeviceConfig {
  bool valid;
  uint16_t version;
  uint8_t rows;
  uint8_t cols;
  uint32_t pitchXUm;
  uint32_t pitchYUm;
  uint16_t activeCount;
  uint16_t shearCount;      // 0 when the sensor has no shear bumps
  uint32_t configId;
  uint32_t crc;
  uint32_t totalBytes;
  const uint8_t *rowOrder;
  const uint8_t *colOrder;
  const uint8_t *rowGeom;
  const uint8_t *colGeom;
  const uint8_t *activeMask;
  const uint8_t *shearGroup;  // NULL when shearCount == 0
};

static inline uint16_t otCfgMaskBytes(uint8_t rows, uint8_t cols) {
  return (uint16_t)(((uint16_t)rows * cols + 7u) / 8u);
}

// Every value below `limit`, no repeats. `allowNone` lets 0xFF appear any
// number of times, for the geometry arrays: a logical row holding no taxels
// has no position, which is not the same as sharing one.
static bool otCfgDistinct(const uint8_t *values, uint8_t count, uint8_t limit,
                          bool allowNone) {
  uint32_t seen = 0;
  for (uint8_t i = 0; i < count; ++i) {
    if (allowNone && values[i] == OTCFG_NO_POSITION) continue;
    if (values[i] >= limit) return false;
    const uint32_t bit = 1UL << values[i];
    if (seen & bit) return false;
    seen |= bit;
  }
  return true;
}

// Reject rather than repair: a config that does not add up means the host and
// the board disagree about the skin, and guessing which is right is exactly
// what this whole mechanism exists to avoid.
static bool otCfgParse(const uint8_t *raw, uint32_t available,
                       OtDeviceConfig *out) {
  memset(out, 0, sizeof(*out));
  if (available < OTCFG_HEADER_BYTES + OTCFG_CRC_BYTES) return false;
  if (otCfgU32(raw + 0) != OTCFG_MAGIC) return false;

  const uint16_t version = otCfgU16(raw + 4);
  if (version != OTCFG_VERSION) return false;

  const uint8_t rows = raw[6];
  const uint8_t cols = raw[7];
  if (rows == 0 || rows > OTCFG_MAX_ROWS) return false;
  if (cols == 0 || cols > OTCFG_MAX_COLS) return false;

  const uint16_t activeCount = otCfgU16(raw + 16);
  const uint16_t shearCount = otCfgU16(raw + 18);
  if (activeCount == 0 || activeCount > (uint16_t)rows * cols) return false;
  if (shearCount != 0 && shearCount != activeCount) return false;

  const uint32_t payloadLen = otCfgU32(raw + 24);
  const uint16_t maskBytes = otCfgMaskBytes(rows, cols);
  const uint32_t expected =
      2UL * rows + 2UL * cols + maskBytes + 2UL * shearCount;
  if (payloadLen != expected) return false;

  const uint32_t total = OTCFG_HEADER_BYTES + payloadLen + OTCFG_CRC_BYTES;
  if (total > available || total > OTCFG_MAX_BYTES) return false;

  const uint32_t stored = otCfgU32(raw + OTCFG_HEADER_BYTES + payloadLen);
  if (otCfgCrc32(raw, OTCFG_HEADER_BYTES + payloadLen) != stored) return false;

  const uint8_t *rowOrder = raw + OTCFG_HEADER_BYTES;
  const uint8_t *colOrder = rowOrder + rows;
  const uint8_t *rowGeom = colOrder + cols;
  const uint8_t *colGeom = rowGeom + rows;
  const uint8_t *mask = colGeom + cols;

  // A duplicate row would make two logical rows share one '595 stage, and a
  // duplicate column would read one pad twice: both scan, both are wrong.
  // Duplicated geometry is the same mistake in physical space: two rows at one
  // position would silently collapse under any future xampling.
  // No physical line can serve two logical rows/columns, across both lists.
  uint64_t seenLine = 0;
  for (uint8_t i = 0; i < (uint8_t)(rows + cols); ++i) {
    const uint8_t code = (i < rows) ? rowOrder[i] : colOrder[i - rows];
    if (code >= OTCFG_LINE_CODES) return false;
    // ...and it must exist on THIS board: a 16x16 node has no step 20.
    if (otCfgLineIndex(code) >=
        (otCfgLineIsMux(code) ? OTCFG_MUX_CHANNELS : OTCFG_SR_STEPS)) {
      return false;
    }
    const uint64_t bit = 1ULL << code;
    if (seenLine & bit) return false;
    seenLine |= bit;
  }
  if (!otCfgDistinct(rowGeom, rows, rows, true)) return false;
  if (!otCfgDistinct(colGeom, cols, cols, true)) return false;

  // Popcount must match, and the tail bits past rows*cols must be clear, or
  // activeCount and the mask describe different sensors.
  uint16_t counted = 0;
  for (uint16_t b = 0; b < maskBytes; ++b) {
    uint8_t byte = mask[b];
    while (byte) { counted += (byte & 1u); byte >>= 1; }
  }
  if (counted != activeCount) return false;
  const uint16_t tailBits = (uint16_t)(maskBytes * 8u - (uint16_t)rows * cols);
  if (tailBits != 0 &&
      (mask[maskBytes - 1] >> (8u - tailBits)) != 0u) {
    return false;
  }

  // Every active taxel must have one driven and one sensed line, or it could
  // never be read -- refuse the config rather than scan around a hole.
  for (uint8_t r = 0; r < rows; ++r) {
    for (uint8_t c = 0; c < cols; ++c) {
      const uint16_t bit = (uint16_t)r * cols + c;
      if (((mask[bit >> 3] >> (bit & 7u)) & 1u) == 0u) continue;
      if (otCfgLineIsMux(rowOrder[r]) == otCfgLineIsMux(colOrder[c])) return false;
    }
  }

  out->valid = true;
  out->version = version;
  out->rows = rows;
  out->cols = cols;
  out->pitchXUm = otCfgU32(raw + 8);
  out->pitchYUm = otCfgU32(raw + 12);
  out->activeCount = activeCount;
  out->shearCount = shearCount;
  out->configId = otCfgU32(raw + 20);
  out->crc = stored;
  out->totalBytes = total;
  out->rowOrder = rowOrder;
  out->colOrder = colOrder;
  out->rowGeom = rowGeom;
  out->colGeom = colGeom;
  out->activeMask = mask;
  out->shearGroup = shearCount ? (mask + maskBytes) : NULL;
  return true;
}

// ----- Runtime accessors ----------------------------------------------------

static inline uint8_t otCfgRowLine(const OtDeviceConfig *c, uint8_t row) {
  return c->rowOrder[row];
}

static inline uint8_t otCfgColLine(const OtDeviceConfig *c, uint8_t col) {
  return c->colOrder[col];
}

// Grid position, not scan position: the row the taxel physically sits on,
// counting from the top. Multiply by pitchYUm for a length. OTCFG_NO_POSITION
// when the logical row holds no taxels at all.
static inline uint8_t otCfgRowGridIndex(const OtDeviceConfig *c, uint8_t row) {
  return c->rowGeom[row];
}

static inline uint8_t otCfgColGridIndex(const OtDeviceConfig *c, uint8_t col) {
  return c->colGeom[col];
}

static inline bool otCfgIsActive(const OtDeviceConfig *c, uint8_t row,
                                 uint8_t col) {
  const uint16_t bit = (uint16_t)row * c->cols + col;
  return (c->activeMask[bit >> 3] >> (bit & 7u)) & 1u;
}

// `taxel` indexes the active taxels in row-major order, i.e. it is an index
// into the emitted frame and into the manifest's `pixels`.
// Whether this skin has shear bumps at all. A pressure-only skin is a
// different sensor for anything that reasons about shear -- notably xampling,
// which samples groups -- so this is asked directly rather than inferred from
// a count being non-zero at each call site.
static inline bool otCfgHasShear(const OtDeviceConfig *c) {
  return c->shearCount != 0;
}

static inline uint16_t otCfgShearGroup(const OtDeviceConfig *c,
                                       uint16_t taxel) {
  if (c->shearCount == 0 || taxel >= c->shearCount) return OTCFG_NO_SHEAR;
  return otCfgU16(c->shearGroup + 2u * (uint32_t)taxel);
}

// ----- Scan plan ------------------------------------------------------------

// Which shift-register step drives a taxel and which mux channel reads it,
// whichever of its row and column happens to be on which side.
static inline bool otCfgTaxelLines(const OtDeviceConfig *c, uint8_t row,
                                   uint8_t col, uint8_t *step, uint8_t *mux) {
  const uint8_t a = c->rowOrder[row];
  const uint8_t b = c->colOrder[col];
  if (otCfgLineIsMux(a) == otCfgLineIsMux(b)) return false;
  *step = otCfgLineIndex(otCfgLineIsMux(a) ? b : a);
  *mux = otCfgLineIndex(otCfgLineIsMux(a) ? a : b);
  return true;
}

// Everything the scan loop needs, flattened at boot so the inner loop touches
// no bitmask and no indirection.
//
// Organised by shift-register STEP, not by logical row: the walking 1 can only
// move forward, so the scan visits steps in order and, at each, reads every
// active taxel with a line on that step. Each entry says which mux channel to
// read and which frame slot the sample belongs in, and that slot is always the
// taxel's index in `pixels` order -- so it no longer matters which side of the
// skin the readout board happens to drive.
struct ScanPlan {
  uint8_t hwRowSteps;                      // '595 stages to clock through
  uint16_t stepStart[OTCFG_SR_STEPS + 1];  // per step, into entryMux/entryOut
  uint8_t entryMux[OTCFG_MAX_TAXELS];      // mux channel to read
  uint16_t entryOut[OTCFG_MAX_TAXELS];     // frame slot = index in `pixels`
};

// Two passes: count the taxels on each step, then place them. A step no taxel
// uses gets no entries and is clocked past without settling or sampling, and
// nothing past the last used step is clocked at all.
//
// `plan` NULL runs the counting pass only: the check a config gets before it is
// committed, without a second 3 KB ScanPlan on the stack. RAM is tight on the
// G431 once xampling's buffers are in.
static bool buildScanPlan(const OtDeviceConfig *config, ScanPlan *plan) {
  uint16_t count[OTCFG_SR_STEPS];
  memset(count, 0, sizeof(count));
  uint16_t counted = 0;
  for (uint8_t row = 0; row < config->rows; ++row) {
    for (uint8_t col = 0; col < config->cols; ++col) {
      if (!otCfgIsActive(config, row, col)) continue;
      uint8_t step, mux;
      if (!otCfgTaxelLines(config, row, col, &step, &mux)) return false;
      ++count[step];
      ++counted;
    }
  }
  if (plan == NULL) return counted == config->activeCount;

  plan->stepStart[0] = 0;
  uint8_t steps = 0;
  for (uint8_t s = 0; s < OTCFG_SR_STEPS; ++s) {
    plan->stepStart[s + 1] = (uint16_t)(plan->stepStart[s] + count[s]);
    if (count[s] != 0u) steps = (uint8_t)(s + 1u);
  }
  plan->hwRowSteps = steps;

  uint16_t fill[OTCFG_SR_STEPS];
  memcpy(fill, plan->stepStart, sizeof(fill));
  uint16_t taxel = 0;
  for (uint8_t row = 0; row < config->rows; ++row) {
    for (uint8_t col = 0; col < config->cols; ++col) {
      if (!otCfgIsActive(config, row, col)) continue;
      uint8_t step, mux;
      otCfgTaxelLines(config, row, col, &step, &mux);
      const uint16_t e = fill[step]++;
      plan->entryMux[e] = mux;
      plan->entryOut[e] = taxel++;
    }
  }
  // otCfgParse() already checked popcount, so this only fires if the two
  // passes disagree -- keep it as a cheap guard on the buffer bound.
  return taxel == config->activeCount;
}

// ----- Flash ----------------------------------------------------------------
//
// Guarded so the parser and accessors above can be compiled and tested on a
// host, where there is no HAL and no flash to speak of.
#ifndef OTCFG_NO_FLASH

static inline const uint8_t *otCfgFlashBase() {
  return (const uint8_t *)OTCFG_FLASH_ADDR;
}

// Why the last flash operation failed, so the reply can say more than "FLASH".
// The first hardware run failed with no way to tell erase from program from
// verify; this is what makes the next failure diagnosable from the host.
#define OTCFG_FLASH_OK       0
#define OTCFG_FLASH_ERASE    1
#define OTCFG_FLASH_PROGRAM  2
#define OTCFG_FLASH_VERIFY   3

struct OtCfgFlashDiag {
  uint8_t stage;       // OTCFG_FLASH_*
  uint32_t halError;   // HAL_FLASH_GetError()
  uint32_t sr;         // FLASH->SR at the failure
  uint32_t addr;       // failing address, or the erase page error
};
static OtCfgFlashDiag otCfgFlashDiag;

static void otCfgFlashFail(uint8_t stage, uint32_t addr) {
  otCfgFlashDiag.stage = stage;
  otCfgFlashDiag.halError = HAL_FLASH_GetError();
  otCfgFlashDiag.sr = FLASH->SR;
  otCfgFlashDiag.addr = addr;
}

// A G4 can come out of reset -- or out of the DFU bootloader -- with an error
// flag already latched in FLASH->SR, and HAL then refuses every erase and
// program until it is cleared. Nothing else in this firmware touches flash, so
// any flag present here is stale by definition.
static inline void otCfgFlashUnlockClean() {
  HAL_FLASH_Unlock();
  __HAL_FLASH_CLEAR_FLAG(FLASH_FLAG_ALL_ERRORS);
}

// Erase stalls the CPU until it finishes -- the G4 cannot fetch code while its
// single flash bank is being written -- so USB just NAKs for ~50 ms. That is
// why provisioning suspends the stream first.
static bool otCfgFlashErase() {
  FLASH_EraseInitTypeDef erase = {};
  erase.TypeErase = FLASH_TYPEERASE_PAGES;
  erase.Banks = FLASH_BANK_1;
  erase.Page = OTCFG_FLASH_PAGE;
  erase.NbPages = OTCFG_FLASH_PAGES;

  uint32_t pageError = 0;
  otCfgFlashUnlockClean();
  const bool ok = (HAL_FLASHEx_Erase(&erase, &pageError) == HAL_OK);
  if (!ok) otCfgFlashFail(OTCFG_FLASH_ERASE, pageError);
  HAL_FLASH_Lock();
  return ok;
}

static bool otCfgFlashWrite(const uint8_t *data, uint32_t length) {
  otCfgFlashDiag.stage = OTCFG_FLASH_OK;
  if (length == 0 || length > OTCFG_FLASH_BYTES) return false;
  if (!otCfgFlashErase()) return false;

  otCfgFlashUnlockClean();
  bool ok = true;
  // The G4 programs 64 bits at a time; the tail doubleword is padded with the
  // erased value so a short config leaves the rest of the page untouched.
  for (uint32_t off = 0; ok && off < length; off += 8) {
    uint8_t chunk[8];
    memset(chunk, 0xFF, sizeof(chunk));
    const uint32_t n = (length - off < 8u) ? (length - off) : 8u;
    memcpy(chunk, data + off, n);
    uint64_t word;
    memcpy(&word, chunk, sizeof(word));
    ok = (HAL_FLASH_Program(FLASH_TYPEPROGRAM_DOUBLEWORD,
                            OTCFG_FLASH_ADDR + off, word) == HAL_OK);
    if (!ok) otCfgFlashFail(OTCFG_FLASH_PROGRAM, OTCFG_FLASH_ADDR + off);
  }
  HAL_FLASH_Lock();
  if (!ok) return false;

  // Reset the data cache before reading back: a line cached from the erased
  // page would make a good write look like a failed one.
  __HAL_FLASH_DATA_CACHE_DISABLE();
  __HAL_FLASH_DATA_CACHE_RESET();
  __HAL_FLASH_DATA_CACHE_ENABLE();

  for (uint32_t i = 0; i < length; ++i) {
    if (otCfgFlashBase()[i] != data[i]) {
      otCfgFlashFail(OTCFG_FLASH_VERIFY, OTCFG_FLASH_ADDR + i);
      return false;
    }
  }
  return true;
}
#endif  // OTCFG_NO_FLASH
