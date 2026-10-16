// OpenTac bus v2 -- one Teensy 4.0 controller, 1..18 sensor nodes, one I2C bus.
//
// A node is not a different kind of board. It runs the single-board firmware
// (stm_test/tactile_imu_fast_cfg) built with -DOT_BUS_NODE, so it is
// provisioned from a manifest, configured with the same parameters and
// xampled exactly as a board on USB is. Only the transport changes. The bus
// carries three things:
//
//   1. Addressing. SMBus ARP as i2c_smoke validated it on three nodes and v1
//      kept it: every node answers the Device Default Address 0x61, the
//      controller hands out 0x10 upward, and a node that reboots is recognised
//      by the CRC-32 of its 96-bit STM32 UID and given its old slot back.
//   2. Text. The board's own command lines, and its replies, carried as byte
//      streams in both directions. So there is ONE command surface -- ot_api.h
//      -- whether a board is on USB or on the bus, and the host library drives
//      both through the same code.
//   3. Frames. Binary, because the bus is the bottleneck: a 12-bit sample is
//      1.5 bytes here and ~5 as text. The controller turns each frame back into
//      the exact F / X / V records a USB board prints, so everything above the
//      controller parses one format.
//
// What is deliberately NOT carried over from v1: the Haar codec, the
// bandwidth scheduler and the tonic sweep. They assume a fixed 16x16 skin
// compiled into the node, and v2 skins come from manifests. v1 stays in
// stm_test/v1 for reference.
//
// Every field is little-endian and written byte by byte, never by casting a
// struct: the two ends are different compilers on different architectures.
//
// Hardware-free. The node, the controller and v2/tests/bus_host_test.cpp all
// include this file, so the test drives the same code both ends run.

#pragma once

#include <stdint.h>
#include <string.h>

#if defined(__cplusplus)
#define OTB_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define OTB_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif

// ----- Versions and addresses -------------------------------------------------

// Carried in the INFO frame. v1 nodes report 4 there, and a different INFO
// magic besides, so neither generation of controller adopts the other's node.
#define OTB_VERSION        5
#define OTB_BUS_HZ         1000000UL

#define OTB_ADDR_ARP       0x61     // SMBus Device Default Address
#define OTB_ADDR_BASE      0x10     // first assigned address
#define OTB_MAX_NODES      18

// ----- SMBus ARP (unchanged bytes from i2c_smoke / v1) -------------------------

#define OTB_ARP_PREPARE    0x01     // clear every node's Address Resolved flag
#define OTB_ARP_RESET      0x02     // clear AR and drop assigned addresses
#define OTB_ARP_GET_UDID   0x03     // block read, arbitrated; lowest UDID wins
#define OTB_ARP_ASSIGN     0x04     // block write of UDID + address

#define OTB_UDID_LEN       16
#define OTB_ARP_BLOCK_LEN  17       // UDID + the device's slave address byte
#define OTB_ARP_READ_LEN   (1 + OTB_ARP_BLOCK_LEN + 1)
#define OTB_ARP_WRITE_LEN  (1 + 1 + OTB_ARP_BLOCK_LEN + 1)
#define OTB_UDID_SPECIFIC  12       // 32-bit: CRC-32 of the STM32 96-bit UID
#define OTB_UID_BYTES      12

// ----- Commands (0x40 block: v1 never used it) --------------------------------

#define OTB_CMD_STATUS     0x40     // r: OTB_STATUS_LEN
#define OTB_CMD_TEXT       0x41     // w: off u16, len, bytes, PEC
#define OTB_CMD_GET_TEXT   0x42     // w: ack u16, max, PEC; r: off u16, len, max bytes, PEC
#define OTB_CMD_GET_FRAME  0x43     // w: chunk, PEC;         r: OTB_CHUNK_RESP_LEN
// A plain read with no command before it returns INFO, as in v1.

// ----- Sizes ------------------------------------------------------------------
//
// Two buffers bound every transaction: the Teensy 4 Wire buffer (136 B) and
// the STM32duino slave buffer, which the node build raises to 160 B with
// -DI2C_TXRX_BUFFER_SIZE=160 (the default of 32 truncates silently).

#define OTB_WIRE_LIMIT     128
#define OTB_TEXT_MAX       96       // text bytes per TEXT write or GET_TEXT read
#define OTB_TEXT_WRITE_LEN(n) (1 + 2 + 1 + (n) + 1)
#define OTB_GET_TEXT_REQ_LEN  5
#define OTB_GET_TEXT_RESP_LEN(max) (2 + 1 + (max) + 1)
#define OTB_CHUNK_REQ_LEN  3
#define OTB_CHUNK_HDR      8        // seq u16, chunk, chunks, total u16, age u16
#define OTB_CHUNK_PAYLOAD  112
#define OTB_CHUNK_RESP_LEN (OTB_CHUNK_HDR + OTB_CHUNK_PAYLOAD + 1)

OTB_STATIC_ASSERT(OTB_TEXT_WRITE_LEN(OTB_TEXT_MAX) <= OTB_WIRE_LIMIT, "TEXT write too long");
OTB_STATIC_ASSERT(OTB_GET_TEXT_RESP_LEN(OTB_TEXT_MAX) <= OTB_WIRE_LIMIT, "GET_TEXT too long");
OTB_STATIC_ASSERT(OTB_CHUNK_RESP_LEN <= OTB_WIRE_LIMIT, "frame chunk too long");
OTB_STATIC_ASSERT(OTB_ARP_WRITE_LEN <= OTB_WIRE_LIMIT, "ARP write too long");

// ----- CRC and PEC --------------------------------------------------------------

// CRC-8, poly 0x07, init 0: the SMBus PEC.
static inline uint8_t otbCrc8Cont(uint8_t crc, const uint8_t *data, uint16_t len) {
  for (uint16_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (uint8_t b = 0; b < 8; b++) {
      crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
    }
  }
  return crc;
}

// CRC-32, zlib's: the UID hash in the UDID.
static inline uint32_t otbCrc32(const uint8_t *data, uint32_t len) {
  uint32_t crc = 0xFFFFFFFFUL;
  for (uint32_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (uint8_t b = 0; b < 8; b++) {
      crc = (crc & 1) ? ((crc >> 1) ^ 0xEDB88320UL) : (crc >> 1);
    }
  }
  return ~crc;
}

// The ARP shapes, as SMBus defines them. Byte-identical to v1's.
static inline uint8_t otbPecSendByte(uint8_t addr, uint8_t cmd) {
  const uint8_t h[2] = {(uint8_t)(addr << 1), cmd};
  return otbCrc8Cont(0, h, 2);
}

static inline uint8_t otbPecBlockRead(uint8_t addr, uint8_t cmd, uint8_t count,
                                      const uint8_t *data) {
  const uint8_t h[4] = {(uint8_t)(addr << 1), cmd, (uint8_t)((addr << 1) | 1), count};
  return otbCrc8Cont(otbCrc8Cont(0, h, 4), data, count);
}

static inline uint8_t otbPecBlockWrite(uint8_t addr, uint8_t cmd, uint8_t count,
                                       const uint8_t *data) {
  const uint8_t h[3] = {(uint8_t)(addr << 1), cmd, count};
  return otbCrc8Cont(otbCrc8Cont(0, h, 3), data, count);
}

// Every v2 transaction: CRC-8 over the node address, the command and the bytes
// that follow it -- for a request the arguments, for a response the payload.
// The address is in it so a reply from the wrong node cannot pass.
static inline uint8_t otbPec(uint8_t addr, uint8_t cmd, const uint8_t *data, uint16_t len) {
  const uint8_t h[2] = {addr, cmd};
  return otbCrc8Cont(otbCrc8Cont(0, h, 2), data, len);
}

static inline void otbPut16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static inline void otbPut32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static inline uint16_t otbGet16(const uint8_t *p) { return (uint16_t)(p[0] | ((uint16_t)p[1] << 8)); }
static inline uint32_t otbGet32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// ----- UDID and INFO ------------------------------------------------------------

// Spec-shaped 128-bit UDID; the device-specific field is the UID's CRC-32, so
// the controller can recognise a node it has seen before.
static inline void otbBuildUdid(uint8_t *udid, const uint8_t *uid) {
  memset(udid, 0, OTB_UDID_LEN);
  udid[0] = 0x80;                    // capabilities: address type 'random'
  udid[1] = 0x08;                    // UDID version
  udid[2] = 0xFF; udid[3] = 0xFF;    // vendor: none registered
  udid[4] = 0x00; udid[5] = 0x02;    // device: OpenTac bus v2 node (v1 was 1)
  udid[6] = 0x00; udid[7] = 0x04;    // interface: SMBus
  const uint32_t h = otbCrc32(uid, OTB_UID_BYTES);
  udid[OTB_UDID_SPECIFIC + 0] = (uint8_t)(h >> 24);
  udid[OTB_UDID_SPECIFIC + 1] = (uint8_t)(h >> 16);
  udid[OTB_UDID_SPECIFIC + 2] = (uint8_t)(h >> 8);
  udid[OTB_UDID_SPECIFIC + 3] = (uint8_t)(h);
}

static inline uint32_t otbUdidUidHash(const uint8_t *udid) {
  return ((uint32_t)udid[OTB_UDID_SPECIFIC] << 24) | ((uint32_t)udid[OTB_UDID_SPECIFIC + 1] << 16) |
         ((uint32_t)udid[OTB_UDID_SPECIFIC + 2] << 8) | (uint32_t)udid[OTB_UDID_SPECIFIC + 3];
}

// INFO, the answer to a plain read: magic, version, address, board, UID, CRC-8.
#define OTB_INFO_MAGIC     0xB5     // v1's INFO magic is 0xA7
#define OTB_INFO_LEN       17
#define OTB_BOARD_READOUT32 0
#define OTB_BOARD_NODE16    1

static inline void otbBuildInfo(uint8_t *s, uint8_t addr, uint8_t board, const uint8_t *uid) {
  s[0] = OTB_INFO_MAGIC;
  s[1] = OTB_VERSION;
  s[2] = addr;
  s[3] = board;
  memcpy(&s[4], uid, OTB_UID_BYTES);
  s[16] = otbCrc8Cont(0, s, 16);
}

static inline bool otbParseInfo(const uint8_t *s, uint8_t addr, uint8_t *board, uint8_t *uid) {
  if (s[0] != OTB_INFO_MAGIC || s[1] != OTB_VERSION || s[2] != addr) return false;
  if (s[16] != otbCrc8Cont(0, s, 16)) return false;
  *board = s[3];
  memcpy(uid, &s[4], OTB_UID_BYTES);
  return true;
}

// ----- STATUS -------------------------------------------------------------------
//
// Polled every round, so it says everything the controller needs to decide
// what to do next: whether a new frame is ready, how much reply text is
// waiting, and how much command text the node has taken.

#define OTB_STATUS_MAGIC   0xB6
#define OTB_STATUS_LEN     15

#define OTB_SF_CONFIG      0x01     // a skin is provisioned
#define OTB_SF_FRAME       0x02     // at least one frame has been published
#define OTB_SF_BUSY        0x04     // provisioning session open
#define OTB_SF_IMU         0x08     // the IMU initialised

struct OtbStatus {
  uint8_t flags;
  uint16_t frameSeq;     // sequence of the newest published frame
  uint16_t txTail;       // node -> controller text: stream offset of the oldest unread byte
  uint16_t txPending;    // ...and how many bytes follow it
  uint16_t rxAccepted;   // controller -> node text: bytes taken so far (mod 65536)
  uint8_t rxFree;        // room for more, capped at 255
  uint16_t generation;   // the board's config generation (ot_api.h)
  uint8_t board;         // OTB_BOARD_*
};

static inline void otbBuildStatus(uint8_t *s, uint8_t addr, const OtbStatus *st) {
  s[0] = OTB_STATUS_MAGIC;
  s[1] = st->flags;
  otbPut16(&s[2], st->frameSeq);
  otbPut16(&s[4], st->txTail);
  otbPut16(&s[6], st->txPending);
  otbPut16(&s[8], st->rxAccepted);
  s[10] = st->rxFree;
  otbPut16(&s[11], st->generation);
  s[13] = st->board;
  s[14] = otbPec(addr, OTB_CMD_STATUS, s, 14);
}

static inline bool otbParseStatus(const uint8_t *s, uint8_t addr, OtbStatus *st) {
  if (s[0] != OTB_STATUS_MAGIC || s[14] != otbPec(addr, OTB_CMD_STATUS, s, 14)) return false;
  st->flags = s[1];
  st->frameSeq = otbGet16(&s[2]);
  st->txTail = otbGet16(&s[4]);
  st->txPending = otbGet16(&s[6]);
  st->rxAccepted = otbGet16(&s[8]);
  st->rxFree = s[10];
  st->generation = otbGet16(&s[11]);
  st->board = s[13];
  return true;
}

// ----- Text ---------------------------------------------------------------------
//
// Both directions are byte streams addressed by a 16-bit stream offset, so a
// transaction lost to a PEC failure or a NACK is simply sent again: the
// receiver takes a byte only at the offset it expects next, and ignores a
// repeat. Nothing is ever duplicated or silently dropped.
//
//   controller -> node   TEXT     [cmd, off u16, len, bytes..., PEC]
//                        The node takes it only if `off` equals its own count
//                        of bytes taken and all `len` fit. STATUS reports that
//                        count, which is how the controller knows to resend.
//   node -> controller   GET_TEXT [cmd, ack u16, max, PEC], then a read of
//                        [off u16, len, max bytes (len used), PEC].
//                        `ack` is the controller's count of bytes received; the
//                        node frees everything before it and serves from there.

static inline uint8_t otbBuildText(uint8_t *out, uint8_t addr, uint16_t off,
                                   const uint8_t *bytes, uint8_t len) {
  out[0] = OTB_CMD_TEXT;
  otbPut16(&out[1], off);
  out[3] = len;
  memcpy(&out[4], bytes, len);
  out[4 + len] = otbPec(addr, OTB_CMD_TEXT, &out[1], (uint16_t)(3 + len));
  return (uint8_t)OTB_TEXT_WRITE_LEN(len);
}

// `in` is the whole write including the command byte.
static inline bool otbParseText(const uint8_t *in, uint16_t n, uint8_t addr,
                                uint16_t *off, const uint8_t **bytes, uint8_t *len) {
  if (n < OTB_TEXT_WRITE_LEN(0) || in[0] != OTB_CMD_TEXT) return false;
  const uint8_t l = in[3];
  if (l > OTB_TEXT_MAX || n < OTB_TEXT_WRITE_LEN(l)) return false;
  if (in[4 + l] != otbPec(addr, OTB_CMD_TEXT, &in[1], (uint16_t)(3 + l))) return false;
  *off = otbGet16(&in[1]);
  *bytes = &in[4];
  *len = l;
  return true;
}

static inline void otbBuildGetTextReq(uint8_t *out, uint8_t addr, uint16_t ack, uint8_t max) {
  out[0] = OTB_CMD_GET_TEXT;
  otbPut16(&out[1], ack);
  out[3] = max;
  out[4] = otbPec(addr, OTB_CMD_GET_TEXT, &out[1], 3);
}

static inline bool otbParseGetTextReq(const uint8_t *in, uint16_t n, uint8_t addr,
                                      uint16_t *ack, uint8_t *max) {
  if (n < OTB_GET_TEXT_REQ_LEN || in[0] != OTB_CMD_GET_TEXT) return false;
  if (in[4] != otbPec(addr, OTB_CMD_GET_TEXT, &in[1], 3)) return false;
  *ack = otbGet16(&in[1]);
  *max = in[3] > OTB_TEXT_MAX ? (uint8_t)OTB_TEXT_MAX : in[3];
  return true;
}

// The response is always `max` bytes long, padded: the controller has to know
// the length before it reads. `len` says how many of them are text.
static inline uint8_t otbBuildGetTextResp(uint8_t *out, uint8_t addr, uint16_t off,
                                          const uint8_t *bytes, uint8_t len, uint8_t max) {
  otbPut16(&out[0], off);
  out[2] = len;
  memcpy(&out[3], bytes, len);
  memset(&out[3 + len], 0, (size_t)(max - len));
  out[3 + max] = otbPec(addr, OTB_CMD_GET_TEXT, out, (uint16_t)(3 + max));
  return (uint8_t)OTB_GET_TEXT_RESP_LEN(max);
}

static inline bool otbParseGetTextResp(const uint8_t *in, uint8_t addr, uint8_t max,
                                       uint16_t *off, const uint8_t **bytes, uint8_t *len) {
  if (in[3 + max] != otbPec(addr, OTB_CMD_GET_TEXT, in, (uint16_t)(3 + max))) return false;
  if (in[2] > max) return false;
  *off = otbGet16(&in[0]);
  *len = in[2];
  *bytes = &in[3];
  return true;
}

// The node's two text rings. Each has exactly one writer and one reader --
// the loop and the I2C interrupt, one on each side -- so neither needs a lock:
// the writer publishes an index only after the bytes are in place, and the
// reader moves only its own. Counters run free and wrap; `size` is a power of
// two, so `counter & (size - 1)` is the slot.
#define OTB_BARRIER() __asm__ __volatile__("" ::: "memory")

// Node -> controller. The loop pushes reply text; the interrupt serves it and
// frees what the controller acknowledges.
struct OtbTxRing {
  uint8_t *buf;
  uint16_t size;
  volatile uint32_t head;     // written by the loop
  volatile uint32_t tail;     // written by the interrupt
};

static inline uint32_t otbTxUsed(const OtbTxRing *r) { return r->head - r->tail; }

// All or nothing, so a reply line is never cut in half by a full ring.
static inline bool otbTxPush(OtbTxRing *r, const uint8_t *bytes, uint16_t n) {
  if ((uint32_t)r->size - otbTxUsed(r) < n) return false;
  const uint32_t h = r->head;
  for (uint16_t i = 0; i < n; ++i) r->buf[(h + i) & (r->size - 1u)] = bytes[i];
  OTB_BARRIER();
  r->head = h + n;
  return true;
}

// `ack` is the controller's count of bytes received (mod 65536). Everything
// before it is freed; an ack from the past, or beyond what exists, changes
// nothing.
static inline void otbTxAck(OtbTxRing *r, uint16_t ack) {
  const uint16_t step = (uint16_t)(ack - (uint16_t)r->tail);
  if (step <= otbTxUsed(r)) r->tail = r->tail + step;
}

// Copies up to `max` bytes from the tail without freeing them; returns how
// many, and the stream offset of the first.
static inline uint8_t otbTxPeek(const OtbTxRing *r, uint8_t *dst, uint8_t max, uint16_t *off) {
  const uint32_t t = r->tail;
  const uint32_t used = r->head - t;
  const uint8_t n = (uint8_t)(used < max ? used : max);
  for (uint8_t i = 0; i < n; ++i) dst[i] = r->buf[(t + i) & (r->size - 1u)];
  *off = (uint16_t)t;
  return n;
}

// Controller -> node. The interrupt takes TEXT writes; the loop reads bytes
// out as its command line input, exactly as it would read USB.
struct OtbRxRing {
  uint8_t *buf;
  uint16_t size;
  volatile uint32_t head;     // written by the interrupt
  volatile uint32_t tail;     // written by the loop
};

static inline uint16_t otbRxFree(const OtbRxRing *r) {
  return (uint16_t)(r->size - (r->head - r->tail));
}

// Takes the bytes only if they start exactly where the stream is: a repeat of
// bytes already taken, or bytes that would leave a gap, are ignored. False
// only when they are new but do not fit (the controller resends later).
static inline bool otbRxAccept(OtbRxRing *r, uint16_t off, const uint8_t *bytes, uint8_t n) {
  const uint32_t h = r->head;
  if (off != (uint16_t)h) return true;
  if (otbRxFree(r) < n) return false;
  for (uint8_t i = 0; i < n; ++i) r->buf[(h + i) & (r->size - 1u)] = bytes[i];
  OTB_BARRIER();
  r->head = h + n;
  return true;
}

static inline int otbRxRead(OtbRxRing *r) {
  const uint32_t t = r->tail;
  if (t == r->head) return -1;
  const uint8_t c = r->buf[t & (r->size - 1u)];
  OTB_BARRIER();
  r->tail = t + 1;
  return c;
}

// The controller's side of a GET_TEXT: `expected` is its count of bytes
// received so far. Returns how many of the `len` bytes at `off` are new, and
// sets `skip` to how many leading bytes it already has. A response that starts
// past `expected` would mean bytes were lost, which the protocol cannot do; it
// is treated as nothing new, and the unchanged ack asks for them again.
static inline uint8_t otbTextNew(uint16_t *expected, uint16_t off, uint8_t len, uint8_t *skip) {
  const uint16_t behind = (uint16_t)(*expected - off);
  if (behind > len) { *skip = 0; return 0; }   // ahead of us (a gap) or long gone
  *skip = (uint8_t)behind;
  const uint8_t fresh = (uint8_t)(len - behind);
  *expected = (uint16_t)(*expected + fresh);
  return fresh;
}

// ----- Frames -------------------------------------------------------------------
//
// After each scan the node publishes one frame record ("blob") into a triple
// buffer. The controller reads it in OTB_CHUNK_PAYLOAD-byte chunks: asking for
// chunk 0 latches the newest blob, and every chunk carries that blob's sequence
// number, so a set read across two publishes is detected and thrown away rather
// than stitched into a frame that never existed.
//
// Blob layout:
//   0  u8   OTB_FRAME_VERSION
//   1  u8   flags: OTB_FF_TACTILE | OTB_FF_SPARSE | OTB_FF_IMU
//   2  u32  frame counter          (the F/X record's <frame>)
//   6  u16  config generation      (<gen>)
//   8  u8   tactile mode           (<mode>)
//   9  u32  dt_us                  (<dt_us>)
//  13  u32  vibration samples dropped (<dropped>)
//  17  u16  N, the skin's active taxels
//  19  u16  n, values carried: N for a raster frame, the sampled count for X
//  21  u8   V, vibration samples carried
//  22  u32  sequence number of the first of them
//  26       sparse only: the sampled mask, ceil(N/8) bytes, taxel t = bit t%8 of byte t/8
//  ..       n values, 12-bit, packed two per three bytes (lo8 of a | hi4 of a, lo4 of b << 4 | hi8 of b)
//  ..       IMU only: accel mg i16 x3, gyro mdps i32 x3, quat w,x,y,z Q14 i16 x4,
//           euler roll,pitch,yaw centidegrees i16 x3   (32 bytes)
//  ..       V x (accel mg i16 x3)

#define OTB_FRAME_VERSION  1
#define OTB_FF_TACTILE     0x01
#define OTB_FF_SPARSE      0x02
#define OTB_FF_IMU         0x04
#define OTB_FRAME_HDR      26
#define OTB_IMU_BYTES      32
#define OTB_MAX_VIB        32       // vibration samples per blob
#define OTB_MAX_TAXELS     1024     // the largest skin any board can hold (32 x 32)

#define OTB_PACKED_BYTES(n) ((uint16_t)(((uint32_t)(n) * 3u + 1u) / 2u))
#define OTB_BLOB_MAX(taxels) \
  (OTB_FRAME_HDR + ((taxels) + 7u) / 8u + OTB_PACKED_BYTES(taxels) + OTB_IMU_BYTES + OTB_MAX_VIB * 6u)

static inline void otbPack12(uint8_t *out, const uint16_t *v, uint16_t n) {
  uint16_t i = 0;
  for (; i + 1u < n; i += 2) {
    const uint16_t a = v[i] & 0x0FFFu, b = v[i + 1] & 0x0FFFu;
    *out++ = (uint8_t)a;
    *out++ = (uint8_t)((a >> 8) | (b << 4));
    *out++ = (uint8_t)(b >> 4);
  }
  if (i < n) {
    const uint16_t a = v[i] & 0x0FFFu;
    *out++ = (uint8_t)a;
    *out++ = (uint8_t)(a >> 8);
  }
}

static inline uint16_t otbUnpack12(const uint8_t *in, uint16_t k) {
  const uint8_t *p = in + (k >> 1) * 3u;
  return (k & 1u) ? (uint16_t)((p[1] >> 4) | ((uint16_t)p[2] << 4))
                  : (uint16_t)(p[0] | ((uint16_t)(p[1] & 0x0Fu) << 8));
}

static inline int16_t otbClamp16(int32_t v) {
  return (int16_t)(v > 32767 ? 32767 : v < -32768 ? -32768 : v);
}

static inline int16_t otbRound16(float v, float scale) {
  const float s = v * scale;
  return otbClamp16((int32_t)(s >= 0.0f ? s + 0.5f : s - 0.5f));
}

// What the node knows about a frame, gathered from its globals.
struct OtbFrame {
  uint8_t flags;              // OTB_FF_*
  uint32_t frame;
  uint16_t generation;
  uint8_t mode;
  uint32_t dtUs;
  uint32_t vibDropped;
  uint16_t active;            // N
  const uint16_t *values;     // indexed by taxel (pixels order), N long
  const uint32_t *mask;       // sparse: bit t of word t/32 = taxel t was read
  int32_t accel[3];           // mg
  int32_t gyro[3];            // mdps
  float quat[4];              // w, x, y, z
  float euler[3];             // degrees
  uint8_t vibCount;
  uint32_t vibSeq;
  const int32_t *vib;         // vibCount x 3, mg
};

// Returns the blob length, or 0 if it would not fit `cap`.
static inline uint16_t otbBuildFrame(uint8_t *out, uint16_t cap, const OtbFrame *f) {
  const bool tactile = (f->flags & OTB_FF_TACTILE) != 0;
  const bool sparse = tactile && (f->flags & OTB_FF_SPARSE) != 0;
  const uint16_t active = tactile ? f->active : 0;
  const uint16_t maskBytes = sparse ? (uint16_t)((active + 7u) / 8u) : 0;
  uint16_t n = active;
  if (sparse) {
    n = 0;
    for (uint16_t t = 0; t < active; ++t) n = (uint16_t)(n + ((f->mask[t >> 5] >> (t & 31u)) & 1u));
  }
  const uint8_t vibCount = f->vibCount > OTB_MAX_VIB ? (uint8_t)OTB_MAX_VIB : f->vibCount;
  const uint32_t total = (uint32_t)OTB_FRAME_HDR + maskBytes + OTB_PACKED_BYTES(n) +
                         ((f->flags & OTB_FF_IMU) ? OTB_IMU_BYTES : 0u) + vibCount * 6u;
  if (total > cap) return 0;

  out[0] = OTB_FRAME_VERSION;
  out[1] = (uint8_t)(f->flags & (OTB_FF_TACTILE | OTB_FF_SPARSE | OTB_FF_IMU));
  otbPut32(&out[2], f->frame);
  otbPut16(&out[6], f->generation);
  out[8] = f->mode;
  otbPut32(&out[9], f->dtUs);
  otbPut32(&out[13], f->vibDropped);
  otbPut16(&out[17], active);
  otbPut16(&out[19], n);
  out[21] = vibCount;
  otbPut32(&out[22], f->vibSeq);
  uint8_t *p = out + OTB_FRAME_HDR;

  if (sparse) {
    for (uint16_t b = 0; b < maskBytes; ++b) *p++ = (uint8_t)(f->mask[b >> 2] >> ((b & 3u) * 8u));
    // Pack the sampled values in ascending taxel order, two at a time, without
    // a gathered copy -- the node has no RAM to spare for one.
    uint16_t pending = 0, have = 0;
    for (uint16_t t = 0; t < active; ++t) {
      if (!((f->mask[t >> 5] >> (t & 31u)) & 1u)) continue;
      if (have == 0) { pending = f->values[t]; have = 1; continue; }
      const uint16_t pair[2] = {pending, f->values[t]};
      otbPack12(p, pair, 2);
      p += 3;
      have = 0;
    }
    if (have) { otbPack12(p, &pending, 1); p += 2; }
  } else if (tactile) {
    otbPack12(p, f->values, n);
    p += OTB_PACKED_BYTES(n);
  }

  if (f->flags & OTB_FF_IMU) {
    for (uint8_t i = 0; i < 3; ++i) { otbPut16(p, (uint16_t)otbClamp16(f->accel[i])); p += 2; }
    for (uint8_t i = 0; i < 3; ++i) { otbPut32(p, (uint32_t)f->gyro[i]); p += 4; }
    for (uint8_t i = 0; i < 4; ++i) { otbPut16(p, (uint16_t)otbRound16(f->quat[i], 16384.0f)); p += 2; }
    for (uint8_t i = 0; i < 3; ++i) { otbPut16(p, (uint16_t)otbRound16(f->euler[i], 100.0f)); p += 2; }
  }
  for (uint16_t k = 0; k < (uint16_t)vibCount * 3u; ++k) {
    otbPut16(p, (uint16_t)otbClamp16(f->vib[k]));
    p += 2;
  }
  return (uint16_t)total;
}

// ----- Frame chunks -------------------------------------------------------------

static inline void otbBuildChunkReq(uint8_t *out, uint8_t addr, uint8_t chunk) {
  out[0] = OTB_CMD_GET_FRAME;
  out[1] = chunk;
  out[2] = otbPec(addr, OTB_CMD_GET_FRAME, &out[1], 1);
}

static inline bool otbParseChunkReq(const uint8_t *in, uint16_t n, uint8_t addr, uint8_t *chunk) {
  if (n < OTB_CHUNK_REQ_LEN || in[0] != OTB_CMD_GET_FRAME) return false;
  if (in[2] != otbPec(addr, OTB_CMD_GET_FRAME, &in[1], 1)) return false;
  *chunk = in[1];
  return true;
}

static inline uint8_t otbChunkCount(uint16_t total) {
  return (uint8_t)((total + OTB_CHUNK_PAYLOAD - 1u) / OTB_CHUNK_PAYLOAD);
}

// `blob` NULL or `total` 0: no frame yet, answered with zero chunks.
static inline void otbBuildChunkResp(uint8_t *out, uint8_t addr, uint16_t seq, uint8_t chunk,
                                     const uint8_t *blob, uint16_t total, uint16_t ageUs) {
  const uint8_t chunks = blob ? otbChunkCount(total) : 0;
  otbPut16(&out[0], seq);
  out[2] = chunk;
  out[3] = chunks;
  otbPut16(&out[4], blob ? total : 0);
  otbPut16(&out[6], ageUs);
  uint16_t len = 0;
  if (blob && chunk < chunks) {
    const uint16_t start = (uint16_t)chunk * OTB_CHUNK_PAYLOAD;
    len = (uint16_t)(total - start < OTB_CHUNK_PAYLOAD ? total - start : OTB_CHUNK_PAYLOAD);
    memcpy(&out[OTB_CHUNK_HDR], blob + start, len);
  }
  memset(&out[OTB_CHUNK_HDR + len], 0, (size_t)(OTB_CHUNK_PAYLOAD - len));
  out[OTB_CHUNK_HDR + OTB_CHUNK_PAYLOAD] =
      otbPec(addr, OTB_CMD_GET_FRAME, out, OTB_CHUNK_HDR + OTB_CHUNK_PAYLOAD);
}

struct OtbChunk {
  uint16_t seq;
  uint8_t chunk;
  uint8_t chunks;
  uint16_t total;
  uint16_t ageUs;
  const uint8_t *payload;
  uint16_t len;              // payload bytes that belong to the blob
};

static inline bool otbParseChunkResp(const uint8_t *in, uint8_t addr, OtbChunk *c) {
  if (in[OTB_CHUNK_HDR + OTB_CHUNK_PAYLOAD] !=
      otbPec(addr, OTB_CMD_GET_FRAME, in, OTB_CHUNK_HDR + OTB_CHUNK_PAYLOAD)) return false;
  c->seq = otbGet16(&in[0]);
  c->chunk = in[2];
  c->chunks = in[3];
  c->total = otbGet16(&in[4]);
  c->ageUs = otbGet16(&in[6]);
  if (c->chunks != otbChunkCount(c->total)) return false;
  c->payload = &in[OTB_CHUNK_HDR];
  const uint32_t start = (uint32_t)c->chunk * OTB_CHUNK_PAYLOAD;
  c->len = (c->chunk < c->chunks)
               ? (uint16_t)(c->total - start < OTB_CHUNK_PAYLOAD ? c->total - start : OTB_CHUNK_PAYLOAD)
               : 0;
  return true;
}

// ----- Text records, as a USB board prints them ----------------------------------
//
// The controller turns each blob back into the records a single board writes
// on USB -- same fields, same order -- so the host parses one format:
//
//   <node>,<t_us>|F,<frame>,<gen>,<mode>,<dt_us>,<dropped>,<N values>,<imu 13>
//   <node>,<t_us>|X,<frame>,<gen>,<mode>,<dt_us>,<dropped>,<n>,<mask hex>,<n values>,<imu 13>
//   <node>,<t_us>|V,<seq>,<ax>,<ay>,<az>
//
// <imu 13> is accel mg x3, gyro mdps x3, quat w,x,y,z (5 decimals), euler
// roll,pitch,yaw (2 decimals); zeros and the identity quaternion when the frame
// carries no IMU block, which is what a USB board prints before its IMU has
// produced anything. <t_us> is
// the controller's clock when the node published the frame, the same clock for
// every node, so frames from different sensors can be put on one timeline.

typedef void (*OtbSink)(void *ctx, const char *text, uint16_t len);

struct OtbOut {
  char buf[256];
  uint16_t n;
  OtbSink sink;
  void *ctx;
};

static inline void otbOutInit(OtbOut *o, OtbSink sink, void *ctx) { o->n = 0; o->sink = sink; o->ctx = ctx; }

static inline void otbFlush(OtbOut *o) {
  if (o->n) o->sink(o->ctx, o->buf, o->n);
  o->n = 0;
}

static inline void otbPutc(OtbOut *o, char c) {
  if (o->n == sizeof(o->buf)) otbFlush(o);
  o->buf[o->n++] = c;
}

static inline void otbPutU64(OtbOut *o, uint64_t v) {
  char r[20];
  uint8_t d = 0;
  do { r[d++] = (char)('0' + (uint8_t)(v % 10u)); v /= 10u; } while (v);
  while (d) otbPutc(o, r[--d]);
}

static inline void otbPutI(OtbOut *o, int32_t v) {
  if (v < 0) { otbPutc(o, '-'); otbPutU64(o, (uint64_t)(-(int64_t)v)); }
  else otbPutU64(o, (uint64_t)v);
}

// Fixed point from an integer numerator: `v / scale` to `decimals` places,
// rounded half away from zero. Integer, so both ends print the same digits.
static inline void otbPutScaled(OtbOut *o, int32_t v, uint32_t scale, uint8_t decimals) {
  uint32_t pow10 = 1;
  for (uint8_t i = 0; i < decimals; ++i) pow10 *= 10u;
  const bool neg = v < 0;
  const uint64_t mag = (uint64_t)(neg ? -(int64_t)v : v);
  const uint64_t scaled = (mag * pow10 * 2u + scale) / (2u * scale);
  if (neg && scaled) otbPutc(o, '-');
  otbPutU64(o, scaled / pow10);
  otbPutc(o, '.');
  uint64_t frac = scaled % pow10;
  for (uint32_t div = pow10 / 10u; div; div /= 10u) {
    otbPutc(o, (char)('0' + (uint8_t)(frac / div)));
    frac %= div;
  }
}

static inline void otbPrefix(OtbOut *o, uint8_t node, uint64_t tUs) {
  otbPutU64(o, node);
  otbPutc(o, ',');
  otbPutU64(o, tUs);
  otbPutc(o, '|');
}

// Checks the blob adds up, then prints its records. False, and nothing
// printed, if it does not: a malformed frame is dropped, never half-printed.
//
// `vibNext`, when given, is the next vibration sequence number this node has
// not printed yet. A node frees samples only once a blob carrying them has been
// read in full, so consecutive blobs can repeat a few; those are skipped here
// and `vibNext` moves past the last one printed. NULL prints every sample.
static inline uint32_t otbFrameVibSeq(const uint8_t *blob) { return otbGet32(&blob[22]); }

static inline bool otbFormatFrame(const uint8_t *blob, uint16_t len, uint8_t node,
                                  uint64_t tUs, uint32_t *vibNext, OtbOut *o) {
  if (len < OTB_FRAME_HDR || blob[0] != OTB_FRAME_VERSION) return false;
  const uint8_t flags = blob[1];
  const bool tactile = flags & OTB_FF_TACTILE, sparse = flags & OTB_FF_SPARSE, imu = flags & OTB_FF_IMU;
  const uint16_t active = otbGet16(&blob[17]), n = otbGet16(&blob[19]);
  const uint8_t vibCount = blob[21];
  if (sparse && !tactile) return false;
  if (!tactile && (active || n)) return false;
  if (tactile && !sparse && n != active) return false;
  if (n > active || active > OTB_MAX_TAXELS || vibCount > OTB_MAX_VIB) return false;
  const uint16_t maskBytes = sparse ? (uint16_t)((active + 7u) / 8u) : 0;
  const uint32_t expect = (uint32_t)OTB_FRAME_HDR + maskBytes + OTB_PACKED_BYTES(n) +
                          (imu ? OTB_IMU_BYTES : 0u) + vibCount * 6u;
  if (expect != len) return false;
  const uint8_t *mask = blob + OTB_FRAME_HDR;
  if (sparse) {
    uint16_t set = 0;
    for (uint16_t t = 0; t < active; ++t) set = (uint16_t)(set + ((mask[t >> 3] >> (t & 7u)) & 1u));
    if (set != n) return false;
  }
  const uint8_t *packed = mask + maskBytes;
  const uint8_t *imuBlock = packed + OTB_PACKED_BYTES(n);
  const uint8_t *vib = imuBlock + (imu ? OTB_IMU_BYTES : 0u);

  const uint32_t vibSeq = otbFrameVibSeq(blob);
  for (uint8_t s = 0; s < vibCount; ++s) {
    const uint32_t seq = vibSeq + s;
    if (vibNext && (int32_t)(seq - *vibNext) < 0) continue;     // printed already
    otbPrefix(o, node, tUs);
    otbPutc(o, 'V'); otbPutc(o, ',');
    otbPutU64(o, seq);
    for (uint8_t a = 0; a < 3; ++a) {
      otbPutc(o, ',');
      otbPutI(o, (int16_t)otbGet16(&vib[(s * 3u + a) * 2u]));
    }
    otbPutc(o, '\n');
    if (vibNext) *vibNext = seq + 1u;
  }

  if (tactile) {
    otbPrefix(o, node, tUs);
    otbPutc(o, sparse ? 'X' : 'F');
    const uint32_t head[5] = {otbGet32(&blob[2]), otbGet16(&blob[6]), blob[8],
                              otbGet32(&blob[9]), otbGet32(&blob[13])};
    for (uint8_t i = 0; i < 5; ++i) { otbPutc(o, ','); otbPutU64(o, head[i]); }
    if (sparse) {
      static const char HEX_DIGITS[] = "0123456789ABCDEF";
      otbPutc(o, ',');
      otbPutU64(o, n);
      otbPutc(o, ',');
      for (uint16_t b = 0; b < maskBytes; ++b) {
        otbPutc(o, HEX_DIGITS[mask[b] >> 4]);
        otbPutc(o, HEX_DIGITS[mask[b] & 0x0Fu]);
      }
    }
    for (uint16_t k = 0; k < n; ++k) { otbPutc(o, ','); otbPutU64(o, otbUnpack12(packed, k)); }
    if (imu) {
      const uint8_t *p = imuBlock;
      for (uint8_t i = 0; i < 3; ++i, p += 2) { otbPutc(o, ','); otbPutI(o, (int16_t)otbGet16(p)); }
      for (uint8_t i = 0; i < 3; ++i, p += 4) { otbPutc(o, ','); otbPutI(o, (int32_t)otbGet32(p)); }
      for (uint8_t i = 0; i < 4; ++i, p += 2) { otbPutc(o, ','); otbPutScaled(o, (int16_t)otbGet16(p), 16384u, 5); }
      for (uint8_t i = 0; i < 3; ++i, p += 2) { otbPutc(o, ','); otbPutScaled(o, (int16_t)otbGet16(p), 100u, 2); }
    } else {
      static const char ZEROS[] = ",0,0,0,0,0,0,1.00000,0.00000,0.00000,0.00000,0.00,0.00,0.00";
      for (const char *z = ZEROS; *z; ++z) otbPutc(o, *z);
    }
    otbPutc(o, '\n');
  }
  otbFlush(o);
  return true;
}
