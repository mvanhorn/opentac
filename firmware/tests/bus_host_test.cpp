// Host test for libraries/OpenTactileBus/src/ot_bus.h -- the code the node and
// the controller both run, driven here without hardware. From firmware/tests:
//
//   c++ -std=c++17 -O2 -Wall -Wextra -I../libraries/OpenTactileBus/src \
//       bus_host_test.cpp -o /tmp/bt && /tmp/bt fixtures/bus_records.txt
//
// The optional argument writes the records the controller would print for a
// set of test frames. tests/readout/test_protocol.py parses that file with the
// host library's own Frame parser, which is what ties the C formatter to the
// Python reader: change either and one of the two tests fails.

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "ot_bus.h"

static int failures = 0, checks = 0;

#define CHECK(cond, ...)                                  \
  do {                                                    \
    ++checks;                                             \
    if (!(cond)) {                                        \
      ++failures;                                         \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);    \
      std::printf(__VA_ARGS__);                           \
      std::printf("\n");                                  \
    }                                                     \
  } while (0)

static uint32_t rng = 12345;
static uint32_t rnd() { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

static void sinkString(void *ctx, const char *text, uint16_t len) {
  static_cast<std::string *>(ctx)->append(text, len);
}

static std::vector<std::string> split(const std::string &s, char sep) {
  std::vector<std::string> out;
  size_t start = 0;
  for (size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == sep) { out.push_back(s.substr(start, i - start)); start = i + 1; }
  }
  return out;
}

// ----- 12-bit packing ----------------------------------------------------------

static void testPack() {
  for (uint16_t n = 0; n < 40; ++n) {
    uint16_t v[40];
    for (uint16_t i = 0; i < n; ++i) v[i] = (uint16_t)(rnd() & 0x0FFFu);
    uint8_t packed[64];
    memset(packed, 0xEE, sizeof(packed));
    otbPack12(packed, v, n);
    CHECK(packed[OTB_PACKED_BYTES(n)] == 0xEE, "pack %u wrote past its length", n);
    for (uint16_t i = 0; i < n; ++i) CHECK(otbUnpack12(packed, i) == v[i], "pack %u slot %u", n, i);
  }
  const uint16_t edge[3] = {0, 4095, 2048};
  uint8_t packed[5];
  otbPack12(packed, edge, 3);
  CHECK(otbUnpack12(packed, 0) == 0 && otbUnpack12(packed, 1) == 4095 && otbUnpack12(packed, 2) == 2048,
        "12-bit extremes");
}

// ----- frames, end to end ----------------------------------------------------

struct Built {
  std::vector<uint8_t> blob;
  std::vector<uint16_t> values;
  std::vector<uint32_t> mask;
};

static Built makeFrame(uint16_t active, bool sparse, bool imu, uint8_t vib, uint32_t frameNo) {
  Built b;
  b.values.resize(active);
  for (auto &v : b.values) v = (uint16_t)(rnd() % 4096u);
  b.mask.assign((active + 31u) / 32u + 1u, 0);
  if (sparse) {
    for (uint16_t t = 0; t < active; ++t) if (rnd() % 3u == 0) b.mask[t >> 5] |= 1u << (t & 31u);
  }
  static int32_t vibs[OTB_MAX_VIB * 3];
  for (int i = 0; i < OTB_MAX_VIB * 3; ++i) vibs[i] = (int32_t)(rnd() % 8000u) - 4000;

  OtbFrame f = {};
  f.flags = (uint8_t)((active ? OTB_FF_TACTILE : 0) | (sparse ? OTB_FF_SPARSE : 0) | (imu ? OTB_FF_IMU : 0));
  f.frame = frameNo;
  f.generation = 7;
  f.mode = sparse ? 2 : 0;
  f.dtUs = 3150;
  f.vibDropped = 4;
  f.active = active;
  f.values = b.values.data();
  f.mask = b.mask.data();
  f.accel[0] = -12; f.accel[1] = 1003; f.accel[2] = 40000;       // last one clamps to i16
  f.gyro[0] = -2000000; f.gyro[1] = 0; f.gyro[2] = 123456;
  f.quat[0] = 0.70711f; f.quat[1] = -0.70711f; f.quat[2] = 0.0f; f.quat[3] = 1.0f;
  f.euler[0] = -179.99f; f.euler[1] = 0.004f; f.euler[2] = 45.5f;
  f.vibCount = vib;
  f.vibSeq = 4294967290u;                                           // wraps mid-blob
  f.vib = vibs;

  b.blob.resize(OTB_BLOB_MAX(OTB_MAX_TAXELS));
  const uint16_t len = otbBuildFrame(b.blob.data(), (uint16_t)b.blob.size(), &f);
  CHECK(len > 0, "frame of %u taxels did not build", active);
  CHECK(otbBuildFrame(b.blob.data(), (uint16_t)(len - 1), &f) == 0, "a blob one byte short must refuse");
  b.blob.resize(len);
  return b;
}

// Chunk the blob as the node would serve it, reassemble as the controller does.
static std::vector<uint8_t> throughChunks(const std::vector<uint8_t> &blob, uint8_t addr, uint16_t seq) {
  std::vector<uint8_t> out(blob.size());
  const uint8_t chunks = otbChunkCount((uint16_t)blob.size());
  for (uint8_t c = 0; c < chunks; ++c) {
    uint8_t req[OTB_CHUNK_REQ_LEN];
    otbBuildChunkReq(req, addr, c);
    uint8_t got = 0xFF;
    CHECK(otbParseChunkReq(req, sizeof(req), addr, &got) && got == c, "chunk request %u", c);
    uint8_t resp[OTB_CHUNK_RESP_LEN];
    otbBuildChunkResp(resp, addr, seq, c, blob.data(), (uint16_t)blob.size(), 321);
    OtbChunk ch;
    CHECK(otbParseChunkResp(resp, addr, &ch), "chunk %u parse", c);
    CHECK(ch.seq == seq && ch.chunk == c && ch.chunks == chunks && ch.total == blob.size() && ch.ageUs == 321,
          "chunk %u header", c);
    memcpy(&out[(size_t)c * OTB_CHUNK_PAYLOAD], ch.payload, ch.len);
    resp[5 + (c % 100)] ^= 0x10;
    CHECK(!otbParseChunkResp(resp, addr, &ch), "a corrupted chunk must fail its PEC");
    otbBuildChunkResp(resp, addr, seq, c, blob.data(), (uint16_t)blob.size(), 321);
    CHECK(!otbParseChunkResp(resp, (uint8_t)(addr + 1), &ch), "a chunk for another node must fail");
  }
  return out;
}

static std::string format(const std::vector<uint8_t> &blob, uint8_t node, uint64_t t, bool *ok,
                          uint32_t *vibNext = nullptr) {
  std::string s;
  OtbOut o;
  otbOutInit(&o, sinkString, &s);
  *ok = otbFormatFrame(blob.data(), (uint16_t)blob.size(), node, t, vibNext, &o);
  return s;
}

// Consecutive blobs that repeat vibration samples print each sample once.
static void testVibDedupe() {
  const Built a = makeFrame(0, false, true, 10, 1);        // seqs S .. S+9
  uint32_t next = otbFrameVibSeq(a.blob.data());
  bool ok = false;
  std::string first = format(a.blob, 3, 1, &ok, &next);
  CHECK(ok && std::count(first.begin(), first.end(), '\n') == 10, "first blob prints all 10");
  CHECK(next == otbFrameVibSeq(a.blob.data()) + 10, "vibNext moves past the last printed");
  std::string again = format(a.blob, 3, 2, &ok, &next);
  CHECK(ok && again.empty(), "the same samples again print nothing");
  // A blob starting 6 samples in overlaps 4 of them: only its last 6 are new.
  Built b = makeFrame(0, false, true, 10, 2);
  otbPut32(&b.blob[22], otbFrameVibSeq(a.blob.data()) + 6);
  const std::string later = format(b.blob, 3, 3, &ok, &next);
  CHECK(ok && std::count(later.begin(), later.end(), '\n') == 6, "overlapping blob prints only new samples");
}

static void checkRecord(const Built &b, const std::string &text, uint16_t active, bool sparse, bool imu,
                        uint8_t vib, uint8_t node, uint64_t t) {
  auto lines = split(text, '\n');
  CHECK(!lines.empty() && lines.back().empty(), "records end in a newline");
  lines.pop_back();
  const size_t expectLines = vib + (active ? 1u : 0u);
  CHECK(lines.size() == expectLines, "%zu lines, expected %zu", lines.size(), expectLines);
  const std::string prefix = std::to_string(node) + "," + std::to_string(t) + "|";
  for (size_t i = 0; i < lines.size(); ++i) {
    CHECK(lines[i].compare(0, prefix.size(), prefix) == 0, "line %zu prefix", i);
  }
  for (uint8_t s = 0; s < vib && s < lines.size(); ++s) {
    const auto f = split(lines[s].substr(prefix.size()), ',');
    CHECK(f.size() == 5 && f[0] == "V", "V record shape");
    if (f.size() == 5) CHECK(std::stoul(f[1]) == (uint32_t)(4294967290u + s), "V sequence wraps");
  }
  if (!active) return;
  const auto f = split(lines.back().substr(prefix.size()), ',');
  CHECK(f[0] == (sparse ? "X" : "F"), "record letter");
  std::vector<uint16_t> want;
  std::vector<uint16_t> idx;
  for (uint16_t k = 0; k < active; ++k) {
    if (!sparse || ((b.mask[k >> 5] >> (k & 31u)) & 1u)) { want.push_back(b.values[k]); idx.push_back(k); }
  }
  const size_t head = sparse ? 8 : 6;          // letter + 5 fields (+ n, mask)
  CHECK(f.size() == head + want.size() + 13, "field count %zu, expected %zu", f.size(), head + want.size() + 13);
  if (f.size() != head + want.size() + 13) return;
  CHECK(f[1] == "99" && f[2] == "7" && f[4] == "3150" && f[5] == "4", "F/X head fields");
  if (sparse) {
    CHECK(std::stoul(f[6]) == want.size(), "X n");
    const std::string &hex = f[7];
    CHECK(hex.size() == 2u * ((active + 7u) / 8u), "X mask length");
    for (size_t j = 0; j < idx.size(); ++j) {
      const unsigned byte = (unsigned)std::stoul(hex.substr(2u * (idx[j] / 8u), 2), nullptr, 16);
      CHECK((byte >> (idx[j] % 8u)) & 1u, "X mask bit for taxel %u", idx[j]);
    }
  }
  for (size_t j = 0; j < want.size(); ++j) {
    CHECK(std::stoul(f[head + j]) == want[j], "value %zu", j);
  }
  const size_t im = head + want.size();
  if (imu) {
    CHECK(f[im + 0] == "-12" && f[im + 2] == "32767", "accel (clamped): %s %s", f[im].c_str(), f[im + 2].c_str());
    CHECK(f[im + 3] == "-2000000" && f[im + 5] == "123456", "gyro");
    CHECK(f[im + 6] == "0.70709" || f[im + 6] == "0.70712", "quat w %s", f[im + 6].c_str());
    CHECK(f[im + 9] == "1.00000", "quat z %s", f[im + 9].c_str());
    CHECK(f[im + 10] == "-179.99" && f[im + 11] == "0.00" && f[im + 12] == "45.50",
          "euler %s %s %s", f[im + 10].c_str(), f[im + 11].c_str(), f[im + 12].c_str());
  } else {
    CHECK(f[im + 6] == "1.00000" && f[im + 12] == "0.00", "no-IMU tail is zeros + identity quaternion");
  }
}

static void testFrames(FILE *fixture) {
  struct Case { uint16_t active; bool sparse, imu; uint8_t vib; };
  const Case cases[] = {
    {81, false, false, 0}, {81, true, false, 0}, {256, false, true, 5}, {256, true, true, 32},
    {1, false, false, 0}, {1, true, false, 0}, {1024, false, true, 0}, {1024, true, false, 3},
    {0, false, true, 7},                 // IMU-only blob: V records, no F
    {434, false, false, 0}, {33, true, true, 1},
  };
  uint8_t node = 0;
  for (const Case &c : cases) {
    const Built b = makeFrame(c.active, c.sparse, c.imu, c.vib, 99);
    const uint8_t addr = (uint8_t)(OTB_ADDR_BASE + node);
    const std::vector<uint8_t> back = throughChunks(b.blob, addr, (uint16_t)(1000 + node));
    CHECK(back == b.blob, "reassembled blob differs (%u taxels)", c.active);
    bool ok = false;
    const uint64_t t = 12345678901ULL + node;
    const std::string text = format(back, node, t, &ok);
    CHECK(ok, "blob of %u taxels failed to format", c.active);
    checkRecord(b, text, c.active, c.sparse, c.imu, c.vib, node, t);
    if (fixture) {
      // One comment line saying what the record must decode to, then the records.
      std::fprintf(fixture, "# expect node=%u active=%u sparse=%d imu=%d vib=%u values=", node, c.active,
                   c.sparse, c.imu, c.vib);
      bool first = true;
      for (uint16_t k = 0; k < c.active; ++k) {
        const bool read = !c.sparse || ((b.mask[k >> 5] >> (k & 31u)) & 1u);
        std::fprintf(fixture, "%s%d", first ? "" : ";", read ? (int)b.values[k] : -1);
        first = false;
      }
      std::fprintf(fixture, "\n%s", text.c_str());
    }
    // Every truncation, and one byte too many, is refused outright.
    for (size_t cut = 0; cut < back.size(); cut += 1 + back.size() / 17) {
      std::vector<uint8_t> shorter(back.begin(), back.begin() + (long)cut);
      bool shortOk = true;
      const std::string none = format(shorter, node, t, &shortOk);
      CHECK(!shortOk && none.empty(), "truncated blob (%zu of %zu) must not print", cut, back.size());
    }
    std::vector<uint8_t> longer = back;
    longer.push_back(0);
    bool longOk = true;
    CHECK(format(longer, node, t, &longOk).empty() && !longOk, "over-long blob must not print");
    ++node;
  }

  // A sparse mask that disagrees with its count is refused.
  Built b = makeFrame(81, true, false, 0, 5);
  b.blob[OTB_FRAME_HDR] ^= 0x01;
  bool ok = true;
  CHECK(format(b.blob, 0, 0, &ok).empty() && !ok, "mask/count mismatch must not print");

  // No frame yet: zero chunks, and the controller sees exactly that.
  uint8_t resp[OTB_CHUNK_RESP_LEN];
  otbBuildChunkResp(resp, 0x10, 0, 0, nullptr, 0, 0);
  OtbChunk ch;
  CHECK(otbParseChunkResp(resp, 0x10, &ch) && ch.chunks == 0 && ch.total == 0, "no-frame chunk");
}

// ----- STATUS, INFO, UDID ------------------------------------------------------

static void testSmallFrames() {
  OtbStatus st = {OTB_SF_CONFIG | OTB_SF_FRAME, 65535, 1234, 96, 65530, 200, 17, OTB_BOARD_NODE16};
  uint8_t s[OTB_STATUS_LEN];
  otbBuildStatus(s, 0x12, &st);
  OtbStatus back;
  CHECK(otbParseStatus(s, 0x12, &back), "status parse");
  CHECK(back.frameSeq == 65535 && back.txTail == 1234 && back.txPending == 96 && back.rxAccepted == 65530 &&
        back.rxFree == 200 && back.generation == 17 && back.board == OTB_BOARD_NODE16 && back.flags == st.flags,
        "status fields");
  CHECK(!otbParseStatus(s, 0x13, &back), "status from another address must fail");

  uint8_t uid[OTB_UID_BYTES];
  for (auto &u : uid) u = (uint8_t)rnd();
  uint8_t info[OTB_INFO_LEN], board = 9, uidBack[OTB_UID_BYTES];
  otbBuildInfo(info, 0x11, OTB_BOARD_NODE16, uid);
  CHECK(otbParseInfo(info, 0x11, &board, uidBack) && board == OTB_BOARD_NODE16 &&
        memcmp(uid, uidBack, sizeof(uid)) == 0, "info round trip");
  info[5] ^= 1;
  CHECK(!otbParseInfo(info, 0x11, &board, uidBack), "corrupted info must fail");

  uint8_t udid[OTB_UDID_LEN];
  otbBuildUdid(udid, uid);
  CHECK(otbUdidUidHash(udid) == otbCrc32(uid, OTB_UID_BYTES), "UDID carries the UID hash");
  CHECK(otbCrc32((const uint8_t *)"123456789", 9) == 0xCBF43926UL, "CRC-32 check value");
  const uint8_t check[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  CHECK(otbCrc8Cont(0, check, 9) == 0xF4, "CRC-8/SMBus check value");
}

// ----- the text streams under loss -------------------------------------------
//
// A node's two rings and the controller's bookkeeping, joined by a "bus" that
// loses a random 20% of transactions in each phase. Whatever is lost, each
// direction must deliver exactly the bytes that were sent, in order, once.

static void testTextStreams() {
  const uint8_t addr = 0x14;
  static uint8_t txMem[2048], rxMem[256];
  OtbTxRing tx = {txMem, sizeof(txMem), 0, 0};
  OtbRxRing rx = {rxMem, sizeof(rxMem), 0, 0};

  // Scaled past the 16-bit offsets, so their wrap is exercised.
  std::string nodeSays, hostSays;
  for (int i = 0; nodeSays.size() < 150000; ++i) nodeSays += "R,PARAM," + std::to_string(i) + ",X,0,100," + std::to_string(rnd() % 999) + "\n";
  for (int i = 0; hostSays.size() < 140000; ++i) hostSays += "D" + std::to_string(i) + "," + std::string(rnd() % 120, 'A') + "\n";

  size_t pushed = 0, sentHost = 0;
  uint16_t ctrlRx = 0;                 // controller: node text received
  uint16_t ctrlTxConfirmed = 0;        // controller: host text the node has taken
  std::string gotAtController, gotAtNode;
  auto lost = []() { return rnd() % 5u == 0; };

  for (int round = 0; round < 400000 && (gotAtController.size() < nodeSays.size() || gotAtNode.size() < hostSays.size()); ++round) {
    // Node loop: push some reply text (a whole line at a time), read commands.
    if (pushed < nodeSays.size()) {
      const size_t end = nodeSays.find('\n', pushed) + 1;
      if (otbTxPush(&tx, (const uint8_t *)nodeSays.data() + pushed, (uint16_t)(end - pushed))) pushed = end;
    }
    for (int k = 0; k < 40; ++k) { const int c = otbRxRead(&rx); if (c < 0) break; gotAtNode += (char)c; }

    // STATUS, as the node builds it and the controller reads it.
    OtbStatus st = {};
    st.txTail = (uint16_t)tx.tail;
    st.txPending = (uint16_t)otbTxUsed(&tx);
    st.rxAccepted = (uint16_t)rx.head;
    const uint16_t f = otbRxFree(&rx);
    st.rxFree = (uint8_t)(f > 255 ? 255 : f);
    uint8_t sbuf[OTB_STATUS_LEN];
    otbBuildStatus(sbuf, addr, &st);
    if (lost()) continue;
    OtbStatus seen;
    CHECK(otbParseStatus(sbuf, addr, &seen), "status parse in stream test");
    ctrlTxConfirmed = seen.rxAccepted;

    // Node -> controller: GET_TEXT with the controller's ack.
    const uint16_t unread = (uint16_t)(seen.txTail + seen.txPending - ctrlRx);
    if (unread || ctrlRx != seen.txTail) {
      const uint8_t max = (uint8_t)(unread < OTB_TEXT_MAX ? unread : OTB_TEXT_MAX);
      uint8_t req[OTB_GET_TEXT_REQ_LEN];
      otbBuildGetTextReq(req, addr, ctrlRx, max);
      if (!lost()) {
        uint16_t ack; uint8_t m;
        CHECK(otbParseGetTextReq(req, sizeof(req), addr, &ack, &m), "get-text request parse");
        otbTxAck(&tx, ack);
        uint8_t bytes[OTB_TEXT_MAX]; uint16_t off;
        const uint8_t n = otbTxPeek(&tx, bytes, m, &off);
        uint8_t resp[OTB_GET_TEXT_RESP_LEN(OTB_TEXT_MAX)];
        otbBuildGetTextResp(resp, addr, off, bytes, n, m);
        if (!lost()) {
          uint16_t roff; const uint8_t *rb; uint8_t rlen;
          CHECK(otbParseGetTextResp(resp, addr, m, &roff, &rb, &rlen), "get-text response parse");
          uint8_t skip = 0;
          const uint8_t fresh = otbTextNew(&ctrlRx, roff, rlen, &skip);
          gotAtController.append((const char *)rb + skip, fresh);
        }
      }
    }

    // Controller -> node: resend from what the node has confirmed.
    const size_t confirmedAbs = sentHost + (uint16_t)(ctrlTxConfirmed - (uint16_t)sentHost);
    sentHost = confirmedAbs;
    if (sentHost < hostSays.size() && seen.rxFree > 0) {
      const size_t room = seen.rxFree < OTB_TEXT_MAX ? seen.rxFree : OTB_TEXT_MAX;
      const uint8_t n = (uint8_t)std::min(room, hostSays.size() - sentHost);
      uint8_t w[OTB_TEXT_WRITE_LEN(OTB_TEXT_MAX)];
      otbBuildText(w, addr, (uint16_t)sentHost, (const uint8_t *)hostSays.data() + sentHost, n);
      if (!lost()) {
        uint16_t off; const uint8_t *b; uint8_t l;
        CHECK(otbParseText(w, OTB_TEXT_WRITE_LEN(n), addr, &off, &b, &l), "text write parse");
        otbRxAccept(&rx, off, b, l);
        if (rnd() % 7u == 0) otbRxAccept(&rx, off, b, l);   // a duplicate must change nothing
      }
    }
  }
  CHECK(gotAtController == nodeSays, "node->controller stream: %zu of %zu bytes, first difference at %zu",
        gotAtController.size(), nodeSays.size(),
        (size_t)(std::mismatch(gotAtController.begin(), gotAtController.end(), nodeSays.begin()).first -
                 gotAtController.begin()));
  CHECK(gotAtNode == hostSays, "controller->node stream: %zu of %zu bytes", gotAtNode.size(), hostSays.size());

  // The overlap case the exchange above never produces (the node always takes
  // an ack before it serves), checked directly: a response that repeats bytes
  // the controller already holds yields only the new ones.
  uint16_t expected = 65530;
  uint8_t skip = 99;
  CHECK(otbTextNew(&expected, 65526, 10, &skip) == 6 && skip == 4 && expected == 0, "overlap across the wrap");
  CHECK(otbTextNew(&expected, 0, 5, &skip) == 5 && skip == 0 && expected == 5, "exact continuation");
  CHECK(otbTextNew(&expected, 9, 5, &skip) == 0 && expected == 5, "a gap is not accepted");
  CHECK(otbTextNew(&expected, 65000, 5, &skip) == 0 && expected == 5, "long-gone bytes are not accepted");
  uint8_t mem[16];
  OtbTxRing r = {mem, sizeof(mem), 0, 0};
  CHECK(otbTxPush(&r, (const uint8_t *)"0123456789", 10) && !otbTxPush(&r, (const uint8_t *)"abcdefg", 7),
        "a push that does not fit is refused whole");
  otbTxAck(&r, 20);
  CHECK(r.tail == 0, "an ack beyond what exists changes nothing");
  otbTxAck(&r, 4);
  CHECK(r.tail == 4, "an ack frees exactly what it covers");
  otbTxAck(&r, 2);
  CHECK(r.tail == 4, "an ack from the past changes nothing");

  // A corrupted TEXT write is refused, not half-taken.
  uint8_t w[OTB_TEXT_WRITE_LEN(4)];
  otbBuildText(w, addr, 0, (const uint8_t *)"abc\n", 4);
  w[5] ^= 0x20;
  uint16_t off; const uint8_t *b; uint8_t l;
  CHECK(!otbParseText(w, sizeof(w), addr, &off, &b, &l), "corrupted text write must fail its PEC");
}

int main(int argc, char **argv) {
  FILE *fixture = nullptr;
  if (argc > 1) {
    fixture = std::fopen(argv[1], "w");
    if (!fixture) { std::perror(argv[1]); return 2; }
  }
  testPack();
  testFrames(fixture);
  testSmallFrames();
  testVibDedupe();
  testTextStreams();
  if (fixture) std::fclose(fixture);
  std::printf("%s: %d checks, %d failures\n", failures ? "FAIL" : "ok", checks, failures);
  return failures ? 1 : 0;
}
