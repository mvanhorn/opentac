// OpenTac bus v2 controller -- Teensy 4.0. Bus on Wire: SDA 18, SCL 19, 1 MHz.
//
// Talks to 1..18 bus nodes (tactile_imu_fast_cfg built with -DOT_BUS_NODE)
// and to the laptop over USB. It adds almost nothing of its own: every node
// speaks the single-board command surface, and this forwards it.
//
// USB, laptop -> controller, one line each:
//   @<n> <line>   send <line> to node n, exactly as it would be typed to a USB
//                 board: `@0 ?`, `@2 p 29 64`, `@1 S1045`, ...
//   N             list nodes: R,NODES,<count>, then per slot
//                 R,NODE,<n>,<addr>,<uid>,<board>,<online>, then R,END
//   ?             R,MCAPS,<bus version>,<max nodes>,<bus hz>, then R,END
//   t / z         bus statistics (R,BUS,...) / zero them (R,OK,z)
//   e             forget every node and enumerate again (R,OK,e,<count>)
//   s             bring-up: SDA/SCL idle levels, then every address that ACKs
//
// USB, controller -> laptop:
//   <n>|<line>             a line node n wrote: its replies (R,...), status (M,...)
//                          and notices (#...), untouched
//   <n>,<t_us>|F,...       a frame, as a USB board prints it; X for xampling
//   <n>,<t_us>|V,...       a vibration sample
//   J,<n>,<uid>,<board>,<how>   node n joined (how: new / rejoin)
//   L,<n>                  node n stopped answering
//   R,... and # ...        the controller's own replies and notices
//
// <t_us> is this controller's clock when the node published the frame -- one
// clock for every node, so frames from different sensors share a timeline.
//
// Build: v2/build.sh (or arduino-cli compile --fqbn
//   'teensy:avr:teensy40:usb=serial,speed=600,opt=o2std' --libraries <Arduino>/libraries v2/ot2_master)

#include <Wire.h>
#include <ot_bus.h>

static const uint32_t HOTPLUG_US = 2000000;      // look for new boards this often
static const uint32_t PING_OFFLINE_US = 1000000; // and knock on silent ones
static const uint8_t FAILS_OFFLINE = 8;          // consecutive failures before L
// A commit (E) or erase (X) writes the node's flash, which stalls its CPU for
// tens of milliseconds -- interrupts included, so an I2C transaction addressed
// to it would clock-stretch the whole bus. Leave it, and ARP (every node
// answers 0x61), alone for this long after sending one.
static const uint32_t FLASH_QUIET_US = 300000;

#define CMD_QUEUE 4
#define LINE_MAX 256

struct Node {
  bool used;                  // this slot belongs to a board we have seen
  bool online;
  uint8_t addr;
  uint8_t uid[OTB_UID_BYTES];
  uint8_t board;
  uint8_t fails;
  uint32_t quietUntilUs;
  uint32_t lastPingUs;

  // laptop -> node: whole lines, sent as a byte stream
  char queue[CMD_QUEUE][LINE_MAX];
  uint16_t queueLen[CMD_QUEUE];
  uint8_t queueHead, queueCount;
  uint16_t txOff;             // stream offset of queue[queueHead][0]

  // node -> laptop: a byte stream, cut back into lines here
  uint16_t rxOff;             // bytes received so far
  char line[LINE_MAX];
  uint16_t lineLen;

  // frames
  bool haveSeq;
  uint16_t lastSeq;
  bool vibKnown;
  uint32_t vibNext;
  uint8_t blob[OTB_BLOB_MAX(OTB_MAX_TAXELS)];

  uint32_t frames, torn, pec, nack;
};

static Node nodes[OTB_MAX_NODES];
static uint8_t slots = 0;                 // slots in use, online or not
static uint32_t arpQuietUntilUs = 0;
static uint32_t lastHotplugUs = 0;

static uint32_t statTxns = 0, statNack = 0, statPec = 0, statTextIn = 0, statTextOut = 0;
static uint32_t statRoundUs = 0, statRoundUsMax = 0;

static const char *const BOARD_NAMES[] = {"READOUT32", "NODE16"};

// ----- time -------------------------------------------------------------------

static uint32_t lastMicros = 0;
static uint64_t microsHigh = 0;

static uint64_t micros64() {
  const uint32_t now = micros();
  if (now < lastMicros) microsHigh += 0x100000000ULL;
  lastMicros = now;
  return microsHigh + now;
}

static bool before(uint32_t now, uint32_t until) { return (int32_t)(now - until) < 0; }

// ----- USB out ----------------------------------------------------------------

static void usbSink(void *, const char *text, uint16_t len) {
  Serial.write((const uint8_t *)text, len);
}

static void emitHex(const uint8_t *b, uint8_t n) {
  static const char H[] = "0123456789ABCDEF";
  for (uint8_t i = 0; i < n; ++i) {
    Serial.write(H[b[i] >> 4]);
    Serial.write(H[b[i] & 0x0F]);
  }
}

static void emitJoin(uint8_t i, bool rejoin) {
  Serial.print("J,");
  Serial.print(i);
  Serial.print(',');
  emitHex(nodes[i].uid, OTB_UID_BYTES);
  Serial.print(',');
  Serial.print(nodes[i].board < 2 ? BOARD_NAMES[nodes[i].board] : "?");
  Serial.println(rejoin ? ",rejoin" : ",new");
}

// ----- bus primitives -----------------------------------------------------------

static bool busWrite(uint8_t addr, const uint8_t *bytes, uint8_t len) {
  Wire.beginTransmission(addr);
  Wire.write(bytes, len);
  const bool ok = Wire.endTransmission() == 0;
  statTxns++;
  if (!ok) statNack++;
  return ok;
}

// Command (and arguments) written, repeated START, `outLen` bytes read. The
// caller checks the PEC, so a corrupted transaction is a failure it can retry.
static bool busCmdRead(uint8_t addr, const uint8_t *req, uint8_t reqLen, uint8_t *out, uint8_t outLen) {
  Wire.beginTransmission(addr);
  Wire.write(req, reqLen);
  statTxns++;
  if (Wire.endTransmission(false) != 0) { statNack++; return false; }
  if (Wire.requestFrom(addr, outLen) != outLen) {
    while (Wire.available()) Wire.read();
    statNack++;
    return false;
  }
  for (uint8_t i = 0; i < outLen; ++i) out[i] = (uint8_t)Wire.read();
  return true;
}

static bool readInfo(uint8_t addr, uint8_t *board, uint8_t *uid) {
  uint8_t s[OTB_INFO_LEN];
  statTxns++;
  if (Wire.requestFrom(addr, (uint8_t)OTB_INFO_LEN) != OTB_INFO_LEN) {
    while (Wire.available()) Wire.read();
    return false;
  }
  for (uint8_t i = 0; i < OTB_INFO_LEN; ++i) s[i] = (uint8_t)Wire.read();
  return otbParseInfo(s, addr, board, uid);
}

static bool readStatus(uint8_t addr, OtbStatus *st) {
  const uint8_t cmd = OTB_CMD_STATUS;
  uint8_t s[OTB_STATUS_LEN];
  if (!busCmdRead(addr, &cmd, 1, s, OTB_STATUS_LEN)) return false;
  if (!otbParseStatus(s, addr, st)) { statPec++; return false; }
  return true;
}

// ----- SMBus ARP, as v1/ot_master.ino does it -------------------------------------

static bool arpSendByte(uint8_t command) {
  const uint8_t b[2] = {command, otbPecSendByte(OTB_ADDR_ARP, command)};
  return busWrite(OTB_ADDR_ARP, b, 2);
}

// Arbitrated by the nodes themselves; lowest UDID wins. A bus with nobody
// unresolved on it reads back all ones, which fails the count check.
static bool arpGetUdid(uint8_t *udid) {
  const uint8_t cmd = OTB_ARP_GET_UDID;
  uint8_t r[OTB_ARP_READ_LEN];
  if (!busCmdRead(OTB_ADDR_ARP, &cmd, 1, r, OTB_ARP_READ_LEN)) return false;
  if (r[0] != OTB_ARP_BLOCK_LEN) return false;
  if (r[OTB_ARP_READ_LEN - 1] != otbPecBlockRead(OTB_ADDR_ARP, OTB_ARP_GET_UDID, OTB_ARP_BLOCK_LEN, &r[1])) {
    return false;
  }
  memcpy(udid, &r[1], OTB_UDID_LEN);
  return true;
}

static bool arpAssign(const uint8_t *udid, uint8_t addr) {
  uint8_t b[OTB_ARP_WRITE_LEN];
  b[0] = OTB_ARP_ASSIGN;
  b[1] = OTB_ARP_BLOCK_LEN;
  memcpy(&b[2], udid, OTB_UDID_LEN);
  b[2 + OTB_UDID_LEN] = (uint8_t)(addr << 1);
  b[OTB_ARP_WRITE_LEN - 1] = otbPecBlockWrite(OTB_ADDR_ARP, OTB_ARP_ASSIGN, OTB_ARP_BLOCK_LEN, &b[2]);
  return busWrite(OTB_ADDR_ARP, b, OTB_ARP_WRITE_LEN);
}

// The node has just been given `addr`; it re-begins Wire from its loop, which
// may be partway through a scan. Give it up to 50 ms to answer there.
static bool awaitInfo(uint8_t addr, uint8_t *board, uint8_t *uid) {
  for (uint8_t tries = 0; tries < 10; ++tries) {
    delay(5);
    if (readInfo(addr, board, uid)) return true;
  }
  return false;
}

// Start the slot's streams where the node's are: its reply stream at the
// oldest byte it still holds (so a boot banner is not lost), its command
// stream at what it has taken.
static bool syncStreams(uint8_t i) {
  Node &n = nodes[i];
  OtbStatus st;
  if (!readStatus(n.addr, &st)) return false;
  n.rxOff = st.txTail;
  n.txOff = st.rxAccepted;
  n.lineLen = 0;
  n.haveSeq = false;
  n.vibKnown = false;
  n.fails = 0;
  return true;
}

// Takes the node that just won arbitration: back into its old slot if its UID
// hash is one we know (a reboot -- every reflash is one), else a new slot.
static bool adopt(const uint8_t *udid) {
  const uint32_t claimed = otbUdidUidHash(udid);
  uint8_t slot = slots;
  bool rejoin = false;
  for (uint8_t i = 0; i < slots; ++i) {
    if (otbCrc32(nodes[i].uid, OTB_UID_BYTES) == claimed) { slot = i; rejoin = true; break; }
  }
  if (slot >= OTB_MAX_NODES) return false;
  const uint8_t addr = (uint8_t)(OTB_ADDR_BASE + slot);
  if (!arpAssign(udid, addr)) return false;
  uint8_t board, uid[OTB_UID_BYTES];
  if (!awaitInfo(addr, &board, uid) || otbCrc32(uid, OTB_UID_BYTES) != claimed) {
    Serial.print("# node at 0x");
    Serial.print(addr, HEX);
    Serial.println(" did not answer after Assign Address");
    return false;
  }
  Node &n = nodes[slot];
  if (!rejoin) {
    memset(&n, 0, sizeof(n));
    slots++;
  }
  n.used = true;
  n.addr = addr;
  n.board = board;
  memcpy(n.uid, uid, OTB_UID_BYTES);
  n.queueCount = 0;           // a rebooted node answers nothing sent before it
  n.queueHead = 0;
  if (!syncStreams(slot)) return false;
  n.online = true;
  emitJoin(slot, rejoin);
  return true;
}

static uint8_t enumerate() {
  for (uint8_t i = 0; i < slots; ++i) {
    if (nodes[i].online) {
      nodes[i].online = false;
      Serial.print("L,");
      Serial.println(i);
    }
  }
  memset(nodes, 0, sizeof(nodes));
  slots = 0;
  // Reset drops every address, so a controller restart re-enumerates without
  // power-cycling the nodes.
  arpSendByte(OTB_ARP_RESET);
  delay(10);
  arpSendByte(OTB_ARP_PREPARE);
  delay(2);
  uint8_t udid[OTB_UDID_LEN];
  while (slots < OTB_MAX_NODES && arpGetUdid(udid)) {
    if (!adopt(udid)) break;
  }
  return slots;
}

// No Reset and no Prepare: a node that owns an address loses the arbitration
// by design, so only a new or rebooted one can answer.
static void hotplugScan() {
  uint8_t udid[OTB_UDID_LEN];
  if (arpGetUdid(udid)) adopt(udid);
}

// ----- per node ---------------------------------------------------------------

static void fail(uint8_t i) {
  Node &n = nodes[i];
  n.nack++;
  if (++n.fails >= FAILS_OFFLINE && n.online) {
    n.online = false;
    n.lastPingUs = micros();
    Serial.print("L,");
    Serial.println(i);
  }
}

static bool isFlashCommand(const char *line) { return line[0] == 'E' || line[0] == 'X'; }

// Commands: one chunk per round, resent from whatever the node reports taking.
static void sendText(uint8_t i, const OtbStatus &st) {
  Node &n = nodes[i];
  if (!n.queueCount) return;
  const char *line = n.queue[n.queueHead];
  const uint16_t len = n.queueLen[n.queueHead];
  const uint16_t taken = (uint16_t)(st.rxAccepted - n.txOff);
  if (taken >= len) {                        // the node has the whole line
    n.txOff = (uint16_t)(n.txOff + len);
    n.queueHead = (uint8_t)((n.queueHead + 1) % CMD_QUEUE);
    n.queueCount--;
    return;
  }
  if (st.rxFree == 0) return;
  uint16_t piece = (uint16_t)(len - taken);
  if (piece > OTB_TEXT_MAX) piece = OTB_TEXT_MAX;
  if (piece > st.rxFree) piece = st.rxFree;
  uint8_t w[OTB_TEXT_WRITE_LEN(OTB_TEXT_MAX)];
  const uint8_t wl = otbBuildText(w, n.addr, (uint16_t)(n.txOff + taken),
                                  (const uint8_t *)line + taken, (uint8_t)piece);
  if (!busWrite(n.addr, w, wl)) { fail(i); return; }
  statTextOut += piece;
  if (taken + piece == len && isFlashCommand(line)) {
    n.quietUntilUs = micros() + FLASH_QUIET_US;
    arpQuietUntilUs = n.quietUntilUs;
  }
}

static void emitNodeByte(uint8_t i, char c) {
  Node &n = nodes[i];
  if (c == '\r') return;
  if (c != '\n' && n.lineLen < LINE_MAX) { n.line[n.lineLen++] = c; return; }
  Serial.print(i);
  Serial.write('|');
  Serial.write((const uint8_t *)n.line, n.lineLen);
  Serial.write('\n');
  n.lineLen = 0;
  if (c != '\n') n.line[n.lineLen++] = c;   // an over-long line is cut, never lost
}

// Replies: everything past what we have, acknowledging what we have.
static void receiveText(uint8_t i, const OtbStatus &st) {
  Node &n = nodes[i];
  uint16_t ahead = (uint16_t)(n.rxOff - st.txTail);
  if (ahead > st.txPending) {               // cannot happen unless the node restarted
    n.rxOff = st.txTail;
    ahead = 0;
  }
  const uint16_t unread = (uint16_t)(st.txPending - ahead);
  if (!unread && n.rxOff == st.txTail) return;
  const uint8_t max = (uint8_t)(unread < OTB_TEXT_MAX ? unread : OTB_TEXT_MAX);
  uint8_t req[OTB_GET_TEXT_REQ_LEN];
  otbBuildGetTextReq(req, n.addr, n.rxOff, max);
  uint8_t resp[OTB_GET_TEXT_RESP_LEN(OTB_TEXT_MAX)];
  if (!busCmdRead(n.addr, req, sizeof(req), resp, (uint8_t)OTB_GET_TEXT_RESP_LEN(max))) { fail(i); return; }
  uint16_t off;
  const uint8_t *bytes;
  uint8_t len;
  if (!otbParseGetTextResp(resp, n.addr, max, &off, &bytes, &len)) { n.pec++; statPec++; fail(i); return; }
  uint8_t skip = 0;
  const uint8_t fresh = otbTextNew(&n.rxOff, off, len, &skip);
  for (uint8_t k = 0; k < fresh; ++k) emitNodeByte(i, (char)bytes[skip + k]);
  statTextIn += fresh;
}

// The newest frame, if it is new: every chunk, all from the same publish.
static void receiveFrame(uint8_t i, const OtbStatus &st) {
  Node &n = nodes[i];
  if (!(st.flags & OTB_SF_FRAME) || (n.haveSeq && st.frameSeq == n.lastSeq)) return;
  uint8_t req[OTB_CHUNK_REQ_LEN];
  uint8_t resp[OTB_CHUNK_RESP_LEN];
  OtbChunk c;
  uint16_t seq = 0, total = 0;
  uint8_t chunks = 0;
  uint64_t tUs = 0;
  for (uint8_t k = 0; k == 0 || k < chunks; ++k) {
    otbBuildChunkReq(req, n.addr, k);
    if (!busCmdRead(n.addr, req, sizeof(req), resp, OTB_CHUNK_RESP_LEN)) { fail(i); return; }
    if (!otbParseChunkResp(resp, n.addr, &c) || c.chunk != k) { n.pec++; statPec++; fail(i); return; }
    if (k == 0) {
      if (c.chunks == 0) return;
      seq = c.seq;
      total = c.total;
      chunks = c.chunks;
      tUs = micros64() - c.ageUs;
    } else if (c.seq != seq || c.total != total) {
      n.torn++;                              // republished mid-read: try again next round
      return;
    }
    memcpy(&n.blob[(uint16_t)k * OTB_CHUNK_PAYLOAD], c.payload, c.len);
  }
  n.fails = 0;
  n.haveSeq = true;
  n.lastSeq = seq;
  if (!n.vibKnown) {
    n.vibNext = otbFrameVibSeq(n.blob);
    n.vibKnown = true;
  }
  OtbOut out;
  otbOutInit(&out, usbSink, NULL);
  if (otbFormatFrame(n.blob, total, i, tUs, &n.vibNext, &out)) n.frames++;
  else n.torn++;
}

static void serviceNode(uint8_t i) {
  Node &n = nodes[i];
  const uint32_t now = micros();
  if (!n.online) {
    // Silent but perhaps still addressed (a long Z, a loose cable): knock.
    if ((uint32_t)(now - n.lastPingUs) >= PING_OFFLINE_US && !before(now, arpQuietUntilUs)) {
      n.lastPingUs = now;
      uint8_t board, uid[OTB_UID_BYTES];
      if (readInfo(n.addr, &board, uid) && memcmp(uid, n.uid, OTB_UID_BYTES) == 0 && syncStreams(i)) {
        n.online = true;
        emitJoin(i, true);
      }
    }
    return;
  }
  if (before(now, n.quietUntilUs)) return;
  OtbStatus st;
  if (!readStatus(n.addr, &st)) { fail(i); return; }
  n.fails = 0;
  receiveText(i, st);
  sendText(i, st);
  if (!before(micros(), n.quietUntilUs)) receiveFrame(i, st);
}

// ----- USB in ------------------------------------------------------------------

static void listNodes() {
  Serial.print("R,NODES,");
  Serial.println(slots);
  for (uint8_t i = 0; i < slots; ++i) {
    Serial.print("R,NODE,");
    Serial.print(i);
    Serial.print(",0x");
    Serial.print(nodes[i].addr, HEX);
    Serial.print(',');
    emitHex(nodes[i].uid, OTB_UID_BYTES);
    Serial.print(',');
    Serial.print(nodes[i].board < 2 ? BOARD_NAMES[nodes[i].board] : "?");
    Serial.print(',');
    Serial.println(nodes[i].online ? 1 : 0);
  }
  Serial.println("R,END");
}

// A refusal of `@n ...` goes out in node n's own stream (`n|R,ERR,...`), where
// the host is waiting for that command's reply, rather than as a controller
// line it is not listening for.
static void refuse(uint16_t idx, const char *reason) {
  Serial.print(idx);
  Serial.print("|R,ERR,");
  Serial.println(reason);
}

static void handleHostLine(char *line) {
  if (line[0] == '@') {
    char *p = line + 1;
    uint16_t idx = 0;
    bool digits = false;
    while (*p >= '0' && *p <= '9' && idx < 1000) { idx = (uint16_t)(idx * 10u + (uint16_t)(*p++ - '0')); digits = true; }
    while (*p == ' ') ++p;
    if (!digits) { Serial.println("R,ERR,NONODE"); return; }
    if (idx >= slots) { refuse(idx, "NONODE"); return; }
    Node &n = nodes[idx];
    if (!n.online) { refuse(idx, "OFFLINE"); return; }
    const uint16_t len = (uint16_t)strlen(p);
    if (len + 1u > LINE_MAX) { refuse(idx, "LONG"); return; }
    if (n.queueCount == CMD_QUEUE) { refuse(idx, "BUSY"); return; }
    const uint8_t slot = (uint8_t)((n.queueHead + n.queueCount) % CMD_QUEUE);
    memcpy(n.queue[slot], p, len);
    n.queue[slot][len] = '\n';
    n.queueLen[slot] = (uint16_t)(len + 1u);
    n.queueCount++;
    return;
  }
  switch (line[0]) {
    case 'N':
      listNodes();
      break;
    case '?':
      Serial.print("R,MCAPS,");
      Serial.print(OTB_VERSION);
      Serial.print(',');
      Serial.print(OTB_MAX_NODES);
      Serial.print(',');
      Serial.println(OTB_BUS_HZ);
      Serial.println("R,END");
      break;
    case 't': {
      uint32_t frames = 0, torn = 0;
      for (uint8_t i = 0; i < slots; ++i) { frames += nodes[i].frames; torn += nodes[i].torn; }
      const uint32_t f[9] = {statTxns, statNack, statPec, frames, torn, statTextIn, statTextOut,
                             statRoundUs, statRoundUsMax};
      Serial.print("R,BUS");
      for (uint8_t k = 0; k < 9; ++k) { Serial.print(','); Serial.print(f[k]); }
      Serial.println();
      break;
    }
    case 'z':
      statTxns = statNack = statPec = statTextIn = statTextOut = statRoundUsMax = 0;
      for (uint8_t i = 0; i < slots; ++i) nodes[i].frames = nodes[i].torn = nodes[i].pec = nodes[i].nack = 0;
      Serial.println("R,OK,z");
      break;
    case 's': {
      // Bring-up: are the lines idle high (pull-ups present, nothing holding
      // them low), and does anything at all acknowledge an address?
      // Each line read three ways: plain, with the weak internal pull-up, and
      // with the weak pull-down. A line that follows the internal pull is
      // floating (no external pull-up); one that ignores it is being driven.
      Wire.end();
      {
        const uint8_t modes[3] = {INPUT, INPUT_PULLUP, INPUT_PULLDOWN};
        const char *names[3] = {"in", "pu", "pd"};
        Serial.print("R,LINES");
        for (uint8_t m = 0; m < 3; ++m) {
          pinMode(18, modes[m]);
          pinMode(19, modes[m]);
          delay(2);
          Serial.print(",");
          Serial.print(names[m]);
          Serial.print(":SDA=");
          Serial.print(digitalRead(18));
          Serial.print("/SCL=");
          Serial.print(digitalRead(19));
        }
        Serial.println();
        pinMode(18, INPUT);
        pinMode(19, INPUT);
      }
      Wire.begin();
      Wire.setClock(OTB_BUS_HZ);
      for (uint8_t a = 1; a < 0x78; ++a) {
        Wire.beginTransmission(a);
        if (Wire.endTransmission() == 0) {
          Serial.print("R,ACK,0x");
          Serial.println(a, HEX);
        }
      }
      Wire.setClock(100000);                  // and once more, slowly
      for (uint8_t a = 1; a < 0x78; ++a) {
        Wire.beginTransmission(a);
        if (Wire.endTransmission() == 0) {
          Serial.print("R,ACK100K,0x");
          Serial.println(a, HEX);
        }
      }
      Wire.setClock(OTB_BUS_HZ);
      Serial.println("R,END");
      break;
    }
    case 'e': {
      const uint8_t count = enumerate();
      Serial.print("R,OK,e,");
      Serial.println(count);
      break;
    }
    default:
      Serial.println("R,ERR,CMD");
      break;
  }
}

static char usbLine[LINE_MAX + 8];
static uint16_t usbLen = 0;
static bool usbOverflow = false;

static void pollUsb() {
  while (Serial.available()) {
    const char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (usbOverflow) Serial.println("R,ERR,LONG");
      else if (usbLen) { usbLine[usbLen] = '\0'; handleHostLine(usbLine); }
      usbLen = 0;
      usbOverflow = false;
    } else if (usbLen + 1u < sizeof(usbLine)) {
      usbLine[usbLen++] = c;
    } else {
      usbOverflow = true;
    }
  }
}

// ----- Arduino ------------------------------------------------------------------

void setup() {
  Serial.begin(115200);                 // USB: the baud rate is ignored
  Wire.begin();
  Wire.setClock(OTB_BUS_HZ);
  delay(200);                           // let the nodes finish booting
  const uint8_t count = enumerate();
  Serial.print("# OpenTac bus v2 controller: ");
  Serial.print(count);
  Serial.println(" node(s). Send N to list them, @<n> <command> to talk to one.");
  lastHotplugUs = micros();
}

void loop() {
  const uint32_t start = micros();
  pollUsb();
  for (uint8_t i = 0; i < slots; ++i) {
    serviceNode(i);
    pollUsb();                          // keep commands flowing on a long bus
  }
  const uint32_t now = micros();
  if ((uint32_t)(now - lastHotplugUs) >= HOTPLUG_US && !before(now, arpQuietUntilUs)) {
    lastHotplugUs = now;
    if (slots < OTB_MAX_NODES) hotplugScan();
  }
  statRoundUs = micros() - start;
  if (statRoundUs > statRoundUsMax) statRoundUsMax = statRoundUs;
}
