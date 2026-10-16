// Daisy chain: three 30-pin pairs (the real KiCad boards, stacked as in make_board_media.py) are
// chained with 4-wire JST-SH cables (5 V, SDA, SCL, GND), board to board and to a Teensy 4.0, which
// is the only USB connection to the computer. Close up, the plugs go into the boards' sockets; then
// the view pulls back to the whole chain. Each board's LED takes its node's colour, as `opentac view`
// sets it, and the controller reads the nodes in turn: a pulse runs from each board along the bus and
// up the USB cable to that node's panel. The presses in the panels are simulated. The Teensy is a
// plain box with the board's dimensions.
import * as THREE from "three";
import { C, W, H, DPR, overlay, text, roundRect, start, inferno, seg, ease, clamp, lerp, pairMaker } from "./common.js";

const NODE = [[255, 64, 64], [64, 255, 64], [64, 128, 255]];        // opentac/view.py PALETTE
const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const X_BOARDS = [-18, 30, 78], X_TEENSY = -84;

// ------------------------------------------------------------------ scene
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(DPR); renderer.setSize(W, H);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.domElement.style.zIndex = 1;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(C.bg);
const camera = new THREE.PerspectiveCamera(22, W / H, 10, 3000);
camera.position.set(-12, 300, 330);
camera.lookAt(-12, 0, 0);
camera.setViewOffset(W, H, 0, -262, W, H);          // the chain sits in the lower part of the frame
scene.add(new THREE.HemisphereLight(0xffffff, 0xd8d2c4, 1.4));
const sun = new THREE.DirectionalLight(0xffffff, 2.0);
sun.position.set(-40, 420, 140); sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
Object.assign(sun.shadow.camera, { left: -180, right: 180, top: 120, bottom: -120, near: 50, far: 800 });
sun.shadow.bias = -0.0004; sun.shadow.radius = 5;
scene.add(sun);
const fill = new THREE.DirectionalLight(0xeef2ff, 0.6); fill.position.set(200, 120, 200); scene.add(fill);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), new THREE.MeshStandardMaterial({ color: C.bg, roughness: 1 }));
floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
const FLOOR_Y = 0;

const mat = (color, o = {}) => new THREE.MeshStandardMaterial(Object.assign({ color, roughness: 0.6, metalness: 0 }, o));
function box(w, h, d, m, x = 0, y = 0, z = 0) {
  const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);
  b.position.set(x, y, z); b.castShadow = b.receiveShadow = true;
  return b;
}

// ---- the 30-pin pairs
const { makePair, parts: P } = await pairMaker(THREE);
const pairs = X_BOARDS.map((x, k) => {
  const { pair, toPair } = makePair();
  pair.position.set(x, FLOOR_Y, 0);
  scene.add(pair);
  const led = toPair(P.LED1, 1.7);
  // LED: a small lit dome and a soft halo, off until the board is set up
  const dome = new THREE.Mesh(new THREE.SphereGeometry(1.4, 16, 12), new THREE.MeshBasicMaterial({ color: 0x222222, toneMapped: false }));
  dome.position.copy(led); pair.add(dome);
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTexture(), color: new THREE.Color(...NODE[k].map(v => v / 255)),
    transparent: true, opacity: 0, depthWrite: false, depthTest: false, toneMapped: false }));
  halo.scale.set(20, 20, 1); halo.position.copy(led).add(new THREE.Vector3(0, 0.8, 0)); pair.add(halo);
  const cn1 = toPair(P.CN1, 4.3).add(pair.position), cn2 = toPair(P.CN2, 4.3).add(pair.position);   // world, as set down
  return { k, pair, dome, halo, cn1, cn2 };
});
function haloTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const g = c.getContext("2d"), r = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  r.addColorStop(0, "rgba(255,255,255,0.95)"); r.addColorStop(0.3, "rgba(255,255,255,0.45)"); r.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = r; g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

// ---- Teensy 4.0: 35.56 x 17.78 mm, as a plain box; micro-USB at the left end, I2C wires at the right end
const teensy = new THREE.Group(); teensy.position.set(X_TEENSY, 0, 0); scene.add(teensy);
teensy.add(box(35.56, 1.6, 17.78, mat("#1f5a3c", { roughness: 0.5 }), 0, 2.6, 0));
teensy.add(box(10, 1.2, 10, mat("#16181b", { roughness: 0.45 }), 2, 4.0, 0));
teensy.add(box(5.6, 2.7, 7.6, mat("#b8bcc2", { metalness: 0.8, roughness: 0.3 }), -16.4, 4.8, 0));
for (let i = 0; i < 14; i++) for (const s of [-1, 1]) {
  const pad = new THREE.Mesh(new THREE.CylinderGeometry(0.85, 0.85, 0.2, 16), mat("#c9a646", { metalness: 0.8, roughness: 0.35 }));
  pad.position.set(-16.5 + i * 2.54, 3.45, s * 7.62); teensy.add(pad);
}
for (const s of [-1, 1]) teensy.add(box(35.56, 2.2, 2.4, mat("#1b1b1d"), 0, 0.9, s * 7.62));   // header rows underneath
const TEENSY_WIRES = new THREE.Vector3(X_TEENSY + 16.0, 3.6, 0);
const TEENSY_USB = new THREE.Vector3(X_TEENSY - 19.4, 4.8, 0);
// USB plug (micro-B), inserted at t_usb
const usbPlug = new THREE.Group();
usbPlug.add(box(6.0, 2.2, 7.0, mat("#b8bcc2", { metalness: 0.8, roughness: 0.3 }), -3, 0, 0));
usbPlug.add(box(9.0, 6.0, 10.5, mat("#2a2d31", { roughness: 0.7 }), -10.5, 0, 0));
scene.add(usbPlug);

// ---- JST-SH cables: four wires (5 V, SDA, SCL, GND), plug housings at the board ends. A plug is
// held HOVER mm above its socket until it is pushed in; the wires follow it.
const WIRE_COL = ["#d83a34", "#2f6fd0", "#e9c02a", "#202124"];
const HOVER = 10;
const wireMats = WIRE_COL.map(c => mat(c, { roughness: 0.45 }));
function cablePath(a, b, h) {
  const m = a.clone().add(b).multiplyScalar(0.5);
  const d = Math.sign(b.x - a.x);
  return new THREE.CatmullRomCurve3([a, a.clone().add(new THREE.Vector3(0, 6, 0)), a.clone().add(new THREE.Vector3(d * 7, h - 3, 0)),
    m.clone().setY(Math.max(a.y, b.y) + h), b.clone().add(new THREE.Vector3(-d * 7, h - 3, 0)), b.clone().add(new THREE.Vector3(0, 6, 0)), b]);
}
function makeCable(a, b, h, plugA) {
  const grp = new THREE.Group(), tubes = WIRE_COL.map((_, w) => { const m = new THREE.Mesh(new THREE.BufferGeometry(), wireMats[w]); grp.add(m); return m; });
  const plugs = [];
  for (const on of [plugA, true]) {
    if (!on) { plugs.push(null); continue; }
    const pl = box(3.4, 4.0, 5.8, mat("#ece4cc", { roughness: 0.55 })); grp.add(pl); plugs.push(pl);
  }
  scene.add(grp);
  const c = { grp, tubes, plugs, a, b, h, key: "", curve: null };
  setCable(c, 0, 0);
  return c;
}
// lift of each end (0 = seated); rebuild the wires only when an end moved
function setCable(c, la, lb) {
  const key = la.toFixed(3) + "/" + lb.toFixed(3);
  if (key === c.key) return;
  c.key = key;
  const a = c.a.clone().add(new THREE.Vector3(0, la, 0)), b = c.b.clone().add(new THREE.Vector3(0, lb, 0));
  c.curve = cablePath(a, b, c.h);
  c.tubes.forEach((m, w) => {
    const off = (w - 1.5) * 1.0;
    const pts = c.curve.getPoints(160).map(p => p.clone().add(new THREE.Vector3(0, 0, off)));
    m.geometry.dispose();
    m.geometry = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 160, 0.45, 8, false);
  });
  [a, b].forEach((p, i) => { if (c.plugs[i]) c.plugs[i].position.set(p.x, p.y + 1.4, p.z); });
}
const cables = [
  makeCable(TEENSY_WIRES, pairs[0].cn1, 18, false),
  makeCable(pairs[0].cn2, pairs[1].cn1, 15, true),
  makeCable(pairs[1].cn2, pairs[2].cn1, 15, true),
];
// solder joints at the Teensy end
for (let w = 0; w < 4; w++) teensy.add(box(1.4, 0.8, 0.9, mat("#c0c4c8", { metalness: 0.7, roughness: 0.3 }), 16.0, 3.6, (w - 1.5) * 1.0));

// ------------------------------------------------------------------ timeline
// close on board 0: the first cable goes in; pan: the second cable's two plugs; pull back to the
// chain (the third cable goes in); USB in; the computer's window comes in; read; then everything
// is undone in reverse, back to the first frame.
const PLUG = [[null, 0.45], [2.25, 2.75], [4.15, 4.35]];   // [end a, end b] push-in start, per cable (null: soldered at the Teensy)
const PUSH = 0.7;                                   // s for a plug to go in
const T_PAN = [1.35, 2.15], T_BACK = [3.55, 5.35];
const T_USB = 5.45, T_WIN = [6.05, 6.85], T_ON = 6.95, T_STREAM = 7.5;
const T_OFF = 12.2, T_WIN_OUT = [12.45, 13.0], T_LIFT = 12.85, T_IN = [13.15, 14.0];
const T_END = 14;
const ROUND = 0.45;                                 // s per node read (slowed down)

// camera: one viewing direction, three framings
const DIR = new THREE.Vector3(0, 300, 330).normalize();
const VIEWS = [
  { target: pairs[0].cn1.clone().add(new THREE.Vector3(6, 3, 0)), dist: 98 },
  { target: pairs[0].cn2.clone().lerp(pairs[1].cn1, 0.5).add(new THREE.Vector3(0, 3, 0)), dist: 104 },
  { target: new THREE.Vector3(-12, 4, 0), dist: 446 },
];
function setCamera(t) {
  let from = 0, to = 0, f = 0;
  if (t < T_PAN[0]) [from, to, f] = [0, 0, 0];
  else if (t < T_PAN[1]) [from, to, f] = [0, 1, ease((t - T_PAN[0]) / (T_PAN[1] - T_PAN[0]))];
  else if (t < T_BACK[0]) [from, to, f] = [1, 1, 0];
  else if (t < T_BACK[1]) [from, to, f] = [1, 2, ease((t - T_BACK[0]) / (T_BACK[1] - T_BACK[0]))];
  else if (t < T_IN[0]) [from, to, f] = [2, 2, 0];
  else [from, to, f] = [2, 0, ease((t - T_IN[0]) / (T_IN[1] - T_IN[0]))];
  const A = VIEWS[from], B = VIEWS[to];
  const target = A.target.clone().lerp(B.target, f), dist = Math.exp(lerp(Math.log(A.dist), Math.log(B.dist), f));
  camera.position.copy(target).addScaledVector(DIR, dist);
  camera.lookAt(target);
  // the window pushes the chain into the lower part of the frame while it is on screen
  const win = seg(t, T_WIN[0], T_WIN[1]) * (1 - seg(t, T_WIN_OUT[0], T_WIN_OUT[1]));
  camera.setViewOffset(W, H, 0, -262 * win, W, H);
  camera.updateMatrixWorld();
  const wide = from === 2 && to === 2 ? 1 : from === 2 || to === 2 ? (from === 2 ? 1 - f : f) : 0;
  return { win, wide };
}
const pushed = (t, t0) => t0 === null ? 1 : ease((t - t0) / PUSH) * (1 - ease((t - T_LIFT) / 0.4));

// ------------------------------------------------------------------ 2D overlay
const ctx = overlay(2);
const WIN = { x: 40, y: 22, w: 1200, h: 420 };
const G = 12;                                       // taxels per side of the simulated sensors

function contact(k, i, j, t) {                      // simulated presses, a different one per node
  const g = (x, y, s) => Math.exp(-(x * x + y * y) / (2 * s * s));
  if (k === 0) {
    const p = 0.55 + 0.45 * Math.sin(2 * Math.PI * t / 3);
    return p * g(i - (5.5 + 3 * Math.sin(1.3 * t)), j - (5.5 + 2.5 * Math.cos(0.9 * t)), 1.7);
  }
  if (k === 1) {
    const s = 3 + 1.5 * Math.sin(2 * Math.PI * t / 4);
    return 0.85 * (g(i - 5.5, j - (5.5 - s), 1.4) + g(i - 5.5, j - (5.5 + s), 1.4));
  }
  const y = 5.5 + 4.5 * Math.sin(2 * Math.PI * t / 2.6);
  return 0.9 * g((i - 2 - 0.65 * j) * 0.9, (j - y) * 0.6, 1.3) * (0.7 + 0.3 * Math.cos(t));
}

function project(v) {
  const p = v.clone().project(camera);
  return [(p.x + 1) / 2 * W, (1 - p.y) / 2 * H];
}
function usbPath(winY) {                            // from the Teensy's USB plug, out and up to the window
  const a = project(new THREE.Vector3(X_TEENSY - 34, 4.8, 0)), e = [150, winY + WIN.h];
  const c1 = [a[0] - 70, a[1] + 10], c2 = [e[0], Math.max(e[1], 0) + 150], out = [];
  for (let i = 0; i <= 40; i++) {
    const f = i / 40, g = 1 - f;
    out.push([0, 1].map(d => g * g * g * a[d] + 3 * g * g * f * c1[d] + 3 * g * f * f * c2[d] + f * f * f * e[d]));
  }
  return [project(TEENSY_USB)].concat(out);
}
function busPath(k, winY) {                         // node k: back along each cable to the Teensy, then USB
  const pts3 = [];
  for (let c = k; c >= 0; c--) {
    if (c < k) pts3.push(pairs[c].cn2.clone(), pairs[c].cn1.clone());
    pts3.push(...cables[c].curve.getPoints(80).reverse());
  }
  pts3.push(TEENSY_WIRES.clone(), TEENSY_USB.clone());
  return pts3.map(project).concat(usbPath(winY).slice(1));
}
function polyAt(pts, f) {
  const L = [0];
  for (let i = 1; i < pts.length; i++) L.push(L[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const s = f * L[L.length - 1];
  let i = 1; while (i < L.length - 1 && L[i] < s) i++;
  const u = (s - L[i - 1]) / Math.max(1e-6, L[i] - L[i - 1]);
  return [lerp(pts[i - 1][0], pts[i][0], u), lerp(pts[i - 1][1], pts[i][1], u)];
}
function drawPoly(pts, f, color, width) {
  const n = Math.max(2, Math.round(pts.length * clamp(f)));
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.beginPath(); ctx.moveTo(...pts[0]); for (let i = 1; i < n; i++) ctx.lineTo(...pts[i]); ctx.stroke(); ctx.restore();
}
function label(s, x, y, a, opts = {}) { text(ctx, s, x, y, Object.assign({ size: 32, color: C.ink, align: "center", alpha: a }, opts)); }

function renderAt(t) {
  const { win, wide } = setCamera(t);
  const out = 1 - seg(t, T_OFF, T_OFF + 0.3);       // LEDs, panels and node labels go off at the end
  const on = k => seg(t, T_ON + 0.18 * k, T_ON + 0.18 * k + 0.3) * out;
  pairs.forEach(pr => {
    const a = on(pr.k);
    pr.dome.material.color.setRGB(...(a > 0.5 ? NODE[pr.k].map(v => v / 255) : [0.13, 0.13, 0.13]));
    pr.halo.material.opacity = a * 0.9;
  });
  cables.forEach((c, k) => setCable(c, HOVER * (1 - pushed(t, PLUG[k][0])), HOVER * (1 - pushed(t, PLUG[k][1]))));
  const usbIn = seg(t, T_USB, T_USB + 0.4) * (1 - seg(t, T_WIN_OUT[0], T_WIN_OUT[0] + 0.3));
  usbPlug.position.copy(TEENSY_USB).add(new THREE.Vector3(-14 * (1 - usbIn), 0, 0));
  usbPlug.visible = usbIn > 0;
  renderer.render(scene, camera);

  ctx.clearRect(0, 0, W, H);
  const winY = lerp(-WIN.h - 40, 22, win);
  // USB cable, in the frame's plane, from the plug up to the computer's window (above the frame until it comes in)
  const usbGrow = seg(t, T_USB + 0.25, T_USB + 0.7) * (1 - seg(t, T_WIN_OUT[0], T_WIN_OUT[0] + 0.3));
  if (usbGrow > 0) drawPoly(usbPath(winY).slice(1), usbGrow, "#2a2d31", 11);
  // computer window: `opentac view`, one panel per node
  if (win > 0) {
    ctx.save();
    roundRect(ctx, WIN.x, winY, WIN.w, WIN.h, 14); ctx.fillStyle = "#161c23"; ctx.fill();
    ctx.restore();
    text(ctx, "computer: opentac view", WIN.x + 26, winY + 42, { size: 30, color: "#aab3bd", weight: 500 });
  }
  const PAN = k => ({ x: WIN.x + 24 + k * 392, y: winY + 62, w: 368, h: 340 });
  pairs.forEach((pr, k) => {
    const a = seg(t, T_ON + 0.18 * k, T_ON + 0.18 * k + 0.4) * out, R = PAN(k);
    if (a <= 0 || win <= 0) return;
    const cs = (R.h - 50) / G, gx = R.x + (R.w - cs * G) / 2, gy = R.y + 46;
    text(ctx, `node ${k}`, gx - 4, R.y + 32, { size: 32, weight: 700, color: rgb(NODE[k]), alpha: a });
    ctx.save(); ctx.globalAlpha = a;
    ctx.fillStyle = "#000004"; ctx.fillRect(gx - 4, gy - 4, cs * G + 8, cs * G + 8);
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      ctx.fillStyle = inferno(clamp(contact(k, i, j, t)));
      ctx.fillRect(gx + j * cs + 1, gy + i * cs + 1, cs - 2, cs - 2);
    }
    ctx.restore();
  });
  // bus traffic: the controller reads node 0, 1, 2, 0, ... each pulse runs to the window
  if (t > T_STREAM && t < T_OFF) {
    const r = (t - T_STREAM) / ROUND, k = Math.floor(r) % 3, f = r - Math.floor(r), path = busPath(k, winY);
    const [px, py] = polyAt(path, f);
    ctx.save();
    ctx.lineCap = "round";
    for (let i = 1; i <= 8; i++) {                  // a short trail behind the pulse
      const [ax, ay] = polyAt(path, Math.max(0, f - 0.012 * i)), [bx2, by2] = polyAt(path, Math.max(0, f - 0.012 * (i - 1)));
      ctx.strokeStyle = rgb(NODE[k], 0.75 * (1 - i / 9)); ctx.lineWidth = 9 * (1 - i / 12);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx2, by2); ctx.stroke();
    }
    const g = ctx.createRadialGradient(px, py, 0, px, py, 22);
    g.addColorStop(0, rgb(NODE[k], 1)); g.addColorStop(0.35, rgb(NODE[k], 0.7)); g.addColorStop(1, rgb(NODE[k], 0));
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(px, py, 22, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = "#ffffff"; ctx.beginPath(); ctx.arc(px, py, 5, 0, 2 * Math.PI); ctx.fill();
    ctx.restore();
  }
  // the panel of each node whose pulse just arrived: an outline that fades over half a second
  if (t > T_STREAM) {
    const done = Math.floor((Math.min(t, T_OFF) - T_STREAM) / ROUND);
    for (let m = done - 1; m >= Math.max(0, done - 3); m--) {
      const a = 0.9 * clamp(1 - (t - (T_STREAM + ROUND * (m + 1))) / 0.5) * out, R = PAN(m % 3);
      if (a <= 0) continue;
      ctx.save(); ctx.strokeStyle = rgb(NODE[m % 3], a); ctx.lineWidth = 4;
      roundRect(ctx, R.x - 12, R.y - 6, R.w + 24, R.h + 12, 10); ctx.stroke(); ctx.restore();
    }
  }
  // labels: the cable close up; the parts in the wide view
  const cl = 1 - seg(wide, 0, 0.3);
  if (cl > 0) {
    ctx.save(); ctx.globalAlpha = cl * 0.9; roundRect(ctx, W / 2 - 320, 868, 640, 60, 12); ctx.fillStyle = C.bg; ctx.fill(); ctx.restore();
    label("JST-SH cable: 5 V, SDA, SCL, GND", W / 2, 910, cl, { size: 36, color: C.ink });
  }
  const tl = project(new THREE.Vector3(X_TEENSY, 0, 16));
  label("Teensy 4.0", tl[0], tl[1] + 40, seg(wide, 0.7, 1));
  pairs.forEach((pr, k) => {
    const p = project(new THREE.Vector3(X_BOARDS[k], 0, 16)), a = on(k) * seg(wide, 0.7, 1);
    if (a <= 0) return;
    ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = rgb(NODE[k]);
    ctx.beginPath(); ctx.arc(p[0] - 62, p[1] + 29, 10, 0, 2 * Math.PI); ctx.fill(); ctx.restore();
    label(`node ${k}`, p[0] + 12, p[1] + 40, a);
  });
  const e = project(new THREE.Vector3(X_BOARDS[2] + 14, 34, 0));
  label("… up to 18 boards", Math.min(e[0] + 40, W - 40), e[1], seg(wide, 0.7, 1) * seg(t, T_BACK[1] - 0.3, T_BACK[1] + 0.2) * (1 - seg(t, T_WIN_OUT[0], T_WIN_OUT[1])), { size: 30, color: C.muted, align: "right" });
  const u = usbPath(winY)[20];
  label("USB", u[0] + 22, u[1] + 10, seg(t, T_USB + 0.5, T_USB + 0.9) * (1 - seg(t, T_WIN_OUT[0], T_WIN_OUT[0] + 0.2)), { size: 32, align: "left" });
}

start(renderAt, T_END);
