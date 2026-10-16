// Daisy chain: three 30-pin pairs (the real KiCad boards, stacked as in make_board_media.py) are
// set down, chained with 4-wire JST-SH cables (5 V, SDA, SCL, GND), board to board and to a Teensy
// 4.0, which is the only USB connection to the computer. Each board's LED takes its node's colour,
// as `opentac view` sets it, and the controller reads the nodes in turn: a pulse runs from each
// board along the bus and up the USB cable to that node's panel. The contacts are made up.
// The Teensy is a plain box with the board's dimensions.
import * as THREE from "three";
import { C, W, H, DPR, overlay, text, roundRect, start, inferno, seg, ease, clamp, lerp, loadBoard } from "./common.js";

const T_END = 12;
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

// ---- the 30-pin pair: bottom board, top board on its sockets
const bot = await loadBoard(THREE, "readout_30pin_bottom");
const top = await loadBoard(THREE, "readout_30pin_top");
const gap = (await (await fetch("/data/stack.json")).json()).gap;
const mid = (P, a, b) => [(P[a][0] + P[b][0]) / 2, (P[a][1] + P[b][1]) / 2];
const [bx, by] = mid(bot.meta.parts, "J2", "J3"), [tx, ty] = mid(top.meta.parts, "J1", "J3");
const [ox0, oy0, ox1, oy1] = bot.meta.outline;
const cx = (ox0 + ox1) / 2, cy = (oy0 + oy1) / 2;
const TOP_Y = 1.6 + gap;                            // top board's underside above the bottom board's

// one pair, in its own frame: KiCad y along +x (CN1 toward the Teensy), board up = +y
function makePair() {
  const pair = new THREE.Group(), inner = new THREE.Group();
  inner.rotation.y = Math.PI / 2;                   // KiCad y (three z) -> +x
  const b = bot.group.clone(), tp = top.group.clone();
  b.position.set(-cx, 0, -cy);
  tp.position.set(bx - tx - cx, TOP_Y, by - ty - cy);
  inner.add(b, tp);
  pair.add(inner);
  const toPair = (P, y) => { const v = new THREE.Vector3(P[0] + bx - tx - cx, TOP_Y + y, P[1] + by - ty - cy); return v.applyEuler(inner.rotation); };
  return { pair, toPair };
}
const P = top.meta.parts;
let LIFT = 0;                                       // the pair stands on its lowest part
const pairs = X_BOARDS.map((x, k) => {
  const { pair, toPair } = makePair();
  if (!LIFT) LIFT = -new THREE.Box3().setFromObject(pair).min.y;
  pair.position.set(x, FLOOR_Y + LIFT, 0);
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

// ---- JST-SH cables: four wires (5 V, SDA, SCL, GND), plug housings at the board ends
const WIRE_COL = ["#d83a34", "#2f6fd0", "#e9c02a", "#202124"];
function cablePath(a, b, h) {
  const m = a.clone().add(b).multiplyScalar(0.5);
  const d = Math.sign(b.x - a.x);
  return new THREE.CatmullRomCurve3([a, a.clone().add(new THREE.Vector3(0, 6, 0)), a.clone().add(new THREE.Vector3(d * 7, h - 3, 0)),
    m.clone().setY(Math.max(a.y, b.y) + h), b.clone().add(new THREE.Vector3(-d * 7, h - 3, 0)), b.clone().add(new THREE.Vector3(0, 6, 0)), b]);
}
function makeCable(a, b, h, plugA) {
  const curve = cablePath(a, b, h), grp = new THREE.Group(), tubes = [];
  for (let w = 0; w < 4; w++) {
    const off = (w - 1.5) * 1.0;
    const pts = curve.getPoints(160).map(p => p.clone().add(new THREE.Vector3(0, 0, off)));
    const geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 160, 0.45, 8, false);
    const t = new THREE.Mesh(geo, mat(WIRE_COL[w], { roughness: 0.45 }));
    grp.add(t); tubes.push(t);
  }
  const plugs = [];
  for (const [p, on] of [[a, plugA], [b, true]]) {
    if (!on) continue;
    const pl = box(3.4, 4.0, 5.8, mat("#ece4cc", { roughness: 0.55 }), p.x, p.y + 1.4, p.z);
    grp.add(pl); plugs.push(pl);
  }
  scene.add(grp);
  return { grp, tubes, plugs, curve };
}
const cables = [
  makeCable(TEENSY_WIRES, pairs[0].cn1, 18, false),
  makeCable(pairs[0].cn2, pairs[1].cn1, 15, true),
  makeCable(pairs[1].cn2, pairs[2].cn1, 15, true),
];
// solder joints at the Teensy end
for (let w = 0; w < 4; w++) teensy.add(box(1.4, 0.8, 0.9, mat("#c0c4c8", { metalness: 0.7, roughness: 0.3 }), 16.0, 3.6, (w - 1.5) * 1.0));

// ------------------------------------------------------------------ timeline
const TB = k => 0.2 + 0.4 * k;                     // board k slides in from the right (0.8 s)
const TC = k => 2.0 + 0.75 * k;                     // cable k drawn
const T_USB = 4.35, T_ON = 5.15, T_STREAM = 5.8;
const T_OFF = 10.45, T_UNPLUG = 10.7, T_EXIT = 11.15;   // stop; unplug; boards slide out, so the last frame is the first
const SLIDE = 200;                                  // mm: off the right edge of the frame
const eOut = f => 1 - Math.pow(1 - clamp(f), 3), eIn = f => Math.pow(clamp(f), 3);
const ROUND = 0.45;                                 // s per node read (slowed down)

// ------------------------------------------------------------------ 2D overlay
const ctx = overlay(2);
const WIN = { x: 40, y: 22, w: 1200, h: 420 };
const PAN = k => ({ x: WIN.x + 24 + k * 392, y: WIN.y + 62, w: 368, h: 340 });
const G = 12;                                       // taxels per side of the made-up sensors

function contact(k, i, j, t) {                      // made-up presses, a different one per node
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

let screenPaths = null;                             // per node: polyline in frame pixels to the window
function project(v) {
  const p = v.clone().project(camera);
  return [(p.x + 1) / 2 * W, (1 - p.y) / 2 * H];
}
function buildPaths() {
  const usb2d = usbPath();
  screenPaths = pairs.map((pr, k) => {
    const pts3 = [];
    for (let c = k; c >= 0; c--) {                  // from board k back along each cable to the Teensy
      if (c < k) pts3.push(pairs[c].cn2.clone(), pairs[c].cn1.clone());
      const pts = cables[c].curve.getPoints(80).reverse();
      pts3.push(...pts);
    }
    pts3.push(TEENSY_WIRES.clone(), TEENSY_USB.clone());
    return pts3.map(project).concat(usb2d.slice(1));
  });
}
function usbPath() {                                // from the Teensy's USB plug, out and up into the window
  const a = project(new THREE.Vector3(X_TEENSY - 34, 4.8, 0)), e = [150, WIN.y + WIN.h];
  const c1 = [a[0] - 70, a[1] + 10], c2 = [e[0], e[1] + 150], out = [];
  for (let i = 0; i <= 40; i++) {
    const f = i / 40, g = 1 - f;
    out.push([0, 1].map(d => g * g * g * a[d] + 3 * g * g * f * c1[d] + 3 * g * f * f * c2[d] + f * f * f * e[d]));
  }
  return [project(TEENSY_USB)].concat(out);
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
  const out = 1 - seg(t, T_OFF, T_OFF + 0.3);       // panels, LEDs and labels go off at the end
  // boards: slide in from the right, one after another; out again at the end, rightmost first
  pairs.forEach(pr => {
    const a = eOut((t - TB(pr.k)) / 0.8), b = eIn((t - (T_EXIT + 0.1 * (2 - pr.k))) / 0.6);
    pr.pair.position.x = X_BOARDS[pr.k] + SLIDE * (1 - a) + SLIDE * b;
    pr.pair.visible = a > 0 && b < 1;
    const on = seg(t, T_ON + 0.18 * pr.k, T_ON + 0.18 * pr.k + 0.3) * out;
    pr.dome.material.color.setRGB(...(on > 0.5 ? NODE[pr.k].map(v => v / 255) : [0.13, 0.13, 0.13]));
    pr.halo.material.opacity = on * 0.9;
  });
  // cables: grow from the Teensy end outward, the plug going in as each arrives; drawn back in at the end
  cables.forEach((c, k) => {
    const u = T_UNPLUG + 0.08 * (2 - k);
    const a = seg(t, TC(k), TC(k) + 0.65) * (1 - seg(t, u, u + 0.35));
    c.grp.visible = a > 0;
    c.tubes.forEach(tb => tb.geometry.setDrawRange(0, Math.round(tb.geometry.index.count * a / 48) * 48));
    c.plugs.forEach((pl, i) => { pl.visible = i === c.plugs.length - 1 ? a > 0.9 : a > 0; });
  });
  const usbIn = seg(t, T_USB, T_USB + 0.4), usbOut = seg(t, T_UNPLUG + 0.25, T_UNPLUG + 0.55);
  usbPlug.position.copy(TEENSY_USB).add(new THREE.Vector3(-14 * (1 - usbIn) - 14 * usbOut, 0, 0));
  usbPlug.visible = t > T_USB && usbOut < 1;
  renderer.render(scene, camera);
  if (!screenPaths) buildPaths();

  ctx.clearRect(0, 0, W, H);
  // USB cable, in the frame's plane, up into the computer window
  const usbGrow = seg(t, T_USB + 0.25, T_USB + 0.85) * (1 - seg(t, T_UNPLUG - 0.1, T_UNPLUG + 0.3));
  if (usbGrow > 0) {
    ctx.save();
    drawPoly(usbPath().slice(1), usbGrow, "#2a2d31", 11);
    ctx.restore();
  }
  // computer window: `opentac view`, one panel per node
  ctx.save();
  roundRect(ctx, WIN.x, WIN.y, WIN.w, WIN.h, 14); ctx.fillStyle = "#161c23"; ctx.fill();
  ctx.restore();
  text(ctx, "computer: opentac view", WIN.x + 26, WIN.y + 42, { size: 30, color: "#aab3bd", weight: 500 });
  pairs.forEach((pr, k) => {
    const a = seg(t, T_ON + 0.18 * k, T_ON + 0.18 * k + 0.4) * out, R = PAN(k);
    if (a <= 0) return;
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
    const r = (t - T_STREAM) / ROUND, k = Math.floor(r) % 3, f = r - Math.floor(r);
    const [px, py] = polyAt(screenPaths[k], f);
    ctx.save();
    ctx.lineCap = "round";
    for (let i = 1; i <= 8; i++) {                  // a short trail behind the pulse
      const [ax, ay] = polyAt(screenPaths[k], Math.max(0, f - 0.012 * i)), [bx2, by2] = polyAt(screenPaths[k], Math.max(0, f - 0.012 * (i - 1)));
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
  // labels
  const tl = project(new THREE.Vector3(X_TEENSY, 0, 16));
  label("Teensy 4.0", tl[0], tl[1] + 40, 1);
  pairs.forEach((pr, k) => {
    const p = project(new THREE.Vector3(X_BOARDS[k], 0, 16)), a = seg(t, T_ON + 0.18 * k, T_ON + 0.18 * k + 0.4) * out;
    if (a <= 0) return;
    ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = rgb(NODE[k]);
    ctx.beginPath(); ctx.arc(p[0] - 62, p[1] + 29, 10, 0, 2 * Math.PI); ctx.fill(); ctx.restore();
    label(`node ${k}`, p[0] + 12, p[1] + 40, a);
  });
  const c1 = project(cables[1].curve.getPoint(0.5).add(new THREE.Vector3(0, 6, 0)));
  label("JST-SH: 5 V, SDA, SCL, GND", c1[0], c1[1] - 34, seg(t, TC(1), TC(1) + 0.4) * out, { size: 30, color: C.muted });
  const u = usbPath()[20];
  label("USB", u[0] + 22, u[1] + 10, seg(t, T_USB + 0.5, T_USB + 0.9) * (1 - seg(t, T_UNPLUG - 0.2, T_UNPLUG)), { size: 32, align: "left" });
  label("… up to 18 boards", WIN.x + WIN.w, c1[1] - 34, seg(t, TC(2) + 0.7, TC(2) + 1.1) * out, { size: 30, color: C.muted, align: "right" });
}

start(renderAt, T_END);
