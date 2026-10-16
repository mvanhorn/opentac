// The IMU on the readout board, from a recorded take (data/imu.json, written by
// make_hw_animations.py from the recording). Everything runs on the recording's own clock:
//   - the 60-pin board, turned by the quaternion the IMU computes on the chip (sensor axes:
//     x and y in the board plane, z out of the top face; "up" is the quaternion's +z)
//   - the angle the board has turned since the clip's first frame, from the same quaternion
//   - the accelerometer, x / y / z as recorded, in g
//   - the tactile frame from the same board at the same instant
import * as THREE from "three";
import { C, W, H, DPR, FONT, overlay, text, roundRect, start, inferno, clamp, lerp, loadBoard } from "./common.js";

const D = await (await fetch("/data/imu.json")).json();
const { group: board, meta } = await loadBoard(THREE, "readout_60pin");
const WINDOW = 4;                         // s of trace on screen
const VMAX = 3500;                        // tactile counts at the top of the colour map

// ------------------------------------------------------------------ data
const N = D.t.length;
const q0 = qAt(0);
const ang = D.quat.map(q => angleBetween(q, q0));

function idxAt(t) {                       // last sample at or before t
  let lo = 0, hi = N - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (D.t[m] <= t) lo = m; else hi = m - 1; }
  return lo;
}
function qAt(t) {
  const i = Math.min(idxAt(t), N - 2), f = clamp((t - D.t[i]) / (D.t[i + 1] - D.t[i]));
  const a = new THREE.Quaternion(D.quat[i][1], D.quat[i][2], D.quat[i][3], D.quat[i][0]);
  const b = new THREE.Quaternion(D.quat[i + 1][1], D.quat[i + 1][2], D.quat[i + 1][3], D.quat[i + 1][0]);
  return [a.slerp(b, f)].map(q => [q.w, q.x, q.y, q.z])[0];
}
function angleBetween(a, b) {
  const d = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
  return 2 * Math.acos(Math.min(1, d)) * 180 / Math.PI;
}

// ------------------------------------------------------------------ 3D
const V3 = { x: 0, y: 0, w: 760, h: 540 };
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(DPR); renderer.setSize(V3.w, V3.h);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
renderer.domElement.style.zIndex = 1;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(C.bg);
const camera = new THREE.PerspectiveCamera(26, V3.w / V3.h, 10, 2000);
camera.position.set(0, 34, 130); camera.lookAt(0, 3, 0);
scene.add(new THREE.HemisphereLight(0xffffff, 0xd8d2c4, 1.5));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(-25, 200, 45);
scene.add(sun);
const fill = new THREE.DirectionalLight(0xeef2ff, 0.8); fill.position.set(80, 40, 120); scene.add(fill);

const [x0, y0, x1, y1] = meta.outline;
const pivot = new THREE.Group(); scene.add(pivot);
board.position.set(-(x0 + x1) / 2, -0.8, -(y0 + y1) / 2);
pivot.add(board);
const imuPart = meta.parts.IC5;           // KiCad mm
const imuLocal = new THREE.Vector3(imuPart[0], 1.8, imuPart[1]);

// sensor axes -> board model axes, and the quaternion's world (z up) -> three.js world (y up):
// x -> x, y -> -z, z -> y
const M = new THREE.Matrix4().set(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);
const Mi = M.clone().invert();
function modelQuat(q) {
  const R = new THREE.Matrix4().makeRotationFromQuaternion(new THREE.Quaternion(q[1], q[2], q[3], q[0]));
  return new THREE.Quaternion().setFromRotationMatrix(M.clone().multiply(R).multiply(Mi));
}
// The quaternion's heading is arbitrary (no magnetometer): turn the scene about the vertical so the
// board's average facing over the clip points at the camera, a little to one side.
const yaw0 = (() => {
  const n = new THREE.Vector3();
  for (let t = 0; t <= D.duration; t += 0.25) n.add(new THREE.Vector3(0, 1, 0).applyQuaternion(modelQuat(qAt(t))));
  return Math.atan2(camera.position.x, camera.position.z) - Math.atan2(n.x, n.z) + 0.35;
})();
const Y0 = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw0);

// ------------------------------------------------------------------ 2D
const ctx = overlay(2);
const PR = { x: 836, y: 160, w: 394, h: 290 };   // rotation trace
const PA = { x: 136, y: 808, w: 1094, h: 104 };  // acceleration trace
const TS = { x: 236, y: 544, w: 994, h: 196 };   // tactile strip

function axes(P, ymin, ymax, ticks, unit) {
  ctx.save();
  ctx.strokeStyle = C.grid; ctx.lineWidth = 2;
  for (const v of ticks) {
    const y = P.y + P.h * (1 - (v - ymin) / (ymax - ymin));
    ctx.beginPath(); ctx.moveTo(P.x, y); ctx.lineTo(P.x + P.w, y); ctx.stroke();
    text(ctx, String(v), P.x - 12, y + 11, { size: 30, color: C.muted, align: "right" });
  }
  ctx.restore();
}
function timeTicks(P) {
  for (const [s, lab] of [[-4, "−4 s"], [-2, "−2 s"], [0, "now"]]) {
    const x = P.x + P.w * (1 + s / WINDOW);
    text(ctx, lab, x, P.y + P.h + 38, { size: 30, color: C.muted, align: s === -4 ? "left" : s === 0 ? "right" : "center" });
  }
}
function trace(P, t, ymin, ymax, value, color) {
  const i0 = idxAt(t - WINDOW), i1 = idxAt(t);
  ctx.save();
  ctx.beginPath(); ctx.rect(P.x, P.y - 2, P.w, P.h + 4); ctx.clip();
  ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.lineJoin = "round";
  ctx.beginPath();
  for (let i = i0; i <= i1; i++) {
    const x = P.x + P.w * (1 + (D.t[i] - t) / WINDOW);
    const y = P.y + P.h * (1 - (value(i) - ymin) / (ymax - ymin));
    if (i === i0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

// taxel layout: along the finger -> across the strip, across the finger -> down the strip
const ys = D.xy.map(p => p[1]), xs = D.xy.map(p => p[0]);
const yMin = Math.min(...ys), yMax = Math.max(...ys), xMin = Math.min(...xs), xMax = Math.max(...xs);
const pitch = 4, k = Math.min((TS.w - 16) / (yMax - yMin + pitch), (TS.h - 16) / (xMax - xMin + pitch));
const cell = pitch * k * 0.86;
const sx = TS.x + (TS.w - (yMax - yMin) * k) / 2, sy = TS.y + (TS.h - (xMax - xMin) * k) / 2;

function renderAt(t) {
  // 3D board
  pivot.quaternion.copy(Y0).multiply(modelQuat(qAt(t)));
  renderer.render(scene, camera);

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = C.bg; ctx.fillRect(V3.w, 0, W - V3.w, V3.y + V3.h); ctx.fillRect(0, V3.y + V3.h, W, H);
  text(ctx, "Orientation", 40, 66, { size: 40, weight: 600 });
  text(ctx, "from the IMU's quaternion", 40, 108, { size: 32, color: C.muted });

  // the IMU on the board: a callout while the top face is toward the camera
  const pw = imuLocal.clone(); board.localToWorld(pw);
  const nrm = new THREE.Vector3(0, 1, 0).applyQuaternion(pivot.quaternion);
  const toCam = camera.position.clone().sub(pw).normalize();
  const vis = clamp((nrm.dot(toCam) - 0.15) / 0.2);
  if (vis > 0) {
    const s = pw.clone().project(camera);
    const px = V3.x + (s.x + 1) / 2 * V3.w, py = V3.y + (1 - s.y) / 2 * V3.h;
    const lx = 96, ly = clamp(py, 200, 470);
    ctx.save(); ctx.globalAlpha = vis; ctx.lineCap = "round";
    for (const [col, wd] of [["rgba(247,246,242,0.9)", 9], [C.ink, 3]]) {
      ctx.strokeStyle = col; ctx.lineWidth = wd;
      ctx.beginPath(); ctx.moveTo(lx + 62, ly); ctx.lineTo(px - 14, py); ctx.stroke();
      ctx.beginPath(); ctx.arc(px, py, 13, 0, 2 * Math.PI); ctx.stroke();
    }
    ctx.restore();
    text(ctx, "IMU", lx + 54, ly + 12, { size: 36, weight: 600, align: "right", alpha: vis });
  }

  // rotation since the clip's first frame
  text(ctx, "Turned from start", PR.x - 72, 66, { size: 36, weight: 600 });
  text(ctx, "degrees", PR.x - 72, 108, { size: 32, color: C.muted });
  axes(PR, 0, 90, [0, 30, 60, 90]);
  trace(PR, t, 0, 90, i => ang[i], C.ink);
  timeTicks(PR);

  // tactile frame
  text(ctx, "Tactile", 40, TS.y + 86, { size: 36, weight: 600 });
  text(ctx, "frame", 40, TS.y + 126, { size: 36, weight: 600 });
  ctx.save();
  roundRect(ctx, sx - cell / 2 - 12, sy - cell / 2 - 12, (yMax - yMin) * k + cell + 24, (xMax - xMin) * k + cell + 24, 10);
  ctx.fillStyle = "#0b0e12"; ctx.fill();
  const fr = D.frames[clamp(Math.round(t * D.fps) - D.frame0, 0, D.frames.length - 1)];
  for (let j = 0; j < fr.length; j++) {
    const cx = sx + (D.xy[j][1] - yMin) * k, cy = sy + (xMax - D.xy[j][0]) * k;
    roundRect(ctx, cx - cell / 2, cy - cell / 2, cell, cell, 4);
    ctx.fillStyle = inferno(fr[j] / VMAX); ctx.fill();
  }
  ctx.restore();

  // accelerometer
  text(ctx, "Acceleration", 40, 784, { size: 36, weight: 600 });
  text(ctx, "g", 270, 784, { size: 32, color: C.muted });
  let lx = 900;
  for (const [a, col] of [["x", C.x], ["y", C.y], ["z", C.z]]) {
    ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 5; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(lx, 773); ctx.lineTo(lx + 34, 773); ctx.stroke(); ctx.restore();
    text(ctx, a, lx + 44, 784, { size: 32, color: C.ink });
    lx += 112;
  }
  axes(PA, -3.4, 3.4, [-3, 0, 3]);
  [[0, C.x], [1, C.y], [2, C.z]].forEach(([a, col]) => trace(PA, t, -3.4, 3.4, i => D.accel[i][a], col));
  timeTicks(PA);
}

start(renderAt, D.duration);
