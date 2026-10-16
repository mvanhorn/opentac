// The IMU on the readout board, from a recorded take (data/imu.json, written by
// make_hw_animations.py from the recording). Everything runs on the recording's own clock:
//   - left: the 60-pin board, turned by the quaternion the IMU computes on the chip (sensor axes:
//     x and y in the board plane, z out of the top face; "up" is the quaternion's +z)
//   - right, top: the angle the board has turned since the clip's first frame, same quaternion
//   - right, below: vibration, the accelerometer with gravity and slow motion removed (each axis
//     high-passed, then the magnitude), in g
//   - bottom: the tactile frame from the same board at the same instant
import * as THREE from "three";
import { C, W, H, DPR, overlay, text, roundRect, start, inferno, clamp, loadBoard } from "./common.js";

const D = await (await fetch("/data/imu.json")).json();
const { group: board, meta } = await loadBoard(THREE, "readout_60pin");
const WINDOW = 4;                         // s of trace on screen
const VMAX = 3500;                        // tactile counts at the top of the colour map
const VIB_MAX = 3.5;                      // g at the top of the vibration panel

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
  const q = a.slerp(b, f);
  return [q.w, q.x, q.y, q.z];
}
function angleBetween(a, b) {
  const d = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
  return 2 * Math.acos(Math.min(1, d)) * 180 / Math.PI;
}

// ------------------------------------------------------------------ 3D: the left half
const V3 = { x: 0, y: 0, w: 640, h: 770 };
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(DPR); renderer.setSize(V3.w, V3.h);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
renderer.domElement.style.zIndex = 1;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(C.bg);
const camera = new THREE.PerspectiveCamera(26, V3.w / V3.h, 10, 2000);
camera.position.set(0, 29, 113); camera.lookAt(0, 5.2, 0);
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
const PR = { x: 750, y: 140, w: 480, h: 190 };   // angle turned
const PV = { x: 750, y: 474, w: 480, h: 236 };   // vibration
const TS = { x: 250, y: 790, w: 990, h: 160 };   // tactile strip

function axes(P, ymin, ymax, ticks) {
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
function tracePath(P, t, ymin, ymax, value) {
  const i0 = idxAt(t - WINDOW), i1 = idxAt(t), pts = [];
  for (let i = i0; i <= i1; i++) {
    pts.push([P.x + P.w * (1 + (D.t[i] - t) / WINDOW), P.y + P.h * (1 - (clamp(value(i), ymin, ymax) - ymin) / (ymax - ymin))]);
  }
  return pts;
}
function line(P, pts, color, width = 3, fillTo = null) {
  ctx.save();
  ctx.beginPath(); ctx.rect(P.x, P.y - 3, P.w, P.h + 6); ctx.clip();
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  if (fillTo !== null) {
    ctx.save();
    ctx.lineTo(pts[pts.length - 1][0], fillTo); ctx.lineTo(pts[0][0], fillTo); ctx.closePath();
    ctx.fillStyle = "rgba(42,120,214,0.22)"; ctx.fill();
    ctx.restore();
    ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  }
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = "round"; ctx.stroke();
  ctx.restore();
}

// taxel layout: along the finger -> across the strip, across the finger -> down the strip
const ys = D.xy.map(p => p[1]), xs = D.xy.map(p => p[0]);
const yMin = Math.min(...ys), yMax = Math.max(...ys), xMin = Math.min(...xs), xMax = Math.max(...xs);
const pitch = 4, k = Math.min((TS.w - 16) / (yMax - yMin + pitch), (TS.h - 16) / (xMax - xMin + pitch));
const cell = pitch * k * 0.86;
const sx = TS.x + (TS.w - (yMax - yMin) * k) / 2, sy = TS.y + (TS.h - (xMax - xMin) * k) / 2;

function renderAt(t) {
  pivot.quaternion.copy(Y0).multiply(modelQuat(qAt(t)));
  renderer.render(scene, camera);

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = C.bg; ctx.fillRect(V3.w, 0, W - V3.w, V3.h); ctx.fillRect(0, V3.h, W, H - V3.h);
  text(ctx, "Orientation", 40, 64, { size: 40, weight: 600 });
  text(ctx, "from the IMU's quaternion", 40, 104, { size: 32, color: C.muted });

  // the IMU on the board: a callout while the top face is toward the camera
  const pw = imuLocal.clone(); board.localToWorld(pw);
  const nrm = new THREE.Vector3(0, 1, 0).applyQuaternion(pivot.quaternion);
  const toCam = camera.position.clone().sub(pw).normalize();
  const vis = clamp((nrm.dot(toCam) - 0.15) / 0.2);
  if (vis > 0) {
    const s = pw.clone().project(camera);
    const px = V3.x + (s.x + 1) / 2 * V3.w, py = V3.y + (1 - s.y) / 2 * V3.h;
    const ly = clamp(py, 200, 700);
    ctx.save(); ctx.globalAlpha = vis; ctx.lineCap = "round";
    for (const [col, wd] of [["rgba(247,246,242,0.9)", 9], [C.ink, 3]]) {
      ctx.strokeStyle = col; ctx.lineWidth = wd;
      ctx.beginPath(); ctx.moveTo(122, ly); ctx.lineTo(px - 15, py); ctx.stroke();
      ctx.beginPath(); ctx.arc(px, py, 15, 0, 2 * Math.PI); ctx.stroke();
    }
    roundRect(ctx, 24, ly - 26, 98, 50, 10); ctx.fillStyle = "rgba(247,246,242,0.94)"; ctx.fill();
    ctx.restore();
    text(ctx, "IMU", 73, ly + 12, { size: 36, weight: 600, align: "center", alpha: vis });
  }

  // angle turned since the clip's first frame
  text(ctx, "Turned from start", 690, 64, { size: 36, weight: 600 });
  text(ctx, "degrees", 690, 104, { size: 32, color: C.muted });
  axes(PR, 0, 90, [0, 30, 60, 90]);
  line(PR, tracePath(PR, t, 0, 90, i => ang[i]), C.ink);

  // vibration: same time axis as the angle above
  text(ctx, "Vibration", 690, 398, { size: 36, weight: 600 });
  text(ctx, "acceleration with gravity removed, g", 690, 438, { size: 32, color: C.muted });
  axes(PV, 0, VIB_MAX, [0, 1, 2, 3]);
  line(PV, tracePath(PV, t, 0, VIB_MAX, i => D.vib[i]), C.x, 3, PV.y + PV.h);
  timeTicks(PV);

  // tactile frame
  text(ctx, "Tactile", 40, TS.y + 70, { size: 36, weight: 600 });
  text(ctx, "frame", 40, TS.y + 110, { size: 36, weight: 600 });
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
}

start(renderAt, D.duration);
