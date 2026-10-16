// Zero-potential readout: a 4 x 4 matrix with three pressed taxels (three corners of a
// rectangle) is scanned twice. First with the undriven rows left open, then held at 0 V as the
// board does. Each step solves the resistor network (the same model as study 1 on the How it
// works page: pressed taxel conductance 1, released 0.002, driven row at +V, the selected
// column at the amplifier's virtual ground, the other columns switched off by the mux) and
// shows the current in every wire as moving dots, at a speed proportional to the current.
import { C, W, H, overlay, text, roundRect, start, seg, lerp, clamp } from "./common.js";

const N = 4, G_ON = 1, G_OFF = 0.002;
const PRESS = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
const GHOST = [1, 0];
const RX = c => 300 + c * 120, RY = r => 262 + r * 110;
const X_DRV = 172, X_END = 700, Y_MUX = 662, MUX_OUT = [482, 722], AMP_IN = [600, 792];
const FX = 922, FY = 236, FC = 70, FG = 8;            // the frame map: origin, cell, gap

// ------------------------------------------------------------------ the network
function solve(row, col, hold) {
  const g = PRESS.map(r => r.map(v => (v ? G_ON : G_OFF)));
  const known = {}, nodes = [];
  for (let r = 0; r < N; r++) {
    if (r === row) known["r" + r] = 1; else if (hold) known["r" + r] = 0; else nodes.push("r" + r);
  }
  for (let c = 0; c < N; c++) { if (c === col) known["c" + c] = 0; else nodes.push("c" + c); }
  const n = nodes.length, idx = {}, A = [], b = [];
  nodes.forEach((nm, i) => { idx[nm] = i; A.push(new Array(n).fill(0)); b.push(0); });
  const nb = nm => {
    const k = +nm.slice(1), out = [];
    for (let q = 0; q < N; q++) out.push(nm[0] === "r" ? ["c" + q, g[k][q]] : ["r" + q, g[q][k]]);
    return out;
  };
  nodes.forEach((nm, i) => nb(nm).forEach(([o, gc]) => {
    A[i][i] += gc;
    if (o in known) b[i] += gc * known[o]; else A[i][idx[o]] -= gc;
  }));
  for (let i = 0; i < n; i++) {                       // Gaussian elimination
    let p = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(A[k][i]) > Math.abs(A[p][i])) p = k;
    [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
    for (let k = i + 1; k < n; k++) {
      const f = A[k][i] / A[i][i];
      for (let j = i; j < n; j++) A[k][j] -= f * A[i][j];
      b[k] -= f * b[i];
    }
  }
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let j = i + 1; j < n; j++) s -= A[i][j] * x[j];
    x[i] = s / A[i][i];
  }
  const V = { ...known };
  nodes.forEach((nm, i) => { V[nm] = x[i]; });
  const cur = [];
  let I = 0;
  for (let r = 0; r < N; r++) {
    cur.push([]);
    for (let c = 0; c < N; c++) cur[r].push(g[r][c] * (V["r" + r] - V["c" + c]));
    I += cur[r][col];
  }
  return { cur, I };
}

// ------------------------------------------------------------------ the timeline
const STEP = 0.2, STEP_PRESSED = 0.38, STEP_GHOST = 1.6;
const phases = [];
let T = 0;
for (const hold of [false, true]) {
  const p = { hold, t0: T, steps: [] };
  T += hold ? 0.35 : 0.3;                             // a moment to read the title
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const d = r === GHOST[0] && c === GHOST[1] ? STEP_GHOST : PRESS[r][c] ? STEP_PRESSED : STEP;
    const sol = solve(r, c, hold);
    p.steps.push({ r, c, t0: T, t1: T + d, sol, counts: Math.min(4095, Math.round(sol.I * 2400)) });
    T += d;
  }
  T += hold ? 1.0 : 0.85;                             // the finished frame
  p.t1 = T;
  phases.push(p);
}
const T_END = T;

// ------------------------------------------------------------------ drawing
const ctx = overlay();

function wire(pts, color, width, dash) {
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = "round"; ctx.lineJoin = "round";
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath(); ctx.moveTo(...pts[0]); for (const p of pts.slice(1)) ctx.lineTo(...p); ctx.stroke();
  ctx.restore();
}

// dots moving from a to b (current > 0) or b to a (< 0), speed proportional to |current|
function flow(a, b, I, t) {
  const m = Math.abs(I);
  if (m < 0.02) return;
  const L = Math.hypot(b[0] - a[0], b[1] - a[1]), sp = 26, off = (t * 170 * m) % sp;
  ctx.save();
  ctx.fillStyle = C.current;
  ctx.globalAlpha = clamp(0.45 + m, 0, 1);
  for (let s = off; s < L; s += sp) {
    const f = (I > 0 ? s : L - s) / L;
    ctx.beginPath(); ctx.arc(lerp(a[0], b[0], f), lerp(a[1], b[1], f), 4 + 3 * clamp(m), 0, 2 * Math.PI); ctx.fill();
  }
  ctx.restore();
}

function ground(x, y) {
  ctx.save(); ctx.strokeStyle = C.muted; ctx.lineWidth = 3;
  for (let i = 0; i < 3; i++) {
    const w = 26 - i * 9;
    ctx.beginPath(); ctx.moveTo(x - w / 2, y + i * 8); ctx.lineTo(x + w / 2, y + i * 8); ctx.stroke();
  }
  ctx.restore();
}

function frameColour(v) {               // released (panel) -> pressed amber -> dark
  const a = [247, 246, 242], b = [224, 138, 44], d = [140, 64, 10];
  const mix = (p, q, f) => `rgb(${p.map((x, i) => Math.round(lerp(x, q[i], f))).join(",")})`;
  return v < 0.6 ? mix(a, b, v / 0.6) : mix(b, d, (v - 0.6) / 0.4);
}

function renderAt(t) {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);

  const pi = t < phases[1].t0 ? 0 : 1, P = phases[pi];
  // the step being read (none before the first and after the last)
  const st = P.steps.find(s => t >= s.t0 && t < s.t1);
  const done = P.steps.filter(s => t >= s.t1 || s === st);
  const hold = P.hold;

  // titles
  text(ctx, hold ? "Other rows held at 0 V, as on the board" : "Other rows left open", 56, 80, { size: 50, weight: 600 });
  text(ctx, hold ? "Only the selected taxel passes current to the amplifier."
                 : "Current takes a path through three pressed taxels.", 56, 132, { size: 36, color: C.muted });

  const row = st ? st.r : -1, col = st ? st.c : -1, sol = st && st.sol;

  // rows: driver terminal + line
  for (let r = 0; r < N; r++) {
    const y = RY(r), drv = r === row;
    const colr = drv ? C.amber : C.rule;
    wire([[X_DRV, y], [X_END, y]], colr, drv ? 5 : 3);
    ctx.save();
    ctx.beginPath(); ctx.arc(X_DRV, y, 9, 0, 2 * Math.PI);
    if (drv || hold) { ctx.fillStyle = drv ? C.amber : C.muted; ctx.fill(); }
    else { ctx.fillStyle = C.bg; ctx.fill(); ctx.strokeStyle = C.muted; ctx.lineWidth = 3; ctx.stroke(); }
    ctx.restore();
    const lab = drv ? "+V" : hold ? "0 V" : "open";
    text(ctx, lab, X_DRV - 24, y + 14, { size: 38, weight: drv ? 700 : 400, color: drv ? C.amberDark : C.muted, align: "right" });
  }
  // columns down to the mux
  for (let c = 0; c < N; c++) {
    const sel = c === col;
    wire([[RX(c), 214], [RX(c), Y_MUX]], sel ? C.ink : C.rule, sel ? 5 : 3);
  }
  // mux
  ctx.save();
  ctx.beginPath(); ctx.moveTo(250, Y_MUX); ctx.lineTo(714, Y_MUX); ctx.lineTo(634, MUX_OUT[1]); ctx.lineTo(330, MUX_OUT[1]); ctx.closePath();
  ctx.fillStyle = C.panel; ctx.fill(); ctx.strokeStyle = C.muted; ctx.lineWidth = 3; ctx.stroke();
  ctx.restore();
  if (st) wire([[RX(col), Y_MUX], MUX_OUT], C.ink, 5);
  text(ctx, "mux", 724, MUX_OUT[1] - 20, { size: 32, color: C.muted });
  // column -> amplifier (inverting, + input grounded), feedback resistor, output
  wire([MUX_OUT, [MUX_OUT[0], AMP_IN[1]], AMP_IN], st ? C.ink : C.rule, st ? 5 : 3);
  text(ctx, "0 V", MUX_OUT[0] - 16, AMP_IN[1] - 18, { size: 32, color: C.muted, align: "right" });
  const ax = 616, ay = AMP_IN[1] + 30;
  ctx.save();
  ctx.beginPath(); ctx.moveTo(ax, ay - 52); ctx.lineTo(ax, ay + 52); ctx.lineTo(ax + 92, ay); ctx.closePath();
  ctx.fillStyle = C.panel; ctx.fill(); ctx.strokeStyle = C.muted; ctx.lineWidth = 3; ctx.stroke();
  ctx.restore();
  wire([AMP_IN, [ax, AMP_IN[1]]], st ? C.ink : C.rule, 3);
  text(ctx, "−", ax + 8, AMP_IN[1] + 10, { size: 30, color: C.muted });
  text(ctx, "+", ax + 8, ay + 34, { size: 30, color: C.muted });
  wire([[ax, ay + 24], [ax - 30, ay + 24], [ax - 30, ay + 44]], C.muted, 3);
  ground(ax - 30, ay + 44);
  wire([[ax - 6, AMP_IN[1]], [ax - 6, AMP_IN[1] - 52], [ax + 120, AMP_IN[1] - 52], [ax + 120, ay], [ax + 92, ay]], C.muted, 3);
  ctx.save(); ctx.fillStyle = C.panel; ctx.strokeStyle = C.muted; ctx.lineWidth = 3;
  ctx.fillRect(ax + 30, AMP_IN[1] - 62, 50, 20); ctx.strokeRect(ax + 30, AMP_IN[1] - 62, 50, 20); ctx.restore();
  wire([[ax + 120, ay], [ax + 150, ay]], C.muted, 3);
  // the reading
  const counts = st ? st.counts : null;
  text(ctx, "reading", ax + 166, ay - 22, { size: 32, color: C.muted });
  text(ctx, counts === null ? "–" : String(counts), ax + 166, ay + 32, { size: 52, weight: 700 });

  // current: rows (driver at the left end), columns (mux at the bottom), mux -> amplifier
  if (sol) {
    for (let r = 0; r < N; r++) {
      const xs = [X_DRV, RX(0), RX(1), RX(2), RX(3)];
      for (let k = 0; k < N; k++) {
        let I = 0; for (let c = k; c < N; c++) I += sol.cur[r][c];
        flow([xs[k], RY(r)], [xs[k + 1], RY(r)], I, t);
      }
    }
    for (let c = 0; c < N; c++) {
      const ys = [RY(0), RY(1), RY(2), RY(3), Y_MUX];
      let I = 0;
      for (let k = 0; k < N; k++) { I += sol.cur[k][c]; flow([RX(c), ys[k]], [RX(c), ys[k + 1]], I, t); }
    }
    flow([RX(col), Y_MUX], MUX_OUT, sol.I, t);
    flow(MUX_OUT, [MUX_OUT[0], AMP_IN[1]], sol.I, t);
    flow([MUX_OUT[0], AMP_IN[1]], AMP_IN, sol.I, t);
  }

  // taxels
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const x = RX(c), y = RY(r), p = PRESS[r][c];
    ctx.save();
    roundRect(ctx, x - 22, y - 22, 44, 44, 7);
    ctx.fillStyle = p ? C.amber : C.panel; ctx.fill();
    ctx.strokeStyle = p ? C.amberDark : C.muted; ctx.lineWidth = 2.5; ctx.stroke();
    if (r === row && c === col) {
      roundRect(ctx, x - 31, y - 31, 62, 62, 10);
      ctx.strokeStyle = C.ink; ctx.lineWidth = 5; ctx.stroke();
    }
    ctx.restore();
  }

  // the frame: each taxel's reading, filled in as it is read
  text(ctx, "Frame", FX, FY - 28, { size: 40, weight: 600 });
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const s = P.steps[r * N + c], x = FX + c * (FC + FG), y = FY + r * (FC + FG);
    const read = done.includes(s);
    ctx.save();
    roundRect(ctx, x, y, FC, FC, 8);
    ctx.fillStyle = read ? frameColour(clamp(s.counts / 2400)) : C.bg; ctx.fill();
    ctx.strokeStyle = s === st ? C.ink : C.rule; ctx.lineWidth = s === st ? 5 : 2.5;
    if (!read) ctx.setLineDash([6, 6]);
    ctx.stroke();
    ctx.restore();
  }
  // the ghost: a reading where nothing is pressed (or none, with the rows held)
  const gs = P.steps[GHOST[0] * N + GHOST[1]];
  if (done.includes(gs)) {
    const a = seg(t, gs.t0 + 0.25, gs.t0 + 0.55);
    const gy = FY + GHOST[0] * (FC + FG) + FC / 2;
    text(ctx, hold ? "no ghost" : "ghost", FX - 18, gy + 6, { size: 38, weight: 600, color: hold ? C.muted : C.ink, align: "right", alpha: a });
    text(ctx, "not pressed", FX - 18, gy + 44, { size: 32, color: C.muted, align: "right", alpha: a });
  }

  // legend
  ctx.save();
  roundRect(ctx, 56, 896, 30, 30, 5); ctx.fillStyle = C.amber; ctx.fill();
  ctx.strokeStyle = C.amberDark; ctx.lineWidth = 2.5; ctx.stroke();
  ctx.fillStyle = C.current; ctx.beginPath(); ctx.arc(286, 911, 8, 0, 2 * Math.PI); ctx.fill();
  ctx.restore();
  text(ctx, "pressed", 100, 923, { size: 34, color: C.muted });
  text(ctx, "current", 304, 923, { size: 34, color: C.muted });
}

start(renderAt, T_END);
