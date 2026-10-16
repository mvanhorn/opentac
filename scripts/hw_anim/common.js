// Shared pieces of the hardware animations: frame size, colours, text, the board loader,
// and the capture hook (window.renderAt / window.T_END / window.ready).
export const Q = new URLSearchParams(location.search);
export const W = +(Q.get("w") || 1280), H = +(Q.get("h") || 960), DPR = +(Q.get("dpr") || 1);
export const FONT = '"Helvetica Neue", Helvetica, Arial, sans-serif';

// Light frame (like the site's concept animation), site ink and amber; x/y/z are the first
// three slots of a colour-blind-checked categorical palette.
export const C = {
  bg: "#eceae4", panel: "#f7f6f2", ink: "#1d232b", muted: "#5b6570", rule: "#c8c5bc", grid: "#dedbd3",
  amber: "#e08a2c", amberDark: "#a5530f", current: "#2a78d6",
  x: "#2a78d6", y: "#eb6834", z: "#1baf7a",
};

export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const ease = x => { x = clamp(x); return x * x * (3 - 2 * x); };
export const seg = (t, a, b) => ease((t - a) / (b - a));
export const lerp = (a, b, f) => a + (b - a) * f;

// matplotlib's inferno, 17 stops (the colour map `opentac view` uses)
const INFERNO = ["#000004", "#0b0724", "#210c4a", "#3d0965", "#57106e", "#71196e", "#8a226a", "#a32c61",
  "#bc3754", "#d24644", "#e45a31", "#f1731d", "#f98e09", "#fcac11", "#f9cb35", "#f2ea69", "#fcffa4"]
  .map(h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)));
export function inferno(v) {
  v = clamp(v) * (INFERNO.length - 1);
  const i = Math.min(INFERNO.length - 2, Math.floor(v)), f = v - i, a = INFERNO[i], b = INFERNO[i + 1];
  return `rgb(${a.map((c, k) => Math.round(lerp(c, b[k], f))).join(",")})`;
}

// A 2D canvas over the whole frame, in frame pixels (W x H) whatever the device pixel ratio.
export function overlay(z = 2) {
  const c = document.createElement("canvas");
  c.width = W * DPR; c.height = H * DPR;
  c.style.width = W + "px"; c.style.height = H + "px"; c.style.zIndex = z;
  document.body.appendChild(c);
  const ctx = c.getContext("2d");
  ctx.scale(DPR, DPR);
  return ctx;
}

export function text(ctx, s, x, y, { size = 40, weight = 400, color = C.ink, align = "left", base = "alphabetic", alpha = 1 } = {}) {
  if (alpha <= 0) return 0;
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.font = `${weight} ${size}px ${FONT}`;
  ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = base;
  ctx.fillText(s, x, y);
  const w = ctx.measureText(s).width;
  ctx.restore();
  return w;
}

export function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// Capture hook: the frame grabber calls renderAt(t) for each frame; a browser plays it in real time.
export function start(renderAt, T_END) {
  window.renderAt = renderAt;
  window.T_END = T_END;
  window.ready = true;
  if (!Q.has("capture")) {
    const t0 = performance.now();
    (function loop() { renderAt(((performance.now() - t0) / 1000) % T_END); requestAnimationFrame(loop); })();
  }
}

// A board exported by make_hw_animations.py: meshes grouped by material, in KiCad millimetres,
// three.js axes (x = KiCad x, y = up out of the top face, z = KiCad y).
export async function loadBoard(THREE, name) {
  const meta = await (await fetch(`/data/${name}.json`)).json();
  const bin = await (await fetch(`/data/${name}.bin`)).arrayBuffer();
  const group = new THREE.Group();
  for (const g of meta.groups) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(bin, g.pos, g.nv * 3), 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(bin, g.nrm, g.nv * 3), 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(bin, g.idx, g.nf * 3), 1));
    const [r, gg, b, a] = g.rgba;
    const col = new THREE.Color().setRGB(r / 255, gg / 255, b / 255, THREE.SRGBColorSpace);
    const mat = new THREE.MeshStandardMaterial({
      color: col, metalness: g.metal * 0.6, roughness: Math.max(0.3, g.rough), side: THREE.DoubleSide,
      transparent: a < 250, opacity: a < 250 ? a / 255 : 1, depthWrite: a >= 250,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = mesh.receiveShadow = true;
    if (a < 250) mesh.renderOrder = 2;
    group.add(mesh);
  }
  return { group, meta };
}

// KiCad (x, y) in mm -> the board group's local coordinates.
export const kpos = (THREE, x, y, up = 0) => new THREE.Vector3(x, up, y);
