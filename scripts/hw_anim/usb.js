// A still: both readout boards with a micro-USB cable plugged in (the boards' USB connectors are
// micro-B receptacles at a board edge: USB2 on the 60-pin board, USB2 on the 30-pin top board).
// The plugs and cables are plain shapes with micro-B proportions.
import * as THREE from "three";
import { C, W, H, DPR, overlay, text, start, loadBoard, pairMaker } from "./common.js";

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(DPR); renderer.setSize(W, H);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.domElement.style.zIndex = 1;
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(C.bg);
const camera = new THREE.PerspectiveCamera(24, W / H, 10, 3000);
camera.position.set(4, 120, 106); camera.lookAt(4, 0, -6);
scene.add(new THREE.HemisphereLight(0xffffff, 0xd8d2c4, 1.4));
const sun = new THREE.DirectionalLight(0xffffff, 2.0);
sun.position.set(-40, 420, 140); sun.castShadow = true;
sun.shadow.mapSize.set(4096, 4096);
Object.assign(sun.shadow.camera, { left: -160, right: 160, top: 120, bottom: -120, near: 50, far: 800 });
sun.shadow.bias = -0.0004; sun.shadow.radius = 5;
scene.add(sun);
const fill = new THREE.DirectionalLight(0xeef2ff, 0.6); fill.position.set(200, 120, 200); scene.add(fill);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), new THREE.MeshStandardMaterial({ color: C.bg, roughness: 1 }));
floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);

const mat = (color, o = {}) => new THREE.MeshStandardMaterial(Object.assign({ color, roughness: 0.6, metalness: 0 }, o));
function box(w, h, d, m, x = 0, y = 0, z = 0) {
  const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);
  b.position.set(x, y, z); b.castShadow = b.receiveShadow = true;
  return b;
}
// a micro-B plug in its own frame: origin at the receptacle's mouth, +z out of it
function plug() {
  const g = new THREE.Group();
  g.add(box(6.9, 1.85, 5.0, mat("#b8bcc2", { metalness: 0.8, roughness: 0.3 }), 0, 0, -1.8));   // shell, mostly inside
  g.add(box(10.6, 6.2, 15.0, mat("#2a2d31", { roughness: 0.7 }), 0, 0, 8.2));                    // overmould
  const relief = new THREE.Mesh(new THREE.CylinderGeometry(1.8, 2.6, 5, 24), mat("#2a2d31", { roughness: 0.7 }));
  relief.rotation.x = Math.PI / 2; relief.position.z = 18.2; relief.castShadow = true; g.add(relief);
  return g;
}
// the cable: on from the plug's back, down to the table and off the frame
function cable(from, dir, len = 160) {
  const d = dir.clone().setY(0).normalize();
  const pts = [from, from.clone().addScaledVector(d, 8), from.clone().addScaledVector(d, 22).setY(1.7),
    from.clone().addScaledVector(d, 45).setY(1.7).add(new THREE.Vector3(0, 0, 10)), from.clone().addScaledVector(d, len).setY(1.7).add(new THREE.Vector3(0, 0, 40))];
  const m = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 200, 1.6, 16, false), mat("#232528", { roughness: 0.55 }));
  m.castShadow = true; scene.add(m);
}
function attach(parent, p) {                        // plug group in `parent`'s frame -> cable from its back
  parent.add(p);
  scene.updateMatrixWorld(true);
  const back = p.localToWorld(new THREE.Vector3(0, 0, 20.6)), dir = p.localToWorld(new THREE.Vector3(0, 0, 30)).sub(back);
  cable(back, dir);
  return p.localToWorld(new THREE.Vector3(0, 3.5, 8));
}

// 60-pin board, its USB edge to the left
const b60 = await loadBoard(THREE, "readout_60pin");
const [x0, y0, x1, y1] = b60.meta.outline;
const g60 = new THREE.Group(); g60.rotation.y = -Math.PI / 2; g60.position.set(-24, 1.0, 0);
b60.group.position.set(-(x0 + x1) / 2, 0, -(y0 + y1) / 2);
g60.add(b60.group); scene.add(g60);
const p60 = plug(); p60.position.set(b60.meta.parts.USB2[0], 2.95, y1);          // mouth at the board's edge
const lab60 = attach(b60.group, p60);

// 30-pin pair, its USB edge to the right
const { makePair, parts, topOutline } = await pairMaker(THREE);
const { pair, toPair } = makePair();
pair.rotation.y = -Math.PI / 2; pair.position.set(36, 0, 0); scene.add(pair);
const p30 = plug(); p30.position.copy(toPair([topOutline[2], parts.USB2[1]], 2.95)); p30.rotation.y = Math.PI;   // KiCad +x is -z in the pair's frame
const lab30 = attach(pair, p30);

const ctx = overlay(2);
const project = v => { const p = v.clone().project(camera); return [(p.x + 1) / 2 * W, (1 - p.y) / 2 * H]; };
function renderAt() {
  renderer.render(scene, camera);
  ctx.clearRect(0, 0, W, H);
  const a = project(new THREE.Vector3(-24, 0, 21)), b = project(new THREE.Vector3(36, 0, 14));
  text(ctx, "60-pin board", a[0], a[1] + 50, { size: 44, weight: 600, align: "center" });
  text(ctx, "30-pin pair", b[0], b[1] + 50, { size: 44, weight: 600, align: "center" });
  const u = project(lab60), v = project(lab30);
  text(ctx, "micro-USB", Math.max(u[0], 120), u[1] - 36, { size: 40, color: C.muted, align: "center" });
  text(ctx, "micro-USB", Math.min(v[0], W - 120), v[1] - 36, { size: 40, color: C.muted, align: "center" });
}
start(renderAt, 1);
