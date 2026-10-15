// Read-only 3D preview for the generated bump sheet.
//
// Deliberately NOT the STL-import viewer in stl-viewer.js: that one owns face
// selection, seam picking and the unfold pipeline, and it lives inside the STL
// input mode. The bump sheet is an output, and it must be viewable no matter
// which input mode built the board (draw / DXF / STL). So this is its own
// minimal scene — load, orbit, reset — with nothing pickable.
import * as THREE from "three";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

let scene, camera, renderer, controls, mesh;
let container = null;
let initialized = false;
let resizeObserver = null;
let homePos = new THREE.Vector3();
let homeTarget = new THREE.Vector3();

function init(el) {
  container = el;
  if (initialized) return;
  initialized = true;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c1116);
  const w = container.clientWidth || 400;
  const h = container.clientHeight || 300;

  camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 5000);
  // The sheet is built Z-up in the board's own frame. three.js defaults to
  // Y-up, which makes the part look tipped on its side and — worse — makes
  // orbit spin about a horizontal axis, so dragging sideways tumbles it.
  camera.up.set(0, 0, 1);
  camera.position.set(60, -60, 60);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(w, h);
  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;
  controls.minDistance = 1;
  controls.maxDistance = 5000;

  scene.add(new THREE.AmbientLight(0xffffff, 0.22));
  // A hemisphere light plus two directionals. The bumps are shallow — 2mm on a
  // 50mm sheet — so flat overhead light leaves them nearly invisible. The key
  // light is deliberately RAKING (low z) because it is the shadowing on the
  // bump flanks that makes them read as bumps at all; the counter-light keeps
  // the underside from going black when you orbit beneath to check the recess.
  scene.add(new THREE.HemisphereLight(0xcfe4ff, 0x1a232b, 0.42));
  const key = new THREE.DirectionalLight(0xffffff, 1.15);
  key.position.set(0.85, -0.9, 0.32);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.45);
  fill.position.set(-0.9, 0.5, 0.55);
  scene.add(fill);

  resizeObserver = new ResizeObserver(() => {
    if (!container.clientWidth || !container.clientHeight) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });
  resizeObserver.observe(container);

  (function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  })();
}

// Open on a three-quarter view from above: the bumps sit on the top face, so a
// near-level camera would show a silhouette and nothing else.
const VIEW_DIR = new THREE.Vector3(0.34, -0.62, 0.71).normalize();
const FILL_FRACTION = 0.88;   // of the smaller half-viewport

function frame(object) {
  const box = new THREE.Box3().setFromObject(object);
  const center = box.getCenter(new THREE.Vector3());
  // Recenter the geometry rather than moving the orbit target far from the
  // origin — panning stays predictable and the near/far planes stay tight.
  object.position.sub(center);
  box.translate(center.clone().negate());

  const corners = [];
  for (const x of [box.min.x, box.max.x]) {
    for (const y of [box.min.y, box.max.y]) {
      for (const z of [box.min.z, box.max.z]) corners.push(new THREE.Vector3(x, y, z));
    }
  }

  // Fit by PROJECTING the bounding box rather than dividing a bounding radius
  // by the field of view. A flat sheet seen from 45 degrees above foreshortens
  // hard, so its bounding radius badly overstates what it occupies on screen —
  // sizing off the radius left the part filling well under half the viewport.
  // A few iterations converge on a snug fit for any shape and any aspect.
  let dist = Math.max(1e-3, box.getSize(new THREE.Vector3()).length());
  for (let i = 0; i < 8; i += 1) {
    camera.position.copy(VIEW_DIR).multiplyScalar(dist);
    camera.near = Math.max(0.01, dist / 500);
    camera.far = dist * 20;
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();
    let extent = 0;
    for (const c of corners) {
      const p = c.clone().project(camera);
      extent = Math.max(extent, Math.abs(p.x), Math.abs(p.y));
    }
    if (!Number.isFinite(extent) || extent <= 1e-6) break;
    dist *= extent / FILL_FRACTION;
  }

  controls.target.set(0, 0, 0);
  controls.update();
  homePos.copy(camera.position);
  homeTarget.copy(controls.target);
}

/** Show an STL held as an ArrayBuffer. Replaces whatever was shown before. */
export function showBumpStl(el, arrayBuffer) {
  init(el);
  if (mesh) {
    scene.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
    mesh = null;
  }
  // The fit below projects the bounding box, so the aspect ratio has to be
  // right BEFORE framing — otherwise the first view is fitted to whatever size
  // the container happened to have when the scene was first built.
  if (container.clientWidth && container.clientHeight) {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }
  const geometry = new STLLoader().parse(arrayBuffer);
  geometry.computeVertexNormals();
  mesh = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({
    color: 0x6fa8e8, specular: 0x223344, shininess: 24,
    side: THREE.DoubleSide, flatShading: false,
  }));
  scene.add(mesh);
  frame(mesh);
  // The container is revealed by the caller in the same tick, so its size can
  // still be 0 here; one deferred resize gets the aspect right.
  requestAnimationFrame(() => {
    if (!container?.clientWidth) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });
}

export function resetBumpView() {
  if (!initialized || !mesh) return;
  camera.position.copy(homePos);
  controls.target.copy(homeTarget);
  controls.update();
}
