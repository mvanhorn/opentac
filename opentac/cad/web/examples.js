// Example design gallery: finished, ready-to-download designs a first-time
// visitor can open with one click. Each entry stores an (outline, params)
// config and regenerates through the normal /generate path on load — the same
// code path as a hand-drawn design — so examples never go stale against the
// pipeline and stay fully editable (taxel size, spacing, cable edge, edge
// cuts, connector) after loading.
//
// The two gripper outlines are real parts, unfolded at true scale with the
// STL import flow and captured here as plain 2D outlines: the soft-fingertip
// wrap comes from finger_tip_soft.stl (flat side + curved contact face + flat
// side, joined in the corner-picking flow) and the Yubi finger pad from the
// Yubi gripper's curved pad surface (screw holes ignored). The "facts" lines
// quote the result of generating the stored config (all DRC-clean).
import { showView } from "./views.js";
import { setSource } from "./design-name.js";
import { applyModeVisibility } from "./landing.js";

// Parameter values every example starts from (the app's own defaults), so a
// card always reproduces its advertised result no matter what was typed into
// the form earlier. Individual examples override just their grid size.
const BASELINE_PARAMS = {
  pixel_w_mm: 4.0, pixel_h_mm: 4.0, pitch_x_mm: 4.2, pitch_y_mm: 4.2,
  edge_keepout_mm: 0.5, trace_w_mm: 0.2, gap_mm: 0.2, clearance_mm: 0.2,
  board_edge_clear_mm: 0.2, edge_clear_mm: 0.1, center_clear_mm: 0.2,
  via_drill_mm: 0.15, via_dia_mm: 0.35, routing_margin_mm: 3.0,
  follow_main_padding_mm: 2.0, cable_length_mm: null, tail_width_mm: null,
  tail_side_padding_mm: 0.0, connector_shoulder_fillet_mm: 1.0,
  board_mode: "expand", router: "hug",
};

const GRID_3MM = { pixel_w_mm: 3.0, pixel_h_mm: 3.0, pitch_x_mm: 3.2, pitch_y_mm: 3.2 };

export const EXAMPLES = [
  {
    id: "square",
    title: "Square patch",
    blurb: "A 40 × 40 mm square, the simplest example.",
    facts: "81 taxels · 4 × 4 mm at 4.2 mm pitch",
    outline: [[-20, -20], [20, -20], [20, 20], [-20, 20]],
    params: {},
  },
  {
    id: "rect",
    title: "Rectangle patch",
    blurb: "A 50 × 30 mm rectangle, for example for a flat gripper palm plate.",
    facts: "66 taxels · 4 × 4 mm at 4.2 mm pitch",
    outline: [[-25, -15], [25, -15], [25, 15], [-25, 15]],
    params: {},
  },
  {
    id: "tee",
    title: "T-shape",
    blurb: "A 24 mm wide stem under a 68 × 18 mm top. The wiring from the top runs down the stem to the connector.",
    facts: "182 taxels · 3 × 3 mm at 3.2 mm pitch",
    outline: [[-12, 0], [12, 0], [12, 34], [34, 34], [34, 52], [-34, 52], [-34, 34], [-12, 34]],
    params: GRID_3MM,
  },
  {
    id: "arch",
    title: "Arch",
    blurb: "A 60 × 40 mm upside-down U with a 24 × 25 mm opening, for example to fit around a bolt or a strap slot.",
    facts: "144 taxels · 3 × 3 mm at 3.2 mm pitch",
    outline: [[0, 0], [18, 0], [18, 25], [42, 25], [42, 0], [60, 0], [60, 40], [0, 40]],
    params: GRID_3MM,
  },
  {
    id: "fingertip-wrap",
    title: "Soft fingertip wrap",
    blurb: "The front face and both sides of a soft gripper fingertip, unfolded flat from its STL model. The flex board wraps back around the fingertip.",
    facts: "64 taxels · 3 × 3 mm at 3.2 mm pitch · from an STL",
    outline: [
      [-24.193, -11.859], [-16.470, -11.935], [-16.076, -11.499],
      [16.156, -8.934], [16.604, -9.315], [24.193, -8.317],
      [21.507, 11.935], [6.329, 10.264], [-7.515, 9.141], [-24.091, 8.501],
    ],
    params: GRID_3MM,
  },
  {
    id: "yubi-pad",
    title: "Yubi finger pad",
    blurb: "The curved finger pad of a Yubi gripper, unfolded flat at true scale from its STL model. Taxels cover the whole pad; its screw holes are ignored.",
    facts: "122 taxels · 3 × 3 mm at 3.2 mm pitch · from an STL",
    outline: [
      [-3.247, -34.645], [13.979, -34.439], [14.854, -30.847],
      [14.636, -15.087], [14.060, 34.645], [-7.099, 34.403],
      [-12.381, 28.098], [-14.849, 25.675], [-14.605, 4.225],
    ],
    params: GRID_3MM,
  },
];

// Small SVG preview of an outline: same colors as the drawing canvas (blue
// shape, orange cable edge) so the gallery teaches the color language before
// the user ever reaches the tool.
function outlineThumbnail(outline) {
  const xs = outline.map((p) => p[0]);
  const ys = outline.map((p) => p[1]);
  const minx = Math.min(...xs), maxx = Math.max(...xs);
  const miny = Math.min(...ys), maxy = Math.max(...ys);
  const w = maxx - minx, h = maxy - miny;
  const pad = Math.max(w, h) * 0.08;
  // mm is y-up, SVG is y-down: flip by mapping y -> maxy - y.
  const pts = outline.map(([x, y]) => `${(x - minx + pad).toFixed(2)},${(maxy - y + pad).toFixed(2)}`);
  const [a, b] = [outline[0], outline[1]];
  const strokeW = Math.max(w, h) / 90;
  return (
    `<svg viewBox="0 0 ${(w + 2 * pad).toFixed(2)} ${(h + 2 * pad).toFixed(2)}" ` +
    `preserveAspectRatio="xMidYMid meet" aria-hidden="true">` +
    `<polygon points="${pts.join(" ")}" fill="rgba(79,157,255,0.16)" ` +
    `stroke="#4f9dff" stroke-width="${strokeW}" stroke-linejoin="round"/>` +
    `<line x1="${(a[0] - minx + pad).toFixed(2)}" y1="${(maxy - a[1] + pad).toFixed(2)}" ` +
    `x2="${(b[0] - minx + pad).toFixed(2)}" y2="${(maxy - b[1] + pad).toFixed(2)}" ` +
    `stroke="#ffb84d" stroke-width="${strokeW * 2.4}" stroke-linecap="round"/>` +
    `</svg>`
  );
}

function loadExample(example) {
  // An example's title is a better download name than a timestamp.
  setSource(example.title);
  applyModeVisibility("draw");
  showView("tool");
  window.dispatchEvent(new CustomEvent("otc:load-example", {
    detail: {
      label: example.title,
      outline: example.outline,
      params: { ...BASELINE_PARAMS, ...example.params },
    },
  }));
}

const galleryEl = document.getElementById("landing-examples");
if (galleryEl) {
  for (const example of EXAMPLES) {
    const card = document.createElement("article");
    card.className = "panel example-card";
    card.innerHTML =
      `<div class="example-card__thumb">${outlineThumbnail(example.outline)}</div>` +
      `<h2>${example.title}</h2>` +
      `<p>${example.blurb}</p>` +
      `<div class="example-card__foot">` +
      `<span class="example-card__facts">${example.facts}</span>` +
      `<button type="button" class="primary">Open this design</button>` +
      `</div>`;
    card.querySelector("button").addEventListener("click", () => loadExample(example));
    galleryEl.appendChild(card);
  }
}

// ---- Real robot end-effector gallery -------------------------------------
// Each card is a surface of an actual robot part, selected by hand in the STL
// tool and saved as an annotation (examples/stl/end_effectors/annotations/);
// stl-examples.json carries the resulting unfolded 2D outline plus a
// thumbnail rendered from the 3D model with the chosen faces highlighted.
//
// Clicking loads the stored outline straight into the board editor — the
// unfold already happened when the example was curated, so the outline needs
// no mesh analysis or re-unfolding at click time; the 3D preview mesh loads
// separately (otcShowExampleInViewer in stl-viewer.js). The cable
// edge is deliberately not preset: which edge the connector should exit is a
// decision about the user's own build, so they choose it exactly as they
// would for a DXF import or a hand-drawn outline.
const stlGalleryEl = document.getElementById("landing-stl-examples");
const stlFilterEl = document.getElementById("stl-example-filter");

function loadStlExample(example) {
  const label = `${example.robot} — ${example.part}`;
  // STL mode, not draw mode: the part stays on screen in the normal 3D
  // viewer — rotatable, with its sensor surface highlighted — so the context
  // of what is being designed for survives the jump from gallery to editor.
  applyModeVisibility("stl");
  showView("tool");
  // Outline first and synchronously (it is already in the manifest), so the
  // board is usable regardless of what the 3D fetch does.
  window.dispatchEvent(new CustomEvent("otc:load-example", {
    detail: {
      label,
      outline: example.outline,
      params: BASELINE_PARAMS,
      requireCableEdge: true,
      autoGenerate: false,
    },
  }));
  window.otcShowExampleInViewer?.(example.mesh, example.triangles, label);
}

if (stlGalleryEl) {
  fetch("stl-examples.json")
    .then((r) => r.json())
    .then(({ examples }) => {
      if (!examples || !examples.length) return;
      const robots = [...new Set(examples.map((e) => e.robot))].sort();
      if (stlFilterEl) {
        stlFilterEl.innerHTML =
          `<option value="">All robots (${examples.length} surfaces)</option>` +
          robots.map((r) => `<option value="${r}">${r}</option>`).join("");
        stlFilterEl.addEventListener("change", () => {
          const want = stlFilterEl.value;
          for (const card of stlGalleryEl.children) {
            card.hidden = Boolean(want) && card.dataset.robot !== want;
          }
        });
      }
      for (const example of examples) {
        const card = document.createElement("article");
        card.className = "panel stl-example-card";
        card.dataset.robot = example.robot;
        card.innerHTML =
          `<div class="stl-example-card__thumb">` +
          `<img src="${example.thumbnail}" alt="3D model of the ${example.robot} ${example.part}, with the sensor surface highlighted" loading="lazy" />` +
          `</div>` +
          `<div class="stl-example-card__body">` +
          `<h3>${example.robot}</h3>` +
          `<p class="stl-example-card__part">${example.part}</p>` +
          `<p class="stl-example-card__meta">${example.size_mm[0]} × ${example.size_mm[1]} mm · ${example.selection}</p>` +
          `</div>` +
          `<button type="button" class="primary">Use this shape</button>`;
        card.querySelector("button").addEventListener("click", () => loadStlExample(example));
        stlGalleryEl.appendChild(card);
      }
    })
    .catch(() => {
      stlGalleryEl.innerHTML =
        '<p class="hint">Could not load the robot example gallery.</p>';
    });
}
