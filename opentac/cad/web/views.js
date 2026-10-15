// Top-level view routing: toggles between the landing, tool, and help
// sections. A plain hash router (#landing / #tool / #help) keeps links
// shareable and the browser back/forward buttons working, without adding a
// framework or a second HTML page (all tool state lives in module-level JS
// in app.js/stl-viewer.js, so a real page navigation would lose it).

const VIEWS = ["landing", "tool", "help"];
const sections = {
  landing: document.getElementById("view-landing"),
  tool: document.getElementById("view-tool"),
  help: document.getElementById("view-help"),
};

function normalizeHash() {
  const raw = (location.hash || "").replace(/^#/, "");
  return VIEWS.includes(raw) ? raw : "landing";
}

export function showView(name) {
  const target = VIEWS.includes(name) ? name : "landing";
  for (const key of VIEWS) {
    if (sections[key]) sections[key].hidden = key !== target;
  }
  if (location.hash.replace(/^#/, "") !== target) {
    location.hash = target;
  }
  // Lets stl-viewer.js re-run its Three.js resize logic once its container
  // is actually visible (a WebGL canvas can't size itself against a
  // display:none/hidden parent).
  window.dispatchEvent(new CustomEvent("otc:view-shown", { detail: { view: target } }));
}

export function currentView() {
  return normalizeHash();
}

window.addEventListener("hashchange", () => showView(normalizeHash()));

document.getElementById("nav-home")?.addEventListener("click", () => showView("landing"));
document.getElementById("nav-tool")?.addEventListener("click", () => showView("tool"));
document.getElementById("nav-help")?.addEventListener("click", () => showView("help"));

// A fresh page load always starts at the landing screen, regardless of any
// hash left over from a previous visit (deep-linking within a session via
// the nav buttons still works normally).
showView("landing");
