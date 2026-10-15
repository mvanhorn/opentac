// Open a design that was built somewhere else — currently the plain-language
// chat prototype in web-prototype/, which produces an outline and a full
// parameter set and then wants to hand the user to the real editor to keep
// working on it by hand.
//
// The payload rides in the URL fragment (#otc=<base64url JSON>), so it never
// reaches a server and needs no endpoint, no storage and no CORS. It is the
// same {outline, params, label} shape the example gallery uses, and it is
// delivered through the same "otc:load-example" event, so a handed-over design
// is editable exactly like an example — no second code path to keep in sync.
//
// Entirely inert unless the fragment is present.

function decode(raw) {
  const b64 = raw.replace(/-/g, "+").replace(/_/g, "/");
  const pad = b64 + "=".repeat((4 - (b64.length % 4)) % 4);
  return JSON.parse(decodeURIComponent(escape(atob(pad))));
}

// Captured at module-evaluation time, NOT on DOMContentLoaded, and this file
// is loaded before app.js so that this runs first. views.js owns the fragment
// as a view router (#landing / #tool / #help): on startup it rewrites
// location.hash to a view name, which would wipe this payload before it could
// be read. Reading it here — synchronously, before any of that — is the whole
// reason the capture and the dispatch are split apart.
const CAPTURED = (() => {
  const m = /[#&]otc=([A-Za-z0-9_-]+)/.exec(location.hash || "");
  if (!m) return null;
  // Strip the payload immediately, so views.js sees a fragment it understands
  // and a reload cannot re-import the original outline over the user's edits.
  try {
    history.replaceState(null, "", location.pathname + location.search + "#tool");
  } catch { /* file:// or a sandbox: harmless, the payload is already read */ }
  try {
    return decode(m[1]);
  } catch (err) {
    console.warn("otc: could not decode the design in the URL", err);
    return null;
  }
})();

function load() {
  const payload = CAPTURED;
  if (!payload) return;

  const { outline, params = {}, label = "imported design" } = payload;
  if (!Array.isArray(outline) || outline.length < 3) {
    console.warn("otc: URL payload has no usable outline");
    return;
  }

  window.dispatchEvent(new CustomEvent("otc:load-example", {
    detail: {
      outline, params, label,
      // The cable edge was already decided when this outline was built —
      // edge 0->1 is flat and deliberate — so this behaves like a built-in
      // example rather than a raw DXF: generate straight away.
      requireCableEdge: false,
      autoGenerate: true,
    },
  }));
}

if (document.readyState === "loading") {
  // The listener lives in app.js, which is a module and therefore deferred;
  // firing on DOMContentLoaded guarantees it is registered first.
  document.addEventListener("DOMContentLoaded", load);
} else {
  load();
}
