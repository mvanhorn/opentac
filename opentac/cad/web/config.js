// API base URL. Empty string = same-origin (the FastAPI backend serves this page,
// so /generate and /health are reachable at the same host). GitHub Pages serves
// this frontend as static files with no backend at the same origin, so there it
// must point at the Render deployment.
//
// Decided at runtime rather than hardcoded: a hardcoded Render URL silently
// sends every request from a locally served page to the DEPLOYED backend — the
// local backend's new features then appear to "not work" while old production
// code answers. (This actually happened with the hug router.)
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]"]);
export const API_BASE = LOCAL_HOSTS.has(location.hostname)
  ? ""                                      // local FastAPI serves this page itself
  : "https://opentac-cad.onrender.com";     // opentac.github.io/cad/ -> Render (render.yaml)

// The OpenTac website, for links out of the tool (header, download panel, help).
// On opentac.github.io, and in scripts/preview_site.py, this page is served at
// <site>/cad/, so the site is one level up; that also keeps a fork's own Pages
// site pointing at itself. Run by `opentac cad` the tool is at the server root,
// and the site is the published one. Links carry the published URL in the HTML,
// so they still work if this script does not run.
const PUBLISHED_SITE = "https://opentac.github.io/";
export const SITE_BASE = /\/cad\/(index\.html)?$/.test(location.pathname)
  ? new URL("../", location.href).href
  : PUBLISHED_SITE;
