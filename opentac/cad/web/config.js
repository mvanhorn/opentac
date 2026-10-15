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
  : "https://custom-pcb-dev.onrender.com";  // GitHub Pages -> Render deployment
