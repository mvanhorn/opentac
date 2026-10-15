"""Preview opentac.github.io on this computer, as it will be published.

    uv run --extra docs python scripts/preview_site.py [--port 8100] [--no-build] [--no-browser]

Serves the website (built from docs/ into site/) at /, OpenTac CAD at /cad/, and
the CAD backend at the same address, so "Try it online" works without Render.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
WEB = ROOT / "opentac" / "cad" / "web"


def make_app():
    from fastapi.routing import APIRoute
    from starlette.responses import RedirectResponse
    from starlette.staticfiles import StaticFiles

    from opentac.cad.main import app as cad

    api_paths = {r.path for r in cad.routes if isinstance(r, APIRoute)}
    site = StaticFiles(directory=SITE, html=True)
    web = StaticFiles(directory=WEB, html=True)

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            return await cad(scope, receive, send)
        path = scope.get("path", "/")
        if path == "/cad":
            return await RedirectResponse("/cad/")(scope, receive, send)
        if path.startswith("/cad/"):
            sub = path[len("/cad"):]
            return await web(dict(scope, path=sub, raw_path=sub.encode(), root_path=""), receive, send)
        if path in api_paths or path.startswith(("/api/", "/dev/")):
            return await cad(scope, receive, send)
        return await site(scope, receive, send)

    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--no-build", action="store_true", help="serve the existing site/ as it is")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not args.no_build:
        subprocess.run([sys.executable, "-m", "mkdocs", "build", "-q", "-d", str(SITE)],
                       cwd=ROOT, check=True)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"\n  Preview: {url}   (OpenTac CAD at {url}cad/)\n  Ctrl-C stops it.\n")
    if not args.no_browser:
        def _open():
            for _ in range(240):
                try:
                    urllib.request.urlopen(url, timeout=1)
                    break
                except Exception:
                    time.sleep(0.25)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()
    import uvicorn
    uvicorn.run(make_app(), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
