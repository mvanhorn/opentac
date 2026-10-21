"""Run OpenTac-CAD on this computer and open it in the default browser.

This is `opentac cad`, which the double-click launchers (Start_OpenTac.bat on
Windows, Start_OpenTac.command on macOS/Linux, both at the repository root) run
through uv. It is deliberately tiny: pick a free port, start the same FastAPI app
the hosted site runs, and open the browser once /health answers. Closing the
window stops the server.

The hosted server runs the same command: opentac cad --host 0.0.0.0 --port 10000 --no-browser
"""

import socket
import threading
import time
import urllib.request
import webbrowser

import uvicorn


def pick_free_port(start: int = 8000, tries: int = 20) -> int:
    """First free localhost port at or after `start` (8000 may be taken by
    another dev server — a busy port must fall through, not crash)."""
    for port in range(start, start + tries):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit(
        f"No free port found between {start} and {start + tries - 1}. "
        "Close some running servers and try again."
    )


def open_browser_when_ready(url: str) -> None:
    # Poll /health rather than sleeping a fixed time: first launch imports
    # numpy/shapely/scipy cold and can take a few seconds; later launches are
    # near-instant. Opening early would show a "connection refused" page.
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url + "health", timeout=1)
            break
        except Exception:
            time.sleep(0.25)
    webbrowser.open(url)


def main(host: str = "127.0.0.1", port: int | None = None, open_browser: bool = True) -> None:
    port = port or pick_free_port()
    url = f"http://127.0.0.1:{port}/"
    print()
    print("  OpenTac-CAD is starting...")
    if open_browser:
        print(f"  It will open in your browser at {url}")
        threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()
    else:
        print(f"  Serving on {host}:{port}")
    print("  Keep this window open while you work — closing it stops the app.")
    print()
    uvicorn.run("opentac.cad.main:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
