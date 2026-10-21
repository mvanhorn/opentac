"""Write a real export ZIP for the web page's tests to read.

    python tests/cad/make_export_zip.py [out.zip]

The page's ZIP reader is hand-written binary parsing, and the archive worth
testing it against is one this pipeline produced: a synthetic one built by the
test would encode whatever assumptions the reader already makes. The default
output is opentac/cad/web/tests/fixtures/export.zip, which
design-import.test.mjs reads automatically (and which is regenerated rather
than committed, so it cannot go stale against the exporter).
"""
import base64
import os
import sys


from opentac.cad.main import generate, GenerateRequest

SQUARE = [[-20, -20], [20, -20], [20, 20], [-20, 20]]

DEFAULT_OUT = os.path.join(
    os.path.dirname(__file__), "..", "..", "opentac", "cad", "web", "tests", "fixtures", "export.zip")


def main(out_path=None):
    out_path = os.path.abspath(out_path or DEFAULT_OUT)
    resp = generate(GenerateRequest(
        outline=SQUARE, router="hug", board_mode="expand",
        pixel_w_mm=4, pixel_h_mm=4, pitch_x_mm=4.2, pitch_y_mm=4.2))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(base64.b64decode(resp.zip_b64))
    print(f"wrote {out_path} ({os.path.getsize(out_path)} bytes)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
