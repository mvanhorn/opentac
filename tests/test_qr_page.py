"""The website's QR decoder (docs/javascripts/qr.js) must download the file the CLI writes.

scripts/check_qr_page.py runs the page's decoder under node on every example
design and compares it byte for byte with `opentac manifest from-qr`. A change
to the manifest text, the format, or the number formatting on one side only
shows up here.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_qr_page_writes_the_same_manifest_as_the_cli():
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_qr_page.py")],
                         cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert res.stdout.startswith("OK:"), res.stdout
