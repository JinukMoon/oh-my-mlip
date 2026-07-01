#!/usr/bin/env python3
"""Pre-stage the MatRIS weight into the framework's own cache.

``MatRIS.load('matris_10m_oam')`` downloads its checkpoint *inside the matris
package* to ``~/.cache/matris/MatRIS_10M_OAM.pth.tar`` via
``torch.hub.download_url_to_file`` against a plain ``figshare.com/ndownloader``
URL, and skips the download if that file already exists. On some networks that
URL 202-blocks and leaves a 0-byte file, which then fails to load while the
package refuses to re-download. See docs/host_requirements.md and the MatRIS
note in models.json.

This helper pre-stages the exact file the loader expects, from the working
``ndownloader.figshare.com`` subdomain. MatRIS is not in the /TGM reference set,
so there is no owner-verified sha256 to check against; instead we require a
plausibly-sized checkpoint (>1 MiB) and print the computed sha256 for the record.
Pure stdlib, idempotent, non-fatal by design.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

# model -> (working ndownloader URL, cache filename). Mirrors MatRIS.model.model
# .MatRIS.load DOWNLOAD_URLS, but uses the subdomain form that returns a real
# 302 -> S3 instead of 202-blocking.
WEIGHTS = {
    "matris_10m_oam": ("https://ndownloader.figshare.com/files/59142728", "MatRIS_10M_OAM.pth.tar"),
    "matris_10m_mp": ("https://ndownloader.figshare.com/files/59143058", "MatRIS_10M_MP.pth.tar"),
}

# Known-good sha256 of the staged checkpoint (verified 2026-07-01 on the RTX 4060
# Ti host). WARN-ONLY: a mismatch is logged but does NOT fail the pre-stage,
# because the upstream figshare file may be legitimately re-published. Models
# without a recorded fingerprint are skipped (no reference to compare).
EXPECTED_SHA256 = {
    "matris_10m_oam": "c033abc53601a74f10d9b7fec0f658220c013c3b11d4d405f1d32136d4c2b067",
}
CACHE_DIR = Path(os.path.expanduser("~/.cache/matris"))
MIN_BYTES = 1 << 20  # a real checkpoint is many MiB; guard against 0-byte/202 bodies


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _verify_sha_warn(model: str, dest: Path) -> None:
    """Warn-only sha256 check: logs a warning on mismatch, never fails."""
    expected = EXPECTED_SHA256.get(model)
    if not expected:
        return
    actual = _sha256(dest)
    if actual != expected:
        print(
            f"prestage_matris: WARNING sha256 mismatch for {dest}: expected "
            f"{expected[:12]}.. got {actual[:12]}.. (upstream figshare file may "
            "have been re-published; not failing).",
            file=sys.stderr,
        )
    else:
        print(f"prestage_matris: sha256 verified ({actual[:12]}..).")


def prestage(model: str) -> int:
    model = model.lower()
    if model not in WEIGHTS:
        print(f"prestage_matris: unknown model {model!r}; known: {list(WEIGHTS)}", file=sys.stderr)
        return 2
    url, fname = WEIGHTS[model]
    dest = CACHE_DIR / fname
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size >= MIN_BYTES:
        print(f"prestage_matris: {dest} already present ({dest.stat().st_size} B); nothing to do.")
        _verify_sha_warn(model, dest)
        return 0
    if dest.exists():
        print(f"prestage_matris: {dest} is missing/too-small (0-byte/202 body); re-fetching.")
        dest.unlink()

    print(f"prestage_matris: downloading {url} -> {dest}")
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(CACHE_DIR), suffix=".part")
    os.close(tmp_fd)
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(url, timeout=180) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        size = tmp.stat().st_size
        if size < MIN_BYTES:
            tmp.unlink(missing_ok=True)
            print(f"prestage_matris: downloaded body too small ({size} B); likely a 202 block, leaving cache empty.", file=sys.stderr)
            return 1
        tmp.replace(dest)
    except Exception as exc:  # noqa: BLE001 - non-fatal helper
        tmp.unlink(missing_ok=True)
        print(f"prestage_matris: download failed ({exc!r}); the framework will retry on first use.", file=sys.stderr)
        return 1
    print(f"prestage_matris: staged {dest} ({dest.stat().st_size} B, sha256 {_sha256(dest)}).")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="matris_10m_oam", help="MatRIS model name (default: matris_10m_oam)")
    ap.add_argument("--target-root", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    return prestage(args.model)


if __name__ == "__main__":
    raise SystemExit(main())
