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
``ndownloader.figshare.com`` subdomain. A checkpoint with a recorded size and
sha256 (below) must match both; it is also on a byte-identical mirror (figshare
30475253 is CC BY 4.0), used only when figshare fails or stays slow and held to
the same size and hash. A checkpoint without a record must arrive whole and be
plausibly sized (>1 MiB).
Pure stdlib, idempotent, non-fatal by design.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _weight_download import FALLBACK_MIN_RATE, download, download_first_available  # noqa: E402

# model -> (working ndownloader URL, cache filename). Mirrors MatRIS.model.model
# .MatRIS.load DOWNLOAD_URLS, but uses the subdomain form that returns a real
# 302 -> S3 instead of 202-blocking.
WEIGHTS = {
    "matris_10m_oam": ("https://ndownloader.figshare.com/files/59142728", "MatRIS_10M_OAM.pth.tar"),
    "matris_10m_mp": ("https://ndownloader.figshare.com/files/59143058", "MatRIS_10M_MP.pth.tar"),
}

# Recorded (size, sha256) of a checkpoint, enforced: a file that differs is never
# staged. Models without a record are only checked for a whole, plausible file.
PINNED = {
    "matris_10m_oam": (42273174, "c033abc53601a74f10d9b7fec0f658220c013c3b11d4d405f1d32136d4c2b067"),
}
MIRROR = "https://huggingface.co/JinukMoon/oh-my-mlip-mirror-matris/resolve/main/{name}"
CACHE_DIR = Path(os.path.expanduser("~/.cache/matris"))
MIN_BYTES = 1 << 20  # a real checkpoint is many MiB; guard against 0-byte/202 bodies


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _present(model: str, dest: Path) -> bool:
    if not dest.exists():
        return False
    if model in PINNED:
        size, sha = PINNED[model]
        return dest.stat().st_size == size and _sha256(dest) == sha
    return dest.stat().st_size >= MIN_BYTES


def prestage(model: str) -> int:
    model = model.lower()
    if model not in WEIGHTS:
        print(f"prestage_matris: unknown model {model!r}; known: {list(WEIGHTS)}", file=sys.stderr)
        return 2
    url, fname = WEIGHTS[model]
    dest = CACHE_DIR / fname
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if _present(model, dest):
        print(f"prestage_matris: {dest} already present ({dest.stat().st_size} B); nothing to do.")
        return 0
    if dest.exists():
        print(f"prestage_matris: {dest} is too small or differs from the record; re-fetching.")
        dest.unlink()

    print(f"prestage_matris: downloading {url} -> {dest}")
    try:
        if model in PINNED:
            size, sha = PINNED[model]
            sources = [("figshare", url),
                       ("the oh-my-mlip mirror on Hugging Face", MIRROR.format(name=fname))]
            used = download_first_available(sources, dest, size=size, sha256=sha,
                                            label="prestage_matris", min_rate=FALLBACK_MIN_RATE)
        else:
            download(url, dest, label="prestage_matris")
            used = "figshare"
            if dest.stat().st_size < MIN_BYTES:
                dest.unlink()
                print("prestage_matris: downloaded body too small; likely a 202 block, "
                      "leaving cache empty.", file=sys.stderr)
                return 1
    except Exception as exc:  # noqa: BLE001 - non-fatal helper
        print(f"prestage_matris: download failed ({exc}); rerun to resume it, or the "
              "framework will retry on first use.", file=sys.stderr)
        return 1
    print(f"prestage_matris: staged {dest} ({dest.stat().st_size} B, sha256 {_sha256(dest)}, from {used}).")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="matris_10m_oam", help="MatRIS model name (default: matris_10m_oam)")
    ap.add_argument("--target-root", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    return prestage(args.model)


if __name__ == "__main__":
    raise SystemExit(main())
