#!/usr/bin/env python3
"""Pre-stage the eqnorm weight into the framework's own cache.

`EqnormCalculator` downloads its checkpoint *inside the eqnorm package* to
``~/.cache/eqnorm/<variant>.pt`` via ``wget`` against a plain ``figshare.com``
URL, and skips the download if that file already exists. On some networks that
plain URL 202-blocks and leaves a 0-byte file, which then fails
``torch.load`` with "Ran out of input" while the package refuses to re-download
(the file "exists"). See docs/host_requirements.md and the Eqnorm note in
models.json.

This helper pre-stages the exact same file the calculator expects, but from the
working ``ndownloader.figshare.com`` subdomain, and verifies the sha256. It is
idempotent: a present, correct file is left untouched; a missing / 0-byte /
wrong-hash file is (re)fetched. Pure stdlib, so it runs with any interpreter
(install.sh invokes it after building the eqnorm env). Non-fatal by design —
if the network is down the framework can still try its own download later.

Model variant weights are keyed below; the default matches models.json's
Eqnorm-MPtrj (EqnormCalculator(..., model_variant="eqnorm-mptrj")).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

# variant -> (working ndownloader URL, sha256). Mirrors eqnorm.calculator.url_dict
# but uses the subdomain form that returns a real 302 -> S3 instead of 202-blocking.
WEIGHTS = {
    "eqnorm-mptrj": (
        "https://ndownloader.figshare.com/files/55429685",
        "9fd5b97a069e03697e41d2e4c468c5c9b487fc42a2842861ea171a23b9706de5",
    ),
}
CACHE_DIR = Path(os.path.expanduser("~/.cache/eqnorm"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prestage(variant: str) -> int:
    if variant not in WEIGHTS:
        print(f"prestage_eqnorm: unknown variant {variant!r}; known: {list(WEIGHTS)}", file=sys.stderr)
        return 2
    url, sha = WEIGHTS[variant]
    dest = CACHE_DIR / f"{variant}.pt"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0 and _sha256(dest) == sha:
        print(f"prestage_eqnorm: {dest} already present and verified; nothing to do.")
        return 0
    if dest.exists():
        print(f"prestage_eqnorm: {dest} is missing/0-byte/wrong-hash; re-fetching.")
        dest.unlink()

    print(f"prestage_eqnorm: downloading {url} -> {dest}")
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(CACHE_DIR), suffix=".part")
    os.close(tmp_fd)
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        got = _sha256(tmp)
        if got != sha:
            tmp.unlink(missing_ok=True)
            print(f"prestage_eqnorm: sha256 mismatch (got {got}, want {sha}); leaving cache empty.", file=sys.stderr)
            return 1
        tmp.replace(dest)
    except Exception as exc:  # noqa: BLE001 - non-fatal helper
        tmp.unlink(missing_ok=True)
        print(f"prestage_eqnorm: download failed ({exc!r}); the framework will retry on first use.", file=sys.stderr)
        return 1
    print(f"prestage_eqnorm: staged {dest} ({dest.stat().st_size} B, sha256 OK).")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="eqnorm-mptrj", help="eqnorm model variant (default: eqnorm-mptrj)")
    # install.sh passes --target-root uniformly to weight helpers; eqnorm ignores it
    # (its cache path is fixed inside the package), but we accept it so the hook stays generic.
    ap.add_argument("--target-root", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    return prestage(args.variant)


if __name__ == "__main__":
    raise SystemExit(main())
