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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _weight_download import FALLBACK_MIN_RATE, download_first_available  # noqa: E402

MIRROR = "https://huggingface.co/JinukMoon/oh-my-mlip-mirror-eqnorm/resolve/main/{name}"

# variant -> (working ndownloader URL, sha256, size). Mirrors eqnorm.calculator.url_dict
# but uses the subdomain form that returns a real 302 -> S3 instead of 202-blocking.
# The file is also on a byte-identical mirror (figshare 29153315 is MIT-licensed),
# used only when figshare fails or stays slow; both are held to this size and hash.
WEIGHTS = {
    "eqnorm-mptrj": (
        "https://ndownloader.figshare.com/files/55429685",
        "9fd5b97a069e03697e41d2e4c468c5c9b487fc42a2842861ea171a23b9706de5",
        21620384,
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
    url, sha, size = WEIGHTS[variant]
    dest = CACHE_DIR / f"{variant}.pt"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size == size and _sha256(dest) == sha:
        print(f"prestage_eqnorm: {dest} already present and verified; nothing to do.")
        return 0
    if dest.exists():
        print(f"prestage_eqnorm: {dest} is missing/0-byte/wrong-hash; re-fetching.")
        dest.unlink()

    print(f"prestage_eqnorm: downloading {url} -> {dest}")
    sources = [("figshare", url),
               ("the oh-my-mlip mirror on Hugging Face", MIRROR.format(name=dest.name))]
    try:
        used = download_first_available(sources, dest, size=size, sha256=sha,
                                        label="prestage_eqnorm", min_rate=FALLBACK_MIN_RATE)
    except Exception as exc:  # noqa: BLE001 - non-fatal helper
        print(f"prestage_eqnorm: download failed ({exc}); rerun to resume it, or the "
              "framework will retry on first use.", file=sys.stderr)
        return 1
    print(f"prestage_eqnorm: staged {dest} ({dest.stat().st_size} B, sha256 OK, from {used}).")
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
