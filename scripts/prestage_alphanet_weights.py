#!/usr/bin/env python3
"""Pre-stage the AlphaNet weight + config into models/alphanet/.

Beta-test finding (RTX A4500 host, 2026-07): the alphanet sidecar builds the env
and fetch.py stages the figshare checkpoint, but the required `oma.json` config
(referenced by the registry inference line) was never auto-fetched, so
single_point failed until it was pulled manually. This helper stages BOTH files
up front, stdlib-only, so install.sh's prestage hook makes a fresh clone
runnable with no manual step:

  * alex_0410.ckpt  <- https://ndownloader.figshare.com/files/53851139
                       (sha256 hard-verified against the registry fingerprint)
  * oma.json        <- raw.githubusercontent.com @ the pinned public SHA
                       65f8ea93 (immutable content, sha256 hard-verified)

Non-fatal by design: on any failure the framework's own fetch path can still
retry on first use; install.sh treats a non-zero exit as a warning.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

_SHA = "65f8ea9330459e0106867d1c694aec4139c6cb19"  # pinned public AlphaNet SHA

# filename -> (url, sha256, size_bytes). Fingerprints host-verified 2026-07-16;
# the ckpt fingerprint mirrors models.json AlphaNet-v1-OMA weights_sha256.
FILES: dict[str, tuple[str, str, int]] = {
    "alex_0410.ckpt": (
        "https://ndownloader.figshare.com/files/53851139",
        "879a477c09e2dd2614e029aeb352892ad21d0bdf80eaccbee40f47cf888e48c5",
        19398138,
    ),
    "oma.json": (
        f"https://raw.githubusercontent.com/zmyybc/AlphaNet/{_SHA}/pretrained/OMA/oma.json",
        "79e2de77bcfd4f97ff96bf2f8716c864d3d3e759f1bcea9c6358ec6d441b919b",
        1059,
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stage(name: str, url: str, sha: str, dest_dir: Path) -> int:
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 0 and _sha256(dest) == sha:
        print(f"prestage_alphanet: {dest} already present and verified; nothing to do.")
        return 0
    if dest.exists():
        print(f"prestage_alphanet: {dest} is missing/0-byte/wrong-hash; re-fetching.")
        dest.unlink()

    print(f"prestage_alphanet: downloading {url} -> {dest}")
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(dest_dir), suffix=".part")
    os.close(tmp_fd)
    tmp = Path(tmp_name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "oh-my-mlip/0.1"})
        with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        got = _sha256(tmp)
        if got != sha:
            tmp.unlink(missing_ok=True)
            print(
                f"prestage_alphanet: sha256 mismatch for {name} (got {got}, want {sha}); "
                "leaving cache empty.",
                file=sys.stderr,
            )
            return 1
        tmp.replace(dest)
    except Exception as exc:  # noqa: BLE001 - non-fatal helper
        tmp.unlink(missing_ok=True)
        print(
            f"prestage_alphanet: download of {name} failed ({exc!r}); "
            "the framework fetch path will retry on first use.",
            file=sys.stderr,
        )
        return 1
    print(f"prestage_alphanet: staged {dest} ({dest.stat().st_size} B, sha256 OK).")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target-root",
        default=None,
        help="models/alphanet directory (default: $OH_MY_MLIP_HOME/models/alphanet)",
    )
    args = ap.parse_args(argv)

    if args.target_root:
        dest_dir = Path(args.target_root)
    else:
        home = os.environ.get("OH_MY_MLIP_HOME") or str(
            Path(__file__).resolve().parent.parent
        )
        dest_dir = Path(home) / "models" / "alphanet"
    dest_dir.mkdir(parents=True, exist_ok=True)

    rc = 0
    for name, (url, sha, _size) in FILES.items():
        rc = max(rc, _stage(name, url, sha, dest_dir))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
