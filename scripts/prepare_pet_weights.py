#!/usr/bin/env python3
"""Prepare PET-OAM-XL weights for inference.

The public HF repo (lab-cosmo/upet) publishes only a **training checkpoint**
`pet-oam-xl-v1.0.0.ckpt`, not an exported metatomic model. `MetatomicCalculator`
needs an **exported** `.pt` (a TorchScript metatomic AtomisticModel); pointing it
at the raw `.ckpt` fails ("not a metatomic model").

This helper makes the on-demand weight materialization self-healing:
  1. download the checkpoint to <target_root>/pet-oam-xl-v1.0.0.ckpt (verbatim),
  2. `mtt export` it into the final inference target <target_root>/pet-oam-xl-v1.0.0.pt.

It is invoked by fetch.py's weights_fetch_command path (by-name), which sets
`target_root` and runs this with the env's own interpreter. `mtt` is resolved
next to that interpreter so it works regardless of the caller's activated env.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

CHECKPOINT_URL = (
    "https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-oam-xl-v1.0.0.ckpt"
)
CKPT_NAME = "pet-oam-xl-v1.0.0.ckpt"
PT_NAME = "pet-oam-xl-v1.0.0.pt"


def _mtt_cmd() -> str:
    sibling = Path(sys.executable).resolve().parent / "mtt"
    return str(sibling) if sibling.is_file() else "mtt"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "oh-my-mlip/0.1"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"downloaded empty checkpoint from {url}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download + export PET-OAM-XL for metatomic inference.")
    ap.add_argument("--target-root", required=True, help="models/pet directory")
    args = ap.parse_args()

    root = Path(args.target_root)
    checkpoint = root / CKPT_NAME
    exported = root / PT_NAME

    if exported.is_file() and exported.stat().st_size > 0:
        print(f"[prepare_pet] exported model already present: {exported}")
        return 0

    if not (checkpoint.is_file() and checkpoint.stat().st_size > 0):
        print(f"[prepare_pet] downloading checkpoint -> {checkpoint}")
        _download(CHECKPOINT_URL, checkpoint)

    print(f"[prepare_pet] exporting {checkpoint} -> {exported}")
    proc = subprocess.run(
        [_mtt_cmd(), "export", str(checkpoint), "-o", str(exported)],
        cwd=str(root),
    )
    if proc.returncode != 0:
        return proc.returncode
    if not (exported.is_file() and exported.stat().st_size > 0):
        print(f"[prepare_pet] export did not produce {exported}", file=sys.stderr)
        return 1
    print(f"[prepare_pet] ready: {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
