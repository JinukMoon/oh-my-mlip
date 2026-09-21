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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _weight_download import download, is_complete  # noqa: E402

CHECKPOINT_URL = (
    "https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-oam-xl-v1.0.0.ckpt"
)
CKPT_NAME = "pet-oam-xl-v1.0.0.ckpt"
PT_NAME = "pet-oam-xl-v1.0.0.pt"
# The checkpoint as Hugging Face stores it (LFS size and sha256).
CKPT_SIZE = 2920687712
CKPT_SHA256 = "c3a67cd019969dfd4dcabe9574682fe035f861d3f1c10190989b36c983699409"


def _mtt_cmd() -> str:
    sibling = Path(sys.executable).resolve().parent / "mtt"
    return str(sibling) if sibling.is_file() else "mtt"


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

    if not is_complete(checkpoint, CKPT_SIZE):
        print(f"[prepare_pet] downloading checkpoint -> {checkpoint}")
        download(CHECKPOINT_URL, checkpoint, size=CKPT_SIZE, sha256=CKPT_SHA256, label="prepare_pet")

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
