#!/usr/bin/env python3
"""Prepare DeePMD DPA-3.1-3M weights for inference.

DPA-3.1-3M.pt on the Hugging Face Hub is a **multi-task training checkpoint**
(a torch.save zip with a `data.pkl`, holding ~31 domain heads). DeePMD's
`DP(model=...)` / `DeepPot` loader needs a **single-head frozen** model
(`constants.pkl` present), so the raw checkpoint cannot be loaded directly:
it raises `Head must be specified in multitask mode` on freeze and
`PytorchStreamReader failed locating file constants.pkl` on direct load.

This helper makes the on-demand weight materialization self-healing:
  1. download the checkpoint to <target_root>/dpa-3.1-3m-ft.pth (verbatim, not unpacked),
  2. `dp --pt freeze` it with a chosen catalysis-appropriate head into the final
     inference target <target_root>/frozen-<head>.pth.

It is invoked by fetch.py's weights_fetch_command path (by-name), which sets
`target_root` and runs this with the env's own interpreter on PATH.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

CHECKPOINT_URL = "https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M/resolve/main/DPA-3.1-3M.pt"
DEFAULT_HEAD = "Omat24"  # OMat24-trained head: general materials, consistent with the OAM roster


def _dp_cmd() -> str:
    # When invoked with the deepmd env interpreter (install.sh weight hook /
    # fetch.py) the env is NOT activated, so `dp` is not on PATH; resolve it next
    # to sys.executable (env bin) and fall back to PATH only if absent.
    sibling = Path(sys.executable).resolve().parent / "dp"
    return str(sibling) if sibling.is_file() else "dp"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "oh-my-mlip/0.1"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"downloaded empty checkpoint from {url}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download + freeze DPA-3.1-3M for DeePMD inference.")
    ap.add_argument("--target-root", required=True, help="models/deepmd directory")
    ap.add_argument("--head", default=DEFAULT_HEAD, help=f"multitask head to freeze (default: {DEFAULT_HEAD})")
    args = ap.parse_args()

    root = Path(args.target_root)
    checkpoint = root / "dpa-3.1-3m-ft.pth"
    frozen = root / f"frozen-{args.head.lower()}.pth"

    if frozen.is_file() and frozen.stat().st_size > 0:
        print(f"[prepare_deepmd] frozen model already present: {frozen}")
        return 0

    if not (checkpoint.is_file() and checkpoint.stat().st_size > 0):
        print(f"[prepare_deepmd] downloading checkpoint -> {checkpoint}")
        _download(CHECKPOINT_URL, checkpoint)

    print(f"[prepare_deepmd] freezing head={args.head} -> {frozen}")
    proc = subprocess.run(
        [_dp_cmd(), "--pt", "freeze", "-c", str(checkpoint), "-o", str(frozen), "--head", args.head],
        cwd=str(root),
    )
    if proc.returncode != 0:
        return proc.returncode
    if not (frozen.is_file() and frozen.stat().st_size > 0):
        print(f"[prepare_deepmd] freeze did not produce {frozen}", file=sys.stderr)
        return 1
    print(f"[prepare_deepmd] ready: {frozen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
