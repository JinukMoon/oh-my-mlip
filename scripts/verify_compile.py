#!/usr/bin/env python3
"""verify_compile.py — GPU-free shape check of every `accel` block in models.json.

The accel blocks (compile/GPU-acceleration recipes for NequIP, Allegro, SevenNet,
and the EquFlash v1 flashtp backend) are curated from upstream/owner docs and are
NOT GPU-verified in this repo. This verifier proves that honesty contract holds at
the data level WITHOUT a GPU: it asserts every accel block

  * carries the required keys (tool, install, compile_cmd, load_note, verify,
    gpu_required, verified, provenance, last_gpu_verified),
  * is marked verified=false (no block silently claims a passed GPU run),
  * has last_gpu_verified=null (nothing claims a verification date),
  * declares a recognized provenance (upstream-doc | owner-doc | prior-local-run).

It prints a per-block table and the banner "shape-only; gpu_unverified". It never
imports torch/ase/conda and never executes any compile command -> safe in CI.

Accel blocks live in two places:
  * framework level:  models[FW]["accel"] and an optional models[FW]["accel_lammps"]
  * version level:    models[FW]["versions"][V]["accel"]

Usage:
    python scripts/verify_compile.py [MODELS_JSON]   # defaults to repo models.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_JSON = _REPO_ROOT / "models.json"

REQUIRED_KEYS = (
    "tool",
    "install",
    "compile_cmd",
    "load_note",
    "verify",
    "gpu_required",
    "verified",
    "provenance",
    "last_gpu_verified",
)
ALLOWED_PROVENANCE = {"upstream-doc", "owner-doc", "prior-local-run"}
ALLOWED_TOOLS = {"openequivariance", "cueq", "flashtp", "lammps-mliap"}


def collect_accel_blocks(models: dict) -> list[tuple[str, dict]]:
    """Return (location-label, accel-block) pairs for every accel block found,
    at both framework and version level (including 'accel_lammps')."""
    blocks: list[tuple[str, dict]] = []
    for framework, info in models.items():
        if framework.startswith("_") or not isinstance(info, dict):
            continue
        for key in ("accel", "accel_lammps"):
            blk = info.get(key)
            if isinstance(blk, dict):
                blocks.append((f"{framework}.{key}", blk))
        for version, vinfo in info.get("versions", {}).items():
            if not isinstance(vinfo, dict):
                continue
            blk = vinfo.get("accel")
            if isinstance(blk, dict):
                blocks.append((f"{framework}.versions.{version}.accel", blk))
    return blocks


def check_block(label: str, blk: dict) -> list[str]:
    """Return a list of contract violations for one accel block (empty == OK)."""
    errs: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in blk:
            errs.append(f"{label}: missing required key '{key}'")
    # If keys are missing we still check the ones that are present.
    if blk.get("verified") is not False:
        errs.append(
            f"{label}: verified must be false (no GPU run is claimed here); "
            f"got {blk.get('verified')!r}"
        )
    if blk.get("last_gpu_verified") is not None:
        errs.append(
            f"{label}: last_gpu_verified must be null (no verification date "
            f"claimed); got {blk.get('last_gpu_verified')!r}"
        )
    prov = blk.get("provenance")
    if prov not in ALLOWED_PROVENANCE:
        errs.append(
            f"{label}: provenance must be one of {sorted(ALLOWED_PROVENANCE)}; "
            f"got {prov!r}"
        )
    if "gpu_required" in blk and not isinstance(blk["gpu_required"], bool):
        errs.append(f"{label}: gpu_required must be a bool; got {blk['gpu_required']!r}")
    tool = blk.get("tool")
    if tool not in ALLOWED_TOOLS:
        errs.append(
            f"{label}: tool must be one of {sorted(ALLOWED_TOOLS)}; got {tool!r}"
        )
    return errs


def verify(models: dict) -> tuple[list[tuple[str, dict]], list[str]]:
    """Return (blocks, errors). errors is empty iff every block honors the
    GPU-unverified contract."""
    blocks = collect_accel_blocks(models)
    errors: list[str] = []
    for label, blk in blocks:
        errors.extend(check_block(label, blk))
    return blocks, errors


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]).resolve() if args else MODELS_JSON
    models = json.loads(path.read_text(encoding="utf-8"))

    blocks, errors = verify(models)

    # Per-block table.
    print(f"verify_compile: {len(blocks)} accel block(s) in {path.name}")
    print(f"{'block':<48} {'tool':<16} {'provenance':<16} verified")
    print(f"{'-' * 48} {'-' * 16} {'-' * 16} --------")
    for label, blk in blocks:
        print(
            f"{label:<48} {str(blk.get('tool')):<16} "
            f"{str(blk.get('provenance')):<16} {str(blk.get('verified'))}"
        )
    print()

    if errors:
        print(
            f"verify_compile: FAIL — {len(errors)} accel-contract violation(s):",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    # Honesty banner: this check is data-shape only; nothing was run on a GPU.
    print("shape-only; gpu_unverified")
    print(
        f"verify_compile: OK — all {len(blocks)} accel block(s) honor the "
        "GPU-unverified contract (verified=false, last_gpu_verified=null)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
