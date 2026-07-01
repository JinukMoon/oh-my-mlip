#!/usr/bin/env python3
"""Prepare GRACE foundation weights for inference.

`grace_models download <name>` fetches a TensorFlow SavedModel, but it lands in a
**nested** layout: <target_root>/<name>/<name>/saved_model.pb (the CLI re-creates
the model-name directory under whatever cache/cwd it runs in). The registry
inference points TPCalculator at <target_root>/<name>, so the SavedModel root must
be flattened to that path (where saved_model.pb sits directly under it).

This helper makes the GRACE on-demand weight materialization self-healing and
layout-robust: it runs the CLI, then locates the directory that actually contains
saved_model.pb and moves its contents up to the inference target directory.

Invoked by fetch.py's weights_fetch_command path (by-name); fetch.py sets
GRACE_CACHE=<target_root> (weights_cache_env) and passes the model name.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _find_saved_model_root(base: Path) -> Path | None:
    for pb in base.rglob("saved_model.pb"):
        return pb.parent
    return None

def _grace_models_cmd() -> str:
    # When fetch.py invokes this with the grace env interpreter, grace_models
    # lives next to that interpreter; prefer it over PATH so the helper works
    # regardless of the caller's activated env.
    sibling = Path(sys.executable).resolve().parent / "grace_models"
    return str(sibling) if sibling.is_file() else "grace_models"


def main() -> int:
    ap = argparse.ArgumentParser(description="Download + flatten a GRACE SavedModel.")
    ap.add_argument("--name", default="GRACE-2L-OAM", help="grace_models model name (default: GRACE-2L-OAM)")
    ap.add_argument("--target-dir", default=None, help="inference target dir (models/grace/<name>); default: <target-root>/<name>")
    ap.add_argument("--target-root", default=None, help="models/grace directory; when --target-dir is omitted it is derived as <target-root>/<name> (uniform install.sh weight-hook interface)")
    args = ap.parse_args()

    if args.target_dir:
        target = Path(args.target_dir)
    elif args.target_root:
        target = Path(args.target_root) / args.name
    else:
        ap.error("provide --target-dir or --target-root")
    if (target / "saved_model.pb").is_file():
        print(f"[prepare_grace] already flattened: {target}/saved_model.pb")
        return 0

    target.mkdir(parents=True, exist_ok=True)
    print(f"[prepare_grace] grace_models download {args.name}")
    proc = subprocess.run([_grace_models_cmd(), "download", args.name], cwd=str(target))
    if proc.returncode != 0:
        return proc.returncode

    if (target / "saved_model.pb").is_file():
        print(f"[prepare_grace] SavedModel already at target root: {target}")
        return 0

    root = _find_saved_model_root(target)
    if root is None:
        print(f"[prepare_grace] saved_model.pb not found anywhere under {target}", file=sys.stderr)
        return 1
    if root == target:
        return 0

    print(f"[prepare_grace] flattening {root} -> {target}")
    # move every entry from the nested SavedModel root up to target
    for entry in root.iterdir():
        dest = target / entry.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(entry), str(dest))
    # prune now-empty nested dirs between target and the old root
    try:
        nested_top = root
        while nested_top != target and nested_top.parent != target:
            nested_top = nested_top.parent
        if nested_top != target and nested_top.exists() and not any(nested_top.iterdir()):
            shutil.rmtree(nested_top)
    except OSError:
        pass

    if not (target / "saved_model.pb").is_file():
        print(f"[prepare_grace] flatten failed: no saved_model.pb at {target}", file=sys.stderr)
        return 1
    print(f"[prepare_grace] ready: {target}/saved_model.pb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
