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
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

# Upstream's own Hugging Face copy. The SavedModel inside GRACE-2L-OAM-model.tar.gz
# is byte-identical, file by file, to the one grace_models fetches from the ICAMS
# share (checked 2026-09-21 against a GPU-verified extraction); only the tarball
# packaging differs. That share has served a few KiB/s with no resume, which put
# a first install hours away; this copy serves the same model in minutes.
HF_URL = ("https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models/"
          "resolve/main/models/{name}-model.tar.gz")


def _find_saved_model_root(base: Path) -> Path | None:
    for pb in base.rglob("saved_model.pb"):
        return pb.parent
    return None

def _cache_roots() -> list[Path]:
    """Places grace_models may have written the SavedModel instead of cwd.

    Observed with grace_models 0.5.3: the CLI
    ignores the working directory and downloads into ~/.cache/grace/<name>/...,
    so the in-target flatten search found nothing and the weight had to be
    copied manually. $GRACE_CACHE (when set) takes precedence."""
    roots: list[Path] = []
    env_cache = os.environ.get("GRACE_CACHE")
    if env_cache:
        roots.append(Path(env_cache).expanduser())
    roots.append(Path.home() / ".cache" / "grace")
    return roots


def _grace_models_cmd() -> str:
    # When fetch.py invokes this with the grace env interpreter, grace_models
    # lives next to that interpreter; prefer it over PATH so the helper works
    # regardless of the caller's activated env.
    sibling = Path(sys.executable).resolve().parent / "grace_models"
    return str(sibling) if sibling.is_file() else "grace_models"


def _registry_expectation(name: str) -> tuple[int, str] | None:
    """(size, sha256) the registry records for the tarball of weights `name`."""
    home = os.environ.get("OH_MY_MLIP_HOME") or str(Path(__file__).resolve().parent.parent)
    try:
        models = json.loads((Path(home) / "models.json").read_text())
    except (OSError, ValueError):
        return None
    for family, info in models.items():
        if family.startswith("_") or not isinstance(info, dict):
            continue
        for spec in (info.get("versions") or {}).values():
            if spec.get("weights_source") == name and spec.get("weights_size") and spec.get("weights_sha256"):
                return int(spec["weights_size"]), str(spec["weights_sha256"]).lower()
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_resumable(url: str, dest: Path, total: int) -> None:
    """Continue a partial download instead of starting over, and say how far it got."""
    have = dest.stat().st_size if dest.exists() else 0
    if have > total:
        dest.unlink()
        have = 0
    if have == total:
        return
    headers = {"User-Agent": "oh-my-mlip/0.1"}
    if have:
        headers["Range"] = f"bytes={have}-"
    with urlopen(Request(url, headers=headers), timeout=60) as resp:
        if have and getattr(resp, "status", None) != 206:   # the server ignored Range
            have = 0
        done, last = have, time.time()
        with open(dest, "ab" if have else "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if time.time() - last > 5:
                    print(f"[prepare_grace] {done / 1e6:.1f} / {total / 1e6:.1f} MB", flush=True)
                    last = time.time()


def _fetch_from_hf(name: str, target: Path) -> bool:
    """Download, verify and flatten `name` from upstream's Hugging Face copy.

    Staged beside the target, never inside it: an interrupted download must
    neither look like weights nor be lost -- the partial file stays in the
    staging dir and the next run resumes it. False when the registry records no
    size/sha for `name` (nothing to verify against)."""
    expected = _registry_expectation(name)
    if expected is None:
        return False
    size, sha = expected
    staging = target.parent / f".{target.name}.hf"
    staging.mkdir(parents=True, exist_ok=True)
    tarball = staging / f"{name}-model.tar.gz"
    print(f"[prepare_grace] downloading {name} from Hugging Face ({size / 1e6:.1f} MB)")
    _download_resumable(HF_URL.format(name=name), tarball, size)
    got_size, got_sha = tarball.stat().st_size, _sha256(tarball)
    if got_size != size or got_sha != sha:
        tarball.unlink()                                   # corrupt: never resume from it
        raise RuntimeError(f"{tarball.name}: size {got_size}, sha256 {got_sha}; "
                           f"the registry records {size}, {sha}")
    unpacked = staging / "unpacked"
    shutil.rmtree(unpacked, ignore_errors=True)
    with tarfile.open(tarball, "r:gz") as tf:
        try:
            tf.extractall(unpacked, filter="data")
        except TypeError:  # pragma: no cover - python without extraction filters
            tf.extractall(unpacked)
    root = _find_saved_model_root(unpacked)
    if root is None:
        raise RuntimeError(f"{tarball.name} holds no saved_model.pb")
    target.mkdir(parents=True, exist_ok=True)
    for entry in root.iterdir():
        shutil.move(str(entry), str(target / entry.name))
    shutil.rmtree(staging, ignore_errors=True)
    return (target / "saved_model.pb").is_file()


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

    try:
        if _fetch_from_hf(args.name, target):
            print(f"[prepare_grace] ready: {target}/saved_model.pb")
            return 0
    except Exception as exc:  # network, verification, archive: fall back, and say why
        print(f"[prepare_grace] Hugging Face download failed ({exc}); "
              f"falling back to grace_models (the ICAMS share, which can be very slow)", file=sys.stderr)
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
        # grace_models >=0.5.3 ignores cwd; look in its cache dirs before
        # giving up, preferring the <cache>/<name> subtree.
        for base in _cache_roots():
            if not base.is_dir():
                continue
            root = _find_saved_model_root(base / args.name) or _find_saved_model_root(base)
            if root is not None:
                print(f"[prepare_grace] found SavedModel in cache: {root}")
                break
    if root is None:
        print(
            f"[prepare_grace] saved_model.pb not found under {target} or "
            f"{[str(r) for r in _cache_roots()]}",
            file=sys.stderr,
        )
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
    # prune now-empty nested dirs between target and the old root (only when
    # the old root actually lived under target; a cache-dir root is left as-is)
    try:
        if target in root.parents:
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
