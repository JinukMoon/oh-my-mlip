#!/usr/bin/env python3
"""verify_weights_integrity.py — GPU-free check: downloaded weight == validated.

Each model version in models.json may carry a `weights_sha256` (+ `weights_size`):
the fingerprint of the EXACT checkpoint oh-my-mlip was validated against. This
verifier sha256s a downloaded weight file and compares it to that recorded
fingerprint, so a user can confirm the file they fetched is byte-identical to the
one we validated (not a silently-diverged official re-upload).

A weight artifact may be a directory rather than a file (GRACE ships a
TensorFlow SavedModel tree). Those are fingerprinted with `sha256_tree` — a
digest over every contained file's relpath/size/sha256 — and recorded in the
same `weights_sha256` field, so callers need no special case.

Per-model status:
  * matches-validated   — recorded fingerprint present AND the supplied file's
                          sha256 (and size) match it.
  * MISMATCH            — recorded fingerprint present but the supplied file
                          differs (official source diverged, or wrong file).
  * fingerprint-pending — no weights_sha256 recorded for this model yet
                          (bundled / HF-cache / gated — to be extracted later).
  * file-not-found      — a path was supplied but does not exist.
  * (no file supplied)  — recorded fingerprint shown; nothing to compare.

GPU-free: stdlib only (json, hashlib). Never imports torch/ase/conda.

Usage:
    # whole-registry table (recorded fingerprints, nothing hashed):
    python scripts/verify_weights_integrity.py

    # check one downloaded file against a model's recorded fingerprint:
    python scripts/verify_weights_integrity.py --model MACE-MPA-0 --file /path/to/weights
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_JSON = _REPO_ROOT / "models.json"

MATCH = "matches-validated"
MISMATCH = "MISMATCH"
PENDING = "fingerprint-pending"
NOT_FOUND = "file-not-found"
RECORDED = "recorded"  # fingerprint present, no file supplied to compare


def index_versions(models: dict) -> dict[str, dict]:
    """Map every version key -> its version dict (registry order preserved)."""
    out: dict[str, dict] = {}
    for framework, info in models.items():
        if framework.startswith("_") or not isinstance(info, dict):
            continue
        for version, vinfo in info.get("versions", {}).items():
            if isinstance(vinfo, dict):
                out[version] = vinfo
    return out


def sha256_file(path: Path) -> tuple[str, int]:
    """Return (hexdigest, size_bytes) for a file, streamed in chunks."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def sha256_tree(root: Path) -> tuple[str, int]:
    """Return (hexdigest, total_size) for a DIRECTORY weight artifact.

    Some frameworks ship a directory, not a file — GRACE loads a TensorFlow
    SavedModel tree (``saved_model.pb`` + ``variables/`` + ``assets/``), so a
    plain file sha256 cannot fingerprint it and the model was left unprotected.
    The digest is over ``<relpath>\\0<size>\\0<file sha256>\\n`` for every file,
    sorted by relative path: stable across hosts, sensitive to any added,
    removed, renamed or edited file. Symlinked roots are resolved first (the
    hub keeps ``models/grace/<name>`` pointing at the framework cache).
    """
    root = root.resolve()
    h = hashlib.sha256()
    total = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest, size = sha256_file(path)
        h.update(f"{path.relative_to(root)}\0{size}\0{digest}\n".encode())
        total += size
    return h.hexdigest(), total


def classify(vinfo: dict, file_path: Path | None) -> tuple[str, str]:
    """Return (status, detail) for one model version against an optional path."""
    recorded = vinfo.get("weights_sha256")
    if not recorded:
        return PENDING, "no weights_sha256 recorded"
    if file_path is None:
        size = vinfo.get("weights_size")
        return RECORDED, f"validated sha256={recorded[:12]}… size={size}"
    if not file_path.exists():
        return NOT_FOUND, f"path does not exist: {file_path}"
    digest, size = (
        sha256_tree(file_path) if file_path.is_dir() else sha256_file(file_path)
    )
    if digest == recorded:
        return MATCH, f"sha256={digest[:12]}… size={size}"
    return MISMATCH, (
        f"file sha256={digest[:12]}… (size={size}) != "
        f"validated {recorded[:12]}… (size={vinfo.get('weights_size')})"
    )


def run(
    models: dict,
    model: str | None = None,
    file_path: Path | None = None,
) -> tuple[list[tuple[str, str, str]], int]:
    """Return (rows, exit_code). rows are (model, status, detail).

    exit_code is non-zero only on an explicit MISMATCH or file-not-found for a
    requested model. A pending fingerprint is NOT a failure.
    """
    versions = index_versions(models)
    rows: list[tuple[str, str, str]] = []
    rc = 0

    if model is not None:
        vinfo = versions.get(model)
        if vinfo is None:
            print(f"verify_weights_integrity: unknown model {model!r}", file=sys.stderr)
            return [], 2
        status, detail = classify(vinfo, file_path)
        rows.append((model, status, detail))
        if status in (MISMATCH, NOT_FOUND):
            rc = 1
        return rows, rc

    # Whole-registry table (recorded fingerprints; no hashing).
    for name, vinfo in versions.items():
        status, detail = classify(vinfo, None)
        rows.append((name, status, detail))
    return rows, rc


def print_table(rows: list[tuple[str, str, str]]) -> None:
    print(f"{'model':<24} {'status':<20} detail")
    print(f"{'-' * 24} {'-' * 20} {'-' * 40}")
    for name, status, detail in rows:
        print(f"{name:<24} {status:<20} {detail}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="version key to check (e.g. MACE-MPA-0)")
    ap.add_argument("--file", help="downloaded weight file to fingerprint")
    ap.add_argument("models_json", nargs="?", help="path to models.json (optional)")
    args = ap.parse_args(argv)

    if args.file and not args.model:
        ap.error("--file requires --model (which fingerprint to compare against)")

    path = Path(args.models_json).resolve() if args.models_json else MODELS_JSON
    models = json.loads(path.read_text(encoding="utf-8"))

    file_path = Path(args.file).resolve() if args.file else None
    rows, rc = run(models, model=args.model, file_path=file_path)
    if rows:
        print_table(rows)
    n_recorded = sum(1 for _, s, _ in rows if s in (RECORDED, MATCH, MISMATCH))
    n_pending = sum(1 for _, s, _ in rows if s == PENDING)
    print()
    print(
        f"verify_weights_integrity: {len(rows)} model(s); "
        f"{n_recorded} with a validated fingerprint, {n_pending} fingerprint-pending."
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
