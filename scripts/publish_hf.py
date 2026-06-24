#!/usr/bin/env python3
"""publish_hf.py — AUTHOR-SIDE: upload one env's conda-pack tarball to its
per-env Hugging Face Hub repo, pin the revision, and append/update the matching
entry in dist_manifest.json.

Pairs with scripts/build_conda_pack.sh: that script strips arch artifacts and
emits ``<env>.tar.gz`` plus sidecars (``.sha256``, ``.unpack_size``,
``.min_driver``); this script publishes the tarball and records its provenance.

One HF repo per env. The pushed revision (a tag or commit) is what
dist_manifest.json pins, so the resolver fetches a reproducible, integrity-checked
artifact rather than a moving ``main``.

Requires:
  * ``HF_TOKEN`` exported with write access to the target repo (gated-model
    weights are NEVER bundled here — only the relocatable env tarball).
  * ``huggingface_hub`` installed (guarded import below).

Usage:
  HF_TOKEN=hf_... python scripts/publish_hf.py \\
      --env mace \\
      --tarball dist/mace.tar.gz \\
      --hf-repo <org>/oh-my-mlip-env-mace \\
      --revision v1 \\
      [--manifest dist_manifest.json]
"""
import argparse
import json
import os
import sys
from pathlib import Path


def _read_sidecar(tarball: Path, suffix: str, default: str = "TODO-on-upload") -> str:
    """Read a build_conda_pack sidecar (e.g. <env>.tar.gz.sha256) if present."""
    cand = tarball.with_name(tarball.name + suffix)
    if cand.exists():
        return cand.read_text().strip()
    # Also try the <env>.<suffix> form (unpack_size / min_driver sidecars).
    stem = tarball.name
    for ext in (".tar.gz", ".tgz"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    alt = tarball.with_name(stem + suffix)
    if alt.exists():
        return alt.read_text().strip()
    return default


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", required=True, help="env name (manifest key, e.g. 'mace')")
    ap.add_argument("--tarball", required=True, type=Path, help="path to <env>.tar.gz")
    ap.add_argument("--hf-repo", required=True, help="target HF repo id (one per env)")
    ap.add_argument("--revision", required=True, help="revision tag to create/pin (never 'main')")
    ap.add_argument("--manifest", default="dist_manifest.json", type=Path)
    ap.add_argument("--path-in-repo", default=None, help="filename in the repo (default: tarball basename)")
    args = ap.parse_args()

    if args.revision == "main":
        print("publish_hf: refuse to pin 'main'; pass an immutable tag/commit.", file=sys.stderr)
        return 2
    if not args.tarball.exists():
        print(f"publish_hf: tarball not found: {args.tarball}", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("publish_hf: HF_TOKEN not set (need write access to the repo).", file=sys.stderr)
        return 1

    # Guarded heavy import: keep the module importable without huggingface_hub.
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("publish_hf: huggingface_hub not installed (pip install huggingface_hub).", file=sys.stderr)
        return 1

    sha256 = _read_sidecar(args.tarball, ".sha256")
    unpack_size = _read_sidecar(args.tarball, ".unpack_size")
    min_driver = _read_sidecar(args.tarball, ".min_driver")
    path_in_repo = args.path_in_repo or args.tarball.name

    api = HfApi(token=token)
    print(f"publish_hf: ensuring repo {args.hf_repo} exists ...")
    api.create_repo(repo_id=args.hf_repo, repo_type="model", exist_ok=True)

    print(f"publish_hf: uploading {args.tarball} -> {args.hf_repo}:{path_in_repo} ...")
    api.upload_file(
        path_or_fileobj=str(args.tarball),
        path_in_repo=path_in_repo,
        repo_id=args.hf_repo,
        repo_type="model",
    )

    print(f"publish_hf: tagging revision {args.revision!r} ...")
    api.create_tag(repo_id=args.hf_repo, tag=args.revision, repo_type="model", exist_ok=True)

    # ── Append/update the manifest entry ──
    manifest = {}
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())

    try:
        unpack_size_val: object = int(unpack_size)
    except (TypeError, ValueError):
        unpack_size_val = unpack_size  # leave the literal marker if not numeric

    manifest[args.env] = {
        "env": args.env,
        "hf_repo": args.hf_repo,
        "revision": args.revision,
        "sha256": sha256,
        "unpack_size_bytes": unpack_size_val,
        "min_driver_version": min_driver,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"publish_hf: updated {args.manifest} entry for '{args.env}'.")
    print("publish_hf: done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
