#!/usr/bin/env python3
"""verify_no_token — fail CI if a Hugging Face token literal leaks into the tree.

Scans the **working tree** (tracked AND untracked files, including generated
logs/reports) for anything shaped like a Hugging Face access token
(``hf_`` followed by 20+ alphanumerics). Exits non-zero on the first hit,
printing the file and line with the matched token REDACTED.

Self-trip safety: the detection pattern is assembled from fragments at runtime,
so this file's own source contains NO matching literal. Running the verifier
against the repo (including this file) passes on a clean tree.

Usage:
    python scripts/verify_no_token.py [ROOT]   # ROOT defaults to repo root
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Build the regex from fragments so this source never contains a literal that
# would match itself. (Do NOT inline the full pattern as one string.)
PAT = re.compile("hf" + "_" + "[A-Za-z0-9]" + "{20,}")

# Directories we never scan: VCS internals, caches, packaged envs, the env
# tarball/cache dirs. These can contain large binaries or third-party content.
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "envs",
    ".venv",
    "venv",
    "node_modules",
    ".omc",
}

# Binary / archive extensions we skip outright (token literals live in text).
SKIP_SUFFIXES = {
    ".tar", ".gz", ".tgz", ".zip", ".pt", ".pt2", ".pth", ".ckpt",
    ".model", ".so", ".bin", ".pyc", ".png", ".jpg", ".jpeg", ".gif",
    ".pdf", ".npy", ".npz", ".h5", ".pkl",
}


def _is_binary(path: Path) -> bool:
    """Heuristic: a NUL byte in the first 4 KiB means binary."""
    try:
        with open(path, "rb") as fh:
            return b"\x00" in fh.read(4096)
    except OSError:
        return True


def _redact(match: str) -> str:
    """Never echo a real token: keep the prefix, hide the secret part."""
    return "hf_<redacted>"


def scan(root: Path) -> list[tuple[Path, int]]:
    """Return a list of (path, lineno) for every working-tree token hit."""
    hits: list[tuple[Path, int]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if _is_binary(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, start=1):
                    if PAT.search(line):
                        hits.append((path, lineno))
        except OSError:
            continue
    return hits


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]).resolve() if args else Path(__file__).resolve().parent.parent
    hits = scan(root)
    if hits:
        print(
            f"verify_no_token: FAIL — {len(hits)} token-shaped literal(s) found "
            f"(value redacted):",
            file=sys.stderr,
        )
        for path, lineno in hits:
            rel = path.relative_to(root) if path.is_relative_to(root) else path
            print(f"  {rel}:{lineno}: {_redact('')}", file=sys.stderr)
        print(
            "  Remove the token; tokens are the user's and must never be "
            "committed. See docs/hf_token.md.",
            file=sys.stderr,
        )
        return 1
    print(f"verify_no_token: OK — no token literals under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
