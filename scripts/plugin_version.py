#!/usr/bin/env python3
"""plugin_version.py — keep the Claude Code plugin version in step with what it carries.

Claude Code keeps an installed plugin on the `version` in
`.claude-plugin/plugin.json` until that string changes, so a skill or
AGENTS.md edit reaches existing users only when the version moves. The
version is `skills-YYYY.MM.DD[.N]`: it tracks the plugin's instructions, not a
package release (`oh_my_mlip.__version__` and git tags are separate).
`metadata.content_sha256` records the sha256 of `skills/**` and `AGENTS.md`;
`tests/test_plugin_manifest.py` fails when that content changes and the
version did not.

Usage:
  python3 scripts/plugin_version.py --check   # exit 1 when a bump is due
  python3 scripts/plugin_version.py --bump    # set today's version and the new hash
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / ".claude-plugin" / "plugin.json"
PREFIX = "skills-"


def content_sha256(repo: Path = REPO) -> str:
    """sha256 over the relative path and bytes of every file under skills/ plus AGENTS.md."""
    h = hashlib.sha256()
    files = sorted(p for p in (repo / "skills").rglob("*") if p.is_file()) + [repo / "AGENTS.md"]
    for path in files:
        h.update(path.relative_to(repo).as_posix().encode() + b"\0")
        h.update(path.read_bytes() + b"\0")
    return h.hexdigest()


def next_version(current: str | None, today: _dt.date) -> str:
    base = f"{PREFIX}{today:%Y.%m.%d}"
    if not current or not current.startswith(base):
        return base
    tail = current[len(base):]
    n = int(tail[1:]) + 1 if tail.startswith(".") and tail[1:].isdigit() else 2
    return f"{base}.{n}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--bump", action="store_true")
    args = ap.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    recorded = (manifest.get("metadata") or {}).get("content_sha256")
    current = content_sha256()
    if args.check:
        if recorded == current:
            print(f"plugin version {manifest.get('version')}: up to date")
            return 0
        print(f"skills/ or AGENTS.md changed since plugin version {manifest.get('version')}; "
              f"run: python3 scripts/plugin_version.py --bump", file=sys.stderr)
        return 1
    if recorded == current:
        print(f"plugin version {manifest.get('version')}: nothing changed, not bumped")
        return 0
    manifest["version"] = next_version(manifest.get("version"), _dt.date.today())
    manifest.setdefault("metadata", {})["content_sha256"] = current
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"plugin version -> {manifest['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
