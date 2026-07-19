#!/usr/bin/env python3
"""adopt_env.py -- adopt an EXISTING interpreter env for a registry env.

Bring-your-own-env: if you already have a working conda env for a framework
(say MACE in ~/miniconda3/envs/MACE), there is no reason to rebuild it under
the hub prefix. This script VERIFIES the env (runs the registry's import
lines inside its interpreter) and only then records it in
``env_map.local.json`` at the hub root — from that point the resolver
(oh_my_mlip.registry.resolve) dispatches to the adopted interpreter, and the
plugin / run() / Worker / setup_verify all use it from any folder.

Verification is the gate: an env that cannot import the framework is refused,
so an adopted entry is always a proven one (same trust rule as install.sh's
sentinel-plus-import-verify).

Usage:
  python3 scripts/adopt_env.py MACE /home/user/miniconda3/envs/MACE
  python3 scripts/adopt_env.py --list
  python3 scripts/adopt_env.py --remove mace
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup_common import load_local_env_map as load_map  # noqa: E402
from _setup_common import resolve_home  # noqa: E402

MAP_NAME = "env_map.local.json"


def save_map(home: Path, data: dict) -> None:
    (home / MAP_NAME).write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def find_family(model: str, registry: dict) -> tuple[str, dict] | None:
    want = model.lower()
    for family, spec in registry.items():
        if family.startswith("_"):
            continue
        if family.lower() == want or spec.get("env", "").lower() == want:
            return family, spec
        for version in (spec.get("versions") or {}):
            if version.lower() == want:
                return family, spec
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", nargs="?", help="family, version, or env name from models.json")
    ap.add_argument("prefix", nargs="?", help="existing env prefix (contains bin/python)")
    ap.add_argument("--list", action="store_true", help="show current adoptions")
    ap.add_argument("--remove", metavar="ENV", help="remove an adoption entry")
    args = ap.parse_args()

    home = resolve_home()
    if args.list:
        data = load_map(home)
        if not data:
            print("no adopted envs (env_map.local.json absent or empty)")
        for env, prefix in sorted(data.items()):
            print(f"{env:<16} -> {prefix}")
        return 0
    if args.remove:
        data = load_map(home)
        if args.remove not in data:
            print(f"not adopted: {args.remove}", file=sys.stderr)
            return 1
        del data[args.remove]
        save_map(home, data)
        print(f"removed adoption: {args.remove}")
        return 0
    if not args.model or not args.prefix:
        ap.error("need MODEL and PREFIX (or --list / --remove)")

    registry = json.loads((home / "models.json").read_text(encoding="utf-8"))
    found = find_family(args.model, registry)
    if found is None:
        print(f"unknown model/env: {args.model}", file=sys.stderr)
        return 1
    family, spec = found
    env_name = spec["env"]

    prefix = Path(args.prefix).expanduser().resolve()
    python = prefix / "bin" / "python"
    if not python.exists():
        print(f"no interpreter at {python}", file=sys.stderr)
        return 1

    imports = "; ".join(spec.get("import") or [])
    proc = subprocess.run(
        [str(python), "-c", imports],
        capture_output=True, text=True, timeout=600,
        env={"PATH": f"{prefix / 'bin'}:/usr/bin:/bin", "HOME": str(Path.home())},
    )
    if proc.returncode != 0:
        tail = (proc.stderr.strip().splitlines() or ["<no stderr>"])[-1]
        print(f"REFUSED: registry imports for {family} fail in {prefix}:\n  {tail}",
              file=sys.stderr)
        return 1

    data = load_map(home)
    data[env_name] = str(prefix)
    save_map(home, data)
    print(f"adopted: {env_name} -> {prefix}  (imports verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
