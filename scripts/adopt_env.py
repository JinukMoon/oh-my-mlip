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

After the imports pass it runs the same post-install steps as install.sh: weight
preparation, then the D3 kernel build (scripts/build_d3.py); a failed D3 build
keeps the adoption and exits 1 with the log path.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup_common import load_local_env_map as load_map  # noqa: E402
from _setup_common import refuse_plugin_copy, resolve_home  # noqa: E402

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


def known_names(registry: dict) -> str:
    return ", ".join(sorted(f for f in registry if not f.startswith("_")))


def prepare_weights(home: Path, env_name: str, python: Path) -> None:
    """Run the weight preparation install.sh runs after it builds an env.

    Adoption skipped this entirely, so a model whose inference names a PATH
    (GRACE's SavedModel, DeePMD's frozen .pth) resolved to a file nobody had
    created: pasting the resolve() lines then failed with a missing file, while
    the same model worked from an install.sh-built env. Mirrors install.sh:
    prestage_* is pure stdlib and runs on any python, prepare_* needs the env's
    own CLI. Failure leaves the adoption standing -- the weights can be fetched
    later -- but it is reported, never swallowed."""
    child_env = dict(os.environ, OH_MY_MLIP_HOME=str(home))
    for argv in weight_steps(home, env_name, python):
        script = Path(argv[1])
        print(f"preparing weights: {script.name} ...")
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=3600, env=child_env)
        if proc.returncode != 0:
            tail = (proc.stderr.strip().splitlines() or ["<no stderr>"])[-1]
            print(f"  {script.name} FAILED: {tail}\n"
                  f"  the adoption stands, but this model's weights are not ready yet; "
                  f"rerun this step with:\n    {shlex.join(argv)}",
                  file=sys.stderr)


def weight_steps(home: Path, env_name: str, python: Path) -> list[list[str]]:
    """The weight-preparation commands install.sh runs after building `env_name`,
    in the same form: prestage_* on any python with no arguments, prepare_* on
    the env's interpreter with --target-root models/<env>."""
    steps = []
    prestage = home / "scripts" / f"prestage_{env_name}_weights.py"
    if prestage.exists():
        steps.append([sys.executable, str(prestage)])
    prepare = home / "scripts" / f"prepare_{env_name}_weights.py"
    if prepare.exists():
        steps.append([str(python), str(prepare), "--target-root", str(home / "models" / env_name)])
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", nargs="?", help="family, version, or env name from models.json")
    ap.add_argument("prefix", nargs="?", help="existing env prefix (contains bin/python)")
    ap.add_argument("--list", action="store_true", help="show current adoptions")
    ap.add_argument("--remove", metavar="ENV", help="remove an adoption entry")
    args = ap.parse_args()

    home = resolve_home()
    refuse_plugin_copy(home)
    if args.list:
        data = load_map(home)
        if not data:
            print("no adopted envs (env_map.local.json absent or empty)")
        for env, prefix in sorted(data.items()):
            print(f"{env:<16} -> {prefix}")
        return 0
    if args.remove:
        data = load_map(home)
        key = args.remove
        if key not in data:
            # adoption is added by family, version or env name but stored by env
            # name; accept the same names on removal
            registry = json.loads((home / "models.json").read_text(encoding="utf-8"))
            found = find_family(key, registry)
            key = found[1]["env"] if found else key
        if key not in data:
            print(f"not adopted: {args.remove} (adopted: {', '.join(sorted(data)) or 'none'})",
                  file=sys.stderr)
            return 1
        del data[key]
        save_map(home, data)
        print(f"removed adoption: {key}")
        return 0
    if not args.model or not args.prefix:
        ap.error("need MODEL and PREFIX (or --list / --remove)")

    registry = json.loads((home / "models.json").read_text(encoding="utf-8"))
    found = find_family(args.model, registry)
    if found is None:
        print(f"unknown model/env: {args.model}; families: {known_names(registry)} "
              "(or any version or env name in models.json)", file=sys.stderr)
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
    prepare_weights(home, env_name, python)
    return 0 if prebuild_d3(home, env_name, prefix) else 1


def prebuild_d3(home: Path, env_name: str, prefix: Path) -> bool:
    """Build the D3 kernel the way install.sh does (scripts/build_d3.py).

    An adopted env is then used through its bare interpreter, which has no env
    bin on PATH, so a first D3 call there could not compile (no ninja, wrong or
    no nvcc). False only when the build itself failed: the adoption stands
    (its imports passed), but D3 is not usable until the logged error is fixed."""
    import build_d3
    log = home / "logs" / f"d3_{env_name}.log"
    state, detail = build_d3.prebuild(prefix, log)
    if state == "failed":
        print(f"D3 kernel build FAILED for {env_name}: {detail}\n"
              f"  the adoption stands, but D3 does not work in this env until this succeeds:\n"
              f"    python3 {build_d3.__file__} {prefix} --log {log}",
              file=sys.stderr)
        return False
    print(f"d3 kernel: {state}: {detail}")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
