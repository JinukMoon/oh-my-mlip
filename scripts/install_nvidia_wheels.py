#!/usr/bin/env python3
"""install_nvidia_wheels.py — the pypi.nvidia.com recovery of AGENTS.md §8, as one command.

torch `+cuNNN` envs pull their `nvidia-*` wheels through pypi.nvidia.com; when
that host is unreachable every torch env fails in the pip step, although the
same wheels are on pypi.org. This script installs exactly the `nvidia-*` pins
torch declares, from pypi.org, into the partially built env, after which
`install.sh <env>` finishes the build as usual.

It acts only on a partial env this hub built (envs/<env> with an interpreter,
no .omm_ready sentinel, not a symlink), and only when the pins PyPI lists for
that torch version are the ones the recipe's CUDA build uses (PyPI describes
one CUDA build per torch version; a recipe on another build is refused rather
than given the wrong libraries).

Usage:
  python3 scripts/install_nvidia_wheels.py <env or model> [--dry-run] [--json]
Exit: 0 installed (or planned with --dry-run); 2 refused (the reason says why);
3 PyPI unreachable or pip failed.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _setup_common import refuse_plugin_copy, resolve_home  # noqa: E402

PYPI_JSON = "https://pypi.org/pypi/torch/{version}/json"


def _fetch_json(url: str, attempts: int = 3, timeout: float = 60) -> dict:
    last = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001 - reported with the URL below
            last = exc
            time.sleep(2 * (attempt + 1))
    raise ConnectionError(f"{url}: {last}")


def env_name(target: str, home: Path) -> str | None:
    """An env name, or the env of a family / variant in models.json."""
    if (home / "envs" / f"{target}.yml").is_file():
        return target
    registry = json.loads((home / "models.json").read_text(encoding="utf-8"))
    want = target.lower()
    for family, info in registry.items():
        if family.startswith("_") or not isinstance(info, dict):
            continue
        names = {family.lower()} | {v.lower() for v in (info.get("versions") or {})}
        if want in names or want == str(info.get("env", "")).lower():
            return info.get("env")
    return None


def torch_pin(recipe: Path) -> tuple[str, str] | None:
    """(torch version, cuda tag) from the recipe, e.g. ('2.7.1', 'cu126')."""
    m = re.search(r"torch==([0-9.]+)\+(cu[0-9]+)", recipe.read_text(encoding="utf-8"))
    return (m.group(1), m.group(2)) if m else None


def nvidia_pins(requires_dist: list[str]) -> list[str]:
    """The nvidia-* requirements torch declares for Linux x86_64, markers dropped."""
    pins = []
    for req in requires_dist or []:
        spec, _, marker = req.partition(";")
        spec = spec.strip()
        if not spec.startswith("nvidia-"):
            continue
        if marker and ("Linux" not in marker or ("platform_machine" in marker and "x86_64" not in marker)):
            continue
        pins.append(spec)
    return pins


def cuda_tag_of(pins: list[str]) -> str | None:
    """cuNNN of the CUDA runtime those pins install (nvidia-cuda-runtime-cu12==12.8.90 -> cu128)."""
    for pin in pins:
        m = re.match(r"nvidia-cuda-runtime(?:-cu\d+)?==(\d+)\.(\d+)", pin)
        if m:
            return f"cu{m.group(1)}{m.group(2)}"
    return None


def plan(target: str, home: Path, fetch=_fetch_json) -> dict:
    env = env_name(target, home)
    if env is None:
        return {"ok": False, "exit": 2, "reason": f"{target!r} is not an env, family or variant in models.json"}
    prefix = home / "envs" / env
    python = prefix / "bin" / "python"
    if prefix.is_symlink():
        return {"ok": False, "exit": 2, "reason": f"{prefix} is a symlink; this recovery only touches envs this hub builds"}
    if not python.exists():
        return {"ok": False, "exit": 2, "reason": f"{prefix} has no interpreter yet: run install.sh {env} first; "
                                                   "this recovery finishes a build that stopped in its pip step"}
    if (prefix / ".omm_ready").exists():
        return {"ok": False, "exit": 2, "reason": f"{env} is already installed (.omm_ready); nothing to recover"}
    pin = torch_pin(home / "envs" / f"{env}.yml")
    if pin is None:
        return {"ok": False, "exit": 2, "reason": f"envs/{env}.yml pins no torch+cuNNN wheel; this recovery does not apply"}
    version, cuda = pin
    try:
        meta = fetch(PYPI_JSON.format(version=version))
    except ConnectionError as exc:
        return {"ok": False, "exit": 3, "reason": f"pypi.org unreachable too ({exc}); retry later or on another network"}
    pins = nvidia_pins(meta.get("info", {}).get("requires_dist") or [])
    if not pins:
        return {"ok": False, "exit": 2, "reason": f"PyPI lists no nvidia-* requirements for torch {version}"}
    pypi_cuda = cuda_tag_of(pins)
    if pypi_cuda != cuda:
        return {"ok": False, "exit": 2,
                "reason": f"the recipe uses torch {version}+{cuda}, but PyPI's torch {version} pins the {pypi_cuda} "
                          f"libraries; installing those would mismatch the build. This case needs pypi.nvidia.com "
                          f"(or another network), not this recovery."}
    command = [str(python), "-m", "pip", "install", "--index-url", "https://pypi.org/simple",
               "--retries", "10", "--timeout", "60", *pins]
    return {"ok": True, "exit": 0, "env": env, "torch": f"{version}+{cuda}", "pins": pins, "command": command,
            "next": f"bash {home / 'install.sh'} {env}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="env name, model family or variant")
    ap.add_argument("--dry-run", action="store_true", help="print the pins and the pip command; install nothing")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    home = resolve_home()
    refuse_plugin_copy(home)
    result = plan(args.target, home)
    if result["ok"] and not args.dry_run:
        rc = subprocess.run(result["command"]).returncode
        if rc != 0:
            result.update(ok=False, exit=3, reason=f"pip exited {rc} (see its output above)")
    if args.json:
        print(json.dumps(result, indent=2))
    elif not result["ok"]:
        print(f"[stop] {result['reason']}", file=sys.stderr)
    else:
        verb = "would install" if args.dry_run else "installed"
        print(f"{verb} into {result['env']} ({result['torch']}) from pypi.org: {' '.join(result['pins'])}")
        print(f"next: {result['next']}")
    return result["exit"]


if __name__ == "__main__":
    raise SystemExit(main())
