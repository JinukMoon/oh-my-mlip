#!/usr/bin/env python3
"""select_recipe.py — pick the env recipe for this host's GPU architecture.

An env has one base recipe, ``envs/<env>.yml``, validated on older GPUs. Some
envs also have a SIBLING recipe for a newer architecture (``envs/<env>.cu128.yml``
for sm120), because the base recipe's torch wheel ships no kernels for it.
``envs/_arch_recipes.json`` says which recipe each env uses on which arch:

  * host arch listed in the table:
      ``{"recipe": "<stem>"}``                 -> build envs/<stem>.yml
      ``{"recipe": null, "reason": "..."}``   -> unsupported there: STOP
      env missing from that arch's block       -> not checked there: STOP
  * host arch not listed, at or below ``base_max_arch`` -> the base recipe
  * host arch not listed, above ``base_max_arch``       -> STOP (unchecked GPU)
  * no GPU detected                                     -> the base recipe

Whatever the recipe, the env is built at ``envs/<env>/`` (models.json paths do
not change); install.sh records the stem in ``envs/<env>/.omm_recipe``.

The host arch is ``detect_host_arch()`` (oh_my_mlip/registry.py, nvidia-smi
compute_cap of the GPU this process would use); ``OMM_INSTALL_ARCH=sm120``
(or ``none`` for a GPU-less host) overrides it, for a dry run of another
host's plan and for tests.

CLI (used by install.sh; stdlib only, never imports torch):
  select_recipe.py <env>            prints the recipe stem; exit 0
                                    exit 3 + reason on stderr when it must stop
  select_recipe.py <env> --json     {"env", "arch", "recipe", "reason"}
  select_recipe.py --host-arch      prints the arch used (or "none")
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TABLE = "_arch_recipes.json"
STOP = 3


def arch_number(arch: str) -> int | None:
    m = re.fullmatch(r"sm_?(\d+)", (arch or "").strip().lower())
    return int(m.group(1)) if m else None


def host_arch() -> str | None:
    override = os.environ.get("OMM_INSTALL_ARCH", "").strip()
    if override.lower() in ("none", "cpu"):
        return None
    if override:
        n = arch_number(override)
        return f"sm{n}" if n is not None else None
    sys.path.insert(0, str(REPO))
    try:
        from oh_my_mlip.registry import detect_host_arch
    except Exception:  # noqa: BLE001 - a broken checkout must not pick a recipe silently
        return None
    return detect_host_arch()


def load_table(envs_dir: Path) -> dict:
    path = envs_dir / TABLE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def select(env: str, arch: str | None, envs_dir: Path) -> tuple[str | None, str]:
    """(recipe stem, reason). A None stem means: stop, do not build."""
    table = load_table(envs_dir)
    if arch is None:
        return env, "no GPU detected: base recipe"
    block = table.get(arch)
    if isinstance(block, dict):
        if env not in block:
            return None, (f"no recipe for env '{env}' has been checked on {arch}; "
                          f"envs/{TABLE} lists none, so nothing is installed")
        entry = block[env] or {}
        stem = entry.get("recipe")
        if not stem:
            return None, (f"'{env}' is not supported on {arch}: "
                          f"{entry.get('reason') or 'no working build is known'}")
        if not (envs_dir / f"{stem}.yml").is_file():
            return None, f"envs/{TABLE} names recipe '{stem}' for {arch}, but envs/{stem}.yml is missing"
        return stem, (f"base recipe; its CUDA build covers {arch}" if stem == env else f"sibling recipe for {arch}")
    limit = arch_number(table.get("base_max_arch", "")) if table else None
    n = arch_number(arch)
    if limit is not None and n is not None and n > limit:
        return None, (f"the GPU here is {arch}, newer than any architecture the recipes were checked on "
                      f"(base recipes up to {table['base_max_arch']}; sibling recipes for "
                      f"{', '.join(k for k in table if k.startswith('sm')) or 'none'}). The base recipe's "
                      f"CUDA wheels may carry no kernels for it, so nothing is installed")
    return env, "base recipe"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("env", nargs="?")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--host-arch", action="store_true", help="print the arch used and exit")
    ap.add_argument("--envs-dir", type=Path, default=REPO / "envs")
    args = ap.parse_args(argv)
    arch = host_arch()
    if args.host_arch:
        print(arch or "none")
        return 0
    if not args.env:
        ap.error("env is required")
    stem, reason = select(args.env, arch, args.envs_dir)
    if args.json:
        print(json.dumps({"env": args.env, "arch": arch, "recipe": stem, "reason": reason}))
    elif stem:
        print(stem)
    else:
        print(reason, file=sys.stderr)
    return 0 if stem else STOP


if __name__ == "__main__":
    raise SystemExit(main())
