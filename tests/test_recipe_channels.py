"""Recipes never resolve against Anaconda's defaults channel.

Miniconda's base configuration appends `defaults` to every solve unless the
recipe says `nodefaults`, and conda 25.x can ask for Anaconda's terms of service
on that channel in a non-interactive install. The recipes use conda-forge and
nvidia only.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENVS = REPO_ROOT / "envs"

# Locks that still reference the defaults channel and await regeneration from a
# build of the current recipe. Empty since 2026-09-23, when the twelve locks
# recorded before `nodefaults` were rebuilt; no lock may reference that channel.
LOCKS_PENDING_REGENERATION: set[str] = set()


def test_every_recipe_excludes_the_defaults_channel():
    missing = []
    for recipe in sorted(ENVS.glob("*.yml")):
        block = re.search(r"^channels:\n((?:[ \t]+- .*\n)+)", recipe.read_text(), re.M)
        channels = re.findall(r"- (\S+)", block.group(1)) if block else []
        if "nodefaults" not in channels or "defaults" in channels:
            missing.append(f"{recipe.name}: {channels}")
    assert not missing, missing


def test_build_sidecars_override_channels():
    for sidecar in sorted(ENVS.glob("*.build.sh")):
        text = sidecar.read_text()
        for line in text.splitlines():
            if " create " in line and "-c conda-forge" in line:
                assert "--override-channels" in line, f"{sidecar.name}: {line.strip()}"


def test_no_new_lock_uses_the_defaults_channel():
    offending = {lock.name.split(".")[0] for lock in (ENVS / "locks").glob("*.conda.txt")
                 if "repo.anaconda.com" in lock.read_text()}
    assert offending <= LOCKS_PENDING_REGENERATION, sorted(offending - LOCKS_PENDING_REGENERATION)


def test_locks_carry_the_build_tool_pins_their_recipe_declares():
    """`pip freeze` hides pip/setuptools/wheel, so locks built from one dropped
    the older setuptools several recipes pin on purpose (their framework imports
    pkg_resources, removed in setuptools 81). A replay then installed conda's
    newer setuptools alone and failed at import."""
    import re
    missing = []
    for recipe in sorted(ENVS.glob("*.yml")):
        pins = [line.strip().lstrip("- ").strip()
                for line in recipe.read_text().splitlines()
                if re.match(r"^\s*-\s*(setuptools|pip|wheel)==\S+$", line)]
        lock = ENVS / "locks" / f"{recipe.stem}.pip.txt"
        if not pins or not lock.is_file() or recipe.stem in LOCKS_PENDING_REGENERATION:
            continue  # replaced wholesale by the regenerated lock
        have = lock.read_text()
        missing += [f"{lock.name}: {pin}" for pin in pins if pin not in have]
    assert not missing, missing
