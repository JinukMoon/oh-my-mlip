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

# Locks recorded before the recipes carried `nodefaults`: each still fetches
# libffi and libglib from repo.anaconda.com and must be regenerated from a build
# of the current recipe. Remove an env from this set once its lock is rebuilt;
# no other lock may reference the defaults channel.
LOCKS_PENDING_REGENERATION = {
    "allegro", "alphanet", "chgnet", "deepmd", "eqnorm", "fairchemv1",
    "mace", "nequip", "nequix", "orb", "sevennet", "uma",
}


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
