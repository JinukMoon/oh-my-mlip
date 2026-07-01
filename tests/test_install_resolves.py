"""GPU-free check that install.sh --dry-run resolves a recipe for every target.

`install.sh --dry-run <X>` must, for each X, print a 'would create env ...' line
and NEVER a 'SKIP ... no recipe' line — for:
  * every registered model name from oh_my_mlip.list_models() (+ versions where
    install.sh accepts them), AND
  * every env name (the 20 envs/*.yml recipes).

This runs install.sh via subprocess with --dry-run only: no conda, no network,
no GPU. Model names are resolved to envs via models.json by install.sh itself.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import oh_my_mlip

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "install.sh"
ENVS_DIR = REPO_ROOT / "envs"


def _env_names() -> list[str]:
    """The 20 recipe stems (excludes _expected.json and other underscore files)."""
    return sorted(p.stem for p in ENVS_DIR.glob("*.yml") if not p.name.startswith("_"))


def _model_names() -> list[str]:
    return list(oh_my_mlip.list_models())


def _dry_run(targets: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(INSTALL_SH), "--dry-run", *targets],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def _assert_resolves(proc: subprocess.CompletedProcess[str], target: str) -> None:
    out = proc.stdout
    assert proc.returncode == 0, f"{target}: exit {proc.returncode}\n{proc.stderr}"
    # No SKIP / "no recipe" lines anywhere.
    bad = [
        ln
        for ln in out.splitlines()
        if "SKIP" in ln or "no recipe" in ln
    ]
    assert not bad, f"{target}: install.sh emitted SKIP/no-recipe lines:\n" + "\n".join(bad)
    # Every target must resolve to a concrete build path: either a single-pass
    # 'would create env' or a multi-pass 'would build env ... via ... sidecar'.
    assert ("would create env" in out) or ("would build env" in out), (
        f"{target}: no 'would create env'/'would build env' line:\n{out}"
    )


@pytest.mark.parametrize("env_name", _env_names())
def test_dry_run_resolves_every_env(env_name: str):
    _assert_resolves(_dry_run([env_name]), env_name)


@pytest.mark.parametrize("model_name", _model_names())
def test_dry_run_resolves_every_model(model_name: str):
    _assert_resolves(_dry_run([model_name]), model_name)


def test_dry_run_all_targets_together_no_skip():
    """All env + model names in a single invocation: exit 0, zero SKIP lines."""
    targets = _env_names() + _model_names()
    proc = _dry_run(targets)
    assert proc.returncode == 0, proc.stderr
    skips = [ln for ln in proc.stdout.splitlines() if "SKIP" in ln or "no recipe" in ln]
    assert not skips, "unexpected SKIP/no-recipe lines:\n" + "\n".join(skips)
    # Every env should have produced a 'would create' line.
    created = sum(1 for ln in proc.stdout.splitlines() if "would create env" in ln)
    assert created >= len(_env_names()), (
        f"expected >= {len(_env_names())} 'would create' lines, got {created}"
    )


@pytest.mark.parametrize("model_name", _model_names())
def test_dry_run_resolves_model_versions(model_name: str):
    """Where a model exposes versions, install.sh should still resolve the model
    name to its env (install.sh keys on the model/env name, not the version, so
    we assert the model name itself resolves — versions ride the same env)."""
    versions = oh_my_mlip.list_versions(model_name)
    assert versions, f"{model_name}: expected at least one version"
    # The model name resolves to a recipe (versions share the env).
    _assert_resolves(_dry_run([model_name]), model_name)
