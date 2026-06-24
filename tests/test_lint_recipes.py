"""Tests for scripts/lint_recipes.py — the GPU-free recipe linter.

Positive: the linter exits 0 on the real envs/ tree (both via main() and CLI).
Negative: a temp recipe with a cuXXX/index mismatch (and a temp _expected.json
mismatch) makes the linter report a structural error — proving CI catches drift.

No GPU, conda, torch, or ase required.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "lint_recipes.py"
ENVS_DIR = REPO_ROOT / "envs"


def _load_lint_module():
    spec = importlib.util.spec_from_file_location("lint_recipes", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ── Positive: real tree passes ────────────────────────────────────────────────

def test_lint_main_exits_zero_on_real_tree():
    mod = _load_lint_module()
    assert mod.main() == 0


def test_lint_cli_exits_zero_on_real_tree():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    # The clean/candidate split is printed (informational).
    assert "build_status:" in proc.stdout
    assert "LINT OK" in proc.stdout


def test_lint_reports_clean_and_candidate_split():
    mod = _load_lint_module()
    errors, split = mod.lint()
    assert errors == []
    assert split["clean"], "expected at least one clean recipe"
    assert split["candidate"], "expected at least one candidate recipe"
    # No env appears in both buckets.
    candidate_envs = {entry.split(":", 1)[0] for entry in split["candidate"]}
    assert not (set(split["clean"]) & candidate_envs)


# ── Negative helpers: build an isolated copy of the tree, then break it ────────

def _stage_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Copy envs/ + models.json into tmp_path; return (envs, models, expected)."""
    envs = tmp_path / "envs"
    shutil.copytree(ENVS_DIR, envs)
    models = tmp_path / "models.json"
    shutil.copyfile(REPO_ROOT / "models.json", models)
    expected = envs / "_expected.json"
    return envs, models, expected


def test_lint_catches_cu_index_mismatch(tmp_path: Path):
    """A recipe whose --extra-index-url cuNNN disagrees with its torch cuNNN
    must be flagged."""
    mod = _load_lint_module()
    envs, models, expected = _stage_tree(tmp_path)

    mace = envs / "mace.yml"
    text = mace.read_text(encoding="utf-8")
    # mace pins cu126; break only the index URL -> cu128 (torch still cu126).
    broken = text.replace("/whl/cu126", "/whl/cu128", 1)
    assert broken != text, "fixture precondition: cu126 index line must exist"
    mace.write_text(broken, encoding="utf-8")

    errors, _ = mod.lint(envs_dir=envs, models_json=models, expected_json=expected)
    assert any("mace.yml" in e and "extra-index-url" in e for e in errors), errors


def test_lint_catches_expected_python_mismatch(tmp_path: Path):
    """A _expected.json whose python pin disagrees with the recipe must be
    flagged."""
    mod = _load_lint_module()
    envs, models, expected = _stage_tree(tmp_path)

    data = json.loads(expected.read_text(encoding="utf-8"))
    data["mace"]["python"] = "3.9.0"  # recipe pins 3.11.13
    expected.write_text(json.dumps(data, indent=2), encoding="utf-8")

    errors, _ = mod.lint(envs_dir=envs, models_json=models, expected_json=expected)
    assert any("mace.yml" in e and "python pin" in e for e in errors), errors


def test_lint_catches_missing_recipe(tmp_path: Path):
    """Deleting a recipe for a framework referenced in models.json is caught by
    check 1."""
    mod = _load_lint_module()
    envs, models, expected = _stage_tree(tmp_path)

    (envs / "mace.yml").unlink()

    errors, _ = mod.lint(envs_dir=envs, models_json=models, expected_json=expected)
    assert any("mace" in e and "no" in e.lower() and "recipe" in e for e in errors), errors
