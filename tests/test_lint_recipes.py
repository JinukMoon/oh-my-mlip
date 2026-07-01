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
    """Copy the recipe FILES under envs/ + models.json into tmp_path; return
    (envs, models, expected). Only files directly under envs/ are copied (recipes:
    *.yml, *.build.sh, _expected.json). A built env is a DIRECTORY and is skipped —
    copying it would drag in a multi-GB conda env (present after a local install)
    and make the copy hang. lint only reads the recipe files, so this is complete."""
    envs = tmp_path / "envs"
    envs.mkdir(parents=True, exist_ok=True)
    for f in ENVS_DIR.iterdir():
        if f.is_file():
            shutil.copyfile(f, envs / f.name)
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


# ── Regression guards for the conda-env-create --no-deps build bug ─────────────

def test_real_recipes_have_no_bare_no_deps_and_no_catbench():
    """The committed recipes must contain no bare '- --no-deps' line and no
    catbench install line in any pip block (both moved to install.sh)."""
    for recipe in sorted(ENVS_DIR.glob("*.yml")):
        if recipe.name.startswith("_"):
            continue
        for lineno, line in enumerate(recipe.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            assert s != "- --no-deps", f"{recipe.name}:{lineno} has a bare '- --no-deps'"
            assert not s.startswith("- catbench=="), (
                f"{recipe.name}:{lineno} still installs catbench in the recipe"
            )


def test_real_tree_lint_passes_new_checks():
    """The real tree (recipes + install.sh) passes the full lint, including the
    no-bare-no-deps guard (check 6) and the install.sh catbench post-step
    (check 7)."""
    mod = _load_lint_module()
    errors, _ = mod.lint()
    assert errors == [], errors


def test_lint_catches_bare_no_deps_in_recipe(tmp_path: Path):
    """Re-introducing a bare '- --no-deps' line in a recipe pip block is caught
    by check 6 (the exact build bug this guards against)."""
    mod = _load_lint_module()
    envs, models, expected = _stage_tree(tmp_path)

    mace = envs / "mace.yml"
    text = mace.read_text(encoding="utf-8")
    # Append a bare --no-deps line to the pip block (6-space indent like siblings).
    broken = text.rstrip("\n") + "\n      - --no-deps\n"
    mace.write_text(broken, encoding="utf-8")

    errors, _ = mod.lint(envs_dir=envs, models_json=models, expected_json=expected)
    assert any("mace.yml" in e and "--no-deps" in e for e in errors), errors


def test_lint_catches_missing_catbench_post_step_in_install_sh(tmp_path: Path):
    """An install.sh without the 'pip install catbench==...' post-step
    is caught by check 7."""
    mod = _load_lint_module()
    envs, models, expected = _stage_tree(tmp_path)

    fake_install = tmp_path / "install.sh"
    fake_install.write_text("#!/usr/bin/env bash\necho no catbench here\n", encoding="utf-8")

    errors, _ = mod.lint(
        envs_dir=envs, models_json=models, expected_json=expected,
        install_sh=fake_install,
    )
    assert any("install.sh" in e and "catbench" in e for e in errors), errors


def test_lint_accepts_present_catbench_post_step_in_install_sh(tmp_path: Path):
    """An install.sh that DOES carry the catbench post-step satisfies check 7."""
    mod = _load_lint_module()
    envs, models, expected = _stage_tree(tmp_path)

    fake_install = tmp_path / "install.sh"
    # catbench is installed WITH deps (NOT --no-deps): the real build showed
    # --no-deps drops requests/xlsxwriter and breaks catbench.adsorption.
    fake_install.write_text(
        '#!/usr/bin/env bash\n"$prefix/bin/pip" install catbench==1.1.2\n',
        encoding="utf-8",
    )

    errors, _ = mod.lint(
        envs_dir=envs, models_json=models, expected_json=expected,
        install_sh=fake_install,
    )
    assert not any("install.sh" in e and "catbench" in e for e in errors), errors
