"""Tests for scripts/verify_determinism.py.

The fixtures are synthetic recipe directories under tmp_path, so these tests are
GPU-free and do not import any model packages.
"""
from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "verify_determinism.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_determinism", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _write_recipe(
    tmp_path: Path,
    name: str,
    pip_lines: list[str],
    *,
    build_status: str = "clean",
    reason: str | None = None,
) -> Path:
    header = f"# build_status: {build_status}\n"
    if reason:
        header += f"# candidate-reason: {reason}\n"
    pip_block = "\n".join(f"      - {line}" for line in pip_lines)
    text = header + textwrap.dedent(f"""\
        name: {name}
        channels:
          - conda-forge
        dependencies:
          - python=3.11.13
          - pip
          - pip:
        """) + pip_block + "\n"
    path = tmp_path / f"{name}.yml"
    path.write_text(text, encoding="utf-8")
    return path


def test_fully_pinned_recipe_passes(tmp_path: Path):
    mod = _load_module()
    _write_recipe(
        tmp_path,
        "good",
        [
            "--extra-index-url https://download.pytorch.org/whl/cu126",
            "torch==2.7.1+cu126",
            "e3nn==0.5.6",
            "pkg @ git+https://github.com/example/pkg.git@0123456789abcdef",
            "-e git+https://github.com/example/editable@abcdef123456#egg=editable",
            "https://example.com/wheels/pkg-1.0.0-py3-none-any.whl",
        ],
    )
    reports = mod.check_all(tmp_path)
    assert len(reports) == 1
    assert reports[0].deterministic
    assert mod.main(["--envs-dir", str(tmp_path)]) == 0


def test_bare_package_fails(tmp_path: Path):
    mod = _load_module()
    _write_recipe(tmp_path, "bad", ["e3nn"])
    reports = mod.check_all(tmp_path)
    assert not reports[0].deterministic
    assert any("bare package" in o.reason for o in reports[0].offenders)
    assert mod.main(["--envs-dir", str(tmp_path)]) == 1


def test_unpinned_git_fails(tmp_path: Path):
    mod = _load_module()
    _write_recipe(
        tmp_path,
        "badgit",
        ["pkg @ git+https://github.com/example/pkg.git"],
    )
    reports = mod.check_all(tmp_path)
    assert not reports[0].deterministic
    assert any("commit SHA" in o.reason for o in reports[0].offenders)


def test_candidate_documented_file_url_passes(tmp_path: Path):
    mod = _load_module()
    _write_recipe(
        tmp_path,
        "private",
        [
            "torch==2.7.1+cu126",
            "privatepkg @ file:///owner/private/pkg  # private local source; owner needed",
        ],
        build_status="candidate",
        reason="privatepkg is a private local file:// source; owner must publish a wheel",
    )
    reports = mod.check_all(tmp_path)
    assert reports[0].deterministic
    assert reports[0].candidate_with_private_source
    assert mod.main(["--envs-dir", str(tmp_path)]) == 0
