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
    conda_lines: list[str] | None = None,
) -> Path:
    header = f"# build_status: {build_status}\n"
    if reason:
        header += f"# candidate-reason: {reason}\n"
    conda_block = "\n".join(
        f"  - {line}" for line in (conda_lines or ["python=3.11.13", "pip"])
    )
    pip_block = "\n".join(f"      - {line}" for line in pip_lines)
    text = header + textwrap.dedent(f"""\
        name: {name}
        channels:
          - conda-forge
        dependencies:
        """) + conda_block + "\n  - pip:\n" + pip_block + "\n"
    path = tmp_path / f"{name}.yml"
    path.write_text(text, encoding="utf-8")
    return path


# ── conda half of the gate ───────────────────────────────────────────────────

def test_bare_conda_package_fails(tmp_path: Path):
    """A floating `- ase` must fail.

    This is the regression the pip-only gate missed: an unpinned conda dep rode
    each new ASE release into the build and broke mattersim/eqnorm, while the
    checker reported the recipe as deterministic.
    """
    mod = _load_module()
    _write_recipe(tmp_path, "bad", ["torch==2.7.1"],
                  conda_lines=["python=3.11.13", "pip", "ase"])
    reports = mod.check_all(tmp_path)
    assert not reports[0].deterministic
    assert any("bare conda package" in o.reason for o in reports[0].offenders)
    assert mod.main(["--envs-dir", str(tmp_path)]) == 1


def test_pinned_conda_package_passes(tmp_path: Path):
    mod = _load_module()
    _write_recipe(tmp_path, "good", ["torch==2.7.1"],
                  conda_lines=["python=3.11.13", "pip", "ase=3.29.0",
                               "libstdcxx-ng=13.2.0=h7e041cc_5"])
    reports = mod.check_all(tmp_path)
    assert reports[0].deterministic, [o.reason for o in reports[0].offenders]


def test_conda_range_constraint_fails(tmp_path: Path):
    mod = _load_module()
    _write_recipe(tmp_path, "range", ["torch==2.7.1"],
                  conda_lines=["python=3.11.13", "pip", "ase>=3.25"])
    reports = mod.check_all(tmp_path)
    assert any("range/version constraint" in o.reason for o in reports[0].offenders)


def test_series_pin_allowed_only_for_toolchain_packages(tmp_path: Path):
    """`cuda-nvcc=12.8.*` is an intentional series pin; `ase=3.29.*` is not."""
    mod = _load_module()
    _write_recipe(tmp_path, "toolchain", ["torch==2.7.1"],
                  conda_lines=["python=3.11.13", "pip", "cuda-nvcc=12.8.*"])
    assert mod.check_all(tmp_path)[0].deterministic

    for path in tmp_path.glob("*.yml"):
        path.unlink()
    _write_recipe(tmp_path, "floaty", ["torch==2.7.1"],
                  conda_lines=["python=3.11.13", "pip", "ase=3.29.*"])
    reports = mod.check_all(tmp_path)
    assert any("series pin floats the patch level" in o.reason for o in reports[0].offenders)


def test_pip_block_items_are_not_read_as_conda_deps(tmp_path: Path):
    """The nested `- pip:` items must be classified by the pip rules only.

    A bare pip requirement is a pip offender ("bare package"), never a conda
    one — if the scanner leaked, every recipe would double-report.
    """
    mod = _load_module()
    _write_recipe(tmp_path, "nested", ["e3nn"])
    reasons = [o.reason for o in mod.check_all(tmp_path)[0].offenders]
    assert reasons == ["bare package is not deterministic"], reasons


def test_real_recipes_pass_the_conda_half():
    """Every shipped recipe must satisfy the extended gate, not just the pip half."""
    mod = _load_module()
    reports = mod.check_all(REPO_ROOT / "envs")
    bad = {r.name: [f"{o.lineno}: {o.requirement} ({o.reason})" for o in r.offenders]
           for r in reports if not r.deterministic}
    assert not bad, bad


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
