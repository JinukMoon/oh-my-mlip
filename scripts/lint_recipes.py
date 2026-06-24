#!/usr/bin/env python3
"""lint_recipes.py — GPU-free structural lint of the envs/*.yml recipes.

Pure-python (stdlib + optional PyYAML dev dep). Needs no torch / ase / conda.

Structural checks (any failure -> non-zero exit):
  1. Every framework env referenced in models.json has an envs/<env>.yml recipe.
  2. Every recipe parses (PyYAML if importable, else a tolerant line check).
  3. Each recipe's `python=X.Y.Z` pin matches envs/_expected.json.
  4. Each recipe's `torch==...+cuNNN` line has a matching
     `--extra-index-url .../cuNNN` (same NNN).
  6. No recipe pip block contains a bare `- --no-deps` line. (Regression guard:
     a standalone `--no-deps` is an invalid requirement to conda's pip-block
     parser and silently breaks `conda env create -f`.) catbench is NOT in any
     recipe; install.sh installs it post-create as `pip install --no-deps
     catbench==...`.
  7. install.sh installs catbench as a post-create step (a real
     `pip install --no-deps catbench==...` line is present), since it was
     removed from the recipe pip blocks.

Informational (never a failure):
  5. Reports the `# build_status: clean|candidate` split, with candidate reasons.
     A `candidate` recipe is allowed and does NOT fail the lint.

The per-env python/torch/cuNNN truth lives in envs/_expected.json (committed so
CI needs no scratch access).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENVS_DIR = REPO_ROOT / "envs"
MODELS_JSON = REPO_ROOT / "models.json"
EXPECTED_JSON = ENVS_DIR / "_expected.json"
INSTALL_SH = REPO_ROOT / "install.sh"

try:  # PyYAML is a dev dependency; prefer it, but degrade gracefully.
    import yaml  # type: ignore

    _HAVE_YAML = True
except Exception:  # noqa: BLE001
    yaml = None  # type: ignore
    _HAVE_YAML = False


_PYTHON_RE = re.compile(r"python\s*=\s*([\d]+\.[\d]+\.[\d]+)")
_TORCH_RE = re.compile(r"torch\s*==\s*\S*\+cu(\d+)")
_INDEX_RE = re.compile(r"--extra-index-url\s+\S*/whl/cu(\d+)")
_STATUS_RE = re.compile(r"^#\s*build_status:\s*(\w+)", re.IGNORECASE)
_REASON_RE = re.compile(r"^#\s*candidate-reason:\s*(.+)$", re.IGNORECASE)
# A bare `- --no-deps` list item inside the pip block (the build-breaking bug).
_BARE_NODEPS_RE = re.compile(r"^-\s+--no-deps\s*$")
# install.sh's catbench post-create step: `pip install --no-deps catbench==...`.
_INSTALL_CATBENCH_RE = re.compile(r"--no-deps\s+catbench==")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_envs(models: dict) -> dict:
    """Map each registered framework name -> its env (skips `_meta` etc.)."""
    out: dict[str, str] = {}
    for name, info in models.items():
        if name.startswith("_") or not isinstance(info, dict):
            continue
        env = info.get("env")
        if isinstance(env, str):
            out[name] = env
    return out


def _parse_recipe(text: str, recipe: Path, errors: list[str]) -> None:
    """Check 2: the recipe parses."""
    if _HAVE_YAML:
        try:
            doc = yaml.safe_load(text)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{recipe.name}: YAML parse error: {exc!r}")
            return
        if not isinstance(doc, dict) or "dependencies" not in doc:
            errors.append(
                f"{recipe.name}: parsed YAML lacks a top-level 'dependencies' mapping"
            )
        return
    # Tolerant fallback: just confirm the recipe has the expected anchors.
    if "dependencies:" not in text:
        errors.append(f"{recipe.name}: no 'dependencies:' block found (tolerant check)")
    if "pip:" not in text:
        errors.append(f"{recipe.name}: no 'pip:' block found (tolerant check)")


def lint(
    envs_dir: Path = ENVS_DIR,
    models_json: Path = MODELS_JSON,
    expected_json: Path = EXPECTED_JSON,
    install_sh: Path = INSTALL_SH,
) -> tuple[list[str], dict[str, list[str]]]:
    """Run the lint. Returns (errors, status_split).

    `errors` is empty iff structural checks (1-4) all pass.
    `status_split` maps {"clean": [...], "candidate": ["env: reason", ...]}.
    """
    errors: list[str] = []
    status_split: dict[str, list[str]] = {"clean": [], "candidate": []}

    models = _load_json(models_json)
    expected = _load_json(expected_json)
    model_envs = _model_envs(models)

    # Check 1: every framework env has a recipe.
    for name, env in sorted(model_envs.items()):
        if not (envs_dir / f"{env}.yml").is_file():
            errors.append(
                f"models.json framework {name!r} -> env {env!r} has no "
                f"recipe at envs/{env}.yml"
            )

    recipes = sorted(p for p in envs_dir.glob("*.yml") if not p.name.startswith("_"))

    for recipe in recipes:
        env = recipe.stem
        text = recipe.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Check 2: parses.
        _parse_recipe(text, recipe, errors)

        # Check 3: python pin matches _expected.json.
        exp = expected.get(env)
        pym = _PYTHON_RE.search(text)
        if exp is None:
            errors.append(f"{recipe.name}: env {env!r} missing from envs/_expected.json")
        elif pym is None:
            errors.append(f"{recipe.name}: no 'python=X.Y.Z' pin found")
        elif pym.group(1) != exp["python"]:
            errors.append(
                f"{recipe.name}: python pin {pym.group(1)} != expected "
                f"{exp['python']} (envs/_expected.json)"
            )

        # Check 4: torch +cuNNN line has a matching --extra-index-url cuNNN.
        tm = _TORCH_RE.search(text)
        im = _INDEX_RE.search(text)
        if tm is None:
            errors.append(f"{recipe.name}: no 'torch==...+cuNNN' pin found")
        else:
            torch_cu = tm.group(1)
            if im is None:
                errors.append(
                    f"{recipe.name}: torch pins cu{torch_cu} but no "
                    f"'--extra-index-url .../cuNNN' line found"
                )
            elif im.group(1) != torch_cu:
                errors.append(
                    f"{recipe.name}: torch cu{torch_cu} != extra-index-url "
                    f"cu{im.group(1)}"
                )
            # Cross-check against _expected.json cu too (defensive).
            if exp is not None and f"cu{torch_cu}" != exp["cu"]:
                errors.append(
                    f"{recipe.name}: torch cu{torch_cu} != expected {exp['cu']} "
                    f"(envs/_expected.json)"
                )

        # Check 6: no bare `- --no-deps` list item in the pip block. A standalone
        # `--no-deps` is an invalid requirement to conda's pip-block parser and
        # silently breaks `conda env create -f` (catbench is installed by
        # install.sh post-create instead).
        for lineno, line in enumerate(lines, start=1):
            if _BARE_NODEPS_RE.match(line.strip()):
                errors.append(
                    f"{recipe.name}:{lineno}: bare '- --no-deps' in pip block "
                    f"(invalid requirement; breaks 'conda env create'). catbench "
                    f"is installed by install.sh post-create instead."
                )

        # Check 5 (informational): build_status header (line 1) + reason.
        status = None
        for line in lines:
            sm = _STATUS_RE.match(line.strip())
            if sm:
                status = sm.group(1).lower()
                break
        if status == "candidate":
            reason = ""
            for line in lines:
                rm = _REASON_RE.match(line.strip())
                if rm:
                    reason = rm.group(1).strip()
                    break
            status_split["candidate"].append(
                f"{env}: {reason}" if reason else env
            )
        elif status == "clean":
            status_split["clean"].append(env)
        else:
            errors.append(
                f"{recipe.name}: missing/invalid '# build_status:' header "
                f"(got {status!r})"
            )

    # Check 7: install.sh installs catbench post-create (a real
    # `pip install --no-deps catbench==...` line), since catbench was removed
    # from the recipe pip blocks.
    if not install_sh.is_file():
        errors.append(f"install.sh not found at {install_sh}")
    else:
        install_text = install_sh.read_text(encoding="utf-8")
        if not _INSTALL_CATBENCH_RE.search(install_text):
            errors.append(
                "install.sh: missing catbench post-create step "
                "(expected a 'pip install --no-deps catbench==...' line)"
            )

    return errors, status_split


def main(argv: list[str] | None = None) -> int:
    errors, status_split = lint()

    clean = sorted(status_split["clean"])
    candidate = sorted(status_split["candidate"])

    print(f"build_status: {len(clean)} clean, {len(candidate)} candidate")
    print("  clean:")
    for env in clean:
        print(f"    - {env}")
    print("  candidate:")
    for entry in candidate:
        print(f"    - {entry}")
    print()

    if errors:
        print(f"LINT FAILED: {len(errors)} structural error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("LINT OK: all structural checks (1-4, 6-7) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
