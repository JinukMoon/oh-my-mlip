"""Syntax-only parse check for every `inference` / `import` string in models.json.

This runs `ast.parse()` on each line WITHOUT executing it — no GPU, no conda
env, no torch import. It is the CI gate that catches accidental Python syntax
errors introduced when editing models.json.

Why this matters: `inference` and `import` strings are `exec()`'d inside each
model's conda env at run time. A typo silently broken in the registry would
only surface when a user actually runs the model. This test catches that class
of error at PR time, without needing any of the heavy runtime dependencies.

Coverage:
  - `import` lines (top-level per framework)
  - `inference` lines (non-arch-pinned versions)
  - `inference_sm86` / `inference_sm89` lines (arch-pinned versions)
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_JSON = REPO_ROOT / "models.json"


def _collect_code_lines() -> list[tuple[str, str]]:
    """Walk models.json and return (label, code_line) pairs for every
    import/inference string that should parse as Python."""
    data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []

    for fw, info in data.items():
        if fw.startswith("_"):
            continue

        # framework-level import lines
        for i, line in enumerate(info.get("import", [])):
            pairs.append((f"{fw}/import[{i}]", line))

        # per-version inference lines (plain + arch-specific)
        for ver, vinfo in info.get("versions", {}).items():
            for key in ("inference", "inference_sm86", "inference_sm89"):
                for i, line in enumerate(vinfo.get(key, [])):
                    pairs.append((f"{fw}/{ver}/{key}[{i}]", line))

    return pairs


# Build the parametrize list at collection time (no GPU/conda needed — it's
# just JSON + stdlib ast).
_CODE_LINES = _collect_code_lines()


@pytest.mark.parametrize("label,code", _CODE_LINES, ids=[l for l, _ in _CODE_LINES])
def test_inference_line_parses(label: str, code: str):
    """Each import/inference line in models.json must parse as valid Python.

    We expand the ${OH_MY_MLIP_HOME} placeholder to a dummy path so the ast
    parser sees a syntactically valid string literal instead of a bare `$`.
    """
    # Substitute the placeholder so ast.parse sees a normal string literal.
    normalised = code.replace("${OH_MY_MLIP_HOME}", "/dummy/home").replace(
        "$OH_MY_MLIP_HOME", "/dummy/home"
    )
    try:
        ast.parse(normalised, mode="exec")
    except SyntaxError as exc:
        pytest.fail(
            f"Syntax error in models.json {label!r}:\n"
            f"  code: {code!r}\n"
            f"  error: {exc}"
        )


def test_all_models_have_at_least_one_inference_line():
    """Every version must expose at least one inference string (plain or arch)."""
    data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    missing: list[str] = []
    for fw, info in data.items():
        if fw.startswith("_"):
            continue
        for ver, vinfo in info.get("versions", {}).items():
            has_inference = bool(
                vinfo.get("inference")
                or vinfo.get("inference_sm86")
                or vinfo.get("inference_sm89")
            )
            if not has_inference:
                missing.append(f"{fw}/{ver}")
    assert not missing, "Versions with no inference string:\n" + "\n".join(missing)
