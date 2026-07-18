"""Drift guard: script paths referenced by skill contracts must exist.

The skill contracts (skills/*/SKILL.md) are prose that points at deterministic
scripts — the Deterministic_First split. A rename under scripts/ or
run_examples/ would today fail at agent runtime on a user's machine, not in
CI. This guard makes every referenced path a CI-checked fact.

Forward direction only for now (every referenced path exists). The inverse
guard — every load-bearing script is referenced by at least one skill — is
added once the setup_verify/setup_sweep wiring lands (plan Step 6), so no
commit ever ships a red test.

GPU-free, stdlib-only.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_FILES = sorted(REPO_ROOT.glob("skills/*/SKILL.md"))

# scripts/foo.py, run_examples/bar.py — optionally followed by :NN[-NN] line
# anchors or CLI flags, which are not part of the path.
_REF = re.compile(r"\b(?:scripts|run_examples)/[A-Za-z0-9_\-./]+")


def _extract_refs(text: str) -> set[str]:
    refs = set()
    for match in _REF.findall(text):
        ref = match.rstrip(".")            # sentence-final period
        ref = re.sub(r":[\d\-]+$", "", ref)  # :18-20 line anchors
        if ref.endswith("/"):
            continue                        # bare directory mention
        refs.add(ref)
    return refs


def test_skill_files_exist():
    assert SKILL_FILES, "no skills/*/SKILL.md found"


def test_all_referenced_script_paths_exist():
    missing = []
    for skill in SKILL_FILES:
        for ref in sorted(_extract_refs(skill.read_text())):
            if not (REPO_ROOT / ref).exists():
                missing.append(f"{skill.relative_to(REPO_ROOT)} -> {ref}")
    assert not missing, (
        "skill contracts reference paths that do not exist "
        "(rename drift):\n" + "\n".join(missing)
    )


def test_extractor_strips_anchors_and_punctuation():
    text = (
        "see `scripts/setup_guardrail.py:18-20` and scripts/setup_survey.py. "
        "run run_examples/single_point.py --json now; ignore scripts/ alone"
    )
    assert _extract_refs(text) == {
        "scripts/setup_guardrail.py",
        "scripts/setup_survey.py",
        "run_examples/single_point.py",
    }
