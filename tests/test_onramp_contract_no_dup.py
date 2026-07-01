"""Regression guard for the on-ramp contract surfaces (Phase 5 / G006).

The catbench and distill on-ramp contracts are documented in dedicated
"frozen" surfaces (skills/catbench/SKILL.md -> AGENTS.md §3B ->
run_examples/catbench_quickstart.py; docs/distillation_onramp.md). The single
source of truth for agent strategy is AGENTS.md; the on-ramp surfaces must
POINT to it, never re-encode it. This guard asserts there is no duplicated
knowledge block — zero shared 10-grams between each on-ramp surface and
AGENTS.md — so a future edit that copy-pastes strategy into a surface fails CI.

GPU-free, dependency-free.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS = REPO_ROOT / "AGENTS.md"

# On-ramp contract surfaces that must reference (not duplicate) AGENTS.md.
ONRAMP_SURFACES = [
    "skills/catbench/SKILL.md",
    "docs/distillation_onramp.md",
    "docs/catbench_data_format.md",
]
NGRAM = 10


def _ngrams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def test_onramp_surfaces_exist():
    for rel in ONRAMP_SURFACES:
        assert (REPO_ROOT / rel).is_file(), f"missing on-ramp surface: {rel}"
    assert AGENTS.is_file()


def test_no_shared_10gram_with_agents():
    agents_grams = _ngrams(AGENTS.read_text(encoding="utf-8"))
    offenders = {}
    for rel in ONRAMP_SURFACES:
        shared = _ngrams((REPO_ROOT / rel).read_text(encoding="utf-8")) & agents_grams
        if shared:
            offenders[rel] = sorted(" ".join(g) for g in shared)[:5]
    assert not offenders, f"on-ramp surfaces re-encode AGENTS.md (shared 10-grams): {offenders}"
