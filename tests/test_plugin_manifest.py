"""The Claude Code plugin manifest: skills resolve, and the version moves with what the plugin carries."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import plugin_version as pv  # noqa: E402

MANIFEST = json.loads((REPO / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))


def test_every_listed_skill_resolves():
    for entry in MANIFEST["skills"]:
        assert (REPO / entry / "SKILL.md").is_file(), entry


def test_version_is_bumped_whenever_skills_or_agents_change():
    # Claude Code keeps users on this version until it changes (plugins-reference, `version`)
    assert MANIFEST["version"].startswith(pv.PREFIX)
    assert MANIFEST["metadata"]["content_sha256"] == pv.content_sha256(), (
        "skills/ or AGENTS.md changed without a plugin version bump: "
        "run python3 scripts/plugin_version.py --bump")


def test_next_version_counts_same_day_bumps():
    import datetime as dt
    day = dt.date(2026, 9, 28)
    assert pv.next_version(None, day) == "skills-2026.09.28"
    assert pv.next_version("skills-2026.09.27", day) == "skills-2026.09.28"
    assert pv.next_version("skills-2026.09.28", day) == "skills-2026.09.28.2"
    assert pv.next_version("skills-2026.09.28.2", day) == "skills-2026.09.28.3"
