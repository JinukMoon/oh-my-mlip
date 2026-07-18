"""Tests for setup_survey.py — the deterministic plan facts for the setup skill.

The skill contract (skills/setup/SKILL.md) mandates that the survey runs before
any disk judgment and that the disk budget counts ONLY envs that will actually
be built. This suite pins those semantics in code: state classification mirrors
install.sh --status, ready envs cost zero disk, shared envs deduplicate, and
the token probe reports a source name without ever leaking a token value.
"""
import importlib.util
import json
import os
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "setup_survey",
    Path(__file__).resolve().parent.parent / "scripts" / "setup_survey.py",
)
survey_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(survey_mod)


def _make_home(tmp_path: Path) -> Path:
    registry = {
        "_meta": {"description": "test registry"},
        "Alpha": {"env": "alpha", "versions": {"A-1": {"gated": False}}},
        "AlphaTwin": {"env": "alpha", "versions": {"A-2": {"gated": False}}},
        "Beta": {"env": "beta", "versions": {"B-1": {"gated": False}}},
        "Gamma": {"env": "gamma", "versions": {"G-1": {"gated": True}}},
        "Delta": {"env": "delta", "versions": {"D-1": {"gated": False}}},
    }
    (tmp_path / "models.json").write_text(json.dumps(registry))
    envs = tmp_path / "envs"
    # alpha: ready (sentinel present)
    (envs / "alpha").mkdir(parents=True)
    (envs / "alpha" / ".omm_ready").touch()
    # beta: partial (interpreter, no sentinel)
    (envs / "beta" / "bin").mkdir(parents=True)
    beta_py = envs / "beta" / "bin" / "python"
    beta_py.touch()
    beta_py.chmod(0o755)
    # gamma: broken (prefix without interpreter)
    (envs / "gamma").mkdir()
    # delta: not installed (no prefix)
    return tmp_path


def test_states_mirror_install_sh_status(tmp_path):
    result = survey_mod.survey(_make_home(tmp_path), [])
    states = {r["env"]: r["state"] for r in result["envs"]}
    assert states == {
        "alpha": "ready",
        "beta": "partial",
        "gamma": "broken",
        "delta": "not_installed",
    }
    assert result["counts"] == {
        "ready": 1,
        "partial": 1,
        "broken": 1,
        "not_installed": 1,
    }


def test_budget_counts_only_envs_that_will_be_built(tmp_path):
    result = survey_mod.survey(_make_home(tmp_path), [])
    assert sorted(result["to_build"]) == ["beta", "delta", "gamma"]
    assert result["disk"]["budget_gb"] == survey_mod.PER_ENV_GB * 3
    assert isinstance(result["disk"]["fits"], bool)


def test_shared_env_families_deduplicate(tmp_path):
    result = survey_mod.survey(_make_home(tmp_path), [])
    alpha_rows = [r for r in result["envs"] if r["env"] == "alpha"]
    assert len(alpha_rows) == 1
    assert sorted(alpha_rows[0]["families"]) == ["Alpha", "AlphaTwin"]


def test_gated_flag_and_target_filter(tmp_path):
    home = _make_home(tmp_path)
    full = survey_mod.survey(home, [])
    assert full["gated_envs"] == ["gamma"]
    # Filter accepts family or env name, case-insensitively.
    by_family = survey_mod.survey(home, ["Gamma"])
    by_env = survey_mod.survey(home, ["BETA"])
    assert [r["env"] for r in by_family["envs"]] == ["gamma"]
    assert [r["env"] for r in by_env["envs"]] == ["beta"]


def test_token_probe_never_leaks_value(tmp_path, monkeypatch):
    secret = "hf_this-value-must-never-appear"
    monkeypatch.setenv("HF_TOKEN", secret)
    result = survey_mod.survey(_make_home(tmp_path), [])
    assert result["token"] == {"available": True, "source": "HF_TOKEN"}
    assert secret not in json.dumps(result)


def test_no_token_sources_reports_none(tmp_path, monkeypatch):
    for var in ("HF_TOKEN", "HF_TOKEN_PATH", "OMM_HF_TOKEN_FILE"):
        monkeypatch.delenv(var, raising=False)
    # Point HOME at the tmp dir so a real ~/.cache/huggingface/token on the
    # test host cannot satisfy the probe.
    monkeypatch.setenv("HOME", str(tmp_path))
    result = survey_mod.survey(_make_home(tmp_path), [])
    assert result["token"] == {"available": False, "source": "none"}
