"""Tests for the adopted-env (bring-your-own-env) feature.

Pins the contract across its three surfaces:
  * oh_my_mlip.registry.resolve dispatches to the adopted interpreter when
    env_map.local.json names one, and fails ACTIONABLY on a stale entry;
  * scripts/adopt_env.py refuses an env whose registry imports fail and
    records only verified adoptions;
  * scripts/setup_survey.py counts an adopted env as ready (zero disk).

GPU-free; fake interpreters are tiny shell scripts.
"""
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oh_my_mlip import registry  # noqa: E402

_SURVEY_SPEC = importlib.util.spec_from_file_location(
    "setup_survey", REPO_ROOT / "scripts" / "setup_survey.py"
)
survey_mod = importlib.util.module_from_spec(_SURVEY_SPEC)
_SURVEY_SPEC.loader.exec_module(survey_mod)


def _fake_env(tmp_path: Path, name: str, import_ok: bool = True) -> Path:
    prefix = tmp_path / name
    (prefix / "bin").mkdir(parents=True)
    py = prefix / "bin" / "python"
    py.write_text(f"#!/bin/sh\nexit {0 if import_ok else 1}\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    return prefix


def _home(tmp_path: Path) -> Path:
    (tmp_path / "models.json").write_text(json.dumps({
        "_meta": {},
        "Alpha": {
            "env": "alpha",
            "python": "${OH_MY_MLIP_HOME}/envs/alpha/bin/python",
            "import": ["import alpha"],
            "default_version": "A-1",
            "versions": {"A-1": {"gated": False, "inference": ["calc = None"]}},
        },
    }))
    (tmp_path / "envs").mkdir()
    return tmp_path


def _models(home: Path) -> dict:
    return json.loads((home / "models.json").read_text())


def test_resolve_uses_adopted_interpreter(tmp_path, monkeypatch):
    home = _home(tmp_path)
    prefix = _fake_env(tmp_path, "external_alpha")
    (home / "env_map.local.json").write_text(json.dumps({"alpha": str(prefix)}))
    monkeypatch.setenv("OH_MY_MLIP_HOME", str(home))
    spec = registry.resolve("Alpha", models=_models(home))
    assert spec["python"] == str(prefix / "bin" / "python")


def test_resolve_without_map_uses_hub_prefix(tmp_path, monkeypatch):
    home = _home(tmp_path)
    monkeypatch.setenv("OH_MY_MLIP_HOME", str(home))
    spec = registry.resolve("Alpha", models=_models(home))
    assert spec["python"] == str(home / "envs" / "alpha" / "bin" / "python")


def test_resolve_stale_adoption_is_actionable(tmp_path, monkeypatch):
    home = _home(tmp_path)
    (home / "env_map.local.json").write_text(json.dumps({"alpha": "/nonexistent/prefix"}))
    monkeypatch.setenv("OH_MY_MLIP_HOME", str(home))
    with pytest.raises(registry.RegistryError, match="adopt_env"):
        registry.resolve("Alpha", models=_models(home))


def test_resolve_invalid_map_raises(tmp_path, monkeypatch):
    home = _home(tmp_path)
    (home / "env_map.local.json").write_text("not json")
    monkeypatch.setenv("OH_MY_MLIP_HOME", str(home))
    with pytest.raises(registry.RegistryError, match="valid JSON"):
        registry.resolve("Alpha", models=_models(home))


def _run_adopt(home: Path, *args: str) -> subprocess.CompletedProcess:
    import os
    env = dict(os.environ, OH_MY_MLIP_HOME=str(home))
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "adopt_env.py"), *args],
        capture_output=True, text=True, env=env,
    )


def test_adopt_records_only_verified_envs(tmp_path):
    home = _home(tmp_path)
    good = _fake_env(tmp_path, "good", import_ok=True)
    bad = _fake_env(tmp_path, "bad", import_ok=False)

    refused = _run_adopt(home, "Alpha", str(bad))
    assert refused.returncode == 1 and "REFUSED" in refused.stderr
    assert not (home / "env_map.local.json").exists()

    adopted = _run_adopt(home, "Alpha", str(good))
    assert adopted.returncode == 0, adopted.stderr
    data = json.loads((home / "env_map.local.json").read_text())
    assert data == {"alpha": str(good)}

    removed = _run_adopt(home, "--remove", "alpha")
    assert removed.returncode == 0
    assert json.loads((home / "env_map.local.json").read_text()) == {}


def test_survey_counts_adopted_as_ready(tmp_path):
    home = _home(tmp_path)
    prefix = _fake_env(tmp_path, "external_alpha")
    (home / "env_map.local.json").write_text(json.dumps({"alpha": str(prefix)}))
    result = survey_mod.survey(home, [])
    row = result["envs"][0]
    assert row["state"] == "ready" and row["adopted"] is True
    assert result["to_build"] == []
    assert result["disk"]["budget_gb"] == 0
