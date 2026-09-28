"""The one hub resolver (oh_my_mlip/hub.py) and the plugin-cache refusal."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from oh_my_mlip import hub, registry  # noqa: E402


def test_env_var_wins_and_is_used_as_given(monkeypatch, tmp_path):
    monkeypatch.setenv("OH_MY_MLIP_HOME", str(tmp_path / "not-yet"))
    assert hub.resolve_home() == (tmp_path / "not-yet").resolve()
    assert registry.home() == str((tmp_path / "not-yet").resolve())


def test_without_env_var_the_code_clone_is_the_hub(monkeypatch):
    monkeypatch.delenv("OH_MY_MLIP_HOME", raising=False)
    assert hub.resolve_home() == REPO


def test_the_legacy_variable_is_ignored(monkeypatch, tmp_path):
    monkeypatch.delenv("OH_MY_MLIP_HOME", raising=False)
    monkeypatch.setenv("OMM_HOME", str(tmp_path))
    assert hub.resolve_home() == REPO


def test_plugin_cache_paths_are_recognised(tmp_path):
    assert hub.in_plugin_cache(tmp_path / ".claude" / "plugins" / "cache" / "x" / "oh-my-mlip" / "abc")
    assert hub.in_plugin_cache(tmp_path / ".codex" / "plugins" / "oh-my-mlip")
    assert not hub.in_plugin_cache(tmp_path / ".oh-my-mlip")
    assert not hub.in_plugin_cache(tmp_path / "plugins" / ".claude")
    msg = hub.plugin_copy_refusal(tmp_path / ".claude" / "plugins" / "c")
    assert "~/.oh-my-mlip" in msg and "§9.0" in msg
    assert hub.plugin_copy_refusal(tmp_path / "hub") is None


def test_install_sh_and_survey_refuse_a_plugin_cache_home(tmp_path):
    fake = tmp_path / ".claude" / "plugins" / "cache" / "m" / "oh-my-mlip" / "sha"
    fake.mkdir(parents=True)
    env = dict(os.environ, OH_MY_MLIP_HOME=str(fake))
    r = subprocess.run(["bash", str(REPO / "install.sh"), "--status", "mace"], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 2 and "plugin cache, not a hub" in r.stderr
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "setup_survey.py"), "--table", "MACE"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "~/.oh-my-mlip" in r.stderr
