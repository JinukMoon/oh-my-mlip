"""GPU-free check that catbench_quickstart skips models whose env is not built.

A fresh clone has no `$OH_MY_MLIP_HOME/envs/<env>/bin/python` built yet. The
roster runner must detect that BEFORE dispatching a subprocess (otherwise the
missing interpreter raises a raw FileNotFoundError that crashes the whole
roster) and instead print a loud, actionable `install.sh <model>` skip.

This test imports the example by file path (it is a script, not a package),
mocks the interpreter-presence check, and asserts the skip-path message fires —
no subprocess, no conda env, no GPU.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "run_examples" / "catbench_quickstart.py"


def _load_quickstart():
    """Import run_examples/catbench_quickstart.py as a module by path."""
    spec = importlib.util.spec_from_file_location("catbench_quickstart", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_env_ready_false_when_interpreter_absent(tmp_path):
    qs = _load_quickstart()
    spec = {"env": "mace", "python": str(tmp_path / "envs" / "mace" / "bin" / "python")}
    assert qs._env_ready(spec) is False


def test_env_ready_true_with_sentinel(tmp_path):
    qs = _load_quickstart()
    env_root = tmp_path / "envs" / "mace"
    (env_root / "bin").mkdir(parents=True)
    (env_root / ".omm_ready").write_text("")
    spec = {"env": "mace", "python": str(env_root / "bin" / "python")}
    assert qs._env_ready(spec) is True


def test_skip_message_when_env_missing(monkeypatch, capsys):
    """With no env materialized, main() must print an actionable skip on stderr
    (mentioning the env name and `install.sh <model>`) and exit non-zero without
    dispatching any subprocess."""
    qs = _load_quickstart()

    # No real dataset / interpreter / subprocess involved.
    monkeypatch.setattr(qs, "_discover_tag", lambda explicit: "demo")
    monkeypatch.setattr(qs, "list_models", lambda: ["MACE"])
    fake_spec = {
        "env": "mace",
        "version": "MACE-MPA-0",
        "python": "/nonexistent/envs/mace/bin/python",
    }
    monkeypatch.setattr(qs, "_resolve_versions_for", lambda model, pins: [fake_spec])
    # Interpreter is absent -> not ready.
    monkeypatch.setattr(qs, "_env_ready", lambda spec: False)

    # A dispatch attempt would mean the skip path failed; make it loud.
    def _boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("dispatched a subprocess despite missing env")

    monkeypatch.setattr(qs, "_run_one_model", _boom)

    monkeypatch.setattr(qs.sys, "argv", ["catbench_quickstart.py", "demo", "--only", "MACE"])
    rc = qs.main()

    err = capsys.readouterr().err
    assert rc != 0
    assert "[skip]" in err
    assert "mace" in err
    assert "install.sh MACE" in err
