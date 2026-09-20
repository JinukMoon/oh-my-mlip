"""The worker subprocess must find oh_my_mlip from any working directory.

`<env>/bin/python -m oh_my_mlip._worker` imports this clone, which the model
envs do not have installed, and a parent-side sys.path.insert never reaches a
subprocess. Without the hub root on the child's PYTHONPATH, run()/Worker only
worked with the clone as cwd -- exactly the "use it from your own folder" path
docs/howto/use-a-model.md advertises.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oh_my_mlip import provider, registry  # noqa: E402


def test_child_env_carries_the_hub_root_on_pythonpath(monkeypatch):
    spec = {"python": sys.executable, "env": "fake-env", "env_run": {}, "model": "Fake"}
    monkeypatch.setattr(provider.registry, "resolve", lambda *a, **k: dict(spec))
    child_env = provider.Worker("Fake")._build_env()
    assert str(registry.home()) in child_env["PYTHONPATH"].split(os.pathsep)


def test_an_existing_pythonpath_is_kept(monkeypatch):
    spec = {"python": sys.executable, "env": "fake-env", "env_run": {}, "model": "Fake"}
    monkeypatch.setattr(provider.registry, "resolve", lambda *a, **k: dict(spec))
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    parts = provider.Worker("Fake")._build_env()["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(registry.home()) and "/somewhere/else" in parts


def test_a_child_started_from_another_cwd_imports_oh_my_mlip(tmp_path, monkeypatch):
    spec = {"python": sys.executable, "env": "fake-env", "env_run": {}, "model": "Fake"}
    monkeypatch.setattr(provider.registry, "resolve", lambda *a, **k: dict(spec))
    env = provider.Worker("Fake")._build_env()
    proc = subprocess.run(
        [sys.executable, "-c", "import oh_my_mlip._worker as w; print(w.__name__)"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "oh_my_mlip._worker" in proc.stdout
