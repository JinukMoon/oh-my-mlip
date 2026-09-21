"""A talkative worker must not deadlock the supervisor.

Worker spawns the child with stderr=PIPE, and oh_my_mlip/_worker.py routes the
backend's own stdout there too. Nothing read that pipe while a request was in
flight, so once the OS pipe buffer (64 KiB on Linux) filled, the child blocked
in write() and the supervisor blocked in readline() -- forever. Mocked workers
never write enough to fill a pipe, so only a real child process shows it.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oh_my_mlip import provider  # noqa: E402

# far past the 64 KiB pipe buffer, written BEFORE the handshake
FLOOD = (
    "import sys\n"
    "sys.stderr.write('noisy-backend-log\\n' * 80000)\n"
    "sys.stderr.flush()\n"
    "print('{\"ready\": true}', flush=True)\n"
    "[None for _ in sys.stdin]\n"
)


def _fake_worker(monkeypatch, script: str) -> provider.Worker:
    spec = {"python": sys.executable, "env": "fake-env", "env_run": {}, "model": "Fake"}
    monkeypatch.setattr(provider.registry, "resolve", lambda *a, **k: dict(spec))
    worker = provider.Worker("Fake")
    monkeypatch.setattr(worker, "_build_cmd", lambda: [sys.executable, "-c", script])
    return worker


def test_start_survives_a_worker_that_floods_stderr(monkeypatch):
    worker = _fake_worker(monkeypatch, FLOOD)
    finished, failure = threading.Event(), []

    def run_start():
        try:
            worker.start()
        except Exception as exc:  # pragma: no cover - only on regression
            failure.append(exc)
        finally:
            finished.set()

    threading.Thread(target=run_start, daemon=True).start()
    try:
        assert finished.wait(60), "Worker.start() blocked on a full stderr pipe"
        assert not failure, failure
        assert worker.alive
    finally:
        worker.shutdown()


def test_stderr_of_a_failed_start_is_reported_from_the_drain(monkeypatch):
    # no handshake, just a diagnostic on stderr: the message must still reach the user
    worker = _fake_worker(monkeypatch, "import sys\nsys.stderr.write('boom: cuda missing\\n')\n")
    with pytest.raises(provider.WorkerError, match="boom: cuda missing"):
        worker.start()


def test_a_worker_killed_before_the_handshake_names_the_oom_killer(monkeypatch):
    # what the kernel's OOM killer does to a large checkpoint mid-load: SIGKILL, no last words
    script = "import os, signal, sys\nsys.stderr.write('loading checkpoint\\n')\nsys.stderr.flush()\nos.kill(os.getpid(), signal.SIGKILL)\n"
    worker = _fake_worker(monkeypatch, script)
    with pytest.raises(provider.WorkerError) as info:
        worker.start()
    message = str(info.value)
    assert "loading checkpoint" in message
    assert "SIGKILL" in message and "out-of-memory killer" in message


def test_an_ordinary_failed_start_carries_no_oom_hint(monkeypatch):
    worker = _fake_worker(monkeypatch, "import sys\nsys.stderr.write('boom\\n')\nsys.exit(1)\n")
    with pytest.raises(provider.WorkerError) as info:
        worker.start()
    assert "out-of-memory" not in str(info.value)
