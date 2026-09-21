"""A failed Worker must not leave debris, and a dead pipe must trigger the
respawn WorkerPool promises.

Two holes: a start that failed after spawning raised and walked away, leaving
the child running (often with a model already on the GPU) and nobody to reap
it; and writing to a child that had just died raised BrokenPipeError out of
request(), while WorkerPool respawns only on the {"error": "worker crashed"}
dict -- so the one failure the respawn exists for skipped it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oh_my_mlip import provider  # noqa: E402

NOT_READY = (
    "import sys, time\n"
    "print('{\"ready\": false, \"error\": \"model load failed\"}', flush=True)\n"
    "time.sleep(60)\n"
)


def test_a_failed_start_terminates_the_child(monkeypatch):
    spec = {"python": sys.executable, "env": "fake-env", "env_run": {}, "model": "Fake"}
    monkeypatch.setattr(provider.registry, "resolve", lambda *a, **k: dict(spec))
    worker = provider.Worker("Fake")
    monkeypatch.setattr(worker, "_build_cmd", lambda: [sys.executable, "-c", NOT_READY])
    with pytest.raises(provider.WorkerError, match="model load failed"):
        worker.start()
    deadline = time.time() + 10
    while worker._proc.poll() is None and time.time() < deadline:
        time.sleep(0.05)
    assert worker._proc.poll() is not None, "the child outlived its failed start"


class _DeadPipe:
    def write(self, _):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        raise BrokenPipeError(32, "Broken pipe")


class _Proc:
    def __init__(self, stdin):
        self.stdin, self.stdout, self.stderr = stdin, None, None

    def poll(self):
        return None


def _worker_with(stdin, monkeypatch) -> provider.Worker:
    spec = {"python": sys.executable, "env": "fake-env", "env_run": {}, "model": "Fake"}
    monkeypatch.setattr(provider.registry, "resolve", lambda *a, **k: dict(spec))
    worker = provider.Worker("Fake")
    worker._proc = _Proc(stdin)
    return worker


def test_writing_to_a_dead_child_is_a_crash_not_an_exception(monkeypatch):
    worker = _worker_with(_DeadPipe(), monkeypatch)
    resp = worker.request({"symbols": ["Cu"], "positions": [[0, 0, 0]]}, request_id=1)
    assert resp == {"id": 1, "ok": False, "error": "worker crashed"}


def test_the_pool_respawns_after_a_broken_pipe(monkeypatch):
    spec = {"python": sys.executable, "env": "fake-env", "env_run": {}, "model": "Fake"}
    monkeypatch.setattr(provider.registry, "resolve", lambda *a, **k: dict(spec))
    started = []

    class FakeWorker:
        def __init__(self, model, **kw):
            self.n = len(started)
            started.append(self)
            self.alive = True

        def start(self):
            return self

        def request(self, atoms, properties, request_id=None):
            if self.n == 0:                    # the first child died under us
                self.alive = False
                return {"id": request_id, "ok": False, "error": "worker crashed"}
            return {"id": request_id, "ok": True, "results": {"energy": -1.0}}

        def shutdown(self):
            pass

    pool = provider.WorkerPool(worker_factory=FakeWorker)
    resp = pool.request("Fake", {"symbols": ["Cu"], "positions": [[0, 0, 0]]}, request_id=7)
    assert resp["ok"] and resp["results"]["energy"] == -1.0
    assert len(started) == 2, "the pool did not respawn after the crash"
