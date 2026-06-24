"""oh_my_mlip.provider — the runnable layers of the tiered teacher-provider.

Layers:

  Layer 2  get_calculator(model, ...) -> ase Calculator   INTRA-ENV ONLY.
  Layer 3  run(model, atoms, ...) -> dict                 CROSS-ENV convenience.
  Layer 4  Worker / WorkerPool                            persistent workers,
           id-routed, one process per env (what a downstream distillation
           tool binds to for bulk teacher labeling).

ase / torch are imported lazily so this module imports on a host with neither
present (the unit tests rely on that). Cross-env work is ALWAYS subprocess +
JSONL protocol — NEVER an in-process import across conda envs.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterable

from oh_my_mlip import registry

__all__ = ["get_calculator", "run", "Worker", "WorkerPool"]


def _env_not_installed_msg(model: str, env: str, python_exe: str) -> str:
    """Actionable message (NOT a raw traceback) for a missing env interpreter.

    Names the exact build command so an agent/user can self-serve. Mirrors the
    hint surfaced by fetch.py / mcp_server.py so every entry point agrees.
    """
    return (
        f"the conda env {env!r} for {model} is not materialized yet "
        f"(interpreter not found: {python_exe}). Install it first:\n"
        f'    bash "$OH_MY_MLIP_HOME/install.sh" {model}\n'
        f"  (or by env name: install.sh {env}), or fetch the prebuilt env via "
        f"oh_my_mlip.fetch.fetch_env({model!r}) / the install_model MCP tool."
    )


# ── Layer 2: get_calculator (INTRA-ENV ONLY) ─────────────────────────────────
def get_calculator(
    model: str,
    version: str | None = None,
    device: str = "cuda",
    apply_d3: bool = False,
    *,
    arch: str | None = None,
):
    """Build and return an ASE Calculator for ``model``.

    PRECONDITION (LOUD): this function exec's the registry `import` + `inference`
    code lines, which import the model's framework (sevenn / mace / nequip /
    fairchem / ...). Those packages live ONLY inside that model's dedicated
    conda env. Therefore **get_calculator MUST be called from within that
    model's interpreter** — i.e. ``<env>/bin/python`` (or via the persistent
    worker, which is launched with exactly that interpreter). Calling it from a
    different env will raise ImportError. For cross-env use, call ``run()`` or
    spawn a ``Worker`` instead; both route to this function inside the correct
    interpreter.

    `apply_d3` wraps the built calculator with catbench's DispersionCorrection
    (``from catbench.dispersion import DispersionCorrection``), matching the
    internal catb_all.py / verify_all.py path.
    """
    spec = registry.resolve(model, version=version, arch=arch)

    # exec the import + inference strings in a shared namespace. `device` is
    # exposed so inference lines that reference a `device` variable resolve; the
    # registry lines themselves carry device='cuda' literally today, so this is
    # belt-and-suspenders and lets future rows parameterize on it.
    ns: dict[str, Any] = {"device": device}
    code = "\n".join(list(spec["imports"]) + list(spec["inference"]))
    exec(compile(code, f"<inference:{model}/{spec['version']}>", "exec"), ns)  # noqa: S102

    if "calc" not in ns:
        raise RuntimeError(
            f"inference for {model}/{spec['version']} did not define `calc`"
        )
    calc = ns["calc"]

    if apply_d3:
        from catbench.dispersion import DispersionCorrection

        calc = DispersionCorrection().apply(calc)
    return calc


# ── Layer 4: persistent Worker (one process per env, id-routed) ──────────────
class WorkerError(RuntimeError):
    """Raised when a worker fails to start or dies unexpectedly."""


class Worker:
    """A persistent single-env MLIP worker speaking the frozen JSONL contract.

    Spawns ``<env>/bin/python -m oh_my_mlip._worker`` for one model, applies the
    registry's parsed ``env_run`` as the subprocess environment (NOT shell),
    performs the ready-handshake, then serves ``request(atoms, ...)`` calls.
    Responses are routed by ``id`` (not FIFO), so a single Worker is safe to
    drive from the supervisor.

    DEFERRED: the 100-call live loop against a real GPU model runs at the
    compute checkpoint. The routing/protocol logic is unit-tested with a mocked
    worker subprocess; it is not exercised against a real model here.
    """

    def __init__(
        self,
        model: str,
        version: str | None = None,
        device: str = "cuda",
        apply_d3: bool = False,
        *,
        arch: str | None = None,
        python_exe: str | None = None,
        env: dict | None = None,
        _popen=subprocess.Popen,
    ):
        self.model = model
        self.version = version
        self.device = device
        self.apply_d3 = apply_d3
        self.spec = registry.resolve(model, version=version, arch=arch)
        self._popen = _popen
        self._proc = None
        self._counter = 0
        self._lock = threading.Lock()
        self._python_exe = python_exe or self.spec["python"]
        self._env_override = env

    # -- lifecycle --
    def _build_env(self) -> dict:
        child_env = dict(os.environ)
        # env_run is already parsed + allowlisted by registry.resolve().
        child_env.update(self.spec["env_run"])
        child_env.setdefault("OH_MY_MLIP_HOME", registry.home())
        if self._env_override:
            child_env.update(self._env_override)
        return child_env

    def _build_cmd(self) -> list[str]:
        cmd = [self._python_exe, "-m", "oh_my_mlip._worker", "--model", self.model]
        if self.version:
            cmd += ["--version", self.version]
        cmd += ["--device", self.device]
        if self.apply_d3:
            cmd.append("--apply-d3")
        return cmd

    def start(self) -> "Worker":
        """Spawn the worker and consume the ready-handshake line.

        If the model's env interpreter is not on disk (the env has not been
        materialized yet), raise a clear, actionable ``WorkerError`` naming the
        exact ``install.sh`` command rather than letting a raw ``FileNotFoundError``
        escape from ``Popen``. This makes the README/AGENTS "actionable message,
        not a traceback" promise true for ``run()`` / ``Worker`` too.
        """
        try:
            self._proc = self._popen(
                self._build_cmd(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=self._build_env(),
            )
        except FileNotFoundError as exc:
            raise WorkerError(
                _env_not_installed_msg(self.model, self.spec["env"], self._python_exe)
            ) from exc
        line = self._proc.stdout.readline()
        if not line:
            err = self._read_stderr()
            raise WorkerError(
                f"worker for {self.model} produced no handshake; stderr:\n{err}"
            )
        try:
            handshake = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorkerError(f"bad handshake from worker: {line!r}") from exc
        if not handshake.get("ready"):
            raise WorkerError(
                f"worker for {self.model} failed to start: "
                f"{handshake.get('error')}"
            )
        return self

    def _read_stderr(self) -> str:
        if self._proc is None or self._proc.stderr is None:
            return ""
        try:
            return self._proc.stderr.read() or ""
        except Exception:  # pragma: no cover
            return ""

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- serve --
    def request(
        self,
        atoms,
        properties: Iterable[str] = ("energy", "forces"),
        *,
        request_id: Any = None,
    ) -> dict:
        """Send one atoms request and return the matched response dict.

        Routes by ``id``: skips/raises if the worker returns a mismatched id
        (the protocol guarantees one response per request, carrying the id).
        Returns ``{"id", "ok", "results"|"error"}``. On worker crash, returns
        ``{"id", "ok": False, "error": "worker crashed"}``.
        """
        if not self.alive:
            raise WorkerError(f"worker for {self.model} is not running")
        from oh_my_mlip._worker import encode_atoms

        with self._lock:
            # Re-check under the lock: shutdown() may have raced us and reaped
            # the process between the alive check above and acquiring the lock.
            # Surface the documented crash dict rather than an AttributeError on
            # a None self._proc.
            if self._proc is None or not self.alive:
                return {"id": request_id, "ok": False, "error": "worker crashed"}
            if request_id is None:
                self._counter += 1
                request_id = self._counter
            req = {
                "id": request_id,
                "atoms": encode_atoms(atoms),
                "properties": list(properties),
            }
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()

            line = self._proc.stdout.readline()
            if not line:
                # Worker died mid-request -> crash semantics.
                return {"id": request_id, "ok": False, "error": "worker crashed"}
            resp = json.loads(line)
            if resp.get("id") != request_id:
                raise WorkerError(
                    f"id mismatch: sent {request_id!r}, got {resp.get('id')!r}"
                )
            return resp

    def shutdown(self) -> None:
        """Send shutdown, close stdin, and reap the process.

        Guarded by ``self._lock`` so it cannot race a concurrent ``request()``
        (which re-checks ``self._proc`` under the same lock).
        """
        with self._lock:
            if self._proc is None:
                return
            try:
                if self.alive and self._proc.stdin:
                    self._proc.stdin.write(json.dumps({"shutdown": True}) + "\n")
                    self._proc.stdin.flush()
                    self._proc.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
            try:
                self._proc.wait(timeout=10)
            except Exception:  # pragma: no cover
                self._proc.kill()
            self._proc = None

    def __enter__(self) -> "Worker":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.shutdown()


class WorkerPool:
    """Supervises one persistent ``Worker`` per (model, version) and routes
    requests to the right env. This is what a downstream distillation tool
    binds to for bulk teacher labeling: it amortizes interpreter+model startup
    across many calls (no subprocess-per-call latency).

    Workers are lazily started on first use and respawned on crash.
    """

    def __init__(
        self,
        device: str = "cuda",
        apply_d3: bool = False,
        *,
        worker_factory=Worker,
    ):
        self.device = device
        self.apply_d3 = apply_d3
        self._worker_factory = worker_factory
        self._workers: dict[tuple[str, str | None], Worker] = {}
        self._lock = threading.Lock()

    def _key(self, model: str, version: str | None) -> tuple[str, str | None]:
        return (model, version)

    def get(self, model: str, version: str | None = None) -> Worker:
        """Return a running Worker for (model, version), starting it if needed."""
        key = self._key(model, version)
        with self._lock:
            worker = self._workers.get(key)
            if worker is None or not worker.alive:
                worker = self._worker_factory(
                    model,
                    version=version,
                    device=self.device,
                    apply_d3=self.apply_d3,
                )
                worker.start()
                self._workers[key] = worker
            return worker

    def request(
        self,
        model: str,
        atoms,
        properties: Iterable[str] = ("energy", "forces"),
        *,
        version: str | None = None,
        request_id: Any = None,
    ) -> dict:
        """Route one request to the model's worker. Respawns once on crash."""
        worker = self.get(model, version)
        resp = worker.request(atoms, properties, request_id=request_id)
        if not resp.get("ok") and resp.get("error") == "worker crashed":
            # respawn and retry once
            worker = self.get(model, version)
            resp = worker.request(atoms, properties, request_id=request_id)
        return resp

    def shutdown(self) -> None:
        with self._lock:
            for worker in self._workers.values():
                worker.shutdown()
            self._workers.clear()

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


# ── Layer 3: run (CROSS-ENV one-shot convenience) ────────────────────────────
def run(
    model: str,
    atoms,
    properties: Iterable[str] = ("energy", "forces"),
    device: str = "cuda",
    apply_d3: bool = False,
    *,
    version: str | None = None,
) -> dict:
    """One-shot cross-env single point: spawn the model's worker, send one
    request, return the ``results`` dict, tear the worker down.

    This is the casual user path (no env management). It uses the SAME persistent
    worker plumbing under the hood — for many calls, prefer a long-lived
    ``Worker``/``WorkerPool`` to avoid per-call startup.

    The parsed ``env_run`` is applied as the subprocess environment, never
    shell-interpolated. Live-GPU execution is deferred to the compute checkpoint;
    the subprocess + protocol plumbing here is correct and unit-tested so it
    works once the envs exist.
    """
    worker = Worker(
        model,
        version=version,
        device=device,
        apply_d3=apply_d3,
    )
    worker.start()
    try:
        resp = worker.request(atoms, properties)
    finally:
        worker.shutdown()
    if not resp.get("ok"):
        raise WorkerError(
            f"run({model}) failed: {resp.get('error', 'unknown error')}"
        )
    return resp["results"]
