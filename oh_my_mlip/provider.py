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

import collections
import glob
import json
import re
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Iterable

from oh_my_mlip.fetch import ensure_weights
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
        f"  (or by env name: install.sh {env})."
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
    ensure_weights(model, version=spec["version"], spec=spec)

    # exec the import + inference strings in a shared namespace. `device` is
    # exposed so inference lines that reference a `device` variable resolve; the
    # registry lines themselves carry device='cuda' literally today, so this is
    # belt-and-suspenders and lets future rows parameterize on it.
    _check_device_honoured(model, spec["version"], spec["inference"], device)
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
def _killed_hint(model: str, returncode: int | None) -> str:
    """Name the usual cause when a worker died from SIGKILL before it was ready.

    Nothing in the child gets to print anything then, so the stderr above ends
    mid-load and reads like a hang. On Linux the sender is almost always the
    kernel's out-of-memory killer, reached while the checkpoint is loaded into
    host RAM."""
    if returncode not in (-9, 137):
        return ""
    return (
        f"\nThe worker was killed by SIGKILL (exit {returncode}) while loading {model}. "
        "On Linux that is almost always the out-of-memory killer: loading the "
        "checkpoint needed more host RAM than was free (check `dmesg` or "
        "`journalctl -k` for 'Out of memory'). Free memory, run on a machine with "
        "more RAM, or choose a smaller variant of the same family."
    )


class DeviceUnavailableError(RuntimeError):
    """Raised when the caller's device cannot be honoured by a registry line."""


_DEVICE_VAR_RE = re.compile(r"device\s*=\s*device\b")
_DEVICE_LITERAL_RE = re.compile(r"""device\s*=\s*['"](cuda|cpu)['"]""")
_CPU_KWARG_RE = re.compile(r"cpu\s*=\s*(True|False)")


def _device_pin(inference_lines) -> str | None:
    """The device a variant's inference lines fix, or None when the caller's
    `device` reaches the calculator (the line references the variable)."""
    text = "\n".join(inference_lines)
    if _DEVICE_VAR_RE.search(text):
        return None
    match = _DEVICE_LITERAL_RE.search(text)
    if match:
        return match.group(1)
    match = _CPU_KWARG_RE.search(text)
    if match:
        return "cpu" if match.group(1) == "True" else "cuda"
    return "unspecified"


def _check_device_honoured(model: str, version: str, inference_lines, device: str) -> None:
    """Refuse a device the registry line cannot deliver.

    The inference lines are used VERBATIM (that is what makes them trustworthy),
    and today's lines pin the device: `device='cuda'` in most, `cpu=False` in a
    few, nothing at all in others. `device` only entered the exec namespace, so
    asking for cpu ran on the GPU and reported cpu -- the CPU fallback promised
    for old-driver hosts did not exist. Saying so is the honest answer; silently
    running elsewhere is not."""
    pin = _device_pin(inference_lines)
    if pin is None or pin == device:
        return
    # No device argument at all (e.g. SevenNetCalculator('7net-mf-ompa')): the
    # framework picks, and on a working host that is the GPU -- the default path
    # every variant uses today. Only a non-default request is unanswerable here.
    if pin == "unspecified" and device == "cuda":
        return
    if pin == "unspecified":
        detail = ("its inference line carries no device argument, so the framework "
                  "chooses the device itself")
    else:
        detail = f"its inference line pins device {pin!r}"
    raise DeviceUnavailableError(
        f"{model}/{version}: cannot run on {device!r} -- {detail}.\n"
        f"  The hub runs registry inference lines verbatim, so it refuses rather than "
        f"reporting a {device!r} run it did not perform.\n"
        f"  For an old driver, install a matching CUDA build or use a host whose driver "
        f"supports the env's build (docs/host_requirements.md)."
    )


class WorkerError(RuntimeError):
    """Raised when a worker fails to start or dies unexpectedly."""


class Worker:
    """A persistent single-env MLIP worker speaking the frozen JSONL contract.

    Spawns ``<env>/bin/python -m oh_my_mlip._worker`` for one model, applies the
    registry's parsed ``env_run`` as the subprocess environment (NOT shell),
    performs the ready-handshake, then serves ``request(atoms, ...)`` calls.
    Responses are routed by ``id`` (not FIFO), so a single Worker is safe to
    drive from the supervisor.

    Live loop on record: 100 consecutive request() calls against MACE-MPA-0,
    65-atom slab rattled between calls, all ok with no restart -- median
    164.5 ms against 161.3 ms for the same calculator called directly inside
    the env (2026-09-16, RTX 4060 Ti; run by the release session, not re-run
    here). That run does NOT cover the stderr drain below: MACE logs only at
    construction, so the pipe never fills. A model that logs on every call is
    what exercises that path.
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
        # stderr is drained continuously into this bounded buffer; see _start_stderr_drain
        self._stderr_lines: collections.deque = collections.deque(maxlen=400)
        self._stderr_thread: threading.Thread | None = None
        self._python_exe = python_exe or self.spec["python"]
        self._env_override = env

    # -- lifecycle --
    def _build_env(self) -> dict:
        child_env = dict(os.environ)
        # Prepend the env's own bin dir to PATH (conda-activate-equivalent for
        # TOOL resolution only — the interpreter is still invoked by absolute
        # path, never via `conda activate`). Some envs shell out to their own
        # console scripts at runtime: NequIP/Allegro load an AOT .pt2 whose
        # OpenEquivariance extension JIT-loads via torch.utils.cpp_extension,
        # which runs `ninja` from PATH. Without the env bin on PATH that `ninja`
        # is invisible and the load fails ("Ninja is required to load C++
        # extensions"). Prepending the env bin fixes this generally.
        env_bin = os.path.dirname(os.path.expandvars(self._python_exe))
        if env_bin:
            existing = child_env.get("PATH", "")
            child_env["PATH"] = env_bin + (os.pathsep + existing if existing else "")
        # Prepend the env's own lib dir to LD_LIBRARY_PATH (conda-activate-
        # equivalent for the LOADER). On hosts with an old system
        # libstdc++ (e.g. Fedora 36), GRACE/TACE scipy imports need CXXABI_1.3.15, which the old system
        # libstdc++ lacks; the env ships a new-enough libstdc++ but — since we
        # never `conda activate` — it was not on the loader path. env_run still
        # wins below (DPA4/DeePMD's LD_LIBRARY_PATH="" override is applied after
        # this prepend). Only applied when the lib dir actually exists.
        env_lib = os.path.join(os.path.dirname(env_bin), "lib") if env_bin else ""
        if env_lib and os.path.isdir(env_lib):
            existing_ld = child_env.get("LD_LIBRARY_PATH", "")
            child_env["LD_LIBRARY_PATH"] = env_lib + (
                os.pathsep + existing_ld if existing_ld else ""
            )
        # Prepend the env's own pip-wheel CUDA headers (site-packages/nvidia/*/include,
        # e.g. nvidia-cuda-nvrtc-cu12's nvrtc.h) to CPATH. OpenEquivariance JIT-builds
        # its extension at first use and needs nvrtc.h; a fresh env has the header
        # but nothing put it on the include path, so the worker failed to start
        # ("fatal error: nvrtc.h: no such file or directory"). Only existing dirs.
        # The conda CUDA toolkit headers (<env>/targets/<arch>/include, e.g.
        # cuda-crt-dev's crt/host_defines.h that nvrtc/cuda headers #include) are
        # needed by the same JIT build and are not in the pip wheels.
        env_prefix = os.path.dirname(env_bin) if env_bin else ""
        nv_includes = (sorted(glob.glob(os.path.join(
            env_prefix, "lib", "python3*", "site-packages", "nvidia", "*", "include")))
            + sorted(glob.glob(os.path.join(env_prefix, "targets", "*", "include")))) if env_prefix else []
        if nv_includes:
            existing_cpath = child_env.get("CPATH", "")
            child_env["CPATH"] = os.pathsep.join(nv_includes) + (
                os.pathsep + existing_cpath if existing_cpath else ""
            )
        # The same JIT build links -lcuda -lnvrtc; gcc/ld search LIBRARY_PATH for
        # them. The env's conda CUDA libs (unversioned .so and the libcuda link
        # stub) live under targets/<arch>/lib[/stubs]; the real driver libcuda is
        # still what loads at runtime.
        lib_dirs = [d for d in (sorted(glob.glob(os.path.join(env_prefix, "targets", "*", "lib")))
                                + sorted(glob.glob(os.path.join(env_prefix, "targets", "*", "lib", "stubs"))))
                    if os.path.isdir(d)] if env_prefix else []
        if lib_dirs:
            existing_lp = child_env.get("LIBRARY_PATH", "")
            child_env["LIBRARY_PATH"] = os.pathsep.join(lib_dirs) + (
                os.pathsep + existing_lp if existing_lp else ""
            )
        # env_run is already parsed + allowlisted by registry.resolve().
        child_env.update(self.spec["env_run"])
        # The child runs `<env>/bin/python -m oh_my_mlip._worker`, and oh_my_mlip
        # lives in this clone -- which the model env does not have installed. The
        # parent's own sys.path.insert does not reach a subprocess, so without this
        # run()/Worker only worked with the clone as cwd and died with
        # ModuleNotFoundError from any other directory.
        hub_root = registry.home()
        if hub_root:
            existing_pp = child_env.get("PYTHONPATH", "")
            child_env["PYTHONPATH"] = str(hub_root) + (
                os.pathsep + existing_pp if existing_pp else ""
            )
        child_env.setdefault("OH_MY_MLIP_HOME", registry.home())
        if self._env_override:
            child_env.update(self._env_override)
        return child_env

    def _build_cmd(self) -> list[str]:
        cmd = [self._python_exe, "-m", "oh_my_mlip._worker", "--model", self.model]
        if self.version:
            cmd += ["--version", self.version]
        cmd += ["--device", self.device]
        # Arch-pinned models: pass the resolved arch explicitly so the in-env
        # worker resolves the SAME inference variant as the supervisor. Without
        # this the worker re-resolves with arch=None (host auto-detect), and a
        # caller-forced arch (Worker(..., arch="sm86")) would be silently lost.
        if self.spec.get("arch"):
            cmd += ["--arch", self.spec["arch"]]
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
        self._start_stderr_drain()
        line = self._proc.stdout.readline()
        if not line:
            err = self._read_stderr()
            code = self._exit_code()
            self._terminate_child()
            raise WorkerError(
                f"worker for {self.model} produced no handshake; stderr:\n{err}"
                + _killed_hint(self.model, code)
            )
        try:
            handshake = json.loads(line)
        except json.JSONDecodeError as exc:
            self._terminate_child()
            raise WorkerError(f"bad handshake from worker: {line!r}") from exc
        if not handshake.get("ready"):
            self._terminate_child()
            raise WorkerError(
                f"worker for {self.model} failed to start: "
                f"{handshake.get('error')}"
            )
        return self

    def _exit_code(self) -> int | None:
        """The child's exit status once its stdout has closed, or None if it is
        still running (or the Popen double cannot say)."""
        try:
            return self._proc.wait(timeout=5)
        except Exception:
            return None

    def _terminate_child(self) -> None:
        """Stop a child that never became a worker.

        A failed start used to raise and walk away, leaving the child running --
        often after it had already loaded a model onto the GPU -- with nobody
        holding a handle to reap it. Tolerant of test doubles that implement
        only part of Popen."""
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        except Exception:  # pragma: no cover - partial doubles, already-reaped children
            pass

    def _start_stderr_drain(self) -> None:
        """Keep the worker's stderr pipe empty for the life of the process.

        Nobody read this pipe while requests were in flight, and _worker.py
        routes the backend's own stdout into it as well. On a talkative model
        the pipe buffer fills, the child blocks in write(), the supervisor
        blocks in readline(), and neither ever wakes -- a deadlock invisible to
        the mocked tests because a mock never writes enough to fill a pipe. The
        drain keeps only the last lines, which is all an error message needs.
        """
        stream = self._proc.stderr if self._proc is not None else None
        if stream is None:
            return

        def _drain() -> None:
            try:
                for line in stream:
                    line = line.rstrip("\n")
                    self._stderr_lines.append(line)
                    if line.startswith("[oh-my-mlip]"):
                        # the hub's own progress (weight downloads, prepare
                        # steps) is for the user now, not for an error later
                        print(line, file=sys.stderr, flush=True)
            except Exception:  # the pipe closes when the worker exits
                pass

        self._stderr_thread = threading.Thread(
            target=_drain, daemon=True, name=f"omm-stderr-{self.model}"
        )
        self._stderr_thread.start()

    def _read_stderr(self) -> str:
        # Give the drain a moment to collect a dying worker's last words, then
        # report what it buffered. The direct read is the fallback for a stream
        # the drain could not iterate (test doubles).
        thread = self._stderr_thread
        if thread is not None:
            thread.join(timeout=0.5)
        buffered = "\n".join(self._stderr_lines)
        if buffered:
            return buffered
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
            try:
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
            except (BrokenPipeError, OSError, ValueError):
                # The child died between the alive check and the write. Same
                # crash semantics as an empty read: WorkerPool respawns only on
                # the crash dict, so an escaping BrokenPipeError skipped the
                # respawn it exists to provide.
                return {"id": request_id, "ok": False, "error": "worker crashed"}
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
    arch: str | None = None,
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
        arch=arch,
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
