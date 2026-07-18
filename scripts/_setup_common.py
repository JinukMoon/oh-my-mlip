"""Shared deterministic core for the setup harness (stdlib-only, GPU-free).

Single home for the process/GPU/parse machinery that the setup oracle
(`setup_verify.py`), the sweep driver (`setup_sweep.py`), and the local sweep
(`sweep_local.py`) all build on. Extracted verbatim from sweep_local.py (the
2026-06-27 18/20 sweep is the provenance of this code) with only injection
points added for testability: `cwd`, `gpu_sample_seconds`, `proc_root`,
`host_cuda`.

Also the single home of:
  * ``resolve_home()`` — OH_MY_MLIP_HOME / OMM_HOME / repo-root resolution
    (moved from setup_survey.py).
  * ``predict_driver_skew()`` — the PREFLIGHT numeric driver-skew predicate.
    It implements the SAME comparison as install.sh's ``warn_driver_skew``
    (recipe ``torch==X.Y.Z+cuNNN`` pin vs the CUDA version the host driver
    exposes), including its early-return semantics: no pinned +cuNNN wheel,
    no nvidia-smi, or unparseable host CUDA all mean "no skew" — GPU is
    attempted and failures surface honestly. A CI test asserts this predicate
    and install.sh's bash implementation agree on representative pairs.

GPU attribution note: the process that actually touches the GPU is a Worker
GRANDCHILD of whatever command we launch (oh_my_mlip.provider spawns a
persistent per-env worker), so GPU PID matching MUST walk /proc parent chains
(`is_descendant_or_self`) — matching the launched PID directly never succeeds.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def command_display(command: Iterable[str]) -> str:
    return " ".join(str(part) for part in command)


def resolve_home() -> Path:
    home = os.environ.get("OH_MY_MLIP_HOME") or os.environ.get("OMM_HOME")
    if home and Path(home).is_dir():
        return Path(home).resolve()
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# /proc walking + GPU attribution (verbatim from sweep_local.py, proc_root
# injectable for fake-/proc unit tests)
# ---------------------------------------------------------------------------

def read_proc_text(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data.replace(b"\x00", b" ").decode("utf-8", errors="replace")


def is_python_pid(pid: int, proc_root: Path = Path("/proc")) -> bool:
    comm = read_proc_text(proc_root / str(pid) / "comm").strip().lower()
    cmdline = read_proc_text(proc_root / str(pid) / "cmdline").lower()
    blob = f"{comm} {cmdline}"
    return "python" in blob


def parent_pid(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    try:
        stat = (proc_root / str(pid) / "stat").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None
    end = stat.rfind(")")
    if end == -1:
        return None
    rest = stat[end + 2 :].split()
    if len(rest) < 2:
        return None
    try:
        return int(rest[1])
    except ValueError:
        return None


def is_descendant_or_self(
    pid: int, root_pid: int, proc_root: Path = Path("/proc")
) -> bool:
    current = pid
    seen: set[int] = set()
    while current > 1 and current not in seen:
        if current == root_pid:
            return True
        seen.add(current)
        parent = parent_pid(current, proc_root)
        if parent is None:
            return False
        current = parent
    return False


def parse_mem_mb(text: str) -> int:
    match = re.search(r"(\d+)", text)
    return int(match.group(1)) if match else 0


def sample_gpu(
    root_pid: int,
    gpu: dict,
    *,
    cwd: Path | None = None,
    proc_root: Path = Path("/proc"),
) -> None:
    if shutil.which("nvidia-smi") is None:
        return
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
            ],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return
    if proc.returncode != 0:
        return
    for row in csv.reader(proc.stdout.splitlines()):
        if len(row) < 2:
            continue
        try:
            pid = int(row[0].strip())
        except ValueError:
            continue
        if not is_descendant_or_self(pid, root_pid, proc_root):
            continue
        if not is_python_pid(pid, proc_root):
            continue
        mem = parse_mem_mb(row[1])
        gpu["seen"] = True
        gpu["max_mem_mb"] = max(int(gpu.get("max_mem_mb", 0)), mem)
        samples = gpu.setdefault("samples", [])
        if len(samples) < 10:
            samples.append({"pid": pid, "used_memory_mb": mem})


# ---------------------------------------------------------------------------
# Streaming subprocess runner with optional GPU monitoring (verbatim from
# sweep_local.py; cwd + sample interval injectable)
# ---------------------------------------------------------------------------

def stream_process(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    stderr_path: Path,
    collect: bool = False,
    monitor_gpu: bool = False,
    cwd: Path | None = None,
    gpu_sample_seconds: float = 0.5,
) -> tuple[int, float, str, str, dict]:
    start = time.monotonic()
    gpu = {"seen": False, "max_mem_mb": 0, "samples": []}
    collected_stdout: list[str] = []
    collected_stderr: list[str] = []
    stop_monitor = threading.Event()
    write_lock = threading.Lock()

    with log_path.open("a", encoding="utf-8", errors="replace") as log_file, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr_file:
        log_file.write(f"\n[{utc_now()}] $ {command_display(command)}\n")
        log_file.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            bufsize=1,
        )

        def emit(line: str, *, is_stderr: bool) -> None:
            with write_lock:
                log_file.write(line)
                log_file.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
                if is_stderr:
                    stderr_file.write(line)
                    stderr_file.flush()
                if collect:
                    if is_stderr:
                        collected_stderr.append(line)
                    else:
                        collected_stdout.append(line)

        def pump(pipe, *, is_stderr: bool) -> None:
            try:
                for line in pipe:
                    emit(line, is_stderr=is_stderr)
            finally:
                pipe.close()

        def monitor() -> None:
            while not stop_monitor.is_set():
                sample_gpu(proc.pid, gpu, cwd=cwd)
                stop_monitor.wait(gpu_sample_seconds)
            sample_gpu(proc.pid, gpu, cwd=cwd)

        threads = [
            threading.Thread(target=pump, args=(proc.stdout,), kwargs={"is_stderr": False}),
            threading.Thread(target=pump, args=(proc.stderr,), kwargs={"is_stderr": True}),
        ]
        for thread in threads:
            thread.daemon = True
            thread.start()

        monitor_thread = None
        if monitor_gpu:
            monitor_thread = threading.Thread(target=monitor)
            monitor_thread.daemon = True
            monitor_thread.start()

        rc = proc.wait()
        stop_monitor.set()
        for thread in threads:
            thread.join()
        if monitor_thread is not None:
            monitor_thread.join()
        elapsed = time.monotonic() - start
        log_file.write(f"[{utc_now()}] returncode={rc} seconds={elapsed:.1f}\n")
        log_file.flush()

    return (
        rc,
        elapsed,
        "".join(collected_stdout),
        "".join(collected_stderr),
        gpu,
    )


def verify_output_ok(stdout: str, stderr: str) -> tuple[bool, bool]:
    combined = stdout + "\n" + stderr
    energy_ok = re.search(r"energy\s*\(eV\)\s*:\s*[-+0-9.eE]+", combined) is not None
    force_ok = re.search(r"max\|force\|\s*:\s*[-+0-9.eE]+", combined) is not None
    return energy_ok, force_ok


# ---------------------------------------------------------------------------
# Preflight driver-skew predicate (numeric twin of install.sh warn_driver_skew)
# ---------------------------------------------------------------------------

# Same extraction as install.sh:201 — torch==X.Y.Z+cuNNN in the env recipe.
_RECIPE_CU = re.compile(r"torch==[0-9.]+\+cu([0-9]+)")
# Same extraction as install.sh:205 — "CUDA Version: 12.9" from nvidia-smi.
_HOST_CU = re.compile(r"CUDA Version: ([0-9]+)\.([0-9]+)")


def parse_recipe_cu(recipe_text: str) -> int | None:
    match = _RECIPE_CU.search(recipe_text)
    return int(match.group(1)) if match else None


def parse_host_cuda(nvidia_smi_text: str) -> int | None:
    match = _HOST_CU.search(nvidia_smi_text)
    if not match:
        return None
    return int(match.group(1)) * 10 + int(match.group(2))


def _probe_host_cuda() -> int | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return parse_host_cuda(proc.stdout)


def predict_driver_skew(
    env_name: str,
    home: Path | None = None,
    host_cuda: int | None = None,
) -> dict:
    """Decide BEFORE running whether GPU inference can work in this env.

    Early-return semantics inherit install.sh warn_driver_skew exactly:
    a CPU-only recipe (no ``+cuNNN`` pin), a host without nvidia-smi, or an
    unparseable host CUDA version all yield ``skew: False`` — the GPU is
    attempted and any failure surfaces honestly; degradation is never
    silent. ``host_cuda`` is injectable for tests (units of major*10+minor).
    """
    home = home or resolve_home()
    recipe = home / "envs" / f"{env_name}.yml"
    result = {"skew": False, "env_cuda": None, "host_cuda": None, "reason": ""}
    try:
        env_cu = parse_recipe_cu(recipe.read_text(encoding="utf-8"))
    except OSError:
        env_cu = None
    if env_cu is None:
        return result
    result["env_cuda"] = env_cu
    if host_cuda is None:
        host_cuda = _probe_host_cuda()
    if host_cuda is None:
        return result
    result["host_cuda"] = host_cuda
    if env_cu > host_cuda:
        result["skew"] = True
        result["reason"] = (
            f"env needs CUDA {env_cu // 10}.{env_cu % 10}, "
            f"host driver exposes CUDA {host_cuda // 10}.{host_cuda % 10}"
        )
    return result
