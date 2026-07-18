#!/usr/bin/env python3
"""Run the local oh-my-mlip environment install/verify/delete sweep.

The sweep is intentionally conservative:
  * one env at a time, in scripts/sweep_config.json order
  * install proof is not enough; run_examples/single_point.py is the oracle
  * verification must print energy and force and show a Python GPU process
  * every env prefix is removed after its attempt
  * scoped cache cleanup is delegated to scripts/setup_guardrail.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "scripts" / "sweep_config.json"
MODELS_JSON = ROOT / "models.json"
GUARDRAIL = ROOT / "scripts" / "setup_guardrail.py"
INSTALL_SH = ROOT / "install.sh"
SINGLE_POINT = ROOT / "run_examples" / "single_point.py"

SWEEP_DIR = ROOT / ".sweep"
LOG_DIR = SWEEP_DIR / "logs"
STATE_DIR = SWEEP_DIR / "state"
RESULTS_JSONL = SWEEP_DIR / "results.jsonl"
RESULTS_MD = SWEEP_DIR / "results.md"
HF_TOKEN_FILE = ROOT / "huggging_token"
MINICONDA_BIN = Path("/home/jumoon/miniconda3/bin")

MAX_REPEAT = 2
# Bounded self-heal stop set (mirrors setup_guardrail.py defaults; AGENTS.md section 8):
# cumulative attempt cap is signature-agnostic (bounds "different strategy each round"),
# wall-clock cap is per-install elapsed (None = off by default).
CUMULATIVE_MAX = 5
WALLCLOCK_MAX_S = None
GPU_SAMPLE_SECONDS = 0.5

# Shared deterministic core (extracted 2026-07-18): the proc/GPU/parse
# machinery lives in scripts/_setup_common.py so the setup oracle, the sweep
# driver, and this sweep share ONE implementation.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup_common import (  # noqa: E402
    command_display,
    is_descendant_or_self,
    is_python_pid,
    parent_pid,
    parse_mem_mb,
    read_proc_text,
    utc_now,
    verify_output_ok,
)
from _setup_common import sample_gpu as _shared_sample_gpu  # noqa: E402
from _setup_common import stream_process as _shared_stream_process  # noqa: E402


def sample_gpu(root_pid: int, gpu: dict) -> None:
    _shared_sample_gpu(root_pid, gpu, cwd=ROOT)


def stream_process(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    stderr_path: Path,
    collect: bool = False,
    monitor_gpu: bool = False,
) -> tuple[int, float, str, str, dict]:
    return _shared_stream_process(
        command,
        env=env,
        log_path=log_path,
        stderr_path=stderr_path,
        collect=collect,
        monitor_gpu=monitor_gpu,
        cwd=ROOT,
        gpu_sample_seconds=GPU_SAMPLE_SECONDS,
    )


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dirs() -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def sweep_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OH_MY_MLIP_HOME", str(ROOT))
    if MINICONDA_BIN.is_dir():
        current_path = env.get("PATH", "")
        parts = current_path.split(os.pathsep) if current_path else []
        miniconda = str(MINICONDA_BIN)
        if miniconda not in parts:
            env["PATH"] = os.pathsep.join([miniconda] + parts)
    return env


def read_existing_passes() -> set[str]:
    passed: set[str] = set()
    if not RESULTS_JSONL.exists():
        return passed
    with RESULTS_JSONL.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("install") == "pass" and row.get("verify") == "pass":
                env = row.get("env")
                if isinstance(env, str):
                    passed.add(env)
    return passed


def all_result_rows() -> list[dict]:
    rows: list[dict] = []
    if not RESULTS_JSONL.exists():
        return rows
    with RESULTS_JSONL.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_result(row: dict) -> None:
    ensure_dirs()
    with RESULTS_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    render_results_md()


def one_line(text: object, limit: int = 180) -> str:
    raw = "" if text is None else str(text)
    raw = raw.replace("|", "\\|")
    raw = re.sub(r"\s+", " ", raw).strip()
    if len(raw) <= limit:
        return raw
    return raw[: limit - 3] + "..."


def render_results_md() -> None:
    headers = [
        "family",
        "env",
        "model",
        "install",
        "verify",
        "gpu_mem_MB",
        "seconds",
        "note",
    ]
    lines = [
        "# Local MLIP Env Sweep",
        "",
        f"Updated: {utc_now()}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in all_result_rows():
        values = [
            one_line(row.get("family")),
            one_line(row.get("env")),
            one_line(row.get("model")),
            one_line(row.get("install")),
            one_line(row.get("verify")),
            one_line(row.get("gpu_mem_MB")),
            one_line(row.get("seconds")),
            one_line(row.get("note")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_json(command: list[str], *, env: dict[str, str] | None = None) -> dict:
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    text = proc.stdout.strip()
    if not text:
        return {
            "status": "error",
            "returncode": proc.returncode,
            "stderr": proc.stderr.strip(),
        }
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "returncode": proc.returncode,
            "stdout": text,
            "stderr": proc.stderr.strip(),
        }
    data.setdefault("returncode", proc.returncode)
    return data


def disk_check(ceiling_gb: float, env: dict[str, str]) -> dict:
    return run_json(
        [
            sys.executable,
            str(GUARDRAIL),
            "disk-check",
            "--ceiling-gb",
            str(ceiling_gb),
            "--path",
            str(ROOT),
        ],
        env=env,
    )


def clean_cache(env: dict[str, str]) -> dict:
    log("Running scoped cache cleanup via setup_guardrail.py clean-cache")
    data = run_json([sys.executable, str(GUARDRAIL), "clean-cache"], env=env)
    freed = data.get("total_bytes_freed")
    if isinstance(freed, int):
        log(f"Scoped cache cleanup freed {freed} bytes")
    return data


def ensure_disk_or_stop(ceiling_gb: float, env: dict[str, str]) -> tuple[bool, str]:
    first = disk_check(ceiling_gb, env)
    if first.get("status") == "ok":
        return True, f"disk ok: {first.get('free_gb')} GB free"

    log(
        "Disk below ceiling "
        f"({first.get('free_gb')} GB free, need {ceiling_gb} GB); cleaning once"
    )
    clean_cache(env)
    second = disk_check(ceiling_gb, env)
    if second.get("status") == "ok":
        return True, f"disk ok after cleanup: {second.get('free_gb')} GB free"
    message = (
        "guardrail_halt: disk free "
        f"{second.get('free_gb')} GB remains below ceiling {ceiling_gb} GB"
    )
    return False, message


def last_lines(path: Path, count: int = 20) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-count:]


def remove_env_prefix(env_name: str, env: dict[str, str]) -> None:
    prefix = ROOT / "envs" / env_name
    if not prefix.exists():
        return
    log(f"Removing env prefix {prefix}")
    conda = shutil.which("conda", path=env.get("PATH"))
    if conda:
        proc = subprocess.run(
            [conda, "env", "remove", "-p", str(prefix), "-y"],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        if proc.returncode == 0 and not prefix.exists():
            return
        log("conda env remove did not fully remove prefix; falling back to rm -rf")
    shutil.rmtree(prefix, ignore_errors=True)


def record_attempt(stderr_file: Path, state_file: Path, env: dict[str, str]) -> dict:
    command = [
        sys.executable,
        str(GUARDRAIL),
        "record-attempt",
        "--state",
        str(state_file),
        "--stderr-file",
        str(stderr_file),
        "--max-repeat",
        str(MAX_REPEAT),
        "--cumulative-max",
        str(CUMULATIVE_MAX),
    ]
    if WALLCLOCK_MAX_S is not None:
        command += ["--wallclock-max-s", str(WALLCLOCK_MAX_S)]
    return run_json(command, env=env)


def looks_gated_error(text: str) -> bool:
    lower = text.lower()
    needles = [
        "gated repo",
        "gated repository",
        "license",
        "401 client error",
        "403 client error",
        "unauthorized",
        "forbidden",
        "repo_not_found",
        "repository not found",
        "hf_token",
        "hugging face token",
        "huggingface token",
        "access to model",
        "access token",
    ]
    return any(needle in lower for needle in needles)


def halt_class(text: str) -> str | None:
    lower = text.lower()
    if "neither 'mamba' nor 'conda'" in lower or "conda: command not found" in lower:
        return "conda_absent"
    if "nvcc" in lower and "not found" in lower and "d3" not in lower:
        return "nvcc_absent"
    return None


def license_url_for_model(model: str) -> str:
    try:
        data = load_json(MODELS_JSON)
    except (OSError, json.JSONDecodeError):
        return ""
    for info in data.values():
        if not isinstance(info, dict):
            continue
        versions = info.get("versions")
        if isinstance(versions, dict):
            version_info = versions.get(model)
            if isinstance(version_info, dict):
                value = version_info.get("license_url")
                return value if isinstance(value, str) else ""
        if info.get("default_version") == model:
            value = info.get("license_url")
            return value if isinstance(value, str) else ""
    return ""


def load_hf_token(env: dict[str, str]) -> str:
    if env.get("HF_TOKEN"):
        return "environment"
    if not HF_TOKEN_FILE.exists():
        return ""
    token = HF_TOKEN_FILE.read_text(encoding="utf-8", errors="replace").strip()
    if not token:
        return ""
    env["HF_TOKEN"] = token
    return str(HF_TOKEN_FILE)


def validate_hf_token(env: dict[str, str]) -> tuple[bool, str]:
    token = env.get("HF_TOKEN", "").strip()
    if not token:
        return False, "HF_TOKEN missing"
    request = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "oh-my-mlip-sweep-local",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if 200 <= response.status < 300:
                return True, "HF_TOKEN validated via Hugging Face whoami"
            return False, f"HF token validation returned HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HF token validation failed: HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"HF token validation failed: {exc}"


def install_env(entry: dict, env: dict[str, str]) -> tuple[str, str, float, list[str]]:
    env_name = entry["env"]
    per_env_log = LOG_DIR / f"{env_name}.log"
    state_file = STATE_DIR / f"{env_name}.guardrail.json"
    total_seconds = 0.0
    attempt = 0

    while True:
        attempt += 1
        stderr_file = LOG_DIR / f"{env_name}.install.attempt{attempt}.stderr.log"
        log(f"Installing env #{attempt} for {env_name}: ./install.sh {env_name}")
        rc, seconds, _, _, _ = stream_process(
            [str(INSTALL_SH), env_name],
            env=env,
            log_path=per_env_log,
            stderr_path=stderr_file,
            collect=False,
            monitor_gpu=False,
        )
        total_seconds += seconds
        sentinel = ROOT / "envs" / env_name / ".omm_ready"
        if rc == 0 and sentinel.exists():
            return "pass", f"installed in {seconds:.1f}s", total_seconds, []

        if rc == 0 and not sentinel.exists():
            stderr_file.write_text(
                "install.sh exited 0 but .omm_ready sentinel was not created\n",
                encoding="utf-8",
            )

        stderr_text = stderr_file.read_text(encoding="utf-8", errors="replace")
        if entry.get("gated") and looks_gated_error(stderr_text):
            url = license_url_for_model(entry.get("verify_model", ""))
            note = "gated fetch failed; accept license and provide a read HF_TOKEN"
            if url:
                note += f" ({url})"
            return "skipped_gated", note, total_seconds, last_lines(stderr_file)

        hclass = halt_class(stderr_text)
        if hclass == "conda_absent":
            return (
                "stalled",
                "halt-and-report: conda/mamba absent; install Miniconda/Miniforge with consent",
                total_seconds,
                last_lines(stderr_file),
            )
        if hclass == "nvcc_absent":
            return (
                "stalled",
                "halt-and-report: nvcc absent; do not install CUDA toolkit automatically",
                total_seconds,
                last_lines(stderr_file),
            )

        verdict = record_attempt(stderr_file, state_file, env)
        vstatus = verdict.get("status")
        if vstatus in ("stalled", "stalled_cumulative", "wallclock_halt"):
            if vstatus == "stalled":
                detail = (
                    f"same install error signature stalled "
                    f"repeat={verdict.get('repeat_count')} signature={verdict.get('signature')}"
                )
            elif vstatus == "stalled_cumulative":
                detail = (
                    f"cumulative attempt cap reached "
                    f"total_attempts={verdict.get('total_attempts')} cumulative_max={verdict.get('cumulative_max')} "
                    f"(signature-agnostic divergence guard)"
                )
            else:  # wallclock_halt
                detail = (
                    f"wall-clock cap reached "
                    f"elapsed_s={verdict.get('elapsed_s')} wallclock_max_s={verdict.get('wallclock_max_s')}"
                )
            return (
                vstatus,
                detail,
                total_seconds,
                last_lines(stderr_file),
            )

        log(
            "Install failed but guardrail allows one more strategy: "
            f"{verdict.get('signature')} repeat={verdict.get('repeat_count')}"
        )
        remove_env_prefix(env_name, env)
        clean_cache(env)


def verify_env(entry: dict, env: dict[str, str]) -> tuple[str, int, float, str, list[str]]:
    env_name = entry["env"]
    model = entry["verify_model"]
    per_env_log = LOG_DIR / f"{env_name}.log"
    stderr_file = LOG_DIR / f"{env_name}.verify.stderr.log"
    log(f"Verifying {env_name} with {sys.executable} run_examples/single_point.py {model}")
    rc, seconds, stdout, stderr, gpu = stream_process(
        [sys.executable, str(SINGLE_POINT), model],
        env=env,
        log_path=per_env_log,
        stderr_path=stderr_file,
        collect=True,
        monitor_gpu=True,
    )
    energy_ok, force_ok = verify_output_ok(stdout, stderr)
    gpu_seen = bool(gpu.get("seen"))
    gpu_mem = int(gpu.get("max_mem_mb") or 0)
    if rc == 0 and energy_ok and force_ok and gpu_seen:
        return "pass", gpu_mem, seconds, "energy+force printed; GPU python PID observed", []
    if rc == 0 and energy_ok and force_ok and not gpu_seen:
        return (
            "cpu_fallback",
            0,
            seconds,
            "energy+force printed, but no descendant Python PID appeared in nvidia-smi",
            last_lines(stderr_file),
        )
    note = (
        f"verify failed rc={rc} energy_ok={energy_ok} "
        f"force_ok={force_ok} gpu_seen={gpu_seen}"
    )
    return "fail", gpu_mem, seconds, note, last_lines(stderr_file)


def row_for_skip(entry: dict, install: str, verify: str, seconds: float, note: str) -> dict:
    return {
        "timestamp": utc_now(),
        "family": entry.get("family", ""),
        "env": entry.get("env", ""),
        "model": entry.get("verify_model", ""),
        "install": install,
        "verify": verify,
        "gpu_mem_MB": 0,
        "seconds": round(seconds, 1),
        "note": note,
    }


def cleanup_entry(env_name: str, env: dict[str, str]) -> None:
    try:
        remove_env_prefix(env_name, env)
    finally:
        clean_cache(env)


def main() -> int:
    ensure_dirs()
    config = load_json(CONFIG_PATH)
    ceiling_gb = float(config.get("host", {}).get("disk_ceiling_gb", 30))
    order = config.get("order", [])
    if not isinstance(order, list):
        raise SystemExit("scripts/sweep_config.json has no order list")

    env = sweep_env()
    token_source = load_hf_token(env)
    if token_source:
        log(f"HF_TOKEN loaded from {token_source}")

    log(f"Starting local sweep in {ROOT}")
    log(f"Results: {RESULTS_JSONL}")
    passed_envs = read_existing_passes()
    if passed_envs:
        log(f"Resuming: already-passed envs will be skipped: {', '.join(sorted(passed_envs))}")

    for index, entry in enumerate(order, start=1):
        env_name = entry.get("env")
        model = entry.get("verify_model")
        family = entry.get("family", "")
        if not env_name or not model:
            log(f"Skipping malformed entry #{index}: {entry}")
            continue
        if env_name in passed_envs:
            log(f"Skipping {env_name}: prior pass recorded in results.jsonl")
            continue

        log(f"===== [{index}/{len(order)}] {family} env={env_name} model={model} =====")
        ok, disk_note = ensure_disk_or_stop(ceiling_gb, env)
        if not ok:
            log(disk_note)
            return 2
        log(disk_note)

        start = time.monotonic()
        last20: list[str] = []
        try:
            if entry.get("gated"):
                token_ok, token_note = validate_hf_token(env)
                if not token_ok:
                    url = license_url_for_model(model)
                    note = f"skipped_gated: {token_note}"
                    if url:
                        note += f"; accept license at {url}"
                    log(note)
                    append_result(row_for_skip(entry, "skipped_gated", "skipped", 0.0, note))
                    continue
                log(token_note)

            install_status, install_note, install_seconds, last20 = install_env(entry, env)
            if install_status != "pass":
                elapsed = time.monotonic() - start
                row = row_for_skip(entry, install_status, "skipped", elapsed, install_note)
                row["last_20_lines"] = last20
                append_result(row)
                log(f"Recorded {env_name}: install={install_status} verify=skipped")
                continue

            verify_status, gpu_mem, verify_seconds, verify_note, verify_last20 = verify_env(entry, env)
            elapsed = install_seconds + verify_seconds
            row = {
                "timestamp": utc_now(),
                "family": family,
                "env": env_name,
                "model": model,
                "install": "pass",
                "verify": verify_status,
                "gpu_mem_MB": gpu_mem,
                "seconds": round(elapsed, 1),
                "note": verify_note,
            }
            if verify_last20:
                row["last_20_lines"] = verify_last20
            append_result(row)
            log(f"Recorded {env_name}: install=pass verify={verify_status} gpu_mem_MB={gpu_mem}")
        finally:
            cleanup_entry(env_name, env)

    log("Sweep complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
