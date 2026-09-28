"""Long MCP tool calls as jobs: start returns at once, job_status polls.

A relaxation or a CatBench run can take minutes to hours, longer than an MCP
call should block. `start()` writes `$OH_MY_MLIP_HOME/.mcp_jobs/<id>/job.json`
and launches `python -m oh_my_mlip.mcp_jobs run <dir>` in its own process
session, so the job outlives the call (and the server) and its whole process
group can be stopped at the time limit. `status()` reads the record back; a
job whose runner died without writing an outcome is reported as failed.

The same pattern as AGENTS.md §9.4 for install.sh: launch detached, then poll.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from oh_my_mlip import registry

LOG_TAIL_LINES = 20


def jobs_root() -> Path:
    return Path(registry.home()) / ".mcp_jobs"


def _write(job_dir: Path, record: dict) -> None:
    tmp = job_dir / "job.json.tmp"
    tmp.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, job_dir / "job.json")


def _read(job_dir: Path) -> dict:
    return json.loads((job_dir / "job.json").read_text(encoding="utf-8"))


def start(kind: str, params: dict, timeout_s: int) -> dict:
    """Record the job and launch its runner detached; returns the job id and paths."""
    job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}-{uuid.uuid4().hex[:8]}"
    job_dir = jobs_root() / job_id
    job_dir.mkdir(parents=True)
    record = {"job_id": job_id, "kind": kind, "params": params, "timeout_s": int(timeout_s),
              "status": "running", "created": time.time(), "log": str(job_dir / "log")}
    _write(job_dir, record)
    env = dict(os.environ, OH_MY_MLIP_HOME=registry.home())
    # the runner starts in the job directory: put this package on its path
    pkg_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with open(job_dir / "log", "ab") as log:
        proc = subprocess.Popen([sys.executable, "-m", "oh_my_mlip.mcp_jobs", "run", str(job_dir)],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                env=env, cwd=str(job_dir), start_new_session=True)
    record["pid"] = proc.pid
    _write(job_dir, record)
    return {"ok": True, "job_id": job_id, "status": "running", "log": record["log"],
            "poll": f"job_status('{job_id}')", "timeout_s": int(timeout_s)}


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:   # a zombie (exited, not yet reaped) is not alive
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
        return state != "Z"
    except OSError:
        return True


def status(job_id: str) -> dict:
    job_dir = jobs_root() / job_id
    if "/" in job_id or not (job_dir / "job.json").is_file():
        return {"ok": False, "error": f"no job {job_id!r} under {jobs_root()}"}
    record = _read(job_dir)
    if record["status"] == "running" and not _alive(record.get("pid")):
        # the runner writes its outcome before exiting; none means it died
        record = _read(job_dir)
        if record["status"] == "running":
            record.update(status="failed", error="the job process exited without recording an outcome; see the log")
            _write(job_dir, record)
    log = Path(record["log"])
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-LOG_TAIL_LINES:] if log.exists() else []
    out = {"ok": record["status"] != "failed", "job_id": job_id, "status": record["status"],
           "log": record["log"], "log_tail": tail, "elapsed_s": round(time.time() - record["created"])}
    for key in ("result", "error"):
        if key in record:
            out[key] = record[key]
    return out


def _run(job_dir: Path) -> None:
    """Inside the detached runner: do the work, record the outcome, honour the limit."""
    record = _read(job_dir)

    def _timeout(signum, frame):
        rec = _read(job_dir)
        rec.update(status="timeout", error=f"stopped after the {record['timeout_s']} s limit")
        _write(job_dir, rec)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.killpg(os.getpgid(0), signal.SIGTERM)       # the catbench jobs this run started go too
        os._exit(124)

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(max(1, int(record["timeout_s"])))
    from oh_my_mlip import mcp_server
    work = {"relax": mcp_server.relax_blocking, "catbench": mcp_server.catbench_blocking}[record["kind"]]
    try:
        result = work(**record["params"])
    except Exception as exc:  # noqa: BLE001 - recorded for job_status
        rec = _read(job_dir)
        rec.update(status="failed", error=f"{exc.__class__.__name__}: {exc}")
        _write(job_dir, rec)
        return
    signal.alarm(0)
    rec = _read(job_dir)
    rec.update(status="done" if result.get("ok", True) else "failed", result=result)
    if not result.get("ok", True) and "error" in result:
        rec["error"] = result["error"]
    _write(job_dir, rec)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        _run(Path(sys.argv[2]))
    else:
        raise SystemExit("usage: python -m oh_my_mlip.mcp_jobs run <job dir>")
