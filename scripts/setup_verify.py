#!/usr/bin/env python3
"""setup_verify.py -- atomic verify oracle for one model (exit 0 iff pass).

ONE command answers "is this model actually computing, and on what device?"
with ONE JSON verdict. The setup skill renders this verdict; it never re-runs
nvidia-smi, re-judges "both conditions", or classifies stderr itself.

Flow (deterministic, decided BEFORE any run):
  0. Preflight: ``predict_driver_skew(env)`` (shared core; the numeric twin of
     install.sh's ``warn_driver_skew``) chooses the device. Skew predicted ->
     the run targets ``--device cpu`` and CPU success is an EXPECTED-DEGRADED
     pass. No skew -> the run targets cuda and a descendant GPU PID is
     REQUIRED for pass.
  1. Launch ``run_examples/single_point.py <model> --json`` (via
     ``sys.executable`` -- run() spawns the env worker itself) under the
     shared-core ``stream_process`` with GPU-PID sampling. The compute PID is
     a Worker GRANDCHILD (oh_my_mlip.provider spawns a per-env worker), so
     attribution walks /proc parent chains -- direct PID equality never works.
  2. Emit the verdict:
     {pass, device, degraded, reason, energy_ev, fmax_ev_a, forces_shape,
      gpu_pid_confirmed}

Exit code: 0 iff ``pass`` -- a DELIBERATE divergence from the
survey/guardrail always-exit-0 convention: verify is test-like (its exit IS
the fact), and the sweep driver records returncodes rather than trusting
them. No stderr classification exists anywhere in this oracle: any nonzero
exit on the chosen device is a plain fail with the normalized stderr tail as
the reason (normalization imported from setup_guardrail -- never copied).

Usage:
  python3 scripts/setup_verify.py MACE --json
  python3 scripts/setup_verify.py TACE --structure POSCAR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup_common import (  # noqa: E402
    predict_driver_skew,
    resolve_home,
    stream_process,
)
from setup_guardrail import _normalize_line  # noqa: E402

STDERR_TAIL_LINES = 20


def find_env(model: str, home: Path) -> str | None:
    """Map a family or version name to its env via models.json (registry-only read)."""
    registry = json.loads((home / "models.json").read_text())
    want = model.lower()
    for family, spec in registry.items():
        if family.startswith("_"):
            continue
        if family.lower() == want:
            return spec["env"]
        for version in (spec.get("versions") or {}):
            if version.lower() == want:
                return spec["env"]
    return None


def parse_witness_json(stdout: str) -> dict | None:
    """Last stdout line that parses as the single_point --json object."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "energy_ev" in obj:
            return obj
    return None


def normalized_tail(stderr: str, lines: int = STDERR_TAIL_LINES) -> str:
    tail = [ln for ln in stderr.splitlines() if ln.strip()][-lines:]
    return "\n".join(_normalize_line(ln) for ln in tail)


def decide_verdict(
    skew: dict,
    returncode: int,
    gpu_seen: bool,
    witness: dict | None,
    stderr: str,
) -> dict:
    """Pure verdict assembly -- the whole decision table, unit-testable."""
    verdict = {
        "pass": False,
        "device": "cpu" if skew["skew"] else "cuda",
        "degraded": bool(skew["skew"]),
        "reason": "",
        "energy_ev": witness.get("energy_ev") if witness else None,
        "fmax_ev_a": witness.get("fmax_ev_a") if witness else None,
        "forces_shape": witness.get("forces_shape") if witness else None,
        "gpu_pid_confirmed": bool(gpu_seen),
    }
    if returncode != 0:
        verdict["reason"] = normalized_tail(stderr) or f"exit {returncode}"
        return verdict
    if witness is None:
        verdict["reason"] = "witness_json_missing"
        return verdict
    if skew["skew"]:
        verdict["pass"] = True
        verdict["reason"] = skew["reason"]
        return verdict
    if not gpu_seen:
        verdict["reason"] = "gpu_not_used"
        return verdict
    verdict["pass"] = True
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="framework or version name from models.json")
    ap.add_argument("--structure", default=None, help="structure file passed through to single_point.py")
    ap.add_argument("--json", action="store_true", help="print the verdict as JSON (agent path)")
    args = ap.parse_args()

    home = resolve_home()
    env_name = find_env(args.model, home)
    if env_name is None:
        verdict = {"pass": False, "reason": f"unknown model: {args.model}"}
        print(json.dumps(verdict) if args.json else f"FAIL: {verdict['reason']}")
        return 1

    skew = predict_driver_skew(env_name, home)
    command = [sys.executable, str(home / "run_examples" / "single_point.py"), args.model, "--json"]
    if skew["skew"]:
        command += ["--device", "cpu"]
    if args.structure:
        command += ["--structure", args.structure]

    log_dir = home / ".sweep" / "verify"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OH_MY_MLIP_HOME=str(home))
    rc, _elapsed, stdout, stderr, gpu = stream_process(
        command,
        env=env,
        log_path=log_dir / f"{env_name}.log",
        stderr_path=log_dir / f"{env_name}.stderr.log",
        collect=True,
        monitor_gpu=not skew["skew"],
        cwd=home,
        gpu_sample_seconds=0.2,
    )

    verdict = decide_verdict(skew, rc, bool(gpu.get("seen")), parse_witness_json(stdout), stderr)
    if args.json:
        print(json.dumps(verdict))
    else:
        state = "PASS" if verdict["pass"] else "FAIL"
        extra = " (degraded: cpu)" if verdict["degraded"] and verdict["pass"] else ""
        print(f"{state}{extra} device={verdict['device']} energy={verdict['energy_ev']} "
              f"gpu_pid_confirmed={verdict['gpu_pid_confirmed']} reason={verdict['reason'] or '-'}")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
