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
  python3 scripts/setup_verify.py EquFlash --version EquFlash-v1   # non-default variant
  python3 scripts/setup_verify.py MACE --all-variants --json       # one verdict per variant
  python3 scripts/setup_verify.py MACE --json --no-local-record    # read-only witness

A bare name is resolved family-first (family -> its default_version), so a
variant is only reachable by its own key or via ``--version``. Without the
flag a family/variant name clash silently verifies the DEFAULT variant twice
and reports it as full coverage; test_registry_integrity forbids new clashes.

``--all-variants`` runs every version of the family in turn (each as
``--version <v>``) and prints one verdict line per variant followed by a
summary line ``{"all_variants": true, "pass": <all passed>, "variants":
[...]}``; exit 0 iff every variant passed. It exists so a fresh-install
cycle can cover a family's whole variant list with one user-runnable
command -- it is the oracle looping, not a new judgment.

``--no-local-record``: NEVER write ``models.local.json``,
even on PASS. This is the read-only regression witness for already-adopted
envs: the real hub's local state must hash identical before and after,
so the materialize-on-verify upsert is skipped and the verdict carries
``local_record: "skipped(--no-local-record)"``.
"""
from __future__ import annotations

import argparse
import json
import math
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
    gpu_mem = witness.get("gpu_mem_allocated_bytes") if witness else None
    verdict = {
        "pass": False,
        "device": "cpu" if skew["skew"] else "cuda",
        "degraded": bool(skew["skew"]),
        "reason": "",
        "energy_ev": witness.get("energy_ev") if witness else None,
        "fmax_ev_a": witness.get("fmax_ev_a") if witness else None,
        "forces_shape": witness.get("forces_shape") if witness else None,
        "gpu_pid_confirmed": bool(gpu_seen),
        "gpu_mem_bytes": gpu_mem,
    }
    if returncode != 0:
        verdict["reason"] = normalized_tail(stderr) or f"exit {returncode}"
        return verdict
    if witness is None:
        verdict["reason"] = "witness_json_missing"
        return verdict
    # A NaN/Inf energy or force is not a pass. This sits above BOTH remaining
    # branches on purpose: the CPU-fallback (skew) branch returns a pass without
    # any GPU proof, so it would otherwise wave a non-finite result through.
    # run_examples/single_point.py always emits both numbers, so a missing one is
    # a broken witness rather than an optional field.
    if not all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        for x in (verdict["energy_ev"], verdict["fmax_ev_a"])
    ):
        verdict["reason"] = "non_finite_result"
        return verdict
    if skew["skew"]:
        verdict["pass"] = True
        verdict["reason"] = skew["reason"]
        return verdict
    # GPU proof: either independent witness suffices — a sampled descendant
    # PID (unavailable on hosts whose driver hides compute-apps, e.g. some
    # virtualized drivers) or the worker's realized CUDA allocation (unavailable in
    # torch-less TF/JAX envs). Both absent => honest fail.
    if not gpu_seen and not (gpu_mem and gpu_mem > 0):
        verdict["reason"] = "gpu_not_used"
        return verdict
    verdict["pass"] = True
    return verdict


def family_versions(model: str, home: Path) -> list[str]:
    """All version keys of the family `model` names (family or version key)."""
    registry = json.loads((home / "models.json").read_text())
    want = model.lower()
    for family, spec in registry.items():
        if family.startswith("_"):
            continue
        versions = list(spec.get("versions") or {})
        if family.lower() == want or any(v.lower() == want for v in versions):
            return versions
    return []


def verify_one(model: str, version: str | None, structure: str | None, home: Path,
               *, quiet: bool, no_local_record: bool) -> dict:
    """Run the oracle for ONE (model, version) and return its verdict dict.
    Exactly the historical main() body, extracted so --all-variants can loop
    it and so the local-record write is one guarded step."""
    env_name = find_env(version or model, home)
    if env_name is None:
        return {"pass": False, "reason": f"unknown model: {model}", "version": version}

    skew = predict_driver_skew(env_name, home)
    command = [sys.executable, str(home / "run_examples" / "single_point.py"), model, "--json"]
    if version:
        command += ["--version", version]
    if skew["skew"]:
        command += ["--device", "cpu"]
    if structure:
        command += ["--structure", structure]

    log_dir = home / ".sweep" / "verify"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OH_MY_MLIP_HOME=str(home))
    log_stem = f"{env_name}" + (f".{version}" if version else "")
    rc, _elapsed, stdout, stderr, gpu = stream_process(
        command,
        env=env,
        log_path=log_dir / f"{log_stem}.log",
        stderr_path=log_dir / f"{log_stem}.stderr.log",
        collect=True,
        monitor_gpu=not skew["skew"],
        cwd=home,
        gpu_sample_seconds=0.2,
        quiet=quiet,
    )

    verdict = decide_verdict(skew, rc, bool(gpu.get("seen")), parse_witness_json(stdout), stderr)
    verdict["model"] = model
    verdict["version"] = version
    verdict["env"] = env_name

    if verdict["pass"] and no_local_record:
        # Read-only witness: the verdict stands, the hub's local state is
        # untouched (no upsert into models.local.json at all).
        verdict["local_record"] = "skipped(--no-local-record)"
    elif verdict["pass"]:
        # Materialize-on-verify: freeze the facts that just computed (exact
        # interpreter, weight paths, evidence) into models.local.json so every
        # later session resolves them deterministically — incremental upsert,
        # so each newly installed model adds its own entry. A local-record write
        # failure is reported in the verdict but never flips a computed pass.
        sys.path.insert(0, str(home))
        try:
            from oh_my_mlip.fetch import weight_targets
            from oh_my_mlip.registry import record_local_verified
            from oh_my_mlip.registry import resolve as registry_resolve
            spec = registry_resolve(model, version=version)
            weights = [w for w in weight_targets(spec) if Path(w).exists()]
            record_local_verified(spec, verdict, weights, str(home))
            verdict["local_record"] = "recorded"
        except Exception as exc:
            verdict["local_record"] = f"failed: {exc}"
    return verdict


def _print_verdict(verdict: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(verdict))
        return
    state = "PASS" if verdict["pass"] else "FAIL"
    extra = " (degraded: cpu)" if verdict.get("degraded") and verdict["pass"] else ""
    label = f" [{verdict['version']}]" if verdict.get("version") else ""
    print(f"{state}{extra}{label} device={verdict.get('device')} energy={verdict.get('energy_ev')} "
          f"gpu_pid_confirmed={verdict.get('gpu_pid_confirmed')} reason={verdict.get('reason') or '-'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="framework or version name from models.json")
    ap.add_argument("--version", default=None,
                    help="specific version, passed through to single_point.py. Required to reach a "
                         "variant whose name is shadowed by its family key (family wins on a bare name)")
    ap.add_argument("--structure", default=None, help="structure file passed through to single_point.py")
    ap.add_argument("--json", action="store_true", help="print the verdict as JSON (agent path)")
    ap.add_argument("--all-variants", action="store_true",
                    help="verify EVERY version of the family (one verdict per variant + a summary line)")
    ap.add_argument("--no-local-record", action="store_true",
                    help="never write models.local.json, even on PASS (read-only witness)")
    args = ap.parse_args()

    home = resolve_home()

    if args.all_variants:
        if args.version:
            print("--all-variants and --version are mutually exclusive", file=sys.stderr)
            return 2
        versions = family_versions(args.model, home)
        if not versions:
            verdict = {"pass": False, "reason": f"unknown model: {args.model}"}
            print(json.dumps(verdict) if args.json else f"FAIL: {verdict['reason']}")
            return 1
        verdicts = []
        for version in versions:
            v = verify_one(args.model, version, args.structure, home,
                           quiet=args.json, no_local_record=args.no_local_record)
            _print_verdict(v, args.json)
            verdicts.append(v)
        all_pass = all(v["pass"] for v in verdicts)
        summary = {"all_variants": True, "model": args.model, "pass": all_pass,
                   "variants": [{"version": v.get("version"), "pass": v["pass"],
                                 "degraded": v.get("degraded"), "reason": v.get("reason")}
                                for v in verdicts]}
        if args.json:
            print(json.dumps(summary))
        else:
            print(f"{'PASS' if all_pass else 'FAIL'} all-variants "
                  f"{sum(v['pass'] for v in verdicts)}/{len(verdicts)} passed")
        return 0 if all_pass else 1

    verdict = verify_one(args.model, args.version, args.structure, home,
                         quiet=args.json, no_local_record=args.no_local_record)
    _print_verdict(verdict, args.json)
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
