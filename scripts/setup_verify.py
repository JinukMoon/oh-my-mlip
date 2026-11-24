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
      gpu_pid_confirmed, arch, recipe}
     ``arch`` is the GPU architecture it ran on and ``recipe`` the recipe the
     env was built from (envs/<env>/.omm_recipe; e.g. mace.cu128 on sm120).
     models.json ``arch_validation.<arch>`` is set from such verdicts only.
  3. D3 check, reported on its own as ``d3`` (never part of ``pass``): one
     catbench D3 calculation through the env's bare interpreter, the way a
     user script runs it, compared with a reference value. ``d3.state`` is
     pass / fail / not_applicable (no torch or no catbench D3 module, e.g.
     GRACE) / skipped (no interpreter, or a CPU run on a skewed driver: the
     kernel is CUDA-only). A model can pass E/F and fail D3; the D3 kernel is
     built at install/adopt time (scripts/build_d3.py), so a missing
     pair_d3.so is a fail here, not a compile.

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
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup_common import (  # noqa: E402
    predict_driver_skew,
    refuse_plugin_copy,
    resolve_home,
    stream_process,
)
from setup_guardrail import _normalize_line  # noqa: E402
import build_d3  # noqa: E402

STDERR_TAIL_LINES = 20

# D3 reference: Pt fcc111 (3,3,4) vacuum 8 + CO ontop at 1.9 A, rattle(0.02, seed=1),
# catbench DispersionCorrection() defaults (damp_bj, pbe). The same numbers on an
# sm89 and an sm120 build (2026-11-19); the kernel is compiled with -fmad=false.
D3_REFERENCE = {"energy_ev": -21.30975002679841, "abs_force_sum": 5.444243171259453}
D3_TOLERANCE = {"energy_ev": 1e-5, "abs_force_sum": 1e-4}
D3_MARK = "OMM_D3_CHECK "
D3_CHECK_CODE = f"""
import json
import numpy as np
from ase.build import add_adsorbate, fcc111, molecule
from catbench.dispersion import DispersionCorrection
slab = fcc111("Pt", size=(3, 3, 4), vacuum=8.0)
add_adsorbate(slab, molecule("CO"), 1.9, "ontop")
slab.rattle(0.02, seed=1)
slab.calc = DispersionCorrection()._get_d3_calculator()
e = float(slab.get_potential_energy())
f = float(np.abs(slab.get_forces()).sum())
print({D3_MARK!r} + json.dumps({{"energy_ev": e, "abs_force_sum": f}}))
"""


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
        # The not-installed message carries the exact install command; the
        # normalization used for error signatures would mangle it into text
        # that cannot be run, and the agent is told to report it as printed.
        if "is not materialized yet" in stderr:
            verdict["reason"] = "\n".join(ln for ln in stderr.splitlines() if ln.strip())[-2000:]
        else:
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


def judge_d3(returncode: int, stdout: str, stderr: str) -> dict:
    """Pure D3 verdict from one D3 check run (unit-testable)."""
    result = None
    for line in stdout.splitlines():
        if line.startswith(D3_MARK):
            try:
                result = json.loads(line[len(D3_MARK):])
            except ValueError:
                result = None
    if returncode != 0 or result is None:
        return {"state": "fail", "reason": normalized_tail(stderr) or f"exit {returncode}, no D3 result"}
    out = {"state": "pass", "reason": "", **result}
    for key, ref in D3_REFERENCE.items():
        got = result.get(key)
        if not isinstance(got, (int, float)) or not math.isfinite(got) or abs(got - ref) > D3_TOLERANCE[key]:
            out["state"] = "fail"
            out["reason"] = f"{key} {got} differs from the reference {ref}"
            break
    return out


def check_d3(model: str, version: str | None, skew: dict, run=subprocess.run) -> dict:
    """The D3 check item: one D3 calculation in the model's env, through its bare
    interpreter (resolve()["python"] plus the env_run prefix, no env bin on PATH)."""
    try:
        from oh_my_mlip.registry import resolve as registry_resolve
        spec = registry_resolve(model, version=version)
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return {"state": "skipped", "reason": f"cannot resolve the interpreter: {exc}"}
    python = Path(os.path.expandvars(spec["python"]))
    if not os.access(python, os.X_OK):
        return {"state": "skipped", "reason": f"no interpreter at {python}"}
    env = dict(os.environ)
    env.update(spec.get("env_run") or {})
    prefix = python.parent.parent
    # the marker build_d3.py left at install/adopt time: a failed build that did
    # not stop the install (an env already in use) is surfaced here
    marker = build_d3.read_marker(prefix)
    build = {k: marker[k] for k in ("state", "log", "path") if marker.get(k)} if marker else None
    result = _check_d3(python, prefix, env, skew, run, marker)
    if build:
        result["build"] = build
    return result


def _check_d3(python: Path, prefix: Path, env: dict, skew: dict, run, marker: dict) -> dict:
    found = build_d3.locate(python, run=run, env=env)
    if found["error"]:
        return {"state": "fail", "reason": found["error"]}
    if not found["applicable"]:
        return {"state": "not_applicable", "reason": "no torch or no catbench D3 module in this env"}
    if skew.get("skew"):
        return {"state": "skipped", "reason": "the D3 kernel is CUDA-only and this env runs on CPU here: "
                                              + str(skew.get("reason") or "driver skew")}
    if not found["present"]:
        if marker.get("state") == "failed":
            return {"state": "fail", "reason": f"the D3 kernel build failed at install/adopt time: "
                                               f"{marker.get('detail', '')} (marker {marker['path']}); rerun: "
                                               f"python3 {build_d3.__file__} {prefix}"}
        return {"state": "fail", "reason": f"pair_d3.so is not built (neither in {found['d3_dir']} nor in "
                                           f"the torch extensions cache); build it with: "
                                           f"python3 {build_d3.__file__} {prefix}"}
    try:
        proc = run([str(python), "-c", D3_CHECK_CODE], capture_output=True, text=True,
                   timeout=900, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"state": "fail", "reason": f"D3 check did not finish: {exc}"}
    return judge_d3(proc.returncode, proc.stdout or "", proc.stderr or "")


def verified_arch_and_recipe(env_name: str, home: Path) -> tuple[str | None, str]:
    """(GPU arch of this host, recipe stem the env was built from)."""
    recipe = env_name
    marker = home / "envs" / env_name / ".omm_recipe"
    try:
        recipe = marker.read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError):
        pass
    arch = None
    try:
        sys.path.insert(0, str(home))
        from oh_my_mlip.registry import detect_host_arch
        arch = detect_host_arch()
    except Exception:  # noqa: BLE001 - the verdict never fails on bookkeeping
        pass
    return arch, recipe


def _known_hint(home: Path) -> str:
    """The names a user can pass, for an 'unknown model' message."""
    try:
        registry = json.loads((home / "models.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "see models.json for the model names"
    families = sorted(f for f in registry if not f.startswith("_"))
    return ("families: " + ", ".join(families)
            + " (or a version name from docs/model_status.md)")


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


def gpu_visible(run=subprocess.run, env: dict | None = None) -> bool:
    """True when this process can see an NVIDIA GPU (nvidia-smi -L lists one and
    CUDA_VISIBLE_DEVICES does not hide them all)."""
    env = os.environ if env is None else env
    if env.get("CUDA_VISIBLE_DEVICES", None) is not None and \
            env["CUDA_VISIBLE_DEVICES"].split(",")[0].strip() in ("", "-1"):
        return False
    import shutil
    wsl = "/usr/lib/wsl/lib/nvidia-smi"          # as oh_my_mlip.registry.nvidia_smi
    smi = shutil.which("nvidia-smi") or (wsl if os.access(wsl, os.X_OK) else None)
    if smi is None:
        return False
    try:
        out = run([smi, "-L"], capture_output=True, text=True, timeout=30).stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        return False
    return any(line.startswith("GPU ") for line in out.splitlines())


NO_GPU_REASON = ("no NVIDIA GPU is visible to this process (nvidia-smi is missing or lists none, or "
                 "CUDA_VISIBLE_DEVICES hides them), so nothing was run. Verify on a GPU: on a cluster, inside "
                 "a GPU job (SLURM: sbatch/srun with --gres=gpu:1 or --gpus=1); a login node without a GPU "
                 "can install, not verify")


def verify_one(model: str, version: str | None, structure: str | None, home: Path,
               *, quiet: bool, no_local_record: bool) -> dict:
    """Run the oracle for ONE (model, version) and return its verdict dict.
    Exactly the historical main() body, extracted so --all-variants can loop
    it and so the local-record write is one guarded step."""
    env_name = find_env(version or model, home)
    if env_name is None:
        return {"pass": False, "reason": f"unknown model: {model}; {_known_hint(home)}", "version": version}

    skew = predict_driver_skew(env_name, home)
    command = [sys.executable, str(home / "run_examples" / "single_point.py"), model, "--json"]
    if version:
        command += ["--version", version]
    if skew["skew"]:
        command += ["--device", "cpu"]
    if structure:
        # The child runs with cwd=home, so a relative path from the user's shell
        # would resolve against the clone instead -- and the tool's own usage line
        # shows `--structure POSCAR`. Resolve it here, against the caller's cwd.
        structure_path = Path(structure).expanduser()
        if not structure_path.is_absolute():
            structure_path = Path.cwd() / structure_path
        if not structure_path.exists():
            return {"pass": False, "version": version,
                    "reason": f"structure not found: {structure_path}"}
        command += ["--structure", str(structure_path)]
    if not skew["skew"] and not gpu_visible():
        # fail at once: a cuda run here can only fail, after minutes of loading
        return {"pass": False, "device": "cuda", "degraded": False, "reason": NO_GPU_REASON,
                "model": model, "version": version, "env": env_name, "gpu_pid_confirmed": False}

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
    # where this verdict holds: the GPU arch it ran on and the recipe the env
    # was built from (install.sh writes envs/<env>/.omm_recipe; an env without
    # it was built from the base recipe). A pass is an arch claim for that pair.
    verdict["arch"], verdict["recipe"] = verified_arch_and_recipe(env_name, home)
    # D3 on its own line of the verdict: it never changes "pass" (E/F on the GPU)
    verdict["d3"] = check_d3(model, version, skew)

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
    d3 = verdict.get("d3")
    if d3:
        print(f"  D3: {d3['state'].upper()}" + (f" ({d3['reason']})" if d3.get("reason") else ""))
        if (d3.get("build") or {}).get("state") == "failed":
            print(f"  D3 build marker: failed (log: {d3['build'].get('log') or '-'}, marker: {d3['build']['path']})")


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
    refuse_plugin_copy(home)

    if args.all_variants:
        if args.version:
            print("--all-variants and --version are mutually exclusive", file=sys.stderr)
            return 2
        versions = family_versions(args.model, home)
        if not versions:
            verdict = {"pass": False, "reason": f"unknown model: {args.model}; {_known_hint(home)}"}
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
                                 "degraded": v.get("degraded"), "reason": v.get("reason"),
                                 "d3": (v.get("d3") or {}).get("state")}
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
