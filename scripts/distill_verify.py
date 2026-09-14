#!/usr/bin/env python3
"""distill_verify.py --work DIR [--json] -- the oracle for recipes/distill.md §5.

Judges a finished (or stopped) onthefly-distill run against the approved
`acceptance.json` that `scripts/distill_bootstrap.py --acceptance` rendered,
and prints ONE verdict. It never re-judges the engine's own stop reason: the
final `run/.al_status` line and each round's `failure.json` are read as
written. What it adds, in order, each step logged to `<work>/verify/verify.log`:

  1. contract      -- `acceptance.json` present (no approved targets, no
                      verdict); `config.yaml`'s `al_loop.*` still equals the
                      approved budget/target (`failed(config_drift)` otherwise
                      -- an edit past the approved values is out of scope);
                      every sha256 the proposal recorded -- config.yaml,
                      run_distill.sh, omm_teacher.py, the structure file, the
                      invoked engine files -- still matches
                      (`failed(provenance_drift)` otherwise: the run is not
                      the approved one). This catches ordinary post-approval
                      edits; it is not a defence against an owner rewriting
                      acceptance.json itself, and claims none. On ANY drift
                      the verifier stops here: `work_dir` comes from the live
                      config.yaml, so a drifted config could redirect every
                      later inspection at another run's directory -- nothing
                      under run/ is read, and the verdict is the drift.
  1b. wall-clock   -- if the approved budget carries a wall-clock bound, the
                      start record `<work>/.distill/attempt.json` and the
                      outcome marker `<work>/.distill/wallclock.json` that
                      run_distill.sh's supervisor wrote (GNU `timeout` over
                      the WHOLE script) are read and bound to each other and
                      to the approved contract: same attempt id and start
                      time, limit/grace as approved, sha256 of the script that
                      ran == the approved run_distill.sh, returncode/state
                      consistent, elapsed == ended - started, and `.al_status`
                      last written inside the attempt's window. Missing,
                      unreadable or unbound evidence => `failed(wallclock_evidence)`
                      (a marker absent while the loop is still running is
                      `incomplete`); `exhausted` => `unmet(budget:wallclock)`
                      whatever `.al_status` says (a status line written in the
                      same second the group was signalled is not a pass).
                      The marker's `attribution` is recomputed from rc +
                      elapsed and must agree: rc 124 = `timeout_term`; rc 137
                      at/after the limit = `kill_at_or_after_limit` (timeout's
                      KILL after the grace OR an external SIGKILL after the
                      deadline -- indistinguishable, recorded as ambiguous,
                      still exhaustion); rc 137 before the limit =
                      `sigkill_before_limit` (source unknown; `engine_exit`,
                      and a marker calling it exhaustion is rejected); rc 0 =
                      `clean`; anything else = `exit_nonzero` (`engine_exit`).
                      A VALID `engine_exit` marker -- the supervised run
                      exited non-zero without expiring -- is
                      `failed(supervisor_exit)` with the actual rc, whatever
                      `.al_status` says (a SUCCESS line under a run that then
                      died is not this attempt's pass), and no metrics or
                      witness subprocess is launched for it.
  2. loop state    -- `.al_status` classified: SUCCESS / STOPPED no_progress
                      (STALLED) / STOPPED backstop / STOPPED label_fail /
                      FAILED / DONE oneshot (teacher could not relabel: the AL
                      loop silently degraded) / ERROR / still running.
  3. relabel rounds-- non-empty `run/al_iter<K>_labeled.extxyz` files (K a
                      decimal integer, exactly the names the loop writes) AND
                      a retrain after the first of them (`model_scratch<R>.bin`,
                      R >= 1). A fixture must show at least one. Any other
                      file the trainer's `al_iter*_labeled.extxyz` glob would
                      absorb (a digit-less K) is foreign pool data:
                      `failed(malformed_evidence)`, never a crash.
  4. held-out      -- path outside the engine work dir, seed distinct from
                      the pool's, file non-empty, sha256 recorded, no frame
                      equal to the init structure (teacher_md.py's pre-MD
                      frame 0, which the pool starts with), and an EMPTY
                      per-frame fingerprint intersection with the pool
                      (dataset.extxyz + every al_iter*_labeled.extxyz):
                      `failed(heldout_leak)` otherwise. The trainer's random
                      validation split is never accepted as held-out.
  5. accuracy      -- `<work>/verify/heldout_eval.py` (written here, then
                      executed with the TEACHER env's interpreter from
                      `config.yaml`, PYTHONPATH = engine + work dir) loads the
                      final `model_scratch<N>.pt` through the engine's own
                      `ontheflydistill.common` (NNMTP, build_cache with the
                      Fmax filter disabled, validate) and writes
                      `metrics.json`: energy MAE meV/atom, force MAE meV/A,
                      frame count, torch/numpy/python versions, device.
  6. lmp witness   -- the loop's own final-round `run_scratch<N>/md.log` +
                      `failure.json` are cited, AND a fresh `run 0` of the
                      final `.bin` under `lmp` (`<work>/verify/lmp_witness/`)
                      must exit 0 with a finite potential energy and no
                      ERROR line. `--no-lmp-witness` records the witness as
                      skipped: a run that would otherwise pass is
                      `incomplete(lmp_witness_skipped)` -- not passed (the
                      witness never ran), not failed (nothing failed).
  7. verdict       -- production: `passed` iff SUCCESS AND both held-out MAEs
                      within thresholds AND witness ok; SUCCESS above a
                      threshold => `unmet(accuracy)` with a reconfigured-rerun
                      proposal inside the approved budget class; STALLED /
                      backstop / label_fail => `unmet(stability|budget)`;
                      FAILED / oneshot => `failed(<class>)`; still running =>
                      `incomplete`. Fixture: `passed` iff >= 1 real relabel
                      round AND terminal state AND metrics computed AND
                      witness ok (thresholds reported, not gating); SUCCESS at
                      round 0 in a fixture is `unmet(fixture:no_relabel)`.

Outputs `<work>/verify/distill_verify.json` and appends one JSONL row to
`<work>/.distill/ledger.jsonl` (or `--ledger`) in the shape the other
campaign ledgers use (`kind`, `phase`, `state`, `verdict`, `evidence`,
`manifest_sha256`, `campaign_id`). Exit 0 iff the state is `passed`.

Held-out metrics come from the `.pt` (torch) model; the exported `.bin`
is the same weights (`common.export_v1`) but is witnessed by `lmp` for
stability and finiteness only, not for per-frame accuracy. The report and
the ledger row carry that split explicitly (`artifacts`: accuracy artifact
= .pt, witness artifact = .bin, witness scope = finiteness/stability, NOT
accuracy equivalence) so a `passed` row is not read as "the deployed .bin
is accurate".
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    import numpy as np
    import yaml
    from ase.io import read as ase_read
except ImportError as exc:  # pragma: no cover - environment hint
    raise ImportError(
        "distill_verify.py needs ase + numpy + pyyaml on the interpreter that "
        "runs it -- the same requirement as distill_bootstrap.py."
    ) from exc

ACCEPTANCE_SCHEMA = "oh-my-mlip.distill.acceptance/1"
WALLCLOCK_SCHEMA = "oh-my-mlip.distill.wallclock/3"  # /3: carries `attribution`
ATTEMPT_SCHEMA = "oh-my-mlip.distill.attempt/1"
WALLCLOCK_EXHAUSTED_RCS = (124, 137)
WALLCLOCK_STATES = ("exhausted", "within", "engine_exit")
# The supervisor's rc/elapsed table (distill_bootstrap.WALLCLOCK_ATTRIBUTION,
# kept in sync by hand: the two hub scripts stay import-free of each other).
# A late rc 137 is AMBIGUOUS -- timeout's KILL after the grace or an external
# SIGKILL after the deadline -- and is never reported as a proven source.
WALLCLOCK_ATTRIBUTION = {
    "timeout_term": "rc 124: GNU timeout's expiry (TERM sufficed) -- budget exhausted",
    "kill_at_or_after_limit": "rc 137 at/after the limit: timeout's KILL after the grace OR an external SIGKILL "
                              "that landed after the deadline -- rc + elapsed cannot tell which (ambiguous, "
                              "recorded as such); the budget had expired either way, so exhausted",
    "sigkill_before_limit": "rc 137 before the limit: SIGKILL of the engine, source unknown (OOM killer, kill -9, "
                            "...) -- not expiry",
    "clean": "rc 0: clean exit within the bound",
    "exit_nonzero": "any other rc: the engine/loop exited non-zero or by another signal (128+N) -- not expiry",
}
# The boundary the witness runs under: the engine's own student MD input
# (ontheflydistill/student_md_lammps.py run(): `boundary p p f`, reflective z
# walls). The held-out accuracy is measured on periodic teacher labels
# (system.force_pbc); that split is the engine's design and is disclosed in
# the report's `artifacts`, not reconciled here.
WITNESS_BOUNDARY = "p p f"
WITNESS_BOUNDARY_SOURCE = "ontheflydistill/student_md_lammps.py run() -- the engine's student MD boundary"
ACCURACY_PBC = "[T, T, T] -- teacher labels forced periodic (system.force_pbc, ontheflydistill/teachers/ase_calculator.py)"


def wallclock_attribution(rc: int, elapsed_s: int, limit_s: int) -> str:
    """The token the supervisor must have written for this rc/elapsed (its
    rule verbatim: rc 137 counts as at/after the limit when elapsed_s >=
    limit_s, integer seconds, no slack -- not tightened to limit + grace,
    since a genuine KILL after the grace is measured in whole seconds)."""
    if rc == 124:
        return "timeout_term"
    if rc == 137:
        return "kill_at_or_after_limit" if elapsed_s >= limit_s else "sigkill_before_limit"
    if rc == 0:
        return "clean"
    return "exit_nonzero"
MTIME_SLACK_S = 2  # .al_status mtime vs the attempt's [started, ended] window
FINGERPRINT_DECIMALS = 4  # positions rounded to 1e-4 A before hashing

_STATUS_RE = {
    "success": re.compile(r"^SUCCESS round(\d+) stable\b"),
    "stalled": re.compile(r"^STOPPED no_progress round(\d+)\b"),
    "backstop": re.compile(r"^STOPPED backstop\b"),
    "label_fail": re.compile(r"^STOPPED label_fail round(\d+)\b"),
    "failed": re.compile(r"^FAILED (\S+) (\S+)"),
    "oneshot": re.compile(r"^DONE oneshot\b"),
    "error": re.compile(r"^ERROR (\S+)"),
}


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[distill_verify] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", required=True, type=Path, help="the distill_bootstrap.py --acceptance work dir")
    ap.add_argument("--python", type=Path, default=None,
                    help="interpreter for the held-out evaluation (default: config.yaml python_bin = the teacher env)")
    ap.add_argument("--lmp-bin", type=Path, default=None, help="LAMMPS binary (default: config.yaml lmp_bin)")
    ap.add_argument("--repo", type=Path, default=None, help="engine checkout (default: acceptance.json provenance)")
    ap.add_argument("--no-lmp-witness", action="store_true",
                    help="skip the fresh lmp run-0 witness (recorded as skipped; the verdict cannot pass)")
    ap.add_argument("--ledger", type=Path, default=None, help="JSONL ledger (default: <work>/.distill/ledger.jsonl)")
    ap.add_argument("--manifest-sha256", default=os.environ.get("OMM_MANIFEST_SHA256"))
    ap.add_argument("--campaign-id", default=os.environ.get("OMM_CAMPAIGN_ID"))
    ap.add_argument("--json", action="store_true", help="print the verdict as one JSON line at the end")
    return ap.parse_args(argv)


# ── 1. contract ──────────────────────────────────────────────────────────────
def load_contract(work: Path, log: Log) -> tuple[dict, dict, dict]:
    acc_path = work / "acceptance.json"
    cfg_path = work / "config.yaml"
    if not acc_path.is_file():
        raise SystemExit(
            f"{acc_path} missing -- no approved targets, no verdict. Render the proposal with "
            "`scripts/distill_bootstrap.py --acceptance ...`, get it approved, run, then verify."
        )
    if not cfg_path.is_file():
        raise SystemExit(f"{cfg_path} missing -- this is not a distill_bootstrap.py work dir")
    acc = json.loads(acc_path.read_text(encoding="utf-8"))
    if acc.get("schema") != ACCEPTANCE_SCHEMA:
        raise SystemExit(f"{acc_path}: schema {acc.get('schema')!r} != {ACCEPTANCE_SCHEMA!r}")
    if acc.get("mode") not in ("fixture", "production"):
        raise SystemExit(f"{acc_path}: mode must be fixture|production, got {acc.get('mode')!r}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    log(f"step 1/7: contract {acc_path} (mode={acc['mode']}) + {cfg_path}")

    approved = {"target_ps": acc["stability"]["target_ps"], "max_iter": acc["budget"]["max_iter"],
                "no_progress_limit": acc["budget"]["no_progress_limit"], "seed": acc["budget"]["seed"]}
    actual = {k: (cfg.get("al_loop") or {}).get(k) for k in approved}
    drift = {k: {"approved": approved[k], "config": actual[k]} for k in approved if actual[k] != approved[k]}
    prov = acc.get("provenance") or {}
    recorded = (prov.get("generated_files") or {}).get("config.yaml") or {}
    contract = {
        "acceptance_path": str(acc_path),
        "config_path": str(cfg_path),
        "config_sha256": _sha256(cfg_path),
        "config_sha256_at_bootstrap": recorded.get("sha256"),
        "al_loop_drift": drift,
        "provenance_drift": provenance_drift(prov),
    }
    contract["config_edited_since_bootstrap"] = (
        recorded.get("sha256") is not None and recorded["sha256"] != contract["config_sha256"])
    if drift:
        log(f"  al_loop drift vs approved values: {drift}")
    if contract["provenance_drift"]:
        log(f"  provenance drift (sha256 recorded at approval != on disk): {sorted(contract['provenance_drift'])}")
    return acc, cfg, contract


def provenance_drift(prov: dict) -> dict:
    """Every file identity the proposal recorded, recomputed: generated files
    (config.yaml, run_distill.sh, omm_teacher.py), the structure, the engine
    files that were invoked. {name: {recorded, actual}} for each mismatch
    (actual None = missing). Ordinary post-approval drift only; acceptance.json
    itself is the reference, not something this can vouch for."""
    checks: dict[str, tuple[Path, str | None]] = {}
    for name, info in (prov.get("generated_files") or {}).items():
        checks[name] = (Path(info["path"]), info.get("sha256"))
    structure = prov.get("structure") or {}
    if structure.get("path"):
        checks["structure"] = (Path(structure["path"]), structure.get("sha256"))
    engine = prov.get("engine") or {}
    for rel, sha in (engine.get("files") or {}).items():
        checks[f"engine:{rel}"] = (Path(engine["repo"]) / rel, sha)
    out = {}
    for name, (path, recorded_sha) in checks.items():
        if recorded_sha is None:
            continue
        actual = _sha256(path) if path.is_file() else None
        if actual != recorded_sha:
            out[name] = {"path": str(path), "recorded": recorded_sha, "actual": actual}
    return out


# ── 1b. wall-clock marker (hub-owned, written by run_distill.sh) ─────────────
def _read_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"{path.name}: unreadable ({exc})"
    if not isinstance(data, dict):
        return None, f"{path.name}: not a JSON object"
    return data, None


def wallclock_evidence(work: Path, acc: dict, run_dir: Path, log: Log) -> dict:
    """The supervisor's evidence, demanded (not merely read) when the approved
    budget carries a bound: start record + outcome marker, bound to each
    other, to the approved plan, to the approved run_distill.sh and to the
    engine's status file's write time. `valid` is False on any problem;
    `pending` is True only for the one legitimate no-marker case (start
    record present, marker not yet written -- an attempt still running)."""
    budget = acc.get("budget") or {}
    enf = budget.get("wallclock_enforcement")
    out = {"limit_h": budget.get("wallclock_max_h"), "enforced": bool(enf), "attempt_record": None, "marker": None,
           "attempt": None, "state": None, "attribution": None, "attribution_note": None, "elapsed_s": None,
           "returncode": None, "limit_s": None, "exhausted": False, "valid": True, "pending": False, "problems": []}
    if not enf:
        if budget.get("wallclock_max_h"):
            out["valid"] = False
            out["problems"].append("bound stated in acceptance.json without an enforcement plan")
            log("step 1b/7: wall-clock bound stated but acceptance.json carries no enforcement plan")
        return out
    problems: list[str] = []
    attempt_path = Path(enf.get("attempt") or (work / ".distill" / "attempt.json"))
    marker_path = Path(enf["marker"])
    out.update({"attempt_record": str(attempt_path), "marker": str(marker_path), "limit_s": enf.get("limit_s")})
    approved_sha = (((acc.get("provenance") or {}).get("generated_files") or {}).get("run_distill.sh") or {}).get("sha256")

    attempt = None
    if not attempt_path.is_file():
        problems.append(f"start record {attempt_path} missing: the run was not launched through run_distill.sh's "
                        "supervisor, so no bound was in force")
    else:
        attempt, err = _read_json(attempt_path)
        if err:
            problems.append(err)
        elif attempt.get("schema") != ATTEMPT_SCHEMA:
            problems.append(f"start record schema {attempt.get('schema')!r} != {ATTEMPT_SCHEMA!r}")
        else:
            out["attempt"] = attempt.get("attempt")
            if attempt.get("limit_s") != enf.get("limit_s") or attempt.get("grace_s") != enf.get("grace_s"):
                problems.append(f"start record limit/grace {attempt.get('limit_s')}/{attempt.get('grace_s')} != "
                                f"approved {enf.get('limit_s')}/{enf.get('grace_s')}")
            if approved_sha and attempt.get("run_script_sha256") != approved_sha:
                problems.append("start record's run_script_sha256 is not the approved run_distill.sh")

    if not marker_path.is_file():
        if not problems:
            out["pending"] = True
            log(f"step 1b/7: attempt {out['attempt']} started, no outcome marker yet at {marker_path}")
        else:
            problems.append(f"outcome marker {marker_path} missing")
    else:
        info, err = _read_json(marker_path)
        if err:
            problems.append(err)
        elif info.get("schema") != WALLCLOCK_SCHEMA:
            problems.append(f"marker schema {info.get('schema')!r} != {WALLCLOCK_SCHEMA!r}")
        else:
            problems.extend(_marker_problems(info, attempt, enf, approved_sha, run_dir))
            out.update({"state": info.get("state"), "elapsed_s": info.get("elapsed_s"),
                        "returncode": info.get("returncode"), "limit_s": info.get("limit_s"),
                        "attribution": info.get("attribution"),
                        "attribution_note": WALLCLOCK_ATTRIBUTION.get(info.get("attribution"))})

    out["problems"] = problems
    out["valid"] = not problems
    out["exhausted"] = out["valid"] and out["state"] == "exhausted"
    if problems:
        log("step 1b/7: wall-clock evidence NOT acceptable: " + "; ".join(problems))
    elif out["state"]:
        log(f"step 1b/7: attempt {out['attempt']} wall-clock {out['state']} ({out['attribution']}): "
            f"{out['elapsed_s']} s of {out['limit_s']} s, rc={out['returncode']} (bound to the approved script "
            "and this attempt)")
    return out


def _marker_problems(info: dict, attempt: dict | None, enf: dict, approved_sha: str | None, run_dir: Path) -> list[str]:
    """Consistency of the outcome marker with the start record, the approved
    plan and the engine's status file. Each string is one reason the marker
    cannot be taken as this attempt's outcome."""
    p: list[str] = []
    state, rc = info.get("state"), info.get("returncode")
    if state not in WALLCLOCK_STATES:
        p.append(f"marker state {state!r} not one of {WALLCLOCK_STATES}")
    limit, grace = enf.get("limit_s") or 0, enf.get("grace_s") or 0
    elapsed_known = isinstance(info.get("elapsed_s"), int)
    # the supervisor's own rule, verbatim: 137 is at/after the limit iff
    # elapsed_s >= limit_s (no slack, not tightened to limit + grace)
    at_or_after_limit = elapsed_known and info["elapsed_s"] >= limit
    if not isinstance(rc, int):
        p.append("marker returncode missing")
    elif state == "exhausted" and rc not in WALLCLOCK_EXHAUSTED_RCS:
        p.append(f"marker says exhausted but rc={rc} is not a GNU timeout expiry code {WALLCLOCK_EXHAUSTED_RCS}")
    elif state == "within" and rc != 0:
        p.append(f"marker says within but rc={rc}")
    elif state == "engine_exit" and rc == 0:
        p.append("marker says engine_exit but rc=0")
    elif state == "engine_exit" and rc == 124:
        p.append("marker says engine_exit but rc=124 is GNU timeout's expiry code")
    elif state == "engine_exit" and rc == 137 and at_or_after_limit:
        # 137 before the limit is a SIGKILL of the engine (a genuine
        # engine_exit); 137 at/after it is exhaustion -- timeout's KILL or an
        # external SIGKILL after the deadline, unknown which
        p.append(f"marker says engine_exit but rc=137 after {info['elapsed_s']} s of a {limit} s bound is at/after "
                 "the limit: timeout's KILL after the grace or an external SIGKILL after the deadline (unknown "
                 "which) -- exhausted, not engine_exit")
    if isinstance(rc, int) and elapsed_known:
        expected = wallclock_attribution(rc, info["elapsed_s"], limit)
        if info.get("attribution") != expected:
            p.append(f"marker attribution {info.get('attribution')!r} != {expected!r} recomputed from rc={rc} and "
                     f"elapsed {info['elapsed_s']} s of {limit} s")
    elif info.get("attribution") not in WALLCLOCK_ATTRIBUTION:
        p.append(f"marker attribution {info.get('attribution')!r} not one of {tuple(WALLCLOCK_ATTRIBUTION)}")
    if info.get("limit_s") != enf.get("limit_s") or info.get("grace_s") != enf.get("grace_s"):
        p.append(f"marker limit/grace {info.get('limit_s')}/{info.get('grace_s')} != approved "
                 f"{enf.get('limit_s')}/{enf.get('grace_s')}")
    if approved_sha and info.get("run_script_sha256") != approved_sha:
        p.append("marker's run_script_sha256 is not the approved run_distill.sh")
    if attempt is not None:
        if info.get("attempt") != attempt.get("attempt"):
            p.append(f"marker attempt {info.get('attempt')!r} != start record {attempt.get('attempt')!r} (stale marker)")
        if info.get("started_epoch_s") != attempt.get("started_epoch_s"):
            p.append("marker start time != start record")
    started, ended, elapsed = info.get("started_epoch_s"), info.get("ended_epoch_s"), info.get("elapsed_s")
    if not all(isinstance(v, int) for v in (started, ended, elapsed)):
        p.append("marker timestamps missing")
    else:
        if ended < started or abs((ended - started) - elapsed) > MTIME_SLACK_S:
            p.append(f"marker elapsed {elapsed} s != ended - started ({ended - started} s)")
        if state == "within" and elapsed > limit + grace + MTIME_SLACK_S:
            p.append(f"marker says within but elapsed {elapsed} s exceeds the {limit} s bound")
        if state == "exhausted" and elapsed < limit - MTIME_SLACK_S:
            p.append(f"marker says exhausted after {elapsed} s, before the {limit} s bound"
                     + (" -- an early rc 137 is a SIGKILL of the engine (source unknown), not expiry"
                        if rc == 137 else ""))
        status_path = run_dir / ".al_status"
        if status_path.is_file():
            mtime = int(status_path.stat().st_mtime)
            if not (started - MTIME_SLACK_S <= mtime <= ended + MTIME_SLACK_S):
                p.append(f".al_status last written at epoch {mtime}, outside this attempt's window "
                         f"[{started}, {ended}] -- the status is not this attempt's")
    return p


# ── 2. loop state ────────────────────────────────────────────────────────────
def classify_status(run_dir: Path, log: Log) -> dict:
    status_path = run_dir / ".al_status"
    out = {"path": str(status_path), "line": None, "kind": None, "round": None, "terminal": False, "detail": None}
    if not status_path.is_file():
        out["kind"] = "missing"
        log(f"step 2/7: {status_path} missing -- the loop never wrote a status")
        return out
    line = status_path.read_text(encoding="utf-8").strip()
    out["line"] = line
    for kind, rx in _STATUS_RE.items():
        m = rx.match(line)
        if m:
            out["kind"] = kind
            out["terminal"] = True
            if kind in ("success", "stalled", "label_fail"):
                out["round"] = int(m.group(1))
            elif kind == "failed":
                out["detail"] = f"{m.group(1)} {m.group(2)}"  # e.g. "train round0", "md oneshot"
            elif kind == "error":
                out["detail"] = m.group(1)
            break
    else:
        out["kind"] = "running"  # "roundN training ..." / "roundN student-MD ..." / "oneshot ..."
    log(f"step 2/7: .al_status = {line!r} -> {out['kind']}" + (f" round {out['round']}" if out["round"] is not None else ""))
    return out


# ── 3. relabel rounds + final model ──────────────────────────────────────────
def _n_frames(path: Path) -> int:
    try:
        return len(ase_read(str(path), index=":"))
    except Exception:
        return 0


_LABELED_RE = re.compile(r"^al_iter(\d+)_labeled\.extxyz$")  # exactly what al_loop_local.sh writes


def relabel_rounds(run_dir: Path, log: Log) -> dict:
    """The engine names its relabel files al_iter<K>_labeled.extxyz with K a
    decimal integer (al_loop_local.sh nextK). The trainer, however, globs
    `al_iter*_labeled.extxyz`, so any other name matching that wildcard is
    trained on too: those are listed as `malformed_files` (never a crash)
    and decide() refuses them as foreign pool data."""
    matched: list[tuple[int, Path]] = []
    malformed: list[str] = []
    for p in sorted(run_dir.glob("al_iter*_labeled.extxyz")):
        m = _LABELED_RE.match(p.name)
        if m:
            matched.append((int(m.group(1)), p))
        else:
            malformed.append(str(p))
    labeled = [p for _, p in sorted(matched)]
    per_file = [{"path": str(p), "n_frames": _n_frames(p)} for p in labeled]
    real = [f for f in per_file if f["n_frames"] > 0]
    models = sorted(int(m.group(1)) for p in run_dir.glob("model_scratch*.bin")
                    if (m := re.match(r"model_scratch(\d+)\.bin$", p.name)))
    retrain_after_relabel = bool(real) and any(r >= 1 for r in models)
    out = {"labeled_files": per_file, "n_relabel_rounds": len(real), "malformed_files": malformed,
           "student_models": [f"model_scratch{r}.bin" for r in models],
           "retrain_after_relabel": retrain_after_relabel}
    log(f"step 3/7: relabel rounds = {len(real)} (non-empty al_iter<K>_labeled), student models = {out['student_models']}, "
        f"retrain after relabel = {retrain_after_relabel}")
    if malformed:
        log(f"  files matching the trainer's al_iter*_labeled.extxyz glob but not the engine's naming: {malformed}")
    return out


def final_model(run_dir: Path, status: dict, log: Log) -> dict:
    """The model the verdict is about: the SUCCESS round's, else the highest
    round that produced a .bin (a stopped run keeps its last student)."""
    rounds = sorted(int(m.group(1)) for p in run_dir.glob("model_scratch*.bin")
                    if (m := re.match(r"model_scratch(\d+)\.bin$", p.name)))
    if status["kind"] == "success":
        n = status["round"]
    elif rounds:
        n = rounds[-1]
    else:
        log("  no model_scratch<N>.bin under run/ -- nothing to evaluate")
        return {"round": None, "bin": None, "pt": None, "md_dir": None}
    out = {"round": n, "bin": str(run_dir / f"model_scratch{n}.bin"), "pt": str(run_dir / f"model_scratch{n}.pt"),
           "md_dir": str(run_dir / f"run_scratch{n}"), "train_log": str(run_dir / f"train_scratch{n}.log")}
    for key in ("bin", "pt"):
        if not Path(out[key]).is_file():
            log(f"  final {key} {out[key]} missing")
            out[key] = None
    out["train_log_best_f_mae_mev_per_a"] = _trainer_best_f(Path(out["train_log"]))
    log(f"  final student: round {n}, bin={out['bin']}, pt={out['pt']}")
    return out


def _trainer_best_f(train_log: Path) -> float | None:
    """The engine's own `best F_MAE: X meV/A` line -- cited as evidence of the
    run, NOT used as the held-out metric (it is the pool's random split)."""
    if not train_log.is_file():
        return None
    m = re.findall(r"best F_MAE:\s*([0-9.]+)\s*meV/A", train_log.read_text(encoding="utf-8", errors="ignore"))
    return float(m[-1]) if m else None


# ── 4. held-out independence ─────────────────────────────────────────────────
def frame_fingerprints(path: Path) -> set[str]:
    fps: set[str] = set()
    for atoms in ase_read(str(path), index=":"):
        pos = np.round(atoms.get_positions(), FINGERPRINT_DECIMALS).astype(np.float64)
        h = hashlib.sha256(atoms.get_atomic_numbers().astype(np.int64).tobytes() + pos.tobytes())
        fps.add(h.hexdigest())
    return fps


def check_heldout(acc: dict, run_dir: Path, log: Log) -> dict:
    h = acc["split"]["heldout"]
    pool_seed = acc["split"]["pool"]["teacher_md_seed"]
    path = Path(h["path"])
    # the init frame teacher_md.py actually saved is the STAGED copy (what
    # config.yaml points at); older proposals recorded only the source
    struct_prov = (acc.get("provenance") or {}).get("structure") or {}
    structure = Path(struct_prov.get("staged_path") or struct_prov.get("path") or "")
    out = {"path": str(path), "source": h["source"], "seed": h.get("seed"), "pool_seed": pool_seed,
           "exists": path.is_file() and path.stat().st_size > 0, "sha256": None, "n_frames": 0,
           "outside_engine_work_dir": not _is_under(path, run_dir),
           "seed_distinct": h["source"] != "separate_teacher_md" or h.get("seed") != pool_seed,
           "init_structure": str(structure) if structure.name else None, "contains_init_frame": None,
           "pool_files": [], "n_pool_frames": 0, "n_overlap": None, "leak": None}
    log(f"step 4/7: held-out {path} (source={h['source']}, seed={h.get('seed')}; pool seed {pool_seed})")
    if not out["outside_engine_work_dir"]:
        log(f"  held-out path is INSIDE the engine work dir {run_dir} -- the trainer glob can absorb it")
    if not out["seed_distinct"]:
        log("  held-out seed equals the pool seed -- same trajectory, not held out")
    if not out["exists"]:
        log("  held-out file missing or empty")
        return out
    out["sha256"] = _sha256(path)
    held = frame_fingerprints(path)
    out["n_frames"] = len(held)
    if not held:
        out["exists"] = False
        log("  held-out file carries no frames")
        return out
    # teacher_md.py's frame 0 is the init structure itself, for the pool's
    # seed and for any other: a held-out set that still carries it shares a
    # frame with the pool whether or not the pool file is on disk to show it.
    if structure.is_file():
        try:
            init_fp = next(iter(frame_fingerprints(structure)), None)
        except Exception as exc:  # unreadable structure: provenance drift reports it; no init check possible
            log(f"  init structure {structure} unreadable ({exc}); init-frame check skipped")
        else:
            out["contains_init_frame"] = init_fp in held
            if out["contains_init_frame"]:
                log(f"  a held-out frame IS the init structure {structure} (teacher_md.py's pre-MD frame 0) -- LEAK")
    pool_files = [run_dir / "dataset.extxyz"] + sorted(run_dir.glob("al_iter*_labeled.extxyz"))
    pool: set[str] = set()
    for pf in pool_files:
        if pf.is_file() and pf.stat().st_size > 0:
            fps = frame_fingerprints(pf)
            out["pool_files"].append({"path": str(pf), "n_frames": len(fps)})
            pool |= fps
    out["n_pool_frames"] = len(pool)
    overlap = held & pool
    out["n_overlap"] = len(overlap)
    out["leak"] = len(overlap) > 0 or bool(out["contains_init_frame"])
    log(f"  {len(held)} held-out frames vs {len(pool)} pool frames: overlap {len(overlap)}"
        + (" -- LEAK" if overlap else " (independent)"))
    return out


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# ── 5. held-out accuracy through the engine's own modules ────────────────────
HELDOUT_EVAL_TEMPLATE = '''\
"""heldout_eval.py -- generated by scripts/distill_verify.py; rerunnable.

Evaluates a student .pt on a held-out extxyz through onthefly-distill's own
`ontheflydistill.common` (same NNMTP class, same neighbour-list cache, same
validate() that reports the trainer's E/F MAE) so the number is comparable to
the trainer's, but computed on frames the student never saw. The Fmax filter
is disabled: every held-out frame counts.
"""
import argparse, json, platform, sys
import numpy, torch
from torch.utils.data import DataLoader
from ontheflydistill import common

ap = argparse.ArgumentParser()
ap.add_argument("--model-pt", required=True)
ap.add_argument("--heldout", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cache-prefix", required=True)
ap.add_argument("--specorder", nargs="+", required=True)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(args.model_pt, map_location=device, weights_only=False)
cfg = ck["model_config"]
model = common.NNMTP(**cfg).to(device)
model.load_state_dict(ck["model_state_dict"])

common.FMAX_FILTER = float("inf")  # held-out: evaluate every frame
stats = common.build_cache(args.heldout, args.cache_prefix + ".cache.pt", args.cache_prefix + ".structure.data",
                           r_max=cfg["r_max"], specorder=args.specorder)
ds = common.CachedDataset(args.cache_prefix + ".cache.pt")
loader = DataLoader(ds, batch_size=common.BATCH_SIZE, shuffle=False, collate_fn=common.collate_fn)
e_mae, f_mae = common.validate(model, loader, device)
json.dump({
    "energy_mae_mev_per_atom": float(e_mae),
    "force_mae_mev_per_a": float(f_mae),
    "n_frames": int(len(ds)),
    "cache_stats": stats,
    "device": device,
    "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__},
    "model_pt": args.model_pt,
    "heldout": args.heldout,
}, open(args.out, "w"), indent=2)
print(f"held-out: E_MAE={e_mae:.2f} meV/atom F_MAE={f_mae:.2f} meV/A on {len(ds)} frames ({device})", flush=True)
'''


def run_heldout_eval(*, work: Path, python: Path, repo: Path, model_pt: str | None, heldout: Path,
                     specorder: list[str], log: Log) -> dict:
    vdir = work / "verify"
    vdir.mkdir(parents=True, exist_ok=True)
    script = vdir / "heldout_eval.py"
    script.write_text(HELDOUT_EVAL_TEMPLATE)
    metrics_path = vdir / "metrics.json"
    out = {"script": str(script), "metrics_path": str(metrics_path), "python": str(python), "ran": False,
           "returncode": None, "metrics": None, "error": None, "log": str(vdir / "heldout_eval.log")}
    if model_pt is None:
        out["error"] = "no final model .pt to evaluate"
        log(f"step 5/7: held-out evaluation skipped -- {out['error']}")
        return out
    if not heldout.is_file():
        out["error"] = f"held-out file {heldout} missing"
        log(f"step 5/7: held-out evaluation skipped -- {out['error']}")
        return out
    if metrics_path.exists():
        metrics_path.unlink()
    cmd = [str(python), str(script), "--model-pt", model_pt, "--heldout", str(heldout), "--out", str(metrics_path),
           "--cache-prefix", str(vdir / "heldout"), "--specorder", *specorder]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo}:{work}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    env["ONTHEFLY_CONFIG"] = str(work / "config.yaml")
    out["command"] = cmd
    log(f"step 5/7: held-out evaluation: {' '.join(cmd)}")
    try:
        with open(out["log"], "w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(vdir))
    except OSError as exc:
        out["error"] = f"could not launch {python}: {exc}"
        log(f"  {out['error']}")
        return out
    out["ran"] = True
    out["returncode"] = proc.returncode
    if proc.returncode != 0 or not metrics_path.is_file():
        out["error"] = f"heldout_eval.py exited {proc.returncode}; see {out['log']}"
        log(f"  {out['error']}")
        return out
    try:
        out["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        e, f = out["metrics"]["energy_mae_mev_per_atom"], out["metrics"]["force_mae_mev_per_a"]
        if not (np.isfinite(e) and np.isfinite(f)):
            out["error"] = f"non-finite held-out metrics E={e} F={f}"
            log(f"  {out['error']}")
        else:
            log(f"  E_MAE={e:.3f} meV/atom  F_MAE={f:.3f} meV/A  on {out['metrics'].get('n_frames')} frames")
    except (ValueError, KeyError, TypeError) as exc:
        out["error"] = f"metrics.json unreadable: {exc}"
        log(f"  {out['error']}")
    return out


# ── 6. lmp witness ───────────────────────────────────────────────────────────
def engine_md_evidence(md_dir: str | None, log: Log) -> dict:
    out = {"md_dir": md_dir, "md_log": None, "failure_json": None, "status": None, "last_ps": None,
           "reached_target": None}
    if md_dir is None:
        return out
    d = Path(md_dir)
    if (d / "md.log").is_file():
        out["md_log"] = str(d / "md.log")
    fj = d / "failure.json"
    if fj.is_file():
        out["failure_json"] = str(fj)
        try:
            info = json.loads(fj.read_text(encoding="utf-8"))
            out["status"] = info.get("status")
            out["last_ps"] = info.get("last_ps")
            out["reached_target"] = info.get("reached_target")
        except ValueError:
            out["status"] = "unreadable"
    log(f"step 6/7: engine's own final-round MD: {md_dir} status={out['status']} last_ps={out['last_ps']}")
    return out


def run_lmp_witness(*, work: Path, lmp_bin: Path, final_bin: str | None, md_dir: str | None,
                    specorder: list[str], masses: list[float], skip: bool, log: Log) -> dict:
    wdir = work / "verify" / "lmp_witness"
    out = {"dir": str(wdir), "lmp_bin": str(lmp_bin), "lmp_bin_sha256": _sha256(lmp_bin) if lmp_bin.is_file() else None,
           "skipped": skip, "ran": False, "returncode": None,
           "pe_ev": None, "log_error": None, "ok": False, "error": None}
    if skip:
        log("  lmp witness skipped by --no-lmp-witness (recorded; cannot pass)")
        return out
    if final_bin is None:
        out["error"] = "no final .bin"
        log(f"  lmp witness impossible: {out['error']}")
        return out
    data = Path(md_dir) / "structure_init.data" if md_dir else None
    if data is None or not data.is_file():
        out["error"] = f"structure_init.data missing under {md_dir} (the engine writes it per MD round)"
        log(f"  lmp witness impossible: {out['error']}")
        return out
    wdir.mkdir(parents=True, exist_ok=True)
    in_file = wdir / "witness.in"
    log_file = wdir / "witness.log"
    mass_lines = "\n".join(f"mass            {i + 1} {m}" for i, m in enumerate(masses))
    in_file.write_text(textwrap.dedent(f"""\
        # generated by scripts/distill_verify.py -- final student .bin under lmp, run 0
        # boundary = the engine's own student MD boundary ({WITNESS_BOUNDARY_SOURCE})
        units           metal
        atom_style      atomic
        atom_modify     map yes
        newton          on
        boundary        {WITNESS_BOUNDARY}
        read_data       {data}
        {mass_lines}
        pair_style      nnmtp
        pair_coeff      * * {Path(final_bin).resolve()} {' '.join(specorder)}
        thermo          1
        thermo_style    custom step pe
        run             0
        """))
    out["input"] = str(in_file)
    out["log"] = str(log_file)
    cmd = [str(lmp_bin), "-in", str(in_file)]
    log(f"  lmp witness: {' '.join(cmd)}")
    try:
        with open(log_file, "w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(wdir))
    except OSError as exc:
        out["error"] = f"could not launch {lmp_bin}: {exc}"
        log(f"  {out['error']}")
        return out
    out["ran"] = True
    out["returncode"] = proc.returncode
    text = log_file.read_text(encoding="utf-8", errors="ignore")
    out["log_error"] = ("ERROR" in text) or ("Lost atoms" in text)
    out["pe_ev"] = _parse_pe(text)
    out["ok"] = bool(proc.returncode == 0 and not out["log_error"] and out["pe_ev"] is not None
                     and np.isfinite(out["pe_ev"]))
    if not out["ok"]:
        out["error"] = f"returncode={proc.returncode} log_error={out['log_error']} pe={out['pe_ev']}"
    log(f"  lmp witness {'ok' if out['ok'] else 'FAILED'}: pe={out['pe_ev']} returncode={proc.returncode}")
    return out


def _parse_pe(text: str) -> float | None:
    """The `Step PotEng` thermo table: first numeric row after the header."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*Step\s+PotEng\b", line):
            for row in lines[i + 1:]:
                parts = row.split()
                if len(parts) >= 2 and re.match(r"^-?\d", parts[0]):
                    try:
                        return float(parts[1])
                    except ValueError:
                        return None
            return None
    return None


# ── 7. verdict ───────────────────────────────────────────────────────────────
def decide(acc: dict, contract: dict, status: dict, relabel: dict, heldout: dict, eval_out: dict,
           witness: dict, wallclock: dict | None = None) -> dict:
    """One state per run. Order of precedence: contract violations (al_loop
    drift, then any other provenance drift -- nothing under run/ was even
    inspected in that case), wall-clock evidence (unbound or missing =>
    failed; exhausted => unmet; a valid engine_exit -- the supervised run
    exited non-zero without expiring -- => failed(supervisor_exit), before
    any status line is believed), then the engine's stop reason, then foreign
    relabel files (malformed_evidence), then held-out integrity, then
    accuracy, then the witness (skipped => incomplete). Returns
    {state, reason, detail, accuracy[, proposal]}."""
    mode = acc["mode"]
    a = acc["accuracy"]
    metrics = eval_out.get("metrics") or {}
    e, f = metrics.get("energy_mae_mev_per_atom"), metrics.get("force_mae_mev_per_a")
    within = (e is not None and f is not None and e <= a["energy_mae_max_mev_per_atom"]
              and f <= a["force_mae_max_mev_per_a"])
    accuracy = {"energy_mae_mev_per_atom": e, "force_mae_mev_per_a": f,
                "energy_mae_max_mev_per_atom": a["energy_mae_max_mev_per_atom"],
                "force_mae_max_mev_per_a": a["force_mae_max_mev_per_a"], "within_thresholds": within}

    if contract["al_loop_drift"]:
        return {"state": "failed", "reason": "config_drift",
                "detail": "config.yaml al_loop.* differs from the approved acceptance.json values; "
                          "a change past the approved values is out of scope (recipes/distill.md §6)",
                "accuracy": accuracy}
    if contract.get("provenance_drift"):
        names = sorted(contract["provenance_drift"])
        return {"state": "failed", "reason": "provenance_drift",
                "detail": f"sha256 recorded at approval != on disk for {names}; the run is not the approved one "
                          "(re-render and re-approve the proposal)", "accuracy": accuracy}
    if wallclock and not wallclock.get("valid"):
        return {"state": "failed", "reason": "wallclock_evidence",
                "detail": "the approved wall-clock bound has no acceptable enforcement evidence: "
                          + "; ".join(wallclock.get("problems") or []), "accuracy": accuracy}
    if wallclock and wallclock.get("enforced"):
        why = wallclock.get("attribution")
        why_note = WALLCLOCK_ATTRIBUTION.get(why, "")
        if wallclock.get("exhausted"):
            return {"state": "unmet", "reason": "budget:wallclock",
                    "detail": f"run_distill.sh's wall-clock bound ({wallclock.get('limit_s')} s) expired after "
                              f"{wallclock.get('elapsed_s')} s (rc={wallclock.get('returncode')}, {why}: {why_note}); "
                              f"the run's process group was terminated, artifacts kept; .al_status = "
                              f"{status['line']!r} is not a pass under an exhausted budget", "accuracy": accuracy,
                    "proposal": _proposal(acc, relabel, wallclock=True)}
        if wallclock.get("pending") and status["terminal"]:
            return {"state": "failed", "reason": "wallclock_evidence",
                    "detail": f"attempt {wallclock.get('attempt')} started and .al_status is terminal "
                              f"({status['line']!r}) but the supervisor recorded no outcome marker at "
                              f"{wallclock.get('marker')}: the bound cannot be shown to have held", "accuracy": accuracy}
        if wallclock.get("valid") and wallclock.get("state") == "engine_exit":
            # The supervised run exited non-zero without expiring: whatever
            # .al_status says (even SUCCESS -- the loop may have written it
            # before something later died, or it may be an earlier line), the
            # attempt did not complete cleanly and nothing is evaluated.
            return {"state": "failed", "reason": "supervisor_exit",
                    "detail": f"run_distill.sh's supervised run exited rc={wallclock.get('returncode')} after "
                              f"{wallclock.get('elapsed_s')} s of {wallclock.get('limit_s')} s ({why}: {why_note}); "
                              f".al_status = {status['line']!r} is not this attempt's clean completion; per-round "
                              "logs kept under run/, no metrics or witness computed", "accuracy": accuracy}
    if status["kind"] in ("missing", "running"):
        return {"state": "incomplete", "reason": "loop_not_terminal",
                "detail": f".al_status = {status['line']!r}", "accuracy": accuracy}
    if status["kind"] == "oneshot":
        return {"state": "failed", "reason": "teacher_cannot_relabel",
                "detail": "the engine took its one-shot branch (can_relabel != '1'): no active learning ran. "
                          "Usually stdout contamination of D/scripts/can_relabel.py by the teacher's banner "
                          "(see distill_bootstrap.render_omm_teacher) or a teacher.type that cannot label.",
                "accuracy": accuracy}
    if status["kind"] in ("failed", "error"):
        return {"state": "failed", "reason": f"engine_{status['kind']}",
                "detail": f".al_status = {status['line']!r}; per-round logs kept under run/", "accuracy": accuracy}
    if relabel.get("malformed_files"):
        return {"state": "failed", "reason": "malformed_evidence",
                "detail": f"{relabel['malformed_files']} match the trainer's al_iter*_labeled.extxyz glob (so the "
                          "student was trained on them) but not the engine's al_iter<K>_labeled naming: foreign "
                          "pool data, not this loop's relabel rounds", "accuracy": accuracy}
    if heldout["leak"] or not heldout["outside_engine_work_dir"] or not heldout["seed_distinct"]:
        why = ("a held-out frame is the init structure (teacher_md.py's pre-MD frame 0, the pool's first frame)"
               if heldout.get("contains_init_frame") else
               f"{heldout['n_overlap']} frame(s) shared with the AL pool" if heldout["leak"] else
               "path inside the engine work dir" if not heldout["outside_engine_work_dir"] else
               "held-out seed equals the pool seed")
        return {"state": "failed", "reason": "heldout_leak", "detail": why, "accuracy": accuracy}
    if not heldout["exists"]:
        return {"state": "failed", "reason": "heldout_missing", "detail": heldout["path"], "accuracy": accuracy}
    if eval_out.get("metrics") is None or eval_out.get("error"):
        return {"state": "failed", "reason": "heldout_eval", "detail": eval_out.get("error"), "accuracy": accuracy}

    witness_ok = bool(witness.get("ok"))
    # A deliberately skipped witness (--no-lmp-witness) is missing evidence,
    # not a failure: incomplete, so the ledger neither passes nor blames it.
    witness_verdict = ({"state": "incomplete", "reason": "lmp_witness_skipped",
                        "detail": "the lmp run-0 witness was skipped by --no-lmp-witness; rerun verify with lmp "
                                  "available to complete the verdict", "accuracy": accuracy}
                       if witness.get("skipped") and not witness.get("ran") else
                       {"state": "failed", "reason": "lmp_witness", "detail": witness.get("error") or "not run",
                        "accuracy": accuracy})
    if mode == "fixture":
        if not (relabel["n_relabel_rounds"] >= 1 and relabel["retrain_after_relabel"]):
            return {"state": "unmet", "reason": "fixture:no_relabel",
                    "detail": f"{relabel['n_relabel_rounds']} relabel round(s), retrain_after_relabel="
                              f"{relabel['retrain_after_relabel']}; the fixture did not exercise crash -> relabel -> "
                              "retrain. Shrink --pool-steps or raise --target-ps and rerun (fixture bound unchanged).",
                    "accuracy": accuracy}
        if not witness_ok:
            return witness_verdict
        return {"state": "passed", "reason": f"fixture: {relabel['n_relabel_rounds']} relabel round(s), "
                                              f"terminal {status['kind']}, held-out metrics computed, lmp witness ok",
                "detail": None, "accuracy": accuracy}

    # production
    if status["kind"] == "stalled":
        return {"state": "unmet", "reason": "stability:stalled",
                "detail": f"no crash-time progress for {acc['budget']['no_progress_limit']} rounds "
                          f"({status['line']}); artifacts kept", "accuracy": accuracy,
                "proposal": _proposal(acc, relabel)}
    if status["kind"] in ("backstop", "label_fail"):
        return {"state": "unmet", "reason": f"budget:{status['kind']}",
                "detail": f"{status['line']}; artifacts kept", "accuracy": accuracy,
                "proposal": _proposal(acc, relabel)}
    assert status["kind"] == "success", status
    if not within:
        return {"state": "unmet", "reason": "accuracy",
                "detail": f"stable to {acc['stability']['target_ps']} ps at round {status['round']} but held-out "
                          f"E_MAE={e:.3f} (max {a['energy_mae_max_mev_per_atom']}) meV/atom, "
                          f"F_MAE={f:.3f} (max {a['force_mae_max_mev_per_a']}) meV/A",
                "accuracy": accuracy, "proposal": _proposal(acc, relabel)}
    if not witness_ok:
        return witness_verdict
    return {"state": "passed",
            "reason": f"SUCCESS at round {status['round']} (stable {acc['stability']['target_ps']} ps), "
                      f"held-out E_MAE={e:.3f} meV/atom F_MAE={f:.3f} meV/A within thresholds, lmp witness ok",
            "detail": None, "accuracy": accuracy}


def _proposal(acc: dict, relabel: dict, *, wallclock: bool = False) -> dict:
    """A reconfigured rerun INSIDE the approved budget class -- more pool data
    and/or more rounds (or, for an exhausted clock, more hours) -- as a
    proposal needing approval, never executed here."""
    b = acc["budget"]
    pool = acc["split"]["pool"]
    options = [
        {"change": "larger initial pool", "flag": f"--pool-steps {pool['steps'] * 2} --pool-save-every {pool['save_every']}"},
        {"change": "more AL rounds", "flag": f"--max-iter {b['max_iter'] * 2} --no-progress-limit {b['no_progress_limit'] + 2}"},
    ]
    if wallclock and b.get("wallclock_max_h"):
        options.insert(0, {"change": "longer wall-clock bound", "flag": f"--wallclock-max-h {b['wallclock_max_h'] * 2}"})
    return {
        "needs_approval": True,
        "options": options,
        "keep": "teacher, structure, target_ps, thresholds, seed (changing any is a new scope)",
        "rounds_used": relabel["n_relabel_rounds"],
    }


# ── output ───────────────────────────────────────────────────────────────────
def append_ledger(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def main(argv=None) -> int:
    args = parse_args(argv)
    work = args.work.resolve()
    log = Log(work / "verify" / "verify.log")
    log(f"verify {work} at {utc_now()}")
    acc, cfg, contract = load_contract(work, log)
    drifted = bool(contract["al_loop_drift"] or contract["provenance_drift"])
    if drifted:
        # `work_dir` (and python_bin, lmp_bin, system.*) come from the LIVE
        # config.yaml: a drifted config could point every inspection below at
        # another run's directory. Nothing under run/ is read; the verdict is
        # the drift itself.
        log("  contract drift: stopping before any run directory is derived from the drifted config -- "
            "status, rounds, models, held-out and wall-clock evidence are NOT inspected")
        run_dir = None
        wallclock = {"limit_h": (acc.get("budget") or {}).get("wallclock_max_h"), "enforced": None,
                     "attempt_record": None, "marker": None, "attempt": None, "state": None, "elapsed_s": None,
                     "returncode": None, "limit_s": None, "exhausted": False, "valid": False, "pending": False,
                     "problems": ["not inspected: contract drift"]}
        status = {"path": None, "line": None, "kind": "not_inspected", "round": None, "terminal": False,
                  "detail": "contract drift"}
        relabel = {"labeled_files": [], "n_relabel_rounds": 0, "malformed_files": [], "student_models": [],
                   "retrain_after_relabel": False}
        final = {"round": None, "bin": None, "pt": None, "md_dir": None}
        heldout = {"path": acc["split"]["heldout"]["path"], "exists": False, "sha256": None, "leak": None,
                   "outside_engine_work_dir": None, "seed_distinct": None, "n_overlap": None,
                   "contains_init_frame": None}
        evaluate = False
    else:
        run_dir = Path(cfg.get("work_dir") or (work / "run"))
        wallclock = wallclock_evidence(work, acc, run_dir, log)
        status = classify_status(run_dir, log)
        relabel = relabel_rounds(run_dir, log)
        final = final_model(run_dir, status, log)
        heldout = check_heldout(acc, run_dir, log)
        # No evaluation subprocess (held-out torch run, lmp witness) is launched
        # unless every earlier gate is clear -- a supervised run that exited
        # non-zero (`engine_exit`) is judged failed(supervisor_exit) without
        # touching its artifacts, whatever .al_status says.
        evaluate = (status["terminal"] and status["kind"] not in ("failed", "error", "oneshot")
                    and not relabel["malformed_files"]
                    and wallclock["valid"] and not wallclock["exhausted"] and not wallclock["pending"]
                    and wallclock["state"] != "engine_exit"
                    and heldout["exists"] and not heldout["leak"]
                    and heldout["outside_engine_work_dir"] and heldout["seed_distinct"])
    repo = (args.repo or Path(acc["provenance"]["engine"]["repo"])).resolve()
    python = args.python or Path(cfg.get("python_bin") or sys.executable)
    lmp_bin = args.lmp_bin or Path(cfg.get("lmp_bin") or acc["provenance"]["lmp_bin"])
    specorder = [str(s) for s in (cfg.get("system") or {}).get("specorder", [])]
    masses = [float(m) for m in (cfg.get("system") or {}).get("masses", [])]
    if evaluate:
        eval_out = run_heldout_eval(work=work, python=python, repo=repo, model_pt=final["pt"],
                                    heldout=Path(heldout["path"]), specorder=specorder, log=log)
        md_evidence = engine_md_evidence(final.get("md_dir"), log)
        witness = run_lmp_witness(work=work, lmp_bin=lmp_bin, final_bin=final["bin"], md_dir=final.get("md_dir"),
                                  specorder=specorder, masses=masses, skip=args.no_lmp_witness, log=log)
    else:
        log("step 5/7 + 6/7: held-out evaluation and lmp witness not run (verdict is decided earlier)")
        eval_out = {"ran": False, "metrics": None, "error": "not run"}
        md_evidence = engine_md_evidence(final.get("md_dir"), log)
        witness = {"ran": False, "ok": False, "skipped": args.no_lmp_witness, "error": "not run"}

    verdict = decide(acc, contract, status, relabel, heldout, eval_out, witness, wallclock)
    log(f"step 7/7: state={verdict['state']} reason={verdict['reason']}")
    # What each gate actually looked at -- machine-readable, so a `passed`
    # row is never read as "the deployed .bin is accurate".
    artifacts = {
        "accuracy_artifact": final.get("pt"),
        "accuracy_scope": "held-out energy/force MAE of the torch checkpoint (.pt) through the engine's validate()",
        "witness_artifact": final.get("bin"),
        "witness_scope": "lmp run 0 of the exported nnmtp binary (.bin): exit 0, finite energy, no ERROR -- "
                         "finiteness/stability only, NOT accuracy equivalence with the .pt",
        "witness_state": ("skipped" if witness.get("skipped") and not witness.get("ran")
                          else "ok" if witness.get("ok") else "failed" if witness.get("ran") else "not_run"),
        "witness_boundary": WITNESS_BOUNDARY,
        "witness_boundary_source": WITNESS_BOUNDARY_SOURCE,
        "accuracy_pbc": ACCURACY_PBC,
        "boundary_note": "the accuracy gate (periodic labels) and the witness (p p f, the deployment boundary) use "
                         "different z boundary conditions by the engine's own design; disclosed, not reconciled",
    }

    report = {
        "schema": "oh-my-mlip.distill.verify/1",
        "utc": utc_now(),
        "work": str(work),
        "mode": acc["mode"],
        "state": verdict["state"],
        "reason": verdict["reason"],
        "detail": verdict.get("detail"),
        "proposal": verdict.get("proposal"),
        "accuracy": verdict["accuracy"],
        "artifacts": artifacts,
        "stability": {"target_ps": acc["stability"]["target_ps"], "al_status": status["line"],
                      "kind": status["kind"], "round": status["round"], "terminal": status["terminal"]},
        "budget": acc["budget"],
        "wallclock": wallclock,
        "relabel": relabel,
        "final_model": final,
        "heldout": heldout,
        "heldout_eval": eval_out,
        "engine_md_evidence": md_evidence,
        "lmp_witness": witness,
        "contract": contract,
        "provenance": acc.get("provenance"),
        "manifest_sha256": args.manifest_sha256,
        "campaign_id": args.campaign_id,
    }
    report_path = work / "verify" / "distill_verify.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    ledger = args.ledger or (work / ".distill" / "ledger.jsonl")
    teacher = (acc.get("provenance") or {}).get("teacher") or {}
    # Same keys the other campaign ledgers carry (env/family/variant/kind/phase/
    # state/verdict/evidence/manifest_sha256/campaign_id) so evidence_report.py
    # keys the row as ("distill", <teacher variant>); any state other than
    # "passed" reads there as not-passed, and a missing manifest as INCOMPLETE.
    row = {"utc": report["utc"], "env": teacher.get("env"), "family": teacher.get("model"),
           "variant": teacher.get("version"), "kind": "distill", "phase": "verify", "mode": acc["mode"],
           "teacher": teacher.get("version"), "state": verdict["state"],
           "verdict": {"reason": verdict["reason"], "detail": verdict.get("detail"), "degraded": False,
                       "device": (eval_out.get("metrics") or {}).get("device"), "artifacts": artifacts},
           "returncode": 0 if verdict["state"] == "passed" else 1,
           "stderr_tail": None if verdict["state"] == "passed" else f"{verdict['reason']}: {verdict.get('detail')}",
           "evidence": {"report": str(report_path), "al_status": status["line"], "final_bin": final["bin"],
                        "metrics": eval_out.get("metrics_path"), "lmp_witness": witness.get("log"),
                        "lmp_bin_sha256": witness.get("lmp_bin_sha256"),
                        "heldout_sha256": heldout["sha256"], "n_relabel_rounds": relabel["n_relabel_rounds"],
                        "wallclock": wallclock["state"], "wallclock_marker": wallclock["marker"],
                        "wallclock_attempt": wallclock["attempt"], "wallclock_valid": wallclock["valid"],
                        "wallclock_attribution": wallclock.get("attribution"),
                        "wallclock_returncode": wallclock.get("returncode"),
                        "provenance_drift": sorted(contract["provenance_drift"]),
                        "malformed_files": relabel.get("malformed_files") or [],
                        "device": (eval_out.get("metrics") or {}).get("device")},
           "manifest_sha256": args.manifest_sha256, "campaign_id": args.campaign_id, "work": str(work)}
    append_ledger(ledger, row)
    log(f"report {report_path}; ledger row -> {ledger}")
    if args.json:
        print(json.dumps({k: report[k] for k in ("state", "reason", "detail", "mode", "accuracy", "stability")}
                         | {"report": str(report_path)}))
    return 0 if verdict["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
