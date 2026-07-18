#!/usr/bin/env python3
"""setup_sweep.py -- deterministic batch driver for `setup all` sweeps.

Complete-then-batch-recover (the consensus decision): the driver NEVER stops
on a failed target. Every phase of every target is appended to a JSONL ledger,
and the final report is generated STRICTLY from that ledger -- never from
anyone's memory of what happened. Agent recovery reasoning runs only after
the sweep completes, over the ledger's `failed` entries.

Per target (sequential -- parallel conda solves corrupt the error signal):
  1. gated bookkeeping: a gated model with NO HF token available is recorded
     `skipped_gated` (the approval gate upstream should already have excluded
     it; double bookkeeping is intentional honesty). With a token it proceeds.
  2. `./install.sh <env>`   -> ledger line (phase "install")
  3. `setup_verify.py <target> --json` -> ledger line (phase "verify",
     verdict embedded). setup_verify owns ALL verify judgment; this driver
     records returncodes and verdicts, it never re-judges.

The sweep itself always exits 0 (completion is the contract; per-target facts
live in the ledger -- same convention as setup_survey/setup_guardrail).

Usage:
  python3 scripts/setup_sweep.py --targets MACE,SevenNet,ORB
  python3 scripts/setup_sweep.py report [--ledger PATH]

Test-only: --install-cmd / --verify-cmd replace the real commands (explicit
injection, no monkeypatching); the target/env is appended as the last arg.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup_common import resolve_home, utc_now  # noqa: E402
from setup_survey import token_source  # noqa: E402
from setup_verify import find_env  # noqa: E402

STDERR_TAIL_CHARS = 2000


def next_ledger_path(home: Path) -> Path:
    sweep_dir = home / ".sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(sweep_dir.glob("setup_sweep_*.jsonl"))
    if existing:
        last = existing[-1].stem.rsplit("_", 1)[-1]
        run_id = int(last) + 1 if last.isdigit() else len(existing) + 1
    else:
        run_id = 1
    return sweep_dir / f"setup_sweep_{run_id:04d}.jsonl"


def latest_ledger_path(home: Path) -> Path | None:
    existing = sorted((home / ".sweep").glob("setup_sweep_*.jsonl"))
    return existing[-1] if existing else None


def append(ledger: Path, record: dict) -> None:
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def is_gated(model: str, home: Path) -> bool:
    registry = json.loads((home / "models.json").read_text())
    want = model.lower()
    for family, spec in registry.items():
        if family.startswith("_"):
            continue
        versions = spec.get("versions") or {}
        if family.lower() == want:
            return any(bool(v.get("gated")) for v in versions.values())
        for version, v in versions.items():
            if version.lower() == want:
                return bool(v.get("gated"))
    return False


def run_phase(command: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    return proc.returncode, proc.stdout, proc.stderr[-STDERR_TAIL_CHARS:]


def last_json_line(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def sweep(targets: list[str], home: Path, ledger: Path,
          install_cmd: list[str] | None, verify_cmd: list[str] | None,
          min_free_gb: float = 10.0) -> None:
    append(ledger, {"seq": 0, "phase": "plan", "targets": targets, "at": utc_now()})
    token_missing = token_source() == "none"
    seq = 0
    for idx, target in enumerate(targets):
        # Disk floor precheck: below min_free_gb every further install is a
        # guaranteed noisy failure, so record THIS and all remaining targets
        # as skipped_disk (honest ledger, no silent truncation) and stop.
        free_gb = shutil.disk_usage(home).free / 1024**3
        if free_gb < min_free_gb:
            for remaining in targets[idx:]:
                seq += 1
                append(ledger, {"seq": seq, "target": remaining,
                                "env": find_env(remaining, home),
                                "phase": "skipped_disk", "returncode": None,
                                "stderr_tail": f"free disk {free_gb:.1f} GB < {min_free_gb:.1f} GB floor",
                                "verdict": None})
            break
        env_name = find_env(target, home)
        if env_name is None:
            seq += 1
            append(ledger, {"seq": seq, "target": target, "env": None,
                            "phase": "resolve", "returncode": 1,
                            "stderr_tail": f"unknown model: {target}", "verdict": None})
            continue
        if is_gated(target, home) and token_missing:
            seq += 1
            append(ledger, {"seq": seq, "target": target, "env": env_name,
                            "phase": "skipped_gated", "returncode": None,
                            "stderr_tail": "", "verdict": None})
            continue

        install = (install_cmd or [str(home / "install.sh")]) + [env_name]
        rc, _out, err = run_phase(install, home)
        seq += 1
        append(ledger, {"seq": seq, "target": target, "env": env_name,
                        "phase": "install", "returncode": rc,
                        "stderr_tail": err, "verdict": None})
        if rc != 0:
            continue  # NEVER stops the sweep; this target is simply failed

        verify = (verify_cmd or [sys.executable,
                                 str(home / "scripts" / "setup_verify.py")]) + [target, "--json"]
        rc, out, err = run_phase(verify, home)
        seq += 1
        append(ledger, {"seq": seq, "target": target, "env": env_name,
                        "phase": "verify", "returncode": rc,
                        "stderr_tail": err, "verdict": last_json_line(out)})


def target_status(records: list[dict]) -> str:
    """Status of one target, strictly from its ledger records."""
    if not records:
        return "not_attempted"
    if any(r["phase"] == "skipped_gated" for r in records):
        return "skipped_gated"
    if any(r["phase"] == "skipped_disk" for r in records):
        return "skipped_disk"
    verifies = [r for r in records if r["phase"] == "verify"]
    if verifies:
        verdict = verifies[-1].get("verdict") or {}
        if verdict.get("pass") and verdict.get("degraded"):
            return "degraded"
        if verdict.get("pass"):
            return "verified"
    return "failed"


def report(ledger: Path) -> str:
    lines = [json.loads(ln) for ln in ledger.read_text().splitlines() if ln.strip()]
    plan = next((ln for ln in lines if ln.get("phase") == "plan"), None)
    targets = plan["targets"] if plan else sorted({ln.get("target") for ln in lines if ln.get("target")})
    out = [f"setup_sweep report -- ledger: {ledger}"]
    counts: dict[str, int] = {}
    for target in targets:
        records = [ln for ln in lines if ln.get("target") == target]
        status = target_status(records)
        counts[status] = counts.get(status, 0) + 1
        detail = ""
        if status == "failed":
            last = records[-1]
            detail = f"  ({last['phase']} rc={last['returncode']})"
        out.append(f"  {target:<22} {status}{detail}")
    out.append("  " + " / ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="sweep", choices=("sweep", "report"))
    ap.add_argument("--targets", default="", help="comma-separated model names (families or versions)")
    ap.add_argument("--ledger", default=None, help="ledger path (default: new for sweep, latest for report)")
    ap.add_argument("--install-cmd", default=None, help="TEST ONLY: replacement install command")
    ap.add_argument("--verify-cmd", default=None, help="TEST ONLY: replacement verify command")
    args = ap.parse_args()

    home = resolve_home()
    if args.mode == "report":
        ledger = Path(args.ledger) if args.ledger else latest_ledger_path(home)
        if ledger is None or not ledger.exists():
            print("no sweep ledger found", file=sys.stderr)
            return 1
        print(report(ledger))
        return 0

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        print("no targets given (--targets M1,M2,...)", file=sys.stderr)
        return 1
    ledger = Path(args.ledger) if args.ledger else next_ledger_path(home)
    sweep(
        targets, home, ledger,
        args.install_cmd.split() if args.install_cmd else None,
        args.verify_cmd.split() if args.verify_cmd else None,
    )
    print(report(ledger))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
