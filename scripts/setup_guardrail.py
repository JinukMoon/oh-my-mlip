"""
setup_guardrail.py -- Deterministic safety-stop helper for the oh-my-mlip self-healing
install loop (Option B-prime).

BOUNDARY (B-prime contract):
  This module owns ONLY the non-negotiable, deterministic safety stops:
    - disk headroom check (disk-check)
    - identical-error-signature stall detection (record-attempt)
    - scoped cache cleanup (clean-cache)
    - combined gate verdict (gate)

  Recovery reasoning -- reading the traceback, classifying the error class, selecting
  retry strategies -- belongs ENTIRELY to the agent via AGENTS.md.  Nothing in this
  file encodes error taxonomy, framework-specific heuristics, or retry strategies.

Usage:
  python3 scripts/setup_guardrail.py disk-check --ceiling-gb 30 [--path PATH]
  python3 scripts/setup_guardrail.py record-attempt --state FILE --stderr-file FILE
  python3 scripts/setup_guardrail.py clean-cache [--dry-run]
  python3 scripts/setup_guardrail.py gate --state FILE --ceiling-gb 30 --stderr-file FILE

All subcommands print a single JSON object to stdout and exit 0 (even on stall/halt so
the caller can parse the verdict rather than relying on exit codes).
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCRUB_PATTERNS = [
    # Absolute paths:  /home/foo/bar/baz.py  or  /tmp/abc
    (re.compile(r"/[^\s:,\"\']+"), "<PATH>"),
    # Windows-style paths (just in case)
    (re.compile(r"[A-Za-z]:\\[^\s,\"\']+"), "<PATH>"),
    # Line numbers: "line 42", ":42:", ":42 "
    (re.compile(r"\bline\s+\d+\b", re.IGNORECASE), "<LINE>"),
    (re.compile(r":\d+[:\s]"), " "),
    # Hex addresses:  0x7f3a4b5c  or  0X...
    (re.compile(r"\b0[xX][0-9a-fA-F]+\b"), "<ADDR>"),
    # PIDs:  "pid 12345"  or  "PID=12345"
    (re.compile(r"\bpid[=:\s]+\d+\b", re.IGNORECASE), "<PID>"),
    # ISO timestamps  2024-01-02T03:04:05  or  2024-01-02 03:04:05
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"), "<TS>"),
    # Plain digit runs of 4+ digits that are not version-like (keep 1.2.3 readable)
    (re.compile(r"(?<![.\w])\d{4,}(?![.\w])"), "<NUM>"),
]


def _normalize_line(line: str) -> str:
    """Apply scrub patterns then lower-case and collapse whitespace."""
    s = line
    for pattern, repl in _SCRUB_PATTERNS:
        s = pattern.sub(repl, s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _stderr_signature(text: str, tail: int = 20) -> str:
    """
    Return a short stable hash for the meaningful tail of a stderr blob.

    Steps:
    1. Split into lines, strip blank/trivial lines (pure whitespace, progress bars,
       lines that are only hyphens/equals/dots).
    2. Keep the last `tail` meaningful lines.
    3. Normalize each line (scrub paths, line-numbers, addresses, pids, timestamps,
       digit runs; lowercase; collapse whitespace).
    4. SHA-256 of the joined result; return first 12 hex chars.
    """
    lines = text.splitlines()

    _trivial = re.compile(r"^[\s\-=\.#*|/\\>]+$")

    meaningful = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            continue
        if _trivial.match(stripped):
            continue
        meaningful.append(stripped)

    tail_lines = meaningful[-tail:] if len(meaningful) >= tail else meaningful
    normalized = [_normalize_line(ln) for ln in tail_lines]
    blob = "\n".join(normalized)
    digest = hashlib.sha256(blob.encode("ascii", errors="replace")).hexdigest()
    return digest[:12]


def _read_state(path: Path) -> dict:
    if path.exists():
        try:
            with path.open("r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"attempts": [], "signature_counts": {}, "first_attempt_ts": None}


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(state, f, indent=2)

def _evaluate_attempt(state: dict, sig: str, max_repeat: int,
                      cumulative_max: int, wallclock_max_s) -> dict:
    """Record one attempt into ``state`` and evaluate all four stop conditions.

    Stop conditions (first one that trips wins; signature-agnostic ones below
    keep the loop bounded even when the agent retries with a *different*
    strategy each round, which produces a fresh signature every time):
      - signature_stall : same normalized stderr signature recurred >= max_repeat
      - cumulative_stall: total attempts reached cumulative_max (signature-agnostic)
      - wallclock_halt  : now - first_attempt_ts >= wallclock_max_s (signature-agnostic)
    Disk headroom is evaluated by the caller (disk-check / gate) and is the 4th.
    """
    now = time.time()
    counts: dict = state.setdefault("signature_counts", {})
    attempts: list = state.setdefault("attempts", [])
    if not state.get("first_attempt_ts"):
        state["first_attempt_ts"] = now

    counts[sig] = counts.get(sig, 0) + 1
    repeat_count = counts[sig]
    attempts.append({"signature": sig, "repeat_count": repeat_count, "ts": now})

    total_attempts = len(attempts)
    elapsed_s = now - state["first_attempt_ts"]

    signature_stalled = repeat_count >= max_repeat
    cumulative_stalled = total_attempts >= cumulative_max
    wallclock_exceeded = wallclock_max_s is not None and elapsed_s >= wallclock_max_s

    return {
        "signature": sig,
        "repeat_count": repeat_count,
        "max_repeat": max_repeat,
        "total_attempts": total_attempts,
        "cumulative_max": cumulative_max,
        "elapsed_s": round(elapsed_s, 1),
        "wallclock_max_s": wallclock_max_s,
        "signature_stalled": signature_stalled,
        "cumulative_stalled": cumulative_stalled,
        "wallclock_exceeded": wallclock_exceeded,
    }


def _default_state_path(model: str = "unknown") -> Path:
    base = Path(".gjc/state")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"setup_guardrail-{model}.json"


def _print_json(obj: dict) -> None:
    print(json.dumps(obj, indent=2))


# ---------------------------------------------------------------------------
# Subcommand: disk-check
# ---------------------------------------------------------------------------

def cmd_disk_check(args: argparse.Namespace) -> None:
    check_path = Path(args.path).resolve() if args.path else Path.cwd().resolve()

    # Walk up until we find a mountpoint that exists
    p = check_path
    while not p.exists():
        parent = p.parent
        if parent == p:
            break
        p = parent

    usage = shutil.disk_usage(str(p))
    free_gb = usage.free / (1024 ** 3)
    ceiling_gb = args.ceiling_gb
    status = "ok" if free_gb >= ceiling_gb else "low"
    _print_json({
        "status": status,
        "free_gb": round(free_gb, 2),
        "ceiling_gb": ceiling_gb,
        "checked_path": str(p),
    })


# ---------------------------------------------------------------------------
# Subcommand: record-attempt
# ---------------------------------------------------------------------------

def cmd_record_attempt(args: argparse.Namespace) -> None:
    state_path = Path(args.state)
    max_repeat = args.max_repeat

    # Read stderr
    if args.stderr_file == "-":
        stderr_text = sys.stdin.read()
    else:
        try:
            with open(args.stderr_file, "r", errors="replace") as f:
                stderr_text = f.read()
        except OSError as e:
            _print_json({"error": f"cannot read stderr_file: {e}"})
            return

    sig = _stderr_signature(stderr_text)

    state = _read_state(state_path)
    ev = _evaluate_attempt(state, sig, max_repeat,
                           args.cumulative_max, args.wallclock_max_s)
    _write_state(state_path, state)

    if ev["signature_stalled"]:
        status = "stalled"
    elif ev["cumulative_stalled"]:
        status = "stalled_cumulative"
    elif ev["wallclock_exceeded"]:
        status = "wallclock_halt"
    else:
        status = "continue"
    _print_json({
        "status": status,
        "signature": ev["signature"],
        "repeat_count": ev["repeat_count"],
        "max_repeat": ev["max_repeat"],
        "total_attempts": ev["total_attempts"],
        "cumulative_max": ev["cumulative_max"],
        "elapsed_s": ev["elapsed_s"],
        "wallclock_max_s": ev["wallclock_max_s"],
    })


# ---------------------------------------------------------------------------
# Subcommand: clean-cache
# ---------------------------------------------------------------------------

def _hf_cache_roots() -> list:
    """Return candidate HuggingFace cache directories."""
    roots = []
    hf_home = os.environ.get("HF_HOME", "")
    if hf_home:
        roots.append(Path(hf_home))
    roots.append(Path.home() / ".cache" / "huggingface")
    return roots


def _collect_torch_extensions(dry_run: bool) -> tuple:
    """Collect (and optionally delete) ~/.cache/torch_extensions contents."""
    removed = []
    freed = 0
    te_dir = Path.home() / ".cache" / "torch_extensions"
    if te_dir.exists():
        for child in list(te_dir.iterdir()):
            try:
                size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file()) if child.is_dir() else child.stat().st_size
                removed.append({"path": str(child), "bytes": size})
                freed += size
                if not dry_run:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            except OSError:
                pass
    return removed, freed


def _collect_hf_partials(dry_run: bool) -> tuple:
    """Collect (and optionally delete) HF partial/incomplete/lock files."""
    import glob as _glob
    removed = []
    freed = 0
    for root in _hf_cache_roots():
        if not root.exists():
            continue
        for pattern in ("**/*.incomplete", "**/*.lock"):
            for hit in root.glob(pattern):
                try:
                    size = hit.stat().st_size
                    removed.append({"path": str(hit), "bytes": size})
                    freed += size
                    if not dry_run:
                        hit.unlink(missing_ok=True)
                except OSError:
                    pass
    return removed, freed


def cmd_clean_cache(args: argparse.Namespace) -> None:
    dry_run = args.dry_run
    te_removed, te_freed = _collect_torch_extensions(dry_run)
    hf_removed, hf_freed = _collect_hf_partials(dry_run)
    total_freed = te_freed + hf_freed
    _print_json({
        "dry_run": dry_run,
        "action": "would_remove" if dry_run else "removed",
        "torch_extensions": {"items": te_removed, "bytes_freed": te_freed},
        "hf_partials": {"items": hf_removed, "bytes_freed": hf_freed},
        "total_bytes_freed": total_freed,
    })


# ---------------------------------------------------------------------------
# Subcommand: gate
# ---------------------------------------------------------------------------

def cmd_gate(args: argparse.Namespace) -> None:
    """
    Combined disk-check + record-attempt. Returns a single verdict (first stop
    condition that trips wins):
      ok                 - disk fine, no stall condition tripped
      stalled            - same error signature repeated >= max_repeat times
      stalled_cumulative - total attempts reached cumulative_max (signature-agnostic;
                           bounds the loop even when each retry uses a different
                           strategy and thus a fresh signature)
      wallclock_halt     - elapsed since first attempt >= wallclock_max_s
      guardrail_halt     - disk headroom below ceiling (highest priority)
    These four conditions are the bounded-self-heal stop set (AGENTS.md section 8).
    """
    # 1. Disk check
    check_path = Path(args.path).resolve() if hasattr(args, "path") and args.path else Path.cwd().resolve()
    p = check_path
    while not p.exists():
        parent = p.parent
        if parent == p:
            break
        p = parent
    usage = shutil.disk_usage(str(p))
    free_gb = usage.free / (1024 ** 3)
    disk_ok = free_gb >= args.ceiling_gb

    # 2. Record attempt / stall check
    state_path = Path(args.state)
    max_repeat = getattr(args, "max_repeat", 2)

    if args.stderr_file == "-":
        stderr_text = sys.stdin.read()
    else:
        try:
            with open(args.stderr_file, "r", errors="replace") as f:
                stderr_text = f.read()
        except OSError as e:
            _print_json({"verdict": "error", "detail": f"cannot read stderr_file: {e}"})
            return

    sig = _stderr_signature(stderr_text)
    state = _read_state(state_path)
    ev = _evaluate_attempt(state, sig, max_repeat,
                           args.cumulative_max, args.wallclock_max_s)
    _write_state(state_path, state)

    if not disk_ok:
        verdict = "guardrail_halt"
    elif ev["signature_stalled"]:
        verdict = "stalled"
    elif ev["cumulative_stalled"]:
        verdict = "stalled_cumulative"
    elif ev["wallclock_exceeded"]:
        verdict = "wallclock_halt"
    else:
        verdict = "ok"

    _print_json({
        "verdict": verdict,
        "detail": {
            "disk": {"free_gb": round(free_gb, 2), "ceiling_gb": args.ceiling_gb, "ok": disk_ok},
            "stall": {"signature": ev["signature"], "repeat_count": ev["repeat_count"],
                      "max_repeat": ev["max_repeat"], "stalled": ev["signature_stalled"]},
            "cumulative": {"total_attempts": ev["total_attempts"], "cumulative_max": ev["cumulative_max"],
                           "stalled": ev["cumulative_stalled"]},
            "wallclock": {"elapsed_s": ev["elapsed_s"], "wallclock_max_s": ev["wallclock_max_s"],
                          "exceeded": ev["wallclock_exceeded"]},
        },
    })


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="setup_guardrail",
        description=(
            "Deterministic safety-stop helper for oh-my-mlip self-healing install loop. "
            "All subcommands print JSON to stdout. Recovery reasoning is the agent's job "
            "via AGENTS.md -- this script owns ONLY disk/stall/cleanup guardrails."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # disk-check
    p_disk = sub.add_parser("disk-check", help="Check free disk headroom at PATH.")
    p_disk.add_argument("--ceiling-gb", type=float, default=30.0,
                        help="Minimum required free space in GB (default: 30).")
    p_disk.add_argument("--path", default=None,
                        help="Filesystem path to check (default: cwd).")

    # record-attempt
    p_rec = sub.add_parser("record-attempt", help="Record one attempt's stderr; detect stall.")
    p_rec.add_argument("--state", required=True,
                       help="Path to JSON state file (created if absent).")
    p_rec.add_argument("--stderr-file", required=True,
                       help="Path to stderr text file, or '-' to read from stdin.")
    p_rec.add_argument("--max-repeat", type=int, default=2,
                       help="Max times the same signature may recur before stall (default: 2).")
    p_rec.add_argument("--cumulative-max", type=int, default=5,
                       help="Max total attempts (signature-agnostic) before stall (default: 5).")
    p_rec.add_argument("--wallclock-max-s", type=float, default=None,
                       help="Max wall-clock seconds since first attempt before halt (default: off).")

    # clean-cache
    p_clean = sub.add_parser("clean-cache",
                              help="Remove torch_extensions and HF partial/lock files only.")
    p_clean.add_argument("--dry-run", action="store_true",
                         help="List what would be removed without deleting anything.")

    # gate
    p_gate = sub.add_parser("gate", help="Combined disk-check + record-attempt verdict.")
    p_gate.add_argument("--state", required=True,
                        help="Path to JSON state file.")
    p_gate.add_argument("--ceiling-gb", type=float, default=30.0,
                        help="Minimum required free space in GB (default: 30).")
    p_gate.add_argument("--stderr-file", required=True,
                        help="Path to stderr text file, or '-' to read from stdin.")
    p_gate.add_argument("--path", default=None,
                        help="Filesystem path to check for disk (default: cwd).")
    p_gate.add_argument("--max-repeat", type=int, default=2,
                        help="Max times the same signature may recur before stall (default: 2).")
    p_gate.add_argument("--cumulative-max", type=int, default=5,
                        help="Max total attempts (signature-agnostic) before stall (default: 5).")
    p_gate.add_argument("--wallclock-max-s", type=float, default=None,
                        help="Max wall-clock seconds since first attempt before halt (default: off).")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "disk-check": cmd_disk_check,
        "record-attempt": cmd_record_attempt,
        "clean-cache": cmd_clean_cache,
        "gate": cmd_gate,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
