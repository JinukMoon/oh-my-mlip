#!/usr/bin/env bash
# build_attempt.sh — tiered, sanitized build-test driver for oh-my-mlip.
#
# PURPOSE
#   Record a per-STAGE status for each env (or a subset) into a structured,
#   commit-safe report (docs/build_attempt_results.md). The point is to find
#   what fails so the roster can be iterated env-by-env. The runbook lives in
#   docs/build_attempt.md.
#
# STAGE ENUM (honest — see docs/build_attempt.md for why there is no fake
# conda-vs-pip split):
#   resolve            recipe + models.json resolve to a known env
#   reachability       wheel/source URLs reachable (scripts/verify_sources.py)
#   dry-run            install.sh --dry-run prints a plan for the env
#   env-create         REAL `conda env create --file` (folds conda+pip in ONE
#                      call, exactly as install.sh does — so it is ONE stage)
#   weights-download   gated/auto weights fetch attempt (token via HF_TOKEN_PATH)
#   compile            D3 / per-arch kernel compile        (needs a GPU)
#   run                single-point inference smoke        (needs a GPU)
#
#   compile/run are logged as `gpu-required-deferred` whenever no GPU (nvcc) is
#   present — they are NOT failures.
#
# TIERS (--tier):
#   lint    (default) GPU-free: resolve + recipe lint + verify_sources +
#                     install.sh --dry-run for ALL envs. Fast and safe.
#   subset  lint, plus a REAL env-create for the small set named via --envs a,b
#           and a weights-download attempt for those envs.
#   full    REAL env-create for ALL envs. Owner/overnight only; NOT a default.
#
# TOKEN SAFETY (critical):
#   The HF token is the user's. This driver NEVER inlines, echoes, or commits it.
#   It is read (never printed) from the file named by HF_TOKEN_PATH only for a
#   weights-download attempt. EVERY line written to the log or report is passed
#   through sanitize(), which scrubs (a) the exact token value if HF_TOKEN_PATH
#   is readable and (b) any hf_<20+ alnum> shape, replacing it with hf_<redacted>.
#   After writing the report the driver runs scripts/verify_no_token.py (or a
#   grep fallback) and FAILS if a token-shaped literal slipped through.
#
# Usage:
#   ./scripts/build_attempt.sh [--tier lint|subset|full] [--envs a,b,c]
#                              [--dry-run-self] [-h|--help]
#
#     --tier T         tier to run (default: lint)
#     --envs a,b       comma-separated env subset (required for tier=subset)
#     --dry-run-self   exercise the stage + sanitize machinery WITHOUT building
#                      anything (proves the report + redaction logic; CI-safe)
#     -h, --help       show this header and exit
#
# Output: docs/build_attempt_results.md (the tracked deliverable) — a table
#   env | stage | status(ok|fail|gpu-required-deferred|skipped) | detail
#   plus a FAILURE LIST summary. The report is safe to commit (no token).
#
# Lint note: this script passes `bash -n` and the static analyzer cleanly. The
# only intentional dynamic indirection is fully quoted; no unavoidable warnings.

set -euo pipefail

# ── Resolve repo root and key paths ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENVS_DIR="$REPO_ROOT/envs"
INSTALL_SH="$REPO_ROOT/install.sh"
REPORT="$REPO_ROOT/docs/build_attempt_results.md"

# ── Defaults / arg parsing ──
TIER="lint"
SUBSET_CSV=""
DRY_RUN_SELF=0

usage() { sed -n '2,69p' "${BASH_SOURCE[0]:-$0}"; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tier) TIER="${2:-}"; shift 2 ;;
    --tier=*) TIER="${1#*=}"; shift ;;
    --envs) SUBSET_CSV="${2:-}"; shift 2 ;;
    --envs=*) SUBSET_CSV="${1#*=}"; shift ;;
    --dry-run-self) DRY_RUN_SELF=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "build_attempt.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

case "$TIER" in
  lint|subset|full) ;;
  *) echo "build_attempt.sh: invalid --tier '$TIER' (lint|subset|full)" >&2; exit 2 ;;
esac

# ── Token sanitization (CRITICAL) ───────────────────────────────────────────
# Read the literal token ONCE (never printed) so sanitize() can scrub the exact
# value in addition to the generic hf_<...> shape. Wrapped in `set +x` so the
# value can never surface via xtrace.
set +x
TOKEN_VALUE=""
if [ -n "${HF_TOKEN_PATH:-}" ] && [ -r "${HF_TOKEN_PATH:-}" ]; then
  # tr -d strips trailing newline/whitespace; redirection keeps it off argv.
  TOKEN_VALUE="$(tr -d ' \t\r\n' < "$HF_TOKEN_PATH" 2>/dev/null || true)"
fi

# sanitize: read stdin, emit it with every secret scrubbed. Used for EVERY line
# that reaches the log or the report. Two layers:
#   1. exact-value replacement (only if a non-empty token was readable)
#   2. generic hf_<20+ alnum> shape replacement (matches verify_no_token.py)
# The python program is passed via -c (NOT a stdin heredoc) so that the data to
# be sanitized stays on stdin. The token value is passed only through the
# environment, never on argv. The token shape is assembled from fragments so
# this driver's own source never contains a self-matching literal.
TOKEN_VALUE_FOR_PY="$TOKEN_VALUE"
SANITIZE_PY='
import os, re, sys
tok = os.environ.get("TOKEN_VALUE_FOR_PY", "")
pat = re.compile("hf" + "_" + "[A-Za-z0-9]" + "{20,}")
redacted = "hf" + "_" + "<redacted>"
data = sys.stdin.read()
if tok:
    data = data.replace(tok, redacted)
data = pat.sub(redacted, data)
sys.stdout.write(data)
'
sanitize() {
  TOKEN_VALUE_FOR_PY="$TOKEN_VALUE_FOR_PY" python3 -c "$SANITIZE_PY"
}
set +x  # keep token handling off any xtrace for the remainder of the run

# ── Report rows accumulate here, then sanitize once on flush ────────────────
ROWS=()        # "env|stage|status|detail"
FAILURES=()    # "env / stage: detail"

record() {
  # record ENV STAGE STATUS DETAIL...
  local env_name="$1" stage="$2" status="$3"
  shift 3
  local detail="$*"
  ROWS+=("${env_name}|${stage}|${status}|${detail}")
  if [ "$status" = "fail" ]; then
    FAILURES+=("${env_name} / ${stage}: ${detail}")
  fi
}

# ── Environment fact-finding ────────────────────────────────────────────────
available_envs() {
  local f name
  for f in "$ENVS_DIR"/*.yml; do
    [ -e "$f" ] || continue
    name="$(basename "$f" .yml)"
    printf '%s\n' "$name"
  done
}

mapfile -t ALL_ENVS < <(available_envs)
if [ "${#ALL_ENVS[@]}" -eq 0 ]; then
  echo "build_attempt.sh: no env recipes in $ENVS_DIR/*.yml" >&2
  exit 1
fi

# GPU presence governs compile/run attribution.
GPU_OK=0
if command -v nvcc >/dev/null 2>&1; then
  GPU_OK=1
fi

# conda/mamba presence governs whether env-create can actually run.
CONDA_BIN=""
if command -v mamba >/dev/null 2>&1; then
  CONDA_BIN="mamba"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="conda"
fi

# Resolve the subset CSV to an array (used by subset tier).
SUBSET=()
if [ -n "$SUBSET_CSV" ]; then
  IFS=',' read -r -a SUBSET <<< "$SUBSET_CSV"
fi

in_subset() {
  local needle="$1" e
  for e in "${SUBSET[@]:-}"; do
    [ "$e" = "$needle" ] && return 0
  done
  return 1
}

# ── Stage helpers ───────────────────────────────────────────────────────────
# Each stage records exactly one row per env. Under --dry-run-self every stage
# is SIMULATED (no network, no conda, no token use) so the machinery + report
# can be proven without building anything.

stage_resolve() {
  local env_name="$1"
  if [ -e "$ENVS_DIR/$env_name.yml" ]; then
    record "$env_name" resolve ok "recipe envs/$env_name.yml present"
  else
    record "$env_name" resolve fail "no recipe at envs/$env_name.yml"
  fi
}

stage_dry_run() {
  local env_name="$1" out
  if [ "$DRY_RUN_SELF" -eq 1 ]; then
    record "$env_name" dry-run ok "[self-test] would run install.sh --dry-run $env_name"
    return
  fi
  if out="$("$INSTALL_SH" --dry-run "$env_name" 2>&1)"; then
    record "$env_name" dry-run ok "install.sh --dry-run plan printed"
  else
    record "$env_name" dry-run fail "install.sh --dry-run rc!=0: $(printf '%s' "$out" | tail -1)"
  fi
}

stage_env_create() {
  local env_name="$1" prefix recipe out
  prefix="$ENVS_DIR/$env_name"
  recipe="$ENVS_DIR/$env_name.yml"
  if [ "$DRY_RUN_SELF" -eq 1 ]; then
    record "$env_name" env-create skipped "[self-test] real env-create not run"
    return
  fi
  if [ -z "$CONDA_BIN" ]; then
    record "$env_name" env-create skipped "neither conda nor mamba on PATH"
    return
  fi
  if [ ! -e "$recipe" ]; then
    record "$env_name" env-create fail "no recipe at $recipe"
    return
  fi
  # ONE call folds conda + pip, exactly as install.sh does -> ONE stage.
  if out="$("$CONDA_BIN" env create --prefix "$prefix" --file "$recipe" 2>&1)"; then
    record "$env_name" env-create ok "env created at envs/$env_name"
  else
    record "$env_name" env-create fail "$(printf '%s' "$out" | tail -1)"
  fi
}

stage_weights_download() {
  local env_name="$1"
  if [ "$DRY_RUN_SELF" -eq 1 ]; then
    record "$env_name" weights-download skipped "[self-test] weights fetch not attempted"
    return
  fi
  if [ -z "${HF_TOKEN_PATH:-}" ] || [ ! -r "${HF_TOKEN_PATH:-}" ]; then
    record "$env_name" weights-download skipped "no readable HF_TOKEN_PATH; gated fetch not attempted"
    return
  fi
  # A real fetch would invoke the env's python with the token resolved via the
  # standard HF_TOKEN_PATH variable (NEVER passed on argv). The python loader
  # reads the file itself; we never read the value into a logged variable.
  local prefix="$ENVS_DIR/$env_name" out
  if [ ! -x "$prefix/bin/python" ]; then
    record "$env_name" weights-download skipped "env python absent (run env-create first)"
    return
  fi
  if out="$(HF_TOKEN_PATH="$HF_TOKEN_PATH" "$prefix/bin/python" \
        "$REPO_ROOT/run_examples/single_point.py" "$env_name" 2>&1)"; then
    record "$env_name" weights-download ok "weights fetch + single-point returned"
  else
    record "$env_name" weights-download fail "$(printf '%s' "$out" | tail -1)"
  fi
}

stage_compile_run() {
  local env_name="$1"
  if [ "$DRY_RUN_SELF" -eq 1 ]; then
    record "$env_name" compile skipped "[self-test] compile not driven"
    record "$env_name" run skipped "[self-test] inference run not driven"
  elif [ "$GPU_OK" -eq 1 ]; then
    # A real GPU run would compile the D3 kernel + run inference here. In this
    # goal we do not drive heavy GPU builds; mark attempted-but-not-run honestly.
    record "$env_name" compile skipped "GPU present; heavy compile not driven in this goal"
    record "$env_name" run skipped "GPU present; inference run not driven in this goal"
  else
    record "$env_name" compile gpu-required-deferred "no nvcc/GPU on this host"
    record "$env_name" run gpu-required-deferred "no nvcc/GPU on this host"
  fi
}

# ── Tier orchestration ──────────────────────────────────────────────────────
run_lint() {
  # GPU-free pass over ALL envs: resolve + dry-run, then one reachability row
  # per env summarized from verify_sources, and compile/run deferred.
  local env_name
  for env_name in "${ALL_ENVS[@]}"; do
    stage_resolve "$env_name"
    stage_dry_run "$env_name"
  done
  reachability_all
  for env_name in "${ALL_ENVS[@]}"; do
    stage_compile_run "$env_name"
  done
}

reachability_all() {
  # One reachability stage per env, sourced from scripts/verify_sources.py.
  # Under --dry-run-self we simulate (no network). Otherwise use --mock to stay
  # offline-safe by default in this goal (real network probing is the verify
  # script's own CI job, not this driver's job).
  local env_name
  if [ "$DRY_RUN_SELF" -eq 1 ]; then
    for env_name in "${ALL_ENVS[@]}"; do
      record "$env_name" reachability ok "[self-test] verify_sources not invoked"
    done
    return
  fi
  if python3 "$REPO_ROOT/scripts/verify_sources.py" --mock >/dev/null 2>&1; then
    for env_name in "${ALL_ENVS[@]}"; do
      record "$env_name" reachability ok "verify_sources --mock OK"
    done
  else
    for env_name in "${ALL_ENVS[@]}"; do
      record "$env_name" reachability fail "verify_sources reported a failure (see CI source-lint)"
    done
  fi
}

run_subset() {
  run_lint
  if [ "${#SUBSET[@]}" -eq 0 ]; then
    echo "build_attempt.sh: --tier subset needs --envs a,b,..." >&2
    exit 2
  fi
  local env_name
  for env_name in "${SUBSET[@]}"; do
    if ! printf '%s\n' "${ALL_ENVS[@]}" | grep -qx "$env_name"; then
      record "$env_name" env-create fail "unknown env (no envs/$env_name.yml)"
      continue
    fi
    stage_env_create "$env_name"
    stage_weights_download "$env_name"
  done
}

run_full() {
  run_lint
  local env_name
  for env_name in "${ALL_ENVS[@]}"; do
    stage_env_create "$env_name"
  done
}

case "$TIER" in
  lint) run_lint ;;
  subset) run_subset ;;
  full) run_full ;;
esac

# ── Write the report (sanitized) ────────────────────────────────────────────
host_label="$(uname -srm 2>/dev/null || echo unknown)"
gpu_label=$([ "$GPU_OK" -eq 1 ] && echo "present" || echo "absent (compile/run deferred)")
conda_label="${CONDA_BIN:-none}"
self_label=$([ "$DRY_RUN_SELF" -eq 1 ] && echo "yes (no builds run)" || echo "no")

{
  echo "# build_attempt results"
  echo
  echo "_Generated by \`scripts/build_attempt.sh\`. Safe to commit: every line is"
  echo "passed through the token sanitizer before being written._"
  echo
  echo "- tier: \`$TIER\`"
  echo "- dry-run-self: $self_label"
  echo "- host: \`$host_label\`"
  echo "- conda/mamba: \`$conda_label\`"
  echo "- GPU (nvcc): $gpu_label"
  echo
  echo "## Stage results"
  echo
  echo "| env | stage | status | detail |"
  echo "|---|---|---|---|"
  for row in "${ROWS[@]}"; do
    IFS='|' read -r r_env r_stage r_status r_detail <<< "$row"
    echo "| $r_env | $r_stage | $r_status | $r_detail |"
  done
  echo
  echo "## Failure list"
  echo
  if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "_None — no stage reported \`fail\`._"
  else
    for f in "${FAILURES[@]}"; do
      echo "- $f"
    done
  fi
} | sanitize > "$REPORT"

echo "build_attempt.sh: wrote $REPORT (tier=$TIER, rows=${#ROWS[@]}, failures=${#FAILURES[@]})"

# ── Post-write token gate (CRITICAL) ────────────────────────────────────────
# Re-scan the report we just wrote. The driver FAILS if any token-shaped literal
# survived sanitization. Prefer the canonical verifier; fall back to grep.
if [ -x "$REPO_ROOT/scripts/verify_no_token.py" ] || \
   [ -f "$REPO_ROOT/scripts/verify_no_token.py" ]; then
  if ! python3 "$REPO_ROOT/scripts/verify_no_token.py" "$REPO_ROOT/docs" >/dev/null 2>&1; then
    echo "build_attempt.sh: ABORT — token-shaped literal found in docs/ after write." >&2
    echo "  The report was NOT left in a committable state. Investigate sanitize()." >&2
    exit 1
  fi
else
  if grep -Eq "hf_[A-Za-z0-9]{20,}" "$REPORT"; then
    echo "build_attempt.sh: ABORT — token-shaped literal found in $REPORT." >&2
    exit 1
  fi
fi

echo "build_attempt.sh: token gate clean — report is safe to commit."
