#!/usr/bin/env bash
# compile_nequip.sh — print (and optionally run) the NequIP/Allegro GPU compile.
#
# NequIP and Allegro both load through nequip's compiled-model ASE path
# (NequIPCalculator.from_compiled_model), so the per-arch .pt2 must be produced
# with `nequip-compile`. This wrapper assembles the curated upstream-doc command
# and, by default, only PRINTS it (--dry-run). The real compile needs an NVIDIA
# GPU + openequivariance and is DEFERRED to the user's host: it is GPU-UNVERIFIED
# here.
#
# Source of the command: NequIP/Allegro upstream docs (provenance=upstream-doc in
# models.json -> NequIP.accel). See docs/compile.md.
#
# Usage:
#   scripts/compile_nequip.sh --dry-run <ckpt> [<out>.nequip.pt2]   # print only
#   scripts/compile_nequip.sh <ckpt> [<out>.nequip.pt2]            # run (needs GPU)
#
# Defaults: <out> = <ckpt-stem>.nequip.pt2 in the cwd.
set -euo pipefail

DRY_RUN=0
ARGS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,21p' "${BASH_SOURCE[0]:-$0}"
      exit 0
      ;;
    -*)
      echo "compile_nequip.sh: unknown option '$arg'" >&2
      exit 2
      ;;
    *) ARGS+=("$arg") ;;
  esac
done

if [ "${#ARGS[@]}" -lt 1 ]; then
  echo "compile_nequip.sh: need a checkpoint path." >&2
  echo "  usage: scripts/compile_nequip.sh [--dry-run] <ckpt> [<out>.nequip.pt2]" >&2
  exit 2
fi

CKPT="${ARGS[0]}"
if [ "${#ARGS[@]}" -ge 2 ]; then
  OUT="${ARGS[1]}"
else
  stem="$(basename "$CKPT")"
  stem="${stem%.*}"
  OUT="${stem}.nequip.pt2"
fi

# Curated upstream-doc command (mirrors models.json NequIP.accel.compile_cmd).
# torchscript .nequip.pth is the torch<2.10 fallback; this wrapper emits the
# aotinductor .pt2 path.
# OMM_COMPILE_MODIFIER selects the accel modifier: NequIP registers
# enable_OpenEquivariance, Allegro registers enable_CuEquivariance (each env
# only knows its own — host-proven 2026-07-19). Default stays the NequIP one.
MODIFIER="${OMM_COMPILE_MODIFIER:-enable_OpenEquivariance}"
CMD=(nequip-compile "$CKPT" "$OUT"
     --mode aotinductor --device cuda --target ase
     --modifiers "$MODIFIER")

echo "# NequIP/Allegro compile (upstream-doc, GPU-unverified)"
echo "# requires: NVIDIA GPU + 'pip install openequivariance' (torch>=2.7, GCC9+)"
printf '%q ' "${CMD[@]}"
echo

if [ "$DRY_RUN" -eq 1 ]; then
  echo "# [--dry-run] printed only; nothing compiled (real GPU run deferred)."
  exit 0
fi

# ── oeq JIT-hang preflight (root-caused 2026-07-16, oeq v0.4.1 dissection) ──
# Importing openequivariance JIT-builds two torch cpp_extension modules
# (extlib/__init__.py: libtorch_tp_jit + generic_module) with ninja at
# MAX_JOBS=<nproc> by default, and torch's FileBaton spin-waits FOREVER on a
# stale build 'lock' left behind if a first build was killed (e.g. a
# cgroup-pids-limited login node) -> every later nequip-compile / model load
# hangs silently with no traceback. Two guards:
#   1. cap the JIT build parallelism (MAX_JOBS, overridable),
#   2. reap stale oeq build locks when no build is actually running.
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
TE_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"
if [ -d "$TE_DIR" ] && ! pgrep -x ninja >/dev/null 2>&1; then
  # shellcheck disable=SC2044
  for lock in $(find "$TE_DIR" -maxdepth 3 -name lock \
      \( -path "*libtorch_tp_jit*" -o -path "*generic_module*" \) 2>/dev/null); do
    echo "# removing stale oeq build lock (no ninja running): $lock"
    rm -f "$lock"
  done
fi

exec "${CMD[@]}"
