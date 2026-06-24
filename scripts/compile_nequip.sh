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
CMD=(nequip-compile "$CKPT" "$OUT"
     --mode aotinductor --device cuda --target ase
     --modifiers enable_OpenEquivariance)

echo "# NequIP/Allegro compile (upstream-doc, GPU-unverified)"
echo "# requires: NVIDIA GPU + 'pip install openequivariance' (torch>=2.7, GCC9+)"
printf '%q ' "${CMD[@]}"
echo

if [ "$DRY_RUN" -eq 1 ]; then
  echo "# [--dry-run] printed only; nothing compiled (real GPU run deferred)."
  exit 0
fi

exec "${CMD[@]}"
