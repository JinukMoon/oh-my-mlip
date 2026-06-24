#!/usr/bin/env bash
# compile_sevennet.sh — print (and optionally run) the SevenNet GPU accel enable.
#
# SevenNet has no separate AOT "compile" step: its GPU acceleration is enabled at
# install time (an extra) and selected at calculator construction. This wrapper
# PRINTS the curated upstream-doc commands by default (--dry-run): how to install
# the accel extra and how to verify it on a GPU host. The real install/verify
# needs an NVIDIA GPU and is DEFERRED -> GPU-UNVERIFIED here.
#
# Source: SevenNet upstream docs (provenance=upstream-doc in models.json ->
# SevenNet.accel). See docs/compile.md.
#
# Usage:
#   scripts/compile_sevennet.sh --dry-run [oeq|cueq12|cueq13]   # print only
#   scripts/compile_sevennet.sh [oeq|cueq12|cueq13]             # run install+verify
#
# Default backend: oeq (openequivariance). cueq12/cueq13 select cuEquivariance by
# CUDA major (12 or 13).
set -euo pipefail

DRY_RUN=0
BACKEND="oeq"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,21p' "${BASH_SOURCE[0]:-$0}"
      exit 0
      ;;
    oeq|cueq12|cueq13) BACKEND="$arg" ;;
    -*)
      echo "compile_sevennet.sh: unknown option '$arg'" >&2
      exit 2
      ;;
    *)
      echo "compile_sevennet.sh: unknown backend '$arg' (use oeq|cueq12|cueq13)" >&2
      exit 2
      ;;
  esac
done

INSTALL=(pip install "sevenn[${BACKEND}]")
VERIFY=(python -c 'from sevenn.nn.oeq_helper import is_oeq_available; print(is_oeq_available())')

echo "# SevenNet accel enable (upstream-doc, GPU-unverified)"
echo "# backend: ${BACKEND}  (oeq=openequivariance; cueq12/cueq13=cuEquivariance by CUDA major)"
echo "# install:"
printf '%q ' "${INSTALL[@]}"
echo
echo "# then enable at construction: SevenNetCalculator(..., enable_oeq=True)"
echo "#   (or --enable_oeq / --enable_cueq on the sevenn CLI)"
echo "# verify (GPU host):"
printf '%q ' "${VERIFY[@]}"
echo

if [ "$DRY_RUN" -eq 1 ]; then
  echo "# [--dry-run] printed only; nothing installed/run (real GPU run deferred)."
  exit 0
fi

"${INSTALL[@]}"
exec "${VERIFY[@]}"
