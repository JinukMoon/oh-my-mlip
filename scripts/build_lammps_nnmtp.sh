#!/usr/bin/env bash
# build_lammps_nnmtp.sh — build a LAMMPS binary carrying the NN-MTP pair style.
#
# Follows onthefly-distill/lammps/BUILD.md verbatim: clone stable LAMMPS, drop in
# pair_nnmtp.* from the sibling repo, cmake Release, make. The pair style is pure
# C++ (no LibTorch, no Python), so no extra packages are needed.
#
# The clone/build root defaults OUTSIDE the oh-my-mlip repo on purpose: the pair
# style is GPL-2.0 (derived from LAMMPS) and must never land inside this MIT
# tree — the repo guard checks committed files, and this script keeps even the
# working tree clean of it.
#
# Usage:
#   scripts/build_lammps_nnmtp.sh [--repo <onthefly-distill dir>] [--prefix <build root>] [-j N]
# Result:
#   <prefix>/lammps/build/lmp   (echoed at the end; export LMP_BIN=<that path>)
# Oracle:
#   "$LMP_BIN" -h | grep -c nnmtp   -> non-zero
set -euo pipefail

REPO="${ONTHEFLY_REPO:-$HOME/01_2026/onthefly-distill}"
PREFIX="${OMM_BUILD_ROOT:-$HOME/.cache/oh-my-mlip}"
JOBS="$(nproc)"
while [ $# -gt 0 ]; do
  case "$1" in
    --repo)   REPO="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    -j)       JOBS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -f "$REPO/lammps/src/pair_nnmtp.cpp" ] || { echo "pair_nnmtp sources not found under $REPO/lammps/src" >&2; exit 1; }

# cmake: prefer PATH, else fall back to any conda-env cmake (host has no system cmake)
CMAKE="$(command -v cmake || true)"
if [ -z "$CMAKE" ]; then
  for c in "$HOME"/miniconda3/envs/*/bin/cmake; do [ -x "$c" ] && CMAKE="$c" && break; done
fi
[ -n "$CMAKE" ] || { echo "cmake not found (PATH or ~/miniconda3/envs/*/bin)" >&2; exit 1; }
mkdir -p "$PREFIX"
cd "$PREFIX"

if [ ! -d lammps/.git ]; then
  git clone -b stable --depth 1 https://github.com/lammps/lammps
fi
cp "$REPO"/lammps/src/pair_nnmtp.*    lammps/src/
cp "$REPO"/lammps/src/pair_nnmtp_v2.* lammps/src/ 2>/dev/null || true

mkdir -p lammps/build
cd lammps/build
"$CMAKE" ../cmake -DCMAKE_BUILD_TYPE=Release >cmake.log 2>&1
make -j "$JOBS" >make.log 2>&1

LMP="$PREFIX/lammps/build/lmp"
n="$("$LMP" -h 2>/dev/null | grep -c nnmtp || true)"
echo "built: $LMP"
echo "nnmtp styles visible in -h: $n"
[ "$n" -gt 0 ] || { echo "ORACLE FAIL: nnmtp not present in $LMP" >&2; exit 1; }
echo "export LMP_BIN=$LMP"
