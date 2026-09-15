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
#   scripts/build_lammps_nnmtp.sh --repo <onthefly-distill dir> [--prefix <build root>]
#                                 [--ref <git-ref>] [-j N]
# --repo (or ONTHEFLY_REPO env var) is required. Provides the path to the
# onthefly-distill checkout containing pair_nnmtp sources.
# --ref pins the LAMMPS clone to a specific tag/branch/commit; default is
# 'stable', which is upstream LAMMPS's own recommendation (BUILD.md), not an
# oh-my-mlip choice -- pass --ref only to deviate from that.
# Result:
#   <prefix>/lammps/build/lmp   (echoed at the end; export LMP_BIN=<that path>)
# Oracle:
#   "$LMP_BIN" -h | grep -c nnmtp   -> non-zero
set -euo pipefail

REPO="${ONTHEFLY_REPO:-}"
PREFIX="${OMM_BUILD_ROOT:-$HOME/.cache/oh-my-mlip}"
REF="stable"
JOBS="$(nproc)"
while [ $# -gt 0 ]; do
  case "$1" in
    --repo)   REPO="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --ref)    REF="$2"; shift 2 ;;
    -j)       JOBS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -n "$REPO" ] || {
  echo "usage: scripts/build_lammps_nnmtp.sh --repo <onthefly-distill dir> [--prefix <build root>] [--ref <git-ref>] [-j N]" >&2
  echo "error: --repo or ONTHEFLY_REPO env var is required" >&2
  exit 2
}
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
  git clone -b "$REF" --depth 1 https://github.com/lammps/lammps
fi
cp "$REPO"/lammps/src/pair_nnmtp.*    lammps/src/
cp "$REPO"/lammps/src/pair_nnmtp_v2.* lammps/src/ 2>/dev/null || true

mkdir -p lammps/build
cd lammps/build

# cmake/make failures are common (missing system libs, compiler mismatches)
# and their real cause lives in the log tail, not in this script's own exit
# message -- point at it before bailing instead of leaving the agent to guess
# which log to open.
_on_build_fail() {
  echo "build failed -- see the log tail below (full logs: $PWD/cmake.log, $PWD/make.log)" >&2
  tail -n 40 cmake.log make.log 2>/dev/null >&2 || true
}
trap _on_build_fail ERR

"$CMAKE" ../cmake -DCMAKE_BUILD_TYPE=Release >cmake.log 2>&1
make -j "$JOBS" >make.log 2>&1
trap - ERR

LMP="$PREFIX/lammps/build/lmp"
n="$("$LMP" -h 2>/dev/null | grep -c nnmtp || true)"
echo "built: $LMP"
echo "nnmtp styles visible in -h: $n"
[ "$n" -gt 0 ] || { echo "ORACLE FAIL: nnmtp not present in $LMP" >&2; exit 1; }
echo "export LMP_BIN=$LMP"
