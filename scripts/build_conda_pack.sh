#!/usr/bin/env bash
# build_conda_pack.sh — AUTHOR-SIDE conda-pack builder for one oh-my-mlip env.
#
# Produces a relocatable tarball that unpacks to $OH_MY_MLIP_HOME/envs/<env> on
# any host. CRITICAL invariant: NEVER bake arch-specific artifacts into the
# tarball. Architecture-pinned files (D3 pair_d3.so, NequIP/Allegro .pt2) are
# recompiled/reselected on the END USER's GPU on first run — baking the author's
# build would silently mismatch a foreign GPU's compute capability (sm86 vs sm89)
# and is the single worst false-pass landmine in this whole pipeline. So we STRIP
# them here, before packing.
#
# It also fixes editable installs: __editable__*.pth / *.egg-link entries that
# point at the author's /home/* (or any absolute build path) break on relocation,
# so we strip/neutralize them before packing.
#
# Usage:
#   ./build_conda_pack.sh <env-prefix> <env-name> [out-dir]
#     env-prefix  path to the BUILT conda env (e.g. .../envs/mace)
#     env-name    logical env name used in the output filename (e.g. mace)
#     out-dir     where to write the tarball + sidecars (default: ./dist)
#
# Emits, in out-dir:
#   <env-name>.tar.gz            the relocatable env
#   <env-name>.tar.gz.sha256     sha256 checksum
#   <env-name>.unpack_size       on-disk size after unpack (bytes)
#   <env-name>.min_driver        minimum NVIDIA driver floor (stub; fill from torch build)
#
# Requires: conda-pack on PATH.

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <env-prefix> <env-name> [out-dir]" >&2
  exit 2
fi

ENV_PREFIX="$1"
ENV_NAME="$2"
OUT_DIR="${3:-./dist}"

if [ ! -d "$ENV_PREFIX" ]; then
  echo "build_conda_pack.sh: env prefix '$ENV_PREFIX' is not a directory" >&2
  exit 1
fi
if ! command -v conda-pack >/dev/null 2>&1; then
  echo "build_conda_pack.sh: conda-pack not found on PATH." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "build_conda_pack: env=$ENV_NAME prefix=$ENV_PREFIX"

# ── 1) STRIP baked arch artifacts (enforces 'never bake arch artifacts') ──
# These MUST recompile/reselect on the user's GPU on first run; shipping the
# author's build would mismatch a foreign GPU's compute capability.
echo "  stripping arch-specific artifacts (.pt2 compiled models, pair_d3.so) ..."
stripped=0
while IFS= read -r f; do
  rm -f "$f" && stripped=$((stripped + 1))
done < <(find "$ENV_PREFIX" \
           \( -path '*/models/compiled/*.pt2' \
           -o -name 'pair_d3.so' \
           -o -name 'pair_d3_for_ase.cuda.o' \) -type f 2>/dev/null)
# Also drop the on-disk ninja build cache for the D3 kernel so it rebuilds clean.
while IFS= read -r d; do
  rm -rf "$d" && stripped=$((stripped + 1))
done < <(find "$ENV_PREFIX" -type d -name 'pair_d3' 2>/dev/null)
echo "  stripped $stripped arch artifact(s)."

# ── 2) FIX editable installs (relocation-breaking absolute paths) ──
# __editable__*.pth and *.egg-link that point at /home/* or any author build dir
# will not exist on the user's machine. Neutralize them: the recipe rebuild
# (install.sh) re-installs those packages non-editable; for a packed env we strip
# the dangling editable pointer so import does not fault on a missing source tree.
echo "  fixing editable installs (__editable__*.pth / *.egg-link) ..."
fixed=0
while IFS= read -r f; do
  # If it references an absolute author path that won't exist post-relocation,
  # remove the editable pointer file entirely.
  if grep -qE '(/home/|/TGM/|/root/|/Users/)' "$f" 2>/dev/null; then
    rm -f "$f" && fixed=$((fixed + 1))
  fi
done < <(find "$ENV_PREFIX" \
           \( -name '__editable__*.pth' -o -name '*.egg-link' \) -type f 2>/dev/null)
echo "  neutralized $fixed editable pointer(s)."

# ── 3) Pack ──
TARBALL="$OUT_DIR/$ENV_NAME.tar.gz"
echo "  conda-pack -> $TARBALL"
conda-pack --prefix "$ENV_PREFIX" --output "$TARBALL" --force

# ── 4) Sidecars: sha256, unpack size, min-driver stub ──
echo "  computing sha256 ..."
sha256sum "$TARBALL" | awk '{print $1}' > "$OUT_DIR/$ENV_NAME.tar.gz.sha256"

echo "  measuring unpack size ..."
# du in bytes (apparent on-disk size of the built env tree).
du -sb "$ENV_PREFIX" | awk '{print $1}' > "$OUT_DIR/$ENV_NAME.unpack_size"

# min-driver floor: derive from the env's torch CUDA build. We emit a stub the
# publisher fills; deriving it programmatically requires importing torch in the
# packed env, which we leave to publish_hf.py (it has the env handy).
echo "TODO-on-upload" > "$OUT_DIR/$ENV_NAME.min_driver"

echo "build_conda_pack: done."
echo "  tarball      : $TARBALL"
echo "  sha256       : $(cat "$OUT_DIR/$ENV_NAME.tar.gz.sha256")"
echo "  unpack_size  : $(cat "$OUT_DIR/$ENV_NAME.unpack_size") bytes"
echo "  min_driver   : $(cat "$OUT_DIR/$ENV_NAME.min_driver") (fill from torch CUDA build at publish)"
