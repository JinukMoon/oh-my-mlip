#!/bin/bash
# oh-my-mlip shared environment (read-only convention file).
# Source this once before running any model: `source "${OH_MY_MLIP_HOME}/env.sh"`.
#
# OH_MY_MLIP_HOME is the clone root (the directory containing this file). It is
# autodetected below if unset, so a fresh clone needs no manual export.

# ── 0) Resolve OH_MY_MLIP_HOME (clone root) ──
if [ -z "${OH_MY_MLIP_HOME:-}" ]; then
  # Directory containing this script, resolved even when sourced.
  _OMM_SRC="${BASH_SOURCE[0]:-$0}"
  OH_MY_MLIP_HOME="$(cd "$(dirname "$_OMM_SRC")" && pwd)"
  export OH_MY_MLIP_HOME
  unset _OMM_SRC
fi

# ── 1) Shared model cache (name-based auto-download models: UMA/fairchem/MACE-mp etc.) ──
export HF_HOME="${OH_MY_MLIP_HOME}/models/hf"            # one shared HuggingFace download cache
# NOTE: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are intentionally NOT forced here.
#   The public edition fetches weights on first run (gated models need the user's HF_TOKEN).
#   Set them to 1 yourself once everything is cached if you want to pin offline:
#   export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export FAIRCHEM_CACHE_DIR="${OH_MY_MLIP_HOME}/models/fairchem"  # UMA + eSEN. NOTE: fairchem reads FAIRCHEM_CACHE_DIR (NOT FAIRCHEM_CACHE)
export TORCH_HOME="${OH_MY_MLIP_HOME}/models/torch"     # torch.hub cache
export CACHED_PATH_CACHE_ROOT="${OH_MY_MLIP_HOME}/models/cached_path"  # ORB (orb_models uses cached_path)

# ── 2) catbench D3 ──
#   The arch-specific pair_d3.so (catbench/dispersion/cuda/pair_d3.so) is NOT baked into
#   distributed tarballs. It compiles on the user GPU on FIRST RUN -> needs `nvcc` on PATH
#   (see section 3). After the first compile it is cached and reused.
export PYTHONUTF8=1                                      # prevents ascii decode crash during D3 (re)compile

# ── 3) nvcc / CUDA_HOME (needed for first-run D3 compile) ──
#   Autodetect CUDA_HOME instead of hardcoding a host path. Order: existing CUDA_HOME ->
#   `nvcc` on PATH -> common /usr/local/cuda symlink. If none is found, D3 first-run compile
#   will fail with a clear message; install/point CUDA_HOME at a CUDA toolkit with nvcc.
if [ -z "${CUDA_HOME:-}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  elif [ -d /usr/local/cuda ]; then
    CUDA_HOME=/usr/local/cuda
  fi
  [ -n "${CUDA_HOME:-}" ] && export CUDA_HOME
fi
if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  # Append (not prepend) so each env's torch-bundled CUDA libs keep priority; this is only a
  # fallback for envs whose torch does not load libcudart.so.12 into memory (e.g. cu118 builds).
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${CUDA_HOME}/lib64:${CUDA_HOME}/lib"
fi
# Per-env LD_LIBRARY_PATH note: a few envs declare an `env_run` prefix in models.json
# (e.g. DPA4 needs LD_LIBRARY_PATH="" to dodge an Intel oneAPI libfabric clash). That prefix
# is applied per-invocation by the provider, NOT exported globally here.

# ── 4) name-based model caches (MatRIS/TACE/eqnorm) ──
#   MatRIS.load -> ~/.cache/matris, TACE -> ~/.cache/tace are hardcoded upstream. Symlink the
#   shared weights into the user cache only when the user does not already have their own.
for _m in matris tace eqnorm; do
  # Skip if anything already occupies the path — a real dir/file ([ -e ]) OR an
  # existing symlink even when its target is missing ([ -L ], i.e. a broken link
  # from a prior run whose models/<m> isn't built yet). Using only [ -e ] would
  # miss a broken symlink and then `ln -s` (no -f) would fail "File exists" and,
  # under the caller's `set -e`, abort install.sh. `|| true` is a final guard so
  # sourcing env.sh can never abort the install on a benign cache-symlink hiccup.
  if [ ! -e "$HOME/.cache/$_m" ] && [ ! -L "$HOME/.cache/$_m" ]; then
    mkdir -p "$HOME/.cache" 2>/dev/null || true
    ln -s "${OH_MY_MLIP_HOME}/models/$_m" "$HOME/.cache/$_m" 2>/dev/null || true
  fi
done
unset _m || true

# ── 5) JIT fast-kernel seed (OpenEquivariance libtorch_tp_jit etc.) ──
#   Some envs JIT-compile fast kernels at runtime; a clean HOME recompiles from scratch
#   (first run can "hang" for minutes). Seed a prebuilt copy into the user cache only-if-absent
#   (cp -n) so there is zero recompile and the same kernel loads immediately.
#   NOTE: NequIP/Allegro do NOT use this JIT path; they load AOT .pt2 (recompiled/reselected on
#   the user GPU) and are unaffected by this seed.
_TE_SRC="${OH_MY_MLIP_HOME}/models/torch_ext"
if [ -d "$_TE_SRC" ]; then
  _TE_DST="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"
  mkdir -p "$_TE_DST" 2>/dev/null && cp -rn "$_TE_SRC"/. "$_TE_DST"/ 2>/dev/null
  unset _TE_DST
fi
unset _TE_SRC
