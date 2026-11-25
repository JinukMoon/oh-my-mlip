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
case "$OH_MY_MLIP_HOME/" in
  */.claude/plugins/*|*/.codex/plugins/*)
    echo "[oh-my-mlip] warning: $OH_MY_MLIP_HOME is the plugin's own copy, not a hub; export" \
         "OH_MY_MLIP_HOME=<your clone> (default ~/.oh-my-mlip, AGENTS.md §9.0) and source that clone's env.sh." >&2 ;;
esac

# ── 1) Weight caches: FRAMEWORK-NATIVE by default; shared root is OPT-IN ──
# Default: env.sh does NOT redirect any download cache. Every framework uses
# its own upstream location (~/.cache/huggingface, ~/.cache/fairchem,
# ~/.cache/torch, ~/.cache/mace, ...), which ADOPTS whatever the user already
# downloaded and leaves `hf auth login` token discovery untouched.
# Why this is NOT redirected: the old always-on redirect (a) forked the
# user's existing caches — a 13 GB UMA re-download of byte-identical weights —
# and (b) moved huggingface_hub's login-token lookup to $HF_HOME/token, so
# every gated fetch went out anonymous and 401'd despite valid credentials.
# The models.json registry records resolved weight paths per model, so no
# central cache directory is needed for lookup.
#
# Building a SHARED hub (multi-user setup, HPC home-quota constraints):
# set OMM_SHARED_CACHE_ROOT to an absolute path BEFORE sourcing env.sh
# and every framework cache is explicitly redirected under it:
if [ -n "${OMM_SHARED_CACHE_ROOT:-}" ]; then
  export HF_HOME="${OMM_SHARED_CACHE_ROOT}/hf"
  # HF_HOME moves huggingface_hub's LOGIN-TOKEN lookup to $HF_HOME/token; keep
  # the standard login visible via HF_TOKEN_PATH — a PATH export, never a
  # token value; an already-set HF_TOKEN_PATH is respected.
  if [ -z "${HF_TOKEN_PATH:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
  fi
  export FAIRCHEM_CACHE_DIR="${OMM_SHARED_CACHE_ROOT}/fairchem"  # UMA + eSEN. NOTE: fairchem reads FAIRCHEM_CACHE_DIR (NOT FAIRCHEM_CACHE)
  export TORCH_HOME="${OMM_SHARED_CACHE_ROOT}/torch"             # torch.hub cache
  export CACHED_PATH_CACHE_ROOT="${OMM_SHARED_CACHE_ROOT}/cached_path"  # ORB (orb_models uses cached_path)
  export NEQUIP_CACHE_DIR="${NEQUIP_CACHE_DIR:-${OMM_SHARED_CACHE_ROOT}/nequip}"  # NequIP/Allegro model cache (~/.nequip)
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${OMM_SHARED_CACHE_ROOT}/xdg}"      # MACE and other XDG-aware caches
  # Frameworks that write ~/.cache/<name> (or ~/.local/<name>) whatever the
  # variables say (SevenNet, MatRIS, TACE, Nequix, Eqnorm, GRACE, DeePMD, MatterSim)
  # are pointed here by install.sh with symlinks (scripts/home_cache_links.sh).
fi
# NOTE: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are NOT forced where there is internet:
#   weights are fetched on first run (gated models need the user's HF_TOKEN).
# Where there is none (HPC compute nodes often drop outbound packets silently, and a
# loader that checks a URL before using its cache then waits until the job's time
# limit), offline mode is switched on: OMM_OFFLINE=1 makes every non-local
# connection in the envs' interpreters fail at once (oh_my_mlip/_offline_hook.py,
# installed into each env by install.sh), so loaders use the weights install.sh
# downloaded. Test (omm_internet_ok): through the proxy when http(s)_proxy is set
# (curl -sI), else a TCP connection, to huggingface.co or pypi.org within 3 s.
# OMM_OFFLINE=0 (or 1) set beforehand is kept and skips the test.
omm_internet_ok() {
  local t="${1:-3}" host
  if [ -n "${https_proxy:-}${HTTPS_PROXY:-}${http_proxy:-}${HTTP_PROXY:-}${all_proxy:-}${ALL_PROXY:-}" ]; then
    # a site that reaches the internet only through a proxy: test through it
    command -v curl >/dev/null 2>&1 || return 0      # cannot test: assume online
    for host in huggingface.co pypi.org; do
      curl -sI -o /dev/null --max-time "$t" "https://$host" 2>/dev/null && return 0
    done
    return 1
  fi
  for host in huggingface.co pypi.org; do
    timeout "$t" bash -c "exec 3<>/dev/tcp/$host/443" 2>/dev/null && return 0
  done
  return 1
}
if [ -z "${OMM_OFFLINE:-}" ]; then
  if ! omm_internet_ok 3; then
    export OMM_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    echo "oh-my-mlip: no internet on this node: offline mode on (OMM_OFFLINE=1); models use downloaded weights." >&2
  fi
fi

# ── 2) catbench D3 ──
#   The arch-specific pair_d3.so (catbench/dispersion/cuda/pair_d3.so) is never shipped;
#   it compiles on the user GPU on FIRST RUN -> needs `nvcc` on PATH
#   (see section 3). After the first compile it is cached and reused.
export PYTHONUTF8=1                                      # prevents ascii decode crash during D3 (re)compile

# ── 3) nvcc / CUDA_HOME (needed for first-run D3 compile) ──
#   Autodetect CUDA_HOME instead of hardcoding a host path. Order: existing CUDA_HOME ->
#   `nvcc` on PATH -> the active conda env's own nvcc ($CONDA_PREFIX/bin/nvcc; every
#   recipe ships cuda-nvcc, and a host may have no system toolkit) -> common
#   /usr/local/cuda symlink. If none is found, D3 first-run compile will fail with a
#   clear message; install/point CUDA_HOME at a CUDA toolkit with nvcc. install.sh and
#   adopt_env.py build the D3 kernel up front with the env's own nvcc (scripts/build_d3.py).
if [ -z "${CUDA_HOME:-}" ]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
    CUDA_HOME="$CONDA_PREFIX"
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
  # Expose the toolkit headers (nvrtc.h, cuda_runtime.h, ...) so first-run CUDA
  # extension JIT builds can find them. NequIP/Allegro load an AOT .pt2 whose
  # OpenEquivariance extension JIT-compiles on a cold cache via torch's
  # cpp_extension (needs ninja on PATH — provided per-env by the provider — AND
  # nvrtc.h on the include path). Append so any env-bundled headers keep priority.
  export CPATH="${CPATH:+$CPATH:}${CUDA_HOME}/include"
fi
# ── 3b) GPU archs for the D3 kernel build ──
#   catbench compiles pair_d3.so for TORCH_CUDA_ARCH_LIST and, when it is unset,
#   for 6.1-9.0 only: on a newer GPU (sm120) that build has no kernel for it and
#   D3 fails at the first call. Add this GPU's compute capability to that list
#   when it is missing (older GPUs: nothing changes). A value you set wins.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  _omm_smi="$(command -v nvidia-smi || { [ -x /usr/lib/wsl/lib/nvidia-smi ] && echo /usr/lib/wsl/lib/nvidia-smi; } || true)"
  _omm_cc="$([ -n "$_omm_smi" ] && "$_omm_smi" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)"
  case "$_omm_cc" in
    ""|6.1|7.0|7.5|8.0|8.6|8.9|9.0) ;;
    [0-9]*.[0-9]*) export TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6;8.9;9.0;$_omm_cc" ;;
  esac
  unset _omm_smi _omm_cc
fi

# Per-env LD_LIBRARY_PATH note: a few envs declare an `env_run` prefix in models.json
# (e.g. DPA4 needs LD_LIBRARY_PATH="" to dodge an Intel oneAPI libfabric clash). That prefix
# is applied per-invocation by the provider, NOT exported globally here.

# ── 4) name-based model caches (MatRIS/TACE/eqnorm) ──
#   MatRIS.load -> ~/.cache/matris, TACE -> ~/.cache/tace are hardcoded upstream. Symlink the
#   shared weights into the user cache only when the user does not already have their own.
for _m in matris tace eqnorm; do
  # ONLY symlink when the repo actually holds pre-staged weights at models/<m>
  # (the share-my-weights case). A fresh clone has no models/<m>, so creating a
  # symlink to an empty/absent target is harmful: these frameworks do a plain
  # `os.mkdir("$HOME/.cache/<m>")` (no exist_ok) on first use, which then fails
  # `FileExistsError` against our symlink and aborts inference. When models/<m>
  # is absent we leave ~/.cache/<m> alone so the package creates + downloads into
  # it normally. Also skip if anything (real path OR any symlink) already exists,
  # and `|| true` so sourcing env.sh can never abort the caller's `set -e`.
  if [ -d "${OH_MY_MLIP_HOME}/models/$_m" ] \
     && [ ! -e "$HOME/.cache/$_m" ] && [ ! -L "$HOME/.cache/$_m" ]; then
    mkdir -p "$HOME/.cache" 2>/dev/null || true
    ln -s "${OH_MY_MLIP_HOME}/models/$_m" "$HOME/.cache/$_m" 2>/dev/null || true
  fi
done
unset _m || true

# ── 5) JIT fast-kernel seed (OpenEquivariance libtorch_tp_jit etc.) ──
#   Some envs JIT-compile fast kernels at runtime; a clean HOME recompiles from scratch
#   (first run can "hang" for minutes). Seed a prebuilt copy into the user cache only-if-absent
#   (cp -n) so there is zero recompile and the same kernel loads immediately.
#   NOTE: NequIP/Allegro DO use this JIT path indirectly: their AOT .pt2 is built
#   with the OpenEquivariance modifier, so loading it imports openequivariance,
#   whose extension JIT-loads via torch's cpp_extension. On a cold cache that
#   recompiles (needs ninja on PATH + nvrtc.h on CPATH; see sections 3 + the
#   provider's per-env PATH injection). Seeding the prebuilt copy here skips that
#   recompile on a same-arch host.
_TE_SRC="${OH_MY_MLIP_HOME}/models/torch_ext"
if [ -d "$_TE_SRC" ]; then
  _TE_DST="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"
  mkdir -p "$_TE_DST" 2>/dev/null && cp -rn "$_TE_SRC"/. "$_TE_DST"/ 2>/dev/null
  unset _TE_DST
fi
unset _TE_SRC
