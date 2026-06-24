#!/usr/bin/env bash
# install.sh — oh-my-mlip build-from-recipe FALLBACK orchestrator.
#
# PRIMARY distribution is the relocatable conda-pack tarball resolved by
# oh_my_mlip/fetch.py (see dist_manifest.json). This script is the FALLBACK: it
# rebuilds a model's env from envs/<env>.yml on the current host (host-correct by
# construction), installs catbench, and triggers the first-run D3 compile. It
# writes the same on-disk layout the tarball path produces:
#   $OH_MY_MLIP_HOME/envs/<env>/bin/python   (+ a .omm_ready sentinel)
#
# Usage:
#   ./install.sh [--dry-run] [TARGET ...]
#     TARGET     one or more env names (mace, sevennet, ...) OR registered model
#                names (MACE, SevenNet, ...), case-insensitive. Model names are
#                resolved to their env via models.json, so 'install.sh MACE',
#                'install.sh mace', and 'install.sh SevenNet' all work.
#                Default: all recipes.
#     --dry-run  print the plan and exit WITHOUT downloading/installing anything
#                (the no-network / contributor inspection path).
#
# Requires conda (or mamba) on PATH. D3 first-run compile needs nvcc; if absent
# this script degrades D3 OFF with a clear message (the MLIP still runs).

set -euo pipefail

# ── Resolve clone root and shared env ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export OH_MY_MLIP_HOME="${OH_MY_MLIP_HOME:-$SCRIPT_DIR}"
ENVS_DIR="$OH_MY_MLIP_HOME/envs"

DRY_RUN=0
REQUESTED=()

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,22p' "${BASH_SOURCE[0]:-$0}"
      exit 0
      ;;
    -*)
      echo "install.sh: unknown option '$arg'" >&2
      exit 2
      ;;
    *) REQUESTED+=("$arg") ;;
  esac
done

# ── Discover available recipes ──
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
  echo "install.sh: no env recipes found in $ENVS_DIR/*.yml" >&2
  exit 1
fi

# ── Resolve an argument to an env name ──
# Accepts EITHER an env name (e.g. 'mace') OR a registered MODEL name
# (e.g. 'MACE', 'SevenNet'), case-insensitively. A model name is mapped to its
# env via the 'env' field in models.json. Prints the resolved env on stdout, or
# echoes the original argument unchanged if nothing matches (the caller then
# SKIPs it with a clear message). This is why 'install.sh MACE' and
# 'install.sh mace' and 'install.sh SevenNet' all work.
MODELS_JSON="$OH_MY_MLIP_HOME/models.json"

resolve_to_env() {
  local arg="$1" lower env_match
  # 1) Direct (case-insensitive) match against an env recipe name.
  lower="$(printf '%s' "$arg" | tr '[:upper:]' '[:lower:]')"
  for env_match in "${ALL_ENVS[@]}"; do
    if [ "$lower" = "$(printf '%s' "$env_match" | tr '[:upper:]' '[:lower:]')" ]; then
      printf '%s\n' "$env_match"
      return 0
    fi
  done
  # 2) Treat the arg as a registered MODEL name; map model -> env via models.json
  #    (case-insensitive on the model key). Python is used only to read JSON; if
  #    it (or the file) is unavailable we fall through to returning the arg as-is.
  if [ -e "$MODELS_JSON" ] && command -v python3 >/dev/null 2>&1; then
    env_match="$(MODELS_JSON="$MODELS_JSON" ARG="$arg" python3 - <<'PYEOF' 2>/dev/null || true
import json
import os

try:
    with open(os.environ["MODELS_JSON"], "r", encoding="utf-8") as fh:
        data = json.load(fh)
except Exception:
    raise SystemExit(0)

want = os.environ["ARG"].lower()
for name, info in data.items():
    if name.startswith("_") or not isinstance(info, dict):
        continue
    if name.lower() == want and isinstance(info.get("env"), str):
        print(info["env"])
        break
PYEOF
)"
    if [ -n "$env_match" ]; then
      printf '%s\n' "$env_match"
      return 0
    fi
  fi
  # 3) No match -> echo the arg unchanged; the caller SKIPs it loudly.
  printf '%s\n' "$arg"
}

# If no env names were given, target every recipe.
if [ "${#REQUESTED[@]}" -eq 0 ]; then
  TARGETS=("${ALL_ENVS[@]}")
else
  TARGETS=()
  for arg in "${REQUESTED[@]}"; do
    TARGETS+=("$(resolve_to_env "$arg")")
  done
fi

# ── Detect conda / mamba (skipped under --dry-run so the plan always prints) ──
CONDA_BIN=""
if [ "$DRY_RUN" -eq 0 ]; then
  if command -v mamba >/dev/null 2>&1; then
    CONDA_BIN="mamba"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN="conda"
  else
    echo "install.sh: neither 'mamba' nor 'conda' found on PATH." >&2
    echo "  Install Miniconda/Miniforge first, then re-run." >&2
    exit 1
  fi
fi

# ── nvcc detection: governs whether first-run D3 compile is possible ──
NVCC_OK=0
if command -v nvcc >/dev/null 2>&1; then
  NVCC_OK=1
fi

echo "oh-my-mlip install (fallback / build-from-recipe)"
echo "  OH_MY_MLIP_HOME = $OH_MY_MLIP_HOME"
echo "  recipes dir     = $ENVS_DIR"
echo "  targets         = ${TARGETS[*]}"
if [ "$NVCC_OK" -eq 1 ]; then
  echo "  nvcc            = $(command -v nvcc)  (D3 first-run compile enabled)"
else
  echo "  nvcc            = NOT FOUND"
  echo "    -> D3 first-run compile of pair_d3.so is not possible on this host."
  echo "    -> Option 1: install a CUDA toolkit providing nvcc, point CUDA_HOME at it,"
  echo "       and re-run; then D3 compiles on first use."
  echo "    -> Option 2: fetch a prebuilt-per-arch pair_d3.so matching this GPU's"
  echo "       compute capability (sm86/sm89) into the env's catbench/dispersion/cuda/."
  echo "    -> Otherwise D3 is degraded OFF: the MLIP still runs; only the dispersion"
  echo "       correction is unavailable. See docs/arch_first_run_compile.md."
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo
  echo "[--dry-run] Plan only; nothing downloaded or installed."
  for env_name in "${TARGETS[@]}"; do
    recipe="$ENVS_DIR/$env_name.yml"
    if [ -e "$recipe" ]; then
      echo "  would create env '$env_name' from $recipe -> $ENVS_DIR/$env_name"
    else
      echo "  SKIP '$env_name': no recipe at $recipe"
    fi
  done
  echo "  would: source $OH_MY_MLIP_HOME/env.sh"
  echo "  would: trigger first-run D3 compile per env (if nvcc present)"
  exit 0
fi

# ── Real install path ──
# shellcheck source=/dev/null
source "$OH_MY_MLIP_HOME/env.sh"

install_one() {
  local env_name="$1"
  local recipe="$ENVS_DIR/$env_name.yml"
  local prefix="$ENVS_DIR/$env_name"
  local sentinel="$prefix/.omm_ready"

  if [ ! -e "$recipe" ]; then
    echo "install.sh: SKIP '$env_name' (no recipe at $recipe)" >&2
    return 1
  fi

  if [ -e "$sentinel" ]; then
    echo "  '$env_name' already installed (sentinel present) — skipping."
    return 0
  fi

  echo "  creating env '$env_name' from $recipe ..."
  "$CONDA_BIN" env create --prefix "$prefix" --file "$recipe"

  # Trigger first-run D3 compile so the user does not pay the cost mid-workflow.
  if [ "$NVCC_OK" -eq 1 ]; then
    echo "  triggering first-run D3 compile for '$env_name' ..."
    "$prefix/bin/python" - <<'PYEOF' || echo "  (D3 warm-up skipped/failed; D3 will retry on first real use)"
try:
    from ase.build import bulk
    from catbench.dispersion import DispersionCorrection
    at = bulk("Cu", "fcc", a=3.61, cubic=True)
    DispersionCorrection()  # constructing triggers the kernel build
    print("  D3 kernel ready")
except Exception as exc:  # noqa: BLE001
    print("  D3 warm-up note:", repr(exc)[:160])
PYEOF
  else
    echo "  nvcc absent: D3 left to degrade off for '$env_name' (MLIP still runs)."
  fi

  : > "$sentinel"
  echo "  '$env_name' ready -> $prefix/bin/python"
}

rc=0
for env_name in "${TARGETS[@]}"; do
  install_one "$env_name" || rc=1
done

echo
if [ "$rc" -eq 0 ]; then
  echo "install.sh: done. Try: $OH_MY_MLIP_HOME/run_examples/single_point.py"
else
  echo "install.sh: completed with one or more skipped/failed envs (see above)." >&2
fi
exit "$rc"
