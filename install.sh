#!/usr/bin/env bash
# install.sh — oh-my-mlip build-from-recipe orchestrator (PRIMARY install path).
#
# Build-from-recipe is the PRIMARY install path today: this script rebuilds a
# model's env from envs/<env>.yml on the current host (host-correct by
# construction), installs catbench, and triggers the first-run D3 compile.
# (Relocatable conda-pack tarballs are planned but NOT yet published — nothing is
# hosted — so do not rely on a tarball path.) It writes the on-disk layout:
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
#     --with-accel  opt-in (default OFF): also PRINT the curated GPU compile/accel
#                commands (NequIP/Allegro/SevenNet) for each targeted env. These
#                are upstream-doc, GPU-UNVERIFIED recipes (see docs/compile.md);
#                this flag only surfaces them — it does NOT run a GPU compile.
#                '--with-accel --dry-run' prints the install plan AND the accel
#                commands, installing nothing.
#
# Requires conda (or mamba) on PATH. D3 first-run compile needs nvcc; if absent
# this script degrades D3 OFF with a clear message (the MLIP still runs).

set -euo pipefail

# ── Resolve clone root and shared env ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export OH_MY_MLIP_HOME="${OH_MY_MLIP_HOME:-$SCRIPT_DIR}"
ENVS_DIR="$OH_MY_MLIP_HOME/envs"

DRY_RUN=0
WITH_ACCEL=0
REQUESTED=()

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --with-accel) WITH_ACCEL=1 ;;
    -h|--help)
      sed -n '2,28p' "${BASH_SOURCE[0]:-$0}"
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
  #    Prefer python3, but fall back to python so a host that only ships `python`
  #    still resolves a registered model instead of silently skipping it.
  local py_bin
  py_bin="$(command -v python3 || command -v python || true)"
  if [ -e "$MODELS_JSON" ] && [ -n "$py_bin" ]; then
    env_match="$(MODELS_JSON="$MODELS_JSON" ARG="$arg" "$py_bin" - <<'PYEOF' 2>/dev/null || true
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

# ── Opt-in accel surfacing (--with-accel) ──
# PRINTS the curated GPU compile/accel command(s) for an env, when the registry
# records an accel block for that framework. Upstream-doc, GPU-UNVERIFIED: this
# only surfaces the command (never runs a GPU compile). See docs/compile.md.
print_accel_for_env() {
  local env_name="$1" py_bin
  py_bin="$(command -v python3 || command -v python || true)"
  [ -e "$MODELS_JSON" ] && [ -n "$py_bin" ] || return 0
  MODELS_JSON="$MODELS_JSON" ENV="$env_name" "$py_bin" - <<'PYEOF' 2>/dev/null || true
import json
import os

try:
    with open(os.environ["MODELS_JSON"], "r", encoding="utf-8") as fh:
        data = json.load(fh)
except Exception:
    raise SystemExit(0)

env = os.environ["ENV"]
for name, info in data.items():
    if name.startswith("_") or not isinstance(info, dict):
        continue
    if info.get("env") != env:
        continue
    blocks = []
    for key in ("accel", "accel_lammps"):
        blk = info.get(key)
        if isinstance(blk, dict):
            blocks.append((key, blk))
    for version, vinfo in info.get("versions", {}).items():
        blk = vinfo.get("accel") if isinstance(vinfo, dict) else None
        if isinstance(blk, dict):
            blocks.append((f"{version}.accel", blk))
    if not blocks:
        print(f"    (no accel block recorded for env '{env}')")
        continue
    for label, blk in blocks:
        print(f"    [{name}.{label}] tool={blk.get('tool')}  (upstream-doc, GPU-unverified)")
        print(f"      install: {blk.get('install')}")
        if blk.get("compile_cmd"):
            print(f"      compile: {blk.get('compile_cmd')}")
        print(f"      load:    {blk.get('load_note')}")
        if blk.get("verify"):
            print(f"      verify:  {blk.get('verify')}")
PYEOF
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
if [ "$WITH_ACCEL" -eq 1 ]; then
  echo "  accel           = ON (--with-accel): curated GPU compile/accel commands"
  echo "                    will be PRINTED per env (upstream-doc, GPU-unverified;"
  echo "                    nothing GPU-compiled). See docs/compile.md."
else
  echo "  accel           = OFF (default). Pass --with-accel to print the curated"
  echo "                    GPU compile/accel commands (NequIP/Allegro/SevenNet)."
fi
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
    build_sidecar="$ENVS_DIR/$env_name.build.sh"
    if [ -e "$recipe" ]; then
      if [ -e "$build_sidecar" ]; then
        echo "  would build env '$env_name' via multi-pass sidecar $build_sidecar -> $ENVS_DIR/$env_name"
        echo "    (sidecar owns env creation + all pip passes; a single 'conda env create' cannot resolve this env)"
      else
        echo "  would create env '$env_name' from $recipe -> $ENVS_DIR/$env_name"
      fi
      echo "    then: $ENVS_DIR/$env_name/bin/pip install catbench==1.1.2"
      prestage="$OH_MY_MLIP_HOME/scripts/prestage_${env_name}_weights.py"
      if [ -e "$prestage" ]; then
        echo "    then: python3 $prestage  (pre-stage upstream weights before first use)"
      fi
    else
      echo "  SKIP '$env_name': no recipe at $recipe"
    fi
    if [ "$WITH_ACCEL" -eq 1 ]; then
      echo "  [--with-accel] curated GPU compile/accel for '$env_name' (printed only):"
      print_accel_for_env "$env_name"
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

  # Multi-pass build sidecar: a few envs cannot be built by a single
  # `conda env create` (e.g. equflash — fairchem-core 1.10 vs torch 2.9.1 is a
  # pip ResolutionImpossible that only a documented multi-pass pip resolves). If
  # `envs/<env>.build.sh` exists it OWNS env creation + all pip passes (invoked
  # with PREFIX as $1); otherwise the single-pass recipe path is used. Either way
  # install.sh still owns the catbench + D3 warm-up + sentinel steps below.
  local build_sidecar="$ENVS_DIR/$env_name.build.sh"
  if [ -e "$build_sidecar" ]; then
    echo "  building env '$env_name' via multi-pass sidecar $build_sidecar ..."
    bash "$build_sidecar" "$prefix"
  else
    echo "  creating env '$env_name' from $recipe ..."
    "$CONDA_BIN" env create --prefix "$prefix" --file "$recipe"
  fi

  # Install catbench as a post-create step (WITH its deps). catbench is
  # deliberately NOT in the recipe pip block: a bare '--no-deps' line inside a
  # conda pip block is an invalid requirement and breaks 'conda env create'.
  # Installing it here (after torch is already pinned by the recipe) is safe: a
  # real-build test confirmed catbench 1.1.2 does NOT downgrade the pinned
  # torch+cuXXX wheel, while '--no-deps' would drop catbench's real runtime deps
  # (requests, xlsxwriter, ...) and break 'from catbench.adsorption import ...'.
  echo "  installing catbench into '$env_name' ..."
  "$prefix/bin/pip" install catbench==1.1.2

  # Pre-stage upstream weights whose in-package downloader can 202-block on some
  # networks. A few frameworks (eqnorm, matris) fetch their checkpoint from a
  # plain figshare.com URL *inside their own package* into ~/.cache/<pkg>/ and
  # then skip re-download if the file "exists" (even a 0-byte one). If a
  # scripts/prestage_<env>_weights.py helper exists, run it now (pure stdlib,
  # env-independent) to stage the file via the working ndownloader subdomain
  # before the user ever builds a calculator. Non-fatal: on failure the
  # framework still falls back to its own download on first use.
  prestage="$OH_MY_MLIP_HOME/scripts/prestage_${env_name}_weights.py"
  if [ -e "$prestage" ]; then
    echo "  pre-staging weights for '$env_name' via $(basename "$prestage") ..."
    python3 "$prestage" || echo "  (weight pre-stage skipped/failed; framework will retry its own download on first use)"
  fi

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

  if [ "$WITH_ACCEL" -eq 1 ]; then
    echo "  [--with-accel] curated GPU compile/accel for '$env_name' (printed only;"
    echo "    upstream-doc, GPU-unverified — run on your GPU host; see docs/compile.md):"
    print_accel_for_env "$env_name"
  fi
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
