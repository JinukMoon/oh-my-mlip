#!/usr/bin/env bash
# alphanet.cu128.build.sh — sm120 sibling of alphanet.build.sh (envs/alphanet.cu128.yml).
# Identical to the base sidecar except torch 2.1.2+cu121 -> 2.7.1+cu128 (no cu128
# build of 2.1.2 exists), torch_scatter +pt21cu121 -> +pt27cu128, cuda-nvcc 12.1 -> 12.8.
# The AlphaNet framework (github.com/zmyybc/AlphaNet) is a git-source package
# (setup.py). install.sh auto-runs this sidecar when present (PREFIX=$1) and
# installs catbench, D3, and sentinel afterwards.
#
# AlphaNet is pinned to a specific public commit for reproducibility.
# See models.json for framework version differences (gas-phase accuracy may vary).
# gxx=14.4.0: the host compiler nvcc 12.8 accepts (a newer system gcc, e.g. gcc 15
# on Ubuntu 26.04, is refused); ninja: torch's extension JIT (D3) runs it from PATH.
# The worker puts this env's bin/ first on PATH.
set -uo pipefail
PREFIX="${1:?usage: alphanet.cu128.build.sh <env-prefix>}"
CONDA_BIN="$(command -v conda || true)"
[ -n "$CONDA_BIN" ] || { echo "conda not found on PATH"; exit 1; }
PIP="$PREFIX/bin/pip"
SHA="65f8ea9330459e0106867d1c694aec4139c6cb19"

echo "== alphanet [0] base env (python 3.11.13 + cuda-nvcc 12.8 + ase) =="
"$CONDA_BIN" create --yes --prefix "$PREFIX" --override-channels -c conda-forge -c nvidia \
  python=3.11.13 "cuda-nvcc=12.8.*" gxx=14.4.0 ninja=1.13.2 ase pip || { echo CREATE_FAILED; exit 11; }

echo "== alphanet [1] torch 2.7.1+cu128 stack + lightning + numpy<2 =="
"$PIP" install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -f https://data.pyg.org/whl/torch-2.7.1+cu128.html \
  torch==2.7.1+cu128 torch-geometric==2.6.1 torch_scatter==2.1.2+pt27cu128 \
  lightning==2.6.6 tensorboard==2.21.0 "numpy==1.26.4" || { echo PASS1_FAILED; exit 12; }

echo "== alphanet [2] AlphaNet package (pinned SHA, --no-deps) + runtime deps =="
"$PIP" install --no-deps "alphanet @ git+https://github.com/zmyybc/AlphaNet.git@${SHA}" \
  || { echo ALPHANET_FAILED; exit 13; }
# AlphaNet runtime deps for the torch inference path. NOTE: do NOT install
# matscipy here (a JAX-path dep, not needed for AlphaNetCalculator). numpy 1.26.4 as in
# the base sidecar; install.sh's catbench step later raises it to 2.x, which torch 2.7.1
# (unlike the base's 2.1.2) supports -- the sm120 verification ran with numpy 2.4.6.
"$PIP" install pydantic==2.13.5 pydantic_settings==2.15.0 rich==15.0.0 scikit-learn==1.9.1 "numpy==1.26.4" \
  || { echo DEPS_FAILED; exit 14; }

echo "== alphanet [3] import smoke =="
"$PREFIX/bin/python" - <<'PYEOF'
from alphanet.infer.calc import AlphaNetCalculator
from alphanet.config import All_Config
print("alphanet import OK")
PYEOF
echo "alphanet sidecar done: $PREFIX/bin/python"
