#!/usr/bin/env bash
# alphanet.build.sh — build sidecar for the alphanet env.
# The AlphaNet framework (github.com/zmyybc/AlphaNet) is a git-source package
# (setup.py, editable in upstream docs). install.sh auto-runs this sidecar when
# present (PREFIX=$1) and still owns catbench + D3 + sentinel afterwards.
#
# Owner repo provided by the user: https://github.com/zmyybc/AlphaNet
# Pinned to an immutable public SHA (not a floating remote). NOTE: this public
# build computes energy+forces (tier PASS) but gas-molecule accuracy differs
# from the internal /TGM build (documented in models.json) — see candidate-reason.
set -uo pipefail
PREFIX="${1:?usage: alphanet.build.sh <env-prefix>}"
CONDA_BIN="$(command -v conda || echo /home/jumoon/miniconda3/condabin/conda)"
PIP="$PREFIX/bin/pip"
SHA="65f8ea9330459e0106867d1c694aec4139c6cb19"

echo "== alphanet [0] base env (python 3.11.13 + cuda-nvcc 12.1 + ase) =="
"$CONDA_BIN" create --yes --prefix "$PREFIX" -c conda-forge -c nvidia \
  python=3.11.13 "cuda-nvcc=12.1.*" ase pip || { echo CREATE_FAILED; exit 11; }

echo "== alphanet [1] torch 2.1.2+cu121 stack + lightning + numpy<2 =="
"$PIP" install --extra-index-url https://download.pytorch.org/whl/cu121 \
  -f https://data.pyg.org/whl/torch-2.1.2+cu121.html \
  torch==2.1.2+cu121 torch-geometric==2.6.1 torch_scatter==2.1.2+pt21cu121 \
  lightning tensorboard "numpy==1.26.4" || { echo PASS1_FAILED; exit 12; }

echo "== alphanet [2] AlphaNet package (pinned SHA, --no-deps) + runtime deps =="
"$PIP" install --no-deps "alphanet @ git+https://github.com/zmyybc/AlphaNet.git@${SHA}" \
  || { echo ALPHANET_FAILED; exit 13; }
# AlphaNet runtime deps for the torch inference path. NOTE: do NOT install
# matscipy here — it forces numpy>=2 which breaks the torch 2.1.2 ABI and is a
# JAX-path dep, not needed for AlphaNetCalculator (torch). Keep numpy==1.26.4.
"$PIP" install pydantic pydantic_settings rich scikit-learn "numpy==1.26.4" \
  || { echo DEPS_FAILED; exit 14; }

echo "== alphanet [3] import smoke =="
"$PREFIX/bin/python" - <<'PYEOF'
from alphanet.infer.calc import AlphaNetCalculator
from alphanet.config import All_Config
print("alphanet import OK")
PYEOF
echo "alphanet sidecar done: $PREFIX/bin/python"
