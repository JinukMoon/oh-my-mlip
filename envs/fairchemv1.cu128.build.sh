#!/usr/bin/env bash
# fairchemv1.cu128.build.sh — sm120 sibling build for the fairchemv1 env
# (recipe: envs/fairchemv1.cu128.yml). The base env (envs/fairchemv1.yml) is a
# single-pass recipe with torch 2.4.1+cu121; no cu128 build of torch 2.4.1 exists,
# so this sibling raises torch to 2.7.1+cu128. fairchem-core 1.10.0 (the last 1.x)
# declares torch~=2.4.0, which makes a single resolve impossible: it is installed
# --no-deps and its runtime deps are installed explicitly, pinned to the versions
# in the base lock (envs/locks/fairchemv1.pip.txt) -- the equflash pattern.
# install.sh runs this sidecar when present (PREFIX=$1) and installs catbench,
# weights, D3 warm-up and the sentinel afterwards.
# gxx=14.4.0: the host compiler nvcc 12.8 accepts (a newer system gcc, e.g. gcc 15
# on Ubuntu 26.04, is refused); ninja: torch's extension JIT (D3) runs it from PATH.
# The worker puts this env's bin/ first on PATH.
set -euo pipefail

PREFIX="${1:?usage: fairchemv1.cu128.build.sh <env-prefix>}"
CONDA_BIN="$(command -v conda || true)"
[ -n "$CONDA_BIN" ] || { echo "conda not found on PATH"; exit 1; }
PIP="$PREFIX/bin/pip"

echo "== fairchemv1 sidecar [0] base env (python 3.11.13 + cuda-nvcc 12.8 + ase 3.25.0) =="
"$CONDA_BIN" create --yes --prefix "$PREFIX" --override-channels -c conda-forge -c nvidia \
  python=3.11.13 "cuda-nvcc=12.8.*" gxx=14.4.0 ninja=1.13.2 ase=3.25.0 pip

echo "== fairchemv1 sidecar [1] torch 2.7.1+cu128 + PyG (+pt27cu128) + pinned base deps =="
"$PIP" install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -f https://data.pyg.org/whl/torch-2.7.1+cu128.html \
  torch==2.7.1+cu128 torch-geometric==2.6.1 torch_scatter==2.1.2+pt27cu128 \
  torch_sparse==0.6.18+pt27cu128 e3nn==0.5.6 scipy==1.16.0 setuptools==78.1.1 numpy==1.26.4

echo "== fairchemv1 sidecar [2a] fairchem-core 1.10.0 WITHOUT deps (it pins torch~=2.4.0) =="
"$PIP" install --no-deps fairchem-core==1.10.0

echo "== fairchemv1 sidecar [2b] fairchem-core runtime deps (versions from the base lock) =="
"$PIP" install huggingface_hub==1.32.0 hydra-core==1.3.7 lmdb==2.3.0 numba==0.67.0 orjson==3.12.0 \
  pydantic==2.13.5 pymatgen==2026.5.4 PyYAML==6.0.3 requests==2.34.2 submitit==1.5.4 \
  tensorboard==2.21.0 torchtnt==0.2.4 tqdm==4.70.1 wandb==0.30.0 numpy==1.26.4

echo "== fairchemv1 sidecar [3] import smoke =="
"$PREFIX/bin/python" -c "import torch, fairchem.core; from fairchem.core import OCPCalculator; print('fairchemv1 import OK, torch', torch.__version__)"
echo "== fairchemv1 sidecar done: $PREFIX/bin/python =="
