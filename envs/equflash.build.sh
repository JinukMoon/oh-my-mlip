#!/usr/bin/env bash
# equflash.build.sh — multi-pass build sidecar for the equflash env.
#
# WHY THIS EXISTS: a single `conda env create --file equflash.yml` is a pip
# ResolutionImpossible (fairchem-core 1.10.0 vs torch 2.9.1+cu126). The
# documented working install (see envs/equflash.yml) is a 2-pass pip:
# pass 1 = torch + PyG + cueq + GGNN; pass 2 = fairchem --no-deps + GGNN runtime deps.
# install.sh auto-runs this sidecar when present (PREFIX passed as $1) and handles
# catbench, D3, and sentinel steps afterwards.
set -euo pipefail

PREFIX="${1:?usage: equflash.build.sh <env-prefix>}"
CONDA_BIN="$(command -v conda || true)"
[ -n "$CONDA_BIN" ] || { echo "conda not found on PATH"; exit 1; }
PIP="$PREFIX/bin/pip"

echo "== equflash sidecar [0] base env (python 3.12.13 + cuda-nvcc 12.6 + ase) =="
"$CONDA_BIN" create --yes --prefix "$PREFIX" -c conda-forge -c nvidia \
  python=3.12.13 "cuda-nvcc=12.6.*" ase pip

echo "== equflash sidecar [1] pass 1: torch + PyG + cueq + GGNN =="
"$PIP" install --extra-index-url https://download.pytorch.org/whl/cu126 \
  -f https://data.pyg.org/whl/torch-2.9.1+cu126.html \
  torch==2.9.1+cu126 torch-geometric==2.6.1 torch_scatter==2.1.2+pt29cu126 \
  torch_sparse==0.6.18+pt29cu126 e3nn==0.5.6 cuequivariance==0.6.0 \
  cuequivariance-torch==0.6.0 cuequivariance-ops-torch-cu12==0.6.0 \
  "git+https://github.com/SamsungDS/GGNN@16b5cae474370977b59120e8bc57e4bcc19cd093#egg=GGNN"

echo "== equflash sidecar [2a] pass 2: fairchem WITHOUT deps =="
"$PIP" install --no-deps fairchem-core==1.10.0

echo "== equflash sidecar [2b] pass 2: GGNN runtime deps (incl. nvalchemi ops) =="
"$PIP" install pyyaml==6.0.3 lmdb==2.3.0 numba==0.67.0 scipy==1.16.0 pymatgen==2026.5.4 orjson==3.12.0 submitit==1.5.4 wandb==0.30.0 \
  torchtnt==0.2.4 pydantic==2.13.5 huggingface_hub==1.31.0 hydra-core==1.3.6 tqdm==4.70.1 pynvml==13.0.1 \
  nvalchemi-toolkit-ops==0.3.0

echo "== equflash sidecar done: $PREFIX/bin/python =="
