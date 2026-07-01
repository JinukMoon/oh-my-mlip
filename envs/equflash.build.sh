#!/usr/bin/env bash
# equflash.build.sh — multi-pass build sidecar for the equflash env.
#
# WHY THIS EXISTS: a single `conda env create --file equflash.yml` is a pip
# ResolutionImpossible (fairchem-core 1.10.0 vs torch 2.9.1+cu126). The
# documented working install (see envs/equflash.yml header) is a 2-pass pip:
# pass 1 = torch + PyG + cueq + GGNN; pass 2 = fairchem --no-deps + GGNN runtime
# deps. install.sh auto-runs this sidecar when present (PREFIX passed as $1) and
# still owns the catbench + D3 warm-up + sentinel steps afterwards.
#
# Verified on host RTX 4060 Ti sm89 (2026-06-30): EquFlashV2 tier-2 GPU PASS,
# energy=-16.391567 eV, torch.cuda.memory_allocated=200097792 B (cueq, 44.9M params).
set -euo pipefail

PREFIX="${1:?usage: equflash.build.sh <env-prefix>}"
CONDA_BIN="$(command -v conda || echo /home/jumoon/miniconda3/condabin/conda)"
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
"$PIP" install pyyaml lmdb numba scipy==1.16.0 pymatgen orjson submitit wandb \
  torchtnt pydantic huggingface_hub hydra-core tqdm pynvml \
  nvalchemi-toolkit-ops==0.3.0

echo "== equflash sidecar done: $PREFIX/bin/python =="
