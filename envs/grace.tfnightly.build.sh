#!/usr/bin/env bash
# grace.tfnightly.build.sh — sm120 sibling build for the grace env
# (recipe and the reason it exists: envs/grace.tfnightly.yml).
# install.sh runs this sidecar when present (PREFIX=$1) and installs catbench,
# prepares the weights (scripts/prepare_grace_weights.py) and writes the sentinel.
set -euo pipefail

PREFIX="${1:?usage: grace.tfnightly.build.sh <env-prefix>}"
CONDA_BIN="$(command -v conda || true)"
[ -n "$CONDA_BIN" ] || { echo "conda not found on PATH"; exit 1; }
PIP="$PREFIX/bin/pip"

echo "== grace sidecar [0] base env (python 3.11.11 + ase 3.29.0) =="
"$CONDA_BIN" create --yes --prefix "$PREFIX" --override-channels -c conda-forge \
  python=3.11.11 ase=3.29.0 pip

echo "== grace sidecar [1] TensorFlow nightly (runs on sm120) + tensorpotential runtime deps =="
"$PIP" install "tf-nightly[and-cuda]==2.23.0.dev20261117" tf_keras-nightly==2.23.0.dev2026111709 \
  keras-nightly==3.16.0.dev2026111703 matscipy==1.3.1 sympy==1.14.0 pandas==2.3.3 pyyaml==6.0.3 tqdm==4.70.1

echo "== grace sidecar [2] tensorpotential 0.5.3 WITHOUT deps (it declares tensorflow<2.20) =="
"$PIP" install --no-deps tensorpotential==0.5.3

echo "== grace sidecar [3] import smoke =="
TF_USE_LEGACY_KERAS=1 "$PREFIX/bin/python" -c "import tensorflow as tf; from tensorpotential.calculator import TPCalculator; print('grace import OK, TF', tf.__version__, tf.config.list_physical_devices('GPU'))"
echo "== grace sidecar done: $PREFIX/bin/python =="
