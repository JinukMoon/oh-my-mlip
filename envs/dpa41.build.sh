#!/usr/bin/env bash
# dpa41.build.sh — build sidecar for the dpa41 env (DPA-4.1).
# deepmd-kit is built from a git commit with its PyTorch C++ op. The op must link
# the CUDA 13 libraries of THIS env, so CUDA_HOME points at the env prefix and the
# env's bin leads PATH during the build (see envs/dpa41.yml). install.sh runs this
# sidecar when present (PREFIX=$1) and installs catbench, D3 and the sentinel after.
set -uo pipefail
PREFIX="${1:?usage: dpa41.build.sh <env-prefix>}"
CONDA_BIN="$(command -v conda || true)"
[ -n "$CONDA_BIN" ] || { echo "conda not found on PATH"; exit 1; }
PIP="$PREFIX/bin/pip"
SHA="1ded5d2429467fb16cb1d6c3d3059fac881f82a7"

echo "== dpa41 [0] base env (python 3.11.15 + CUDA 13.0 toolkit + ase) =="
"$CONDA_BIN" create --yes --prefix "$PREFIX" --override-channels -c conda-forge -c nvidia \
  python=3.11.15 pip "cuda-nvcc=13.0.*" "cuda-cudart-dev=13.0.96" "cuda-nvrtc-dev=13.0.88" ase=3.28.0 \
  || { echo CREATE_FAILED; exit 11; }

echo "== dpa41 [1] torch 2.11.0+cu130 =="
"$PIP" install --extra-index-url https://download.pytorch.org/whl/cu130 torch==2.11.0+cu130 \
  || { echo TORCH_FAILED; exit 12; }

echo "== dpa41 [2] deepmd-kit @ ${SHA} (PyTorch op only, against this env's CUDA 13) =="
# torch's CMake config calls the legacy FindCUDA, which takes the toolkit root
# from nvcc's location and looks for <root>/include and <root>/lib64; the conda
# CUDA toolkit keeps headers and libraries in targets/<arch>/ instead ("Could NOT
# find CUDA (missing: CUDA_INCLUDE_DIRS)"). deepmd's build backend does not pass
# CMAKE_ARGS on, so give the build a standard layout: <prefix>/cuda_root with
# bin, include, lib64 and nvvm linking into this env, first on PATH and as CUDA_HOME.
TGT="$(ls -d "$PREFIX"/targets/* | head -1)"
ROOT="$PREFIX/cuda_root"
rm -rf "$ROOT"; mkdir -p "$ROOT"
# bin is the env's own bin (nvcc finds nvcc.profile and crt/ next to itself, and
# reads TOP as <bin>/../targets/<arch>, so the other links resolve through here)
ln -sfn "$PREFIX/bin" "$ROOT/bin"
ln -sfn "$TGT/include" "$ROOT/include"
ln -sfn "$TGT/lib" "$ROOT/lib64"
ln -sfn "$PREFIX/targets" "$ROOT/targets"
# CMake's CUDA language detection takes <root>/nvvm/libdevice as the library root
ln -sfn "$PREFIX/nvvm" "$ROOT/nvvm"
PATH="$ROOT/bin:$PREFIX/bin:$PATH" CUDA_HOME="$ROOT" CUDA_PATH="$ROOT" CUDAToolkit_ROOT="$ROOT" CUDA_TOOLKIT_ROOT_DIR="$ROOT" \
  DP_ENABLE_TENSORFLOW=0 DP_ENABLE_PYTORCH=1 CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}" \
  "$PIP" install --extra-index-url https://download.pytorch.org/whl/cu130 \
  "deepmd-kit @ git+https://github.com/deepmodeling/deepmd-kit@${SHA}" e3nn==0.6.0 mpich==5.0.1 dpdata==1.1.0 \
  || { echo DEEPMD_FAILED; exit 13; }

echo "== dpa41 [3] the op links CUDA 13 nvrtc, not 12 =="
op="$(ls "$PREFIX"/lib/python3.11/site-packages/deepmd/lib/libdeepmd_op_pt.so 2>/dev/null | head -1)"
if [ -n "$op" ] && ldd "$op" | grep -q "libnvrtc.so.12"; then
  echo "  $op links libnvrtc.so.12 (a CUDA 12 toolkit was found first); it fails under a clean LD_LIBRARY_PATH"
  exit 14
fi
echo "== dpa41 [4] import smoke (clean LD_LIBRARY_PATH) =="
LD_LIBRARY_PATH="" "$PREFIX/bin/python" -c "from deepmd.calculator import DP; print('deepmd import OK')" \
  || { echo IMPORT_FAILED; exit 15; }
echo "dpa41 sidecar done: $PREFIX/bin/python"
