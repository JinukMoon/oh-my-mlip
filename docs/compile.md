# GPU compile / acceleration recipes (NequIP, Allegro, SevenNet, EquFlash)

> **Provenance: upstream-doc — GPU-UNVERIFIED.**
> Every command on this page is curated from the model's upstream (or owner)
> documentation. It has **not** been run end-to-end on a GPU in this repo. The
> machine-readable source of truth is the `accel` blocks in `models.json`, where
> each block carries `verified: false`, `provenance`, and `last_gpu_verified:
> null`. `scripts/verify_compile.py` asserts that contract on every block and
> prints `shape-only; gpu_unverified`. Treat these as the starting recipe to run
> on **your** GPU host, not as a passed result.

Some equivariant MLIPs need a GPU-side acceleration step to reach their fast
path: an AOT compile (`.pt2`) for NequIP/Allegro, or an acceleration extra for
SevenNet. This is **opt-in** and separate from the normal `install.sh` env build.
`install.sh --with-accel` prints these same commands per env (it never runs a GPU
compile).

## Why this is opt-in and unverified

The accel backends (OpenEquivariance, cuEquivariance, flashTP, LAMMPS MLIAP)
require an NVIDIA GPU plus a toolchain (`torch>=2.7`, GCC9+ for OpenEquivariance).
The compiled artifact is **arch-specific** and is produced on the user's GPU — it
is never baked into a distributed tarball (see `docs/arch_first_run_compile.md`).
Because this repo's CI is GPU-free, we surface the recipe and verify its *shape*,
but we do not claim a passed GPU run.

## NequIP / Allegro — OpenEquivariance AOT compile

Both load through nequip's compiled-model ASE path
(`NequIPCalculator.from_compiled_model`), so a per-arch `.pt2` must be produced
with `nequip-compile`.

```bash
# 1. install the accel backend (NVIDIA GPU, torch>=2.7, GCC9+)
pip install openequivariance

# 2. compile the checkpoint to an AOT .pt2 (wrapper: scripts/compile_nequip.sh)
nequip-compile <ckpt> <out>.nequip.pt2 \
    --mode aotinductor --device cuda --target ase \
    --modifiers enable_OpenEquivariance

# 3. load in Python
python - <<'PY'
import openequivariance
from nequip.ase import NequIPCalculator
calc = NequIPCalculator.from_compiled_model("<out>.nequip.pt2", device="cuda")
PY
```

* For `torch < 2.10`, compile to a TorchScript `.nequip.pth` instead of the
  AOTInductor `.pt2`.
* LAMMPS deploy (Allegro): `nequip-prepare-lmp-mliap <ckpt> <out>` and build
  LAMMPS with the MLIAP/KOKKOS C++ path (see the `Allegro.accel_lammps` block).

Wrapper: `scripts/compile_nequip.sh --dry-run <ckpt> [<out>.nequip.pt2]` prints
the exact command; drop `--dry-run` to run it on a GPU host.

## SevenNet — OpenEquivariance / cuEquivariance extra

SevenNet has no separate AOT compile step. Acceleration is an install-time extra,
selected at calculator construction.

```bash
# install one extra (oeq = OpenEquivariance; cueq12/cueq13 = cuEquivariance by CUDA major)
pip install sevenn[oeq]
# or: pip install sevenn[cueq12]    # CUDA 12
# or: pip install sevenn[cueq13]    # CUDA 13

# enable at construction
python - <<'PY'
from sevenn.calculator import SevenNetCalculator
calc = SevenNetCalculator("7net-mf-ompa", modal="mpa", enable_oeq=True)
PY
# (or pass --enable_oeq / --enable_cueq on the sevenn CLI)

# verify the backend is importable on the GPU host
python -c 'from sevenn.nn.oeq_helper import is_oeq_available; print(is_oeq_available())'
```

Wrapper: `scripts/compile_sevennet.sh --dry-run [oeq|cueq12|cueq13]`.

## EquFlash v1 — flashTP backend (owner-doc)

The `flashtp` backend applies to **EquFlash v1 only**. EquFlashV2 is cueq-only and
rejects `conv_type='flashtp'` (TypeError). Recorded under
`EquFlash.versions.EquFlash.accel`.

```bash
git clone https://github.com/SNU-ARC/flashTP.git
cd flashTP && pip install -r requirements.txt
CUDA_ARCH_LIST="80;90" pip install . --no-build-isolation
python -c 'import flashTP_e3nn'   # verify
```

## Verifying the contract (GPU-free)

```bash
python scripts/verify_compile.py
```

This asserts every `accel` block carries the required keys, is marked
`verified: false` with `last_gpu_verified: null`, and declares a recognized
`provenance`, then prints `shape-only; gpu_unverified`. When a recipe is actually
run and validated on a GPU, the owner flips `verified` to `true` and sets
`last_gpu_verified` to the date — at which point this banner no longer applies to
that block.
