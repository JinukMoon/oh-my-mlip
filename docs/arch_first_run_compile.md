# First-run compilation on your GPU (D3 `.so`, NequIP/Allegro `.pt2`)

Some artifacts are **architecture-specific** — they are tied to your GPU's CUDA
compute capability. oh-my-mlip **never bakes these into the distributed tarballs**
(see `scripts/build_conda_pack.sh`, which strips them). Instead they are compiled
or reselected on **your** GPU the first time you run. This page explains what
compiles, why, and what to do if your machine lacks `nvcc`.

## Compute capability: sm86 vs sm89

| Compute capability | Example GPUs |
|---|---|
| `sm86` | NVIDIA A5000, A6000 (Ampere) |
| `sm89` | NVIDIA L40S (Ada Lovelace) |

An artifact compiled for one is not guaranteed to run on the other, which is
exactly why we never ship a prebuilt one — the author's GPU may differ from yours.

## What compiles / reselects on first run

### 1. D3 dispersion kernel — `pair_d3.so`

catbench's D3 correction uses a CUDA kernel (`catbench/dispersion/cuda/pair_d3.so`).
On first D3 use it is JIT-compiled for your GPU and then cached (under your torch
extensions cache) and reused. `env.sh` sets `PYTHONUTF8=1` (prevents an
ascii-decode crash during the build) and autodetects `CUDA_HOME`.

### 2. NequIP / Allegro AOT models — `.pt2`

Models marked `arch_pinned: true` in `models.json` (NequIP, Allegro) load an
AOT-compiled `.pt2` via `NequIPCalculator.from_compiled_model(...)`. The registry
exposes `inference_sm86` / `inference_sm89`; the variant matching your GPU is
selected and the `.pt2` is recompiled/reselected into
`models/compiled/{sm86,sm89}/` on first run. Selecting the wrong arch is a runtime
error — let the tooling pick by host GPU rather than hard-coding a path.

> Note: NequIP/Allegro use the AOT `.pt2` path, not the runtime JIT-kernel seed in
> `env.sh` section 5 — that seed is for OpenEquivariance-style kernels only.

## `nvcc` requirement

First-run compilation of the D3 kernel needs the CUDA toolkit's `nvcc` on `PATH`.
`env.sh` autodetects `CUDA_HOME` in this order: an existing `CUDA_HOME` →
`nvcc` on `PATH` → `/usr/local/cuda`. Check with:

```bash
source env.sh
command -v nvcc && nvcc --version
```

## If `nvcc` is absent — fallbacks

`install.sh` detects a missing `nvcc` and tells you which path it is taking. Your
options:

1. **Install a CUDA toolkit** that provides `nvcc`, point `CUDA_HOME` at it, and
   re-run. D3 then compiles on first use. This is the recommended path.
2. **Fetch a prebuilt-per-arch `pair_d3.so`** matching your GPU's compute
   capability (sm86 or sm89) and drop it into the env's
   `catbench/dispersion/cuda/` directory. Use this only when you cannot install a
   toolkit and you are certain of your GPU's arch.
3. **Degrade D3 off.** The MLIP itself does not need `nvcc` and runs normally; only
   the dispersion correction is unavailable. `install.sh` prints a clear message
   and never silently produces wrong numbers.

The MLIP forward pass (energy/forces) does **not** require `nvcc` — only the D3
add-on and the arch-pinned `.pt2` selection do. So a no-`nvcc` host can still run
every non-arch-pinned model without dispersion.
