# GPU-architecture compilation

Two kinds of artifact are tied to the GPU they were built on, so oh-my-mlip builds
them on your machine instead of downloading them:

- the compiled NequIP and Allegro models (`.pt2`), built during install;
- the CUDA kernel for CatBench's D3 dispersion correction, built on first use.

## Ask your LLM

```text
I'm moving my NequIP jobs to a machine with a different GPU. Prepare the model for it.
```

## GPU architecture names

The architecture is the GPU's CUDA compute capability, written `sm<major><minor>`:

| Arch | Example GPUs |
|---|---|
| `sm80` | A100 |
| `sm86` | RTX A5000, RTX A6000, RTX 30xx |
| `sm89` | L40S, RTX 40xx |

`nvidia-smi --query-gpu=compute_cap --format=csv,noheader` prints yours (e.g.
`8.9` = `sm89`). A model compiled for one architecture is not guaranteed to run on
another.

## NequIP and Allegro models

During install, `scripts/prepare_nequip_weights.py` and
`scripts/prepare_allegro_weights.py` compile each checkpoint for the GPU in the
machine into `models/compiled/<arch>/`. See [Accelerators](compile.md) for the
compile options.

At run time `resolve()` fills in the architecture: it detects the current GPU
with `nvidia-smi` and puts that arch into the calculator line's
`models/compiled/<arch>/...` path. To write jobs for another machine, pass it
explicitly:

```python
spec = resolve("NequIP", arch="sm86")
```

The `.pt2` for that arch must exist. Compile it on a machine with that GPU by
running the prepare script there (it detects the arch from the GPU); see
"Run it yourself" on [Accelerators](compile.md).

## D3 dispersion kernel

CatBench's D3 correction (`catbench.dispersion`) uses a CUDA kernel that is
compiled for your GPU and cached. `install.sh` triggers that build once when
`nvcc` is available, so a later calculation does not stop to compile.

The build needs the CUDA toolkit's `nvcc`. `env.sh` finds the toolkit
(`CUDA_HOME`) automatically. Check with:

```bash
source env.sh
command -v nvcc && nvcc --version
```

If `nvcc` is missing, install a CUDA toolkit and run `install.sh` again. The
models themselves (energy and forces) do not need `nvcc`; without it, only D3 is
unavailable, and `install.sh` says so.
