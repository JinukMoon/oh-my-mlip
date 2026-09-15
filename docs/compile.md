# Accelerators

Several equivariant models run through GPU acceleration libraries. oh-my-mlip
pins those libraries in each env and does the compile step during install, so
normally there is nothing to do by hand.

| Framework | Accelerator | What happens at install |
|---|---|---|
| NequIP | OpenEquivariance | the checkpoint is compiled into a `.pt2` for your GPU |
| Allegro | cuEquivariance | the checkpoint is compiled into a `.pt2` for your GPU |
| SevenNet | OpenEquivariance | the library's extension is built; the calculator enables it (`enable_oeq=True`) |
| EquFlash | cuEquivariance (EquFlashV2); flashTP (`EquFlash-v1`, optional) | nothing extra for EquFlashV2 |

## Ask your LLM

```text
Install NequIP and check it runs on my GPU.
```

```text
I moved to a different GPU. Recompile NequIP and Allegro for it.
```

## NequIP and Allegro

Both load an AOT-compiled model through
`NequIPCalculator.from_compiled_model(...)`, so each checkpoint must be compiled
with `nequip-compile` for the GPU it will run on. During install,
`scripts/prepare_nequip_weights.py` (NequIP) and
`scripts/prepare_allegro_weights.py` (Allegro) download the checkpoint, check its
MD5, and compile it for the current GPU into:

```
models/compiled/<arch>/<Version>_<arch>.nequip.pt2     # e.g. sm89/NequIP-OAM-L_sm89.nequip.pt2
```

The compile options follow each project's own documentation:

| Model | `nequip-compile` modifier | Import before loading |
|---|---|---|
| NequIP-OAM-XL, NequIP-OAM-L | `--modifiers enable_OpenEquivariance` | `import openequivariance` |
| Allegro-OAM-L | `--modifiers enable_CuEquivarianceContracter` | `import cuequivariance_torch` |

`resolve()` returns calculator lines that already point at the right file and
include the import, so your scripts need no change.

??? note "Run it yourself"

    Compile again for the GPU in the current machine (the arch is detected from
    that GPU):

    ```bash
    NEQUIP_PY=$(python -c 'from oh_my_mlip import resolve; print(resolve("NequIP")["python"])')
    ALLEGRO_PY=$(python -c 'from oh_my_mlip import resolve; print(resolve("Allegro")["python"])')
    "$NEQUIP_PY"  scripts/prepare_nequip_weights.py  --target-root models/nequip
    "$ALLEGRO_PY" scripts/prepare_allegro_weights.py --target-root models/allegro
    ```

    Run from the repository root; add `--dry-run` to print the commands only.
    The underlying command is:

    ```bash
    nequip-compile <ckpt> <out>.nequip.pt2 --mode aotinductor --device cuda --target ase \
        --modifiers enable_OpenEquivariance            # Allegro: enable_CuEquivarianceContracter
    ```

The first load of a NequIP model builds OpenEquivariance's kernels once and
caches them; later loads are fast.

## SevenNet

There is no separate compile step. OpenEquivariance is pinned in the SevenNet env
and built when the env is installed, and the registry's calculator line turns it
on:

```python
calc = SevenNetCalculator('7net-mf-ompa', modal='mpa', enable_oeq=True)
```

To check the backend is available inside the SevenNet env:

```bash
python -c 'from sevenn.nn.oeq_helper import is_oeq_available; print(is_oeq_available())'
```

## EquFlash

The default `EquFlashV2` uses cuEquivariance and needs nothing extra. The older
`EquFlash-v1` can use the flashTP backend, which you build yourself on a GPU
machine:

```bash
git clone https://github.com/SNU-ARC/flashTP.git
cd flashTP && pip install -r requirements.txt
CUDA_ARCH_LIST="80;90" pip install . --no-build-isolation
python -c 'import flashTP_e3nn'   # check
```

`EquFlashV2` does not accept `conv_type='flashtp'`.

## LAMMPS (Allegro)

For MD in LAMMPS instead of ASE, prepare the model for the ML-IAP interface:

```bash
nequip-prepare-lmp-mliap <ckpt> <out>
```

and build LAMMPS with ML-IAP (and KOKKOS for GPU).

See also [GPU-architecture compilation](arch_first_run_compile.md).
