# Host requirements

Every env pins its own PyTorch (or TensorFlow) and CUDA build, so what your
machine needs depends on which models you install.

## In short

- **Linux with an NVIDIA GPU.** Installation counts as done only after the model
  computes energy and forces on the GPU.
- **conda or mamba** on `PATH`.
- **An NVIDIA driver new enough for the env's CUDA build** — see the table. Check
  yours with `nvidia-smi` (the CUDA version at the top right).
- **Disk:** each env takes several GB, plus the model weights.
- **Host RAM:** 16 GB is enough for every model except `UMA-m-1p1-*`, which needs
  32 GB or more.

## Per env

| Env | Python | Framework | CUDA | Driver | Notes |
|---|---|---|---|---|---|
| allegro | 3.11 | torch 2.8.0 | 12.8 | 570+ | compiled for your GPU at install (CuEquivariance) |
| alphanet | 3.11 | torch 2.1.2 | 12.1 | 525+ | |
| chgnet | 3.11 | torch 2.7.1 | 12.6 | 525+ | |
| deepmd | 3.11 | torch 2.8.0 | 12.8 | 570+ | |
| dpa4 | 3.11 | torch 2.11.0 | 13.0 | 580+ | |
| eqnorm | 3.11 | torch 2.6.0 | 11.8 | 450+ | |
| equflash | 3.12 | torch 2.9.1 | 12.6 | 525+ | |
| equiformer_v3 | 3.11 | torch 2.7.1 | 12.8 | 570+ | |
| fairchemv1 | 3.11 | torch 2.4.1 | 12.1 | 525+ | gated weights (eSEN) |
| grace | 3.11 | TensorFlow 2.16.2 | 12.3 | 525+ | |
| mace | 3.11 | torch 2.7.1 | 12.6 | 525+ | |
| matris | 3.11 | torch 2.12.1 | 13.0 | 580+ | |
| mattersim | 3.10 | torch 2.6.0 | 12.4 | 525+ | |
| nequip | 3.11 | torch 2.9.1 | 12.8 | 570+ | compiled for your GPU at install (OpenEquivariance) |
| nequix | 3.11 | torch 2.10.0 | 12.6 | 525+ | |
| orb | 3.11 | torch 2.7.1 | 12.6 | 525+ | |
| pet | 3.11 | torch 2.9.1 | 12.8 | 570+ | 2.8 GB checkpoint |
| sevennet | 3.11 | torch 2.7.1 | 12.6 | 525+ | OpenEquivariance kernels built at install |
| tace | 3.11 | torch 2.11.0 | 13.0 | 580+ | |
| uma | 3.11 | torch 2.8.0 | 12.8 | 570+ | gated weights; `UMA-m-1p1-*` needs 32 GB+ host RAM |

Versions come from the lock files in `envs/locks/`; driver floors are the
approximate Linux minimums for each CUDA build.

## Common problems

- **GRACE weights download very slowly:** `GRACE-2L-OAM` (about 100 MB) is fetched
  by `grace_models` from the upstream ICAMS share, which has been seen serving a few
  KiB/s -- several hours for the full file. The download does not resume: an
  interrupted fetch starts over. Let the first `setup_verify.py GRACE` run finish
  undisturbed, on a host with outbound network, before using the model elsewhere.

- **Driver too old for a CUDA 13.0 env** (dpa4, matris, tace):
  `torch.cuda.is_available()` is `False`. Upgrade the driver, or use a host whose
  driver matches the build. A CPU run is not a general fallback here: these
  models' inference lines pin the GPU, and the hub refuses a CPU request rather
  than running on the GPU and reporting CPU.
- **A different GPU than the one you installed on:** NequIP and Allegro load a
  model compiled for a specific GPU architecture. See
  [GPU-architecture compilation](arch_first_run_compile.md).
- **Loading `UMA-m-1p1-*` dies with no error message:** the checkpoint (11.2 GB)
  is loaded into host RAM first, and the kernel's out-of-memory killer ends the
  process (exit code 137). Use a machine with 32 GB or more, or a `UMA-s-*` model.
- **Gated models** (UMA, eSEN) need your own Hugging Face login — see
  [Hugging Face token](hf_token.md).
