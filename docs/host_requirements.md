# Host requirements

Every env pins its own PyTorch (or TensorFlow) and CUDA build, so what your
machine needs depends on which models you install.

## In short

- **Linux with an NVIDIA GPU.** Installation counts as done only after the model
  computes energy and forces on the GPU.
- **conda or mamba** on `PATH`. On every host, `install.sh` runs its recipe and lock builds with
  `CONDA_DEFAULT_CHANNELS=conda-forge` (unless you set it), so conda's default Anaconda channels and
  their Terms-of-Service prompt are not used; the recipes list `nodefaults` anyway.
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

## Blackwell GPUs (sm120)

A GPU of compute capability 12.0 needs CUDA 12.8 or newer builds: the CUDA 11.8 to
12.6 torch wheels in the table above carry no kernels for it, and fail at the
first calculation with `CUDA error: no kernel image is available`. On such a GPU
`install.sh` builds a different recipe where one exists:

- allegro and nequip (already CUDA 12.8, with `ninja`) use their usual recipe;
- uma, deepmd, equiformer_v3 and pet (CUDA 12.8) and dpa4, matris and tace
  (CUDA 13.0) use a sibling (`envs/<env>.cu128.yml` / `envs/<env>.cu130.yml`)
  with the same pins plus conda's `ninja` and a pinned host compiler (`gxx` 14.4
  for CUDA 12.8, 15.3 for CUDA 13.0), so the D3 kernel builds there;
- mace, sevennet, chgnet, orb, equflash and nequix use a sibling recipe
  (`envs/<env>.cu128.yml`) with the same torch release built for CUDA 12.8;
- alphanet, fairchemv1, eqnorm and mattersim use a sibling recipe that raises
  torch to 2.7.1+cu128, because their pinned torch has no CUDA 12.8 build. This
  changes the numerical stack slightly; measured against the base stack (table
  below);
- grace uses a sibling on a TensorFlow nightly (`envs/grace.tfnightly.yml`):
  the released TensorFlow wheels up to 2.21 do not run on sm120. It is marked
  `candidate` (a nightly, and outside tensorpotential's declared TensorFlow
  range); see the recipe header;
- for an env with no working recipe there, `install.sh` stops before
  downloading anything and prints the reason.

Raised-torch siblings against the base stack (its lock replayed, run on CPU, as
the base wheels have no sm120 kernels), on a 38-atom Cu(111) slab with CO,
rattled; sibling on an sm120 GPU. ΔE in meV/atom, largest force-component
difference in meV/Å (measured 2026-11):

| Env | base torch | sibling torch | ΔE | max ΔF |
|---|---|---|---|---|
| alphanet | 2.1.2+cu121 | 2.7.1+cu128 | 0.0000 | 0.001 |
| eqnorm | 2.6.0+cu118 | 2.7.1+cu128 | 0.0000 | 0.002 |
| fairchemv1 | 2.4.1+cu121 | 2.7.1+cu128 | -0.0004 | 0.014 |
| mattersim | 2.6.0+cu124 | 2.7.1+cu128 | +0.0004 | 0.005 |

The sibling recipes also carry conda's `gxx` (14 for CUDA 12.x, 15 for CUDA 13.0)
and `ninja`: CUDA 12.x refuses a newer host gcc (Ubuntu 26.04 ships gcc 15), and
the D3 kernel and OpenEquivariance compile with them. `install.sh` (and
`adopt_env.py`) build the D3 kernel at install time with `scripts/build_d3.py`:
the env's `bin/` first on `PATH`, its own `nvcc` as `CUDA_HOME`, its conda
compiler as `CC`/`CXX`, and `TORCH_CUDA_ARCH_LIST` with the GPU's arch added (as
`env.sh` does) and cut to what that `nvcc` compiles. Nothing then compiles at the
first D3 call, which runs through the bare interpreter with no env `bin/` on
`PATH`.

`envs/_arch_recipes.json` is the list, [Recipes](recipes.md) shows each sibling's
pins, and the **sm120** column of [Supported models](model_status.md) shows what
was verified on such a GPU. The driver must support CUDA 12.8 (570+). Older
GPUs keep the recipes in the table above; nothing changes for them.

## HPC clusters: install on the login node, verify in a GPU job

On many clusters the login node has internet but no GPU, and the GPU nodes have
a GPU but no internet. So: build every env on the login node, then verify (and
compute) in batch jobs.

- **Pick the GPU architecture by hand.** The login node has no GPU to detect, so
  tell `install.sh` which GPU the jobs will use: `OMM_INSTALL_ARCH=sm80` (A100),
  `sm90` (H100/H200), `sm86`, `sm89`, `sm120`. Recipes and the D3 kernel follow it.
- **Keep envs and caches out of `$HOME`.** Home directories on clusters often allow
  only a few GB or a few tens of thousands of files; an env alone has more. Put the
  clone (`OH_MY_MLIP_HOME`) and every cache on project storage:
  ```bash
  P=/path/to/project/storage/$USER
  export OH_MY_MLIP_HOME=$P/oh-my-mlip OMM_SHARED_CACHE_ROOT=$P/cache/shared
  export CONDA_PKGS_DIRS=$P/cache/conda_pkgs PIP_CACHE_DIR=$P/cache/pip XDG_CACHE_HOME=$P/cache/xdg \
         TORCH_EXTENSIONS_DIR=$P/cache/torch_ext TRITON_CACHE_DIR=$P/cache/triton TMPDIR=$P/tmp
  ```
  Export the same lines in every job script, so the job finds the weights.
  With `OMM_SHARED_CACHE_ROOT` set, `env.sh` also moves the NequIP cache and
  `XDG_CACHE_HOME` there, and `install.sh` replaces the caches that some frameworks
  write to a fixed place in your home (`~/.cache/sevennet`, `matris`, `tace`, `nequix`,
  `eqnorm`, `grace`, `deepmd`, `~/.local/mattersim`) with symlinks into
  `$OMM_SHARED_CACHE_ROOT/home`. One that already exists as a directory is left
  where it is and listed, so you can move it yourself. The links stay in your home
  after you delete the hub or the cache root; a later install with a new root
  points them at the new one.
- **No conda on the cluster?** Some sites offer only Anaconda modules or none.
  Install Miniforge on project storage (not in `$HOME`) and put its `bin/` on
  `PATH`; `install.sh` then finds `mamba` there.
- **Task limits.** A login node may cap the processes and threads one user can
  run (cgroup `pids.max`, often 100 to 1000; `ulimit -u` does not show it). Over
  the cap, the install hangs and your shell and new logins fail too.
  `install.sh` reads the cap and, when it is below 4096, builds with at most 2
  parallel jobs and threads and skips the `.pyc` precompile (Python writes it on
  first import); it says so at the start. `OMM_MAX_PARALLEL=<n>` sets the number
  yourself. Run nothing else heavy on the login node while it builds; an AI agent
  driving the install counts against the same cap.
- **Stay logged in.** Some login nodes end every process of a user when the
  user's last session closes (systemd `KillUserProcesses=yes`), `nohup` and
  `setsid` included. Keep the session open until the install ends. If it was cut,
  run the same `install.sh` command again: finished envs are skipped, and an env
  whose build was interrupted is reported with the `--rebuild <env>` command
  that deletes and rebuilds it.
- **Weights are downloaded during the install.** Each env's install builds every
  variant's calculator once on the CPU, which downloads weights that frameworks
  fetch by name (MACE, SevenNet, ORB, ...), and fetches the rest by URL. The
  result is in `envs/<env>/.omm_prefetch.json`; a variant listed as failed there
  may not load on a node without internet. Gated models need your Hugging Face
  login before the install ([Hugging Face token](hf_token.md)).
- **NequIP and Allegro** load a model compiled for one GPU architecture, and the
  compile needs that GPU. On a login node without a GPU, the install downloads the
  model packages and prints the compile command; run it first in a GPU job (no
  internet needed there).
- **Offline GPU nodes.** A node that drops outbound packets without answering makes
  a loader that checks a URL before using its cache (ORB, for one) wait until the
  job's time limit. `env.sh` tests the network (3 s) and, without internet, sets
  `OMM_OFFLINE=1`; every env `install.sh` built then fails network access at once,
  and the loaders use the downloaded weights. `run()`, workers and `setup_verify`
  test it the same way. Set `OMM_OFFLINE=1` (or `0`) yourself to skip the test.
  Private addresses (another compute node, a site mirror) and the proxy stay
  reachable. A site host with a public address that a job must reach (for example
  a login node) goes in `OMM_OFFLINE_ALLOW=<host>[,<host>...]`.
- **Verify in a GPU job.** `setup_verify.py` refuses at once on a node without a
  GPU. A SLURM job for one model (account, partition and GPU syntax are site-specific):
  ```bash
  #!/bin/bash
  #SBATCH --account=<account> --partition=<gpu partition> --gres=gpu:1 --time=00:20:00
  #SBATCH -o verify_%j.out
  # the same exports as for the install, then:
  source "$OH_MY_MLIP_HOME/env.sh"
  export OMM_OFFLINE=1             # the node has no internet: use the downloaded weights only
  python3 "$OH_MY_MLIP_HOME/scripts/setup_verify.py" MACE-MPA-0 --json
  ```

## Common problems

- **GRACE weights:** `GRACE-2L-OAM` (about 100 MB) comes from upstream's Hugging Face
  copy, checked against the recorded sha256, and an interrupted download resumes
  where it stopped. If Hugging Face is unreachable it falls back to `grace_models`
  and the upstream ICAMS share, which has served a few KiB/s (hours for the file)
  with no resume -- let that run finish undisturbed.

- **Driver too old for a CUDA 13.0 env** (dpa4, matris, tace):
  `torch.cuda.is_available()` is `False`. Upgrade the driver, or use a host whose
  driver matches the build. A CPU run is not a general fallback here: these
  models' inference lines pin the GPU, and the hub refuses a CPU request rather
  than running on the GPU and reporting CPU.
- **`no kernel image is available` on a new GPU:** the env was built from a
  recipe for older GPUs (for example copied from another machine, or built
  before the sibling recipes existed). `./install.sh --status <env>` says which
  recipe this GPU needs; `./install.sh --rebuild <env>` rebuilds it.
- **A different GPU than the one you installed on:** NequIP and Allegro load a
  model compiled for a specific GPU architecture. See
  [GPU-architecture compilation](arch_first_run_compile.md).
- **Loading `UMA-m-1p1-*` dies with no error message:** the checkpoint (11.2 GB)
  is loaded into host RAM first, and the kernel's out-of-memory killer ends the
  process (exit code 137). Use a machine with 32 GB or more, or a `UMA-s-*` model.
- **GRACE fine-tuning on WSL crashes right after `Compiled cluster using XLA`**
  (segmentation fault, no Python traceback): rerun with
  `--set jit_compile=false`. Native Linux runs the default (`jit_compile=true`).
- **Gated models** (UMA, eSEN) need your own Hugging Face login — see
  [Hugging Face token](hf_token.md).
