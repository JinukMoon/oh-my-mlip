# Host requirements & equivalence matrix

Each env is pinned to a specific torch/CUDA build, so the **host must meet that
recipe's NVIDIA-driver floor** (and, for a few, a GPU-arch compile). This is the
honest portability story: oh-my-mlip removes the env/calculator/weights pain on a
**compatible host** — it is not magic universal portability. Driver floors are
approximate Linux minimums for the bundled CUDA runtime.

**Real-usage bar (builds + runs).** On the maintainer's RTX 4060 Ti (sm89, CUDA
12.8) all **20/20 envs build and compute energy + forces** — 17 on the GPU
directly, and **dpa4 / tace / matris on CPU** (their cu130 build needs a CUDA-13
driver). The only out-of-the-box gap is a **weights auto-download** quirk in a
couple of upstream packages on some networks (eqnorm / matris — see their notes),
which has a one-line recovery.

**Stricter bar (bit-reproduces our reference).** Equivalence = single-point energy
vs the validated /TGM hub on BackSingle2018 (per-atom, tol 1e-3 eV/atom).
**17/20 envs matched** (most ≤ 1e-6, several bit-identical), across PyTorch +
TensorFlow, incl. both gated envs. The other 3 are **not wrong-value failures**:
AlphaNet runs and uses the **same weights** (slab energies are bit-identical) but
the public commit's gas-phase energies drift ~0.24 eV/atom (owner must pin the
exact commit); MatRIS and Nequix are simply **not in /TGM**, so there is no
reference to compare. See `docs/equiv_results.md` for the full per-model numbers.

| env | status | py | torch | CUDA | driver floor | equivalence | host requirement / note |
|---|---|---|---|---|---|---|---|
| chgnet | clean | 3.11.13 | 2.7.1 | cu126 | 525+ | ✅ matched (9.2e-07 eV/atom) | — |
| deepmd | clean | 3.10.19 | 2.8.0 | cu128 | 570+ | ✅ matched (1.5e-07 eV/atom) | needs pip `mpich` |
| dpa4 | candidate | 3.11.15 | 2.11.0 | cu130 | 580+ (CUDA 13) | ✅ matched (CPU) (2.2e-07 eV/atom) | needs CUDA-13 driver for GPU; ran CPU on CUDA-12 box; pip `mpich` |
| eqnorm | clean | 3.11.13 | 2.6.0 | cu118 | 450+ | ✅ matched (0.0 (all bit-identical)) | runs on GPU; pkg auto-dl 202-blocks on some nets → 1-line curl recovery (see note) |
| equiformer_v3 | candidate | 3.11.15 | 2.7.1 | cu128 | 570+ | ✅ matched (0.0 (all bit-identical)) | vendored-fairchem editable build; owner-pin sha pending |
| fairchemv1 | clean | 3.11.13 | 2.4.1 | cu121 | 525+ | ✅ matched (0.0 (all bit-identical)) | gated weights (HF token) |
| grace | clean | 3.11.11 | (TF) | — | — | ✅ matched (machine precision (TensorFlow)) | TF-only; no GPU driver floor for inference |
| mace | clean | 3.11.13 | 2.7.1 | cu126 | 525+ | ✅ matched (machine precision (float64)) | — |
| mattersim | clean | 3.10.16 | 2.6.0 | cu124 | 525+ | ✅ matched (0.0 (all bit-identical)) | — |
| orb | clean | 3.11.13 | 2.7.1 | cu126 | 525+ | ✅ matched (7.5e-05 eV/atom (float32-high)) | — |
| pet | clean | 3.11.14 | 2.9.1 | cu128 | 570+ | ✅ matched (9.2e-07 eV/atom) | 2.8GB model |
| sevennet | clean | 3.11.13 | 2.7.1 | cu126 | 525+ | ✅ matched (9.2e-07 eV/atom) | — |
| tace | candidate | 3.11.15 | 2.11.0 | cu130 | 580+ (CUDA 13) | ✅ matched (CPU) (1.9e-06 eV/atom) | C/CUDA ext compiles; needs CUDA-13 driver for GPU |
| uma | clean | 3.11.13 | 2.8.0 | cu128 | 570+ | ✅ matched (1.6e-07 eV/atom) | gated weights (HF token); UMA proven via HF_TOKEN_PATH |
| allegro | candidate | 3.11.13 | 2.8.0 | cu128 | 570+ | ✅ matched (2.9e-07 eV/atom via AOT .pt2) | needs cueq-ops kernels + a per-arch .pt2 (nequip-compile on the user GPU) |
| alphanet | candidate | 3.11.13 | 2.1.2 | cu121 | 525+ | ⚠ version drift (public HEAD ≠ /TGM) | owner must pin the exact commit (gas energies drift 0.24 eV/atom) |
| equflash | candidate | 3.12.13 | 2.9.1 | cu126 | 525+ | ✅ matched (9.8e-07 eV/atom, multi-pass) | needs a 2-pass install (fairchem --no-deps) + nvalchemi-toolkit-ops; install.sh multi-pass pending |
| matris | candidate | 3.11.15 | 2.12.1 | cu130 | 580+ (CUDA 13) | not in /TGM (no ref) | builds+runs (CPU here; GPU needs CUDA-13 driver); pkg auto-dl can write 0-byte on some nets |
| nequip | candidate | 3.11.13 | 2.9.1 | cu128 | 570+ | ✅ matched (3.8e-07 eV/atom via AOT .pt2) | oeq JIT needs ninja + nvrtc.h on CPATH; + a per-arch .pt2 (nequip-compile) |
| nequix | candidate | 3.10.20 | 2.10.0 | cu126 | 525+ | ✅ builds+imports (no ref) | needs ninja + nvrtc.h on CPATH; extjax(JAX) accel build still fails (optional); not in /TGM |

## The two host-floor limits (found empirically)

1. **`torch +cu130` needs a CUDA-13-class driver** (≈ 580+). On a CUDA-12.8 box
   (driver 576) `torch.cuda.is_available()` is False → those envs (**dpa4, tace,
   matris**) run **CPU-only**. CPU still reproduces /TGM (float64), just slower.
2. **`openequivariance` needs torch ≥ 2.10** for its precompiled extension, else it
   JIT-compiles against nvcc — which can fail. Affects **nequip** (won't load) and
   **nequix** (wheel build fails). Allegro avoids it (uses cuequivariance) but still
   needs an AOT-compiled `.pt2`.

## How to read this for your host
- Check your driver: `nvidia-smi` (top-right CUDA version) vs the **driver floor** column.
- A `clean` row with a met floor → `install.sh <env>` then run; weights fetch on first use.
- A `cu130` row on a CUDA-12 host → works on **CPU** (set `device='cpu'`), or upgrade the driver.
- Gated rows (fairchemv1/eSEN, uma) → set up an HF token (see `docs/hf_token.md`).
- `⏳`/`❌`/`⚠` rows → see the per-env `# candidate-reason:` in `envs/<env>.yml` and
  `docs/equiv_results.md` for the exact blocker and the path to resolve it.

