# Host requirements & equivalence matrix

Each env is pinned to a specific torch/CUDA build, so the **host must meet that
recipe's NVIDIA-driver floor** (and, for a few, a GPU-arch compile). This is the
honest portability story: oh-my-mlip removes the env/calculator/weights pain on a
**compatible host** — it is not magic universal portability. Driver floors are
approximate Linux minimums for the bundled CUDA runtime.

Equivalence = single-point energy vs the validated /TGM hub on BackSingle2018
(per-atom, tol 1e-3 eV/atom). **14/20 envs matched** (most bit-identical), across
PyTorch + TensorFlow, incl. both gated envs. See `docs/equiv_results.md` for the
full per-model numbers.

| env | status | py | torch | CUDA | driver floor | equivalence | host requirement / note |
|---|---|---|---|---|---|---|---|
| chgnet | clean | 3.11.13 | 2.7.1 | cu126 | 525+ | ✅ matched (9.2e-07 eV/atom) | — |
| deepmd | clean | 3.10.19 | 2.8.0 | cu128 | 570+ | ✅ matched (1.5e-07 eV/atom) | needs pip `mpich` |
| dpa4 | candidate | 3.11.15 | 2.11.0 | cu130 | 580+ (CUDA 13) | ✅ matched (CPU) (2.2e-07 eV/atom) | needs CUDA-13 driver for GPU; ran CPU on CUDA-12 box; pip `mpich` |
| eqnorm | clean | 3.11.13 | 2.6.0 | cu118 | 450+ | ✅ matched (0.0 (all bit-identical)) | ⚠ by-name weights fetch broken (stage manually) |
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
| allegro | candidate | 3.11.13 | 2.8.0 | cu128 | 570+ | ⏳ AOT compile (builds+imports OK) | needs AOT-compiled .pt2 (NequIPCalculator.from_compiled_model) |
| alphanet | candidate | 3.11.13 | 2.1.2 | cu121 | 525+ | ⚠ version drift (public HEAD ≠ /TGM) | owner must pin the exact commit (gas energies drift 0.24 eV/atom) |
| equflash | candidate | 3.12.13 | 2.9.1 | cu126 | 525+ | ❌ resolver (env doesn't resolve) | fairchem-core 1.10 ↔ torch 2.9.1 conflict; needs --no-deps recipe |
| matris | candidate | 3.11.15 | 2.12.1 | cu130 | 580+ (CUDA 13) | ⚠ no ref (builds+imports OK) | by-name weights fetch broken; not in /TGM (no reference) |
| nequip | candidate | 3.11.13 | 2.9.1 | cu128 | 570+ | ⏳ oeq+AOT (builds+imports OK) | openequivariance needs torch>=2.10 to load; + AOT .pt2 |
| nequix | candidate | 3.10.20 | 2.10.0 | cu126 | 525+ | ❌ oeq build (openequivariance wheel build fails) | JAX; not in /TGM |

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

