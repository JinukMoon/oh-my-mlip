# Equivalence results — per-MLIP proof ledger

Per-model evidence that an env rebuilt here from the recipe reproduces the
validated /TGM hub. Tiers (see `docs/equiv_protocol.md`): **T1** weight bytes ·
**T2a** single-point energy (cross-GPU OK) · **T2b** catbench-relax (same-GPU) ·
**T3** time (same-GPU). `validated` ≡ T1 + T2b + T3.

The ladder is run efficiently: (0) the recipe builds + the model imports + a
calculation runs → (1) the same structure gives the same energy as /TGM → (2)
catbench. A model only advances when the cheaper rung passes.

## MACE (mace-mpa-0)

| Rung | Result |
|---|---|
| **0. build + import + calc** | ✅ `install.sh mace` builds clean on WSL (RTX 4060 Ti, CUDA 12.6): torch 2.7.1+cu126 CUDA-available, mace 0.3.15, e3nn 0.4.4 (the /TGM-authoritative pin), catbench 1.1.2, D3 first-run compile OK. `mace_mp` imports and computes. |
| **T1. weight bytes** | ✅ downloaded `mace_mp(model="medium-mpa-0")` == /TGM `mace-mpa-0.model` BYTE-IDENTICAL (sha256 `75428afe…604fb638`, 79462305 B). |
| **T2a. single-point energy (cross-GPU)** | ✅ **PASS** — 6 fixed BackSingle2018 structures, identical `mace_mp(dispersion=False, default_dtype='float64', device='cuda')` on both sides. Our RTX 4060 Ti (sm89) vs /TGM RTX A4500 (sm86). |
| T2b. catbench-relax (same-GPU) | ⏳ pending (D4) |
| T3. time (same-GPU) | ⏳ pending (owner relocation) |

**T2a numbers** (eV):

| structure | ours (4060 Ti) | /TGM (A4500) | \|diff\| |
|---|---|---|---|
| H2O (gas) | -13.784034257636 | -13.784034257636 | 1.8e-15 |
| H2 (gas) | -6.515047362424 | -6.515047362424 | 8.9e-16 |
| C26N4Ni (slab) | -271.602681613699 | -271.602681613699 | **0** |
| C26HN4NiO2 (HO2*) | -284.597784576809 | -284.597784576809 | **0** |
| C26HN4NiO (HO*) | -280.633681719903 | -280.633681719903 | **0** |

**max |diff| = 1.78e-15 eV · max |diff|/atom = 5.9e-16 eV/atom** (T2a tol = 1e-3 eV/atom).
The slab and adslab energies are bit-identical across the two GPUs; the gas
molecules differ only by float64 round-off. The deterministic rebuild reproduces
the validated hub to machine precision.

_Provenance:_ dataset `raw_data/BackSingle2018_adsorption.json` (fetched by
`catbench.adsorption.get_benchmark`, 141 systems); calc construction mirrors the
/TGM `models.json` MACE-MPA-0 inference line exactly. The rebuilt env was deleted
after the run (disk reclaim).

## SevenNet (7net-mf-ompa, modal=mpa)

| Rung | Result |
|---|---|
| **0. build + import + calc** | ✅ `install.sh sevennet` builds clean on WSL (RTX 4060 Ti): torch 2.7.1+cu126 CUDA-available, sevenn 0.12.2.dev0 (git sha `e72eb2c9`, the /TGM pin), e3nn 0.5.6, cuequivariance 0.8.0. `SevenNetCalculator` imports and computes; checkpoint `7net-mf-ompa` auto-downloads. |
| **T2a. single-point energy (cross-GPU)** | ✅ **PASS** — 6 fixed BackSingle2018 structures, identical `SevenNetCalculator('7net-mf-ompa', modal='mpa', enable_oeq=False)` on both sides (plain path; oeq is a speed accel, not a different model). Our RTX 4060 Ti vs /TGM A4500. |
| T2b / T3 | ⏳ pending |

**max |diff| = 3.05e-05 eV · max |diff|/atom = 9.2e-07 eV/atom** (tol 1e-3 eV/atom).
Most structures bit-identical; the two adslabs differ by 3e-5 eV — float32 round-off
(SevenNet's default dtype is float32, vs MACE's float64), well within tolerance.
Our deterministically-rebuilt SevenNet env reproduces the validated hub. Env deleted after.

## Method (reproducible)
- Our side: `install.sh mace` → `envs/mace/bin/python` runs the single-point on the
  fixed structures with `mace_mp(model="medium-mpa-0", dispersion=False,
  default_dtype="float64", device="cuda")`.
- /TGM side: the same script in `/TGM/Apps/MLIP/conda/envs/mace` with
  `model=/TGM/Apps/MLIP/models/mace/mace-mpa-0.model`, run via a plain `ssh 147`
  command (no Claude Code on 147; token-free).
- Cross-GPU is valid for T2a because a single-point is a pure function of the fixed
  structure (no chaotic relaxation). T2b/T3 require the same GPU and are deferred.
