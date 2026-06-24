# Equivalence validation protocol

This document defines what `validated` means for a model in oh-my-mlip and how
the in-repo harness (`scripts/run_equiv.py`) and comparator
(`scripts/compare_equiv.py`) prove it. The bar is **clean-host rebuild +
reference-result auto-check**: an env rebuilt here from the recipe must reproduce
the validated reference both byte-for-byte (weights) and result-for-result
(energies, geometries, timing).

## The 4-tier model

A capped relaxation (`n_crit_relax=5`) is a *dynamical* workflow, not a pure
function of fixed structures: tiny force differences perturb the optimizer steps
and can drive the short trajectory to a different geometry/energy. Therefore the
owner's "results match" for a relaxed run requires the **same GPU**. Only
fixed-geometry single-points are a pure function of the structure and are thus
cross-GPU tolerant. This splits the proof into four tiers:

| Tier | Claim | Needs | Cross-GPU? |
|---|---|---|---|
| **T1 weight-integrity** | downloaded checkpoint == reference checkpoint | sha256 both sides | N/A (bytes) |
| **T2a single-point** | fixed-geometry energy + forces agree | run BOTH on the SAME fixed structures; compare within tolerance | yes (pure fn of structure) |
| **T2b catbench-relax** | relaxed adsorption energies + anomaly + terminal geometries match | BOTH on the SAME GPU; compare energies AND end-geometries | no (same hardware) |
| **T3 time-match** | mean time_per_step matches | BOTH on the SAME GPU | no (same hardware) |

### Definition of `validated` (honesty contract)

A model is **`validated` only when T1 + T2b + T3 all pass.**

- **T2a is a cross-GPU numerical sanity check** — necessary, not sufficient.
- **T1 alone is a byte-match** and is never reported as a results-match.
- **T2b and T3 are ONE same-GPU dual-run**: relocating the reference env once
  yields both. The real split is "T2a now (cross-GPU single-point), T2b+T3
  together after the env relocation," NOT "results now, time later."

Every equivalence claim must name its tier. T2a and T1 are explicitly sub-proofs
and are never dressed up as the full `validated` bar.

## Reference-provenance schema

Every reference number must carry a complete provenance block. A reference
missing any field is rejected by the comparator. The harness stamps:

| Field | Meaning |
|---|---|
| `mode` | `single-point` or `relax` |
| `model`, `version` | framework + resolved version key |
| `tag` | benchmark tag (`raw_data/<tag>_adsorption.json`) |
| `dataset_sha256` | sha256 of the raw dataset file bytes |
| `n_systems` | empirical reaction-system count (counted, never assumed) |
| `catbench_version` | catbench package version |
| `python_version` | interpreter version |
| `backend` | `torch` / `jax` / `none` |
| `torch_version`, `jax_version` | backend versions (the unused one is null) |
| `cuda_version` | CUDA toolkit version (torch.version.cuda) |
| `gpu_name` | `torch.cuda.get_device_name(0)` or null |
| `driver` | NVIDIA driver (best-effort via nvidia-smi, else null) |
| `d3` | D3 dispersion on/off (bool) |
| `weights_sha256` | sha256 of the local checkpoint if resolvable, else null |
| `command` | full `sys.argv` of the run |
| `timestamp` | ISO string supplied by the caller (the harness never calls a clock) |

The `timestamp` is passed in via `--timestamp` so the caller — not the harness —
stamps the time; this keeps the run output deterministic for the same inputs.

## Tolerances and their justification

The comparator's defaults, all overridable on the CLI:

- **T2a single-point energy — `--tol` 1e-3 eV/atom** (max-abs-diff per atom).
  Cross-GPU floating-point differences for MACE single-points are typically in
  the 1e-4 to 1e-3 eV/atom range, so 1e-3 eV/atom is a defensible upper bound
  that catches a genuinely different numerical path while tolerating
  hardware-level fp scatter. Tunable per backend.
- **T2a forces — `--force-tol` 1e-2 eV/Angstrom** (max-abs-diff on per-structure
  fmax). Forces are a derivative and carry more fp noise than energies; 1e-2
  eV/Angstrom is the matching cross-GPU band.
- **T2b terminal-geometry — `--geom-tol` 1e-2 Angstrom** RMSD over matched
  atoms. Same-GPU, this captures "the optimizer reached the same local minimum"
  rather than a bitwise-identical trajectory. Geometry is compared by coordinate
  RMSD, **never** by a hash of float positions (a SHA256 of floats never matches
  across fp noise, even same-GPU). The cheap first-line check is the
  `slab_max_disp` / `adslab_max_disp` scalars catbench already records; the full
  check is the coordinate RMSD.
- **T3 time — `--time-band` 0.8,1.25** ratio on mean `time_per_step`. Compared
  only when both provenance `gpu_name` values are equal and non-null. Use **>= 3
  interleaved warm runs**; record mean and variance; **exclude the first run**
  (it pays a one-time JIT-compile and/or weight-download cost — see below).

## D3 and JIT-warm note

T2a and T2b must use **identical D3 on/off on both sides** (the provenance `d3`
field enforces this; the comparator hard-fails on a mismatch). The compiled
torch-dftd D3 path pays a one-time JIT cost on its first call, which can add a
spurious per-step difference. **Warm the calculator** (run one throwaway
structure) before timing so the JIT-compile and any weight download are excluded
from the T3 measurement.

## catbench-version-skew consequence path

The comparator **hard-fails** on a `catbench_version` provenance mismatch — there
is no silent compare across result formats. The consequence path:

1. Re-pin the reference-side run to `catbench==1.1.2` and re-run, so the
   comparison is apples-to-apples.
2. If the reference side cannot be re-pinned, record that model's T2a/T2b as
   **"version-skew — not comparable"**. It is **never** recorded as a pass.

## catbench has no single-point API

`catbench.adsorption.AdsorptionCalculation` **always relaxes**
(`n_crit_relax` → ASE optimizer `opt.run`). There is no single-point entry point.
So `run_equiv.py --mode single-point` (T2a) is a **custom per-structure loop**:
for each reaction system, for each structure's ASE `Atoms`, attach the
calculator, take `get_potential_energy()` and `abs(get_forces()).max()`, with
**no optimizer**. Structures (including gas references) are loaded with
`catbench.utils.data_utils.load_catbench_json`, which rehydrates the deduplicated
`_structures` map into real ASE `Atoms` at
`d[reaction_key]["raw"][structure_key]["atoms"]`. The top-level `_structures` key
is skipped when iterating reaction systems.

## Where terminal geometries come from

Terminal geometries are **not** in catbench's result dict. `result["final"]`
holds energies, timing, and the scalar displacement proxies `slab_max_disp` /
`adslab_max_disp` — but no coordinates. Terminal coordinates are written to disk
only as `traj/<key>/<structure>_<i>` extxyz files, and only when catbench runs
with `save_files=True`. So T2b geometry capture = run `--mode relax` (which sets
`save_files=True`), read each `traj/<key>` final frame, and emit per-system
positions + cell. The harness rounds positions to 6 decimal places before
emitting them.

## Stage map: where each tier runs

| Stage | Tier | Where it runs |
|---|---|---|
| Rebuild env from recipe + T1 fingerprint | T1 | WSL-local (`install.sh`, `verify_weights_integrity.py`) |
| `run_equiv.py --mode single-point` (our side) | T2a | WSL-local GPU |
| Reference single-point | T2a | reference host via reviewed, token-free `ssh 147` (cross-GPU is fine) |
| `compare_equiv.py` (T2a) | T2a | WSL-local |
| `run_equiv.py --mode relax`, both envs, same GPU | T2b + T3 | same-GPU owner-action (requires env relocation) |
| `compare_equiv.py` (T2b + T3) | T2b + T3 | WSL-local |

Claude Code never runs on the reference compute hosts; the reference side is
reached only by reviewed, checked-in, token-free scripts run via plain `ssh`, or
by owner-issued commands. Because T2a single-point is cross-GPU tolerant and the
dataset is a public download, the two sides can rely on
**identical-by-`dataset_sha256`** rather than transferring the dataset.

## Comparator hard-fail summary

`compare_equiv.py OURS.json REF.json` exits non-zero on any of:

- provenance mismatch on `catbench_version`, `dataset_sha256`, `model`,
  `weights_sha256` (only when both non-null), or `d3`;
- missing or extra system/structure keys;
- single-point: `|energy_ours - energy_ref| / natoms > --tol`, or
  `|fmax_ours - fmax_ref| > --force-tol`;
- relax: per-system relaxed-adsorption-energy diff > `--tol`, or terminal
  coordinate RMSD > `--geom-tol` (terminal geometry absent on either side is a
  FAIL);
- time: when both `gpu_name` are equal and non-null, a `time_per_step` ratio
  outside `--time-band`. When the GPUs differ or are null, time is **SKIPPED**
  and never fails the run.

The output is a per-check PASS/FAIL table ending in `EQUIV: PASS` or
`EQUIV: FAIL (<reasons>)`.
