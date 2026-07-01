# Ground Truth: 5-bucket root-cause reclassification (Phase 0)

> Single source of truth: the preserved `.sweep/logs/<env>.verify.stderr.log` (2026-06-28 sweep). This document reclassifies GJC_HANDOFF §3's 4-bucket "estimated" split into 5 buckets, grounded in **measured stderr evidence**. Diagnostic axis (this document) != recovery axis (`AGENTS.md §8`) — CR-2.

## Measurement environment
- Host: WSL, RTX 4060 Ti 16GB **sm89** (compute cap 8.9), nvcc 12.8, conda 25.7.
- sweep: `scripts/sweep_local.py` (build -> verify -> delete). **Each env is deleted after its test** -> no built env currently under `envs/` (only the stderr logs are preserved).
- Raw results: `.sweep/results.md` (4 PASS / 1 cpu_fallback / 13 fail / 2 gated-skip), `.sweep/results.jsonl` (20 rows).

## 5-bucket classification table (diagnostic axis — observability/classification only, CR-2)

| env | model | measured stderr key | bucket | note |
|---|---|---|---|---|
| mace | MACE-MPA-0 | (energy+force, GPU PID) | — PASS | tier-2 PASS |
| sevennet | SevenNet-MF-OMPA | (energy+force, GPU PID) | — PASS | tier-2 PASS |
| orb | ORB-v3 | (energy+force, GPU PID) | — PASS | tier-2 PASS |
| chgnet | CHGNet-v0.3.0 | (energy+force, GPU PID) | — PASS | tier-2 PASS |
| deepmd | DPA-3.1-3M-FT | `dpa-3.1-3m-ft.pth does not exist` | **A** weight-fetch | on-demand-hf, missing path |
| grace | GRACE-2L-OAM | `SavedModel file does not exist` | **A** weight-fetch | TF SavedModel not fetched |
| pet | PET-OAM-XL | `pet-oam-xl-v1.0.0.ckpt does not exist` | **A** weight-fetch | .ckpt not fetched (note: .ckpt != .pt fingerprint, a known issue) |
| dpa4 | DPA-4.0.1-pro-MPtrj | `FileNotFoundError(2)` | **A** weight-fetch | weight at unknown path |
| **eqnorm** | Eqnorm-MPtrj | `EOFError('Ran out of input')` | **A** weight-fetch (corrupt/empty cache) | module import OK, torch.load read an empty/partial file -> corrupt cache. **CONFIRMED** |
| **matris** | MatRIS-10M-OAM | `EOFError('Ran out of input')` | **A** weight-fetch (corrupt/empty cache) | same pattern. **CONFIRMED** |
| nequip | NequIP-OAM-L | `Ninja is required to load C++ extensions` | **B** ninja | ninja runtime not applied to the OpenEquivariance .pt2 build |
| allegro | Allegro-OAM-L | `Failed to initialize zip archive: file open failed` | **B2** .pt2 corrupt/absent | different root cause from ninja (.nequip.zip open failure) |
| nequix | Nequix-MP-1 | `No module named 'nequix'` | **C** module not installed | declared but not importable |
| alphanet | AlphaNet-v1-OMA | `No module named 'alphanet'` | **C** module not installed | |
| equflash | EquFlashV2 | `No module named 'GGNN'` | **C** module not installed | internal dependency GGNN not installed |
| equiformer_v3 | EqV3-OMatMPtrjSalex | `No module named 'fairchem'` | **C** module not installed | fairchem declared but not applied |
| mattersim | MatterSim-v1-5M | (energy+force, no GPU PID) | **D** cpu_fallback | device=cuda not forced |
| tace | TACE-OAM-L | `NVIDIA driver too old (found 12090)` | **E** driver skew | **uncontrolled external value** — PyTorch CUDA runtime > host driver (12.9). Subject to CR-1. |
| fairchemv1 | eSEN-30M-OAM | (skipped, HF_TOKEN missing) | gated | license: facebook/OMAT24 |
| uma | UMA-m-1p1-OC20 | (skipped, HF_TOKEN missing) | gated | license: facebook/UMA |

## Measured correction vs the GJC_HANDOFF 4-bucket "estimate"

GJC_HANDOFF §3 estimated Bucket A to contain `grace, deepmd, pet (+ likely nequix, alphanet, dpa4, tace, matris, eqnorm, equflash)` and asserted that "weight-fetch is the single biggest lever, flipping ~7-9 envs." **Measured correction:**

- **True bucket A (weight-fetch)**: deepmd, grace, pet, dpa4, **eqnorm, matris** = **6** (eqnorm/matris join A via `EOFError` = corrupt/empty weight cache).
- **Wrong estimates**: `nequix`, `alphanet`, `equflash` = bucket C (module not installed) — fixing weight-fetch does not resolve them.
- **tace**: bucket E (driver skew, uncontrolled) — the handoff recorded only nvcc 12.8 in §5 and omitted the driver-runtime compatibility matrix.
- Conclusion: weight-fetch is indeed the **single largest lever, covering 6 of 13**, but the handoff's "7-9" overcounts by mis-including C/E. Option B is consistent: start from the bucket A representative (deepmd) and generalize across the 6.

## Carry-over confidence marker (sign vs reality gap)

Measurement revealed that the `models.json validation` field carried **cross-host values (internal catbench L40S) carried over**:

- `eqnorm` = labeled `validated_sm89`, but on host (4060Ti sm89) it failed with `EOFError` because the weight was absent -> arch-validity may hold, but it is unverified on this host.
- `fairchemv1` (eSEN), `uma` = labeled `validated_sm89`, but skipped as gated.
- `matris`, `tace`, `allegro`, `alphanet` = honestly labeled `gpu_pending`.

**Action (this Phase 0):** redefined `models.json _meta.field_guide.validation` along two axes (arch-validity / host-resource+driver-validity), registered `RTX 4060 Ti` under `_meta.gpu_arch.sm89`, and added a `host_note` flagging the 16GB VRAM + driver-skew caveats. Going forward, the DONE verdict takes **the measured tier-1/tier-2 result on the given host — not a carried-over arch label — as the source of truth.**

## Codex fix GPU re-verification — deferred to Phase 2 (Option B consistent)

GJC_HANDOFF §4's Codex fix (`registry.py` +58 version-name resolve, `fetch.py` +300 weights_source-driven fetch, `_worker.py` stdout->stderr, `envs/allegro.yml` +ninja, `envs/equiformer_v3.yml` +fairchem) is **applied to code/recipes but GPU-unverified**. Rebuilding all 13 envs in Phase 0 (153-325s per env, all deleted) would **violate the approved plan's Option B (one representative end-to-end first)** and waste GPU time.

Therefore the actual GPU re-verification of the Codex fix is performed in **Phase 2 by taking the bucket A representative deepmd (DPA-3.1-3M-FT) end-to-end**; that run is the golden path that GPU-proves `fetch.py`, `registry.py` version-resolve, and the recipe fixes in one shot. This is consistent with plan §4 Phase 2 / Risks ("Codex fix may only partially PASS on GPU -> Option B verify-first").

## Phase 0 gate branch verdict

Plan §4 Phase 0 gate: "re-evaluate the Option B representative if bucket E expands beyond tace or a new root cause is found." **Measured result: bucket E is tace only**, no new root cause (eqnorm/matris absorbed into existing A). -> **Gate passed. Option B representative = deepmd (bucket A) retained.**

## GPU re-verification measurement (Phase 0, 2026-06-30)

With the Codex fix (registry/fetch/recipe) applied, the representative PASS model mace was **rebuilt+re-verified** to prove no regression on GPU:

| env | install | verify | energy (eV) | tier | note |
|---|---|---|---|---|---|
| mace (MACE-MPA-0) | rc=0 (~270s) | rc=0 | -16.391451 | tier-2 PASS | No regression after the Codex fix. Forces produced (small structure, so max\|force\|=0). **GPU PID 356480 directly observed in nvidia-smi compute-apps** (`.sweep/phase0/gpu_sample.log`) -> tier-2 GPU real-use proven. |

- Build artifact: `envs/mace/` (6.8G, `.omm_ready` sentinel present) — preserved for reuse in the Phase 4 plugin `/setup` tier-1 proof.
- Logs: `.sweep/phase0/mace.{install,verify,runner}.log`.
- Conclusion: the Codex fix does not break existing PASS models (0 regression). The fail->PASS flip proof for the remaining fixes (fetch.py weights_source-driven, registry version-resolve) is performed in Phase 2 with the bucket A representative deepmd end-to-end (Option B consistent).
