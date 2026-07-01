# Model status — validation, gating, and v1 distribution

This is the detailed companion to the README's
[`## Supported MLIPs`](../README.md#supported-mlips) list. The table below is
**generated from `models.json`** by `scripts/gen_status_table.py` (CI runs
`--check` to keep it byte-for-byte in sync — this doc and the README can never
drift from the registry, which is the guard against an "all validated"
overclaim).

How to read the columns:

- **Validation** is each model's true per-GPU validation state from the
  registry. `validated (sm89)` means a GPU single-point (and D3 where relevant)
  was run and recorded on that architecture; rows marked `gpu pending` are
  env/load-verified only (the env imports and the calculator builds, but the GPU
  single-point and/or D3 has not been verified yet).
- **v1 tarball** marks the frameworks whose relocatable conda-pack distribution
  is authored for v1 — **exactly MACE and SevenNet**, shown as `upload-pending`
  because their tarballs are not yet uploaded (the build+publish and the binding
  foreign-host end-to-end run are the deferred compute checkpoint). Everything
  else is a `Phase 2` packaging target. This is kept distinct from the per-model
  GPU validation state.
- **Weights** is how a model's weights are obtained: `bundled` (inside the
  conda-pack tarball), `auto-download` (fetched by name to a shared cache on
  first run), or `on-demand-hf` (fetched from Hugging Face Hub, may be gated).
- **Gated** flags weights that sit behind an upstream license you must accept
  with your own Hugging Face token — see
  [`gated_models.md`](gated_models.md). Licenses for every framework are in
  [`model_licenses.md`](model_licenses.md).

<!-- STATUS_TABLE_DETAILED_START -->
| Model | Framework | Weights | Validation | Gated | v1 tarball |
|---|---|---|---|---|---|
| SevenNet-MF-OMPA | SevenNet | bundled | validated (sm89) | no | upload-pending |
| SevenNet-Omni | SevenNet | bundled | validated (sm89) | no | upload-pending |
| MACE-MPA-0 | MACE | auto-download | validated (sm89) | no | upload-pending |
| MACE-MH-1-OMAT | MACE | auto-download | validated (sm89) | no | upload-pending |
| MACE-MH-1-OC20 | MACE | auto-download | validated (sm89) | no | upload-pending |
| NequIP-OAM-XL | NequIP | on-demand-hf | validated (sm89) | no | Phase 2 |
| NequIP-OAM-L | NequIP | on-demand-hf | validated (sm89) | no | Phase 2 |
| Allegro-OAM-L | Allegro | on-demand-hf | validated (sm89) | no | Phase 2 |
| Nequix-MP-1 | Nequix | on-demand-hf | validated (sm89) | no | Phase 2 |
| DPA-3.1-3M-FT | DeePMD | on-demand-hf | validated (sm89) | no | Phase 2 |
| ORB-v3 | ORB | auto-download | validated (sm89) | no | Phase 2 |
| GRACE-2L-OAM | GRACE | on-demand-hf | validated (sm89) | no | Phase 2 |
| MatterSim-v1-5M | MatterSim | bundled | validated (sm89) | no | Phase 2 |
| CHGNet-v0.3.0 | CHGNet | bundled | validated (sm89) | no | Phase 2 |
| AlphaNet-v1-OMA | AlphaNet | on-demand-hf | validated (sm89) | no | Phase 2 |
| Eqnorm-MPtrj | Eqnorm | auto-download | validated (sm89) | no | Phase 2 |
| eSEN-30M-OAM | fairchemv1 | on-demand-hf | validated (sm89) | yes | Phase 2 |
| EqV3-OMatMPtrjSalex | EquiformerV3 | on-demand-hf | validated (sm89) | no | Phase 2 |
| UMA-m-1p1-OC20 | UMA | on-demand-hf | validated (sm89) | yes | Phase 2 |
| UMA-m-1p1-OMAT | UMA | on-demand-hf | validated (sm89) | yes | Phase 2 |
| UMA-s-1p1-OC20 | UMA | on-demand-hf | validated (sm89) | yes | Phase 2 |
| UMA-s-1p1-OMAT | UMA | on-demand-hf | validated (sm89) | yes | Phase 2 |
| UMA-s-1p2-OC20 | UMA | on-demand-hf | validated (sm89) | yes | Phase 2 |
| UMA-s-1p2-OC22 | UMA | on-demand-hf | validated (sm89) | yes | Phase 2 |
| UMA-s-1p2-OMAT | UMA | on-demand-hf | validated (sm89) | yes | Phase 2 |
| PET-OAM-XL | PET | on-demand-hf | validated (sm89) | no | Phase 2 |
| EquFlashV2 | EquFlash | on-demand-hf | validated (sm89) | no | Phase 2 |
| EquFlash | EquFlash | on-demand-hf | validated (sm89) | no | Phase 2 |
| MatRIS-10M-OAM | MatRIS | auto-download | tier-1 CPU (driver skew) | no | Phase 2 |
| DPA-4.0.1-pro-MPtrj | DPA4 | on-demand-hf | tier-1 CPU (driver skew) | no | Phase 2 |
| TACE-OAM-L | TACE | auto-download | tier-1 CPU (driver skew) | no | Phase 2 |
<!-- STATUS_TABLE_DETAILED_END -->
