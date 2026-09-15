# Supported models

Every model variant in the registry, one row per version. The table is
**generated from `models.json`** by `scripts/gen_status_table.py`, and CI keeps
it in sync with the registry.

How to read the columns:

- **Weights** — how a model's weights are obtained: `bundled` (inside the
  framework package), `auto-download` (fetched by name on first run), or
  `on-demand-hf` (fetched from the Hugging Face Hub, possibly gated).
- **Gated** — the weights sit behind an upstream license you accept with your
  own Hugging Face account; see [Gated models](gated_models.md). Licenses for
  every framework are in [Model licenses](model_licenses.md).
- **v1 tarball** — `published (<rev>)` means a prebuilt, relocatable env is
  available from the Hugging Face Hub with a pinned revision and sha256;
  `upload-pending` and `Phase 2` mean the env is built from its recipe
  (`./install.sh <model>`).

<!-- STATUS_TABLE_DETAILED_START -->
| Model | Framework | Weights | Gated | v1 tarball |
|---|---|---|---|---|
| SevenNet-MF-OMPA | SevenNet | bundled | no | published (v1) |
| SevenNet-Omni | SevenNet | bundled | no | published (v1) |
| MACE-MPA-0 | MACE | auto-download | no | published (v1) |
| MACE-MH-1-OMAT | MACE | auto-download | no | published (v1) |
| MACE-MH-1-OC20 | MACE | auto-download | no | published (v1) |
| NequIP-OAM-XL | NequIP | on-demand-hf | no | Phase 2 |
| NequIP-OAM-L | NequIP | on-demand-hf | no | Phase 2 |
| Allegro-OAM-L | Allegro | on-demand-hf | no | Phase 2 |
| Nequix-MP-1 | Nequix | on-demand-hf | no | Phase 2 |
| DPA-3.1-3M-FT | DeePMD | on-demand-hf | no | Phase 2 |
| ORB-v3 | ORB | auto-download | no | Phase 2 |
| GRACE-2L-OAM | GRACE | on-demand-hf | no | Phase 2 |
| MatterSim-v1-5M | MatterSim | bundled | no | Phase 2 |
| CHGNet-v0.3.0 | CHGNet | bundled | no | Phase 2 |
| AlphaNet-v1-OMA | AlphaNet | on-demand-hf | no | Phase 2 |
| Eqnorm-MPtrj | Eqnorm | auto-download | no | Phase 2 |
| eSEN-30M-OAM | fairchemv1 | on-demand-hf | yes | Phase 2 |
| EqV3-OMatMPtrjSalex | EquiformerV3 | on-demand-hf | no | Phase 2 |
| UMA-m-1p1-OC20 | UMA | on-demand-hf | yes | Phase 2 |
| UMA-m-1p1-OMAT | UMA | on-demand-hf | yes | Phase 2 |
| UMA-s-1p1-OC20 | UMA | on-demand-hf | yes | Phase 2 |
| UMA-s-1p1-OMAT | UMA | on-demand-hf | yes | Phase 2 |
| UMA-s-1p2-OC20 | UMA | on-demand-hf | yes | Phase 2 |
| UMA-s-1p2-OC22 | UMA | on-demand-hf | yes | Phase 2 |
| UMA-s-1p2-OC25 | UMA | on-demand-hf | yes | Phase 2 |
| UMA-s-1p2-OMAT | UMA | on-demand-hf | yes | Phase 2 |
| PET-OAM-XL | PET | on-demand-hf | no | Phase 2 |
| EquFlashV2 | EquFlash | on-demand-hf | no | Phase 2 |
| EquFlash | EquFlash | on-demand-hf | no | Phase 2 |
| MatRIS-10M-OAM | MatRIS | auto-download | no | Phase 2 |
| DPA-4.0.1-pro-MPtrj | DPA4 | on-demand-hf | no | Phase 2 |
| TACE-OAM-L | TACE | auto-download | no | Phase 2 |
<!-- STATUS_TABLE_DETAILED_END -->
