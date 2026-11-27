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
- **sm120** — the state on a Blackwell GPU (compute capability 12.0), measured
  there on its own: `validated` (`scripts/setup_verify.py` passed on such a GPU,
  with the recipe `install.sh` builds there), `candidate` (it passed, but
  `envs/_arch_recipes.json` marks that recipe a candidate, e.g. GRACE on a
  TensorFlow nightly that PyPI may not keep; read its reason before relying on it),
  `failed`, `unsupported` (no build is known to work; `install.sh` stops before
  installing), or `not checked`.
  Older GPUs use the base recipes `envs/<env>.yml`; on sm120 some envs are built
  from a sibling recipe such as `envs/mace.cu128.yml`, because the base CUDA
  wheels carry no sm120 kernels. `envs/_arch_recipes.json` lists which recipe
  each env uses there; see [Host requirements](host_requirements.md).
- **Code** — the framework's upstream repository.

<!-- STATUS_TABLE_DETAILED_START -->
| Model | Framework | Weights | Gated | sm120 | Code |
|---|---|---|---|---|---|
| SevenNet-MF-OMPA | SevenNet | auto-download | no | validated | [GitHub](https://github.com/MDIL-SNU/SevenNet){ .md-button .omm-repo } |
| SevenNet-Omni | SevenNet | auto-download | no | validated | [GitHub](https://github.com/MDIL-SNU/SevenNet){ .md-button .omm-repo } |
| MACE-MPA-0 | MACE | auto-download | no | validated | [GitHub](https://github.com/ACEsuit/mace){ .md-button .omm-repo } |
| MACE-MH-1-OMAT | MACE | auto-download | no | validated | [GitHub](https://github.com/ACEsuit/mace){ .md-button .omm-repo } |
| MACE-MH-1-OC20 | MACE | auto-download | no | validated | [GitHub](https://github.com/ACEsuit/mace){ .md-button .omm-repo } |
| NequIP-OAM-XL | NequIP | on-demand-hf | no | validated | [GitHub](https://github.com/mir-group/nequip){ .md-button .omm-repo } |
| NequIP-OAM-L | NequIP | on-demand-hf | no | validated | [GitHub](https://github.com/mir-group/nequip){ .md-button .omm-repo } |
| Allegro-OAM-L | Allegro | on-demand-hf | no | validated | [GitHub](https://github.com/mir-group/allegro){ .md-button .omm-repo } |
| Nequix-MP-1 | Nequix | on-demand-hf | no | validated | [GitHub](https://github.com/atomicarchitects/nequix){ .md-button .omm-repo } |
| DPA-3.1-3M-FT | DeePMD | on-demand-hf | no | candidate | [GitHub](https://github.com/deepmodeling/deepmd-kit){ .md-button .omm-repo } |
| ORB-v3 | ORB | auto-download | no | validated | [GitHub](https://github.com/orbital-materials/orb-models){ .md-button .omm-repo } |
| GRACE-2L-OAM | GRACE | on-demand-hf | no | candidate | [GitHub](https://github.com/ICAMS/grace-tensorpotential){ .md-button .omm-repo } |
| MatterSim-v1-5M | MatterSim | auto-download | no | validated | [GitHub](https://github.com/microsoft/mattersim){ .md-button .omm-repo } |
| CHGNet-v0.3.0 | CHGNet | bundled | no | validated | [GitHub](https://github.com/CederGroupHub/chgnet){ .md-button .omm-repo } |
| AlphaNet-v1-OMA | AlphaNet | on-demand-hf | no | validated | [GitHub](https://github.com/zmyybc/AlphaNet){ .md-button .omm-repo } |
| Eqnorm-MPtrj | Eqnorm | auto-download | no | validated | [GitHub](https://github.com/yzchen08/eqnorm){ .md-button .omm-repo } |
| eSEN-30M-OAM | fairchemv1 | on-demand-hf | yes | validated | [GitHub](https://github.com/facebookresearch/fairchem){ .md-button .omm-repo } |
| EqV3-OMatMPtrjSalex | EquiformerV3 | on-demand-hf | no | candidate | [GitHub](https://github.com/atomicarchitects/equiformer_v3){ .md-button .omm-repo } |
| UMA-m-1p1-OC20 | UMA | on-demand-hf | yes | candidate | [GitHub](https://github.com/facebookresearch/fairchem){ .md-button .omm-repo } |
| UMA-m-1p1-OMAT | UMA | on-demand-hf | yes | candidate | [GitHub](https://github.com/facebookresearch/fairchem){ .md-button .omm-repo } |
| UMA-s-1p1-OC20 | UMA | on-demand-hf | yes | candidate | [GitHub](https://github.com/facebookresearch/fairchem){ .md-button .omm-repo } |
| UMA-s-1p1-OMAT | UMA | on-demand-hf | yes | candidate | [GitHub](https://github.com/facebookresearch/fairchem){ .md-button .omm-repo } |
| UMA-s-1p2-OC20 | UMA | on-demand-hf | yes | candidate | [GitHub](https://github.com/facebookresearch/fairchem){ .md-button .omm-repo } |
| UMA-s-1p2-OC22 | UMA | on-demand-hf | yes | candidate | [GitHub](https://github.com/facebookresearch/fairchem){ .md-button .omm-repo } |
| UMA-s-1p2-OC25 | UMA | on-demand-hf | yes | candidate | [GitHub](https://github.com/facebookresearch/fairchem){ .md-button .omm-repo } |
| UMA-s-1p2-OMAT | UMA | on-demand-hf | yes | candidate | [GitHub](https://github.com/facebookresearch/fairchem){ .md-button .omm-repo } |
| PET-OAM-XL | PET | on-demand-hf | no | candidate | [GitHub](https://github.com/metatensor/metatrain){ .md-button .omm-repo } |
| EquFlashV2 | EquFlash | on-demand-hf | no | validated | [GitHub](https://github.com/SamsungDS/GGNN){ .md-button .omm-repo } |
| EquFlash-v1 | EquFlash | on-demand-hf | no | validated | [GitHub](https://github.com/SamsungDS/GGNN){ .md-button .omm-repo } |
| MatRIS-10M-OAM | MatRIS | auto-download | no | candidate | [GitHub](https://github.com/HPC-AI-Team/MatRIS){ .md-button .omm-repo } |
| DPA-4.0.1-pro-MPtrj | DPA4 | on-demand-hf | no | candidate | [GitHub](https://github.com/deepmodeling/deepmd-kit){ .md-button .omm-repo } |
| TACE-OAM-L | TACE | auto-download | no | candidate | [GitHub](https://github.com/xvzemin/tace){ .md-button .omm-repo } |
| Prophet-OAME-MBD | Prophet | on-demand-hf | no | unsupported | [GitHub](https://github.com/kairosmaterial/prophet){ .md-button .omm-repo } |
<!-- STATUS_TABLE_DETAILED_END -->
