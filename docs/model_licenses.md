# Model & framework licenses

`oh-my-mlip` is an **MIT-licensed installer / orchestrator**. It does **not**
redistribute any framework's source code — every framework is installed from its
own official channel (PyPI / GitHub / Hugging Face / Zenodo) — and every weight
is downloaded from its upstream host first. The one exception is a fallback
mirror for a few CC-BY-4.0 weights, described [below](#fallback-mirror). **Each one stays under its
own upstream license, which you must comply with** — especially for commercial
use.

This page is a convenience summary of what we found by reading each project's
actual `LICENSE` file (as of 2026-06). It is **not legal advice** and licenses
change — always check the upstream repo before commercial use or redistribution.
Two licenses matter separately:

- **Code license** — the framework's Python package (what you `pip install`).
- **Weights license** — the pretrained checkpoint (often *different* from the
  code, e.g. a permissive code base with CC-BY or non-commercial weights).

## ⚠️ Read this first — the restricted ones

| Restriction | Frameworks (our variants) | What it means |
|---|---|---|
| 🚫 **Non-commercial only** | **GRACE** (`GRACE-2L-OAM`), **MACE-MH** weights (`MACE-MH-1-OMAT`, `MACE-MH-1-OC20`) | Academic / research use only. Commercial use needs a separate license from the owner. |
| ⚖️ **Upstream declarations conflict** | **EquFlash** (`EquFlashV2`, `EquFlash-v1`), **AlphaNet** (`AlphaNet-v1-OMA`) | The record hosting the weights says one license and the project says another (details in the table below). Nothing upstream resolves it; ask the authors before relying on either. EquFlash's code is non-commercial (CC BY-NC-SA 4.0); AlphaNet's code is GPL-3.0. |
| ❔ **Weights license not stated** | **SevenNet** (`SevenNet-MF-OMPA`, `SevenNet-Omni`) | The checkpoint files oh-my-mlip downloads carry no license statement (details below). |
| 🔗 **Weak copyleft (LGPL-3.0)** | **DeePMD** (`DPA-3.1-3M-FT`), **DPA4** (`DPA-4.0.1-pro-MPtrj`) | Using the library is fine; modifications *to the library* must stay LGPL. |
| 🔒 **Gated weights** | **UMA** (all `UMA-*` variants), **eSEN** (`eSEN-30M-OAM`) | Code is MIT, but the weights are under Meta licenses on Hugging Face (FAIR Chemistry License for UMA, OMat24 License for eSEN): accept the license, get access approved, log in with your token. Commercial use is permitted, subject to Meta's Acceptable Use Policy; the FAIR Chemistry License (UMA) also requires compliance with applicable laws, including trade control laws. |
| 📎 **Attribution (CC-BY-4.0) weights** | **NequIP** (`NequIP-OAM-*`), **Allegro** (`Allegro-OAM-L`), **TACE** (`TACE-OAM-L`), **DPA** weights (`DPA-3.1-3M-FT`, `DPA-4.0.1-pro-MPtrj`), **MatRIS** (`MatRIS-10M-OAM`, see below) | Free incl. commercial, but you must credit the authors. |

Everything else below is permissive (MIT / Apache-2.0 / BSD-3-Clause), commercial
use allowed.

## Full table — 20 frameworks / 32 model variants

| Framework | Our variant(s) | GitHub | Code license | Weights license | Gated |
|---|---|---|---|---|---|
| **SevenNet** | SevenNet-MF-OMPA, SevenNet-Omni | [MDIL-SNU/SevenNet](https://github.com/MDIL-SNU/SevenNet) | MIT (since 2025-06-20; GPL-3.0 before) | **Not stated** on the GitHub release assets that are downloaded. The authors' figshare deposit of MF-ompa ([28590722](https://figshare.com/articles/28590722), a different file) says GPL 3.0+. | no |
| **MACE** | MACE-MPA-0, MACE-MH-1-OMAT, MACE-MH-1-OC20 | [ACEsuit/mace](https://github.com/ACEsuit/mace) | MIT | **Mixed**: MPA-0 = MIT; **MH-1 = ASL (non-commercial)** | no |
| **NequIP** | NequIP-OAM-XL, NequIP-OAM-L | [mir-group/nequip](https://github.com/mir-group/nequip) | MIT | CC-BY-4.0 | no |
| **Allegro** | Allegro-OAM-L | [mir-group/allegro](https://github.com/mir-group/allegro) | MIT | CC-BY-4.0 | no |
| **Nequix** | Nequix-MP-1 | [atomicarchitects/nequix](https://github.com/atomicarchitects/nequix) | MIT | MIT (the file is in the MIT-licensed repo); the author's figshare copy of the same file ([29927606](https://figshare.com/articles/29927606)) says CC BY 4.0 | no |
| **DeePMD** | DPA-3.1-3M-FT | [deepmodeling/deepmd-kit](https://github.com/deepmodeling/deepmd-kit) | **LGPL-3.0** | CC-BY-4.0 ([deepmodelingcommunity/DPA-3.1-3M](https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M) on HF) | no |
| **ORB** | ORB-v3 | [orbital-materials/orb-models](https://github.com/orbital-materials/orb-models) | Apache-2.0 | Apache-2.0 (code + weights) | no |
| **GRACE** | GRACE-2L-OAM | [ICAMS/grace-tensorpotential](https://github.com/ICAMS/grace-tensorpotential) | **ASL — non-commercial** | ASL — non-commercial | no |
| **MatterSim** | MatterSim-v1-5M | [microsoft/mattersim](https://github.com/microsoft/mattersim) | MIT | MIT (the 5M checkpoint is downloaded from the GitHub repo on first use) | no¹ |
| **CHGNet** | CHGNet-v0.3.0 | [CederGroupHub/chgnet](https://github.com/CederGroupHub/chgnet) | BSD-3-Clause | BSD-3-Clause | no |
| **AlphaNet** | AlphaNet-v1-OMA | [zmyybc/AlphaNet](https://github.com/zmyybc/AlphaNet) | **GPL-3.0** | **Conflict**: the hosting figshare record ([28831364](https://figshare.com/articles/28831364)) says CC BY 4.0; the author's Matbench Discovery submission says GPL-3.0 | no |
| **Eqnorm** | Eqnorm-MPtrj | [yzchen08/eqnorm](https://github.com/yzchen08/eqnorm) | MIT | MIT | no |
| **fairchem (eSEN)** | eSEN-30M-OAM | [facebookresearch/fairchem](https://github.com/facebookresearch/fairchem) | MIT (code) | **OMat24 License** (Meta; [facebook/OMAT24](https://huggingface.co/facebook/OMAT24)) | **yes** |
| **EquiformerV3** | EqV3-OMatMPtrjSalex | [atomicarchitects/equiformer_v3](https://github.com/atomicarchitects/equiformer_v3) | MIT | MIT (`mirror-physics/equiformer_v3` on HF) | no |
| **UMA** | UMA-s-1p2-* / UMA-s-1p1-* / UMA-m-1p1-* (8) | [facebookresearch/fairchem](https://github.com/facebookresearch/fairchem) | MIT (code) | **FAIR Chemistry License** ([facebook/UMA](https://huggingface.co/facebook/UMA)) | **yes** |
| **PET** | PET-OAM-XL | [metatensor/metatrain](https://github.com/metatensor/metatrain) | BSD-3-Clause | BSD-3-Clause ([lab-cosmo/upet](https://huggingface.co/lab-cosmo/upet)) | no |
| **EquFlash** | EquFlashV2, EquFlash-v1 | [SamsungDS/GGNN](https://github.com/SamsungDS/GGNN) | **CC BY-NC-SA 4.0 — non-commercial** | **Conflict**: the hosting figshare record ([32630910](https://figshare.com/articles/32630910), a Matbench Discovery submission bundle) says CC BY 4.0; the repo license (Samsung Electronics) is CC BY-NC-SA 4.0 and no statement covers the weights | no |
| **MatRIS** | MatRIS-10M-OAM | [HPC-AI-Team/MatRIS](https://github.com/HPC-AI-Team/MatRIS) | BSD-3-Clause | CC BY 4.0 on the hosting figshare record ([30475253](https://figshare.com/articles/30475253)); the repo and [Matbench Discovery](https://matbench-discovery.materialsproject.org/models/matris-10m-oam) declare BSD-3-Clause (both permissive) | no |
| **DPA4** | DPA-4.0.1-pro-MPtrj | [deepmodeling/deepmd-kit](https://github.com/deepmodeling/deepmd-kit) | **LGPL-3.0** | CC-BY-4.0 (the hosting figshare record [32647224](https://figshare.com/articles/32647224); also declared on [Matbench Discovery](https://matbench-discovery.materialsproject.org/models/dpa-4.0.1-pro-mptrj)) | no |
| **TACE** | TACE-OAM-L | [xvzemin/tace](https://github.com/xvzemin/tace) | MIT | CC-BY-4.0 ([xvzemin/tace-foundations](https://huggingface.co/xvzemin/tace-foundations)) | no |

¹ MatterSim: only the two open checkpoints (1M / 5M) are MIT; more advanced versions are gated behind Azure Quantum Elements (commercial Microsoft platform).
Weights licenses are read from the record that hosts the checkpoint file oh-my-mlip downloads: the Hugging Face license tag, the Zenodo or figshare record's license, or the repository LICENSE when the file lives in the repository. Where another upstream declaration disagrees, the table gives both.

## Notes & caveats

- **MACE is the per-model trap.** The repo is MIT, but only the Materials-Project
  line (`MACE-MPA-0`) ships MIT weights. The **`MACE-MH-1` weights are under the
  ASL (Academic Software License) — non-commercial**. So of our three MACE
  variants, only `MACE-MPA-0` is commercial-safe.
- **Training-data terms ≠ model terms.** CHGNet (MPtrj) and others were trained on
  datasets with their own terms of use; that constrains re-training/redistributing
  the *data*. The OMat24 dataset card licenses the dataset under CC-BY-4.0 and says
  nothing about models trained on it; each model's own license is what applies.
- **`oh-my-mlip` never redistributes framework code or gated weights** — gated
  weights (UMA, eSEN) are fetched on first run with *your* token after *you*
  accept the upstream license. See [gated_models.md](gated_models.md).

## Fallback mirror

| Files | Official source | Mirror | Why it is allowed |
|---|---|---|---|
| `NequIP-OAM-XL-0.1.nequip.zip`, `NequIP-OAM-L-0.1.nequip.zip`, `Allegro-OAM-L-0.1.nequip.zip` | Zenodo [10.5281/zenodo.18775904](https://doi.org/10.5281/zenodo.18775904) ("NequIP & Allegro Foundation Potentials", Kavanagh, S. R.; MIR Group @ Harvard) | [JinukMoon/oh-my-mlip-mirror-nequip](https://huggingface.co/JinukMoon/oh-my-mlip-mirror-nequip) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) permits redistribution with attribution; the mirror's model card credits the authors and the record. |

The files are byte-identical copies, unchanged. `scripts/prepare_nequip_weights.py`
downloads from Zenodo first and uses the mirror only when Zenodo fails or stays
slow; whichever source delivers the file, it must match Zenodo's published size
and MD5, and the log names the source used. No other weights are mirrored.

## Quick reference: commercial use

| Safe for commercial use (permissive) | Restricted — check before commercial use |
|---|---|
| MACE-MPA-0, NequIP⁴, Allegro⁴, Nequix, ORB, MatterSim (open), CHGNet, Eqnorm, EquiformerV3, PET, MatRIS⁴, TACE⁴, UMA¹, eSEN¹, DeePMD/DPA²,⁴ | **GRACE** (non-commercial), **MACE-MH-1** (non-commercial), **EquFlash** (non-commercial code; weight declarations conflict), **AlphaNet** (GPL-3.0 code; weight declarations conflict), **SevenNet** (weights license not stated) |

¹ UMA and eSEN: commercial use allowed under Meta's FAIR Chemistry License (UMA) and OMat24 License (eSEN), but the weights are gated and Meta's Acceptable Use Policy applies; the FAIR Chemistry License also requires compliance with applicable laws, including trade control laws.
² DeePMD / DPA4: LGPL-3.0 — commercial use of the library is allowed; copyleft applies only if you modify the library itself.
⁴ CC-BY-4.0 weights (NequIP, Allegro, TACE, DPA, MatRIS) require attribution.

*Code licenses read from each upstream LICENSE file (2026-06); weights licenses checked against the record hosting each checkpoint (2026-09-21). Verify upstream before relying on any entry.*
