# Model & framework licenses

`oh-my-mlip` is an **MIT-licensed installer / orchestrator**. It does **not**
redistribute any framework's source code or model weights — every framework is
installed from its own official channel (PyPI / GitHub / Hugging Face / Zenodo)
and every weight is downloaded from its upstream host. **Each one stays under its
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
| 🚫 **Non-commercial only** | **GRACE** (`GRACE-2L-OAM`), **EquFlash** (`EquFlashV2`, `EquFlash-v1`), **MACE-MH** weights (`MACE-MH-1-OMAT`, `MACE-MH-1-OC20`) | Academic / research use only. Commercial use needs a separate license from the owner. |
| 🔁 **Strong copyleft (GPL-3.0)** | **AlphaNet** (`AlphaNet-v1-OMA`) | Derivative works must also be GPL-3.0. |
| 🔗 **Weak copyleft (LGPL-3.0)** | **DeePMD** (`DPA-3.1-3M-FT`), **DPA4** (`DPA-4.0.1-pro-MPtrj`) | Using the library is fine; modifications *to the library* must stay LGPL. |
| 🔒 **Gated weights** | **UMA** (all `UMA-*` variants), **eSEN** (`eSEN-30M-OAM`) | Code is MIT, but the weights are under Meta licenses on Hugging Face (FAIR Chemistry License for UMA, OMat24 License for eSEN): accept the license, get access approved, log in with your token. Commercial use is permitted, subject to Meta's Acceptable Use Policy (UMA also has geographic exclusions). |
| 📎 **Attribution (CC-BY-4.0) weights** | **NequIP** (`NequIP-OAM-*`), **Allegro** (`Allegro-OAM-L`), **TACE** (`TACE-OAM-L`), **DPA** weights (`DPA-3.1-3M-FT`, `DPA-4.0.1-pro-MPtrj`); OMat24-trained variants of **SevenNet** | Free incl. commercial, but you must credit the source / dataset. |

Everything else below is permissive (MIT / Apache-2.0 / BSD-3-Clause), commercial
use allowed.

## Full table — 20 frameworks / 32 model variants

| Framework | Our variant(s) | GitHub | Code license | Weights license | Gated |
|---|---|---|---|---|---|
| **SevenNet** | SevenNet-MF-OMPA, SevenNet-Omni | [MDIL-SNU/SevenNet](https://github.com/MDIL-SNU/SevenNet) | MIT | MIT (OMat24-trained → CC-BY-4.0 attribution) | no |
| **MACE** | MACE-MPA-0, MACE-MH-1-OMAT, MACE-MH-1-OC20 | [ACEsuit/mace](https://github.com/ACEsuit/mace) | MIT | **Mixed**: MPA-0 = MIT; **MH-1 = ASL (non-commercial)** | no |
| **NequIP** | NequIP-OAM-XL, NequIP-OAM-L | [mir-group/nequip](https://github.com/mir-group/nequip) | MIT | CC-BY-4.0 | no |
| **Allegro** | Allegro-OAM-L | [mir-group/allegro](https://github.com/mir-group/allegro) | MIT | CC-BY-4.0 | no |
| **Nequix** | Nequix-MP-1 | [atomicarchitects/nequix](https://github.com/atomicarchitects/nequix) | MIT | MIT (weights in-repo) | no |
| **DeePMD** | DPA-3.1-3M-FT | [deepmodeling/deepmd-kit](https://github.com/deepmodeling/deepmd-kit) | **LGPL-3.0** | CC-BY-4.0 (`deepmodelingcommunity/DPA` on HF) | no |
| **ORB** | ORB-v3 | [orbital-materials/orb-models](https://github.com/orbital-materials/orb-models) | Apache-2.0 | Apache-2.0 (code + weights) | no |
| **GRACE** | GRACE-2L-OAM | [ICAMS/grace-tensorpotential](https://github.com/ICAMS/grace-tensorpotential) | **ASL — non-commercial** | ASL — non-commercial | no |
| **MatterSim** | MatterSim-v1-5M | [microsoft/mattersim](https://github.com/microsoft/mattersim) | MIT | MIT (open checkpoints) | no¹ |
| **CHGNet** | CHGNet-v0.3.0 | [CederGroupHub/chgnet](https://github.com/CederGroupHub/chgnet) | BSD-3-Clause | BSD-3-Clause | no |
| **AlphaNet** | AlphaNet-v1-OMA | [zmyybc/AlphaNet](https://github.com/zmyybc/AlphaNet) | **GPL-3.0** | GPL-3.0 | no |
| **Eqnorm** | Eqnorm-MPtrj | [yzchen08/eqnorm](https://github.com/yzchen08/eqnorm) | MIT | MIT | no |
| **fairchem (eSEN)** | eSEN-30M-OAM | [facebookresearch/fairchem](https://github.com/facebookresearch/fairchem) | MIT (code) | **OMat24 License** (Meta; [facebook/OMAT24](https://huggingface.co/facebook/OMAT24)) | **yes** |
| **EquiformerV3** | EqV3-OMatMPtrjSalex | [atomicarchitects/equiformer_v3](https://github.com/atomicarchitects/equiformer_v3) | MIT | MIT (`mirror-physics/equiformer_v3` on HF) | no |
| **UMA** | UMA-s-1p2-* / UMA-s-1p1-* / UMA-m-1p1-* (8) | [facebookresearch/fairchem](https://github.com/facebookresearch/fairchem) | MIT (code) | **FAIR Chemistry License** ([facebook/UMA](https://huggingface.co/facebook/UMA)) | **yes** |
| **PET** | PET-OAM-XL | [metatensor/metatrain](https://github.com/metatensor/metatrain) | BSD-3-Clause | BSD-3-Clause ([lab-cosmo/upet](https://huggingface.co/lab-cosmo/upet)) | no |
| **EquFlash** | EquFlashV2, EquFlash-v1 | [SamsungDS/GGNN](https://github.com/SamsungDS/GGNN) | **CC BY-NC-SA 4.0 — non-commercial** | inherits (non-commercial) | no |
| **MatRIS** | MatRIS-10M-OAM | [HPC-AI-Team/MatRIS](https://github.com/HPC-AI-Team/MatRIS) | BSD-3-Clause | BSD-3-Clause (declared on [Matbench Discovery](https://matbench-discovery.materialsproject.org/models/matris-10m-oam)) | no |
| **DPA4** | DPA-4.0.1-pro-MPtrj | [deepmodeling/deepmd-kit](https://github.com/deepmodeling/deepmd-kit) | **LGPL-3.0** | CC-BY-4.0 (declared on [Matbench Discovery](https://matbench-discovery.materialsproject.org/models/dpa-4.0.1-pro-mptrj)) | no |
| **TACE** | TACE-OAM-L | [xvzemin/tace](https://github.com/xvzemin/tace) | MIT | CC-BY-4.0 ([xvzemin/tace-foundations](https://huggingface.co/xvzemin/tace-foundations)) | no |

¹ MatterSim: only the two open checkpoints (1M / 5M) are MIT; more advanced versions are gated behind Azure Quantum Elements (commercial Microsoft platform).
Weights licenses are taken from the Hugging Face license tag of the repository that hosts the checkpoint, or, for checkpoints hosted on figshare (DPA-4, MatRIS), the license declared on the model's Matbench Discovery page.

## Notes & caveats

- **MACE is the per-model trap.** The repo is MIT, but only the Materials-Project
  line (`MACE-MPA-0`) ships MIT weights. The **`MACE-MH-1` weights are under the
  ASL (Academic Software License) — non-commercial**. So of our three MACE
  variants, only `MACE-MPA-0` is commercial-safe.
- **OMat24-trained weights** (several frameworks include an OMat24 variant) carry
  the dataset's **CC-BY-4.0 attribution** requirement even when the model code is
  MIT.
- **Training-data terms ≠ model terms.** CHGNet (MPtrj) and others were trained on
  datasets with their own terms of use; that constrains re-training/redistributing
  the *data*, not normal use of the released model.
- **`oh-my-mlip` never redistributes weights or framework code** — gated weights
  (UMA, eSEN) are fetched on first run with *your* token after *you* accept the
  upstream license. See [gated_models.md](gated_models.md).

## Quick reference: commercial use

| Safe for commercial use (permissive) | Restricted — check before commercial use |
|---|---|
| SevenNet, MACE-MPA-0, NequIP⁴, Allegro⁴, Nequix, ORB, MatterSim (open), CHGNet, Eqnorm, EquiformerV3, PET, MatRIS, TACE⁴, UMA¹, eSEN¹, DeePMD/DPA²,⁴ | **GRACE** (non-commercial), **EquFlash** (non-commercial), **MACE-MH-1** (non-commercial), **AlphaNet** (GPL-3.0) |

¹ UMA and eSEN: commercial use allowed under Meta's FAIR Chemistry License (UMA) and OMat24 License (eSEN), but the weights are gated and Meta's Acceptable Use Policy applies; UMA also has geographic exclusions.
² DeePMD / DPA4: LGPL-3.0 — commercial use of the library is allowed; copyleft applies only if you modify the library itself.
⁴ CC-BY-4.0 weights (NequIP, Allegro, TACE, DPA, OMat24 variants) require attribution.

*Code licenses read from each upstream LICENSE file (2026-06); weights licenses checked against each checkpoint's host (2026-09). Verify upstream before relying on any entry.*
