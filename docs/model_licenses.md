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
| 🚫 **Non-commercial only** | **GRACE** (`GRACE-2L-OAM`), **EquFlash** (`EquFlash`, `EquFlashV2`), **MACE-MH** weights (`MACE-MH-1-OMAT`, `MACE-MH-1-OC20`) | Academic / research use only. Commercial use needs a separate license from the owner. |
| 🔁 **Strong copyleft (GPL-3.0)** | **AlphaNet** (`AlphaNet-v1-OMA`) | Derivative works must also be GPL-3.0. |
| 🔗 **Weak copyleft (LGPL-3.0)** | **DeePMD** (`DPA-3.1-3M-FT`), **DPA4** (`DPA-4.0.1-pro-MPtrj`) | Using the library is fine; modifications *to the library* must stay LGPL. |
| 🔒 **Gated weights** | **UMA** (all `UMA-*` variants) | Code is MIT, but weights are under the **FAIR Chemistry License**: accept the license on Hugging Face, get access approved, authenticate with an `HF_TOKEN`. Commercial use is permitted but subject to an Acceptable Use Policy and geographic exclusions (no China / Russia / Belarus / sanctioned regions). |
| 📎 **Attribution (CC-BY-4.0) weights** | **NequIP** (`NequIP-OAM-*`), **Allegro** (`Allegro-OAM-L`), **DPA** weights; OMat24-trained variants of **SevenNet** | Free incl. commercial, but you must credit the source / dataset. |

Everything else below is permissive (MIT / Apache-2.0 / BSD-3-Clause), commercial
use allowed.

## Full table — 20 frameworks / 31 model variants

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
| **fairchem (eSEN)** | eSEN-30M-OAM | [facebookresearch/fairchem](https://github.com/facebookresearch/fairchem) | MIT | per model card (e.g. `facebook/OMC25` on HF) | no² |
| **EquiformerV3** | EqV3-OMatMPtrjSalex | [atomicarchitects/equiformer_v3](https://github.com/atomicarchitects/equiformer_v3) | MIT | MIT (`mirror-physics/equiformer_v3` on HF) | no |
| **UMA** | UMA-s-1p2-* / UMA-s-1p1-* / UMA-m-1p1-* (7) | [facebookresearch/fairchem](https://github.com/facebookresearch/fairchem) | MIT (code) | **FAIR Chemistry License** ([facebook/UMA](https://huggingface.co/facebook/UMA)) | **yes** |
| **PET** | PET-OAM-XL | [spozdn/pet](https://github.com/spozdn/pet) | MIT | unconfirmed³ | no |
| **EquFlash** | EquFlash, EquFlashV2 | [SamsungDS/GGNN](https://github.com/SamsungDS/GGNN) | **CC BY-NC-SA 4.0 — non-commercial** | inherits (non-commercial) | no |
| **MatRIS** | MatRIS-10M-OAM | [HPC-AI-Team/MatRIS](https://github.com/HPC-AI-Team/MatRIS) | BSD-3-Clause | BSD-3-Clause (presumed; in-repo) | unknown |
| **DPA4** | DPA-4.0.1-pro-MPtrj | [deepmodeling/deepmd-kit](https://github.com/deepmodeling/deepmd-kit) | **LGPL-3.0** | CC-BY-4.0 (DPA HF) — DPA-4 card unconfirmed³ | no |
| **TACE** | TACE-OAM-L | [xvzemin/tace](https://github.com/xvzemin/tace) | MIT | unconfirmed³ | unknown |

¹ MatterSim: only the two open checkpoints (1M / 5M) are MIT; more advanced versions are gated behind Azure Quantum Elements (commercial Microsoft platform).
² fairchem: the *code* is MIT and ungated; individual model weights carry their own per-card terms (UMA weights are the gated exception — listed separately above).
³ Unconfirmed = the code license is verified from the repo, but the exact weights-license file/host could not be confirmed; it is *presumed* to inherit the repo license. Verify the model card before commercial use.

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
  (UMA) are fetched on first run with *your* `HF_TOKEN` after *you* accept the
  upstream license. See [gated_models.md](gated_models.md).

## Quick reference: commercial use

| Safe for commercial use (permissive) | Restricted — check before commercial use |
|---|---|
| SevenNet, MACE-MPA-0, NequIP, Allegro, Nequix, ORB, MatterSim (open), CHGNet, Eqnorm, fairchem/eSEN, EquiformerV3, MatRIS, TACE, PET, UMA¹, DeePMD/DPA²,⁴ | **GRACE** (non-commercial), **EquFlash** (non-commercial), **MACE-MH-1** (non-commercial), **AlphaNet** (GPL-3.0) |

¹ UMA: commercial use allowed under the FAIR Chemistry License, but gated + Acceptable Use Policy + geographic exclusions apply.
² DeePMD / DPA4: LGPL-3.0 — commercial use of the library is allowed; copyleft applies only if you modify the library itself.
⁴ CC-BY-4.0-weighted models (NequIP, Allegro, DPA, OMat24 variants) require attribution.

*Compiled 2026-06 by reading each upstream LICENSE file. Verify upstream before relying on any entry.*
