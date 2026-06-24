# Source research — public sources for the "source-unknown" frameworks and low-confidence weights

Research date: 2026-06-24. Method: WebSearch + WebFetch + GitHub/HF API, verified against
the actual repos and Hugging Face metadata. Every claim below cites the URL it came from.
Where a fact could not be confirmed, it is marked NEEDS-OWNER with the best lead, not fabricated.

`oh-my-mlip` hosts nothing — every value here is the framework's own official public channel.

---

## Part A — 6 "source-unknown" frameworks (find the PUBLIC source + how to build/install)

### Summary table

| # | Framework | Official public source | Install / build | Needs compile? | Verdict |
|---|---|---|---|---|---|
| 1 | allegro | `mir-group/allegro` + `mir-group/pair_nequip_allegro` | python parts: pip; LAMMPS deploy: C++ cmake | YES (LAMMPS only) | **RESOLVED** |
| 2 | alphanet | `zmyybc/AlphaNet` (GPL-3.0) | `pip install git+https://github.com/zmyybc/AlphaNet.git` | NO | **RESOLVED** |
| 3 | equflash | `SamsungDS/GGNN` + (v1 only) `SNU-ARC/flashTP` (MIT) | pip reqs; flashTP = source build | YES (flashTP only, v1) | **RESOLVED** |
| 4 | grace | `ICAMS/grace-tensorpotential` | `pip install tensorpotential` OR `pip install .` | NO | **RESOLVED** |
| 5 | tace | `xvzemin/tace` (MIT) | `pip install git+https://github.com/xvzemin/tace.git` | NO (pure Python) | **RESOLVED** |
| 6 | equiformer_v3 | `atomicarchitects/equiformer_v3` (MIT); fairchem-core is **vendored** | `pip install -e packages/fairchem-core` (bundled) | NO | **RESOLVED** |

**Part A result: 6 RESOLVED / 0 NEEDS-OWNER.** All six "local build" blockers trace to a
public source. Two still need a real compile step on the target host (allegro's LAMMPS, and
flashTP for EquFlash v1) — but the source for that compile is public, so they are not owner-private.

---

### 1. allegro — LAMMPS-MLIAP wheel blocker

- **Official source:**
  - Allegro python model: `https://github.com/mir-group/allegro`
  - NequIP framework (dependency): `https://github.com/mir-group/nequip`
  - LAMMPS pair styles: `https://github.com/mir-group/pair_nequip_allegro`
    (replaces the deprecated `mir-group/pair_allegro` / `pair_nequip`)
- **Python install (pip, no compile):**
  ```bash
  pip install nequip            # the framework
  git clone --depth 1 https://github.com/mir-group/allegro.git
  cd allegro && pip install .   # or: pip install git+https://github.com/mir-group/allegro.git
  ```
  The `nequip` / `nequip-allegro` python side IS pip-installable.
- **LAMMPS deployment (the actual blocker — C++ cmake, NOT pip):**
  ```bash
  git clone --depth=1 https://github.com/lammps/lammps
  git clone --depth=1 https://github.com/mir-group/pair_nequip_allegro
  ./pair_nequip_allegro/patch_lammps.sh /path/to/lammps/
  # cmake build with libtorch / KOKKOS, e.g.:
  cmake ../cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') \
        -DNEQUIP_AOT_COMPILE=ON -DPKG_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON
  make -j$(nproc)
  ```
  Requires the **10 Sep 2025 LAMMPS release or newer**, KOKKOS in double-double precision.
  Models are compiled to `.nequip.pth` / `.nequip.pt2` via `nequip-compile` before LAMMPS use.
  There is also a newer **LAMMPS ML-IAP integration** path (build LAMMPS with the ML-IAP
  package, then `nequip-prepare-lmp-mliap`) — still a LAMMPS build, not a pip wheel.
- **Needs compile?** YES — strictly a local LAMMPS C++ cmake build per host/GPU. **There is no pip path for the LAMMPS deployment.** (The python `nequip`/`allegro` parts are pip.)
- **Verdict: RESOLVED.** Confidence: **high**.
- Sources:
  - https://github.com/mir-group/pair_nequip_allegro/blob/main/README.md
  - https://github.com/mir-group/allegro
  - https://github.com/mir-group/nequip/releases

### 2. alphanet — zmyybc/AlphaNet

- **Official source:** `https://github.com/zmyybc/AlphaNet`
- **License:** **GPL-3.0** (confirmed via GitHub API + LICENSE file).
- **Install (pip from github — confirmed in README):**
  ```bash
  pip install git+https://github.com/zmyybc/AlphaNet.git
  # or editable from a clone:
  git clone https://github.com/zmyybc/AlphaNet.git && cd AlphaNet && pip install -e .
  ```
- **Needs compile?** NO — standard pip install.
- **Verdict: RESOLVED.** Confidence: **high**.
- Sources:
  - https://github.com/zmyybc/AlphaNet
  - https://github.com/zmyybc/AlphaNet/blob/main/LICENSE
  - paper: https://www.nature.com/articles/s41524-025-01817-w

### 3. equflash — Samsung GGNN + flashTP_e3nn

- **Official source:** `https://github.com/SamsungDS/GGNN` (license: NOASSERTION — non-standard;
  owner should confirm exact terms, but the repo IS public).
- **flashTP_e3nn blocker — RESOLVED:** flashTP is **public** at `https://github.com/SNU-ARC/flashTP`
  (Seoul National University ARC lab, **MIT** license). The importable module is `flashTP_e3nn`.
  Build (CUDA compile required):
  ```bash
  git clone https://github.com/SNU-ARC/flashTP.git && cd flashTP
  pip install -r requirements.txt
  CUDA_ARCH_LIST="80;90" pip install . --no-build-isolation   # auto-detects CC if omitted; ~10 min
  ```
- **Important nuance (matches the registry's existing note):** GGNN's own `requirements.txt`
  pins **`cuequivariance` / `cuequivariance-torch` / `cuequivariance-ops-torch-cu12` (==0.6.0)**
  and does **NOT** list flashTP. flashTP is only needed for the **EquFlash v1** `flashtp` backend;
  **EquFlashV2 is cueq-only** (FullConv rejects `conv_type='flashtp'`). So:
  - EquFlashV2: no flashTP needed — pure pip (cuequivariance from PyPI).
  - EquFlash v1: needs flashTP — public source build (`SNU-ARC/flashTP`), CUDA compile.
- **GGNN install (deps only shown in README; package install command not stated explicitly):**
  ```bash
  pip install -r requirements.txt
  pip install --no-deps -r requirements-no-deps.txt   # pins fairchem.core==1.10.0
  # then install the GGNN package itself (pip install -e . / pip install . — confirm with repo)
  ```
- **Needs compile?** Only for EquFlash v1's flashTP backend (public). V2 = no compile.
- **Verdict: RESOLVED.** Confidence: **high** for flashTP source/build; **medium** on the exact
  GGNN package-install command (README shows only the two requirements installs).
- Sources:
  - https://github.com/SamsungDS/GGNN
  - https://github.com/SNU-ARC/flashTP
  - GGNN requirements.txt (cuequivariance pins) / requirements-no-deps.txt (fairchem.core==1.10.0)

### 4. grace — ICAMS/grace-tensorpotential

- **Official source:** `https://github.com/ICAMS/grace-tensorpotential`
  (the `tensorpotential` python package lives here). Also `https://github.com/ICAMS/TensorPotential`.
- **Install (pip — confirmed in README + readthedocs):**
  ```bash
  pip install tensorpotential        # PyPI — TensorFlow pulled automatically
  # developer version from source:
  git clone https://github.com/ICAMS/grace-tensorpotential.git
  cd grace-tensorpotential && pip install .
  ```
  Note: needs legacy Keras — `export TF_USE_LEGACY_KERAS=1` if `keras>=3.0` is present.
- **Needs compile?** NO — pip / TensorFlow.
- **Verdict: RESOLVED.** Confidence: **high**.
- Sources:
  - https://github.com/ICAMS/grace-tensorpotential
  - https://gracemaker.readthedocs.io/en/latest/gracemaker/install/

### 5. tace — xvzemin/tace

- **Official source:** `https://github.com/xvzemin/tace` (**MIT** license, confirmed via API + LICENSE.md).
- **Build system:** `setuptools>=61` / `setuptools.build_meta`, **pure Python**
  (pyproject.toml shows **no** C/CUDA/cmake/pybind extension). Pip package name: `TACE`, version 0.2.0.
- **Install:**
  ```bash
  pip install git+https://github.com/xvzemin/tace.git
  # or from a clone:
  git clone https://github.com/xvzemin/tace.git && cd tace && pip install .
  ```
  (README.rst shows `git clone` then running examples; pyproject confirms it is a normal
  pip-installable pure-Python package — no separate local/external compiled library.)
- **Needs compile?** NO.
- **Verdict: RESOLVED.** Confidence: **high** (pure-Python pip confirmed from pyproject.toml).
- Sources:
  - https://github.com/xvzemin/tace
  - https://github.com/xvzemin/tace/blob/main/pyproject.toml
  - docs: https://tace.readthedocs.io/en/latest/index.html
  - paper: https://arxiv.org/abs/2509.14961

### 6. equiformer_v3 — atomicarchitects/equiformer_v3 + its fairchem-core

- **Official source:** `https://github.com/atomicarchitects/equiformer_v3` (**MIT**).
- **The fairchem-core question — RESOLVED:** the equiformer_v3 repo **vendors** (bundles) its
  own copy of fairchem-core under `packages/fairchem-core`, and you install **that local copy
  editable**. It is NOT a separate VCS fork URL — the README states the repo "is based on
  [this version of fairchem]" = **facebookresearch/fairchem commit `977a80328f2be44649b414a9907a1d6ef2f81e95`**,
  with the EquiformerV3 code under `experimental/`. So the "editable local fairchem-core with no
  VCS URL" in the internal env is exactly the bundled `packages/fairchem-core` from this repo.
- **Install (from `experimental/docs/env_setup.md`):**
  ```bash
  conda create -n equiformer_v3 python=3.11 -c conda-forge && conda activate equiformer_v3
  pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
  pip install torch_geometric
  pip install -r experimental/env/conda_requirements.txt
  pip install -e packages/fairchem-core      # <-- the bundled fairchem-core, editable
  # (matbench-discovery @ commit 375a8d6 for evaluation, optional)
  ```
  Reproduce on a foreign host by cloning equiformer_v3 (which carries `packages/fairchem-core`)
  and running the editable install — no external fork URL needed.
- **Needs compile?** NO (pure pip/editable).
- **Verdict: RESOLVED.** Confidence: **high**. (This corrects the prior weights_source_review note
  that called the fork "unknown / not pip-resolvable" — it is the vendored `packages/fairchem-core`,
  upstream commit `977a8032...`.)
- Sources:
  - https://github.com/atomicarchitects/equiformer_v3 (README — "based on this version of fairchem" → commit 977a8032)
  - https://github.com/atomicarchitects/equiformer_v3/blob/main/experimental/docs/env_setup.md
  - upstream: https://github.com/facebookresearch/fairchem/tree/977a80328f2be44649b414a9907a1d6ef2f81e95

---

## Part B — 5 low-confidence weights (confirm the exact official download)

### Summary table

| # | Model | Exact official source | Gated? | Confidence | Verdict vs registry |
|---|---|---|---|---|---|
| 1 | eSEN-30M-OAM | `facebook/OMAT24` → file `esen_30m_oam.pt` | **YES (gated: manual)** | **high** | **registry `gated:false` is WRONG → must be `gated:true`** |
| 2 | PET-OAM-XL | `lab-cosmo/upet` → `models/pet-oam-xl-v1.0.0.ckpt` | No | **high** | only `.ckpt` exists, no `.pt` |
| 3 | EqV3-OMatMPtrjSalex | `mirror-physics/equiformer_v3` → `checkpoint/omat24-mptrj-salex_gradient.pt` | No | **high** | registry path missing `checkpoint/` prefix (→ 404) |
| 4 | EquFlash v1 | figshare `ndownloader/files/65435004` | No | **high** | v1 figshare file now found |
| 5 | MACE-MH-1-OC20 | `mace_mp(model="mh-1", head="oc20_usemppbe")` | No | **high** | head is `oc20_usemppbe`, NOT `oc20` |

**Part B result: 5/5 confirmed (all upgraded from low → high). Two registry corrections needed
(eSEN gated flag; MACE OC20 head name) and two path fixes (EqV3 filename prefix; PET .ckpt vs .pt).**

---

### 1. eSEN-30M-OAM (fairchem) — THE CRITICAL ONE

- **Exact source:** Hugging Face repo **`facebook/OMAT24`** (note: `fairchem/OMAT24` HTTP-redirects
  to `facebook/OMAT24`), file **`esen_30m_oam.pt`**. Sibling files also present:
  `esen_30m_mptrj.pt`, `esen_30m_omat.pt`.
- **GATED? → YES.** The repo is gated: the HF page shows *"You need to agree to share your contact
  information to access this model"*, and the HF API reports **`gated: manual`** (manual approval).
  License is the custom **"OMat24 License"** (not standard OSS).
  - **License URL:** `https://huggingface.co/facebook/OMAT24/blob/main/LICENSE` (confirmed HTTP 200).
  - Repo/gate page: `https://huggingface.co/fairchem/OMAT24` (== `https://huggingface.co/facebook/OMAT24`).
- **Implication for models.json:** eSEN-30M-OAM **must be `gated: true`** with
  `license_url: https://huggingface.co/facebook/OMAT24` (or the LICENSE blob URL). The current
  `gated: false` is incorrect — downloading `esen_30m_oam.pt` requires HF login + license acceptance + token,
  exactly like the UMA gating flow already documented in `docs/gated_models.md`.
- **Confidence: high.**
- Sources:
  - https://huggingface.co/fairchem/OMAT24 (gate banner)
  - https://huggingface.co/api/models/facebook/OMAT24 (`gated: manual`)
  - https://huggingface.co/facebook/OMAT24/blob/main/LICENSE

### 2. PET-OAM-XL

- **Exact source:** repo **`lab-cosmo/upet`** (NOT `pet-mad`), file **`models/pet-oam-xl-v1.0.0.ckpt`**.
- **Format:** only a **`.ckpt`** exists — there is **no `.pt`** in the repo (full file list confirmed via
  HF API; the `.pt` the registry references must be produced by a TorchScript/export step, or the
  registry path should point at the `.ckpt`).
- **Gated? No** (HF API `gated: False`; license **BSD-3-Clause**).
- **Confidence: high** (exact filename verified).
- Sources:
  - https://huggingface.co/lab-cosmo/upet
  - https://huggingface.co/api/models/lab-cosmo/upet (siblings list)

### 3. EqV3-OMatMPtrjSalex

- **Exact source:** repo **`mirror-physics/equiformer_v3`**, file
  **`checkpoint/omat24-mptrj-salex_gradient.pt`** (the registry guessed
  `omat24-mptrj-salex_gradient.pt` WITHOUT the `checkpoint/` directory prefix → that is why it 404'd).
- **All 4 files in the repo (verified via HF API):**
  - `checkpoint/mptrj_gradient.pt`
  - `checkpoint/omat24-mptrj-salex_gradient.pt`  ← this one
  - `checkpoint/omat24_direct.pt`
  - `checkpoint/omat24_gradient.pt`
- **Gated? No** (HF API `gated: False`; license **MIT**).
- **Confidence: high** (exact filenames verified).
- Sources:
  - https://huggingface.co/mirror-physics/equiformer_v3
  - https://huggingface.co/api/models/mirror-physics/equiformer_v3 (siblings list)

### 4. EquFlash v1

- **Exact source (figshare direct download):** **`https://figshare.com/ndownloader/files/65435004`**
  (this is EquFlash **v1**; EquFlashV2 is `.../files/65435007`, already in the registry).
- **Gated? No** (public figshare).
- **Confidence: high** — the GGNN README lists both checkpoints on figshare and distinguishes v1
  (65435004) from v2 (65435007). This upgrades the prior NEEDS-OWNER on EquFlash v1.
- Sources:
  - https://github.com/SamsungDS/GGNN (README — figshare checkpoint links for v1 and v2)

### 5. MACE-MH-1-OC20

- **Model name / download keyword:** `mh-1` (confirmed — `mace_mp(model="mh-1", ...)`).
  Released via `ACEsuit/mace-foundations` / HF `mace-foundations/mace-mh-1`.
- **OC20 head — CORRECTION:** the OC20 head name is **`oc20_usemppbe`**, NOT `oc20`.
  MACE-MH-1 is multi-head; its heads are `omat_pbe` (default), `omol`, `spice_wB97M`,
  **`oc20_usemppbe`** (PBE surfaces — the OC20/catalysis head), `matpes_r2scan`, `rgd1_b3lyp`.
  Load:
  ```python
  from mace.calculators import mace_mp
  calc = mace_mp(model="mh-1", default_dtype="float64", device="cuda", head="oc20_usemppbe")
  ```
- **Gated? No.**
- **Implication for models.json:** the MACE-MH-1-OC20 inference line should use
  `head='oc20_usemppbe'`, not `head='oc20'` (the latter is not a valid head name). The `mh-1`
  weights value is correct.
- **Confidence: high.**
- Sources:
  - https://huggingface.co/mace-foundations/mace-mh-1 (head list incl. `oc20_usemppbe`)
  - https://github.com/ACEsuit/mace-foundations
  - https://github.com/ACEsuit/mace/issues/1329 (MH-1 multi-head)

---

## Overall result

- **Part A: 6 RESOLVED / 0 NEEDS-OWNER.** Every "local build" framework has a public source:
  - allegro → `mir-group/{allegro,nequip,pair_nequip_allegro}` (python pip; LAMMPS = local C++ cmake, no pip path)
  - alphanet → `zmyybc/AlphaNet` (GPL-3.0, `pip install git+https`)
  - equflash → `SamsungDS/GGNN` + `SNU-ARC/flashTP` (MIT, v1-only CUDA build; V2 is cueq-only pip)
  - grace → `ICAMS/grace-tensorpotential` (`pip install tensorpotential`)
  - tace → `xvzemin/tace` (MIT, pure-Python `pip install git+https`)
  - equiformer_v3 → `atomicarchitects/equiformer_v3` (MIT) with **vendored** `packages/fairchem-core`
    (upstream facebookresearch/fairchem @ `977a80328f2be44649b414a9907a1d6ef2f81e95`) — `pip install -e packages/fairchem-core`

- **Part B: 5/5 confirmed (all low → high).** Net registry actions:
  1. **eSEN-30M-OAM must be `gated: true`** (license_url `https://huggingface.co/facebook/OMAT24`).
  2. EqV3 path → `checkpoint/omat24-mptrj-salex_gradient.pt` (add the `checkpoint/` prefix).
  3. PET-OAM-XL is `.ckpt` only (`models/pet-oam-xl-v1.0.0.ckpt`) — no `.pt` in repo.
  4. EquFlash v1 weights URL → `https://figshare.com/ndownloader/files/65435004`.
  5. MACE-MH-1-OC20 head → `oc20_usemppbe` (not `oc20`).

- **THE CRITICAL eSEN ANSWER: YES, eSEN-30M-OAM is GATED.** `esen_30m_oam.pt` lives in
  `facebook/OMAT24` (= `fairchem/OMAT24`), which is gated (`gated: manual`, "agree to share contact
  information") under the custom OMat24 License. `models.json` should set `gated: true` with
  `license_url: https://huggingface.co/facebook/OMAT24`.
