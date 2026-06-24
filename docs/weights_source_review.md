# Weights-source owner-review table

This table records the **real official-source** `weights_fetch` / `weights_source`
values now committed to `models.json`, replacing the prior `TODO-owner-review`
placeholders. `oh-my-mlip` hosts nothing — every value is the framework's own
official channel (PyPI-bundled name, vendor download URL, or HF repo id).

- **`by-name`** — the loader resolves the weights by a name/cache key. `weights_source`
  is that loader name. A public model-card URL is carried in the optional
  `weights_source_url` field for transparency.
- **`url`** — `weights_source` is an `http(s)://` official download/page URL.
  `weights_source_url` (when present) points at the exact file.
- **`gated-hf`** — `weights_source` is the gated HF repo id (`owner/name`). All 7
  UMA variants; carried verbatim from the prior goal.

`confidence` = how sure we are the exact official name/URL is correct.
`needs-owner-review?` = YES where the value is best-effort and the owner should
confirm before it is relied on (see notes below the table).

## All 31 model versions

| Model | Framework | weights | weights_fetch | weights_source | confidence | needs-owner-review? |
|---|---|---|---|---|---|---|
| SevenNet-MF-OMPA | SevenNet | bundled | by-name | `7net-mf-ompa` | high | no |
| SevenNet-Omni | SevenNet | bundled | by-name | `7net-omni` | high | no |
| MACE-MPA-0 | MACE | auto-download | by-name | `medium-mpa-0` | high | no |
| MACE-MH-1-OMAT | MACE | auto-download | by-name | `mh-1` | high | no |
| MACE-MH-1-OC20 | MACE | auto-download | by-name | `mh-1` | low | **YES** |
| NequIP-OAM-XL | NequIP | on-demand-hf | url | https://www.nequip.net/models/mir-group/NequIP-OAM-XL:0.1 | high | no |
| NequIP-OAM-L | NequIP | on-demand-hf | url | https://www.nequip.net/models/mir-group/NequIP-OAM-L:0.1 | high | no |
| Allegro-OAM-L | Allegro | on-demand-hf | url | https://www.nequip.net/models/mir-group/Allegro-OAM-L:0.1 | high | no |
| Nequix-MP-1 | Nequix | on-demand-hf | url | https://github.com/atomicarchitects/nequix/raw/7c2854d.../models/nequix-mp-1.nqx | high | no |
| DPA-3.1-3M-FT | DeePMD | on-demand-hf | url | https://huggingface.co/deepmodelingcommunity/DPA | high | no |
| ORB-v3 | ORB | auto-download | by-name | `orb_v3_conservative_inf_omat` | high | no |
| GRACE-2L-OAM | GRACE | on-demand-hf | by-name | `GRACE-2L-OAM` | high | no |
| MatterSim-v1-5M | MatterSim | bundled | by-name | `MatterSim-v1.0.0-5M.pth` | high | no |
| CHGNet-v0.3.0 | CHGNet | bundled | by-name | `0.3.0` | high | no |
| AlphaNet-v1-OMA | AlphaNet | on-demand-hf | url | https://ndownloader.figshare.com/files/53851139 | high | no |
| Eqnorm-MPtrj | Eqnorm | auto-download | by-name | `eqnorm-mptrj` | high | no |
| eSEN-30M-OAM | fairchemv1 | on-demand-hf | url | https://huggingface.co/fairchem/OMAT24 | low | **YES** |
| EqV3-OMatMPtrjSalex | EquiformerV3 | on-demand-hf | url | https://huggingface.co/mirror-physics/equiformer_v3 | low | **YES** (exact HF filename unconfirmed) |
| UMA-m-1p1-OC20 | UMA | on-demand-hf | gated-hf | `facebook/UMA` | high | no |
| UMA-m-1p1-OMAT | UMA | on-demand-hf | gated-hf | `facebook/UMA` | high | no |
| UMA-s-1p1-OC20 | UMA | on-demand-hf | gated-hf | `facebook/UMA` | high | no |
| UMA-s-1p1-OMAT | UMA | on-demand-hf | gated-hf | `facebook/UMA` | high | no |
| UMA-s-1p2-OC20 | UMA | on-demand-hf | gated-hf | `facebook/UMA` | high | no |
| UMA-s-1p2-OC22 | UMA | on-demand-hf | gated-hf | `facebook/UMA` | high | no |
| UMA-s-1p2-OMAT | UMA | on-demand-hf | gated-hf | `facebook/UMA` | high | no |
| PET-OAM-XL | PET | on-demand-hf | url | https://huggingface.co/lab-cosmo/upet | low | **YES** |
| EquFlashV2 | EquFlash | on-demand-hf | url | https://figshare.com/ndownloader/files/65435007 | high | no |
| EquFlash | EquFlash | on-demand-hf | url | https://github.com/SamsungDS/GGNN | low | **YES** |
| MatRIS-10M-OAM | MatRIS | auto-download | by-name | `matris_10m_oam` | low | **YES** |
| DPA-4.0.1-pro-MPtrj | DPA4 | on-demand-hf | url | https://matbench-discovery.materialsproject.org/models/dpa-4.0.1-pro-mptrj | high | no |
| TACE-OAM-L | TACE | auto-download | by-name | `TACE-OAM-L` | high | no |

**Confidence distribution:** 26 high / 5 low. Needs-owner-review: 5 models.

## Notes on the LOW-confidence / needs-owner-review entries

1. **MACE-MH-1-OC20** — The `mace_mp` keyword `mh-1` and the mace-foundations
   release (`mace_mh_1`) are confirmed, so the **weights** value is correct. The
   open question is the **inference `head=`** (this row uses `head='oc20'`); the
   existing registry note already flags "confirm OC20 head with jumoon". The
   inference line is owned by a separate agent — flagged here only so the owner
   reconciles the OC20 head when confirming weights.

2. **eSEN-30M-OAM** — The checkpoint `esen_30m_oam.pt` is confirmed to live in
   HF repo **`fairchem/OMAT24`** (not `facebook/OMC25`/`facebook/eSEN`). However
   that repo is **GATED** (license acceptance + HF login required to download),
   which conflicts with the registry's `gated: false` for this version. The
   `gated` honesty field is owned by another agent; the owner must reconcile
   whether eSEN-30M-OAM should be marked gated (with a `license_url`) given its
   actual host. `weights_source` is set to the repo URL as the official channel.

3. **PET-OAM-XL** — Official repo `lab-cosmo/upet` confirmed, but the only file
   present is **`pet-oam-xl-v1.0.0.ckpt`** (2.92 GB) — there is **no `.pt`
   variant**, whereas the registry's `model_source`/inference reference a `.pt`
   file. `weights_source_url` points at the real `.ckpt`. Owner should confirm
   how the `.pt` is produced (TorchScript export step) or update the path.

4. **EquFlash (v1)** — Repo `SamsungDS/GGNN` confirmed; only **EquFlashV2's**
   figshare checkpoint URL was verifiable
   (`.../ndownloader/files/65435007`). A v1-specific figshare file URL could not
   be confirmed, so `weights_source` is the repo URL as best-effort. Owner should
   supply the exact EquFlash-v1 checkpoint URL.

5. **MatRIS-10M-OAM** — The by-name loader key `matris_10m_oam` is confirmed from
   the repo README usage examples (correct for `weights_fetch: by-name`). The
   underlying download backend (reportedly a figshare `MatRIS_10M_OAM.pth.tar`)
   could **not** be confirmed, so no exact file URL is recorded; `weights_source_url`
   is the repo. The by-name value is usable as-is; owner may add the figshare URL.

## Upstream-unknown / not pip-resolvable (recipe-build blockers — distinct from weights)

These two are flagged for **owner input on the install recipe**, not on weights.
The weights values above are independently fine; the blockers are about *building
the environment*.

| Item | Blocker | Owner input needed |
|---|---|---|
| `equiformer_v3` (EquiformerV3 env) | Needs a specific **fairchem-core fork** (editable `dev6` install per the registry note). The exact fork repo/commit URL is unknown and not pip-resolvable. | Provide the fairchem-core fork URL + pin so `install.sh` can recreate the editable install on a foreign host. |
| `allegro` (Allegro env) | Requires a **local LAMMPS-MLIAP build** (compiled `.pt2` reselected/recompiled per host GPU); not a pip-resolvable wheel. | Confirm the LAMMPS + MLIAP build recipe / source so the env can be reproduced. |

Note: **MatRIS weights are fine** (the by-name value above is usable); its *recipe*
is merely a candidate and is **not** in this build-blocker list.
