# oh-my-mlip

<p align="center">
  <img src="assets/logo.png" alt="oh-my-mlip — machine learning interatomic potentials" width="560">
</p>

[![CI (GPU-free)](https://github.com/JinukMoon/oh-my-mlip/actions/workflows/ci.yml/badge.svg)](https://github.com/JinukMoon/oh-my-mlip/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> oh-my-mlip won't pick the best MLIP for you — but you'll never fight a conda
> environment, a training config, or a benchmark script by hand again.

**One registry, many MLIPs.** 20 machine-learning interatomic-potential
frameworks (32 model variants), each in its own conda env built from a
versioned recipe. Facts live in `models.json`; every procedure is a file on
disk you can rerun without an agent. It drives the real upstream frameworks
and never reimplements a model.

| | Feature | Command |
|---|---|---|
| 1 | [Install & environments](#install--environments) | `/oh-my-mlip:setup MACE` |
| 2 | [Benchmark (catbench)](#benchmark-catbench) | `/oh-my-mlip:catbench` |
| 3 | [Fine-tune](#fine-tune) | `/oh-my-mlip:finetune` |
| 4 | [Distill](#distill) | `/oh-my-mlip:distill` |

## Quickstart (Claude Code)

```
/plugin marketplace add JinukMoon/oh-my-mlip
```

```
/plugin install oh-my-mlip@oh-my-mlip
```

Then ask in plain language — "install MACE", "benchmark my adsorption set",
"fine-tune SevenNet on my data". Every push to `main` is a new plugin version
(`/plugin` → Marketplaces → Enable auto-update). Details:
[`docs/claude_plugin.md`](docs/claude_plugin.md).

## Install & environments

```bash
git clone https://github.com/JinukMoon/oh-my-mlip.git && cd oh-my-mlip
source env.sh
./install.sh MACE                                   # build from envs/mace.yml
python run_examples/single_point.py MACE            # energy + forces on your GPU
python scripts/setup_verify.py MACE-MPA-0 --json    # the completion check
```

- **Recipes, not improvisation.** `envs/<env>.yml` (+ `.build.sh` where needed)
  pins the package set; `install.sh` runs the steps in a fixed order —
  create env, catbench, weight preparation, first-use compilation
  (NequIP with OpenEquivariance, Allegro with CuEquivariance), D3.
- **Exact replay.** `envs/locks/<env>.{conda,pip}.txt` record the full
  package set of a verified build; `OMM_USE_LOCK=1 ./install.sh <env>`
  reinstalls exactly that set (no dependency resolution).
- **Fresh-environment proof.** `scripts/setup_sweep.py --fresh-root` builds
  each env in a disposable root, verifies every variant on the GPU, preserves
  the evidence and removes the root.
- **Bring your own env.** `scripts/adopt_env.py MACE ~/miniconda3/envs/MACE`.

```python
import oh_my_mlip
from ase.build import bulk

out = oh_my_mlip.run("MACE", bulk("Cu", "fcc", a=3.61, cubic=True))
print(out["energy"])
```

`resolve(model)` returns the exact interpreter, import and calculator lines to
paste into your own scripts. Per-framework procedures:
[`docs/recipes.md`](docs/recipes.md); whole workflows: [`recipes/`](recipes/).

## Benchmark (catbench)

Every env ships [catbench](https://github.com/JinukMoon/catbench) (adsorption
energies, D3 included).

```bash
cd my_benchmark/
python <repo>/run_examples/catbench_quickstart.py <TAG> --only MACE,SevenNet   # fetches TAG if missing
python <repo>/run_examples/catbench_quickstart.py <TAG> --all-versions --slurm --emit-only
python <repo>/scripts/catbench_report.py --result ./result --out ./report
```

- One job file per model (`jobs/catbench_<MLIP>.py` + `.sh`), rerunnable as-is.
- Your own VASP calculations: `scripts/catbench_vasp_stage.py` copies them
  (originals never touched) and converts with catbench's `vasp_preprocessing`.
- The report adds MAE tables and catbench's threshold-sensitivity analysis;
  official leaderboard values are fetched for comparison, never recomputed.

Full procedure: [`recipes/catbench.md`](recipes/catbench.md).

## Fine-tune

```bash
python scripts/ft_run.py MACE --dataset my_frames.traj --out ft_mace
python scripts/ft_verify.py ft_mace/<checkpoint> --model MACE --json
```

Datasets are anything ASE reads; each framework's upstream trainer, config and
checkpoint handling are in [`docs/finetune.md`](docs/finetune.md). Licenses of
non-commercial checkpoints are shown before training.

## Distill

```bash
python scripts/distill_bootstrap.py --teacher MACE-MPA-0 --structure slab.vasp --work ./distill
cd distill && sh run_distill.sh
```

A teacher MLIP labels structures for a small NN-MTP student that runs in
LAMMPS on CPUs, in an active-learning loop driven by the separate GPL-2.0
project [`onthefly-distill`](https://github.com/JinukMoon/onthefly-distill)
(invoked, never copied in).

## Supported MLIPs

Status, licenses and host requirements:
[`docs/model_status.md`](docs/model_status.md),
[`docs/model_licenses.md`](docs/model_licenses.md),
[`docs/host_requirements.md`](docs/host_requirements.md).

<!-- STATUS_TABLE_START -->
| Framework | Models |
|---|---|
| SevenNet | SevenNet-MF-OMPA, SevenNet-Omni |
| MACE | MACE-MPA-0, MACE-MH-1-OMAT, MACE-MH-1-OC20 |
| NequIP | NequIP-OAM-XL, NequIP-OAM-L |
| Allegro | Allegro-OAM-L |
| Nequix | Nequix-MP-1 |
| DeePMD | DPA-3.1-3M-FT |
| ORB | ORB-v3 |
| GRACE | GRACE-2L-OAM |
| MatterSim | MatterSim-v1-5M |
| CHGNet | CHGNet-v0.3.0 |
| AlphaNet | AlphaNet-v1-OMA |
| Eqnorm | Eqnorm-MPtrj |
| fairchemv1 | eSEN-30M-OAM |
| EquiformerV3 | EqV3-OMatMPtrjSalex |
| UMA | UMA-m-1p1-OC20, UMA-m-1p1-OMAT, UMA-s-1p1-OC20, UMA-s-1p1-OMAT, UMA-s-1p2-OC20, UMA-s-1p2-OC22, UMA-s-1p2-OC25, UMA-s-1p2-OMAT |
| PET | PET-OAM-XL |
| EquFlash | EquFlashV2, EquFlash |
| MatRIS | MatRIS-10M-OAM |
| DPA4 | DPA-4.0.1-pro-MPtrj |
| TACE | TACE-OAM-L |
<!-- STATUS_TABLE_END -->

## Gated models

UMA and eSEN weights are gated on Hugging Face. Accept the license on the
model page with your own account, then run `huggingface-cli login` (or point
`HF_TOKEN_PATH` at a token file). Never paste a token into a chat or a command
line. Weights are always fetched with your token and never hosted here —
see [`docs/hf_token.md`](docs/hf_token.md).

## MCP server

```bash
pip install -r requirements-mcp.txt && source env.sh && python -m oh_my_mlip.mcp_server
```

## Contributing & license

A new framework is one `models.json` entry plus an env recipe — see
[`CONTRIBUTING.md`](CONTRIBUTING.md). MIT licensed ([`LICENSE`](LICENSE));
frameworks and weights keep their upstream licenses
([`docs/model_licenses.md`](docs/model_licenses.md)).
