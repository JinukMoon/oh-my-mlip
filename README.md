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

**Documentation:** https://jinukmoon.github.io/oh-my-mlip/

| | Feature | Command |
|---|---|---|
| 1 | [Install & environments](#install--environments) | `/oh-my-mlip:setup MACE` |
| 2 | [Benchmark (catbench)](#benchmark-catbench) | `/oh-my-mlip:catbench` |
| 3 | [Fine-tuning](#fine-tuning) | `/oh-my-mlip:finetune` |
| 4 | [Distillation](#distillation) | `/oh-my-mlip:distill` |

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
./install.sh MACE                                  # its own conda env, from envs/mace.yml
python scripts/setup_verify.py MACE-MPA-0 --json   # energy + forces on your GPU
```

Every framework gets its own env (their torch/CUDA stacks conflict), built from a
pinned recipe. `OMM_USE_LOCK=1 ./install.sh <env>` replays a verified build
exactly (`envs/locks/`); an env you already have can be adopted with
`scripts/adopt_env.py MACE <prefix>`. Details: [`recipes/setup.md`](recipes/setup.md).

## Use a model

**In your own script** (MD, relax, anything long) — copy the lines once, run with
that env's interpreter; oh-my-mlip is not needed at run time:

```python
import sys; sys.path.insert(0, "<repo>")        # or $OH_MY_MLIP_HOME
from oh_my_mlip import resolve
spec = resolve("MACE")                          # or a version, e.g. "MACE-MH-1-OMAT"
print(spec["python"], spec["imports"], spec["inference"], spec["env_run"])
```

```python
# my_run.py  ->  <spec["python"]> my_run.py   (export spec["env_run"] first, if any)
from ase.io import read
from mace.calculators import mace_mp            # spec["imports"], verbatim
atoms = read("POSCAR")
calc = mace_mp(model='medium-mpa-0', dispersion=False, default_dtype='float64', device='cuda')  # spec["inference"]
atoms.calc = calc
```

**Several models from one script** (quick comparisons) — each model runs in its
own env process, so the calling Python needs no model installed:

```python
import oh_my_mlip
out = oh_my_mlip.run("MACE", atoms)             # one call -> {"energy": ..., "forces": ...}
with oh_my_mlip.WorkerPool() as pool:           # many calls: workers stay alive
    for name in ("MACE", "SevenNet"):
        print(name, pool.request(name, atoms)["results"]["energy"])
```

Every call crosses a process boundary — use the first way for MD loops.
Per-framework lines and weights: [`docs/recipes.md`](docs/recipes.md).

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

## Fine-tuning

```bash
python scripts/ft_run.py MACE --dataset my_frames.traj --out ft_mace
python scripts/ft_verify.py ft_mace/<checkpoint> --model MACE --json
```

Every framework fine-tunes differently (CLI flags, YAML or JSON configs, its
own dataset layout). One command converts an ASE-readable dataset, writes the
framework's own config and command, and runs the upstream trainer. Per-framework
details: [`docs/finetune.md`](docs/finetune.md); licenses of non-commercial
checkpoints are shown before training.

## Distillation

```bash
python scripts/distill_bootstrap.py --teacher MACE-MPA-0 --structure slab.vasp --work ./distill
cd distill && sh run_distill.sh
```

Distill any hub model into an **NN-MTP** student: a compact potential with a
ready LAMMPS pair style, so the student runs large MD on CPUs with no Python or
LibTorch. Any installed model can be the teacher; the active-learning loop is
the separate GPL-2.0 project
[`onthefly-distill`](https://github.com/JinukMoon/onthefly-distill) (invoked,
never copied in).

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
