# oh-my-mlip

<p align="center">
  <img src="assets/logo.png" alt="oh-my-mlip — machine learning interatomic potentials" width="560">
</p>

[![CI (GPU-free)](https://github.com/JinukMoon/oh-my-mlip/actions/workflows/ci.yml/badge.svg)](https://github.com/JinukMoon/oh-my-mlip/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> oh-my-mlip won't pick the best MLIP for you — but you'll never fight a conda
> environment, a training config, or a benchmark script by hand again.

**One registry, many MLIPs.** 20 machine-learning interatomic-potential
frameworks (32 model variants) catalogued behind one convention. Each framework
lives in its own conda env, built from a curated recipe; every capability below
speaks the same language: facts live in `models.json`, procedures live in
generated docs, and everything that executes is a file on disk you can rerun
yourself — with or without an agent.

`oh-my-mlip` is a **convenience layer**, not a model and not a benchmark
verdict. It removes the things that actually stop people from using many MLIP
frameworks at once — the install layer, the per-framework training dialects,
the benchmarking boilerplate — and never reimplements a model: it packages and
drives the real upstream frameworks (MACE, SevenNet, NequIP, ORB, UMA, …).

## The four things it does

| # | Feature | One command |
|---|---------|-------------|
| 1 | [Install & environments](#1--install--environments) | `/oh-my-mlip:setup MACE` |
| 2 | [Run & benchmark](#2--run--benchmark-catbench) | `/oh-my-mlip:catbench` |
| 3 | [Fine-tune](#3--fine-tune) | `/oh-my-mlip:finetune` |
| 4 | [Distill](#4--distill) | `/oh-my-mlip:distill` |

More capabilities land the same way — as one skill, one registry extension,
one generated doc — so the list grows without the convention changing.

## Quickstart — let an agent do everything (Claude Code)

This is an **agent-first** hub. The repo doubles as a self-serve Claude Code
marketplace — two commands, once (enter them one at a time):

```
/plugin marketplace add JinukMoon/oh-my-mlip
```

```
/plugin install oh-my-mlip@oh-my-mlip
```

The plugin installs at **user scope** — the skills work from any directory; the
hub itself lives in `~/.oh-my-mlip` (or `$OH_MY_MLIP_HOME`). Natural language
works everywhere: "which MLIPs can I install?", "relax this POSCAR with an
MLIP", "benchmark my adsorption set", "fine-tune SevenNet on my dataset" all
route to the right skill. Details: [`docs/claude_plugin.md`](docs/claude_plugin.md).
Tool-calling agents can use the [MCP server](#mcp-server) instead. Prefer
typing commands yourself? Every feature below is driven by plain scripts —
the agent is optional by design.

### Updating

The plugin has no version pins — **every push to `main` is a new version**
(identified by its git commit). Enable auto-update once (`/plugin` →
Marketplaces → **Enable auto-update**) or pull explicitly:
`claude plugin update oh-my-mlip@oh-my-mlip`, then `/reload-plugins`.
A manually cloned hub updates with plain `git pull`.

---

## 1 — Install & environments

Environment-solving is the tax every MLIP user pays first. Here it is paid
once, in a reviewed recipe, instead of per person per machine.

```
/oh-my-mlip:setup MACE                  # one model — zero prompts
/oh-my-mlip:setup MACE SevenNet ORB     # several models, one command
/oh-my-mlip:setup all                   # the full registry, after your approval
```

Each setup clones the hub if needed, builds the env from its curated recipe,
runs any first-use GPU compilation, and verifies energy + forces on your GPU
before reporting back. `install.sh` exiting cleanly is treated as necessary,
never sufficient — the only completion test is a real computation.

No agent required:

```bash
git clone https://github.com/JinukMoon/oh-my-mlip.git
cd oh-my-mlip
source env.sh

python -c "import oh_my_mlip; print(oh_my_mlip.list_models())"   # registry read
./install.sh MACE                                                # build one env
python run_examples/single_point.py MACE                         # energy + forces
python scripts/setup_verify.py MACE-MPA-0 --json                 # the completion oracle
```

```python
import oh_my_mlip
from ase.build import bulk

atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
out = oh_my_mlip.run("MACE", atoms, properties=("energy", "forces"))
print(out["energy"], out["forces"][0])
```

**Bring your own envs.** Already have a working conda env for a framework?
Adopt it instead of rebuilding — zero disk, verified before it is trusted:

```bash
python3 scripts/adopt_env.py MACE ~/miniconda3/envs/MACE
python3 scripts/adopt_env.py --list
```

**API surface (small and stable).**

| Call | Layer | Use |
|---|---|---|
| `list_models()` | registry | enumerate registered frameworks |
| `resolve(model, version=None)` | registry | codegen dict (env `python`, `imports`, `inference`) — no model loaded |
| `get_calculator(model, ...)` | intra-env | an ASE `Calculator` inside that model's own env |
| `run(model, atoms, ...)` | cross-env | one-shot compute; spawns the right env interpreter |
| `Worker` / `WorkerPool` | cross-env | persistent per-env worker for many repeated calls |

Want the explicit procedure rather than the wrapper — which pip packages each
framework wants, the upstream command that fetches (and compiles) its
checkpoint, and the calculator line that loads it? That is
[`docs/recipes.md`](docs/recipes.md), generated from `models.json` so it
cannot drift.

## 2 — Run & benchmark (catbench)

Every env ships with [catbench](https://github.com/JinukMoon/catbench)
pre-wired, so the moment a model installs it can be benchmarked on adsorption
energies against your reference data — no per-model boilerplate.

```bash
# materialize + run one job file per model (the files are the actions):
python run_examples/catbench_quickstart.py MyDataset --only MACE,SevenNet
# jobs/catbench_<MLIP>.py        the exact per-model benchmark script
# jobs/run_catbench_<MLIP>.sh    the runner (env vars + interpreter) you can rerun

# on a SLURM cluster, emit sbatch-ready wrappers instead:
python run_examples/catbench_quickstart.py MyDataset --slurm --emit-only

# aggregate everything into one report (MAE table + comparison plot):
python scripts/catbench_report.py --result ./result --out ./report
```

Every executed command exists on disk before it runs, so a benchmark is
rerunnable — byte-identically — without the agent that launched it.

## 3 — Fine-tune

Foundation checkpoints are a starting point; your system usually deserves a
few epochs of its own data. Fine-tuning dialects differ wildly between
frameworks — CLI flags here, YAML keys there, a different dataset format
everywhere. The hub absorbs those differences.

```
/oh-my-mlip:finetune            # asks: which model, which dataset — then runs
```

```bash
# your dataset: anything ASE reads (extxyz, .traj, OUTCAR, ...) — converted
# automatically to whatever the chosen framework's trainer eats:
python scripts/ft_dataset.py --input my_frames.traj --to mace --out ft_data
# the exact training command, written to disk, then executed:
python scripts/ft_run.py MACE --dataset my_frames.traj --out ft_mace
# the completion oracle: the produced checkpoint loads and computes:
python scripts/ft_verify.py ft_mace/<checkpoint> --model MACE --json
```

Per-framework fine-tuning procedures — the exact upstream command, the config
it consumes, the dataset format, and how the pretrained checkpoint is
referenced — live in [`docs/finetune.md`](docs/finetune.md), generated from
the registry with one classification per variant. A model whose upstream
publishes no training path says so honestly rather than pretending.

Some checkpoints carry non-commercial licenses (for example CC BY-NC or
academic-use terms). Fine-tuning them is supported, and the hub prints the
license and its URL before starting — a derivative of a non-commercial
checkpoint inherits its terms, and that is worth knowing *before* training.

## 4 — Distill

A large, accurate teacher is expensive at MD time. Distillation turns it into
a tiny NN-MTP student that runs in LAMMPS on CPUs — no Python, no LibTorch —
validated against the teacher's own surface.

```
/oh-my-mlip:distill             # asks: which teacher, which material — then runs
```

The skill orchestrates the sibling project
[`onthefly-distill`](https://github.com/JinukMoon/onthefly-distill): it
renders the config, generates the teacher hook from the registry's own
calculator lines, and invokes the sibling's published entrypoints verbatim —

```bash
python scripts/distill_bootstrap.py --teacher MACE-MPA-0 --structure slab.vasp --work ./distill
cd distill && sh run_distill.sh     # teacher MD → student training → student LAMMPS MD, looped
scripts/build_lammps_nnmtp.sh       # one-time: LAMMPS with the NN-MTP pair style
```

The loop labels structures with the teacher, trains the student, runs the
student's own MD, and folds every failure back into training until the student
survives its target trajectory — an active-learning loop that hardens the
student exactly where it is weak. `onthefly-distill` is a separately-licensed
**GPL-2.0** sibling project; oh-my-mlip invokes it and copies nothing in.

---

## Supported MLIPs

Per-model state, gated flag, and licenses:
[`docs/model_status.md`](docs/model_status.md),
[`docs/model_licenses.md`](docs/model_licenses.md); driver/CUDA floors:
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

## How distribution works

- **Envs build from recipes.** `install.sh <model>` builds the env on the current
  host from a curated recipe (`envs/<env>.yml`, plus a `.build.sh` sidecar where a
  framework needs a pinned side-install). Re-runs are safe by construction:
  an interrupted env is import-verified and **adopted or rebuilt — never
  duplicated**, and a ready sentinel is only trusted after its imports re-verify.
- **Weights are never hosted here.** They download from each framework's official
  channel on first run — into that framework's own native cache, so anything you
  already downloaded is reused as-is and your `huggingface-cli login` keeps
  working for gated models (fetched with *your* token). Building a shared
  multi-user hub? Set `OMM_SHARED_CACHE_ROOT=/path` before sourcing `env.sh`.
  Each successful verification freezes the resolved interpreter/weight facts
  into the per-machine ledger `models.local.json`.
- **Arch-pinned artifacts compile on your GPU.** The D3 CUDA kernel and the
  NequIP/Allegro AOT `.pt2` are compiled/reselected for your compute capability
  (auto-detected — sm86/sm89/anything newer) on first run — never shipped. See
  [`docs/arch_first_run_compile.md`](docs/arch_first_run_compile.md).
- **Self-healing install loop.** A bounded, agent-driven setup loop with hard
  stop conditions (disk headroom, signature stall, cumulative-attempt cap,
  wall-clock) — the policy is the single source of truth in `AGENTS.md` §8.
- **Relocatable env tarballs.** Selected envs are published on the Hugging Face
  Hub with pinned revisions + sha256 (`dist_manifest.json`);
  `oh_my_mlip.fetch.fetch_env("MACE")` downloads, integrity-checks, and unpacks
  one. Recipes remain the always-available fallback. Author side, a release is
  one deterministic command with a hard pre-upload relocation gate
  (`scripts/release_env.sh`).

## Gated models

Most of the roster is open-weight. A few (e.g. all **UMA** variants, eSEN) are
**gated**: accept the upstream license with your own Hugging Face account, then
`export HF_TOKEN=hf_...` before running. `env.sh` never sets the token, so nothing
is baked into the repo or a shared cache. If the token is missing or the license
is unaccepted, the fetch fails **by design** and surfaces the `license_url`. See
[`docs/gated_models.md`](docs/gated_models.md).

## MCP server

The same registry is exposed as a [Model Context Protocol](https://modelcontextprotocol.io)
server — a thin adapter over the public `oh_my_mlip` API. The `mcp` SDK is an
optional extra (`import oh_my_mlip` never requires it):

```bash
pip install -r requirements-mcp.txt
source env.sh
python -m oh_my_mlip.mcp_server     # serves over stdio
```

Tools: `list_models`, `describe_model`, `model_status` (GPU-free) and
`run_singlepoint`, `run_relax`, `install_model`, `run_catbench` (compute runtime).

## Contributing

The registry (`models.json`) is the single source of truth, so adding a framework
is one data-shaped entry (its env, import line, calculator line) plus an env
recipe — no package internals to touch. See
[`CONTRIBUTING.md`](CONTRIBUTING.md). Suggestions for more integrations or faster
env setups are very welcome.

## License

See [`LICENSE`](LICENSE). Individual frameworks and their weights carry their own
upstream licenses — see [`docs/model_licenses.md`](docs/model_licenses.md).
`onthefly-distill`, orchestrated by the distill feature, is a separate GPL-2.0
project.
