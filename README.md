# oh-my-mlip

[![CI (GPU-free)](https://github.com/JinukMoon/oh-my-mlip/actions/workflows/ci.yml/badge.svg)](https://github.com/JinukMoon/oh-my-mlip/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> oh-my-mlip won't pick the best MLIP for you — but you'll never solve a conda
> environment by hand again.

**One registry, many MLIPs.** 20 machine-learning interatomic-potential
frameworks (31 model variants) catalogued behind one convention. Each framework
lives in its own validated conda env, built from a curated recipe, with
[catbench](https://github.com/JinukMoon/catbench) adsorption benchmarking
pre-wired and an **agent-native** surface (`AGENTS.md` + a Claude Code plugin +
an MCP server).

`oh-my-mlip` is a **convenience layer**, not a model and not a benchmark verdict.
It removes the one thing that actually stops people from using many MLIP
frameworks at once: **the install / environment-solving layer.** It does not pick
a model for you, and it never reimplements a model — it packages the real
upstream frameworks (MACE, SevenNet, NequIP, ORB, UMA, …).

## Quickstart — let an agent do everything (Claude Code)

This is an **agent-first** hub, and the agent path is the quickstart. The repo
doubles as a self-serve Claude Code marketplace — two commands, once
(enter them one at a time; the first is the marketplace source, the second is
the install):

```
/plugin marketplace add JinukMoon/oh-my-mlip
```

```
/plugin install oh-my-mlip@oh-my-mlip
```

The plugin installs at **user scope** — the skills work from any directory, in
any project; the hub itself lives in `~/.oh-my-mlip` (or `$OH_MY_MLIP_HOME`).
From then on, in any Claude Code session:

```
/oh-my-mlip:setup MACE               # one model
/oh-my-mlip:setup MACE SevenNet ORB  # several models, one command
/oh-my-mlip:setup all                # the whole registry (gated models are
                                     # skipped with instructions)
```

Natural language works too — "which MLIPs can I install?" lists the registry
roster (a pure read, nothing is installed), and "relax this POSCAR with an
MLIP" or "install every MLIP you support" routes to the same skills. Each
setup clones the hub if needed, builds the env, runs any first-use GPU
compilation, and verifies energy + forces on your GPU before reporting back.
Zero manual steps. Details:
[`docs/claude_plugin.md`](docs/claude_plugin.md). (Tool-calling agents can use
the [MCP server](#mcp-server) instead.)

### Updating

The plugin has no version pins — **every push to `main` is a new version**
(identified by its git commit). Third-party marketplaces have auto-update
**off** by default in Claude Code, so either enable it once
(`/plugin` → Marketplaces → **Enable auto-update**) or pull updates explicitly,
from your terminal:

```bash
claude plugin update oh-my-mlip@oh-my-mlip
```

(or in-session: `/plugin` → **Installed** → oh-my-mlip → Update.)

Updates never apply to a running session — run `/reload-plugins` (or start a
new session) to load the new version. A manually cloned hub updates with plain
`git pull`.

### Manual quickstart (no agent required)

`oh_my_mlip` is **path-importable, not a pip package**. `source env.sh` once per
shell sets `OH_MY_MLIP_HOME`, the shared caches, and the D3/CUDA environment.

```bash
git clone https://github.com/JinukMoon/oh-my-mlip.git
cd oh-my-mlip
source env.sh

# List registered models — pure registry read, no GPU / model env needed.
python -c "import oh_my_mlip; print(oh_my_mlip.list_models())"

# Build a model's env from its curated recipe (one-time, on your host).
./install.sh MACE

# Several at once — or everything: no argument builds every recipe.
# Budget ~5-10 GB of disk per env; gated envs need your own HF token
# (docs/hf_token.md) and are skipped without one.
./install.sh MACE SevenNet ORB
./install.sh

# Single-point energy + forces (spawns the right env interpreter for you).
python run_examples/single_point.py MACE
```

```python
import oh_my_mlip
from ase.build import bulk

atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
out = oh_my_mlip.run("MACE", atoms, properties=("energy", "forces"))
print(out["energy"], out["forces"][0])
```

### API surface (small and stable)

| Call | Layer | Use |
|---|---|---|
| `list_models()` | registry | enumerate registered frameworks |
| `resolve(model, version=None)` | registry | codegen dict (env `python`, `imports`, `inference`, flags) — no model loaded |
| `get_calculator(model, ...)` | intra-env | build an ASE `Calculator` from inside that model's own env |
| `run(model, atoms, ...)` | cross-env | one-shot compute; spawns the right env interpreter |
| `Worker` / `WorkerPool` | cross-env | persistent per-env worker for many repeated calls |

## Why oh-my-mlip?

The name is a promise, in the [oh-my-zsh](https://github.com/ohmyzsh/ohmyzsh)
sense: a framework whose whole job is removing setup pain. Excellent
neighboring tools exist, and nearly all of them **assume the hard part — a
working environment for each framework — is already solved**:

- **Evaluation suites** (mlipx, MLIP Arena, mlipaudit) benchmark models you
  have already installed.
- **Reimplementation stacks** (e.g. JAX rewrites of popular architectures)
  unify frameworks by rewriting them — the opposite strategy; oh-my-mlip only
  ever packages the real upstream code, so you compute with the authors' own
  implementation.
- **Single-backend unifiers** standardize on one tensor stack; oh-my-mlip
  isolates instead (one env per framework), so PyTorch, TensorFlow, and JAX
  frameworks ride along unchanged and never fight over CUDA pins.

oh-my-mlip is the layer those tools stand on: curated, validated envs behind
one registry, with an agent that does the installing — plus
[catbench](https://github.com/JinukMoon/catbench) pre-wired the moment an env
exists. Your favorite MLIP tool is welcome on top of it
([Contributing](#contributing)).

## Supported MLIPs

20 frameworks / 31 model variants, install-verified on **two independent hosts**:
the maintainer's RTX 4060 Ti (sm89, CUDA 12.x driver) and a third-party
RTX A4500 login node (sm86, CUDA 13.0 driver) where 27/31 variants — including
the CUDA-13-only trio (dpa4/tace/matris) — computed energy + forces on the GPU
from a fresh clone. On a CUDA-12 driver that trio runs CPU-only and says so up
front. Every push also runs a **GPU smoke test** (MACE + SevenNet single-points
on a self-hosted runner) alongside the GPU-free CI. Separately,
17/20 envs also **bit-reproduce** our internal `/TGM` reference (the rest run fine
but aren't in that reference set, or are a public build that drifts). Per-model
state, gated flag, and licenses: [`docs/model_status.md`](docs/model_status.md),
[`docs/model_licenses.md`](docs/model_licenses.md); driver/CUDA floors and the
equivalence matrix: [`docs/host_requirements.md`](docs/host_requirements.md).

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
| UMA | UMA-m-1p1-OC20, UMA-m-1p1-OMAT, UMA-s-1p1-OC20, UMA-s-1p1-OMAT, UMA-s-1p2-OC20, UMA-s-1p2-OC22, UMA-s-1p2-OMAT |
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
  `install.sh --status` inspects install state, an interrupted env is
  import-verified and **adopted or rebuilt — never duplicated**, and a ready
  sentinel is only trusted after its imports re-verify. Per-model caveats live in
  `models.json` and [`docs/model_status.md`](docs/model_status.md).
- **Weights are never hosted here.** They download from each framework's official
  channel on first run — by name to a shared cache, from an official URL, or from
  Hugging Face with *your* own token for gated models. oh-my-mlip redistributes no
  weights.
- **Arch-pinned artifacts compile on your GPU.** The D3 CUDA kernel and the
  NequIP/Allegro AOT `.pt2` are compiled/reselected for your compute capability
  (auto-detected — sm86/sm89/anything newer) on first run — never shipped. See
  [`docs/arch_first_run_compile.md`](docs/arch_first_run_compile.md).
- **Self-healing install loop.** A bounded, agent-driven setup loop with hard
  stop conditions (disk headroom, signature stall, cumulative-attempt cap,
  wall-clock) — the policy is the single source of truth in `AGENTS.md` §8.
- **Relocatable env tarballs (rolling out).** The conda-pack pipeline is
  validated end-to-end — a packed env unpacked at a foreign prefix reproduces
  the source env's energy bit-identically — and per-env tarballs are being
  published to the Hugging Face Hub via `dist_manifest.json` (recipes remain
  the always-available fallback).

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
