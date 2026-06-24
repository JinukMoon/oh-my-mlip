# oh-my-mlip

> Won't pick a model for you — but it gives you one convention for reaching
> many MLIP frameworks, without solving a single conda environment by hand.

**One registry, many MLIPs.** 20 machine-learning interatomic-potential
frameworks (31 model variants) catalogued behind one convention — each meant to
live in its own validated env, built from a curated recipe, with catbench
adsorption benchmarking pre-wired by its author. **Building today:** MACE and
SevenNet ship as validated `install.sh` recipes now; the rest of the roster is
rolling out as recipes are added (see [How distribution works](#how-distribution-works)).

- [What this is](#what-this-is)
- [Philosophy](#philosophy)
- [Install & quickstart](#install--quickstart)
- [Supported MLIPs](#supported-mlips)
- [How distribution works](#how-distribution-works)
- [MCP server](#mcp-server)
- [Gated models](#gated-models)
- [Mental model](#mental-model)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

## What this is

`oh-my-mlip` is a **convenience service**, not a model and not a benchmark
verdict. It removes the one thing that actually stops people from using many MLIP
frameworks at once: **the install / environment-solving layer.**

Each framework is meant to live in its own validated conda environment, built
from a curated recipe. A single trusted `models.json` registry is the source of
truth. A small path-importable Python package (`oh_my_mlip`) turns that registry
into a uniform way to run any model — each in its own interpreter, with **no
`conda activate`, no dependency conflicts, and no GPU lock-in** between
frameworks.

That is the whole value, and it is value nobody else hands you in one place: the
install/env layer curated for **real upstream frameworks** (MACE, SevenNet,
NequIP, ORB, UMA, ...) — not reimplementations — with **catbench** catalysis
benchmarking wired into every env, and an **agent-native** surface (`AGENTS.md` +
an MCP server) so a tool-calling agent can drive the whole hub.

What you actually get:

| Capability | What it means |
|---|---|
| **Uniform access to 20 frameworks / 31 variants** | Clone the hub, point `OH_MY_MLIP_HOME` at it, and call one model after another from the same convention. Each per-framework env is built locally by `install.sh <model>` from a curated recipe (MACE and SevenNet build today; the rest are rolling out). |
| **catbench, easily** | catbench (adsorption / heterogeneous-catalysis benchmarking, authored by the project owner) is pre-wired into every env. You bring your own DFT/reference data; the wiring is done. |
| **Agent-native** | `AGENTS.md` is the agent contract, and the same registry is exposed over MCP. Hand the repo to an agent and it can run a model. |

It deliberately does **not** auto-select a model for you, and it does **not**
reimplement any model — it packages the real upstream frameworks. (See
[Mental model](#mental-model) for the full not-goals list.)

Distillation is **out of scope for now**: a future, *separate* tool can bind to
the stable teacher-provider interface exposed here as an optional, **planned
teacher-query on-ramp** — cheap bulk teacher labeling to train a CPU-deployable
NN-MTP/LAMMPS student. If you only want to run MLIPs or catbench, you never touch
it.

## Philosophy

`oh-my-mlip` is curated but open, and it is meant to **grow with the community**
— in the spirit of the other `oh-my-*` projects. The registry (`models.json`) is
the single source of truth, so a contribution is small, data-shaped, and easy to
review.

**Anyone can add an MLIP.** Adding a framework is one `models.json` entry — its
env, its import line, and the one validated calculator line — plus an env recipe.
That is the whole change; no package internals to touch. PRs are welcome.

We would also love your recommendations:

- **More integrations** — a framework or model you want supported. Suggest it,
  or open a PR with the `models.json` entry.
- **Faster / better env setups** — tighter version locks, quicker installs,
  cleaner GPU-arch handling. If your recipe for an env beats ours, send it.

Because the registry is the source of truth, every such contribution stays small
and reviewable. The how-to (the add-a-model steps and the trust policy that
governs `models.json`) lives in [Contributing](#contributing) and
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Install & quickstart

`oh_my_mlip` is **path-importable, not a pip package**: you put the clone root on
`sys.path` (or set `OH_MY_MLIP_HOME`) and import it. `source env.sh` once per
shell sets `OH_MY_MLIP_HOME`, the shared caches, and the D3/CUDA environment.

### Let an agent do it

Hand this to Claude Code (or any tool-calling agent) and let it bootstrap the hub
for you:

> Clone this repo into `oh-my-mlip` and `cd` into it. Read `AGENTS.md` — it is
> the agent contract for this hub. Then `source env.sh` to set
> `OH_MY_MLIP_HOME`, the shared caches, and the D3/CUDA environment. List the
> registered models with `python -c "import oh_my_mlip; print(oh_my_mlip.list_models())"`
> (this is a pure registry read — no GPU or model env needed). Finally, run a
> single-point energy/forces calculation with MACE via
> `python run_examples/single_point.py MACE`, and tell me the energy you got
> back. If MACE's env is not materialized yet, follow the actionable message it
> prints (`install.sh MACE`).

### Or do it by hand

```bash
# 1. Clone the hub and enter it.
git clone https://github.com/JinukMoon/oh-my-mlip.git
cd oh-my-mlip

# 2. Set OH_MY_MLIP_HOME (autodetected from the clone), caches, and D3/CUDA env.
source env.sh

# 3. List the registered models (no GPU / model env needed — pure registry read).
python -c "import oh_my_mlip; print(oh_my_mlip.list_models())"
```

Run a single-point calculation. `run()` is the cross-env convenience path: it
spawns the correct env interpreter for the model, computes, and returns a results
dict — you do **not** need to be inside the model's conda env:

```python
import oh_my_mlip
from ase.build import bulk

atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
out = oh_my_mlip.run("MACE", atoms, properties=("energy", "forces"))
print(out["energy"], out["forces"][0])
```

`run("MACE", atoms)` works with the version omitted — it falls back to the
model's `default_version`.

Or straight from the shell, using the bundled examples:

```bash
python run_examples/single_point.py MACE          # or SevenNet, ...
python run_examples/single_point.py MACE --d3      # with D3 dispersion
python run_examples/relax.py SevenNet --fmax 0.05  # persistent-Worker relaxation
```

The public API surface is small and stable:

| Call | Layer | Use |
|---|---|---|
| `oh_my_mlip.list_models()` | registry | enumerate registered frameworks |
| `oh_my_mlip.resolve(model, version=None)` | registry | get the codegen dict (env `python`, `imports`, `inference`, `env_run`, flags) — no model loaded |
| `oh_my_mlip.get_calculator(model, ...)` | intra-env | build an ASE `Calculator` **from inside that model's own env** |
| `oh_my_mlip.run(model, atoms, ...)` | cross-env | one-shot compute; spawns the right env interpreter for you |
| `oh_my_mlip.Worker` / `WorkerPool` | cross-env | persistent per-env worker for many repeated calls (e.g. a relaxation, or bulk labeling a structure set) |

For gated models (e.g. UMA) export `HF_TOKEN` and accept the upstream license
first — see [Gated models](#gated-models).

## Supported MLIPs

20 frameworks / 31 model variants. Some weights are gated (need a Hugging Face
token); some carry non-commercial or copyleft licenses — see the per-model
details below.

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

Per-model validation state, gated flag, and v1 distribution status: see
[`docs/model_status.md`](docs/model_status.md). Licenses: see
[`docs/model_licenses.md`](docs/model_licenses.md).

## How distribution works

After a fresh clone the `envs/<env>/` directories do not exist yet. They are
materialized on first use into the on-disk layout
`$OH_MY_MLIP_HOME/envs/<env>/bin/python`:

1. **`install.sh` build-from-recipe (primary today).** `install.sh <model>`
   builds the env on the current host from a curated recipe (`envs/<env>.yml`):
   create env + catbench + first-run compiles. It is host-correct by
   construction. Today only **MACE and SevenNet** have validated recipes; the
   other frameworks are being added as recipes and are not yet build-verified
   from a fresh clone. This is the documented path you should expect to use now.
2. **Weights are never hosted by oh-my-mlip.** The recipe builds the environment
   only — model weights download from each framework's **official channel** on
   first run: by name to a shared cache (`auto-download`), from an official URL,
   or from Hugging Face with *your* own `HF_TOKEN` after you accept the upstream
   license for gated weights (see [Gated models](#gated-models)). oh-my-mlip
   redistributes no weights.
3. **conda-pack tarballs (planned, not yet live).** The intended future
   convenience path is a relocatable, prebuilt env packed once and hosted on the
   Hugging Face Hub, resolved by `oh_my_mlip/fetch.py` (`fetch_env`) against
   `dist_manifest.json` (repo id + pinned revision + sha256 + unpack size +
   minimum NVIDIA driver). **This is not live:** every `dist_manifest.json` entry
   still carries the `TODO-on-upload` placeholder and nothing is hosted yet, so
   the resolver treats those entries as not-yet-publishable and points you at the
   `install.sh` build instead. When this ships it will yield the same on-disk
   layout as the recipe build.
4. **Arch-specific artifacts compile on first run, on your GPU.** Architecture-
   pinned artifacts — the D3 CUDA kernel `pair_d3.so` and the NequIP/Allegro AOT
   `.pt2` — are compiled or reselected for *your* GPU's compute capability
   (sm86 = A5000/A6000, sm89 = L40S) the first time you run (and would never be
   baked into any future tarball). See
   [`docs/arch_first_run_compile.md`](docs/arch_first_run_compile.md).

## MCP server

The same registry that drives `AGENTS.md` is also exposed as a
[Model Context Protocol](https://modelcontextprotocol.io) server
(`oh_my_mlip/mcp_server.py`), so a tool-calling agent can drive the hub directly.
It is a **thin adapter** — every tool forwards to the public `oh_my_mlip` API and
reimplements nothing.

The `mcp` SDK is an **optional extra** (it is *not* on the core import path —
`import oh_my_mlip` never requires it). Install it and launch the server over
stdio:

```bash
pip install -r requirements-mcp.txt
source env.sh
python -m oh_my_mlip.mcp_server     # serves over stdio
```

Tools exposed:

| Tool | Maps to | Runtime |
|---|---|---|
| `list_models` | `list_models()` (+ versions) | GPU-free (pure registry read) |
| `describe_model` | `resolve(model, version)` codegen dict | GPU-free |
| `model_status` | the detailed per-model status data (see [`docs/model_status.md`](docs/model_status.md)) | GPU-free |
| `run_singlepoint` | `run(model, atoms, ...)` energy + forces | GPU runtime |
| `run_relax` | persistent `Worker` + ASE `BFGS` | GPU runtime |
| `install_model` | `fetch.fetch_env(model)` (prebuilt-env fetch; falls back to the `install.sh` recipe build, the live path today) | compute runtime |
| `run_catbench` | the catbench roster runner | GPU runtime |

The three GPU-free tools work now from any host. The `run_*` / `install_model`
tools execute at **GPU runtime** once the model's conda env is built; if an env
is not installed yet they return an actionable message pointing at
`install_model` / `install.sh <model>` rather than a traceback.
`run_singlepoint` / `run_relax` accept a structure as a
file path, an `Atoms.todict()` dict, or a simple `{symbols, positions, cell?, pbc?}`
dict.

## Gated models

Most of the roster is open-weight and needs no token. A few models are **gated**
(in the v1 roster, all **UMA** variants): their weights sit behind an upstream
license you must accept with your own Hugging Face account. **oh-my-mlip never
redistributes gated weights** — they are always fetched on first run with *your*
token, after *you* accept the license.

The flow, one time per model + machine:

1. Accept the license at the model's `license_url`
   (`https://huggingface.co/facebook/UMA` for UMA) while logged into Hugging Face
   with the account whose token you will use.
2. Export a read token: `export HF_TOKEN=hf_...` (`env.sh` intentionally does
   **not** set this, so no token is ever baked into the repo or a shared cache).
3. Run normally — the first call downloads the weights into the shared cache.

If `HF_TOKEN` is missing or the license has not been accepted, the fetch fails
**by design**: the resolver surfaces the `license_url` and stops rather than
working around the gate. Full details and the agent contract are in
[`docs/gated_models.md`](docs/gated_models.md).

## Mental model

`oh-my-mlip` is opinionated about what it is *not*. It does not:

- **pick the best model for you** — it gives you uniform multi-MLIP access,
  catbench wiring, and guidance; *you* bring the data and judge the results.
- **reimplement any model** — it packages the real upstream frameworks (MACE,
  SevenNet, NequIP, ORB, UMA, ...), never rewrites of them.
- **be a pip library** — `oh_my_mlip` is path-importable; you put the clone root
  on `sys.path` (or `source env.sh`), and there is no `setup.py`.
- **redistribute gated weights** — gated weights are always fetched on first run
  with *your* token, after *you* accept the upstream license.
- **do distillation now** — that would land later as a *separate* tool binding
  to the planned teacher-query on-ramp.

What it *is*: one registry, many real MLIPs, each in its own validated env, with
catbench bundled and an agent-native surface on top.

## Roadmap

- **Now.** Small, focused core. **MACE and SevenNet** build today from validated
  `install.sh` recipes (create env + catbench + first-run compiles), with weights
  pulled from their official channels. Every framework is carried in `models.json`
  at its true validation state. The **MCP server** (`list_models`,
  `describe_model`, `model_status`, `install_model`, `run_singlepoint`,
  `run_relax`, `run_catbench`) is **already shipped** (see the
  [MCP server](#mcp-server) section): its GPU-free tools work now, and its `run_*`
  / `install_model` tools execute once a model's env is built.
- **Rolling out.** Add the remaining **recipes** so the full 20-framework /
  31-variant roster builds from a fresh clone via `install.sh`. Then the
  **conda-pack / instant-download path**: pack each validated env once, host it
  on the Hugging Face Hub, and fill `dist_manifest.json` so `fetch_env` can
  download + relocate a prebuilt env instead of rebuilding it (today that
  manifest is all `TODO-on-upload` and nothing is hosted).
- **Later.** Compile-command curation (per-arch first-run build recipes) and a
  **teacher-query on-ramp** for distillation. The downstream distillation tool
  (cheap bulk teacher labeling → CPU-deployable NN-MTP/LAMMPS student) would land
  **separately** on its own timeline and bind to the teacher-provider interface
  (`get_calculator` intra-env, `Worker`/`WorkerPool` cross-env) exposed here. This
  is a *planned* on-ramp, not yet built. See
  [`docs/distillation_onramp.md`](docs/distillation_onramp.md). Container images
  (Docker / Apptainer) and expanded CI follow.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide.

`models.json` is treated as **code-equivalent / trusted** — its `import` and
`inference` lines are `exec`'d inside each model's env, and its `env_run` strings
are applied as subprocess environments through a strict key=value allowlist
(`oh_my_mlip.registry.parse_env_run`). Changes to it are therefore gated behind
review exactly like code. Found a bug or want a model added? **Open an issue.**
When you add or change a model, regenerate the status table so CI's
`python scripts/gen_status_table.py --check` stays green:

```bash
python scripts/gen_status_table.py > /tmp/table.md
# replace the block between the STATUS_TABLE markers in README.md with /tmp/table.md
```

## Further reading

- [`AGENTS.md`](AGENTS.md) — the agent guide (read this if you are an LLM asked
  to use a model).
- [`docs/model_licenses.md`](docs/model_licenses.md) — every framework's GitHub +
  code/weights license, with the non-commercial / copyleft / gated ones flagged.
- [`docs/gated_models.md`](docs/gated_models.md) — gated-weight license + token
  flow.
- [`docs/arch_first_run_compile.md`](docs/arch_first_run_compile.md) — first-run
  D3 `.so` and NequIP/Allegro `.pt2` compilation per GPU arch.
- [`docs/distillation_onramp.md`](docs/distillation_onramp.md) — the (Phase-2,
  separate-tool) teacher-query on-ramp and the frozen worker contract.
