# oh-my-mlip

> Install-free, agent-native access to 20 machine-learning interatomic-potential
> frameworks (31 model variants). One convention to run them all — catbench
> adsorption benchmarking bundled by its author.

## What this is

`oh-my-mlip` is a **convenience service**, not a model or a benchmark verdict.
It removes the one thing that actually stops people from using many MLIP
frameworks at once: **the install / environment-solving layer.** Every framework
ships as its own validated conda environment; a single trusted `models.json`
registry is the source of truth; and a small path-importable Python package
(`oh_my_mlip`) turns that registry into a uniform way to run any model — each in
its own interpreter, with no `conda activate`, no dependency conflicts, and no
GPU lock-in between frameworks.

What you get:

- **Run 20 frameworks / 31 model variants painlessly** — clone the hub, point `OH_MY_MLIP_HOME` at it,
  and call one model after another from the same convention. The heavy
  per-framework environments are reproduced from prebuilt, relocatable
  conda-pack tarballs (or rebuilt locally by `install.sh`).
- **catbench, easily** — catbench (adsorption / heterogeneous-catalysis
  benchmarking, authored by the project owner) is pre-wired into every env. You
  bring your own DFT/reference data; the wiring is done.

It deliberately does **not** auto-select a model for you: it gives you easy
multi-MLIP access plus catbench wiring plus guidance, and *you* supply the data
and judge the results. It also does **not** reimplement any model — it packages
the real upstream frameworks (MACE, SevenNet, NequIP, ORB, UMA, …), not rewrites
of them.

Distillation is **out of scope for v1**: a future, *separate* tool can bind to
the stable teacher-provider interface exposed here as an optional Phase-2 on-ramp
(cheap bulk teacher labeling to train a CPU-deployable NN-MTP/LAMMPS student). If
you only want to run MLIPs or catbench, you never touch it.

**Tagline:** *One registry, many MLIPs — install nothing, solve no
environments, just run.*

## Quickstart

`oh_my_mlip` is **path-importable, not a pip package**: you put the clone root on
`sys.path` (or set `OH_MY_MLIP_HOME`) and import it. `source env.sh` once per
shell sets `OH_MY_MLIP_HOME`, the shared caches, and the D3/CUDA environment.

```bash
# 1. Clone the hub and enter it (replace the URL with this repo's clone URL).
git clone <this-repo-url> oh-my-mlip
cd oh-my-mlip

# 2. Set OH_MY_MLIP_HOME (autodetected from the clone), caches, and D3/CUDA env.
source env.sh                       # sets OH_MY_MLIP_HOME, caches, D3/CUDA env

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

Or from the shell, using the bundled example:

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
| `oh_my_mlip.Worker` / `WorkerPool` | cross-env | persistent per-env worker for many repeated calls (e.g. a relaxation or AL loop) |

For gated models (e.g. UMA) export `HF_TOKEN` and accept the upstream license
first — see [Gated models](#gated-models).

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
source env.sh                       # sets OH_MY_MLIP_HOME, caches, D3/CUDA env
python -m oh_my_mlip.mcp_server     # serves over stdio
```

Tools exposed:

| Tool | Maps to | Runtime |
|---|---|---|
| `list_models` | `list_models()` (+ versions) | GPU-free (pure registry read) |
| `describe_model` | `resolve(model, version)` codegen dict | GPU-free |
| `model_status` | the `## Models & status` table data | GPU-free |
| `run_singlepoint` | `run(model, atoms, …)` energy + forces | GPU runtime |
| `run_relax` | persistent `Worker` + ASE `BFGS` | GPU runtime |
| `install_model` | `fetch.fetch_env(model)` (download + relocate env) | HF / compute runtime |
| `run_catbench` | the catbench roster runner | GPU runtime |

The three GPU-free tools work now from any host. The `run_*` / `install_model`
tools execute at **GPU runtime** once the model's conda env is materialized — they
are validated end-to-end at the compute checkpoint; if an env is not installed yet
they return an actionable message pointing at `install_model` / `install.sh`
rather than a traceback. `run_singlepoint` / `run_relax` accept a structure as a
file path, an `Atoms.todict()` dict, or a simple `{symbols, positions, cell?, pbc?}`
dict.

## Distinct from peers

We solve the **install / environment layer** that the neighbors all assume away,
package the **real upstream frameworks** (not reimplementations), bundle
**catbench catalysis** benchmarking, and ship an **agent-native** interface
(`AGENTS.md` + an MCP server) that no competitor has.

| Project | Solves install/env layer? | Packages originals vs reimplements? | catbench catalysis? | Agent-native? |
|---|---|---|---|---|
| **oh-my-mlip** | **yes** — relocatable per-model conda envs + `install.sh` fallback | **packages originals** (MACE, SevenNet, NequIP, ORB, UMA, …) | **yes** — pre-wired, bring your own data | **yes** — `AGENTS.md` + MCP server |
| `basf/mlipx` | no — assumes the MLIP is already installed | uses originals (eval/compare + ZnDraw viz) | no (general materials eval) | no |
| `instadeepai/mlipaudit` | no — assumes calculators exist | uses originals (benchmark suite + leaderboard) | no (molecular / bio: proteins, peptides, liquids) | no |
| `instadeepai/mlip` | partial — one JAX env, but JAX lock-in | **reimplements** MACE/NequIP/ViSNet/eSEN in JAX | no | no |
| `amaceing-toolkit` | partial — 6 models in isolated envs + input-file gen | uses originals (materials / MD) | no | no |

## Models & status

This table is **generated from `models.json`** by
`scripts/gen_status_table.py` (CI runs `--check` to keep it byte-for-byte in
sync — the README can never drift from the registry, which is the guard against
an "all validated" overclaim). The `Validation` column is each model's true
per-GPU validation state from the registry; the **`v1 tarball`** column marks the
frameworks whose relocatable conda-pack distribution is authored for v1 —
**exactly MACE and SevenNet**, shown as `upload-pending` because their tarballs
are not yet uploaded (the build+publish and the binding foreign-host end-to-end
run are the deferred compute checkpoint). Everything else is a `Phase 2`
packaging target. Rows marked `gpu pending` are env/load-verified only.

<!-- STATUS_TABLE_START -->
| Model | Framework | Weights | Validation | Gated | v1 tarball |
|---|---|---|---|---|---|
| SevenNet-MF-OMPA | SevenNet | bundled | validated (sm89) | no | upload-pending |
| SevenNet-Omni | SevenNet | bundled | validated (sm89) | no | upload-pending |
| MACE-MPA-0 | MACE | bundled | validated (sm89) | no | upload-pending |
| MACE-MH-1-OMAT | MACE | bundled | validated (sm89) | no | upload-pending |
| MACE-MH-1-OC20 | MACE | bundled | validated (sm89) | no | upload-pending |
| NequIP-OAM-XL | NequIP | on-demand-hf | validated (sm89) | no | Phase 2 |
| NequIP-OAM-L | NequIP | on-demand-hf | validated (sm89) | no | Phase 2 |
| Allegro-OAM-L | Allegro | on-demand-hf | gpu pending | no | Phase 2 |
| Nequix-MP-1 | Nequix | on-demand-hf | validated (sm89) | no | Phase 2 |
| DPA-3.1-3M-FT | DeePMD | on-demand-hf | validated (sm89) | no | Phase 2 |
| ORB-v3 | ORB | auto-download | validated (sm89) | no | Phase 2 |
| GRACE-2L-OAM | GRACE | on-demand-hf | validated (sm89) | no | Phase 2 |
| MatterSim-v1-5M | MatterSim | bundled | validated (sm89) | no | Phase 2 |
| CHGNet-v0.3.0 | CHGNet | bundled | validated (sm89) | no | Phase 2 |
| AlphaNet-v1-OMA | AlphaNet | on-demand-hf | gpu pending | no | Phase 2 |
| Eqnorm-MPtrj | Eqnorm | auto-download | validated (sm89) | no | Phase 2 |
| eSEN-30M-OAM | fairchemv1 | on-demand-hf | validated (sm89) | no | Phase 2 |
| EqV3-OMatMPtrjSalex | EquiformerV3 | on-demand-hf | gpu pending | no | Phase 2 |
| UMA-m-1p1-OC20 | UMA | on-demand-hf | validated (sm89) | yes | Phase 2 |
| UMA-m-1p1-OMAT | UMA | on-demand-hf | validated (sm89) | yes | Phase 2 |
| UMA-s-1p1-OC20 | UMA | on-demand-hf | validated (sm89) | yes | Phase 2 |
| UMA-s-1p1-OMAT | UMA | on-demand-hf | validated (sm89) | yes | Phase 2 |
| UMA-s-1p2-OC20 | UMA | on-demand-hf | validated (sm89) | yes | Phase 2 |
| UMA-s-1p2-OC22 | UMA | on-demand-hf | validated (sm89) | yes | Phase 2 |
| UMA-s-1p2-OMAT | UMA | on-demand-hf | validated (sm89) | yes | Phase 2 |
| PET-OAM-XL | PET | on-demand-hf | gpu pending | no | Phase 2 |
| EquFlashV2 | EquFlash | on-demand-hf | validated (sm89) | no | Phase 2 |
| EquFlash | EquFlash | on-demand-hf | validated (sm89) | no | Phase 2 |
| MatRIS-10M-OAM | MatRIS | auto-download | gpu pending | no | Phase 2 |
| DPA-4.0.1-pro-MPtrj | DPA4 | on-demand-hf | gpu pending | no | Phase 2 |
| TACE-OAM-L | TACE | auto-download | gpu pending | no | Phase 2 |
<!-- STATUS_TABLE_END -->

## How distribution works

After a fresh clone the `envs/<env>/` directories do not exist yet. They are
materialized on first use, and either path yields the identical on-disk layout
`$OH_MY_MLIP_HOME/envs/<env>/bin/python`:

1. **conda-pack tarballs (primary).** Each env is packed once and hosted on the
   Hugging Face Hub. `dist_manifest.json` maps an env to its tarball (repo id +
   pinned revision + sha256 + unpack size + minimum NVIDIA driver). The resolver
   in `oh_my_mlip/fetch.py` (`fetch_env`) downloads it, verifies the sha256, runs
   `conda-unpack` once (sentinel-guarded), then probes `torch.cuda`. When the
   manifest records a `min_driver_version` (it carries the `TODO-on-upload`
   placeholder until upload, which skips the check), the resolver compares it
   against the host CUDA driver and, on a host below it, prints the exact
   `install.sh` fallback command rather than a raw traceback.
2. **`install.sh` (fallback).** When no tarball is publishable for an env (its
   manifest entry still carries the `TODO-on-upload` marker) or the prebuilt env
   cannot run on this host, `install.sh <env>` rebuilds it from a recipe (create
   env + catbench + first-run compiles).
3. **Gated weights on-demand.** The published tarballs contain the **environment
   only** — never gated weights. Gated weights are fetched on first run with the
   user's own `HF_TOKEN` after the user accepts the upstream license (see
   [Gated models](#gated-models)).
4. **Arch-specific artifacts compile on first run.** Architecture-pinned
   artifacts — the D3 CUDA kernel `pair_d3.so` and the NequIP/Allegro AOT `.pt2`
   — are **never baked into the distributed tarballs**. They are compiled or
   reselected for *your* GPU's compute capability (sm86 = A5000/A6000, sm89 =
   L40S) the first time you run. See
   [`docs/arch_first_run_compile.md`](docs/arch_first_run_compile.md).

## Gated models

Most of the roster is open-weight and needs no token. A few models are
**gated** (in the v1 roster, all **UMA** variants): their weights sit behind an
upstream license you must accept with your own Hugging Face account.
**oh-my-mlip never redistributes gated weights** — they are always fetched on
first run with *your* token, after *you* accept the license.

The flow, one time per model + machine:

1. Accept the license at the model's `license_url`
   (`https://huggingface.co/facebook/UMA` for UMA) while logged into Hugging
   Face with the account whose token you will use.
2. Export a read token: `export HF_TOKEN=hf_...` (`env.sh` intentionally does
   **not** set this, so no token is ever baked into the repo or a shared cache).
3. Run normally — the first call downloads the weights into the shared cache.

If `HF_TOKEN` is missing or the license has not been accepted, the fetch fails
**by design**: the resolver surfaces the `license_url` and stops rather than
working around the gate. Full details and the agent contract are in
[`docs/gated_models.md`](docs/gated_models.md).

## Roadmap

- **v1 (now).** Small, focused core: the **MACE + SevenNet** distribution
  pipeline is authored and ready (relocatable conda-pack → fetch → relocate →
  single-point + catbench), with the tarballs **upload-pending the compute
  checkpoint** (build+publish and the binding foreign-host end-to-end run have
  not executed yet). Every framework is carried in `models.json` at its true
  validation state and runnable from a local `install.sh` build.
- **Phase 2.** Expand the conda-pack/`install.sh` path to the **full 31-model**
  roster; and open the **teacher-query on-ramp** for distillation. The
  **`oh-my-mlip` MCP server** (`list_models`, `describe_model`, `model_status`,
  `install_model`, `run_singlepoint`, `run_relax`, `run_catbench`) — the headline
  agent-native differentiator — is **already included** (see the [MCP
  server](#mcp-server) section); its GPU-free tools work now and its `run_*` /
  `install_model` tools are validated end-to-end at the compute checkpoint. The
  downstream distillation tool (cheap bulk teacher labeling → CPU-deployable
  NN-MTP/LAMMPS student) lands **separately** on its own timeline and binds to
  the stable teacher-provider interface (`get_calculator` intra-env,
  `Worker`/`WorkerPool` cross-env) exposed here. The binding 100-call worker
  acceptance test has **not** run yet, so this is described as a *planned*
  on-ramp: **teacher-query on-ramp planned (Phase 2)**. See
  [`docs/distillation_onramp.md`](docs/distillation_onramp.md).
- **Phase 3.** Docker / Apptainer images and CI (including the generated
  status-table `--check` and the foreign-host relocation acceptance test).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full guide.

`models.json` is treated as **code-equivalent / trusted** — its `import` and
`inference` lines are `exec`'d inside each model's env, and its `env_run` strings
are applied as subprocess environments through a strict key=value allowlist
(`oh_my_mlip.registry.parse_env_run`). Changes to it are therefore gated behind
review exactly like code. When you add or change a model, regenerate the status
table so CI's `python scripts/gen_status_table.py --check` stays green:

```bash
python scripts/gen_status_table.py > /tmp/table.md
# replace the block between the STATUS_TABLE markers in README.md with /tmp/table.md
```

## Further reading

- [`AGENTS.md`](AGENTS.md) — the agent guide (read this if you are an LLM asked
  to use a model).
- [`docs/gated_models.md`](docs/gated_models.md) — gated-weight license + token
  flow.
- [`docs/arch_first_run_compile.md`](docs/arch_first_run_compile.md) — first-run
  D3 `.so` and NequIP/Allegro `.pt2` compilation per GPU arch.
- [`docs/distillation_onramp.md`](docs/distillation_onramp.md) — the (Phase-2,
  separate-tool) teacher-query on-ramp and the frozen worker contract.
