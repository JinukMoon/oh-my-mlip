# oh-my-mlip — Agent Guide

This file is written for an AI coding agent (Claude Code or any other LLM) that
has cloned this repo and is asked to **use** one or more MLIPs — to write a
single-point/relaxation script, run catbench, or wire an MLIP calculator into a
larger workflow. A human can follow it verbatim too.

> **This runs locally — no SLURM, no job scheduler, no cluster.** Every model is
> a plain Python interpreter call inside its own conda environment, executed on
> whatever machine you are on (laptop, workstation, single GPU box). There is
> nothing to submit to a queue.

## 0. Read these first (do NOT rely on memory — read them every time)

The two repo-root files are the single source of truth. Read them before
writing any code:

1. **`models.json`** — the model registry. For each model it gives the env name,
   the env's Python interpreter path, the `import` lines, the per-version
   `inference` (calculator-construction) code, `arch_pinned`/`gated` flags, and
   any `note`/`status`/`env_run` caveats. Resolve everything from here; never
   hard-code a model fact.
2. **`dist_manifest.json`** — maps each env to its conda-pack tarball on the
   Hugging Face Hub (repo id + pinned revision + sha256 + unpack size + minimum
   NVIDIA driver). The resolver in `oh_my_mlip/fetch.py` uses this to fetch and
   relocate an env on first use.

Also `source env.sh` once per shell before running anything — it sets the shared
model cache, the D3/CUDA environment, and the name-based model cache symlinks.

## 1. Hub layout

```
oh-my-mlip/                          # = $OH_MY_MLIP_HOME (clone root, autodetected by env.sh)
├── models.json                      # model registry (source of truth)
├── dist_manifest.json               # env -> HF tarball resolution
├── env.sh                           # shared cache + D3/CUDA env (source this)
├── oh_my_mlip/                      # the tiered Python interface (import by path, NOT pip)
├── envs/                            # conda env recipes (install.sh fallback)
├── models/                          # weights + caches (mostly auto-populated)
│   ├── compiled/{sm86,sm89}/        # arch-pinned .pt2 (recompiled on YOUR GPU; never shipped)
│   └── hf/ fairchem/ torch/ ...     # shared download caches
├── run_examples/                    # single_point.py / relax.py / catbench_quickstart.py
└── scripts/                         # author-side build/publish/verify tooling
```

After a fresh clone the `envs/<env>/` directories do not exist yet. They are
materialized on first use either by fetching a relocatable conda-pack tarball
(primary; `oh_my_mlip.fetch`) or by building from a recipe (`install.sh`,
fallback). Either way the on-disk layout is identical:
`$OH_MY_MLIP_HOME/envs/<env>/bin/python`.

## 2. The interface — use `oh_my_mlip`, not raw interpreter strings

Earlier conventions had agents construct the model's run script by hand and call
`<env>/bin/python` directly. The public repo replaces that with a **tiered
Python interface** in `oh_my_mlip/`. Pick the tier that matches what you are
doing:

| You want to… | Use | Notes |
|---|---|---|
| Get the resolved env/interpreter/import/inference for code generation | `resolve(model, version=None)` | Returns a machine-readable dict; the codegen contract. No model is loaded. |
| Build an ASE calculator **from inside that model's own env** | `get_calculator(model, version=None, device="cuda", apply_d3=False)` | **Intra-env only.** Precondition: the current interpreter is that env's interpreter. |
| Run a calculation casually **without managing envs** | `run(model, atoms, properties=("energy","forces"), device="cuda", apply_d3=False)` | **Cross-env.** Spawns the right `<env>/bin/python` worker for you and returns a results dict. |
| Make many repeated calls (e.g. bulk teacher labeling) | `Worker` (persistent-worker protocol) | One long-lived process per env; avoids per-call subprocess spawn. JSONL request/response keyed by `id`. |

```python
import os, sys
sys.path.insert(0, os.environ["OH_MY_MLIP_HOME"])  # import by path, not pip
from oh_my_mlip import resolve, get_calculator, run, Worker
```

`resolve()` is what a code-generating agent should call when it needs to emit a
standalone script: read the returned `python`, `imports`, `inference`, `env_run`
and write them into the file. `run()` is the convenience path for "just compute
this" when you are not already inside the target env.

## 3. Two request branches — keep them separate

**Decide which branch you are in before writing anything; they behave differently.**

### (A) Run one (or a few) specific models — `run_singlepoint` / `run_relax`

Examples: "single-point this structure with MACE", "relax this POSCAR with
SevenNet-Omni". Do **not** use the catbench roster runner for this.

1. **`resolve(model)`** to obtain `python`, `import`, the chosen version's
   `inference`, `arch_pinned`, and any `note`/`status`/`env_run`. **Honor
   `note`/`status`/`env_run` if present** — they encode special run conditions
   and validation state.
2. If you only need a result, call **`run(model, atoms, ...)`** — it spawns the
   correct env interpreter and returns `{"energy":..., "forces":...}`. If you
   must emit a standalone script, write the `import` + `inference` lines verbatim
   into `main.py` and run it with that model's `python` path.
3. **arch-pinned models** (`arch_pinned: true`, e.g. NequIP/Allegro): the
   `inference_sm86` / `inference_sm89` variant is selected by your host GPU
   (sm86 = A5000/A6000, sm89 = L40S). The matching `.pt2` is recompiled/reselected
   on your GPU on first run (see §6). Picking the wrong arch is a runtime error.
4. **`env_run` prefix** (e.g. `LD_LIBRARY_PATH=""` for DPA4): it must wrap the
   interpreter invocation. `run()`/`Worker` apply it automatically; if you write
   a script by hand, prepend it to the `python` line.

Minimal example (cross-env, no env management):

```python
from ase.build import bulk
from oh_my_mlip import run
atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
out = run("MACE", atoms, properties=("energy", "forces"), apply_d3=False)
print(out["energy"])
```

### (B) Full-roster catbench pipeline — `run_catbench`

Examples: "run catbench across all shipped models", "benchmark every model on
this adsorption dataset". Because each model is a different conda env, **no single
process can host two models** — the pattern is *one process per model*, each
using its own env interpreter, all writing into a shared `result/` directory that
catbench aggregates at the end.

`run_examples/catbench_quickstart.py` is the public, local (no scheduler) form of
the internal roster runner:

1. Put your dataset at `raw_data/<tag>_adsorption.json` in the working directory
   (this repo **bundles no benchmark data** — bring your own; see
   `run_examples/README.md`).
2. For each model in the roster it builds the catbench skeleton:
   `calc_num = 3` calculator instances, `config = {"mlip_name", "benchmark"}`,
   results into `cwd/result/`. With D3 enabled, the `mlip_name` gets a `_D3`
   suffix so D3/non-D3 results stay distinct.
3. Run each model with its own `<env>/bin/python` (applying `env_run` where the
   registry declares it), then aggregate with catbench's
   `AdsorptionAnalysis().analysis()` + `threshold_sensitivity_analysis()`.

> Speed caveat carried from the internal guide: catbench's `Time_per_step` is an
> aggregate (total time / total steps) and is sensitive to host CPU load — a
> busy machine inflates the *speed* number, but **accuracy (energy / MAE /
> anomaly) is unaffected**. Compare speed only across identical GPU/host
> conditions; models with heavy CPU-side neighbor lists (e.g. ORB conservative+inf)
> are the most contention-sensitive.

## 4. D3 dispersion correction

Every env ships catbench, so D3 is available everywhere:

```python
from catbench.dispersion import DispersionCorrection
calc_d3 = DispersionCorrection().apply(calc)   # calc = the MLIP calculator
```

With the tiered interface, pass `apply_d3=True` to `get_calculator()` / `run()`
and it is applied for you. The D3 CUDA kernel (`pair_d3.so`) is **not baked into
distributed tarballs** — it compiles on **your** GPU on first run and is then
cached (see §6). `source env.sh` first: it sets the shared caches, `PYTHONUTF8=1`
(prevents an ascii-decode crash during the D3 compile), and a fallback
`CUDA_HOME`/`LD_LIBRARY_PATH` so D3 works even in envs whose torch does not load
`libcudart.so.12` itself.

## 5. Gated vs open models

Read the `gated`, `license_url`, and `weights` fields in `models.json` before
fetching weights:

- **Open models** (`gated: false`, most of the roster): weights are either
  `bundled` (inside the conda-pack tarball), `auto-download` (fetched by name to
  the shared cache on first run), or `on-demand-hf` (pulled from the Hugging Face
  Hub). No token, no license step.
- **Gated models** (`gated: true`, e.g. all UMA variants): the weights require
  (1) accepting the upstream license at the model's `license_url`
  (`https://huggingface.co/facebook/UMA` for UMA) with the same Hugging Face
  account, and (2) exporting your own token: `export HF_TOKEN=hf_...`. **This
  repo never redistributes gated weights** — they are always fetched on first run
  with the user's own token after the user accepts the license. If `HF_TOKEN` is
  missing or the license has not been accepted, the fetch fails by design; surface
  the `license_url` to the user rather than retrying.

See `docs/gated_models.md` for the full flow.

## 6. First-run compilation (D3 `.so`, NequIP/Allegro `.pt2`)

Some artifacts are **architecture-specific** and are therefore *never* shipped in
the relocatable tarballs — they are produced on **your** GPU the first time you
run:

- **D3 kernel `pair_d3.so`**: compiled once on first D3 use. Needs `nvcc`
  (a CUDA toolkit) on `PATH`; `env.sh` autodetects `CUDA_HOME`.
- **NequIP / Allegro `.pt2`**: the AOT-compiled model is selected/recompiled for
  your GPU's compute capability — `sm86` (A5000/A6000) vs `sm89` (L40S) — into
  `models/compiled/{sm86,sm89}/`. `arch_pinned: true` in the registry marks these.

**`nvcc` requirement and fallback**: first-run compilation needs the CUDA
toolkit's `nvcc`. If `nvcc` is absent, `install.sh` either fetches a
prebuilt-per-arch D3 artifact or degrades D3 **off** with a clear message (the
MLIP itself still runs; only the dispersion correction is unavailable). It never
silently produces wrong numbers. See `docs/arch_first_run_compile.md`.

## 7. Checklist before you hand back code

- [ ] Read `models.json` (not memory) for `python` / `import` / `inference`.
- [ ] Used the `oh_my_mlip` interface (`resolve` / `get_calculator` / `run` /
      `Worker`) rather than a hand-built interpreter string where possible.
- [ ] For arch-pinned models, selected the `sm86`/`sm89` variant for the host GPU.
- [ ] Applied any `env_run` prefix and respected `note`/`status`.
- [ ] For gated models, checked `HF_TOKEN` + surfaced `license_url`; never wrote
      gated weights into a file.
- [ ] `source env.sh` happens before the run (D3/cache setup).
- [ ] No scheduler/queue assumptions — this runs locally.

---

### Section → MCP tool

These top-level sections map onto the MCP tools in `oh_my_mlip/mcp_server.py`, so
the same guide drives both a human-following agent and a tool-calling one. Launch
the server with `python -m oh_my_mlip.mcp_server` (see the README "MCP server"
section). `list_models` / `describe_model` / `model_status` are GPU-free; the
`run_*` / `install_model` tools execute at GPU/compute runtime.

| Section | MCP tool |
|---|---|
| §0–1 read registry / layout | `list_models` |
| §0 resolve one model's codegen dict | `describe_model` |
| §5 per-model validation / gated / weights status | `model_status` |
| §1, §6 materialize an env / first-run compile | `install_model` |
| §3A `run(model, atoms)` single-point | `run_singlepoint` |
| §3A relaxation | `run_relax` |
| §3B full-roster catbench | `run_catbench` |
