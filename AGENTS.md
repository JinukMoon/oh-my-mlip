# oh-my-mlip — Agent Guide

This file is written for an AI coding agent (Claude Code or any other LLM) that
has cloned this repo and is asked to **use** one or more MLIPs — to write a
single-point/relaxation script, run catbench, or wire an MLIP calculator into a
larger workflow. A human can follow it verbatim too.

> **This runs locally — no SLURM, no job scheduler, no cluster.** Every model is
> a plain Python interpreter call inside its own conda environment, executed on
> whatever machine you are on (laptop, workstation, single GPU box). There is
> nothing to submit to a queue.

## ⚠️ Hard operating constraints (omm self-healing setup) — read before any install

These are non-negotiable and override convenience:

1. **WSL-ONLY install / compile / delete.** Every MLIP env build, GPU compile
   (D3 `.so`, NequIP/Allegro `.pt2`), verification, and env deletion happens
   **only on this local WSL host**. **NEVER** install, build, compile, or delete
   on remote servers (e.g. 147 / 114) — their login nodes are blocked and SLURM
   is a bottleneck. The setup sweep is a local-WSL operation, full stop.
2. **ONE ENV AT A TIME — never parallelize installs/builds.** Build, compile, and
   verify exactly one env end-to-end before starting the next. **NEVER** run two
   `install.sh` / `conda env create` / pip-into-env builds concurrently: parallel
   builds race on the shared conda/pip package cache and the env prefix, contend
   for disk and host/GPU memory, and stack driver-touching compiles (D3 `.so`,
   NequIP/Allegro `.pt2`) that destabilize each other. `install.sh` and
   `scripts/sweep_local.py` are serial by design (one env, in order, prefix
   removed after its attempt); do not defeat that with background `&`, `xargs -P`,
   or multiple concurrent shells.
   Before starting or *retrying* a build, first reap any orphan `install.sh` /
   `conda` / pip-into-env processes left by a prior attempt (they still hold the
   env prefix and package-cache locks — see `scripts/sweep_watchdog.sh`), and
   confirm disk headroom up front (>= 30 GB default) instead of only reacting to
   the guardrail mid-build.
3. **Success = ASE energy + forces on the GPU, not "install succeeded".** A model
   is only DONE when `run_examples/single_point.py <model>` actually returns
   energy and forces (GPU-backed). `install.sh` exit-0 / the `.omm_ready`
   sentinel is necessary but **not** sufficient.
4. **Self-healing loop.** The agent autonomously installs → compiles → verifies,
   retrying with *different* strategies (see §8 error-class policy) until ASE
   works, or the same error signature repeats (deterministic stall stop via
   `scripts/setup_guardrail.py`). Bounded disk; clean up each env after testing.
5. **Driven by a fresh Codex agent.** The autonomous install/compile/verify loop
   is handed to a fresh Codex agent (independent perspective, separate runtime).
6. **End vision — omm embeds in the session like OMC.** The Claude Code plugin
   (`.claude-plugin/` + `skills/`) is the point: drop into any session and
   `/oh-my-mlip:setup <model>` makes MLIP install trivial. Keep that the north star.
7. **Host-scoped truth — a carried-over label is NEVER proof.** A `models.json`
   `validation` value (or any status) carried over from another host/GPU — e.g.
   the internal L40S catbench — is a *claim*, not evidence that the model runs
   **here**. The DONE verdict is the measured tier-1/tier-2 result on THIS host
   (see the two-axis `_meta.field_guide.validation`: arch-validity vs
   host-resource/driver-validity). If a `validated`-labeled model has not produced
   energy+forces on this host, treat it as unverified and say so — do not inherit
   the label. This is the "sign vs reality gap" documented in
   `docs/ground_truth_reclassification.md` (carried-over `validated_sm89` labels
   that actually failed or were skipped on the 4060 Ti host).
8. **Weight-integrity preflight — never load a 0-byte / partial weight.** A weight
   pre-staged or fetched to a cache MUST pass a non-empty size guard (and a sha256
   guard when a fingerprint is recorded) before it is loaded. A 0-byte / 202-block
   body is re-fetched, NEVER loaded: an `EOFError('Ran out of input')` or
   "... does not exist" at load time is the bucket-A signature (corrupt/empty cache,
   not a broken model). Frameworks that download inside their own package
   (deepmd/grace/pet/dpa4/eqnorm/matris) are served by
   `scripts/{prestage,prepare}_<env>_weights.py`, wired into `install.sh` and run
   with the ENV interpreter — do NOT trust the in-package downloader's "file exists"
   check (it accepts a 0-byte file).
9. **Driver ↔ CUDA-runtime preflight — degrade to CPU, do not crash.** Before the
   build, compare the recipe's torch `+cuNNN` (the CUDA runtime the wheel needs)
   against the CUDA version the host NVIDIA driver exposes. If the host is lower
   (e.g. host CUDA 12.9 vs a `+cu130` build) the env still builds and runs on CPU,
   but `torch.cuda` is unavailable — record `validation=tier1_cpu_driver_skew` and
   say so up front (`install.sh::warn_driver_skew`) instead of letting GPU inference
   crash with a cryptic "driver too old" (bucket E: tace/dpa4/matris). The fix is a
   newer host driver, NOT a code change; never silently claim GPU validation on a
   driver-skewed host.

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
2. Two equal first-class ways to compute — pick by what the user needs:
   - **One-shot result**: call **`run(model, atoms, ...)`** — it spawns the
     correct env interpreter and returns `{"energy":..., "forces":...}`.
   - **Calculator embedded in the USER'S OWN code** (MD loops, relax scripts,
     custom pipelines — the most common real request): take `resolve(model)`'s
     `import` + `inference` lines and paste them into the user's script
     **VERBATIM — never edit a character**. These exact lines are what passed
     equivalence validation; a "small improvement" to them is an unvalidated
     model. The inference line creates `calc`; follow with
     `atoms.calc = calc`. Execute the script with `spec["python"]` — never a
     guessed interpreter or `conda activate`.
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
  account, and (2) making the user's own read token available (preferred:
  `huggingface-cli login`; or `HF_TOKEN_PATH` / `OMM_HF_TOKEN_FILE` / `HF_TOKEN`).
  **This repo never redistributes gated weights** — they are always fetched on
  first run with the user's own token after the user accepts the license. If no
  token is resolvable or the license has not been accepted, the fetch fails by
  design; surface the `license_url` to the user rather than retrying.

**When a gated model is requested:** do not just dump a traceback. GUIDE the user
through token setup per [`docs/hf_token.md`](docs/hf_token.md) — accept the
license, create a READ token, make it available the leak-safe way (never
instruct pasting the token literal on the command line) — and fail any gated
fetch with an actionable pointer to `docs/hf_token.md` and the model's
`license_url`, not a raw exception.

See `docs/gated_models.md` for the full flow and
[`docs/hf_token.md`](docs/hf_token.md) for the canonical token setup.

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

## 8. Self-healing setup: error-class policy (retryable vs halt-and-report)

This section is the **single source of truth** for recovery strategy selection
in the self-healing install loop. The agent reads the traceback and classifies
it here; `scripts/setup_guardrail.py` owns only the deterministic stop set (disk
ceiling, identical-signature stall N=2, cumulative-attempt cap N=5, wall-clock
cap, and scoped cache cleanup — see "Deterministic stop" below). Neither
`skills/setup/SKILL.md` nor `setup_guardrail.py` copies or re-encodes this
taxonomy.

### RETRYABLE — agent picks a different strategy; loop continues

| Error class | Recovery action |
|---|---|
| **GPU arch mismatch** (`sm86` ↔ `sm89`; e.g. wrong `.pt2` loaded for host) | Reselect the arch-matched compiled artifact (`models/compiled/{sm86,sm89}/`) or recompile for the host arch via `scripts/compile_nequip.sh`. Do NOT delete the mismatched artifact first — rename to avoid a re-fetch race. |
| **Transient network / partial weight download** (connection reset, incomplete HF tarball, partial `.nequip.zip`) | Re-fetch the artifact. Clean the partial file before retrying so the fetch does not resume a corrupt state. |
| **Stale weight-fetch debris shadowing presence checks** (an earlier FAILED fetch left `tmp.tar.gz` / an empty or partial directory under `$OH_MY_MLIP_HOME/models/<framework>/`; later runs either treat weights as "present" or re-fetch forever, while a complete copy often already sits in the framework's upstream cache `~/.cache/<framework>/`) | Remove ONLY the debris under `models/<framework>/` (never touch upstream-cache real copies), then let `ensure_weights` re-run; when a complete upstream-cache copy exists, symlink it into the hub models path instead of re-downloading. NEVER hand-write a new filesystem-walking path resolver — weight resolution is owned by `oh_my_mlip/fetch.py`. (Host-proven 2026-07-19: GRACE `tmp.tar.gz` debris beside an intact `~/.cache/grace/GRACE-2L-OAM`.) |
| **`pypi.nvidia.com` unreachable** (torch `+cuNNN` recipes: the `nvidia-*-cu12` wheels resolve through the PyTorch index to pypi.nvidia.com; when that host times out, the pip stage of `conda env create` fails for EVERY torch env even though identical wheels exist on pypi.org) | Sideload the nvidia wheels from pypi.org into the partial env, then re-run `install.sh` (adopt-or-heal completes the rest). Host-proven 2026-07-17: read the exact pins from `https://pypi.org/pypi/torch/<X.Y.Z>/json` `requires_dist` (pypi's default torch X.Y.Z is the same cuNNN binary), `pip install --index-url https://pypi.org/simple <nvidia pins>` with the env's pip, then `./install.sh <env>`. Diagnose with `pip install --dry-run -r <pip-block>` — a ConnectTimeout to pypi.nvidia.com is this class. |

### HALT-AND-REPORT — do NOT auto-retry; surface an actionable message and stop

| Error class | Required action |
|---|---|
| **`nvcc` absent** | D3 dispersion compilation falls back: `install.sh` either fetches a prebuilt per-arch D3 artifact or degrades D3 **off** with a clear message (the MLIP itself still runs; only D3 is unavailable). See `docs/arch_first_run_compile.md` and `§6` of this file. Do not attempt to install `nvcc` or the CUDA toolkit. Report the degrade state to the user. |
| **Gated weights** (`gated: true`) — token missing or license not accepted | Surface the model's `license_url` and point to `docs/hf_token.md`. **Never auto-retry** a gated fetch (consistent with `§5`). The fetch failing here is by design; the user must accept the license and supply a read token. |
| **conda / mamba absent** | Surface an actionable install guide pointing the user to Miniconda/Mambaforge (`https://docs.conda.io/projects/miniconda`). A scoped Miniconda install is acceptable **only after the user explicitly consents**; the host must not be mutated without consent. |
| **Undocumented / unclear install or weight source** — the model's install recipe or `weights_*` fetch path is missing or wrong in `models.json` and the official page does not resolve it | Make **one bounded autonomous attempt** (check the official model/docs page + 1–2 real fetch/build tries). If that fails, **STOP — do not burn tokens on open-ended self-heal.** Ask the user to share the model's official install/weight-download docs or link, set the version's `models.json` `note` to start with `awaiting user docs:` describing what was tried, and report the blocker. This is the same trigger as the deterministic stop set (identical signature ×2, cumulative N=5, or wall-clock): on reaching it, switch to this docs-request path instead of auto-retrying. (Precedent: GRACE's nested-SavedModel flatten pattern was fixed fast once the user supplied the `gracemaker` foundation-models docs.) |

### Deterministic stop (owned by `scripts/setup_guardrail.py`, not this section)

`setup_guardrail.py` enforces **four bounded-self-heal stop conditions**
regardless of error class; whichever trips first stops the loop unconditionally:

1. **disk headroom** below the ceiling (default **30 GB**) → `guardrail_halt`.
2. **same normalized stderr signature** recurring ≥ N=2 times → `stalled`.
3. **cumulative attempts** reaching `cumulative_max` (default **5**, signature-agnostic)
   → `stalled_cumulative`. This bounds the loop even when the agent retries with a
   *different* strategy each round (which produces a fresh signature every time, so
   the signature stall alone would never fire — the divergence guard).
4. **wall-clock** elapsed since the first attempt reaching `wallclock_max_s`
   (off by default; set per host/model) → `wallclock_halt`.

Partial-artifact cleanup runs between attempts (`clean-cache`). When the helper
returns `"stalled"`, `"stalled_cumulative"`, `"wallclock_halt"`, or
`"guardrail_halt"`, the agent stops unconditionally and surfaces the helper's
verdict. These four conditions are the stop set; the recovery *strategy* axis
(retryable vs halt-and-report) lives in §8 above, and the root-cause *diagnosis*
axis (the install-failure buckets) is a separate, observability-only concern that
this stop set never re-encodes.

---

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

### Plugin vs MCP — surfaces, not knowledge homes

The **Claude Code plugin** (`.claude-plugin/` + `skills/`) is the **primary
onboarding surface**: it provides `/oh-my-mlip:setup`, `/oh-my-mlip:run`, and
`/oh-my-mlip:catbench` as slash commands in Claude Code with zero extra server
setup. The plugin skills are thin pointers — they contain no duplicate knowledge;
all strategy and model facts live here in `AGENTS.md` and `models.json`.

The **MCP server** (`python -m oh_my_mlip.mcp_server`) is an **optional
structured-query surface** for tool-calling agents that prefer typed tool
invocations over reading this document. It exposes the same registry information
and the same run/install entry points. No knowledge has moved from `AGENTS.md`
into the MCP server — it is a thin adapter only (see docstring in
`oh_my_mlip/mcp_server.py`).

**Single source of truth:** all model facts in `models.json`; all agent strategy
in `AGENTS.md`. Neither surface (plugin nor MCP) duplicates or extends these.

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
