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
3. **Materialize, then execute — never `python -c`.** Each model+version is
   ALWAYS written to disk first via `scripts/catbench_jobgen.py`:
   `jobs/catbench_<MLIP>.py` (verbatim `resolve()` codegen) and
   `jobs/run_catbench_<MLIP>.sh` (the rerun unit — `cd`s to the absolute
   workdir, `export`s `env_run` one key per line, then `exec`s the `.py`).
   The quickstart script runs that `.sh`, not the interpreter directly.
   `--emit-only` stops after writing the files; `--slurm [--partition P]`
   additionally emits `jobs/run_slurm_<MLIP>.sh` (identical body, SBATCH
   header, no `sbatch` issued); `--submit` dispatches through
   `catbench_jobgen.submit()` behind an injectable hook instead of running
   the local `.sh` in-process, and is never combined with `--emit-only`.
4. Once `cwd/result/` is populated, aggregate with
   `scripts/catbench_report.py --result <result dir> --out <report dir>` —
   it wraps catbench's own `AdsorptionAnalysis().analysis()` (re-execing
   itself under a catbench-bearing interpreter first, since catbench never
   lives in the ambient one) and derives `report/mae_table.{md,csv}` +
   `report/mae_comparison.png`, plus a `report/run_report.sh` that
   reproduces the report from any working directory.

> Speed caveat carried from the internal guide: catbench's `Time_per_step` is an
> aggregate (total time / total steps) and is sensitive to host CPU load — a
> busy machine inflates the *speed* number, but **accuracy (energy / MAE /
> anomaly) is unaffected**. Compare speed only across identical GPU/host
> conditions; models with heavy CPU-side neighbor lists (e.g. ORB conservative+inf)
> are the most contention-sensitive.

### (C) Fine-tune a model on your own dataset — `ft_run.py`

Examples: "fine-tune MACE on my dataset", "파인튜닝 해줘", "continue training SevenNet
from the foundation checkpoint with my extxyz". Ask for **which model** and **where
the dataset is** if either is missing, then follow this flow — never hand-write a
training command or config from memory.

1. **Convert first.** `scripts/ft_dataset.py` is the single canonical converter:
   anything `ase.io.read` handles goes in, and it writes a `SinglePointCalculator`
   **and** duplicates the same energy/forces into `REF_energy`/`REF_forces` on the
   same extxyz file — one file feeds MACE (which reads `REF_*`, not the calculator)
   and SevenNet/NequIP/Allegro/GRACE/MatterSim/PET/TACE (which read the calculator)
   at once. `--to deepmd`/`dpa4` writes the npy system layout instead. `ft_run.py`
   calls this for you; only invoke it directly for a converted dataset with no
   training.
2. **`scripts/ft_run.py <model> --dataset <path> [--out DIR] [--epochs N]`** resolves
   the version's `finetune` block in `models.json` (ingested from
   `scripts/upstream_finetune.py`) and writes **every artifact before running
   anything**: the converted dataset, a patched config file when the family's
   `selector_kind` is `config_key` (most families — a YAML/JSON patch, not a
   hand-built flag string), and `<out>/finetune_<version>.sh` — `cd`s to the
   absolute `<out>` dir, exports any `env_run`, then `exec`s the real command. Run
   that `.sh` (or let `ft_run.py` run it for you without `--emit-only`); rerunning
   it later from anywhere reproduces the same fine-tune.
   - `--emit-only` stops after writing the files. `--slurm [--partition P]` also
     emits an SBATCH-headed twin of the same body — never submits it.
   - Refusal is by design, not a bug: a `finetune.status` of `not-supported` or
     `code-excavation-needed` exits **2** (no training code exists yet, or exists
     but is undocumented — Eqnorm/MatRIS/AlphaNet/EquiformerV3); `runnable_as_installed:
     false` exits **3**, naming each remaining blocker phrased as its own fix
     (a missing import, a file to symlink). Relay that message to the user; do
     not retry with guessed flags.
   - Eleven families have a command builder built from the installed package's
     own sources (`BUILDERS` in `scripts/ft_run.py`: MACE, SevenNet, DeePMD, DPA4,
     GRACE, PET, NequIP, Allegro, MatterSim, TACE, CHGNet; citations in the
     `note` fields of `scripts/upstream_finetune.py`). There is no generic
     renderer. Any other family that passes the registry gate exits **4** before
     any dataset conversion or file write — "no real command builder", an
     implementation gap in this hub, not an upstream limitation. Nothing is
     emitted for it, `--emit-only` included; the sweep records it as
     `failed(no_builder)`. With the current registry every non-builder family is
     already stopped by exit 2 or 3, so exit 4 is the fail-closed guard. Eleven
     builders is this hub's implementation coverage, not the fine-tuning
     requirement met: the exit-3 families (UMA, ORB, Nequix, EquFlash,
     fairchemv1) and the unwritten dataset writers are gaps pending
     implementation here, never "unsupported" — that word is reserved for a
     cited upstream absence. Exit
     **1** is a usage error or unknown model; exit **5** is an explicit `--seed`
     the family cannot honour in full (NequIP/Allegro seed only the data split;
     their training seed is fixed upstream — `--allow-partial-seed` accepts the
     narrower scope, recorded in `<out>/ft_run.json` as `seed_control.scope`);
     any other code is the training
     process's own. Some emitted `finetune_<version>.sh` (SevenNet, NequIP/Allegro,
     TACE) carry one prestage line before `exec` — still a single rerunnable file.
     `<out>/ft_run.json` records the run's inputs, seed scope, designated
     checkpoint glob, artifacts and the `rematerialize` argv; the per-family
     seed table is in `recipes/finetune.md`.
3. **Verify with `scripts/ft_verify.py <ckpt> --model <M> [--version <V>] --json`** — loads the
   produced checkpoint through that framework's own ASE calculator, inside that
   framework's own env, and reports pass iff energy is finite, forces are finite
   with the probe's `[4,3]` shape, and the backend execution record agrees with
   `gpu_used` — on `--device cuda` that record must show GPU compute ops. This
   is what turns a run into a `demonstrated` claim; a checkpoint that merely exists
   on disk is not yet a demonstrated fine-tune.
4. **Licence-gated checkpoints are advertised, not refused (Part 8 decision (a),
   option B).** A version whose `finetune.licence` is set (MACE-MH-1: `ASL`;
   EquFlash: `CC-BY-NC-SA-4.0`) still runs — `ft_run.py` prints the licence and a
   URL before it starts. Pass that notice on to the user; `oh-my-mlip` is MIT and
   redistributes no weights either way, but a fine-tuned derivative of one of
   these checkpoints inherits that licence's terms.

### (D) Distill a teacher into a CPU LAMMPS student — `distill_bootstrap.py`

Examples: "distill MACE into a LAMMPS student", "compress this teacher into a
fast CPU potential", "run the onthefly-distill active-learning loop on my
structure". This branch orchestrates a **separate, GPL-2.0 sibling repo**,
`onthefly-distill` (default `~/01_2026/onthefly-distill`, override with
`--repo` or `$ONTHEFLY_REPO`) — it is invoked, never absorbed: nothing from it
is copied into this repo, and this repo's MIT licence covers only what lives
here. Ask for **which teacher** (any oh-my-mlip model/version) and **which
structure** (an ASE-readable file with **at most 4 distinct elements** — see
below) if either is missing.

1. **One-time prerequisite**: `scripts/build_lammps_nnmtp.sh` builds a LAMMPS
   binary with `pair_style nnmtp` (following `onthefly-distill/lammps/BUILD.md`
   verbatim, cloned outside this repo under `$HOME/.cache/oh-my-mlip/lammps`).
   Run it once; it is long and unattended. Skip it if
   `$HOME/.cache/oh-my-mlip/lammps/build/lmp -h` already lists `nnmtp`.
2. **`scripts/distill_bootstrap.py --teacher <variant> --structure <file>
   --work <dir> [--target-ps F]`** renders three files into the (absolute)
   work dir and executes nothing:
   - `omm_teacher.py` — a zero-arg `make_calc()` whose body is
     `oh_my_mlip.resolve(<teacher>)`'s import + inference lines pasted
     **VERBATIM inside the function** (never edit a character — a modified
     line is an unvalidated model), wrapped in a stdout/stderr-suppression
     context. That suppression is load-bearing, not decoration: several
     teacher frameworks (MACE's `mace_mp()` among them) print banner text at
     import/construction time, and `onthefly-distill/scripts/can_relabel.py`
     captures this exact process's stdout via a bash command substitution
     compared against the literal string `"1"` — unsuppressed banner noise
     makes an AL-capable teacher misread as unable to relabel, and the loop
     silently falls back to one-shot distillation instead of running the
     active-learning loop. Confirmed against this hub's MACE-MPA-0 teacher.
   - `config.yaml` — overlaid onto `onthefly-distill/config.example.yaml`'s
     own structure (so every key its config loader reads stays present):
     `teacher.type: ase_calculator`, `teacher.calculator:
     "omm_teacher:make_calc"`, `python_bin` set to the **teacher's own env
     interpreter** (so every AL-loop subprocess — train / student-MD / label —
     runs inside an env that already has the teacher framework), `lmp_bin`
     from step 1, and `system.{init_structure, species, specorder, masses}`
     derived from `--structure` via `ase.io.read`. **`work_dir` is rendered as
     an ABSOLUTE path** — `onthefly-distill`'s own loop script `cd`s into
     itself before reading the config, so a relative `work_dir` would resolve
     inside the sibling repo instead of your work dir.
   - `run_distill.sh` — the rerun unit. Three environment facts, each
     preventing a distinct silent failure: a `PATH` pin to the teacher env's
     `bin` (the loop bootstraps its own config reads with bare `python`),
     `PYTHONPATH` covering **both** the sibling repo (every AL-loop stage runs
     as `python -m ontheflydistill.<mod>`) **and** the work dir (`omm_teacher`
     is imported via `importlib` from a script whose own `sys.path[0]` is the
     sibling repo's `scripts/`, so the work dir must be on the path too), and
     `ONTHEFLY_CONFIG` pointing at the generated `config.yaml` (without it the
     loop silently falls back to its built-in Pt-water defaults and ignores
     everything just rendered). It then seeds the initial AL-pool dataset —
     `scripts/teacher_md.py` (a short teacher MD trajectory) followed by
     `python -m ontheflydistill.merge_xyz` — the same two steps
     `onthefly-distill/examples/ptwater_acid/README.md` documents by hand,
     automated and skipped on rerun once `dataset.extxyz` exists. Only then
     does it `exec onthefly-distill/scripts/al_loop_local.sh` **verbatim** —
     never reimplemented.
3. **Run it**: `cd <work> && bash run_distill.sh`. This is genuinely long
   (teacher MD, then repeated train → student-LAMMPS-MD → relabel rounds) —
   run it in the background with `tee` and poll, per the house long-running-
   script convention. It stops on its own: `stable_in_dump` (student reached
   `al_loop.target_ps`), `STALLED` (no crash-time progress for
   `no_progress_limit` rounds), or the `max_iter` backstop. The result is
   always in `<work>/run/.al_status`.
4. **`<=4 distinct elements`, stated as a `pair_nnmtp` v1 bound.** The LAMMPS
   pair style the loop's student uses hard-codes `species_Z[4]`
   (`pair_nnmtp_v2` lifts this but is documented as not used by the loop) —
   `distill_bootstrap.py` refuses a richer structure with an actionable error
   before writing anything, rather than failing deep inside a LAMMPS run.

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

## 9. Installing model environments — single, multiple, or all

This is the procedure behind `/oh-my-mlip:setup` (or any request to install,
set up, or "get working" one or more MLIPs). It runs upstream of §3: once an
env is installed and verified here, §3A–D take over for actual use. The
`skills/setup/SKILL.md` contract is a pointer into this section — it carries
no procedure of its own.

### 9.0 Bootstrapping when the repo is not yet on disk

The plugin must behave identically from any working directory — the cwd plays
no role in locating the repo:

- If `$OH_MY_MLIP_HOME` is set and the path exists, use it.
- Else if `$OMM_HOME` is set and the path exists, use it as `OH_MY_MLIP_HOME`.
- Otherwise clone `https://github.com/JinukMoon/oh-my-mlip.git` into
  `~/.oh-my-mlip` (no ref pinned yet).
- Export `OH_MY_MLIP_HOME` so every child process inherits it; do not rely on
  `env.sh`'s relative-path autodetection.
- Confirm `which conda || which mamba` before entering any install loop —
  `install.sh` hard-exits without one; guide the user to a scoped Miniconda
  install only with explicit consent (the "conda / mamba absent" row of §8).
- `source $OH_MY_MLIP_HOME/env.sh` before any `install.sh` invocation (§0/§4).

### 9.1 Roster listing (read-only, no install)

"Which models can I install?" never installs anything: resolve the repo per
§9.0 (a clone is cheap; no conda/GPU/env build needed for a listing), then
read `models.json` and report name / framework / `gated` / validation status
per model — `oh_my_mlip.list_models()` is the one-liner if Python is
available. `docs/model_status.md` is the human-readable rendering of the same
registry. Close by offering the install step and flagging gated models as
needing the user's own HF token (§5).

### 9.2 Target resolution — single, several, all, or all-except

- Several explicit names — resolve each via the registry, preserve the given
  order; `install.sh` natively accepts the same multi-target list.
- `all` (or "everything") — every model family in `models.json`, one
  representative version per env; bare `install.sh` builds every recipe.
- `all except <names>` — resolve `all`, then drop the named families; state
  the exclusions in the plan so the user sees what was left out.
- A single or explicitly-listed target skips the approval gate in §9.3 —
  naming the model(s) IS the approval, and §9.4's loop runs zero-prompt.

### 9.3 The `all`-target survey-plan-approve gate

An `all` sweep is a 100+ GB, multi-hour commitment, so it gets the one
deliberate human checkpoint in this flow:

1. Run `scripts/setup_survey.py` first, before anything else. It computes
   atomically the one fact set the plan needs: per-env state (ready / partial
   / missing), the disk budget counting only the envs that will actually
   build, leak-safe token availability (§5 — the value is never read), and
   the gated list. Render this output as-is; never recompute, reorder, or
   partially re-derive it, and never open with a disk question before it has
   run.
2. Show the table before asking: the resolved `OH_MY_MLIP_HOME`, the per-env
   state table plus gated models and exclusions, and the disk verdict — in
   that order, before the approval question. The question restates the
   counts in its own text; an option that offers to install an already-
   `ready` env is a contract violation.
3. If the budget does not fit, that shortfall is part of the same approval
   question, not a separate upfront alarm — the selection in step 4 decides
   what makes the cut.
4. Ask for approval using the host's native interactive question UI when one
   exists (e.g. Claude Code's selectable-option tool), not a plain-text
   prompt: one single-select question (install everything in the plan / only
   what's missing-or-broken / choose exclusions), and if exclusions are
   chosen, follow-up multi-select question(s) listing the pending models
   (batched to the UI's per-question option limit). Free-text exclusions
   ("skip everything except MACE") are always honored too. Fall back to a
   text plan + text approval only when no interactive question UI exists.
   After approval the sweep runs to completion with zero further prompts.
5. Gated targets with no token found: the plan must include a token request
   spelling out the literal commands/URLs from `docs/hf_token.md` — each
   gated model's `license_url`, the token-creation page, and
   `huggingface-cli login` to run in the user's OWN terminal (never paste the
   token into the conversation — §5's leak-safe rule). The user can say
   "token is set" afterward and the sweep re-checks before proceeding.
6. An approved subset ("skip the last three") is an exclusion list — restate
   the final targets in one line before starting.

### 9.4 Batch execution and the per-target self-healing loop

After approval (or immediately, for single/explicit targets), run each
target through the loop below. `scripts/setup_sweep.py --targets
<M1,M2,...>` is the deterministic driver for a batch: it enforces
one-env-at-a-time execution (the hard constraint in this file's header),
`skipped_gated` bookkeeping for gated targets without a token, never
stopping on one target's failure, a 10 GB disk-floor re-check before each
target (below it, that and every remaining target are recorded
`skipped_disk` and the sweep ends), and one JSONL ledger line per phase
under `.sweep/`. Do not iterate `install.sh` over targets by hand.

Per target, whether inside a sweep or standalone:

0. **Take stock first**, always, before any mutation:
   `scripts/setup_survey.py --table <model>` (read-only) reports
   ready / partial / broken / not-installed, using the same state rules as
   `install.sh --status`. `ready` jumps straight to step 3 (verify only —
   report "already installed and verified" and stop if it passes; fall into
   step 1 if it fails, since `install.sh`'s adopt-or-heal repairs rather
   than duplicates an env). Anything else proceeds to step 1, naming the
   state that was found.
1. **Install**: `install.sh <model>` (with `OH_MY_MLIP_HOME` exported),
   capturing stdout/stderr.
2. **Guardrail check**: write the attempt's raw stderr to a file — the
   helper owns normalization, never pre-normalize or hash it yourself —
   then `scripts/setup_guardrail.py gate --state <state-file> --ceiling-gb
   30 --stderr-file <stderr-file>`. It always exits 0 and prints one JSON
   verdict; parse the JSON, not the exit code: `guardrail_halt` /
   `wallclock_halt` stop unconditionally; `stalled` / `stalled_cumulative`
   stop and switch to the docs-request path (§8); `ok` continues.
3. **On failure, recover**: classify the traceback against §8's
   retryable/halt-and-report table and apply the matching strategy, then
   return to step 1. A missing or unclear install recipe / weight source, or
   any guardrail stop above, means exactly one bounded attempt, then the §8
   docs-request path — not an open-ended retry loop.
4. **On success, verify with one command**:
   `scripts/setup_verify.py <model> --json`. It preflights the driver-skew
   predicate (constraint 9 in this file's header) to pick the device, runs
   the single-point witness, samples GPU PIDs with descendant attribution
   (the compute PID is a worker grandchild — never run `nvidia-smi`
   yourself), and prints one JSON verdict (`pass`, `device`, `degraded`,
   `reason`, `energy_ev`, `fmax_ev_a`, `forces_shape`, `gpu_pid_confirmed`,
   `gpu_mem_bytes`, `local_record`); exit 0 iff `pass`. On pass it also
   upserts the machine-local verified ledger `models.local.json`, which
   `resolve()` exposes as `spec["local_verified"]`. Render the verdict as
   printed — never re-judge it or run a second GPU check.
   `pass:true, degraded:false` is a clean GPU verification;
   `pass:true, degraded:true` is a pass-with-caveat (report `device=cpu` and
   the computed `reason`); `pass:false` returns to step 3 using `reason` as
   the error signature.
5. **Declare success**: report the verdict fields as-is — model name, env
   path, `energy_ev`, `forces_shape`, `device`, `degraded` (with its reason
   when true), `gpu_pid_confirmed`.

After a batch sweep completes, run the recovery pass for each `failed`
ledger entry through steps 1–4 above, then finish with
`scripts/setup_sweep.py report` — the final report comes strictly from the
ledger (a target absent from it appears as `not_attempted`, never silently
dropped). A single-target gated request still halts per §5; inside a batch
the driver's `skipped_gated` bookkeeping replaces the halt.

### 9.5 Arch-pinned first-run compilation

For `arch_pinned: true` models (e.g. NequIP, Allegro), the loop above must
also cover first-run compilation of the `.pt2` artifact — see §6 for the
compilation flow, artifact locations, and arch-selection logic. The loop
body in §9.4 applies equally; checkpoint acquisition and
`scripts/compile_nequip.sh` are the arch-pinned-specific piece.

---

## 10. End-to-end recipes — a request becomes plan → approval → execution → evidence

`recipes/` holds one workflow protocol per request kind. Each recipe layers
the six-stage job protocol (ask, plan, approve, execute, verify, out of
scope — defined once in `recipes/README.md`) on top of the mechanics
sections of this file, and closes with a "Planned helpers (not
implemented)" section so a reader can tell an existing command from a
design target. Read the mechanics section first, then the recipe, and only
for the request at hand:

| Request | Mechanics here | Recipe |
|---|---|---|
| install / verify envs | §9 | `recipes/setup.md` |
| single-point / relax | §3A | `recipes/run.md` |
| catbench | §3B | `recipes/catbench.md` |
| fine-tune | §3C | `recipes/finetune.md` |
| distill | §3D | `recipes/distill.md` |

Discovery is host-neutral: Claude Code reaches a recipe through the
matching `skills/<name>/SKILL.md`; Codex, and any agent that reads this
file, reaches it through the table above. Recipe text is never copied into
this file or into a skill (`tests/test_onramp_contract_no_dup.py`,
`tests/test_recipe_contract.py`).

To exercise a candidate checkout rather than the plugin a host already has
installed, point the host at that checkout: Claude Code with
`claude --plugin-dir <candidate root>` (session-local; loads that root's
`.claude-plugin/` and `skills/`, installs nothing) and `OH_MY_MLIP_HOME`
set to the same root; Codex by starting in that root, whose `AGENTS.md` it
reads from its cwd. A session started without either runs the installed
plugin, which may be older than the candidate, and its results are not
evidence about the candidate (`recipes/README.md`, candidate-local
discovery).

A recipe explains and invokes; it does not own knowledge that an
executable file already owns. Package combinations and their order live
in `install.sh` with `envs/<env>.yml` / `envs/<env>.build.sh`; run
commands live in the emitters under `scripts/` and the run files they
write; the agent's contribution is the scientific inputs (models,
dataset, targets, budget). A failure those files do not already handle is
recorded as failed and fixed by a reviewed change to the owning file —
never by installing or upgrading something inside an env during a job.
Every recipe therefore opens with an executable-chain table (step → owner
file) and closes with the helpers still planned and the steps that lack an
owner file.

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
in `AGENTS.md`; the per-request job protocols in `recipes/` (§10). Neither
surface (plugin nor MCP) duplicates or extends these.

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
