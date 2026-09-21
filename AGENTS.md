# oh-my-mlip — Agent Guide

This file is for an AI coding agent (Claude Code, Codex or any other) working in
a clone of this repository on a user's behalf: installing machine-learning
interatomic potentials (MLIPs), running them, benchmarking them with CatBench,
fine-tuning them, or distilling them. A person can follow it step by step too.

Everything runs on the user's own Linux machine with an NVIDIA GPU — a
workstation, or a node of a cluster. Each framework lives in its own conda env
and is called through that env's Python interpreter. Nothing here needs a job
scheduler; where it helps, the tools can also write a SLURM script for the user
to submit themselves.

## Ground rules — read before installing anything

1. **One env at a time.** Build, compile and verify one env completely before
   starting the next. Never run two `install.sh`, `conda env create` or
   pip-into-env builds at the same time: they race on the shared package cache
   and the env prefix, and stack GPU compiles that interfere with each other.
   `install.sh` and `scripts/setup_sweep.py` are serial by design; do not work
   around that with `&`, `xargs -P` or several shells. Before retrying a failed
   build, make sure no process from the earlier attempt is still running, and
   check free disk space first (30 GB is the default floor).
2. **Done means energy and forces on the GPU.** A model is installed only when
   `scripts/setup_verify.py <model> --json` passes: it computes energy and forces
   and confirms the GPU was used. A zero exit code from `install.sh` is not
   enough.
3. **Recover, but boundedly.** When an install fails, classify the error with §8
   and retry with a *different* strategy. `scripts/setup_guardrail.py` stops the
   loop when the same error repeats or the attempt budget runs out; then report to
   the user instead of retrying.
4. **A label in the registry is not proof for this machine.** A `validation`
   value or note in `models.json` describes where the model was checked before.
   Whether it works *here* is decided only by `setup_verify.py` on this machine;
   say so when a model has not been verified here.
5. **Never load an empty or partial weight file.** A cached or downloaded weight
   must pass a size check (and a sha256 check when a fingerprint is recorded)
   before it is loaded. `EOFError('Ran out of input')` or "... does not exist" at
   load time usually means a broken download, not a broken model: remove it and
   fetch again. Frameworks whose own downloader accepts an empty file are
   pre-fetched by `scripts/{prestage,prepare}_<env>_weights.py`, which
   `install.sh` runs with the env's interpreter.
6. **Driver too old for an env's CUDA build: say so, do not pretend to fall back.**
   Before a build, `install.sh` compares the torch `+cuNNN` build with the CUDA
   version the NVIDIA driver supports. If the driver is older (for example CUDA
   12.x against a `+cu130` build), the env still installs, but a CPU run is only
   available where the variant's own inference line takes a device: the lines run
   verbatim and most pin `device='cuda'`. For those, `get_calculator` refuses the
   CPU request instead of building a CUDA calculator and calling it CPU, and
   `setup_verify.py` reports that failure. Tell the user a newer driver (or a host
   whose driver matches the build) is the fix; never report a GPU run as CPU, and
   never report a degraded run as GPU verification.

## 0. Read these first

1. **`models.json`** — the registry, and the only source of model facts. For
   each model: its env, `import` lines, per-version `inference` lines (the code
   that creates the calculator), `arch_pinned` and `gated` flags, `env_run`
   prefixes and notes. Look facts up here every time; never write them from
   memory.
2. **`envs/<env>.yml`** — each env's recipe (conda + PyPI), which `install.sh`
   builds; this is the only way an env is installed.

Run `source env.sh` once per shell before anything else: it sets up the model
caches and the CUDA environment that the D3 correction needs.

## 1. Layout

```
oh-my-mlip/                          # $OH_MY_MLIP_HOME (the clone root)
├── models.json                      # model registry
├── env.sh                           # caches + CUDA environment (source it)
├── install.sh                       # builds envs from envs/<env>.yml
├── oh_my_mlip/                      # Python interface (import by path, not pip)
├── envs/                            # env recipes; envs/locks/ = exact package sets
├── models/                          # hub-managed weights, compiled/<arch>/ models
├── run_examples/                    # single_point.py, relax.py, catbench_quickstart.py
├── recipes/                         # one workflow per request kind (§10)
└── scripts/                         # install, verify, benchmark, fine-tune, distill tools
```

Envs built by `install.sh` live at `$OH_MY_MLIP_HOME/envs/<env>/`. An env the user
already has can be registered instead with `scripts/adopt_env.py`. Either way,
`resolve(model)["python"]` gives the interpreter to use.

## 2. The interface — `oh_my_mlip`, not hand-built interpreter paths

| You want to… | Use | Notes |
|---|---|---|
| The interpreter, imports and calculator lines for a script | `resolve(model, version=None)` | Returns a dict; loads no model. |
| A calculator object **inside that model's env** | `get_calculator(model, version=None, device="cuda", apply_d3=False)` | Only when the current interpreter is that env's. |
| A quick result **from any Python** | `run(model, atoms, properties=("energy","forces"), device="cuda", apply_d3=False)` | Starts the right env's worker and returns a dict. |
| Many repeated calls | `Worker` / `WorkerPool` | One long-lived process per env; JSONL requests keyed by `id`. |

```python
import os, sys
sys.path.insert(0, os.environ["OH_MY_MLIP_HOME"])  # import by path
from oh_my_mlip import resolve, get_calculator, run, Worker
```

When writing a standalone script, take `python`, `imports`, `inference` and
`env_run` from `resolve()` and put them in the file.

## 3. Four kinds of request — decide which one you are in first

### (A) One or a few models — single point, relaxation, MD, custom scripts

Examples: "single-point this structure with MACE", "relax this POSCAR with
SevenNet-Omni", "write an MD script with UMA". Do not use the CatBench runner
for this.

1. Call **`resolve(model)`**. Respect `env_run` and any note: they carry run
   conditions.
2. Either:
   - **one-shot result** — `run(model, atoms, ...)` returns
     `{"energy": ..., "forces": ...}`; or
   - **calculator in the user's own script** (the usual case) — paste
     `resolve()`'s `imports` and `inference` lines **unchanged**. These are the
     lines that were checked; an edited line is an unchecked model. The inference
     line creates `calc`; then `atoms.calc = calc`. Run the script with
     `spec["python"]`, never a guessed `python` or `conda activate`.
3. **GPU-architecture-specific models** (`arch_pinned: true`: NequIP, Allegro)
   load a model compiled for one GPU architecture. `resolve()` detects the
   current GPU and fills the `models/compiled/<arch>/` path; `resolve(model,
   arch="sm86")` writes a job for a different GPU, whose compiled file must exist
   (§6).
4. **`env_run`** (for example an `LD_LIBRARY_PATH` setting) must prefix the
   interpreter call. `run()` and `Worker` apply it; in a hand-written command,
   put it in front of the `python` call.

```python
from ase.build import bulk
from oh_my_mlip import run
atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
print(run("MACE", atoms, properties=("energy", "forces"))["energy"])
```

### (B) Benchmark many models with CatBench — `run_catbench`

Examples: "benchmark every installed model on this adsorption dataset",
"compare MACE, SevenNet and UMA on CO2 reduction on Cu". No single process can
host two models, so each model runs in its own env process, and all write into
one `result/` directory that CatBench aggregates.

`run_examples/catbench_quickstart.py` drives this:

1. The dataset is `raw_data/<tag>_adsorption.json` in the working directory.
   The quickstart can fetch published CatBench datasets itself; the user's own
   VASP calculations are converted with `scripts/catbench_vasp_stage.py`, which
   works on a copy. Data format: `docs/catbench_data_format.md`.
2. For each model it sets up CatBench with `calc_num = 3` calculator instances and
   `config = {"mlip_name", "benchmark"}`, writing to `result/`. With D3 on,
   `mlip_name` gets a `_D3` suffix so both runs stay separate.
3. **Write the job files, then run them — never `python -c`.**
   `scripts/catbench_jobgen.py` writes `jobs/catbench_<MLIP>.py` (the
   `resolve()` lines) and `jobs/run_catbench_<MLIP>.sh` (changes to the absolute
   work dir, exports `env_run`, runs the `.py`); the quickstart runs the `.sh`.
   `--emit-only` stops after writing. `--slurm [--partition P]` also writes
   `jobs/run_slurm_<MLIP>.sh` without submitting it; `--submit` dispatches
   through `catbench_jobgen.submit()` instead of running locally.
4. When `result/` is filled, `scripts/catbench_report.py --result <dir> --out
   <dir>` runs CatBench's `AdsorptionAnalysis`, including the threshold
   sensitivity analysis, and writes `mae_table.{md,csv}`, `mae_comparison.png`
   and a `run_report.sh` that rebuilds the report.

> CatBench's `Time_per_step` is total time over total steps, so a busy CPU makes
> models look slower; energies, MAE and anomaly counts are not affected. Compare
> speed only on the same machine under the same load. Models with heavy CPU-side
> neighbour lists (for example ORB conservative) are the most sensitive.

### (C) Fine-tune a model on the user's data — `ft_run.py`

Examples: "fine-tune MACE on my dataset", "adapt ORB to my data", "continue
training SevenNet from the foundation checkpoint with my extxyz". Ask **which
model** and **where the data is** if either is missing. Never write a training
command or config from memory.

0. **Read the framework's settings before asking about hyperparameters.**
   `scripts/ft_run.py <model> --show-settings` prints the settings that framework
   really has (from `finetune/settings/<Framework>.json`, taken from its pinned
   source): native name, the flag that sets it (if any), upstream default, the
   official fine-tuning value where upstream publishes one, and whether a value
   is required. Frameworks differ: DeePMD trains by steps, not epochs; MACE and
   SevenNet turn stress on through its weight; some have no EMA or early stopping.
   The run uses the defaults unless the user says otherwise. Show the user the
   handful that matter for their run (training length, batch size, learning rate,
   energy/force/stress weights, stress on or off) with the value each will take
   and where it comes from, and ask whether to keep them or change any. Use the
   values the user gives; do not tune or second-guess them. Values resolve as:
   user > official fine-tuning value > upstream default; a required setting with
   none of these (e.g. DPA4's learning rate) must be asked, and is refused if
   still missing.
   Common knobs are flags (`--epochs`, `--max-steps`, `--batch-size`, `--lr`,
   `--energy-weight`, `--force-weight`, `--stress-weight`, `--include-stress` /
   `--no-stress`, `--patience`, `--ema`, `--precision`); any other setting passes
   as `--set <native name>=<value>`. A flag sets only the settings listed against it
   (`--lr` is the start learning rate, not the final one); settings marked "related
   to <flag>" are not touched by that flag and take `--set`. A knob the framework
   does not have is refused with its list of available knobs; never map it to
   something else. `ft_run.json`
   records every emitted value with its origin (user, official-finetune, default).
1. **Convert the data.** `scripts/ft_dataset.py` reads anything `ase.io.read`
   reads and writes one extxyz with a `SinglePointCalculator` **and** the same
   values as `REF_energy`/`REF_forces`, so a single file serves MACE (reads
   `REF_*`) and SevenNet, NequIP, Allegro, GRACE, MatterSim, PET and TACE (read
   the calculator). `--to deepmd` / `--to dpa4` writes the npy layout instead.
   `ft_run.py` calls it for you.
2. **`scripts/ft_run.py <model> --dataset <path> [--out DIR] [--epochs N]`**
   reads the version's `finetune` block in `models.json` and writes everything
   before running: the converted data, a patched config where the framework uses
   one, and `<out>/finetune_<version>.sh`, which reproduces the run from
   anywhere.
   - `--emit-only` stops after writing; `--slurm [--partition P]` also writes an
     SBATCH version without submitting it.
   - Exit codes are answers, not bugs. **2**: no documented training code for
     this framework (for example Eqnorm, MatRIS, AlphaNet).
     **3**: `runnable_as_installed: false`; the message lists each remaining
     blocker as the fix to apply. **4**: this hub has no command builder for the
     framework yet; nothing is written. **5**: an explicit `--seed` the framework
     cannot fully honour (NequIP/Allegro seed only the data split;
     `--allow-partial-seed` accepts that). **1**: usage error or unknown model.
     Pass the message to the user; do not retry with guessed flags.
   - Command builders exist for MACE, SevenNet, DeePMD, DPA4, GRACE, PET, NequIP,
     Allegro, MatterSim, TACE, CHGNet, UMA, fairchemv1 (eSEN), EquFlash, Nequix, ORB
     and EquiformerV3 (`BUILDERS` in `scripts/ft_run.py`). EquiformerV3's status is
     `source-derived (hub builder)`: upstream documents no user fine-tune, and its
     builder follows the installed trainer source; say so when you offer it.
   - `<out>/ft_run.json` records inputs, seed scope, the checkpoint to use and
     the command to rebuild the run; per-framework seed behaviour is in
     `recipes/finetune.md`.
3. **Verify: `scripts/ft_verify.py <ckpt> --model <M> [--version <V>] --json`**
   loads the new checkpoint through the framework's own ASE calculator in its
   own env, and passes when energy and forces are finite with the expected shape
   and, on `--device cuda`, the GPU was used. A checkpoint that only exists on
   disk is not a verified fine-tune.
4. **Weights with restrictive licences are flagged, not refused.** When a
   version has `finetune.licence` set (MACE-MH-1: `ASL`; EquFlash:
   `CC-BY-NC-SA-4.0`, the project's licence -- the record hosting the EquFlash
   weights says CC BY 4.0, see docs/model_licenses.md), `ft_run.py` prints the
   licence and its URL before starting. Tell the user: a fine-tuned model
   derived from those weights inherits their terms.

### (D) Distill a teacher into an NN-MTP student for LAMMPS — `distill_bootstrap.py`

Examples: "distill MACE into a LAMMPS student", "compress this model into a fast
CPU potential". Distillation uses the `onthefly-distill` engine, a separate
GPL-2.0 project (https://github.com/JinukMoon/onthefly-distill). If the user
has no checkout, ask where to clone it. It is invoked from that checkout
(`--repo` or `$ONTHEFLY_REPO`, required) and never copied into this repository. Ask for the
**teacher** (any installed model) and the **structure** (an ASE-readable file
with **at most 4 elements**) if missing.

1. **Once:** `scripts/build_lammps_nnmtp.sh --repo <engine>` builds LAMMPS with
   `pair_style nnmtp` under `$HOME/.cache/oh-my-mlip/lammps`. It is long and
   unattended; skip it if `$HOME/.cache/oh-my-mlip/lammps/build/lmp -h` already
   lists `nnmtp`.
2. **`scripts/distill_bootstrap.py --teacher <variant> --structure <file> --work
   <dir> --repo <engine> [--target-ps F]`** writes three files into a new or empty
   work dir and runs nothing:
   - `omm_teacher.py` — `make_calc()` built from `resolve(<teacher>)`'s import and
     inference lines, unchanged, with the teacher's start-up output silenced
     (the engine reads that process's stdout, and banner text would make it
     believe the teacher cannot relabel);
   - `config.yaml` — the engine's example config with the teacher hook, the
     teacher env's interpreter, the LAMMPS binary, the species from the structure
     and an absolute `work_dir`;
   - `run_distill.sh` — sets `PATH`, `PYTHONPATH` (engine and work dir) and
     `ONTHEFLY_CONFIG`, seeds the training pool with a short teacher MD, then runs
     the engine's own active-learning loop.
   For a run whose result must be judged, add `--acceptance` with every target
   stated explicitly (`recipes/distill.md`) and check it afterwards with
   `scripts/distill_verify.py --work <dir> --json`.
3. **Run it:** `cd <work> && bash run_distill.sh`, in the background with its
   output logged; it takes long (teacher MD, then rounds of training, student MD
   and relabelling). It stops by itself when the student survives the target MD
   length, when rounds stop making progress, or at the round limit; the outcome
   is in `<work>/run/.al_status`.
4. **At most 4 elements:** the student's LAMMPS pair style supports four species.
   `distill_bootstrap.py` rejects larger systems with a clear message before
   writing anything.

## 4. D3 dispersion correction

CatBench is installed in every env, so D3 is available everywhere:

```python
from catbench.dispersion import DispersionCorrection
calc_d3 = DispersionCorrection().apply(calc)   # calc = the MLIP calculator
```

Or pass `apply_d3=True` to `get_calculator()` / `run()`. The D3 CUDA kernel is
compiled for the user's GPU and cached; `install.sh` triggers that build once
when `nvcc` is available (§6). `source env.sh` first: it sets `PYTHONUTF8=1`
(avoids a decode error during that build) and finds `CUDA_HOME`.

## 5. Gated and open models

Check `gated`, `license_url` and `weights` in `models.json` before fetching:

- **Open models** (`gated: false`, most of them): weights are `bundled` in the
  framework package, `auto-download`ed by name on first use, or `on-demand-hf`
  from the Hugging Face Hub. No token needed.
- **Gated models** (`gated: true`: all UMA variants and eSEN-30M-OAM): the user
  must (1) accept the license at the model's `license_url` with their own
  Hugging Face account (`https://huggingface.co/facebook/UMA`,
  `https://huggingface.co/facebook/OMAT24`) and (2) make a read token available,
  preferably with `hf auth login` in their own terminal (alternatives:
  `HF_TOKEN_PATH`, `OMM_HF_TOKEN_FILE`, `HF_TOKEN`). This repository never
  redistributes gated weights. Without an accepted license or a token the
  download fails by design.

**When a gated model is requested:** check only whether a token is available,
never read or print it. If it is missing, give the user the `license_url` and the
steps in `docs/hf_token.md`, and ask them to run the login themselves. Never ask
them to paste a token into the conversation or onto a command line. HTTP 401 or
403 means access has not been granted yet. Full flow: `docs/gated_models.md`.

## 6. GPU-specific compilation (D3 kernel, NequIP/Allegro models)

Some artifacts only work on the GPU architecture they were built for, so they are
built on the user's machine:

- **NequIP and Allegro** (`arch_pinned: true`): during install,
  `scripts/prepare_nequip_weights.py` and `scripts/prepare_allegro_weights.py`
  download each checkpoint, check its MD5 and compile it with `nequip-compile`
  for the current GPU into `models/compiled/<arch>/` (`sm86`, `sm89`, …).
  NequIP uses `--modifiers enable_OpenEquivariance`; Allegro uses
  `--modifiers enable_CuEquivarianceContracter`. For another GPU architecture,
  run the prepare script on a machine with that GPU.
- **D3 kernel**: compiled once with `nvcc`; `env.sh` finds `CUDA_HOME`.

**Without `nvcc`**, only D3 is unavailable; the models themselves still run.
`install.sh` says so. Details: `docs/arch_first_run_compile.md`,
`docs/compile.md`.

## 7. Checklist before handing back code

- [ ] Model facts came from `models.json` via `resolve()`, not memory.
- [ ] Used `resolve` / `get_calculator` / `run` / `Worker` rather than a
      hand-built interpreter path.
- [ ] Calculator lines pasted unchanged; script run with `spec["python"]`.
- [ ] `env_run` applied; notes respected.
- [ ] For NequIP/Allegro, the compiled model exists for the GPU the job runs on.
- [ ] For gated models, token presence checked and `license_url` given; no token
      or gated weight written into any file.
- [ ] `source env.sh` before running.

## 8. Recovering from install errors

The agent classifies each failure here and picks the strategy.
`scripts/setup_guardrail.py` only decides when to stop (see below).

### RETRYABLE — try a different strategy, then continue

| Error | Recovery |
|---|---|
| **Wrong GPU architecture** (a `.pt2` compiled for another GPU) | Compile for this GPU with the prepare script (§6). Rename the mismatched file rather than deleting it first. |
| **Network error or partial download** (connection reset, incomplete archive or `.nequip.zip`) | Delete the partial file and fetch again, so the retry does not resume a corrupt download. |
| **Leftover files from a failed download** (a `tmp.tar.gz` or an empty or partial directory under `$OH_MY_MLIP_HOME/models/<framework>/` makes the weights look present, or makes every run download again; a complete copy may already be in the framework's own cache such as `~/.cache/<framework>/`) | Remove only the leftovers under `models/<framework>/` (never the framework's own cache), then let `fetch.ensure_weights` run again. Do not write your own path-searching resolver: weight locations belong to `oh_my_mlip/fetch.py`. |
| **`pypi.nvidia.com` unreachable** (torch `+cuNNN` envs pull `nvidia-*` wheels through that host; when it times out, every torch env fails in the pip step, although the same wheels are on pypi.org) | Install those wheels from pypi.org into the partial env, then run `install.sh` again. Take the exact pins from `https://pypi.org/pypi/torch/<X.Y.Z>/json` (`requires_dist`) and install them with the env's pip and `--index-url https://pypi.org/simple`. A `ConnectTimeout` to pypi.nvidia.com in `pip install --dry-run` identifies this case. |

### HALT AND REPORT — do not retry; give the user an actionable message

| Error | What to do |
|---|---|
| **`nvcc` missing** | D3 is unavailable; the models still run (§6). Tell the user; installing a CUDA toolkit is their decision. |
| **Gated weights without token or accepted license** | Give the `license_url` and point to `docs/hf_token.md` (§5). Do not retry the download. |
| **conda / mamba missing** | Point the user to Miniconda (`https://docs.conda.io/projects/miniconda`). Install it only with the user's explicit consent. |
| **Install recipe or weight source missing or wrong in `models.json`**, and the official documentation does not resolve it | Make **one bounded attempt** (read the official page, try one or two real fetches or builds). If that fails, stop: ask the user for the model's official install or download instructions, and report what was tried. The guardrail stops below lead here too. |

### When to stop (enforced by `scripts/setup_guardrail.py`)

Whatever the error, the first of these ends the loop:

1. free disk below the floor (default **30 GB**) → `guardrail_halt`;
2. the same normalized error **twice** → `stalled`;
3. **5** attempts in total, even with different errors → `stalled_cumulative`;
4. a wall-clock limit, when one is set → `wallclock_halt`.

Partial downloads are cleaned between attempts (`clean-cache`). On any of these
verdicts, stop and report the verdict.

## 9. Installing model envs — one, several or all

This is the procedure behind `/oh-my-mlip:setup` and any request to install or
"get working" MLIPs. After an env is installed and verified here, §3 takes over.
`skills/setup/SKILL.md` only points into this section.

### 9.0 When the repository is not on disk yet

The working directory does not matter:

- use `$OH_MY_MLIP_HOME` if it is set and exists, else `$OMM_HOME`;
- otherwise clone `https://github.com/JinukMoon/oh-my-mlip.git` into
  `~/.oh-my-mlip`;
- export `OH_MY_MLIP_HOME` for every child process;
- check `which conda || which mamba` before any install (§8 if missing);
- `source $OH_MY_MLIP_HOME/env.sh` before `install.sh`.

### 9.1 Listing what can be installed (installs nothing)

Read `models.json`, or call `oh_my_mlip.list_models()`, and report name,
framework and whether each model is gated. `docs/model_status.md` is the same list
for people. Offer the install step, and mention that gated models need the user's
own Hugging Face login (§5).

### 9.2 Which targets

- Named models: resolve each through the registry, keep the user's order;
  `install.sh` accepts several targets.
- `all`: every framework in `models.json`, one env each.
- `all except <names>`: `all` minus those; list the exclusions in the plan.
- Naming the models is the approval; only `all` needs the gate in §9.3.

### 9.3 Installing everything: survey, plan, approve

Installing everything takes 100+ GB and hours, so ask once, clearly:

1. Run `scripts/setup_survey.py` first. It reports each env's state (ready /
   partial / missing), the disk needed for what will actually be built, whether a
   Hugging Face token is available (presence only) and the gated models. Show its
   output as printed.
2. Show the plan before asking: `OH_MY_MLIP_HOME`, the state table with gated
   models and exclusions, and the disk verdict. Never offer to install an env that
   is already `ready`.
3. If disk is short, include that in the same question.
4. Ask through selectable options if the agent's interface offers them: install
   everything planned / only what is missing or broken / choose exclusions (then
   a multi-select of the pending models). Accept free-text exclusions too. After
   approval, run to the end without more questions.
5. Gated models without a token: include the token steps in the plan — each
   `license_url`, the token page, and `hf auth login` to run in the user's own
   terminal. Re-check when the user says the token is set.
6. Restate the final target list in one line before starting.

### 9.4 Running the installs

`scripts/setup_sweep.py --targets <M1,M2,...>` runs a batch: one env at a time,
gated models without a token recorded as `skipped_gated`, a failure never stops
the rest, a 10 GB free-disk check before each target (below it, the remaining
targets are recorded as `skipped_disk`), and one JSONL line per step under
`.sweep/`. Do not loop over `install.sh` by hand.

For each target:

0. **State first:** `scripts/setup_survey.py --table <model>` (read-only). If
   `ready`, go to step 4 and stop when verification passes; otherwise continue
   (`install.sh` repairs a partial env instead of duplicating it).
1. **Install:** `install.sh <model>` with `OH_MY_MLIP_HOME` exported; keep stdout
   and stderr.
2. **Guardrail:** save the attempt's stderr to a file as-is, then run
   `scripts/setup_guardrail.py gate --state <state-file> --ceiling-gb 30
   --stderr-file <stderr-file>`. Read its JSON verdict (the exit code is always
   0): `guardrail_halt` / `wallclock_halt` stop; `stalled` /
   `stalled_cumulative` stop and ask for documentation (§8); `ok` continues.
3. **On failure:** classify with §8, apply that strategy, back to step 1.
4. **Verify:** `scripts/setup_verify.py <model> --json`. It picks CPU when the
   driver is too old (ground rule 6), computes energy and forces, confirms GPU use,
   and prints one verdict (`pass`, `device`, `degraded`, `reason`, `energy_ev`,
   `fmax_ev_a`, `forces_shape`, `gpu_pid_confirmed`, `gpu_mem_bytes`,
   `local_record`); exit 0 means pass. A pass is recorded in the machine-local
   `models.local.json`, which `resolve()` exposes as `local_verified`. Report the
   verdict as printed; do not run your own GPU check. `pass: true, degraded:
   false` is a clean GPU pass; `degraded: true` is a CPU pass with a reason;
   `pass: false` goes back to step 3.
5. **Report:** model, env path, `energy_ev`, `forces_shape`, `device`, `degraded`
   (with reason), `gpu_pid_confirmed`.

After a batch, retry each failed target through steps 1–4, then run
`scripts/setup_sweep.py report`; targets that never ran show as `not_attempted`.
A single gated request without a token stops per §5; in a batch it is recorded
as `skipped_gated`.

### 9.5 NequIP and Allegro

These also need the compiled model for the GPU (§6). `install.sh` runs the
prepare scripts; if verification fails with a missing or mismatched `.pt2`, run
the prepare script again on this machine.

## 10. Recipes — from request to plan, approval, run and evidence

`recipes/` has one workflow per kind of request. Each follows the same six stages
(ask, plan, approve, execute, verify, out of scope — defined in
`recipes/README.md`) on top of the sections above, and ends with the helpers that
are planned but not implemented yet. Read the section here, then the recipe for
the request at hand:

| Request | Section | Recipe |
|---|---|---|
| install / verify envs | §9 | `recipes/setup.md` |
| single point / relaxation | §3A | `recipes/run.md` |
| CatBench | §3B | `recipes/catbench.md` |
| fine-tuning | §3C | `recipes/finetune.md` |
| distillation | §3D | `recipes/distill.md` |

Claude Code reaches a recipe through `skills/<name>/SKILL.md`; Codex and other
agents through the table above. Recipe text is not copied here or into the
skills (`tests/test_onramp_contract_no_dup.py`, `tests/test_recipe_contract.py`).

To test a different checkout than the installed plugin: in Claude Code, start
with `claude --plugin-dir <checkout>` and set `OH_MY_MLIP_HOME` to the same
checkout; in Codex, start inside that checkout so its `AGENTS.md` is read. A
session started without either uses the installed plugin, and its results say
nothing about the other checkout (`recipes/README.md`).

A recipe explains and calls; it does not duplicate what executable files own.
Package sets and their order live in `install.sh` with `envs/<env>.yml` and
`envs/<env>.build.sh`; run commands live in the generators under `scripts/` and
the files they write; the agent supplies the scientific inputs (models, data,
targets, budget). A failure those files do not handle is reported as failed and
fixed by changing the owning file, never by installing or upgrading packages
inside an env during a job.

### Plugin and MCP server

The **Claude Code plugin** (`.claude-plugin/` + `skills/`) adds
`/oh-my-mlip:setup`, `/oh-my-mlip:run`, `/oh-my-mlip:catbench`,
`/oh-my-mlip:finetune` and `/oh-my-mlip:distill`. The skills only point here and
to the recipes; model facts stay in `models.json`.

The **MCP server** (`python -m oh_my_mlip.mcp_server`) is optional: it offers the
same registry information and run/install entry points as typed tools, as a thin
adapter (see `oh_my_mlip/mcp_server.py`).

| Section | MCP tool |
|---|---|
| §0–1 registry and layout | `list_models` |
| §0 one model's `resolve()` dict | `describe_model` |
| §5 gated / weights status | `model_status` |
| §1, §6 is an env installed / the command that builds it | `install_model` |
| §3A single point | `run_singlepoint` |
| §3A relaxation | `run_relax` |
| §3B CatBench across models | `run_catbench` |
