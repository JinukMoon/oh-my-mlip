# build_attempt — the tiered build-test runbook

This is the runbook a fresh agent (or maintainer) follows to find **what fails**
across the env roster and iterate on it env-by-env. The driver is
[`scripts/build_attempt.sh`](../scripts/build_attempt.sh); the tracked output it
produces is [`docs/build_attempt_results.md`](build_attempt_results.md).

> **Never echo or commit a Hugging Face token.** The driver reads your token
> only to *attempt* a gated download, never prints it, and runs every report
> line through a sanitizer. After writing the report it re-scans it and **fails**
> if a token-shaped literal slipped through. See [`hf_token.md`](hf_token.md).

## Purpose

`install.sh` is the primary install path, but it can fail per env for many
reasons (a wheel went 404, a CUDA pin drifted, a gated license was not accepted).
This driver records a **per-stage status for each env** so you get a single
table of exactly which env failed at which stage — the iterate list. It does
**not** drive heavy GPU builds itself; it is meant to be fast and safe to run
repeatedly, with the expensive tiers reserved for an owner/overnight run.

## Stage enum

Each env is walked through these stages, in order. The enum is deliberately
**honest**: `install.sh` creates an env with `conda env create --file <recipe>`,
which folds the conda solve **and** the pip block into one call. You cannot
cleanly separate a "conda-create" stage from a "pip-install" stage, so this
driver does **not** fake that split — there is a single `env-create` stage.

| stage | meaning |
|---|---|
| `resolve` | the recipe `envs/<env>.yml` exists and the env name is known |
| `reachability` | wheel / source URLs resolve (via `scripts/verify_sources.py`) |
| `dry-run` | `install.sh --dry-run <env>` prints a plan (no network, no install) |
| `env-create` | **real** `conda env create --file` — conda + pip in ONE call |
| `weights-download` | gated/auto weights fetch attempt (token via `HF_TOKEN_PATH`) |
| `compile` | D3 / per-arch kernel compile — **needs a GPU** |
| `run` | single-point inference smoke — **needs a GPU** |

`compile` and `run` are logged as **`gpu-required-deferred`** on any host without
`nvcc`/a GPU. That is a deferral, not a failure.

### Status values

Every row carries one of:

- `ok` — the stage succeeded.
- `fail` — the stage failed; the (sanitized) error is in the detail column and
  the env/stage is repeated in the **Failure list** summary.
- `gpu-required-deferred` — the stage needs a GPU this host does not have.
- `skipped` — the stage was not applicable (e.g. `env-create` skipped because no
  conda is on PATH, or any stage under `--dry-run-self`).

## Tiers

Select with `--tier`:

| tier | what it does | who runs it |
|---|---|---|
| `lint` (default) | GPU-free: `resolve` + recipe reachability + `install.sh --dry-run` for **all** envs. Fast and safe. | anyone, any time |
| `subset` | `lint`, plus a **real** `env-create` for the envs named via `--envs a,b` and a `weights-download` attempt for them | a representative spot-check |
| `full` | `lint`, plus a **real** `env-create` for **all** envs | owner / overnight — **not** a default |

```bash
# default GPU-free lint over the whole roster
./scripts/build_attempt.sh --tier lint

# spot-check two representative envs with a real build + weights attempt
./scripts/build_attempt.sh --tier subset --envs mace,sevennet

# full overnight rebuild of every env (slow, owner only)
./scripts/build_attempt.sh --tier full
```

## Setting the token (for weights-download)

Gated weights (e.g. UMA) need *your* Hugging Face token. The canonical, leak-safe
setup is in [`hf_token.md`](hf_token.md); the gated roster is in
[`gated_models.md`](gated_models.md). This driver reads the token **only** via
the standard `HF_TOKEN_PATH` file variable — it never accepts the token literal
on the command line and never reads the value into a logged variable.

```bash
# point at a token file OUTSIDE the repo (the file contains only the token)
export HF_TOKEN_PATH=/path/outside/repo/token
./scripts/build_attempt.sh --tier subset --envs uma
```

For local testing the token file on this machine is
`/home/jumoon/01_2026/17_MLIP_HUB/huggging_token`:

```bash
HF_TOKEN_PATH=/home/jumoon/01_2026/17_MLIP_HUB/huggging_token \
  ./scripts/build_attempt.sh --tier subset --envs uma
```

If `HF_TOKEN_PATH` is unset or unreadable, `weights-download` is recorded as
`skipped` (gated fetch not attempted) — it is not a failure.

## How token safety works (and why the report is commit-safe)

1. The token literal is read **once**, never printed, inside a `set +x` block so
   it can never surface via shell xtrace.
2. **Every** line written to the report passes through a sanitizer that scrubs
   (a) the exact token value, if the `HF_TOKEN_PATH` file was readable, **and**
   (b) any `hf_<20+ alphanumerics>` shape — both replaced with `hf_<redacted>`.
3. After writing the report, the driver runs `scripts/verify_no_token.py` against
   `docs/` (grep fallback otherwise) and **aborts non-zero** if any token-shaped
   literal survived. A committed `docs/build_attempt_results.md` is therefore
   guaranteed token-free. The same `verify_no_token.py` runs in CI.

## Self-test mode (no builds)

`--dry-run-self` exercises the full stage + sanitize + report + token-gate
machinery **without** building anything — no network, no conda, no token use.
Use it to confirm the driver itself works (and in CI) before an expensive tier.

```bash
./scripts/build_attempt.sh --tier lint --dry-run-self
```

## Output

The tracked deliverable is [`docs/build_attempt_results.md`](build_attempt_results.md):
a header (tier, host, conda, GPU presence), a **Stage results** table
(`env | stage | status | detail`), and a **Failure list** that repeats every
`fail` row with its sanitized error. Start your iterate loop from the Failure
list: fix the recipe (which is a separate goal — do not edit `models.json`,
`envs/*.yml`, or `fetch.py` from this runbook), then re-run the relevant tier.
