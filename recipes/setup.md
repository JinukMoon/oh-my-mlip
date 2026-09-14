# Recipe: install and verify model environments

Mechanics live in `AGENTS.md §9` (bootstrapping, roster listing, target
resolution, the survey → approve → sweep loop), `§5` (gated weights), `§6`
(first-run compilation) and `§8` (error-class policy). This recipe adds the
job protocol from `recipes/README.md`, the executable chain that owns every
package decision, and the fresh-versus-adopted distinction; it repeats none
of the mechanics.

## Executable chain

The user's words choose the targets and the kind of proof. Every package,
version, order and host branch below is fixed by a file, and the agent
runs those files unchanged:

| Step | Owner file | What it fixes |
|---|---|---|
| target → env | `install.sh` resolves a model or env name through the `env` field of `models.json` | which env recipe builds |
| env creation | `envs/<env>.yml` — one conda solve with pinned python, pinned `torch==X.Y.Z+cuNNN` from the matching wheel index, framework pinned to a release or git SHA; or `envs/<env>.build.sh` — an ordered multi-pass sidecar with one pinned pass per numbered step and a named exit code per failure | the exact package combination and its order |
| pin expectation | `envs/_expected.json` (python + CUDA-runtime tag per env), checked GPU-free by `python3 scripts/lint_recipes.py` | recipe drift before any solve |
| benchmark package | `CATBENCH_PIN` in `install.sh`, a post-create step in every env; `OMM_CATBENCH_VERSION=<ver>` overrides it for one build | catbench version |
| weights | `scripts/prestage_<env>_weights.py` (stdlib fetch) and `scripts/prepare_<env>_weights.py --target-root models/<env>` (env-interpreter transform), each only where the file exists | checkpoint staging |
| first-run kernels | the D3 warm-up inside `install.sh` when `nvcc` is on `PATH`; arch-pinned artifacts per `docs/arch_first_run_compile.md` | compiled extensions |
| ready marker | `envs/<env>/.omm_ready`, written by `install.sh` after the env solve and the catbench post-step succeeded; the weight pre-stage, weight prepare and D3 warm-up may fail non-fatally before it (`install.sh` prints a "skipped/failed" line for each and continues) | adopt-or-heal resume point — proof that the build finished, not that inference works; `scripts/setup_verify.py` is that proof |
| shell facts | `env.sh` (cache routes, `CUDA_HOME`, `PYTHONUTF8`) | the environment every step sees |
| verdict | `scripts/setup_verify.py` running `run_examples/single_point.py --json` in the env | pass / degraded / fail per variant |
| batch record | `scripts/setup_sweep.py` → `.sweep/setup_sweep_<NNNN>.jsonl` | one row per phase per target |
| fresh-install cycle | `scripts/setup_sweep.py --fresh-root` — per env, in a fixed order: mandatory-hash targets read (the real hub's local state and the user envs, compared again at the attribution step) → measured start gate (free space ≥ conservative peak + the measured cold size of the host's `$HOME` caches, which an isolated `HOME` re-downloads, + the first-run fine-tune downloads + `--start-reserve-gib`; the peak is the preflight's estimate or the operator's `--peak-gib`, and with neither the `budget` row is `resource-blocked` ("no conservative peak estimate … the measured floor alone is not a budget") — there is no floor-only pass. The cold-cache measurement is read-only on the host `HOME`; known cache symlinks are resolved and measured (provenance `resolved`), while a dangling link, a link to a file or an unreadable subtree makes it `partial`, which is `resource-blocked` with `home_cold_cache_provenance: "unknown(partial_measurement)"` unless `--cold-cache-gib` supplies an operator estimate — a complete measurement is never replaced by an operator value. Budget evidence keys: `peak_estimate_gib`, `peak_estimate_provenance` (`preflight` / `operator_estimate` / `unknown`), `estimate_status`, `home_cold_cache_gib`, `home_cold_cache_provenance`, `first_run_ft_downloads` (per item: provenance, `copies`, `measured_gib`; provenance `measured_host:<path> x<copies>` when copies ≠ 1), `first_run_ft_downloads_gib`, `first_run_ft_downloads_provenance` (`measured_host` / `measured_host+operator_estimate` / `operator_estimate` / `partial(unknown items excluded)`), `first_run_ft_downloads_operator` {`gib`, `covers_unknown`, `semantics`}, `gate_gib`, `floor_gate_gib`, `start_reserve_gib`, `min_free_gib`. The first-run download table names nine families: MatterSim (`.local/mattersim/pretrained_models`, x1), TACE (`.cache/tace`, x2) and NequIP (`.nequip/model_cache`, x2, a whole-cache upper bound) are measured on the host; GRACE, Allegro, PET, MACE, DeePMD and DPA4 are unknown, not zero (items named "… audit NOT completed (FT owner 2026-09-14) -- unknown, not zero"). `--ft-download-gib` is added to the measured known part and is the only accepted bound for the unknown items; without it any unknown item blocks the FT phase ("… (no --ft-download-gib estimate given; unknown is never zero)"), and `--peak-gib` never covers downloads) → working-tree snapshot check → snapshot → materialize (re-checks the tree; no skip flag exists) → child environment built (the campaign environment row below; routes, install, verify and the fine-tune sweep all run with it) → seed-cache (copy only) → routes verify → isolation pre-check (the runtime copy's own `resolve()`) → `install.sh` inside the root, free space re-measured while it runs and the build stopped in order below `--min-free-gib` → isolation post-check → inventory (`conda list --explicit`, `pip freeze`, `pip list --format json` and the registry resolve under the owned interpreter, line-sanitized before any write, into `<root>/.sweep/inventory/<env>.{conda_explicit.txt,pip_freeze.txt,pip_list.json,resolve.json}` — required preserve artifacts; empty, malformed or non-zero output is `failed(inventory)`, after which preserve is `failed(preserve:required_missing)` and cleanup is withheld; a missing prefix is `failed(inventory:prefix_missing)`; a disk-floor trip during it is `resource-blocked` with the abort reason "during inventory") → `setup_verify` per variant with `--no-local-record` → `scripts/ft_sweep.py` (an `ft_run` exit 5 is `failed(seed_unhonoured)`, a refusal class the harness never reissues with `--seed` / `--allow-partial-seed`, out dir left empty; `ft_run` exit 0 without a readable `<out>/ft_run.json` JSON object — missing, a symlink, unreadable — is `failed(no_provenance_record)`; a gated checkpoint with no token source present is `access-blocked` on a presence-only check, nothing fetched — see §5; FT rows carry `ft_run_json`, `ft_run_json_sha256`, `ft_run_json_schema`, `ft_verify_reason`, `ft_verify_version` (null when absent), a verify-failed tail reading "<why> (ft_verify reason: <reason>)" when both exist, and the fallback `CKPT_GLOBS` mirrors `ft_run.FAMILY_CHECKPOINT_GLOBS` exactly, pinned by a test, used only when the shipped table cannot be parsed) → source-hash verify → attribution hashes of the real hub and user envs → budget row → preserve (verified copies of ledgers, logs, configs and named artifacts; a passed FT row must preserve its ckpt, sh AND `ft_run.json` — one missing at preserve time is `failed(preserve:required_missing)`, cleanup refused with `evidence_not_durable`, root retained) → guarded cleanup (result key `removal` {`files`, `dirs`, `removed`}) → post-cleanup row; ledger at `.sweep/campaign_<id>/ledger.jsonl`, every row carrying `manifest_sha256` | fresh-install proof without touching the user's hub or envs |
| fresh-root primitives | `scripts/fresh_root.py` (`allowlist`, `snapshot`, `materialize`, `seed-cache`, `verify`, `preserve`, `cleanup`), the building blocks the cycle calls; used directly only to inspect, preserve from, or clean up a retained root. `materialize` writes the campaign-external ownership record (`--owned-registry`, default `<campaign dir>/owned_roots.json`); `preserve --runtime <root> --dest <outside dir> --file <artifact>` treats every named artifact as required (`--require` is the same flag) — one missing, symlinked or outside-the-root artifact is `failed(preserve:required_missing)` with nothing partial written; `cleanup` requires `--owned-registry` and `--allowed-parent` and refuses a root not bound in that record, whose direct parent is not the canonical allowed parent, or that sits behind a symlink, and its quiescence scan refuses a root with a live process (`--owned-group <sid>`, repeatable, names launched phase children — `failed(cleanup:in_use)`), open file descriptors or mapped files inside the root, and fails closed when `/proc` cannot be read (`failed(cleanup:proc_unreadable)`). The `/proc` scan classifies PIDs as users / inspected / uninspectable / vanished / unknown_use / excluded_by_perm with `quiescence: total|partial`, recorded on the cleanup row under the `quiescence` key; a vanished PID (ENOENT/ESRCH) is tolerated, while any uninspectable or unknown-use PID refuses cleanup with nothing deleted (`failed(cleanup:proc_uninspectable)`, `failed(cleanup:proc_unknown_use)`; an EACCES `/proc` entry is uninspectable, so cleanup is refused without deletion). Before any `rmtree` a mountinfo + `st_dev` walk refuses a mount inside the root (`failed(cleanup:mount_inside_root)`; mount dirs are never descended) and an unreadable mount table or subtree (`failed(cleanup:mounts_unreadable)`, also raised when the mount table re-read inside removal is unreadable). Removal itself can stop part way: `failed(cleanup:remove_stopped)` with detail {`at`, `reason`: `mount_point_appeared` | `mount_or_device_boundary` | `unreadable_during_listing` | `"os_error: …"`, `removed`: exact count}, root retained — "root retained" means the root directory and not-yet-processed entries remain; entries already unlinked before the stop are gone (files are removed before directories). The mount checks are point-in-time and removal is pathname-based, not race-proof; `materialize` refuses a root that is not mode 0700 (`failed(materialize:root_not_private)`); preserve evidence lists `bundle_dirs` (SavedModel directories, including an empty `assets/`). `allowlist` is the plan Part 3.1 list plus main-approved additions (`tests/test_relax_input.py`, `tests/test_ft_verify.py`, `tests/test_catbench_report.py`, `tests/test_catbench_quickstart.py`, `docs/host_sessions/*.md`); `setup_sweep.py` passes the session ids itself | one owned, disposable runtime root from a content-addressed snapshot of the working tree — never a `git` clone of `HEAD`, which would miss uncommitted edits |
| campaign environment | every phase runs with `HOME=<root>/home` (an isolated campaign home), `HUGGINGFACE_HUB_CACHE` / `TRANSFORMERS_CACHE` / `HF_ASSETS_CACHE` / `HF_DATASETS_CACHE` / `CONDA_ENVS_PATH` / `OMM_HOME` unset, and `HF_TOKEN_PATH` / `CONDARC` pointing at the user's real files (path exports, never copies); `<root>/.omm_fresh_env.sh` carries the same exports and unsets, and `fresh_root.py verify --routes` expects `HOME`, `HF_HUB_CACHE`, `CONDA_ENVS_DIRS`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH`, `TORCHINDUCTOR_CACHE_DIR` inside the root | nothing a campaign writes lands in the user's home or caches; a recipe that sources `.omm_fresh_env.sh` never also sets `HOME` |
| verdict table | `scripts/evidence_report.py --ledger <ledger> [--models <models.json>] [--strict] [--audit <json>] [--bundle <dir>]` | the five-state accounting over a campaign ledger |

Host and arch branches already inside those files — the agent names the
branch, never adds one: a host driver below the recipe's CUDA runtime →
`install.sh` prints a driver-skew warning and the env runs CPU-only;
`nvcc` absent → D3 is left off and the MLIP still runs; an arch-pinned
model → `--arch` on `run_examples/single_point.py`; a gated checkpoint →
the token rules of `AGENTS.md §5`.

Anything the chain does not do — another package, a different version, a
dependency workaround, an edit inside an env prefix — is not done during
a job. It ends as `failed(<class>)` with the log, and the remedy is a
reviewed change to the owner file followed by a rebuild through
`install.sh`.

## 1. Ask

Required inputs (skip any the user already gave):

- **Targets** — one model, several, `all`, or `all except ...`. A bare "set
  up an MLIP" gets the roster (`oh_my_mlip.list_models()`) and a confirmed
  choice.
- **Kind of proof** —
  (a) *use*: an existing healthy env may be adopted (`install.sh`
  adopt-or-heal, or `scripts/adopt_env.py <model> <prefix>` for an env the
  user built elsewhere). This is the default.
  (b) *fresh-install proof*: the env must be built from its recipe inside an
  isolated runtime root that holds no prior envs. Only when the user asks
  for fresh evidence; it costs a full download and build.
- **Gated targets** — whether the upstream licence is accepted and a read
  token is reachable (the token value itself is never requested or echoed).

Do not open with a disk question; the survey answers it.

## 2. Plan

1. `python3 scripts/setup_survey.py --table <targets...>` — per-env state,
   disk budget for the envs that will actually build, token presence.
2. `bash install.sh --dry-run <targets...>` — prints, per target, whether
   the single-solve recipe or the multi-pass sidecar builds it, the catbench
   pin, the weight steps that will follow, and any driver-skew warning.
   That output is pasted into `PLAN.md` unedited: it is the install plan.
3. `python3 scripts/lint_recipes.py` — the recipes' structural check; a
   failure here stops the plan before any solve.
4. Write `PLAN.md`: targets in order; for each, the survey state and the
   resulting action (verify only / install / adopt-or-heal); the variants
   that will be witnessed (every version of the env in `models.json`, named
   individually); expected outputs (`envs/<env>/bin/python`, `.omm_ready`,
   verdict JSON per variant, a `models.local.json` row per PASS, the sweep
   ledger path); the disk floor; exclusions; the hub commit
   (`git rev-parse HEAD`) and, when the tree is dirty, `git status
   --porcelain` verbatim.
5. Fresh-install proof, only when asked. One command owns the whole
   ordered cycle; the plan quotes it with every option filled in:

   ```bash
   python3 scripts/setup_sweep.py --fresh-root --targets <env> --campaign-id <id> \
       --preflight <preflight.json> --ft-dataset <frames> --ft-audit <audit.json> \
       [--seed-spec <seed.json>] [--runtime-parent <dir>] \
       [--min-free-gib <G>] [--start-reserve-gib <G>] [--no-cleanup] \
       [--peak-gib <G>] [--cold-cache-gib <G>] [--ft-download-gib <G>]
   ```

   Disk figures are GiB. The runtime root lands under `--runtime-parent`
   (default: `.omm_fresh` next to the hub), the ledger under
   `.sweep/campaign_<id>/ledger.jsonl` outside every root. The start gate
   measures free space itself — an estimate in the plan is not a
   measurement; `--peak-gib`, `--cold-cache-gib` and `--ft-download-gib`
   are operator estimates recorded as such (`operator_estimate`
   provenance) and each one quoted in the plan — `--peak-gib` and
   `--cold-cache-gib` accepted only where no measurement or preflight
   value exists, `--ft-download-gib` added to the measured known
   downloads as the only bound for the unknown families, never a
   substitute for `--peak-gib` — and a build that drives the
   host below `--min-free-gib`
   is stopped in order and its remaining variants recorded
   `resource-blocked`. `--seed-spec` names native-cache files to COPY
   (never link) into the root so a checkpoint is not downloaded twice;
   each item is listed in the plan. Without `--ft-dataset` the cycle's
   fine-tune rows cannot pass — say so in the plan when fine-tuning is
   not part of the proof. `--no-cleanup` keeps the root for inspection
   and records the cleanup row as retained. The manual primitives
   (`python3 scripts/fresh_root.py allowlist|snapshot|materialize|seed-cache|verify|preserve|cleanup ...`,
   contract as `--help` prints) are for inspecting, preserving from or
   removing a retained root, not a substitute for the cycle. To exercise
   the candidate's own skills and `AGENTS.md` inside a retained root, the
   host is pointed at that root as `recipes/README.md` describes
   (candidate-local discovery); the default-installed plugin is not it.
6. One env at a time. Two builds never overlap (the header constraint of
   `AGENTS.md`); a batch is serialized by `scripts/setup_sweep.py`.

## 3. Approve

- A single or explicitly listed target is approved by being named
  (`AGENTS.md §9.2`).
- `all` or a multi-target batch goes through the `§9.3` gate with the
  survey table rendered first.
- A fresh-install root always needs an explicit yes, whatever the target
  count, and so does every `seed-cache` item.

## 4. Execute

- Single target: `bash install.sh <model>` with stdout and stderr captured
  to `<work>/install_<env>.log` → on failure
  `python3 scripts/setup_guardrail.py gate --state <state.json> --ceiling-gb 30 --stderr-file <stderr.txt>`
  → the `§8` recovery for the class it names (free disk, supply a token,
  rerun) → `bash install.sh <model>` again, which adopts what was built
  and resumes the post-steps.
- Batch: `python3 scripts/setup_sweep.py --targets M1,M2,...` then
  `python3 scripts/setup_sweep.py report`.
- Verdicts, every variant and not only the family default:
  `python3 scripts/setup_verify.py <Family> --all-variants --json` (one
  verdict per version plus a summary object), or
  `python3 scripts/setup_verify.py <Version> --json` per version. Add
  `--no-local-record` when the job is a read-only witness that must leave
  `models.local.json` untouched. Keep each JSON verdict as a file under
  the work dir.
- A failure outside the guardrail's classes — a solver conflict, a wheel
  missing for this platform, an import error after the build — is
  recorded `failed(<class>)` with the log excerpt. The remedy is a change
  to `envs/<env>.yml`, `envs/<env>.build.sh` or `install.sh`, reviewed on
  its own, then a rebuild. No package is added, removed or upgraded inside
  the env by hand, and no sentinel is written by hand.
- Fresh root: the `--fresh-root` cycle verifies sources, hashes the real
  hub's local state and the user envs before and after (the attribution
  row records `targets_before` / `targets_after` / `changed` with
  `attribution: "unknown"` on a change — no writer is claimed; a change
  pauses the campaign as `failed(attribution)`), refuses to start next
  to another campaign and fails closed when `/proc` cannot be read
  (`failed(concurrency:proc_unreadable)`), and cleans up its own root at
  the end. A root that was retained (`--no-cleanup`, or a recorded
  cleanup refusal) is removed only by
  `python3 scripts/fresh_root.py cleanup --runtime <root> --evidence-ledger <ledger> --preserved <preserved dir> --owned-registry <campaign dir>/owned_roots.json --allowed-parent <runtime parent>`,
  which refuses any root the ledger or the ownership record does not
  own, any root whose direct parent is not the allowed parent, and any
  symlinked root or parent; a root materialized without an ownership
  record can never be removed by the tool. Preserve runs first and
  requires every fine-tune final artifact by name, plus `.catbench/` and
  `.distill/` under the root (jsonl, log, sh, json, yaml/yml, txt, csv,
  md). Nothing else removes a root, and never the user's hub,
  `~/.cache/huggingface`, or any conda package cache.

## 5. Verify

Evidence per variant is the `setup_verify.py --json` verdict, whose fields
`AGENTS.md §9.4` step 4 lists; the three that decide the state are `pass`,
`degraded` and `reason`. Map the verdict onto the result states of
`recipes/README.md`:

| Evidence | State |
|---|---|
| `pass:true`, `degraded:false` | `passed` (GPU energy + forces) |
| `pass:true`, `degraded:true` (CPU fallback, driver below the CUDA runtime) | `resource-blocked` (driver); report `reason` — not a GPU pass |
| sweep ledger `skipped_gated`, or an FT row classified "access prerequisite missing/unverified: gated checkpoint and no HF token source present (HF_TOKEN / HF_TOKEN_PATH / hf cache / OMM_HF_TOKEN_FILE); no remote denial observed (nothing fetched)" with audit `access` {prerequisite `hf_token_source`, status `missing/unverified`, checked `presence_only`, `remote_denial_observed: false`, `fetched: false`} | `access-blocked` — a presence-only token check, never a remote denial |
| FT row `failed(seed_unhonoured)` (`ft_run` exit 5) | `failed` — refusal, not reissued with `--seed` / `--allow-partial-seed` |
| FT row `failed(no_provenance_record)` (`ft_run` exit 0, `<out>/ft_run.json` missing / symlink / unreadable / not an object) | `failed` — no fine-tune evidence without its record |
| sweep ledger `skipped_disk` | `resource-blocked` (disk) |
| `pass:false`, or install stopped by the guardrail | `failed` (`reason` / stop verdict as the class) |
| no ledger row | not attempted — say so |

The record for a job carries: the hub commit; the recipe file that built
each env (`envs/<env>.yml` or the sidecar) with its sha256; the
`CATBENCH_PIN` value in force; the `install.sh --dry-run` output; the
install logs; the verdict JSON per variant with its `local_record` field
(`recorded` or `skipped(--no-local-record)`); the sweep ledger path; for a
fresh root, the campaign ledger with `manifest_sha256` on every row.

For a campaign ledger the table itself is printed by
`python3 scripts/evidence_report.py --ledger .sweep/campaign_<id>/ledger.jsonl --strict [--audit <audit.json>] [--bundle <dir>]`:
`--strict` exits zero only when every required (env, kind, variant) row
is `passed` — a degraded CPU pass fails strict unless `--allow-degraded`
is given deliberately and disclosed; `--bundle` writes
`evidence_bundle.md` and `evidence_bundle.json`, for a paused campaign as
much as for a finished one. The report's states are rendered as printed;
the agent does not re-total them.

Two kinds of evidence are labelled separately in the report and never
merged: a verdict from an adopted env proves the model runs here; only a
verdict from the fresh-install root proves the recipe builds from scratch.
Historical verdicts from another host or campaign are claims, not evidence
(`AGENTS.md` header constraint 7).

## 6. Out of scope

- Building on a remote machine, deleting any env this job did not create,
  purging caches, installing conda or a CUDA toolkit — each needs its own
  renewed approval (`§8` conda row).
- Retrying past the guardrail stop set is not a choice the agent makes; the
  docs-request path of `§8` follows instead.
- Changing a recipe pin, adding a package, or bumping `CATBENCH_PIN` —
  a separate reviewed change, never part of a setup job.

## Planned helpers (not implemented)

None remaining for this recipe: the consensus plan's setup targets
(`fresh_root.py`, the `--fresh-root` cycle, `--all-variants`,
`--no-local-record`, `evidence_report.py`) all exist and are named above.
Not implemented anywhere, and not to be described as existing: a real
end-to-end campaign run, a skip flag for the materialize re-check, any
decimal-GB disk flag.

Landed without a unit test: none for this recipe (every helper named
above has a dedicated test under `tests/`; a test's existence is not an
independent review of it).

Missing executable parts (described by function; no owner file yet):

- capturing the installed package inventory of an adopted (non-fresh-root)
  env into the job record — the fresh-root cycle's `inventory` phase now
  writes the resolved closure for its own owned env, but a single
  `install.sh` target or a plain sweep still records only the recipe
  file's sha256 and the pins;
- a lock of that closure per env, so a rebuild resolves the same set
  rather than the same pins;
- changing the catbench version an existing env holds — today only a
  rebuild with `OMM_CATBENCH_VERSION` set changes it (the mismatch itself
  is reported by `scripts/catbench_version.py`, see `recipes/catbench.md`).
