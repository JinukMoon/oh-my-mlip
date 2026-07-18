---
name: setup
description: Install an oh-my-mlip MLIP model environment and verify energy+force on GPU with zero human intervention. Triggers on requests to "install", "set up", "setup", or "get working" any MLIP model (MACE, SevenNet, NequIP, Allegro, ORB, UMA, etc.) via oh-my-mlip. Also triggers when a user wants to run a model for the first time and the env is not yet materialized. Also triggers on GENERIC natural-language intent to use a machine-learning interatomic potential — "set up an MLIP", "install an ML potential / machine-learned force field / foundation interatomic potential" — even when no specific model and no "oh-my-mlip" is named; in that case list the registry roster (oh_my_mlip.list_models()) and confirm the model choice (MACE is the quickstart default). Also triggers on requests to install SEVERAL models or ALL models at once — "install everything", "set up all the MLIPs", "MACE and SevenNet and ORB" — see <Multi_Target>. Also triggers on ROSTER questions — "which MLIPs can I install", "list the available models", "what does oh-my-mlip support" — answered as a pure registry read with no install (see <Roster_Listing>). Works from ANY working directory: the skill never assumes the current directory is the oh-my-mlip repo (see <Bootstrap>). Do NOT trigger on purely informational MLIP discussion (papers, theory, definitions) with no install/run/roster intent.
argument-hint: "<model … | all | all except <model …>>  e.g. MACE-MPA-0 · MACE SevenNet ORB · all · all except UMA eSEN"
---

<Endpoint>
Goal: `run_examples/single_point.py <model>` prints energy and forces, GPU is
confirmed in use (nvidia-smi compute-apps shows the PID). Zero human prompts
from clone to verified compute. This is the only success oracle; `install.sh`
exit-0 alone is not sufficient, and `scripts/verify_compile.py` is a JSON-shape
lint, not a compute witness.
</Endpoint>

<Deterministic_First>
Every fact this skill needs that CAN be computed by code IS computed by a
script, and the agent must use the script instead of re-deriving the fact:
survey/plan numbers -> `scripts/setup_survey.py`; stop conditions ->
`scripts/setup_guardrail.py`; install state transitions -> `install.sh`;
the verify verdict -> `scripts/setup_verify.py`; batch sweeps + the final
report -> `scripts/setup_sweep.py` (JSONL ledger is the only report source);
the compute witness -> `run_examples/single_point.py`. The agent's own job is
exactly three things: render script output, drive the approval UI, and reason
about failures. Prose ordering promises are not trusted — where ordering
matters, it is enforced by making one script compute all the dependent facts
atomically.
</Deterministic_First>

<Bootstrap>
Preconditions to establish before the loop:

1. Locate or clone the repo.
   - The current working directory plays NO role in resolution: the plugin is
     installed user-scope, so this skill must behave identically from any
     folder. Never assume cwd is (or contains) the oh-my-mlip repo.
   - If `$OH_MY_MLIP_HOME` is set and the path exists, use it.
   - If `$OMM_HOME` is set and the path exists, use it as `OH_MY_MLIP_HOME`.
   - Otherwise clone `https://github.com/JinukMoon/oh-my-mlip.git` into
     `~/.oh-my-mlip` (no specific ref pinned at this skeleton stage; later
     stories will pin a ref).
   - Export `OH_MY_MLIP_HOME` so all child processes inherit it. Do not rely
     on `env.sh:7` auto-detection from relative path.

2. Detect conda/mamba.
   `install.sh` hard-exits if neither is present (`install.sh` lines 187-196).
   Check `which conda || which mamba` before entering the loop. If absent, guide
   the user to install Miniconda (offer a scoped Miniconda install only with
   explicit user consent); never silently mutate the host.

3. Source `$OH_MY_MLIP_HOME/env.sh` to set shared model caches, D3/CUDA
   environment variables, and name-based cache symlinks. This must happen before
   any `install.sh` invocation.
</Bootstrap>

<Read_Order>
Before writing any code or starting the install loop, read these files in the
order specified in `AGENTS.md §0`. Do not rely on memory for model facts:

1. `$OH_MY_MLIP_HOME/models.json` -- model registry (env, interpreter, imports,
   inference code, arch_pinned, gated, weights, license_url, note, status).
2. `$OH_MY_MLIP_HOME/dist_manifest.json` -- env -> HF tarball map used by
   `oh_my_mlip/fetch.py`.
3. `$OH_MY_MLIP_HOME/AGENTS.md` -- agent contract, run branches (§3), gated
   model policy (§5), first-run compilation (§6), recovery strategies.

Resolve ALL model facts from `models.json`; never hard-code them here.
</Read_Order>

<Gated_Model_Gate>
Before entering the loop, check `gated` in `models.json` for the requested
model (policy specified in `AGENTS.md §5`).

If `gated: true`:
- Surface the model's `license_url` and pointer to `docs/hf_token.md`.
- Do NOT proceed to install. Do NOT retry automatically.
- Explain what the user must do (accept license, create HF read token, make it
  available via `huggingface-cli login` or `HF_TOKEN`), then stop.
- **Never ask for, accept, or echo the token value in the chat** — instruct
  the user to run `huggingface-cli login` in their own terminal instead, and
  warn them not to paste the token into the conversation (it would persist in
  the transcript and logs).
</Gated_Model_Gate>

<Roster_Listing>
"Which models can I install?" is answered WITHOUT installing anything.

- Resolve the repo per <Bootstrap> step 1 (clone is cheap; no conda, no GPU,
  no env build is needed for a listing).
- Read `models.json` and report per model: name, framework, `gated`, and the
  per-model validation status. `docs/model_status.md` is the human-readable
  rendering of the same registry if the user wants the full table.
- If Python is available, `oh_my_mlip.list_models()` is the one-liner
  (pure registry read); otherwise read `models.json` directly.
- End the listing by offering the install step (`/oh-my-mlip:setup <model>`
  or `all`), and flag gated models as requiring the user's own HF token.
</Roster_Listing>

<Multi_Target>
The skill accepts one model, several models, or `all`.

Target resolution:
- Several names (`MACE SevenNet ORB`) — resolve each via the registry, keep
  the given order. `install.sh` natively accepts the same multi-target list.
- `all` (or natural-language "everything") — the target list is every model
  family in `models.json` (one representative version per env; `install.sh`
  with no argument builds every recipe).
- Exclusions — `all except UMA eSEN` / `all --except UMA eSEN` (or natural
  language "everything except ...") resolves `all` first, then removes the
  named families from the target list. Report the exclusions in the plan so
  the user sees what was left out.

Survey-plan-approve gate (applies to `all` targets ONLY):
- The survey is ONE deterministic command and it is the FIRST action:
  `python3 $OH_MY_MLIP_HOME/scripts/setup_survey.py`
  It computes ATOMICALLY every fact the plan needs — per-env state, the disk
  budget counting ONLY envs that will actually be built (ready envs cost
  zero), leak-safe token availability, and the gated list. The agent renders
  this output; it must NOT recompute, reorder, or partially re-derive any of
  it. Never open with "not enough disk, what do you want to do?" — that
  question, asked pre-survey, is exactly the failure mode this design
  prevents (its numbers are wrong until the survey has run).
- RENDER BEFORE ASKING: the survey table must be shown to the user as
  visible output BEFORE the approval question appears — never summarized
  away into the question options. Show, in this order: (1) the resolved
  `OH_MY_MLIP_HOME` (so a fresh-clone-vs-existing-hub mixup is visible at a
  glance), (2) the per-env table — ready (will skip) / partial (will
  adopt-or-heal) / missing (will build) — plus gated models (skipped
  without a token) and exclusions, (3) the survey's disk verdict
  (`disk.fits`). The approval question comes AFTER this table and must
  restate the counts in its text ("N ready — skipped, M to build, K gated").
  A question whose options mention installing models the table shows as
  `ready` is a contract violation — ready envs are never proposed for
  install.
- If the post-survey budget still does not fit the free space, that is part
  of the SAME plan-approval question, not a separate upfront alarm: show how
  many envs fit, and let the selection UI (below) drive which ones make the
  cut.
- Then STOP and ask for approval. This is the one deliberate human checkpoint
  in an `all` sweep — a full-registry install is a 100+ GB, multi-hour
  commitment the user should see before it starts. This is an agent-first
  hub, so use the host agent's NATIVE interactive question UI whenever one
  exists (in Claude Code, the structured question tool with selectable
  options) — not a plain-text "reply yes" prompt:
  - Question 1 (single-select): (a) install everything in the plan,
    (b) only what is missing/broken, (c) choose models to exclude.
  - If (c): follow up with MULTI-SELECT questions listing the pending models
    as options, batched to fit the UI's per-question option limit (batch by
    a sensible grouping — e.g. gated / large-disk / the rest); each answer
    marks exclusions. Free-text answers ("skip everything except MACE") are
    always honored too.
  - Only when no interactive question UI exists (plain MCP callers, other
    agents) fall back to a text plan + text approval.
  After approval the sweep runs to completion with zero further prompts (the
  <Endpoint> zero-prompt contract applies from approval onward).
- Token check for gated targets: the survey output already reports it
  (`token.available` / `token.source`, probed in the `fetch.py` resolution
  order without ever reading the token value).
  If gated models are in scope and NO token is found, the plan must include a
  token request that spells out the LITERAL commands and URLs for the user to
  run/visit (sourced from `docs/hf_token.md`, not paraphrased): the
  `license_url` of each gated model in scope, the token-creation page
  (huggingface.co/settings/tokens, role: read), and the exact command to run
  **in the user's own terminal**:
  `huggingface-cli login`
  (alternative: `export HF_TOKEN_PATH=/path/outside/repo/token`).
  **Tell the user explicitly: NEVER paste the token into this chat.** Anything
  typed here lands in the conversation transcript and logs; the token must
  reach the machine out-of-band (the interactive `huggingface-cli login`
  prompt is the leak-safe path). The agent must never ask for the token value,
  never echo it, and never write it into any file inside the repo. The user
  can say "token is set" afterwards and the sweep re-checks and proceeds.
- The user may approve a subset ("skip the last three") — treat that as an
  exclusion list and restate the final targets in one line before starting.
- Single-model and explicitly-listed targets do NOT get this gate: naming the
  models IS the approval; proceed zero-prompt as always.

Batch execution (deterministic driver, complete-then-batch-recover):
- After approval, run the sweep DRIVER once:
    `python3 $OH_MY_MLIP_HOME/scripts/setup_sweep.py --targets <M1,M2,...>`
  The driver enforces what used to be prose rules: sequential one-env-at-a-
  time execution, `skipped_gated` bookkeeping for gated targets without a
  token, never stopping on a failed target, a 10 GB disk-floor check before
  each target (below it, this and all remaining targets are recorded
  `skipped_disk` and the sweep ends — guaranteed failures are not attempted),
  and one JSONL ledger line per phase under `.sweep/`. Do NOT iterate
  install.sh over targets yourself.
- After the sweep completes, run the recovery pass: for each ledger entry
  whose status is `failed`, work that ONE target through the
  <Self_Healing_Loop> (Steps 1-4; the guardrail bounds still apply per
  target). Recovery reasoning stays agent-owned (AGENTS.md §8).
- Finish with `python3 $OH_MY_MLIP_HOME/scripts/setup_sweep.py report`.
  The final report comes STRICTLY from the ledger — never compose it from
  memory of what happened; targets absent from the ledger appear as
  `not_attempted` (no silent truncation).
- A single-target gated request still halts per <Gated_Model_Gate>; inside a
  batch the driver's `skipped_gated` recording replaces the halt.
</Multi_Target>

<Self_Healing_Loop>
Entry condition: bootstrap complete, conda present, model is not gated.

Step 0 -- Take stock FIRST (always, for every target):
  Before any install attempt, check the target's install state:
  `python3 $OH_MY_MLIP_HOME/scripts/setup_survey.py --table <model>`
  (read-only; reports ready / partial / broken / not installed per env, with
  state rules identical to `install.sh --status`).
  - `ready` -> do NOT install. Jump straight to Step 4 (verify). If the
    verification passes, report "already installed and verified" and stop —
    the whole call cost one single-point run. If it fails, fall into Step 1
    (install.sh's adopt-or-heal will repair the env, never duplicate it).
  - `partial` / `broken` / `not installed` -> proceed to Step 1; say which
    state was found so the user knows why an install is happening.
  This ordering is what makes repeated setup calls cheap and honest: the
  answer to "is it already there?" always comes before any mutation.

Step 1 -- Install:
  Run `bash $OH_MY_MLIP_HOME/install.sh <model>` (with `OH_MY_MLIP_HOME`
  exported). Capture stdout and stderr.

Step 2 -- Check guardrail:
  Capture the attempt's raw stderr to a file, then call the combined gate.
  The SCRIPT owns stderr normalization — never pre-normalize, summarize, or
  hash the stderr yourself before passing it:
    `python3 $OH_MY_MLIP_HOME/scripts/setup_guardrail.py gate \
        --state <state-file> --ceiling-gb 30 --stderr-file <stderr-file>`
  The helper prints ONE JSON verdict and always exits 0 — parse the JSON,
  not the exit code. Verdicts:
  - `guardrail_halt` (disk headroom below ceiling; highest priority) and
    `wallclock_halt`: stop unconditionally; surface the message.
  - `stalled` (same normalized signature recurred >= 2) and
    `stalled_cumulative` (attempt cap): stop and switch to the docs-request
    path (AGENTS.md §8).
  - `ok`: continue.

Step 3 -- On install failure: recover.
  - Read the traceback.
  - Classify the error and select a recovery strategy from `AGENTS.md §6`.
    The strategy table lives in `AGENTS.md`, not here; do not reproduce it.
  - Apply the strategy (e.g. select alternate arch, clean partial env, retry
    with adjusted flags).
  - When the failure is a missing/unclear install recipe or weight source (the
    fetch path in `models.json` is absent or wrong and the official page does
    not resolve it), do NOT loop indefinitely: make one bounded attempt, then
    follow the docs-request path in `AGENTS.md §8` (ask the user for the
    official docs/link, mark the version note `awaiting user docs:`). Likewise,
    if a guardrail stop (`stalled` / `stalled_cumulative` / `wallclock_halt` /
    `guardrail_halt`) has tripped, switch to that docs-request path instead of
    retrying.
  - Otherwise return to Step 1.

Step 4 -- On install success: verify with ONE command.
  Run:
    `python3 $OH_MY_MLIP_HOME/scripts/setup_verify.py <model> --json`
  It preflights the driver-skew predicate to choose the device, runs the
  single_point witness, samples GPU PIDs with descendant attribution (the
  compute PID is a worker grandchild — do not run nvidia-smi yourself), and
  prints ONE JSON verdict:
  `{pass, device, degraded, reason, energy_ev, fmax_ev_a, forces_shape,
  gpu_pid_confirmed}` — exit 0 iff `pass`. Render the verdict; NEVER
  re-judge it or run a second GPU check.
  - `pass:true, degraded:false` -> verified on GPU.
  - `pass:true, degraded:true` -> pass-with-caveat: report device=cpu and
    the verdict's computed reason (legitimate driver skew, AGENTS.md §1.9).
  - `pass:false` -> return to Step 3 using `reason` as the error signature.

Step 5 -- Declare success.
  Report the verdict fields as-is: model name, env path, energy_ev,
  forces_shape, device, degraded (with reason when true), gpu_pid_confirmed.
  The guardrail helper will have already confirmed disk headroom
  and absence of a stall signature.
</Self_Healing_Loop>

<Arch_Pinned_Models>
For models where `arch_pinned: true` in `models.json` (e.g. NequIP, Allegro),
the install loop must also handle first-run compilation of the `.pt2` artifact.
See `AGENTS.md §6` for the compilation flow, artifact locations
(`models/compiled/{sm86,sm89}/`), and the arch-selection logic. This path is
the Phase 2 headline gate; the loop body here applies equally, but checkpoint
acquisition and `scripts/compile_nequip.sh` invocation are Phase 2 work.
</Arch_Pinned_Models>

<Related_Skills>
`/oh-my-mlip:run <model>` and `/oh-my-mlip:catbench` are thin pointers to
`AGENTS.md §3A` and `§3B` respectively. They are added by later stories and are
not part of this skeleton.
</Related_Skills>
