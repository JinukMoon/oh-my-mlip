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
- Before installing ANYTHING, take stock: run `install.sh --status` and read
  the registry. Present one plan table — per env: ready (will skip) / partial
  (will adopt-or-heal) / missing (will build), plus gated models (skipped
  without a token), exclusions, and the disk budget vs free space.
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
- Token check for gated targets: while surveying, detect whether an HF token
  is already available (the `fetch.py` resolution order: `HF_TOKEN` env →
  standard `huggingface_hub` cache/`HF_TOKEN_PATH` → `OMM_HF_TOKEN_FILE`).
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

Batch rules (the per-model contract is unchanged — each target still goes
through <Gated_Model_Gate> and the full <Self_Healing_Loop>):
- Run targets SEQUENTIALLY, one env at a time — never parallel conda solves
  (disk and GPU contention corrupt the error signal the loop depends on).
- Gated models inside a batch are SKIPPED with the <Gated_Model_Gate> notice
  recorded, not a batch halt. A single-target gated request still halts as
  specified there.
- Scale the disk precheck: budget ~10 GB per remaining env against the free
  space before starting each target; if the budget no longer fits, stop the
  batch and report which targets were not attempted.
- A target whose loop ends in `stalled`/`guardrail_halt` fails ONLY that
  target; continue with the next one.
- Final report: one line per target — verified (energy + GPU PID) / skipped
  (gated / disk) / failed (last error signature) — so a partial sweep is
  honest about what is and is not usable.
</Multi_Target>

<Self_Healing_Loop>
Entry condition: bootstrap complete, conda present, model is not gated.

Step 0 -- Take stock FIRST (always, for every target):
  Before any install attempt, check the target's install state:
  `bash $OH_MY_MLIP_HOME/install.sh --status` (read-only; reports
  ready / partial / broken / not installed per env).
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
  After each attempt (success or failure), call:
    `python $OH_MY_MLIP_HOME/scripts/setup_guardrail.py \
        --free-disk-gb <ceiling> \
        --error-sig "<normalized_stderr_signature>" \
        --attempt <n>`
  The helper returns structured JSON: `{status: "ok"|"stalled"|"guardrail_halt",
  message: "..."}`.
  - `guardrail_halt`: stop unconditionally; surface the message.
  - `stalled`: same normalized error signature has recurred >= N=2 times; stop
    and report.
  - `ok`: continue.

  NOTE: `scripts/setup_guardrail.py` is added by a later story (Phase 1 step 3
  in the plan). Until it exists, the agent must enforce the same bounds manually:
  halt on the same stderr signature repeating twice; check `df -h` against a
  30 GB headroom default before each attempt.

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

Step 4 -- On install success: verify.
  Run:
    `$OH_MY_MLIP_HOME/envs/<env>/bin/python \
        $OH_MY_MLIP_HOME/run_examples/single_point.py <model>`
  AND assert GPU is in active use:
    `nvidia-smi --query-compute-apps=pid --format=csv,noheader`
  Confirm the Python process PID appears in the nvidia-smi output.

  Both conditions must hold. If only one holds, treat as failure and return to
  Step 3 with a descriptive error signature (e.g. "gpu_not_used").

Step 5 -- Declare success.
  Report: model name, env path, energy value, forces shape, GPU PID confirmed.
  The guardrail helper (when present) will have already confirmed disk headroom
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
