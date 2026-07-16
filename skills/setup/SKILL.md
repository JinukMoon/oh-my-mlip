---
name: setup
description: Install an oh-my-mlip MLIP model environment and verify energy+force on GPU with zero human intervention. Triggers on requests to "install", "set up", "setup", or "get working" any MLIP model (MACE, SevenNet, NequIP, Allegro, ORB, UMA, etc.) via oh-my-mlip. Also triggers when a user wants to run a model for the first time and the env is not yet materialized. Also triggers on GENERIC natural-language intent to use a machine-learning interatomic potential — "set up an MLIP", "install an ML potential / machine-learned force field / foundation interatomic potential" — even when no specific model and no "oh-my-mlip" is named; in that case list the registry roster (oh_my_mlip.list_models()) and confirm the model choice (MACE is the quickstart default). Do NOT trigger on purely informational MLIP discussion (papers, theory, definitions) with no install/run intent.
argument-hint: "<model name e.g. MACE-MPA-0>"
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
</Gated_Model_Gate>

<Self_Healing_Loop>
Entry condition: bootstrap complete, conda present, model is not gated.

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
