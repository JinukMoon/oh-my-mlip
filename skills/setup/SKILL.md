---
name: setup
description: Install an oh-my-mlip MLIP model environment and verify energy+force on GPU with zero human intervention. Triggers on requests to "install", "set up", "setup", or "get working" any MLIP model (MACE, SevenNet, NequIP, Allegro, ORB, UMA, etc.) via oh-my-mlip. Also triggers when a user wants to run a model for the first time and the env is not yet materialized. Also triggers on GENERIC natural-language intent to use a machine-learning interatomic potential — "set up an MLIP", "install an ML potential / machine-learned force field / foundation interatomic potential" — even when no specific model and no "oh-my-mlip" is named; in that case list the registry roster (oh_my_mlip.list_models()) and confirm the model choice (MACE is the quickstart default). Also triggers on requests to install SEVERAL models or ALL models at once — "install everything", "set up all the MLIPs", "MACE and SevenNet and ORB". Also triggers on ROSTER questions — "which MLIPs can I install", "list the available models", "what does oh-my-mlip support" — answered as a pure registry read with no install. Works from ANY working directory: the skill never assumes the current directory is the oh-my-mlip repo. Do NOT trigger on purely informational MLIP discussion (papers, theory, definitions) with no install/run/roster intent.
argument-hint: "<model … | all | all except <model …>>  e.g. MACE-MPA-0 · MACE SevenNet ORB · all · all except UMA eSEN"
---

Defer entirely to `AGENTS.md §9` (installing model environments — single,
multiple, or all), plus `AGENTS.md §5` (gated models) and `AGENTS.md §8`
(error-class recovery policy) that §9 leans on. Read them now and follow
them verbatim; this skill carries no procedure of its own.

Success oracle (zero human prompts, clone to verified compute):
`run_examples/single_point.py <model>` prints energy and forces with the GPU
confirmed in use. `install.sh` exit-0 alone is never sufficient.

Deterministic entry scripts, called in the order `AGENTS.md §9` drives them:
`scripts/setup_survey.py` (state + disk-budget read; `--table <model>` for a
single target), `install.sh` (build / adopt-or-heal),
`scripts/setup_guardrail.py` (stop-condition verdict),
`scripts/setup_verify.py` (compute witness + local-ledger upsert),
`scripts/setup_sweep.py` (batch driver + JSONL report, for `all` or several
targets). Use these scripts instead of re-deriving any fact they compute.

Routing notes:
- Bootstrapping the repo from an arbitrary cwd (locating vs. cloning,
  `OH_MY_MLIP_HOME`/`OMM_HOME`, conda/mamba check, `env.sh`) —
  `AGENTS.md §9.0`.
- A pure roster/listing question installs nothing — `AGENTS.md §9.1`.
- Target resolution for one model, several, `all`, or `all except <...>`,
  and which of those skip the approval gate — `AGENTS.md §9.2`.
- The mandatory human approval checkpoint before any `all`/multi-model build
  starts (survey → render → approve) — `AGENTS.md §9.3`.
- The per-target install → guardrail → verify loop and the batch sweep +
  recovery pass — `AGENTS.md §9.4`.
- Arch-pinned (NequIP/Allegro) first-run compilation — `AGENTS.md §9.5`
  and `§6`.
- Gated models (`gated: true` in `models.json`): never proceed without a
  license + token; never ask for, echo, or store the token value —
  `AGENTS.md §5` / `docs/hf_token.md`.

`/oh-my-mlip:run <model>` and `/oh-my-mlip:catbench` are the next skills once
a model is installed and verified here.
