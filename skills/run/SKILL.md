---
name: run
description: Run a single-point energy/forces calculation or structure relaxation with any oh-my-mlip MLIP. Triggers on requests to run, compute, calculate, or relax a structure with a specific model (MACE, SevenNet, NequIP, ORB, etc.) via oh-my-mlip. Also triggers on GENERIC natural-language compute intent — "single-point this structure with an MLIP", "relax this POSCAR with a machine-learning (interatomic) potential / ML force field" — even when no specific model and no "oh-my-mlip" is named; when no model is given, present oh_my_mlip.list_models() and confirm the choice (MACE is the quickstart default). Do NOT trigger on purely conceptual MLIP questions with no structure/compute intent.
argument-hint: "<model name> [--relax] [--d3] [--structure PATH]"
---

Defer entirely to `AGENTS.md §3A` (run branches — single-point / relax).

Read `AGENTS.md §3A` now. Do not reproduce its content here; follow it verbatim.
Key entry points: `oh_my_mlip.run()` for cross-env single calls; `Worker` for
repeated calls; `resolve()` when emitting a standalone script.

If the model env is not yet materialized, run `/oh-my-mlip:setup <model>` first.
