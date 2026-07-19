---
name: run
description: Run a single-point energy/forces calculation or structure relaxation with any oh-my-mlip MLIP. Triggers on requests to run, compute, calculate, or relax a structure with a specific model (MACE, SevenNet, NequIP, ORB, etc.) via oh-my-mlip. Also triggers on GENERIC natural-language compute intent — "single-point this structure with an MLIP", "relax this POSCAR with a machine-learning (interatomic) potential / ML force field" — even when no specific model and no "oh-my-mlip" is named; when no model is given, present oh_my_mlip.list_models() and confirm the choice (MACE is the quickstart default). Do NOT trigger on purely conceptual MLIP questions with no structure/compute intent.
argument-hint: "<model name> [--relax] [--d3] [--structure PATH]"
---

Defer entirely to `AGENTS.md §3A` (run branches — single-point / relax).

Read `AGENTS.md §3A` now. Do not reproduce its content here; follow it verbatim.
Key entry points — two first-class compute paths per §3A: `oh_my_mlip.run()`
for cross-env one-shot results (`Worker` for repeated calls), and `resolve()`
codegen when the calculator goes INTO the user's own script (paste the
returned import/inference lines unmodified; execute with the returned
interpreter). After a verification passes, `models.local.json` records the
machine-verified interpreter/weights (exposed as `spec["local_verified"]`).

If the model env is not yet materialized, run `/oh-my-mlip:setup <model>` first.
