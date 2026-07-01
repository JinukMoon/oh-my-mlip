---
name: run
description: Run a single-point energy/forces calculation or structure relaxation with any oh-my-mlip MLIP. Triggers on requests to run, compute, calculate, or relax a structure with a specific model (MACE, SevenNet, NequIP, ORB, etc.) via oh-my-mlip.
argument-hint: "<model name> [--relax] [--d3] [--structure PATH]"
---

Defer entirely to `AGENTS.md §3A` (run branches — single-point / relax).

Read `AGENTS.md §3A` now. Do not reproduce its content here; follow it verbatim.
Key entry points: `oh_my_mlip.run()` for cross-env single calls; `Worker` for
repeated calls; `resolve()` when emitting a standalone script.

If the model env is not yet materialized, run `/oh-my-mlip:setup <model>` first.
