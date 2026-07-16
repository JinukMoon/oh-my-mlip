---
name: catbench
description: Run the full-roster catbench adsorption benchmarking pipeline across oh-my-mlip models. Triggers on requests to benchmark, compare, or evaluate MLIPs on catalysis/adsorption tasks — including generic phrasing like "benchmark several machine-learning potentials on my adsorption dataset" — even when "oh-my-mlip" or "catbench" is not named. Do NOT trigger on literature-only comparison questions with no dataset/compute intent.
argument-hint: "[--dataset PATH] [--models MODEL1,MODEL2,...] [--d3]"
---

Defer entirely to `AGENTS.md §3B` (full-roster catbench pipeline).

Read `AGENTS.md §3B` now. Do not reproduce its content here; follow it verbatim.
Key entry point: `run_examples/catbench_quickstart.py` — bring your own dataset
at `raw_data/<tag>_adsorption.json`; each model runs in its own subprocess with
the model's env interpreter; results aggregate into `cwd/result/`.

If any model env is not yet materialized, run `/oh-my-mlip:setup <model>` first.
