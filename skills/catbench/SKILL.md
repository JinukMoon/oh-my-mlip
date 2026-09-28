---
name: catbench
description: Run the full-roster catbench adsorption benchmarking pipeline across oh-my-mlip models. Triggers on requests to benchmark, compare, or evaluate MLIPs on catalysis/adsorption tasks — including generic phrasing like "benchmark several machine-learning potentials on my adsorption dataset" — even when "oh-my-mlip" or "catbench" is not named. Do NOT trigger on literature-only comparison questions with no dataset/compute intent.
argument-hint: "[--dataset PATH] [--models MODEL1,MODEL2,...] [--d3]"
---

The hub is `$OH_MY_MLIP_HOME` (default `~/.oh-my-mlip`); if neither exists,
clone it there first (`$OH_MY_MLIP_HOME/AGENTS.md §9.0`). Read the AGENTS.md
and run the scripts of that clone, never the copy inside the plugin directory.

Defer entirely to `AGENTS.md §3B` (full-roster catbench pipeline).

Read `AGENTS.md §3B` now. Do not reproduce its content here; follow it verbatim.
Key entry point: `run_examples/catbench_quickstart.py` — bring your own dataset
at `raw_data/<tag>_adsorption.json`; each model runs in its own subprocess with
the model's env interpreter; results aggregate into `cwd/result/`.

If any model env is not yet materialized, run `/oh-my-mlip:setup <model>` first.

Underlying job templates/emitters: `scripts/catbench_jobgen.py`.
Result-directory aggregation into a report: `scripts/catbench_report.py`.

Whole-job requests (interview, scoped plan, approval, execution, evidence)
follow the shared recipe `recipes/catbench.md` — read it after the section
above; it says which helpers exist today and which are still planned.
