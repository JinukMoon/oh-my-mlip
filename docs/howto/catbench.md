# Benchmark with CatBench

Compare models on adsorption energies with
[catbench](https://github.com/JinukMoon/catbench), which ships in every env.

## Ask your LLM

```text
Benchmark MACE, SevenNet and UMA on adsorption energies for CO2 reduction on Cu, with D3.
```

## What you prepare

- **A dataset** — a published CatBench dataset name, an existing
  `raw_data/<tag>_adsorption.json`, or your own VASP calculations
  ([convert them first](vasp-to-catbench.md)). If you only describe the
  chemistry, the agent recommends a published dataset.
- **The models** — installed ones; gated models need a Hugging Face login.

## What it asks you

- which dataset, if you described only the chemistry;
- which models — or your own ASE calculator file;
- D3 on or off;
- a new benchmark or a rerun of an earlier one (a rerun keeps the catbench
  version it used).

## What you get

- `jobs/` — one rerunnable job script per model;
- `result/` — catbench's results for every model;
- `report/` — the MAE table and plot, and catbench's analysis workbook with
  threshold-sensitivity charts;
- optionally, official leaderboard values side by side — fetched, never
  recomputed.

??? note "Run it yourself"

    A first run (20 small-adsorbate reactions, one model: a few minutes):

    ```bash
    mkdir my_benchmark && cd my_benchmark
    python <repo>/run_examples/catbench_quickstart.py MamunHighT2019 --only MACE --max-reactions 20
    python <repo>/scripts/catbench_report.py --result ./result --out ./report
    ```

    The full benchmark drops `--max-reactions`; pick the dataset and models
    with `python3 scripts/catbench_datasets.py --list`. A dataset's download
    size says little about its run time: a small file of large molecules can
    take hours, so try a subset first. A `<TAG>` not yet in `raw_data/` is
    downloaded first (a Zenodo benchmark, otherwise CatHub).

    | Option | Effect |
    |---|---|
    | `--only MACE,SevenNet` | frameworks to run |
    | `--max-reactions N` | only the first N reaction ids (sorted, so the same N every run); the report marks it "Subset: N of M" |
    | `--all-versions` | every version of each framework (versions marked `"catbench": false` are skipped) |
    | `--d3` | add D3 dispersion (`_D3` is appended to the model name) |
    | `--calc-num N` | calculator instances per model, all loaded at once (default 3): GPU memory grows with N, so use 1 for large models on a small GPU |
    | `--slurm --emit-only --partition <p>` | write SLURM job scripts without running them |
    | `--arch sm86` | pick the NequIP/Allegro build for a different GPU |
    | `--catbench-version <v>` | refuse to run under any other catbench version |

    The SLURM header sets only the job name, partition, log files, one task and
    one GPU. Wall time, memory, CPUs and account are left to your cluster's
    defaults: add them to the scripts in `jobs/` before submitting if those
    defaults do not suit. `--submit` prints the header it sends.

    Leaderboard comparison: `scripts/catbench_leaderboard.py`. Full procedure:
    [`recipes/catbench.md`](https://github.com/JinukMoon/oh-my-mlip/blob/main/recipes/catbench.md);
    data format: [CatBench data format](../catbench_data_format.md).
