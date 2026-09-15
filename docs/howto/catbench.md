# Benchmark with CatBench

Every env ships [catbench](https://github.com/JinukMoon/catbench) for
adsorption-energy benchmarks. Results from all models land in one `result/`
directory and are analysed together.

## Choose a dataset

```bash
python3 scripts/catbench_datasets.py --list                  # published benchmarks
python3 scripts/catbench_datasets.py --target "CO2 reduction on Cu"
```

Your own VASP calculations can be converted too — see
[Convert VASP results](vasp-to-catbench.md).

## Run models

```bash
mkdir my_benchmark && cd my_benchmark
python <repo>/run_examples/catbench_quickstart.py <TAG> --only MACE,SevenNet
```

A `<TAG>` that is not in `raw_data/` yet is downloaded first (a Zenodo
benchmark, otherwise CatHub). For each model the script writes
`jobs/catbench_<MLIP>.py` and `jobs/run_catbench_<MLIP>.sh`, then runs the
runner; rerunning the runner reproduces the job.

| Option | Effect |
|---|---|
| `--only MACE,SevenNet` | frameworks to run |
| `--all-versions` | every version of each framework (versions marked `"catbench": false` are skipped) |
| `--d3` | add D3 dispersion (`_D3` is appended to the model name) |
| `--slurm --emit-only --partition <p>` | write SLURM job scripts without running them |
| `--arch sm86` | pick the NequIP/Allegro build for a different GPU |
| `--catbench-version <v>` | refuse to run under any other catbench version |

## Report

```bash
python <repo>/scripts/catbench_report.py --result ./result --out ./report
```

The report holds the MAE table and plot, and catbench's own analysis
workbook with threshold-sensitivity charts. Official leaderboard values can be
fetched for side-by-side comparison with `scripts/catbench_leaderboard.py`;
they are never recomputed.

Full procedure: [`recipes/catbench.md`](https://github.com/JinukMoon/oh-my-mlip/blob/main/recipes/catbench.md) ·
data format: [CatBench data format](../catbench_data_format.md).
