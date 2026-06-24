# run_examples

Minimal, copy-pasteable examples that use **only** the public `oh_my_mlip`
interface. They run **locally** — there is no job scheduler. `source env.sh`
once per shell first (it sets `OH_MY_MLIP_HOME`, the shared caches, and the
D3/CUDA environment).

| Example | What it shows | MCP-tool analogue |
|---|---|---|
| `single_point.py` | Cross-env single-point energy/forces via `run()` | `run_singlepoint` |
| `relax.py` | Relaxation driven by a persistent `Worker` (`.request()`) behind a small ASE adapter | `run_relax` |
| `catbench_quickstart.py` | Full-roster catbench on **your** data: one `resolve()`-dispatched subprocess per model | `run_catbench` |

```bash
source env.sh
python run_examples/single_point.py MACE          # or SevenNet, ...
python run_examples/single_point.py MACE --d3     # with D3 dispersion
python run_examples/relax.py SevenNet --fmax 0.05
```

## Bring your own data (catbench)

**This repo bundles no benchmark data.** `catbench_quickstart.py` reads
`raw_data/<tag>_adsorption.json` from the **current working directory**:

```bash
mkdir -p my_bench/raw_data
# put your dataset here, named <tag>_adsorption.json, e.g.:
#   my_bench/raw_data/MyDataset_adsorption.json
cd my_bench
python <repo>/run_examples/catbench_quickstart.py MyDataset --only MACE,SevenNet
```

- The file must be named exactly `<tag>_adsorption.json`; `<tag>` is what you
  pass on the command line (omit it and the script auto-detects a single file).
- Results are written to `cwd/result/`. With `--d3`, each model's `mlip_name`
  gets a `_D3` suffix so D3 and non-D3 results stay distinct.
- catbench's preprocessing helpers (e.g. `cathub_preprocessing`,
  `zenodo_download`) can produce this JSON from public sources — run them in any
  env that ships catbench. This repo does not ship or redistribute any dataset.

## Gated models

For gated models (e.g. UMA) accept the upstream license and export your own
token before running any example:

```bash
export HF_TOKEN=hf_...
```

The weights are fetched on first run with your token; they are never
redistributed by this repo. See `docs/gated_models.md`.
