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

### `<tag>_adsorption.json` schema

The internal schema is owned by the external **catbench** library
(`catbench.adsorption.AdsorptionCalculation` reads it via
`catbench.utils.data_utils.load_catbench_json`). The shape below is **confirmed**
against that loader; a minimal, loadable example ships at
[`raw_data_example/Example_adsorption.json`](raw_data_example/Example_adsorption.json)
and the full reference is in [`docs/catbench_data_format.md`](../docs/catbench_data_format.md).

Top level is a JSON object mapping a **reaction key** (any unique string) to a
reaction entry:

```jsonc
{
  "<reaction_key>": {
    "raw": {
      "star":           { "atoms_json": "<ase json str>", "energy_ref": -111.111, "stoi": -1 },
      "<adslab_key>":   { "atoms_json": "<ase json str>", "energy_ref": -126.222, "stoi":  1 },
      "gas-<species>":  { "atoms_json": "<ase json str>", "energy_ref":  -14.345, "stoi": -1 }
    },
    "ref_ads_eng": -0.766,
    "adsorbate_indices": [12, 13]
  }
}
```

| Field | Where | Meaning (confirmed from the catbench loader) |
|---|---|---|
| `<reaction_key>` | top level | unique name for one adsorption reaction (becomes a `result/` subdir). |
| `raw` | per reaction | map of structure name -> structure entry. A clean slab is keyed **`star`**; the adsorbate-covered slab is any non-`star`, non-`gas` key; gas-phase references are keyed with **`gas`** in the name (e.g. `gas-CO`). |
| `atoms_json` | per structure | the structure serialized with ASE's JSON writer: `buf=io.StringIO(); atoms.write(buf, format="json")`. `load_catbench_json` reads it back with `ase.io.read(buf, format="json")`. |
| `energy_ref` | per structure | the DFT reference total energy (eV) of that structure. |
| `stoi` | per structure | stoichiometric coefficient in the adsorption reaction (slab `-1`, adslab `+1`, each gas its coefficient). |
| `ref_ads_eng` | per reaction | the reference (DFT) adsorption energy (eV) the benchmark compares against. |
| `adsorbate_indices` | per reaction | indices (into the adslab structure) of the adsorbate atoms. **Required** — `AdsorptionCalculation` raises `KeyError` if missing. |

Generate it from public sources with catbench's own helpers
(`cathub_preprocessing`, `zenodo_download` / `download`) inside any env that ships
catbench — they emit exactly this file and auto-detect `adsorbate_indices`. This
repo bundles no dataset; the shipped example is a **format illustration only**,
not benchmark data.

## Gated models

For gated models (e.g. UMA) accept the upstream license and export your own
token before running any example:

```bash
export HF_TOKEN=hf_...
```

The weights are fetched on first run with your token; they are never
redistributed by this repo. See `docs/gated_models.md`.
