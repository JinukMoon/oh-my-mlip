# catbench `<tag>_adsorption.json` data format

`run_examples/catbench_quickstart.py` runs a catbench adsorption benchmark on
**your** data. It reads `raw_data/<tag>_adsorption.json` from the current working
directory (`catbench.utils.io_utils.get_raw_data_path` -> `cwd/raw_data/<tag>_adsorption.json`).
This repo bundles no dataset; this page documents the file so you can build your
own from the repo alone.

The schema is owned by the external **catbench** library. Everything below is
**confirmed** by reading catbench's loader
(`catbench.utils.data_utils.load_catbench_json`) and the consumer
(`catbench.adsorption.AdsorptionCalculation._process_reaction_basic`), unless a
field is explicitly flagged as inferred.

## Top-level structure

The file is a JSON object. Each top-level key is a **reaction key** (any unique
string) mapping to one reaction entry:

```jsonc
{
  "<reaction_key>": {
    "raw": {
      "star":          { "atoms_json": "...", "energy_ref": -111.111, "stoi": -1 },
      "<adslab_key>":  { "atoms_json": "...", "energy_ref": -126.222, "stoi":  1 },
      "gas-CO":        { "atoms_json": "...", "energy_ref":  -14.345, "stoi": -1 }
    },
    "ref_ads_eng": -0.766,
    "adsorbate_indices": [12, 13]
  }
}
```

## Reaction entry fields (confirmed)

| Field | Required | Meaning |
|---|---|---|
| `raw` | yes | object mapping a structure name to a structure entry (see below). |
| `ref_ads_eng` | yes | reference (DFT) adsorption energy in eV; read as `reaction_data["ref_ads_eng"]`. |
| `adsorbate_indices` | yes | list of atom indices (into the adslab) that are the adsorbate. `AdsorptionCalculation` raises `KeyError` if absent. |

### Structure names inside `raw` (confirmed)

`_process_reaction_basic` classifies each structure by its key string:

- **`star`** — the clean slab. Its single-point energy is cached/reused.
- any **non-`star`, non-`gas`** key — the adsorbate-covered slab (adslab).
- any key containing **`gas`** (e.g. `gas-CO`) — a gas-phase reference molecule;
  detected via `"gas" in str(structure)`.

## Structure entry fields (confirmed)

Each structure entry under `raw` is an object:

| Field | Required | Meaning |
|---|---|---|
| `atoms_json` | yes | the structure serialized by ASE's JSON writer. Produce with `buf = io.StringIO(); atoms.write(buf, format="json"); atoms_json = buf.getvalue()`. `load_catbench_json` rehydrates it with `ase.io.read(io.StringIO(atoms_json), format="json")` and replaces it with an `atoms` key. |
| `energy_ref` | yes | DFT reference total energy (eV) of that structure (`reaction_data["raw"][s]["energy_ref"]`). |
| `stoi` | yes | stoichiometric coefficient of that structure in the adsorption reaction (clean slab `-1`, adslab `+1`, each gas its own coefficient). Used as `energy * stoi`. |

## Deduplicated form (confirmed, optional)

`load_catbench_json` also accepts a deduplicated layout: a top-level
`"_structures"` map from an id to an `atoms_json` string, with each structure
entry carrying a `"ref": "<id>"` instead of an inline `atoms_json`. The loader
rehydrates `ref` pointers from `_structures` and then proceeds identically.
Legacy files with inline `atoms_json` (no `_structures`) are fully supported. The
shipped example uses the simple inline form.

## How to generate it

Use catbench's own preprocessing inside any env that ships catbench
(every oh-my-mlip model env does):

- `catbench.adsorption.data.cathub.cathub_preprocessing(benchmark, ...)` pulls a
  CatHub dataset and writes `<benchmark>_adsorption.json`, auto-detecting
  `adsorbate_indices`.
- `catbench.adsorption.data` also exposes `download` / zenodo helpers.

This repo never ships or redistributes a dataset.

## Shipped example

`run_examples/raw_data_example/Example_adsorption.json` is a minimal, single-
reaction file (CO* on a small Cu(111) slab) that loads cleanly through
`load_catbench_json`. It is a **format illustration only** — the energies are
placeholders, not real DFT data. Copy it to `raw_data/<your_tag>_adsorption.json`
and replace the structures/energies with your own to use it as a template.
