# VASP results to CatBench

Turn your own VASP calculations into a CatBench dataset. The conversion runs on
a copy: your original calculations are hashed before and checked after, and
never modified.

## Ask your LLM

```text
Turn my VASP calculations in ./my_dataset into a CatBench dataset called my_dataset.
```

## What you prepare

Finished calculations with a `CONTCAR` and an `OSZICAR` in every directory,
laid out like this:

```
my_dataset/
├── Pt111/                 # one directory per surface
│   ├── slab/              # the clean slab
│   ├── H/                 # one directory per adsorbate
│   └── OH/
└── gas/
    ├── H2gas/             # one directory per gas reference
    └── H2Ogas/
```

## What it asks you

- **the reaction coefficients** for each adsorbate — the adsorption energy is
  adslab − slab − Σ gas terms, e.g. H = adslab − slab − ½ H2. The agent
  proposes candidates from your gas references; you confirm them;
- the dataset name.

## What you get

- `stage/raw_data/<name>_adsorption.json` — ready for
  [Benchmark with CatBench](catbench.md);
- a check that every original file is unchanged;
- `stage/run_stage.sh` — reruns the conversion.

Symlinks, mounted trees and existing files the tool did not create are refused
rather than guessed.

??? note "Run it yourself"

    ```bash
    # 1. scan: complete and incomplete pairs, proposed coefficients
    python3 scripts/catbench_vasp_stage.py scan --source my_dataset --out stage

    # 2. confirm coefficients in a file outside stage/, e.g. coeff_setting.json:
    #    {"H":  {"slab": -1, "adslab": 1, "H2gas": -0.5},
    #     "OH": {"slab": -1, "adslab": 1, "H2Ogas": -1, "H2gas": 0.5}}

    # 3. stage: copy what catbench reads and write the runner
    python3 scripts/catbench_vasp_stage.py stage --source my_dataset --dest stage \
        --dataset-name my_dataset --coeff coeff_setting.json --python <env>/bin/python

    # 4. convert on the copy
    sh stage/run_stage.sh

    # 5. verify the originals and the JSON
    python3 scripts/catbench_vasp_stage.py verify --source my_dataset --dest stage
    <env>/bin/python scripts/catbench_datasets.py --check stage/raw_data/my_dataset_adsorption.json
    ```

    `<env>/bin/python` is any installed model env; every env has catbench.
