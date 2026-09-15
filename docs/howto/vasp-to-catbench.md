# Convert VASP results for CatBench

catbench's `vasp_preprocessing` deletes every file except `CONTCAR` and
`OSZICAR` in the tree it converts. `scripts/catbench_vasp_stage.py` therefore
works on a **copy**: your original calculations are hashed before and checked
after, and are never modified.

## Expected layout

```
my_dataset/
├── Pt111/                 # one directory per surface
│   ├── slab/              # CONTCAR + OSZICAR of the clean slab
│   ├── H/                 # CONTCAR + OSZICAR of each adsorbate
│   └── OH/
└── gas/
    ├── H2gas/             # CONTCAR + OSZICAR of each gas reference
    └── H2Ogas/
```

## Steps

**1. Scan** — lists complete and incomplete pairs and proposes coefficients:

```bash
python3 scripts/catbench_vasp_stage.py scan --source my_dataset --out stage
```

**2. Confirm the coefficients** (reaction energy = adslab − slab − Σ gas terms)
in a file outside `stage/`, for example `coeff_setting.json`:

```json
{
  "H":  {"slab": -1, "adslab": 1, "H2gas": -0.5},
  "OH": {"slab": -1, "adslab": 1, "H2Ogas": -1, "H2gas": 0.5}
}
```

**3. Stage** — copies the files catbench reads and writes a runner:

```bash
python3 scripts/catbench_vasp_stage.py stage --source my_dataset --dest stage \
    --dataset-name my_dataset --coeff coeff_setting.json --python <env>/bin/python
```

`<env>/bin/python` is any installed model env (every env has catbench).

**4. Convert** — runs catbench on the copy:

```bash
sh stage/run_stage.sh        # -> stage/raw_data/my_dataset_adsorption.json
```

**5. Verify** — the originals are unchanged and the JSON is valid:

```bash
python3 scripts/catbench_vasp_stage.py verify --source my_dataset --dest stage
<env>/bin/python scripts/catbench_datasets.py --check stage/raw_data/my_dataset_adsorption.json
```

Copy `raw_data/` into your benchmark directory and follow
[Benchmark with CatBench](catbench.md). The stage refuses symlinks, mounted
trees and existing files it did not create, rather than guessing.
