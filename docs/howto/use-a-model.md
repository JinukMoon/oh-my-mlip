# Use a model

Every framework lives in its own env, so a model always runs under that env's
interpreter. There are two ways to use one.

## In your own script

Use this for MD, relaxations and anything long. Copy the lines once; your
script does not need oh-my-mlip at run time.

```python
import sys; sys.path.insert(0, "<repo>")       # or $OH_MY_MLIP_HOME
from oh_my_mlip import resolve

spec = resolve("MACE")                         # a family, or a version such as "MACE-MH-1-OMAT"
print(spec["python"])      # the interpreter to run your script with
print(spec["imports"])     # import lines
print(spec["inference"])   # the line that creates `calc`
print(spec["env_run"])     # environment variables to export first (often empty)
```

Paste `imports` and `inference` verbatim:

```python
# my_run.py  ->  run with spec["python"]
from ase.io import read
from mace.calculators import mace_mp
atoms = read("POSCAR")
calc = mace_mp(model='medium-mpa-0', dispersion=False, default_dtype='float64', device='cuda')
atoms.calc = calc
print(atoms.get_potential_energy())
```

Inside that env you can also build the calculator in one call:
`oh_my_mlip.get_calculator("MACE")`.

## From one script, across models

Use this for quick comparisons. Each model runs in its own env process, so the
calling Python needs no model installed.

```python
import oh_my_mlip

out = oh_my_mlip.run("MACE", atoms)            # {"energy": ..., "forces": ...}

with oh_my_mlip.WorkerPool() as pool:          # workers stay alive between calls
    for name in ("MACE", "SevenNet"):
        resp = pool.request(name, atoms)
        print(name, resp["results"]["energy"])
```

Every call crosses a process boundary, so use the first way for MD loops.

## Ready-made scripts

```bash
python run_examples/single_point.py MACE --structure POSCAR
<spec["python"]> run_examples/relax.py MACE --structure POSCAR    # writes relaxed.extxyz
```

## D3 dispersion

`oh_my_mlip.run(..., apply_d3=True)`, or inside the model's env:

```python
from catbench.dispersion import DispersionCorrection
calc = DispersionCorrection().apply(calc)
```

## NequIP and Allegro

Their calculators load a model compiled for your GPU architecture during
install. To prepare jobs for a different GPU, pass its architecture:
`resolve("NequIP", arch="sm86")`. See
[GPU-architecture compilation](../arch_first_run_compile.md).

Full procedure: [`recipes/run.md`](https://github.com/JinukMoon/oh-my-mlip/blob/main/recipes/run.md) ·
per-framework lines and weights: [Environment recipes](../recipes.md).
