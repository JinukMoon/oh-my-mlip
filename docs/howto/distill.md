# Distill

Distillation turns a large teacher MLIP into a small NN-MTP student that runs
in LAMMPS on CPUs. The active-learning loop is the separate GPL-2.0 project
[onthefly-distill](https://github.com/JinukMoon/onthefly-distill); oh-my-mlip
prepares the run and invokes it, copying nothing in.

## Prerequisites

- the teacher model installed (see [Install models](install.md))
- an onthefly-distill checkout
- a LAMMPS binary with the NN-MTP pair style: `scripts/build_lammps_nnmtp.sh`

## Prepare

```bash
python scripts/distill_bootstrap.py --teacher MACE-MPA-0 --structure slab.vasp \
    --work ./distill --repo <onthefly-distill> --lmp-bin <lmp>
```

`--work` must be a new or empty directory; nothing already in it is ever
overwritten. The directory receives the engine config, the teacher hook built
from the registry's own calculator lines, and `run_distill.sh`.

## Run

```bash
cd distill
sh run_distill.sh > distill.log 2>&1
```

The loop labels structures with the teacher, trains the student, runs the
student's own MD in LAMMPS, and adds every failure back into training until the
student survives the target trajectory.

## Accept a student

For a run you want to judge, prepare it with acceptance criteria and check it
afterwards:

```bash
python scripts/distill_bootstrap.py --teacher MACE-MPA-0 --structure slab.vasp \
    --work ./distill --acceptance --target-ps 10 --repo <onthefly-distill> --lmp-bin <lmp>
python scripts/distill_verify.py --work ./distill --json
```

The verdict requires both held-out accuracy of the student and a stable LAMMPS
run within the time budget.

Full procedure: [`recipes/distill.md`](https://github.com/JinukMoon/oh-my-mlip/blob/main/recipes/distill.md).
