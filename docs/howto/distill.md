# Distillation

Distill any model in the hub into an **NN-MTP** student. NN-MTP is a compact
machine-learning potential with a ready LAMMPS pair style (`nnmtp`), so the
student runs large MD on CPUs — no Python and no LibTorch at run time. The
teacher hook is built from the registry's own calculator lines, so any
installed model can be the teacher without teacher-specific code. The
active-learning loop is the separate GPL-2.0 project
[onthefly-distill](https://github.com/JinukMoon/onthefly-distill), invoked and
never copied in.

## Ask your LLM

```text
Distill UMA-s-1p2-OMAT into an NN-MTP student for 300 K NVT MD of slab.vasp.
```

## What you prepare

- **the teacher** — an installed model;
- **a structure file** with at most four chemical elements;
- **a LAMMPS binary with the NN-MTP pair style** — the agent can build it once
  with `scripts/build_lammps_nnmtp.sh`;
- **an onthefly-distill checkout**.

## What it asks you

- the teacher variant;
- how the student will be used — temperature, ensemble, MD length; the
  stability target is set from this;
- a quick trial run or a production run that must meet accuracy and stability
  targets;
- where LAMMPS and the onthefly-distill checkout are.

## What you get

- a work directory with the engine config, the teacher hook and
  `run_distill.sh`;
- the NN-MTP student, trained where the student failed — teacher labels, student
  training and student MD in LAMMPS, looped until it survives the target
  trajectory;
- for a production run, a verdict that needs both held-out accuracy and a
  stable LAMMPS run within the time budget.

??? note "Run it yourself"

    ```bash
    scripts/build_lammps_nnmtp.sh --repo <onthefly-distill>  # once
    python scripts/distill_bootstrap.py --teacher MACE-MPA-0 --structure slab.vasp \
        --work ./distill --repo <onthefly-distill> --lmp-bin <lmp>
    cd distill && sh run_distill.sh > distill.log 2>&1
    ```

    `--work` must be a new or empty directory; nothing in it is overwritten.

    For a run you want judged, the bootstrap also takes every target
    explicitly (values in `< >` are yours to choose):

    ```bash
    python scripts/distill_bootstrap.py --teacher MACE-MPA-0 --structure slab.vasp \
        --work ./distill --repo <onthefly-distill> --lmp-bin <lmp> \
        --acceptance --mode production \
        --energy-mae-max <meV/atom> --force-mae-max <meV/A> \
        --target-ps <stable MD length, ps> --max-iter <AL rounds> --no-progress-limit <rounds>
    ```

    and afterwards:

    ```bash
    python scripts/distill_verify.py --work ./distill --json
    ```

    `--mode fixture` (at most 3 rounds) only exercises the loop: even a passing
    fixture says nothing about the student's accuracy. Use `--mode production`
    for a student you will use.

    Full procedure:
    [`recipes/distill.md`](https://github.com/JinukMoon/oh-my-mlip/blob/main/recipes/distill.md).
