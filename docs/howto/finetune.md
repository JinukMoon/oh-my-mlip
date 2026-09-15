# Fine-tuning

Every framework fine-tunes differently: MACE takes command-line flags, NequIP
and SevenNet read YAML configs, DeePMD reads `input.json`, and each expects its
own dataset layout. oh-my-mlip converts your data to that framework's format,
writes the framework's **own** config and command, and runs the upstream
trainer — nothing is reimplemented.

## Ask your LLM

```text
Fine-tune MACE on frames.traj — just a quick check that fine-tuning works.
```

```text
Fine-tune SevenNet-MF-OMPA on my DFT frames for 100 epochs.
```

## What you prepare

- **A dataset** that ASE reads (extxyz, `.traj`, OUTCAR, …) with energies and
  forces;
- **the model** to start from; which models can be fine-tuned is listed in
  [Fine-tuning per framework](../finetune.md).

## What it asks you

- the exact variant, e.g. `MACE-MH-1-OMAT`;
- where energies and forces are stored, if they are not on a calculator;
- a real fine-tune (epochs, batch size, split, seed) or a quick check on small
  data;
- the output directory.

## What you get

- the converted dataset, the framework's config and training script, and
  `ft_run.json` recording what was run;
- the fine-tuned checkpoint, checked to reload and compute energy and forces;
- the license of the starting checkpoint — a model fine-tuned from a
  non-commercial checkpoint inherits its terms.

??? note "Run it yourself"

    ```bash
    python scripts/ft_run.py MACE --dataset frames.traj --out ft_mace --epochs 50 --seed 0
    python scripts/ft_verify.py ft_mace/<checkpoint> --model MACE --json
    ```

    | Option | Effect |
    |---|---|
    | `--version MACE-MH-1-OMAT` | start from a specific checkpoint |
    | `--emit-only` | write the dataset, config and command without running |
    | `--slurm --partition <p>` | also write a SLURM script (nothing is submitted) |
    | `--allow-partial-seed` | NequIP/Allegro only: their trainers fix part of the seed internally |

    Convert a dataset without training:
    `python scripts/ft_dataset.py --input frames.traj --to mace --out ft_data`.
    Full procedure:
    [`recipes/finetune.md`](https://github.com/JinukMoon/oh-my-mlip/blob/main/recipes/finetune.md).
