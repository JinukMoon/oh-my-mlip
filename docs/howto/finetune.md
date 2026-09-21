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

Your agent first reads which settings that framework really has — they differ:
DeePMD counts steps instead of epochs, MACE and SevenNet switch stress on through
its weight, and not every trainer has EMA or early stopping. Then it asks:

- the exact variant, e.g. `MACE-MH-1-OMAT`;
- where energies and forces are stored, if they are not on a calculator;
- whether to keep the defaults it shows for training length, batch size, learning
  rate, energy/force/stress weights and stress on or off, or change any of them
  (a setting upstream gives no value for must be set, e.g. DPA4's learning rate);
- the output directory.

Every value comes from you, from the framework's official fine-tuning example, or
from its upstream default, and the run records which one it was.

## What you get

- the converted dataset, the framework's config and training script, and
  `ft_run.json` recording what was run;
- the fine-tuned checkpoint, checked to reload and compute energy and forces;
- the license of the starting checkpoint — a model fine-tuned from a
  non-commercial checkpoint inherits its terms.

??? note "Run it yourself"

    ```bash
    python scripts/ft_run.py MACE --show-settings      # the settings this framework has
    python scripts/ft_run.py MACE --dataset frames.traj --out ft_mace --epochs 50 --seed 0
    python scripts/ft_verify.py ft_mace/<checkpoint> --model MACE --json
    ```

    | Option | Effect |
    |---|---|
    | `--show-settings` | list native settings, the flag that sets each (if any), default, official fine-tuning value and whether they are required |
    | `--epochs`, `--max-steps`, `--batch-size`, `--lr` | training length, batch size, learning rate |
    | `--energy-weight`, `--force-weight`, `--stress-weight` | loss weights |
    | `--include-stress`, `--no-stress` | train on stress or not |
    | `--set NAME=VALUE` | any other native setting, by its name in `--show-settings` |
    | `--version MACE-MH-1-OMAT` | start from a specific checkpoint |
    | `--emit-only` | write the dataset, config and command without running |
    | `--slurm --partition <p>` | also write a SLURM script (nothing is submitted). Its header sets only the job name, partition, log files, one task and one GPU; add wall time, memory, CPUs or account if your cluster's defaults do not suit |
    | `--allow-partial-seed` | NequIP/Allegro only: their trainers fix part of the seed internally |

    Convert a dataset without training:
    `python scripts/ft_dataset.py --input frames.traj --to mace --out ft_data`.
    Full procedure:
    [`recipes/finetune.md`](https://github.com/JinukMoon/oh-my-mlip/blob/main/recipes/finetune.md).
