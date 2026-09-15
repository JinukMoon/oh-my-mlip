# Fine-tuning

Every framework fine-tunes differently: MACE takes command-line flags, NequIP
and SevenNet read YAML configs, DeePMD reads `input.json`, and each expects its
own dataset layout. You give oh-my-mlip one command and one ASE-readable
dataset; it converts the data to that framework's format, writes the
framework's **own** training config and command, and runs the upstream
trainer — nothing is reimplemented.

Which models can be fine-tuned, with the exact upstream command, config and
dataset format for each: [Fine-tuning per framework](../finetune.md).

## Train

Your dataset is anything ASE reads (extxyz, `.traj`, OUTCAR, …) with energies
and forces.

```bash
python scripts/ft_run.py MACE --dataset my_frames.traj --out ft_mace --epochs 50 --seed 0
```

| Option | Effect |
|---|---|
| `--version MACE-MH-1-OMAT` | start from a specific checkpoint |
| `--emit-only` | write the dataset, config and command without running |
| `--slurm --partition <p>` | also write a SLURM script (nothing is submitted) |
| `--allow-partial-seed` | NequIP/Allegro only: their trainers fix part of the seed internally |

The output directory holds the converted dataset, the exact training script
and `ft_run.json` with what was run.

To convert a dataset without training:

```bash
python scripts/ft_dataset.py --input my_frames.traj --to mace --out ft_data
```

## Check the result

```bash
python scripts/ft_verify.py ft_mace/<checkpoint> --model MACE --json
```

The fine-tuned checkpoint must load and compute energy and forces.

## Licenses

Some checkpoints carry non-commercial terms. The license is printed before
training starts; a model fine-tuned from such a checkpoint inherits its terms.

Per-framework commands, configs and dataset formats:
[Fine-tuning per framework](../finetune.md) ·
full procedure: [`recipes/finetune.md`](https://github.com/JinukMoon/oh-my-mlip/blob/main/recipes/finetune.md).
