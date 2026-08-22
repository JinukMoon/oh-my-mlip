---
name: finetune
description: Fine-tune an oh-my-mlip model on your own dataset from a foundation checkpoint. Triggers on requests to fine-tune, continue training, or adapt a model to custom data — "fine-tune MACE on my dataset", "파인튜닝 해줘", "continue training SevenNet from the foundation checkpoint", "train DeePMD on my extxyz" — even when "oh-my-mlip" or "finetune" is not named.
argument-hint: "<model> --dataset PATH [--out DIR] [--epochs N] [--emit-only] [--slurm]"
---

Defer entirely to `AGENTS.md §3C` (fine-tune a model on your own dataset).

Read `AGENTS.md §3C` now. Do not reproduce its content here; follow it verbatim.
Key entry points: `scripts/ft_dataset.py` (canonical dataset converter),
`scripts/ft_run.py <model> --dataset <path>` (resolves `models.json`'s
`finetune` block and materializes the run), `scripts/ft_verify.py <ckpt>
--model <M> --json` (the checkpoint-loads-and-computes oracle).

If the model's env is not yet materialized, run `/oh-my-mlip:setup <model>` first.
