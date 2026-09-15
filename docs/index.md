# oh-my-mlip

<p align="center">
  <img src="assets/logo.png" alt="oh-my-mlip — machine learning interatomic potentials" width="560">
</p>

**One registry, many MLIPs.** oh-my-mlip covers 20 machine-learning
interatomic-potential frameworks (32 model variants) — MACE, SevenNet, NequIP,
ORB, UMA, … — and lets you:

- **install** any of them, each in its own conda env from a pinned recipe;
- **use** them in your own scripts with the exact interpreter and calculator lines;
- **benchmark** them on adsorption energies with CatBench, including your own VASP calculations;
- **fine-tune** them on your data through each framework's own trainer;
- **distill** any of them into an NN-MTP student that runs in LAMMPS.

It drives the real upstream frameworks and never reimplements a model.

## Ask your LLM

oh-my-mlip is built to be driven by your coding agent. Set it up once, then ask.

**Claude Code** — add the plugin:

```
/plugin marketplace add JinukMoon/oh-my-mlip
```

```
/plugin install oh-my-mlip@oh-my-mlip
```

**Codex or any other coding agent** — tell it:

```text
Clone https://github.com/JinukMoon/oh-my-mlip, read its AGENTS.md and the
documentation at https://jinukmoon.github.io/oh-my-mlip/, and use it for the
MLIP work I ask for.
```

Then ask in plain language, for example:

| You say | Guide |
|---|---|
| "Install MACE and SevenNet and check they run on my GPU." | [Install models](howto/install.md) |
| "Relax this POSCAR with UMA and give me the relaxed structure." | [Use a model](howto/use-a-model.md) |
| "Benchmark MACE, SevenNet and UMA on adsorption energies for CO2 reduction on Cu." | [Benchmark with CatBench](howto/catbench.md) |
| "Turn my VASP calculations in ./dft into a CatBench dataset." | [VASP results to CatBench](howto/vasp-to-catbench.md) |
| "Fine-tune MACE on frames.traj." | [Fine-tuning](howto/finetune.md) |
| "Distill MACE into an NN-MTP student for 300 K MD of this slab." | [Distillation](howto/distill.md) |

Your agent asks for what only you can decide — models, data, D3, reference
coefficients, how long the MD must be stable — and shows its plan before
running anything long. Every step it takes is written to a file you can rerun.

## Without an agent

```bash
git clone https://github.com/JinukMoon/oh-my-mlip.git && cd oh-my-mlip
source env.sh
./install.sh MACE                                  # MACE's own conda env
python scripts/setup_verify.py MACE-MPA-0 --json   # energy + forces on your GPU
python run_examples/single_point.py MACE           # a first calculation
```

Each guide ends with the commands to run a workflow yourself.

## How it works

- **One env per framework.** Their torch and CUDA stacks conflict, so each
  framework lives in its own conda env; `resolve()` tells you which
  interpreter and which calculator lines belong together.
- **Recipes, not improvisation.** Package sets are pinned in `envs/<env>.yml`,
  and verified builds are recorded as exact lock files in `envs/locks/`.
- **Every procedure is a file.** Job scripts, training commands and reports
  are written to disk before they run, so you can rerun them without an agent.
- **Weights are never hosted here.** They come from each framework's official
  channel; gated models use your own Hugging Face login.
