# Get started

**Before you start:** Linux with an NVIDIA GPU, and conda (or mamba). Gated
models (UMA, eSEN) also need a Hugging Face account; your agent asks when it
gets there. Details: [Host requirements](host_requirements.md).

## 1. Connect your agent

=== "Claude Code"

    Add the plugin once:

    ```
    /plugin marketplace add JinukMoon/oh-my-mlip
    ```

    ```
    /plugin install oh-my-mlip@oh-my-mlip
    ```

    The plugin works from any directory and fetches the hub itself on first
    use. Details: [Use with Claude Code](claude_plugin.md).

=== "Codex or another agent"

    Tell your agent:

    ```text
    Clone https://github.com/JinukMoon/oh-my-mlip, read its AGENTS.md and the
    documentation at https://jinukmoon.github.io/oh-my-mlip/, and use it for the
    MLIP work I ask for.
    ```

=== "No agent"

    ```bash
    git clone https://github.com/JinukMoon/oh-my-mlip.git && cd oh-my-mlip
    source env.sh
    ```

## 2. Install your first model

Ask your agent:

```text
Install MACE and check it runs on my GPU.
```

or run it yourself:

```bash
./install.sh MACE
python scripts/setup_verify.py MACE-MPA-0 --json
```

The model counts as installed only when it has computed energy and forces on
your GPU.

## 3. Do something with it

```text
Relax POSCAR with MACE-MPA-0 and give me the relaxed structure.
```

From here:

- [Install models](howto/install.md) — more models, exact replay, envs you already have
- [Use a model](howto/use-a-model.md) — in your own scripts or across models
- [Benchmark with CatBench](howto/catbench.md) · [Fine-tuning](howto/finetune.md) · [Distillation](howto/distill.md)
