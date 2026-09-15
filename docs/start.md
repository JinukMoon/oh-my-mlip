# Get started

Getting oh-my-mlip running takes three steps: check your machine, connect your
coding agent (or clone the repository yourself), and install a first model.

## 1. Check your machine

| You need | Why |
|---|---|
| Linux with an NVIDIA GPU and a recent driver | every model is verified by computing energy and forces on the GPU; each env pins a CUDA build that needs a matching driver ([Host requirements](host_requirements.md)) |
| conda or mamba on `PATH` | every framework gets its own conda env |
| git | the hub is a git repository |
| disk space | each framework env takes several GB |
| a Hugging Face account | only for gated models (UMA, eSEN) — [Gated models](gated_models.md) |

## 2. Connect your agent

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

## 3. Install your first model

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

## 4. Do something with it

```text
Relax POSCAR with MACE-MPA-0 and give me the relaxed structure.
```

From here:

- [Install models](howto/install.md) — more models, exact replay, envs you already have
- [Use a model](howto/use-a-model.md) — in your own scripts or across models
- [Benchmark with CatBench](howto/catbench.md) · [Fine-tuning](howto/finetune.md) · [Distillation](howto/distill.md)
