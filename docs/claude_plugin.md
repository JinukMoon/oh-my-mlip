# Use with Claude Code

The Claude Code plugin lets you run oh-my-mlip by asking in plain language. It
works from any directory.

## Add the plugin

```
/plugin marketplace add JinukMoon/oh-my-mlip
```

```
/plugin install oh-my-mlip@oh-my-mlip
```

The repository is its own marketplace (`.claude-plugin/marketplace.json`), so no
hosted registry is involved. To use a local clone instead, add its directory as
the marketplace:

```
/plugin marketplace add ~/oh-my-mlip
```

## What it adds

You rarely need to type these; Claude picks the right one from your request.

| Command | What it does |
|---|---|
| `/oh-my-mlip:setup <model>` | install a model's env, compile it for your GPU if needed, and verify energy and forces on the GPU |
| `/oh-my-mlip:run <model>` | single-point energy and forces, or a structure relaxation |
| `/oh-my-mlip:catbench` | CatBench adsorption benchmark across models, including your own VASP data |
| `/oh-my-mlip:finetune` | fine-tune a model on your data with its framework's own trainer |
| `/oh-my-mlip:distill` | distill a model into an NN-MTP student for LAMMPS, with the [onthefly-distill](https://github.com/JinukMoon/onthefly-distill) engine |

The skills are thin: model facts live in `models.json` and the procedures in
`AGENTS.md` and `recipes/`, so the plugin and a Codex session follow the same
steps.

## Check that it works

Ask:

```text
Install MACE-MPA-0 and check it runs on my GPU.
```

It is done when Claude reports energy and forces computed on your GPU.

## Notes

- **Gated models** (UMA, eSEN): Claude shows the license page to accept and
  asks you to log in to Hugging Face yourself; it never asks you to paste a
  token. See [Gated models](gated_models.md).
- An optional MCP server (`python -m oh_my_mlip.mcp_server`) offers structured
  registry queries; the plugin does not need it.
