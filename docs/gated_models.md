# Gated models

Most of the roster is open-weight and needs no token. A few models are **gated**:
their weights sit behind an upstream license that you must accept with your own
Hugging Face account. **oh-my-mlip never redistributes gated weights** — they are
always fetched on first run with *your* token, after *you* accept the license.

## How to tell if a model is gated

Read the model's entry in `models.json`:

| Field | Open model | Gated model |
|---|---|---|
| `gated` | `false` | `true` |
| `license_url` | `null` | the upstream license page, e.g. `https://huggingface.co/facebook/UMA` |
| `weights` | `bundled` / `auto-download` / `on-demand-hf` | `on-demand-hf` |

In the v1 roster, all **UMA** variants are gated. Everything else is open.

## The flow (one-time per model + per machine)

1. **Accept the license.** Open the model's `license_url` while logged into
   Hugging Face with the account whose token you will use, and accept the terms.
   For UMA that is `https://huggingface.co/facebook/UMA`.
2. **Make your token available.** Create a read token at
   `https://huggingface.co/settings/tokens`, then make it available to
   oh-my-mlip. See [`hf_token.md`](hf_token.md) for the canonical, leak-safe
   setup (the preferred path is `huggingface-cli login`; avoid pasting the token
   literal into your shell). `source env.sh` does **not** set `HF_TOKEN` for you
   — that is intentional, so no token is ever baked into the repo or a shared
   cache.
3. **Run normally.** The first call downloads the weights into the shared cache
   (`HF_HOME` / `FAIRCHEM_CACHE_DIR`, set by `env.sh`). Subsequent runs reuse the
   cache.

```bash
source env.sh
huggingface-cli login          # or: export HF_TOKEN="$(< /path/outside/repo/token)"
python run_examples/single_point.py UMA --version UMA-s-1p2-OMAT
```

See [`hf_token.md`](hf_token.md) for the full, leak-safe token setup.

## What happens without a token / without accepting the license

The fetch **fails by design** — the resolver does not retry or fall back to a
mirror. The correct response is to surface the model's `license_url` to the user
and stop, not to work around the gate. An agent following `AGENTS.md` should:

- check `gated` in `models.json` before attempting a gated model,
- verify `HF_TOKEN` is set,
- on failure, print the `license_url` and the two steps above — never attempt to
  obtain the weights by any other route.

## Policy

- Gated weights are **never** committed to this repo, baked into a conda-pack
  tarball, or uploaded to any oh-my-mlip Hugging Face repo.
- The conda-pack tarballs published by this project contain the **environment
  only** (Python + libraries). Gated weights are downloaded on the user's machine
  with the user's own credentials.
- Open-weight models may be bundled or auto-downloaded; gated models are always
  `on-demand-hf` and user-token-gated.
