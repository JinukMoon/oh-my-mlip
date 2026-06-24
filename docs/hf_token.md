# Hugging Face token setup (canonical)

This is the canonical, leak-safe guide for making your Hugging Face token
available to oh-my-mlip so it can fetch **gated** model weights (e.g. UMA).

**The token is yours.** oh-my-mlip *reads* it to download weights on your
machine; it never **writes**, **echoes**, **commits**, or **redistributes** it.
Gated weights are fetched on first run with *your* token, after *you* accept the
upstream license — they are never baked into this repo or any published tarball.

> See [`gated_models.md`](gated_models.md) for which models are gated and what
> happens when a token is missing. This file is the canonical token-setup doc.

## 1. Make a Hugging Face account

Sign up (free) at <https://huggingface.co/join>.

## 2. Accept the model license

Open the gated model's page while logged in with the **same** account whose
token you will use, and accept the terms. For UMA:

  <https://huggingface.co/facebook/UMA>

Without an accepted license the download fails by design — no retry, no mirror.

## 3. Create a READ token

At <https://huggingface.co/settings/tokens>, create a token with the **read**
role. A read token is sufficient to download gated weights; do not use a write
token for fetching.

## 4. Make the token available — supported ways (in order of preference)

Pick **one**. They are listed best-first.

### a) `huggingface-cli login` (preferred)

```bash
huggingface-cli login
```

Paste the token at the interactive prompt. It is stored in your HF cache
(`~/.cache/huggingface/token`) and resolved automatically by `huggingface_hub`.
Nothing lands in your shell history, the repo, or a shared cache.

### b) Point at a token file with `HF_TOKEN_PATH`

Keep the token in a file **outside the repo** and export its path (the standard
Hugging Face variable, also honored by oh-my-mlip):

```bash
export HF_TOKEN_PATH=/path/outside/repo/token   # file contains only the token
```

oh-my-mlip also accepts `OMM_HF_TOKEN_FILE=/path/outside/repo/token` as a
convenience; when set, it is exported as `HF_TOKEN_PATH` for child processes so
third-party loaders resolve it the standard way. oh-my-mlip never reads the
file's contents into a variable or a log.

### c) Read the file into `HF_TOKEN` at use time

If you must use the `HF_TOKEN` variable, read it from a file outside the repo so
the literal never appears as a command argument:

```bash
export HF_TOKEN="$(< /path/outside/repo/token)"
```

### Do NOT do this

```text
# ❌ NEVER paste the token literal after HF_TOKEN= on the command line —
#    it leaks into your shell history and the process list.
#    (use 4a/4b/4c above instead)
```

Never type the token literal on the command line, paste it into a script, or
commit it. Keep token files outside the repository (the repo `.gitignore`
ignores `*token*` paths defensively).

## 5. Verify

```bash
source env.sh
python run_examples/single_point.py UMA --version UMA-s-1p2-OMAT
```

The first call downloads the weights into the shared cache; later runs reuse it.
If the token or license is missing, the fetch fails with an actionable message
pointing back here — not a raw traceback.
