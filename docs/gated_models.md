# Gated models

Most models are open-weight and need no token. A few are **gated**: their
weights sit behind an upstream license that you accept with your own Hugging
Face account. **oh-my-mlip never redistributes gated weights** — they are
downloaded on your machine with *your* token, after *you* accept the license.

## Which models are gated

| Models | License page |
|---|---|
| all **UMA** variants | <https://huggingface.co/facebook/UMA> |
| **eSEN-30M-OAM** | <https://huggingface.co/facebook/OMAT24> |

Everything else is open. The registry marks each model: `gated` is `true` and
`license_url` points at the page to accept (see
[Supported models](model_status.md)).

## Ask your LLM

```text
Install UMA and check it runs on my GPU.
```

Your agent notices the model is gated, shows you the license page to accept, and
asks you to log in to Hugging Face in your own terminal. It never asks you to
paste a token into the chat.

## The flow (once per model and machine)

1. **Accept the license** on the model's license page, logged in with the account
   whose token you will use.
2. **Log in** so the token is available: `hf auth login` (details and
   alternatives: [Hugging Face token](hf_token.md)). `source env.sh` does **not**
   set a token for you, so no token ends up in the repo.
3. **Run normally.** The first call downloads the weights into the framework's
   own cache; later runs reuse it.

??? note "Run it yourself"

    ```bash
    source env.sh
    hf auth login
    python run_examples/single_point.py UMA --version UMA-s-1p2-OMAT
    ```

## Without a token or an accepted license

The download fails on purpose: nothing retries or falls back to a mirror. An
HTTP 401 or 403 response means the license is not accepted yet, or the token
belongs to a different account. Accept the license, log in, and run again.

## Policy

- Gated weights are never committed to this repository or uploaded anywhere
  by oh-my-mlip.
- They are always downloaded on demand with the user's own credentials.
