# Contributing to oh-my-mlip

Thank you for helping improve oh-my-mlip.
This document covers the trust-boundary policy, how to add a model, how to run
the GPU-free checks locally, and the deferred GPU/compute checkpoint.

---

## Trust boundary: `models.json` is code-equivalent

**`models.json` must be treated as code, not data.**

Every entry in `models.json` contains Python strings (`import`, `inference`)
that are `exec()`'d inside each model's conda env at run time, and an
optional `env_run` prefix that is applied as subprocess environment variables.
A malicious or careless edit to `models.json` is therefore equivalent to a
malicious edit to a Python source file.

Consequences:

- **Changes to `models.json` are gated behind PR review**, the same as any
  `.py` file.  No direct pushes to `main` that touch `models.json`.
- The `inference` and `import` strings are syntax-checked by CI (`ast.parse`)
  on every PR — without executing them and without a GPU.
- The `env_run` field is constrained to a strict `KEY=VALUE` allowlist
  (`oh_my_mlip.registry.ENV_RUN_ALLOWLIST`) and is **never** passed through a
  shell.  The parser (`parse_env_run`) raises `RegistryError` for any token
  that is not a bare, allow-listed `KEY=VALUE` pair — command substitution
  (`$(…)`), pipes (`|`), semicolons (`;`), backticks, and any other shell
  metacharacter are rejected.  See `oh_my_mlip/registry.py` for the allowlist
  and `tests/test_registry_integrity.py` for the negative tests.
- The JSON-schema in `tests/schema/models.schema.json` enforces the four
  **honesty fields** on every version entry.  CI will reject any version that
  omits `gated`, `weights`, `validation`, or `inference`.

---

## How to add a model

Adding a model is a **data-only** change (no Python code) but it still goes
through a normal PR and must pass all CI checks before merging.

### 1. Append a row to `models.json`

Add a new top-level key (the framework name) or a new version inside an
existing framework.  All four honesty fields are **required** on every version:

```jsonc
"MyModel": {
  "env": "mymodel",
  "python": "${OH_MY_MLIP_HOME}/envs/mymodel/bin/python",
  "import": ["from mymodel.calculator import MyCalc"],
  "versions": {
    "MyModel-v1": {
      "mlip_name": "MyModel-v1",
      "training_set": ["MPtrj"],
      "gated": false,           // REQUIRED — true if weights need HF_TOKEN + license
      "license_url": null,      // REQUIRED — URL when gated=true, null when false
      "weights": "bundled",     // REQUIRED — "bundled" | "auto-download" | "on-demand-hf"
      "validation": "gpu_pending", // REQUIRED — start here; upgrade after GPU test
      "inference": ["calc = MyCalc(model='${OH_MY_MLIP_HOME}/models/mymodel/v1.pt')"]
    }
  }
}
```

Rules:
- **Never** set `validation` to `validated_sm86` or `validated_sm89` unless you
  have run a GPU single-point check on that architecture and recorded the result.
  New rows start at `gpu_pending`.
- **Never** set `gated: false` for weights that require accepting an upstream
  license.  If in doubt, set `gated: true`.
- `env_run` (optional) must contain only allow-listed `KEY=VALUE` tokens —
  see `oh_my_mlip/registry.py:ENV_RUN_ALLOWLIST`.  Any shell metacharacter
  will be rejected by CI.

### 2. Regenerate the status table

The README `## Models & status` table is generated from `models.json`.
After editing `models.json`, regenerate it:

```bash
python scripts/gen_status_table.py
```

Paste the output between the `<!-- STATUS_TABLE_START -->` and
`<!-- STATUS_TABLE_END -->` markers in `README.md`.

CI runs `python scripts/gen_status_table.py --check` and fails if the table is
out of sync — so you must do this before opening a PR.

### 3. Run the GPU-free CI checks locally

Install dev dependencies (once):

```bash
pip install -r requirements-dev.txt
```

Then run all four checks:

```bash
# 1. JSON-schema validation (models.json + dist_manifest.json) and all tests
python -m pytest tests/ -q

# 2. Status-table sync check
python scripts/gen_status_table.py --check

# 3. Shellcheck (requires shellcheck binary — see requirements-dev.txt)
shellcheck install.sh scripts/*.sh

# 4. Inference-string ast.parse (covered by pytest above, but runnable standalone)
python -m pytest tests/test_inference_parses.py -v
```

All four checks run on every PR in CI (`.github/workflows/ci.yml`) without a
GPU, conda env, torch, or ase.

### 4. Open a PR

CI must be green before merging.  A reviewer will check:

- Honesty fields are accurate (not aspirational).
- `inference`/`import` strings match the upstream library's current API.
- `env_run` tokens are on the allowlist.
- The status table in README.md matches `models.json`.

---

## Deferred GPU/compute checkpoint

The following acceptance checks **require human action** (a GPU machine with
the appropriate driver, access to a prebuilt source environment, and an
`HF_TOKEN`) and are therefore **not run in CI**:

- GPU validate: `install.sh <env>` on a GPU box and verify D3 compiles or
  degrades gracefully.
- End-to-end: `run("MACE", atoms)` and `run("SevenNet", atoms)` on a
  **foreign** GPU host (different driver/glibc than the build host) returning
  finite energy within tolerance of the `ref_energy_*` fixtures.
- Worker 100-call loop: one long-lived MACE worker returns 100 results
  without a respawn (persistent-worker protocol proof).
- Upgrading `validation` from `gpu_pending` to `validated_sm86/sm89`: must be
  done by the maintainer after a successful GPU single-point run and recorded
  in `models.json` via a PR.

---

## Code style

- Python: PEP 8, type hints on public functions, no heavy imports in
  `oh_my_mlip/registry.py` (it must import cleanly with no torch/ase/conda).
- Shell: `set -euo pipefail`; pass `shellcheck`.
- Tests: no GPU, no conda env, no torch, no ase required for `tests/`.
- Commit messages: imperative mood, present tense, ≤72 chars subject line.
- No Co-authored-by trailers for AI assistants.

---

## Questions?

Open an issue on GitHub.
