# Contributing to oh-my-mlip

Thank you for helping improve oh-my-mlip.
This document covers the trust-boundary policy, how to add a model, how to add
a tool or workflow, how to run the GPU-free checks locally, and the GPU proof a
change needs.

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

### 2. Regenerate the status lists

`models.json` is the single source of truth for **two** generated views:

- the light `| Framework | Models |` list in the README `## Supported MLIPs`
  section, between the `<!-- STATUS_TABLE_START -->` /
  `<!-- STATUS_TABLE_END -->` markers in `README.md`; and
- the full detailed table (Model / Framework / Weights / Gated / Code) in
  `docs/model_status.md`, between the
  `<!-- STATUS_TABLE_DETAILED_START -->` /
  `<!-- STATUS_TABLE_DETAILED_END -->` markers.

After editing `models.json`, regenerate both:

```bash
python scripts/gen_status_table.py
```

It prints both blocks (clearly delimited). Paste the simple list between the
`STATUS_TABLE` markers in `README.md`, and the detailed table between the
`STATUS_TABLE_DETAILED` markers in `docs/model_status.md`.

CI runs `python scripts/gen_status_table.py --check`, which verifies **both**
files are byte-for-byte in sync and fails if either is out of sync — so you must
do this before opening a PR.

### 3. Run the GPU-free CI checks locally

Install dev dependencies (once):

```bash
pip install -r requirements-dev.txt
```

Then run all four checks:

```bash
# 1. JSON-schema validation (models.json) and all tests
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

## How to add a tool or workflow

The five existing workflows (`setup`, `run`, `catbench`, `finetune`, `distill`)
all have the same shape. A new tool follows it; there is no plugin framework to
learn. Not every tool needs all of it: a script that serves an existing
workflow only extends that workflow's recipe and tests.

| Part | Where | What it holds |
|---|---|---|
| Scripts | `scripts/<tool>.py` (or `run_examples/`) | every action the workflow executes, as a file a user can rerun without an agent |
| Recipe | `recipes/<tool>.md` | the agent procedure: inputs to ask for, the plan, approval, the commands, and the evidence that counts as done |
| Agent section | `AGENTS.md` §3 (a new request kind) and the §10 recipe map | when this workflow applies and which recipe it follows |
| Skill | `skills/<tool>/SKILL.md` | the trigger description and a pointer to its `AGENTS.md` section and recipe; no procedure of its own |
| How-to page | `docs/howto/<tool>.md`, plus a `mkdocs.yml` nav entry | the user's view: what to ask, what to prepare, what it asks, what you get, and a "Run it yourself" block |
| Tests | `tests/test_<tool>*.py` | GPU-free tests of the scripts, run in CI |

Rules every tool keeps:

- **Models only through the hub API.** Get a calculator or a result with
  `oh_my_mlip.resolve()`, `run()`, `Worker` or `get_calculator()`, and run
  model code with the interpreter `resolve()` returns. Never hard-code an env
  path, a weight path or a calculator constructor; the registry owns them.
- **Files, not inline code.** Anything the tool executes is written to disk
  first (a `.sh`, `.py` or config) and then run; never `python -c` with a
  generated body.
- **Refuse rather than guess.** Unsupported inputs stop with a message that
  names the next command; exit codes are documented in the recipe.
- **`--help` works anywhere.** Parse arguments before importing numpy, ase or
  a framework, and point at the right interpreter when one is missing.
- **One source per fact.** A skill names its `AGENTS.md` section and recipe;
  it does not restate them (`tests/test_onramp_contract_no_dup.py` and
  `tests/test_skill_contract_paths.py` check this).

Separately licensed engines stay in their own repository. oh-my-mlip is MIT;
distillation runs the GPL-2.0 [onthefly-distill](https://github.com/JinukMoon/onthefly-distill)
engine from its own checkout (`--repo <path>`), never copies its code into
this repository, and records the engine commit in the run's provenance. A
tool built on another licensed engine does the same, and its how-to page
names the engine and its license.

Registering a new skill with the Claude Code plugin:

1. add `./skills/<tool>/` to `skills` in `.claude-plugin/plugin.json` and
   mention it in the plugin description in `.claude-plugin/marketplace.json`;
2. run `python3 scripts/plugin_version.py --bump` (any change under
   `skills/` or to `AGENTS.md` needs a new plugin version, or installed
   plugins never receive it; `tests/test_plugin_manifest.py` enforces this);
3. add the `AGENTS.md` §3 section the skill points to, and the how-to page
   to the docs navigation.

A tool PR carries GPU-free tests that pass in CI and, if it runs models, a
GPU proof (next section).

---

## GPU proof

CI has no GPU, driver or Hugging Face token, so a change that installs or runs
models also needs a real run on a GPU host. Put the command, the commit it ran
at and the JSON verdict in the PR description.

- **A model or env change:** `python3 scripts/setup_verify.py <variant> --json`
  passes with `"degraded": false` (energy and forces computed, GPU use
  confirmed). Several variants: `python3 scripts/setup_sweep.py --targets
  <A,B,...>`. A new row enters `models.json` as `validation: "gpu_pending"`;
  the maintainer changes it to `validated_<arch>` (for example
  `validated_sm89`) in a PR once that verdict exists.
- **A tool:** the tool's own verifier on a real run, as the existing
  workflows do: `scripts/ft_verify.py` for a fine-tuned checkpoint,
  `scripts/distill_verify.py` for a distilled student,
  `scripts/catbench_report.py` for a benchmark.
- **Gated models** need your own Hugging Face login and an accepted license;
  never put a token in the PR.

Public docs state what the hub does, not measurements from one machine: keep
timings and host details in the PR, not in `docs/` or the README.

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
