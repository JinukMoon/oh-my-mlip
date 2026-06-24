"""Negative / integrity tests that prove CI catches breakage.

Three categories:

(a) A deliberately malformed models.json fragment FAILS schema validation.
(b) parse_env_run rejects raw-shell strings (trust boundary enforcement).
(c) gen_status_table --check would fail when the README table drifts.

These tests do NOT need a GPU, conda env, torch, or ase.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import jsonschema
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = Path(__file__).resolve().parent / "schema"

from oh_my_mlip.registry import RegistryError, parse_env_run

# ── (a) Schema validation: malformed models.json fragments ────────────────────

def _models_schema() -> dict:
    return json.loads((SCHEMA_DIR / "models.schema.json").read_text(encoding="utf-8"))


def _good_version() -> dict:
    """A minimal valid version entry that passes the schema."""
    return {
        "gated": False,
        "license_url": None,
        "weights": "bundled",
        "validation": "validated_sm89",
        "inference": ["calc = SomeCalc()"],
    }


def _minimal_valid_doc() -> dict:
    """A minimal models.json document that passes the schema."""
    return {
        "_meta": {"description": "test"},
        "TestModel": {
            "env": "test_env",
            "python": "${OH_MY_MLIP_HOME}/envs/test_env/bin/python",
            "versions": {
                "TestModel-v1": _good_version(),
            },
        },
    }


def test_valid_fragment_passes_schema():
    """Baseline: the minimal valid doc must pass (sanity check on the schema itself)."""
    jsonschema.validate(instance=_minimal_valid_doc(), schema=_models_schema())


@pytest.mark.parametrize(
    "mutation,description",
    [
        # missing 'gated'
        (
            lambda v: v.pop("gated"),
            "missing required field 'gated'",
        ),
        # missing 'weights'
        (
            lambda v: v.pop("weights"),
            "missing required field 'weights'",
        ),
        # missing 'validation'
        (
            lambda v: v.pop("validation"),
            "missing required field 'validation'",
        ),
        # missing 'inference'
        (
            lambda v: v.pop("inference"),
            "missing required field 'inference'",
        ),
        # bad validation enum value
        (
            lambda v: v.update({"validation": "not_a_real_state"}),
            "invalid validation enum value",
        ),
        # bad weights enum value
        (
            lambda v: v.update({"weights": "by-magic"}),
            "invalid weights enum value",
        ),
        # missing top-level 'env'
        (
            None,  # handled specially below
            "missing top-level 'env'",
        ),
        # missing top-level 'versions'
        (
            None,  # handled specially below
            "missing top-level 'versions'",
        ),
    ],
    ids=[
        "missing_gated",
        "missing_weights",
        "missing_validation",
        "missing_inference",
        "bad_validation_enum",
        "bad_weights_enum",
        "missing_top_env",
        "missing_top_versions",
    ],
)
def test_malformed_fragment_fails_schema(mutation, description):
    """Each of these deliberately broken fragments must be REJECTED by the schema."""
    doc = _minimal_valid_doc()

    if description == "missing top-level 'env'":
        del doc["TestModel"]["env"]
    elif description == "missing top-level 'versions'":
        del doc["TestModel"]["versions"]
    else:
        # Apply the mutation to a copy of the version entry.
        version_entry = _good_version()
        mutation(version_entry)
        doc["TestModel"]["versions"]["TestModel-v1"] = version_entry

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=doc, schema=_models_schema())


def test_invalid_json_would_fail_parse():
    """Proves that bad JSON is caught before schema validation even runs."""
    bad_json = '{"MACE": {"env": "mace", "python": "...", "versions": {}'
    with pytest.raises(json.JSONDecodeError):
        json.loads(bad_json)


# ── (b) parse_env_run raw-shell rejection (trust boundary) ───────────────────
# These cases are already parametrised in test_registry.py; we include a
# representative subset here so this file is self-contained as a
# "trust-boundary policy" reference.

@pytest.mark.parametrize(
    "bad_env_run",
    [
        "$(rm -rf /)",              # command substitution; not KEY=VALUE
        "FOO=bar; rm x",            # semi-colon injection; FOO not allowlisted
        "LD_LIBRARY_PATH=$(id)",    # allowlisted key but shell substitution in value
        "PATH=/usr/bin",            # PATH not on the allowlist
        "LD_LIBRARY_PATH=a`b`",     # backtick metacharacter
        "OMP_NUM_THREADS=4|cat",    # pipe metacharacter
        "CUDA_VISIBLE_DEVICES=0 && rm -rf /",  # && injection
    ],
    ids=[
        "cmd_substitution",
        "semicolon_injection",
        "shell_substitution_in_value",
        "path_not_allowlisted",
        "backtick_in_value",
        "pipe_in_value",
        "and_injection",
    ],
)
def test_parse_env_run_rejects_raw_shell(bad_env_run):
    """parse_env_run must raise RegistryError for ANY non-KEY=VALUE / non-allowlisted token.

    This is the trust boundary: env_run is applied as subprocess env, never
    shell-interpolated. A single rejected token keeps the subprocess safe.
    """
    with pytest.raises(RegistryError):
        parse_env_run(bad_env_run)


def test_parse_env_run_accepts_allowlisted_tokens():
    """Confirm the allowlist itself still works after the negative tests."""
    result = parse_env_run("LD_LIBRARY_PATH=\"\" OMP_NUM_THREADS=4")
    assert result == {"LD_LIBRARY_PATH": "", "OMP_NUM_THREADS": "4"}


# ── (c) gen_status_table --check fails on drift ───────────────────────────────

def test_status_table_check_passes_on_current_readme():
    """gen_status_table --check must exit 0 on the current (committed) README."""
    script = REPO_ROOT / "scripts" / "gen_status_table.py"
    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        "gen_status_table --check failed on the current README.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_status_table_check_fails_on_drift(tmp_path, monkeypatch):
    """gen_status_table --check must exit non-zero when the README table is stale.

    We make a modified README that has an obviously wrong status table and run
    the check against it via monkeypatching the script's README path constant.
    """
    # Build a README with a deliberately wrong table block.
    stale_readme = textwrap.dedent("""\
        # oh-my-mlip

        <!-- STATUS_TABLE_START -->
        | Model | Framework | Weights | Validation | Gated | Shipped (v1) |
        |---|---|---|---|---|---|
        | FakeModel | FakeFramework | bundled | validated (sm89) | no | yes |
        <!-- STATUS_TABLE_END -->
    """)
    readme_path = tmp_path / "README.md"
    readme_path.write_text(stale_readme, encoding="utf-8")

    # Copy models.json to tmp_path so the script can find it.
    models_json = REPO_ROOT / "models.json"
    (tmp_path / "models.json").write_text(
        models_json.read_text(encoding="utf-8"), encoding="utf-8"
    )

    # Run the check script with its repo root overridden to tmp_path.
    # We use an env-var trick: monkeypatch PYTHONPATH isn't needed here;
    # we just pass the script directly and let it find files relative to itself.
    # Instead, we run it as a module and inject a patched environment by
    # temporarily rewriting the script's file pointers — the cleanest approach
    # is to run via subprocess with a helper that overrides REPO_ROOT.
    helper = tmp_path / "run_check.py"
    helper.write_text(
        textwrap.dedent(f"""\
            import sys
            from pathlib import Path
            import types

            # Patch the module-level paths before importing gen_status_table.
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "gen_status_table",
                {str(REPO_ROOT / "scripts" / "gen_status_table.py")!r},
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Override the file pointers to use tmp_path.
            mod.MODELS_JSON = Path({str(tmp_path)!r}) / "models.json"
            mod.README = Path({str(tmp_path)!r}) / "README.md"

            sys.exit(mod.main(["--check"]))
        """),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(helper)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        "gen_status_table --check should have failed on a stale README but returned 0.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
