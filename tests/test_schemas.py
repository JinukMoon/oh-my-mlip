"""JSON-schema validation tests for models.json and dist_manifest.json.

These run on every PR without a GPU, conda env, torch, or ase.
They guard against:
  - structural drift in models.json (missing honesty fields, bad enum values)
  - structural drift in dist_manifest.json (missing required keys)

The models schema requires the four honesty fields (gated, weights, validation,
inference) on every version entry — this guards against an "all-validated"
overclaim by keeping each model's true validation state in the registry.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = Path(__file__).resolve().parent / "schema"

MODELS_JSON = REPO_ROOT / "models.json"
DIST_MANIFEST_JSON = REPO_ROOT / "dist_manifest.json"
MODELS_SCHEMA_FILE = SCHEMA_DIR / "models.schema.json"
DIST_MANIFEST_SCHEMA_FILE = SCHEMA_DIR / "dist_manifest.schema.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def models_schema() -> dict:
    return _load(MODELS_SCHEMA_FILE)


@pytest.fixture(scope="module")
def dist_manifest_schema() -> dict:
    return _load(DIST_MANIFEST_SCHEMA_FILE)


@pytest.fixture(scope="module")
def models_data() -> dict:
    return _load(MODELS_JSON)


@pytest.fixture(scope="module")
def dist_manifest_data() -> dict:
    return _load(DIST_MANIFEST_JSON)


# ── models.json ──────────────────────────────────────────────────────────────

def test_models_json_is_valid_json():
    """models.json must be parseable (most basic gate)."""
    data = _load(MODELS_JSON)
    assert isinstance(data, dict)


def test_models_json_passes_schema(models_data, models_schema):
    """models.json must satisfy the full schema (incl. honesty fields on every version)."""
    jsonschema.validate(instance=models_data, schema=models_schema)


def test_models_json_has_mace_and_sevennet(models_data):
    """v1 shipped models must be present in the registry."""
    assert "MACE" in models_data
    assert "SevenNet" in models_data


def test_models_json_honesty_fields_all_versions(models_data):
    """Every version entry must carry gated, weights, validation.
    This is an explicit belt-and-suspenders check beyond the schema."""
    missing: list[str] = []
    for fw, info in models_data.items():
        if fw.startswith("_"):
            continue
        for ver, vinfo in info.get("versions", {}).items():
            for field in ("gated", "weights", "validation"):
                if field not in vinfo:
                    missing.append(f"{fw}/{ver}: missing '{field}'")
    assert not missing, "Honesty fields missing:\n" + "\n".join(missing)


def test_models_json_validation_enum_values(models_data):
    """validation field must be one of the four allowed enum values."""
    allowed = {"validated_sm86", "validated_sm89", "gpu_pending", "cpu_only"}
    bad: list[str] = []
    for fw, info in models_data.items():
        if fw.startswith("_"):
            continue
        for ver, vinfo in info.get("versions", {}).items():
            val = vinfo.get("validation", "")
            if val not in allowed:
                bad.append(f"{fw}/{ver}: validation={val!r}")
    assert not bad, "Invalid validation values:\n" + "\n".join(bad)


def test_models_json_weights_enum_values(models_data):
    """weights field must be one of the three allowed enum values."""
    allowed = {"bundled", "auto-download", "on-demand-hf"}
    bad: list[str] = []
    for fw, info in models_data.items():
        if fw.startswith("_"):
            continue
        for ver, vinfo in info.get("versions", {}).items():
            w = vinfo.get("weights", "")
            if w not in allowed:
                bad.append(f"{fw}/{ver}: weights={w!r}")
    assert not bad, "Invalid weights values:\n" + "\n".join(bad)


def test_models_json_gated_models_have_license_url(models_data):
    """gated=true requires a non-null license_url so users can find the license."""
    bad: list[str] = []
    for fw, info in models_data.items():
        if fw.startswith("_"):
            continue
        for ver, vinfo in info.get("versions", {}).items():
            if vinfo.get("gated") and not vinfo.get("license_url"):
                bad.append(f"{fw}/{ver}: gated=true but license_url is null/missing")
    assert not bad, "\n".join(bad)


def test_models_json_default_version_is_a_valid_version_key(models_data):
    """If a framework declares default_version, it must be one of its version keys."""
    bad: list[str] = []
    for fw, info in models_data.items():
        if fw.startswith("_"):
            continue
        dv = info.get("default_version")
        if dv is not None and dv not in info.get("versions", {}):
            bad.append(f"{fw}: default_version={dv!r} not in {list(info.get('versions', {}))}")
    assert not bad, "\n".join(bad)


def test_models_json_multiversion_frameworks_have_default_version(models_data):
    """Every multi-version framework must declare default_version so resolve(model, None) works."""
    missing: list[str] = []
    for fw, info in models_data.items():
        if fw.startswith("_"):
            continue
        versions = info.get("versions", {})
        if len(versions) > 1 and "default_version" not in info:
            missing.append(fw)
    assert not missing, (
        "multi-version frameworks without default_version (resolve(model, None) "
        "would raise): " + ", ".join(missing)
    )


def test_models_json_no_tgm_paths(models_data):
    """Public registry must not contain any /TGM/ absolute paths."""
    raw = MODELS_JSON.read_text(encoding="utf-8")
    assert "/TGM/" not in raw, "models.json contains /TGM/ path(s) — use $OH_MY_MLIP_HOME"


# ── dist_manifest.json ────────────────────────────────────────────────────────

def test_dist_manifest_is_valid_json():
    data = _load(DIST_MANIFEST_JSON)
    assert isinstance(data, dict)


def test_dist_manifest_passes_schema(dist_manifest_data, dist_manifest_schema):
    jsonschema.validate(instance=dist_manifest_data, schema=dist_manifest_schema)


def test_dist_manifest_v1_envs_present(dist_manifest_data):
    """v1-shipped envs (mace, sevennet) must appear in the manifest."""
    assert "mace" in dist_manifest_data
    assert "sevennet" in dist_manifest_data


def test_dist_manifest_required_keys(dist_manifest_data):
    """Every non-meta entry must carry all required distribution keys."""
    required = {"env", "hf_repo", "revision", "sha256", "unpack_size_bytes", "min_driver_version"}
    missing: list[str] = []
    for env, entry in dist_manifest_data.items():
        if env.startswith("_"):
            continue
        for key in required:
            if key not in entry:
                missing.append(f"{env}: missing '{key}'")
    assert not missing, "\n".join(missing)
