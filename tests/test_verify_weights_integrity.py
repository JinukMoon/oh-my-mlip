"""Tests for scripts/verify_weights_integrity.py — GPU-free weight-fingerprint check.

These tests are MOCKED: instead of downloading multi-GB weights, a tiny temp file
is created and its sha256 is written into a stand-in models.json so the match /
mismatch paths are exercised deterministically. The real registry's table view is
also smoke-tested (recorded fingerprints, nothing hashed).

Covers:
  * matches-validated   — file sha256 equals the recorded weights_sha256.
  * MISMATCH            — file differs from the recorded fingerprint (exit 1).
  * fingerprint-pending — no weights_sha256 recorded for the model.
  * file-not-found      — a path is supplied but absent (exit 1).
  * the real models.json table shows the 15 validated fingerprints.

No GPU, conda, torch, or ase required.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "verify_weights_integrity.py"
MODELS_JSON = REPO_ROOT / "models.json"


def _load():
    spec = importlib.util.spec_from_file_location("verify_weights_integrity", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _mock_models(weights_sha256: str | None, size: int) -> dict:
    """A minimal models.json with one model carrying (or not) a fingerprint."""
    ver: dict = {
        "gated": False,
        "license_url": None,
        "weights": "on-demand-hf",
        "weights_fetch": "url",
        "weights_source": "https://example.invalid/w",
        "validation": "validated_sm89",
        "inference": ["calc = X()"],
    }
    if weights_sha256 is not None:
        ver["weights_sha256"] = weights_sha256
        ver["weights_size"] = size
    return {
        "_meta": {"description": "mock"},
        "Mock": {
            "env": "mock",
            "python": "${OH_MY_MLIP_HOME}/envs/mock/bin/python",
            "versions": {"Mock-1": ver},
        },
    }


# ── Mocked match / mismatch ───────────────────────────────────────────────────

def test_matches_validated(tmp_path):
    mod = _load()
    weight = tmp_path / "w.bin"
    weight.write_bytes(b"some-weight-bytes")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    models = _mock_models(digest, weight.stat().st_size)

    rows, rc = mod.run(models, model="Mock-1", file_path=weight)
    assert rc == 0
    assert rows[0][1] == mod.MATCH


def test_mismatch_exit_nonzero(tmp_path):
    mod = _load()
    weight = tmp_path / "w.bin"
    weight.write_bytes(b"the-downloaded-bytes")
    # Record a DIFFERENT fingerprint -> the file must MISMATCH.
    wrong = hashlib.sha256(b"a-different-validated-checkpoint").hexdigest()
    models = _mock_models(wrong, 999)

    rows, rc = mod.run(models, model="Mock-1", file_path=weight)
    assert rc == 1
    assert rows[0][1] == mod.MISMATCH


def test_fingerprint_pending(tmp_path):
    mod = _load()
    models = _mock_models(None, 0)
    rows, rc = mod.run(models, model="Mock-1", file_path=None)
    assert rc == 0
    assert rows[0][1] == mod.PENDING


def test_file_not_found_exit_nonzero(tmp_path):
    mod = _load()
    models = _mock_models("a" * 64, 10)
    missing = tmp_path / "does-not-exist.bin"
    rows, rc = mod.run(models, model="Mock-1", file_path=missing)
    assert rc == 1
    assert rows[0][1] == mod.NOT_FOUND


def test_unknown_model_exits_two():
    mod = _load()
    models = _mock_models("a" * 64, 10)
    _rows, rc = mod.run(models, model="No-Such-Model", file_path=None)
    assert rc == 2


# ── Real registry smoke test ──────────────────────────────────────────────────

def test_real_registry_has_fifteen_recorded_fingerprints():
    mod = _load()
    models = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    rows, rc = mod.run(models)
    assert rc == 0
    recorded = [name for name, status, _ in rows if status == mod.RECORDED]
    assert len(recorded) == 15, recorded


def test_cli_table_runs_on_real_registry(capsys):
    mod = _load()
    rc = mod.main([str(MODELS_JSON)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "verify_weights_integrity:" in out
    assert "fingerprint-pending" in out
