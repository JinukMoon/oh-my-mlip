"""Tests for scripts/verify_compile.py — GPU-free accel-block shape check.

Positive: the real models.json passes; every accel block honors the
GPU-unverified contract (verified=false, last_gpu_verified=null, recognized
provenance) and the "shape-only; gpu_unverified" banner is printed.

Negative: an accel block that claims verified=true, or carries a non-null
last_gpu_verified, or an unknown provenance, is flagged.

No GPU, conda, torch, or ase required.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "verify_compile.py"
MODELS_JSON = REPO_ROOT / "models.json"


def _load():
    spec = importlib.util.spec_from_file_location("verify_compile", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _models() -> dict:
    return json.loads(MODELS_JSON.read_text(encoding="utf-8"))


# ── Positive ──────────────────────────────────────────────────────────────────

def test_real_registry_has_accel_blocks_and_all_pass():
    mod = _load()
    models = _models()
    blocks, errors = mod.verify(models)
    # NequIP, Allegro (accel + accel_lammps), SevenNet, EquFlash v1 -> at least 4.
    assert len(blocks) >= 4, blocks
    assert errors == [], errors


def test_cli_prints_banner_and_exits_zero(capsys):
    mod = _load()
    rc = mod.main([str(MODELS_JSON)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "shape-only; gpu_unverified" in out


def test_every_block_marked_unverified():
    mod = _load()
    for _label, blk in mod.collect_accel_blocks(_models()):
        assert blk.get("verified") is False
        assert blk.get("last_gpu_verified") is None
        assert blk.get("provenance") in mod.ALLOWED_PROVENANCE


# ── Negative ──────────────────────────────────────────────────────────────────

def test_verified_true_is_flagged():
    mod = _load()
    models = copy.deepcopy(_models())
    models["NequIP"]["accel"]["verified"] = True
    _blocks, errors = mod.verify(models)
    assert any("verified must be false" in e for e in errors), errors


def test_nonnull_last_gpu_verified_is_flagged():
    mod = _load()
    models = copy.deepcopy(_models())
    models["NequIP"]["accel"]["last_gpu_verified"] = "2026-01-01"
    _blocks, errors = mod.verify(models)
    assert any("last_gpu_verified must be null" in e for e in errors), errors


def test_unknown_provenance_is_flagged():
    mod = _load()
    models = copy.deepcopy(_models())
    models["NequIP"]["accel"]["provenance"] = "made-up"
    _blocks, errors = mod.verify(models)
    assert any("provenance must be one of" in e for e in errors), errors


def test_missing_required_key_is_flagged():
    mod = _load()
    models = copy.deepcopy(_models())
    del models["NequIP"]["accel"]["load_note"]
    _blocks, errors = mod.verify(models)
    assert any("missing required key 'load_note'" in e for e in errors), errors
