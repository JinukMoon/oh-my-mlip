"""Tests for scripts/compare_equiv.py — the equivalence honesty gate.

GPU-free: builds synthetic equiv_result.json dicts in tmp_path and exercises the
comparator via importlib (matching tests/test_lint_recipes.py's load convention).
compare_equiv is pure stdlib + numpy, so no ase/torch/catbench is needed.

Coverage:
  * identical single-point        -> exit 0 / EQUIV: PASS
  * energy drift > tol            -> FAIL
  * missing system                -> FAIL
  * extra system                  -> FAIL
  * catbench_version mismatch     -> FAIL
  * relax terminal-geom RMSD>tol  -> FAIL
  * relax missing terminal_geom   -> FAIL
  * cross-GPU time                -> SKIPPED (never fails)
  * same-GPU time out-of-band     -> FAIL
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "compare_equiv.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_equiv", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


MOD = _load_module()


# ── fixture builders ──────────────────────────────────────────────────────────

def _prov(**over) -> dict:
    base = {
        "mode": "single-point",
        "model": "MACE",
        "dataset_sha256": "abc123",
        "catbench_version": "1.1.2",
        "weights_sha256": "wsha",
        "d3": False,
        "gpu_name": "NVIDIA RTX A5000",
    }
    base.update(over)
    return base


def _sp_doc(**prov_over) -> dict:
    return {
        "provenance": _prov(mode="single-point", **prov_over),
        "single_point": {
            "rxnA::slab": {"energy": -10.0, "fmax": 0.05, "natoms": 20, "formula": "Cu20"},
            "rxnA::adslab": {"energy": -12.0, "fmax": 0.04, "natoms": 22, "formula": "Cu20CO"},
        },
        "summary": {"n_structures": 2, "n_errors": 0},
    }


def _geom(shift: float = 0.0) -> dict:
    return {
        "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0 + shift], [0.0, 1.0, 0.0]],
        "cell": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]],
        "natoms": 3,
    }


def _relax_doc(*, geom_shift=0.0, drop_geom=False, time=1.0, gpu="NVIDIA RTX A5000") -> dict:
    sysd = {
        "relaxed_ads_energy": -1.5,
        "ref_ads_eng": -1.4,
        "anomaly": None,
        "slab_max_disp": 0.01,
        "adslab_max_disp": 0.02,
        "mean_time_per_step": time,
        "natoms": 3,
        "terminal_geom": None if drop_geom else _geom(geom_shift),
    }
    return {
        "provenance": _prov(mode="relax", gpu_name=gpu),
        "relax": {"rxnA": sysd},
        "aggregate": {
            "MAE": 0.1, "RMSE": 0.1, "anomaly_counts": {},
            "mean_time_per_step_overall": time, "n_systems": 1,
        },
    }


def _run_cli(tmp_path: Path, ours: dict, ref: dict, *extra: str):
    op = tmp_path / "ours.json"
    rp = tmp_path / "ref.json"
    op.write_text(json.dumps(ours), encoding="utf-8")
    rp.write_text(json.dumps(ref), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(op), str(rp), *extra],
        capture_output=True, text=True,
    )


# ── single-point ──────────────────────────────────────────────────────────────

def test_identical_single_point_passes(tmp_path: Path):
    proc = _run_cli(tmp_path, _sp_doc(), _sp_doc())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "EQUIV: PASS" in proc.stdout


def test_energy_drift_fails(tmp_path: Path):
    ours = _sp_doc()
    ours["single_point"]["rxnA::slab"]["energy"] = -10.0 + 0.5  # 0.025 eV/atom > 1e-3
    proc = _run_cli(tmp_path, ours, _sp_doc())
    assert proc.returncode != 0
    assert "EQUIV: FAIL" in proc.stdout
    assert "single_point.energy" in proc.stdout


def test_force_drift_fails(tmp_path: Path):
    ours = _sp_doc()
    ours["single_point"]["rxnA::slab"]["fmax"] = 0.05 + 0.5  # > force-tol 1e-2
    proc = _run_cli(tmp_path, ours, _sp_doc())
    assert proc.returncode != 0
    assert "single_point.forces" in proc.stdout


def test_missing_system_fails(tmp_path: Path):
    ours = _sp_doc()
    del ours["single_point"]["rxnA::adslab"]  # present in ref, absent here
    proc = _run_cli(tmp_path, ours, _sp_doc())
    assert proc.returncode != 0
    assert "single_point.keys" in proc.stdout


def test_extra_system_fails(tmp_path: Path):
    ours = _sp_doc()
    ours["single_point"]["rxnB::slab"] = {"energy": -5.0, "fmax": 0.01, "natoms": 10, "formula": "Ni10"}
    proc = _run_cli(tmp_path, ours, _sp_doc())
    assert proc.returncode != 0
    assert "single_point.keys" in proc.stdout


def test_catbench_version_mismatch_fails(tmp_path: Path):
    ours = _sp_doc(catbench_version="1.1.1")
    proc = _run_cli(tmp_path, ours, _sp_doc())
    assert proc.returncode != 0
    assert "provenance.catbench_version" in proc.stdout


def test_dataset_sha_mismatch_fails(tmp_path: Path):
    ours = _sp_doc(dataset_sha256="different")
    proc = _run_cli(tmp_path, ours, _sp_doc())
    assert proc.returncode != 0
    assert "provenance.dataset_sha256" in proc.stdout


# ── relax ─────────────────────────────────────────────────────────────────────

def test_identical_relax_passes(tmp_path: Path):
    proc = _run_cli(tmp_path, _relax_doc(), _relax_doc())
    assert proc.returncode == 0, proc.stdout
    assert "EQUIV: PASS" in proc.stdout


def test_relax_terminal_geometry_rmsd_fails(tmp_path: Path):
    ours = _relax_doc(geom_shift=0.5)  # RMSD ~ 0.29 Angstrom > geom-tol 1e-2
    proc = _run_cli(tmp_path, ours, _relax_doc())
    assert proc.returncode != 0
    assert "relax.geometry_rmsd" in proc.stdout


def test_relax_missing_terminal_geom_fails(tmp_path: Path):
    ours = _relax_doc(drop_geom=True)
    proc = _run_cli(tmp_path, ours, _relax_doc())
    assert proc.returncode != 0
    assert "relax.geometry_rmsd" in proc.stdout


# ── time (T3) ─────────────────────────────────────────────────────────────────

def test_cross_gpu_time_skipped_not_fail(tmp_path: Path):
    # Different GPUs -> time SKIPPED. Times differ wildly but must NOT fail.
    ours = _relax_doc(time=5.0, gpu="NVIDIA RTX A6000")
    ref = _relax_doc(time=1.0, gpu="NVIDIA RTX A5000")
    proc = _run_cli(tmp_path, ours, ref)
    assert proc.returncode == 0, proc.stdout
    assert "SKIPPED (not same-GPU)" in proc.stdout
    assert "EQUIV: PASS" in proc.stdout


def test_same_gpu_time_out_of_band_fails(tmp_path: Path):
    ours = _relax_doc(time=5.0)  # 5x ref -> ratio 5.0 outside [0.8,1.25]
    ref = _relax_doc(time=1.0)
    proc = _run_cli(tmp_path, ours, ref)
    assert proc.returncode != 0
    assert "time.ratio" in proc.stdout


def test_same_gpu_time_in_band_passes(tmp_path: Path):
    ours = _relax_doc(time=1.1)  # ratio 1.1 inside band
    ref = _relax_doc(time=1.0)
    proc = _run_cli(tmp_path, ours, ref)
    assert proc.returncode == 0, proc.stdout


# ── module-level compare() direct ─────────────────────────────────────────────

def test_compare_returns_all_ok_for_identical():
    checks, ok = MOD.compare(_sp_doc(), _sp_doc())
    assert ok
    assert all(c.ok for c in checks)


def test_coord_rmsd_shape_mismatch_returns_none():
    assert MOD.coord_rmsd([[0, 0, 0]], [[0, 0, 0], [1, 1, 1]]) is None
