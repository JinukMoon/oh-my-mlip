"""GPU-free tests for scripts/ft_run.py.

Covers the refusal gate (AC5's own failable oracle, C18), the live blocker
recheck (dpdata-style "pip install X" blockers, and the deepmd symlink
special-case), N4's `cd` requirement in the emitted `.sh`, and the generic
command renderer's registry-entrypoint + variant_args merge (exercised via a
UMA variant, which is where `variant_args` first earns its keep — 9.4).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")  # GPU-free CI has no numpy/ase
pytest.importorskip("ase")
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ft_run  # noqa: E402
from oh_my_mlip import registry as reg  # noqa: E402

FT_RUN = REPO_ROOT / "scripts" / "ft_run.py"


@pytest.fixture(autouse=True)
def _neutralize_local_state(monkeypatch):
    """Per-machine env adoptions / verified ledger must not change what these
    tests assert -- mirrors tests/test_catbench_jobgen.py."""
    monkeypatch.setattr(reg, "local_env_map", lambda *a, **k: {})
    monkeypatch.setattr(reg, "load_local_models", lambda *a, **k: {})


@pytest.fixture()
def synthetic_traj(tmp_path) -> Path:
    rng = np.random.default_rng(0)
    frames = []
    for i in range(5):
        a = bulk("Cu", "fcc", a=3.61, cubic=True)
        a.rattle(stdev=0.02, seed=i)
        a.calc = SinglePointCalculator(
            a, energy=-14.0 + float(rng.normal(0, 0.01)), forces=rng.normal(0, 0.1, size=(len(a), 3))
        )
        frames.append(a)
    path = tmp_path / "synthetic5.traj"
    write(path, frames)
    return path


def _find_shellcheck() -> str | None:
    found = shutil.which("shellcheck")
    if found:
        return found
    candidate = Path(sys.exec_prefix) / "bin" / "shellcheck"
    return str(candidate) if candidate.is_file() else None


# ── refusal gate: status-based exit 2 (AC5, C18) ─────────────────────────────
def test_eqnorm_not_supported_exits_2(tmp_path, synthetic_traj):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(FT_RUN), "Eqnorm", "--dataset", str(synthetic_traj), "--out", str(out), "--emit-only"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "not-supported" in proc.stderr
    assert "evidence" in proc.stderr
    assert not out.exists() or not any(out.iterdir())  # nothing materialized on refusal


# ── refusal gate: runnable_as_installed-based exit 3 (AC5, C18) ──────────────
def test_orb_not_runnable_exits_3(tmp_path, synthetic_traj):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(FT_RUN), "ORB", "--dataset", str(synthetic_traj), "--out", str(out), "--emit-only"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 3
    assert "wandb" in proc.stderr
    assert "finetune.py" in proc.stderr


def test_unknown_model_is_a_clean_usage_error(tmp_path, synthetic_traj):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(FT_RUN), "NotAModel", "--dataset", str(synthetic_traj), "--out", str(out), "--emit-only"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1


# ── live blocker recheck ──────────────────────────────────────────────────────
def test_live_recheck_drops_resolved_pip_install_blocker():
    blockers = ["yaml is not installed in the X env -- pip install yaml before doing anything"]
    remaining = ft_run.live_recheck_blockers("SomeFamily", blockers, sys.executable)
    assert remaining == []


def test_live_recheck_keeps_unresolved_pip_install_blocker():
    blockers = ["missing -- pip install this_package_does_not_exist_xyz123"]
    remaining = ft_run.live_recheck_blockers("SomeFamily", blockers, sys.executable)
    assert remaining == blockers


def test_live_recheck_drops_deepmd_symlink_blocker_only_for_deepmd_family():
    blockers = ["the shipped weight foo.pth must be symlinked to a .pt name before dp dispatches on it"]
    assert ft_run.live_recheck_blockers("DeePMD", blockers, sys.executable) == []
    assert ft_run.live_recheck_blockers("DPA4", blockers, sys.executable) == []
    assert ft_run.live_recheck_blockers("ORB", blockers, sys.executable) == blockers


# ── N4: emitted .sh cd's to the absolute out dir before exec ─────────────────
def test_render_sh_cds_to_absolute_out_dir(tmp_path):
    resolved = {"python": "/fake/env/bin/python", "env_run": {}}
    text = ft_run.render_sh(resolved, ["/fake/env/bin/mace_run_train", "--epochs=2"], tmp_path)
    lines = text.splitlines()
    cd_lines = [ln for ln in lines if ln.startswith("cd ")]
    assert len(cd_lines) == 1
    assert cd_lines[0] == f'cd "{tmp_path.resolve()}"'
    exec_idx = next(i for i, ln in enumerate(lines) if ln.startswith("exec "))
    assert lines.index(cd_lines[0]) < exec_idx


def test_render_sh_exports_env_run(tmp_path):
    resolved = {"python": "/fake/env/bin/python", "env_run": {"LD_LIBRARY_PATH": ""}}
    text = ft_run.render_sh(resolved, ["/fake/env/bin/dp"], tmp_path)
    assert 'export LD_LIBRARY_PATH=""' in text


def test_slurm_body_identical_below_header(tmp_path):
    resolved = {"python": "/fake/env/bin/python", "env_run": {}}
    argv = ["/fake/env/bin/mace_run_train"]
    local_text = ft_run.render_sh(resolved, argv, tmp_path)
    slurm_text = ft_run.render_slurm(resolved, argv, tmp_path, job_name="j", partition="g1")
    body = ft_run._render_body(resolved, argv, tmp_path)
    assert local_text.endswith(body)
    assert slurm_text.endswith(body)
    assert "#SBATCH --partition=g1" in slurm_text


# ── build_generic: registry entrypoint + variant_args merge (UMA variant) ───
def test_build_generic_uma_variant_merges_variant_args(tmp_path):
    resolved = reg.resolve("UMA-s-1p2-OC20")
    finetune = reg.load_models()["UMA"]["versions"]["UMA-s-1p2-OC20"]["finetune"]
    assert finetune.get("variant_args"), "fixture assumption: this variant carries variant_args"

    out = tmp_path / "ft_uma"
    out.mkdir()
    ctx = ft_run.Context(
        family="UMA", version="UMA-s-1p2-OC20", finetune=finetune, resolved=resolved,
        out=out, epochs=2, batch_size=2, device="cpu", dataset_paths={}, elements=["Cu"],
    )
    spec = ft_run.build_generic(ctx)

    entrypoint_first_token = finetune["entrypoint"].split()[0]
    assert entrypoint_first_token in spec.argv[0]
    for k, v in finetune["variant_args"].items():
        assert f"{k}={v}" in spec.argv

    if spec.config_path is not None:
        spec.config_path.write_text(spec.config_text)

    sh_path = out / "finetune_UMA-s-1p2-OC20.sh"
    sh_path.write_text(ft_run.render_sh(resolved, spec.argv, out))
    sh_path.chmod(0o755)

    shellcheck = _find_shellcheck()
    if shellcheck is None:
        pytest.skip("shellcheck not available on PATH")
    proc = subprocess.run([shellcheck, str(sh_path)], capture_output=True, text=True)
    assert proc.returncode == 0, f"{sh_path}: {proc.stdout}\n{proc.stderr}"


def test_entrypoint_bin_honors_resolved_python_dir():
    resolved = {"python": "/some/env/bin/python"}
    assert ft_run.entrypoint_bin(resolved, "mace_run_train") == "/some/env/bin/mace_run_train"


def test_resolve_foundation_checkpoint_from_inference_line():
    resolved = reg.resolve("MACE", "MACE-MPA-0")
    assert ft_run.resolve_foundation_checkpoint("MACE-MPA-0", resolved) == "medium-mpa-0"


def test_resolve_foundation_checkpoint_override_for_deepmd_ft():
    resolved = reg.resolve("DeePMD")
    ckpt = ft_run.resolve_foundation_checkpoint("DPA-3.1-3M-FT", resolved)
    assert ckpt.endswith("dpa-3.1-3m-ft.pth")
    assert "frozen-omat24" not in ckpt
