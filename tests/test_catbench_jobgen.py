"""GPU-free tests for scripts/catbench_jobgen.py.

catbench_jobgen owns the three emitted catbench job artifacts:
  - jobs/catbench_<MLIP>.py           verbatim resolve() codegen
  - jobs/run_catbench_<MLIP>.sh       the AC7 rerun unit (cd + env_run exports + exec)
  - jobs/run_slurm_<MLIP>.sh          SBATCH header + the IDENTICAL body as the runner

These tests assert the properties that actually make AC7 (F14) hold: a
missing `env_run` export or a missing `cd` used to make the `.py` (or the old
`.sh`-less flow) silently unrerunnable for NequIP/DeePMD/DPA4 while a MACE
spot-check could never catch it (MACE declares no `env_run` — C4). There is
DELIBERATELY no host-independence assertion here: resolve() emits absolute
host weight paths for 11/20 families (F13), so such an assertion would pass
in CI and fail on the maintainer's box.
"""
from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import catbench_jobgen  # noqa: E402
from oh_my_mlip import registry as reg  # noqa: E402
from oh_my_mlip import resolve  # noqa: E402


@pytest.fixture(autouse=True)
def _neutralize_local_state(monkeypatch):
    """Per-machine local state (env adoptions + verified ledger) must not
    change what these tests assert — mirrors tests/test_registry.py."""
    monkeypatch.setattr(reg, "local_env_map", lambda *a, **k: {})
    monkeypatch.setattr(reg, "load_local_models", lambda *a, **k: {})


def _spec(model: str, version: str | None = None) -> dict:
    return resolve(model, version)


# ── render_model_script: ast.parse ───────────────────────────────────────────
def test_generated_py_parses(tmp_path):
    spec = _spec("MACE", "MACE-MPA-0")
    text = catbench_jobgen.render_model_script(spec, "MACE-MPA-0", "Example", 3, False)
    ast.parse(text)  # raises SyntaxError on failure


def test_generated_py_parses_with_d3():
    spec = _spec("MACE", "MACE-MPA-0")
    text = catbench_jobgen.render_model_script(spec, "MACE-MPA-0_D3", "Example", 3, True)
    ast.parse(text)
    assert "DispersionCorrection" in text


# ── byte-identical regeneration (C3's determinism property) ─────────────────
def test_render_model_script_byte_identical_regeneration():
    spec = _spec("MACE", "MACE-MPA-0")
    a = catbench_jobgen.render_model_script(spec, "MACE-MPA-0", "Example", 3, False)
    b = catbench_jobgen.render_model_script(spec, "MACE-MPA-0", "Example", 3, False)
    assert a == b


def test_render_runner_sh_byte_identical_regeneration(tmp_path):
    spec = _spec("MACE", "MACE-MPA-0")
    jobfile = tmp_path / "jobs" / "catbench_MACE-MPA-0.py"
    a = catbench_jobgen.render_runner_sh(spec, jobfile, tmp_path)
    b = catbench_jobgen.render_runner_sh(spec, jobfile, tmp_path)
    assert a == b


def test_emit_byte_identical_regeneration(tmp_path):
    spec = _spec("MACE", "MACE-MPA-0")
    first = catbench_jobgen.emit(spec, "MACE-MPA-0", "Example", 3, False, tmp_path)
    text1_py = first["py"].read_text()
    text1_sh = first["sh"].read_text()
    second = catbench_jobgen.emit(spec, "MACE-MPA-0", "Example", 3, False, tmp_path)
    assert second["py"].read_text() == text1_py
    assert second["sh"].read_text() == text1_sh


# ── env_run exports: both directions (C4 — never MACE alone) ────────────────
@pytest.mark.parametrize(
    "model, expected_key, expected_value",
    [
        ("NequIP", "MAX_JOBS", "4"),
        ("DeePMD", "LD_LIBRARY_PATH", ""),
        ("DPA4", "LD_LIBRARY_PATH", "/usr/lib/wsl/lib"),
    ],
)
def test_runner_sh_exports_env_run_key(tmp_path, model, expected_key, expected_value):
    spec = _spec(model)
    jobfile = tmp_path / "jobs" / f"catbench_{spec['version']}.py"
    text = catbench_jobgen.render_runner_sh(spec, jobfile, tmp_path)
    assert f'export {expected_key}="{expected_value}"' in text
    # exactly one export line per env_run key (not zero, not duplicated)
    assert text.count(f"export {expected_key}=") == 1


def test_runner_sh_no_export_for_mace(tmp_path):
    spec = _spec("MACE", "MACE-MPA-0")
    jobfile = tmp_path / "jobs" / "catbench_MACE-MPA-0.py"
    text = catbench_jobgen.render_runner_sh(spec, jobfile, tmp_path)
    assert spec["env_run"] == {}
    assert "export " not in text


def test_runner_sh_export_count_matches_env_run(tmp_path):
    spec = _spec("DPA4")
    jobfile = tmp_path / "jobs" / f"catbench_{spec['version']}.py"
    text = catbench_jobgen.render_runner_sh(spec, jobfile, tmp_path)
    export_lines = [ln for ln in text.splitlines() if ln.startswith("export ")]
    assert len(export_lines) == len(spec["env_run"])


# ── cd to the absolute workdir before exec (N4) ──────────────────────────────
def test_runner_sh_cds_to_absolute_workdir(tmp_path):
    spec = _spec("MACE", "MACE-MPA-0")
    jobfile = tmp_path / "jobs" / "catbench_MACE-MPA-0.py"
    text = catbench_jobgen.render_runner_sh(spec, jobfile, tmp_path)
    lines = text.splitlines()
    cd_lines = [ln for ln in lines if ln.startswith("cd ")]
    assert len(cd_lines) == 1
    assert cd_lines[0] == f'cd "{tmp_path.resolve()}"'
    # cd must precede exec
    assert lines.index(cd_lines[0]) < next(i for i, ln in enumerate(lines) if ln.startswith("exec "))


def test_runner_sh_execs_the_materialized_py(tmp_path):
    spec = _spec("MACE", "MACE-MPA-0")
    jobfile = tmp_path / "jobs" / "catbench_MACE-MPA-0.py"
    text = catbench_jobgen.render_runner_sh(spec, jobfile, tmp_path)
    assert f'exec "{spec["python"]}" "{jobfile.resolve()}"' in text
    assert text.startswith("#!/bin/sh\n")


# ── SLURM: identical body below the header (C2/C17) ──────────────────────────
def test_slurm_body_byte_identical_below_header(tmp_path):
    spec = _spec("MACE", "MACE-MPA-0")
    jobfile = tmp_path / "jobs" / "catbench_MACE-MPA-0.py"
    local_text = catbench_jobgen.render_runner_sh(spec, jobfile, tmp_path)
    slurm_text = catbench_jobgen.render_slurm_sh(
        spec, jobfile, tmp_path, job_name="catbench_MACE-MPA-0", partition="test"
    )
    body = catbench_jobgen._render_body(spec, jobfile, tmp_path)
    assert local_text.endswith(body)
    assert slurm_text.endswith(body)
    assert slurm_text != local_text  # the header must actually differ
    assert "#SBATCH --partition=test" in slurm_text


def test_slurm_sh_byte_identical_regeneration(tmp_path):
    spec = _spec("MACE", "MACE-MPA-0")
    jobfile = tmp_path / "jobs" / "catbench_MACE-MPA-0.py"
    a = catbench_jobgen.render_slurm_sh(spec, jobfile, tmp_path, job_name="j", partition="gpu")
    b = catbench_jobgen.render_slurm_sh(spec, jobfile, tmp_path, job_name="j", partition="gpu")
    assert a == b


# ── emit(): both files (+ slurm) land on disk ────────────────────────────────
def test_emit_writes_py_and_sh(tmp_path):
    spec = _spec("MACE", "MACE-MPA-0")
    artifacts = catbench_jobgen.emit(spec, "MACE-MPA-0", "Example", 3, False, tmp_path)
    assert artifacts["py"] == tmp_path / "jobs" / "catbench_MACE-MPA-0.py"
    assert artifacts["sh"] == tmp_path / "jobs" / "run_catbench_MACE-MPA-0.sh"
    assert artifacts["py"].is_file()
    assert artifacts["sh"].is_file()
    assert "slurm" not in artifacts
    ast.parse(artifacts["py"].read_text())


def test_emit_slurm_true_also_writes_slurm_sh(tmp_path):
    spec = _spec("DPA4")
    mlip_name = spec["version"]
    artifacts = catbench_jobgen.emit(
        spec, mlip_name, "Example", 3, False, tmp_path, slurm=True, partition="g1",
    )
    assert artifacts["slurm"] == tmp_path / "jobs" / f"run_slurm_{mlip_name}.sh"
    assert artifacts["slurm"].is_file()
    text = artifacts["slurm"].read_text()
    assert "#SBATCH --partition=g1" in text
    assert 'export LD_LIBRARY_PATH="/usr/lib/wsl/lib"' in text


# ── _D3 suffix present iff --d3 (one artifact triple per model+version) ─────
def test_d3_suffix_present_iff_d3(tmp_path):
    spec = _spec("MACE", "MACE-MPA-0")
    no_d3 = catbench_jobgen.emit(spec, "MACE-MPA-0", "Example", 3, False, tmp_path)
    with_d3 = catbench_jobgen.emit(spec, "MACE-MPA-0_D3", "Example", 3, True, tmp_path)
    assert no_d3["py"].name == "catbench_MACE-MPA-0.py"
    assert with_d3["py"].name == "catbench_MACE-MPA-0_D3.py"
    assert "_D3" not in no_d3["py"].read_text()
    assert "DispersionCorrection" in with_d3["py"].read_text()
    assert "DispersionCorrection" not in no_d3["py"].read_text()


# ── submit(): the injectable hook, called once per job ───────────────────────
def test_submit_calls_hook_once(tmp_path):
    calls = []
    target = tmp_path / "jobs" / "run_catbench_MACE-MPA-0.sh"
    target.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\ntrue\n")

    def stub_hook(t):
        calls.append(Path(t))
        return 0

    rc = catbench_jobgen.submit(target, hook=stub_hook)
    assert rc == 0
    assert calls == [target]


def test_default_submit_hook_dispatches_by_filename_prefix(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, *a, **k):
        seen["cmd"] = cmd

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(catbench_jobgen.subprocess, "run", fake_run)

    local = tmp_path / "run_catbench_MACE-MPA-0.sh"
    local.write_text("#!/bin/sh\n")
    catbench_jobgen.submit(local)
    assert seen["cmd"] == ["sh", str(local)]

    slurm = tmp_path / "run_slurm_MACE-MPA-0.sh"
    slurm.write_text("#!/bin/sh\n")
    catbench_jobgen.submit(slurm)
    assert seen["cmd"] == ["sbatch", str(slurm)]


# ── quickstart CLI wiring: --submit calls the hook once per job, never under
#    --emit-only ─────────────────────────────────────────────────────────────
@pytest.fixture()
def quickstart_module():
    run_examples = REPO_ROOT / "run_examples"
    sys.path.insert(0, str(run_examples))
    import importlib

    mod = importlib.import_module("catbench_quickstart")
    return importlib.reload(mod)


def test_run_one_model_emit_only_never_calls_submit_hook(tmp_path, quickstart_module):
    calls = []
    spec = _spec("MACE", "MACE-MPA-0")
    rc = quickstart_module._run_one_model(
        "MACE", spec, "Example", 1, False,
        workdir=tmp_path, emit_only=True, submit=True,
        submit_hook=lambda t: calls.append(t) or 0,
    )
    assert rc == 0
    assert calls == []  # never fired under --emit-only, even with submit=True
    assert (tmp_path / "jobs" / "catbench_MACE-MPA-0.py").is_file()


def test_run_one_model_submit_calls_hook_once(tmp_path, quickstart_module):
    calls = []
    spec = _spec("MACE", "MACE-MPA-0")
    rc = quickstart_module._run_one_model(
        "MACE", spec, "Example", 1, False,
        workdir=tmp_path, emit_only=False, submit=True,
        submit_hook=lambda t: calls.append(t) or 0,
    )
    assert rc == 0
    assert len(calls) == 1
    assert calls[0] == tmp_path / "jobs" / "run_catbench_MACE-MPA-0.sh"


def test_run_one_model_submit_with_slurm_targets_slurm_file(tmp_path, quickstart_module):
    calls = []
    spec = _spec("DPA4")
    mlip_name = spec["version"]
    rc = quickstart_module._run_one_model(
        "DPA4", spec, "Example", 1, False,
        workdir=tmp_path, emit_only=False, use_slurm=True, partition="g1",
        submit=True, submit_hook=lambda t: calls.append(t) or 0,
    )
    assert rc == 0
    assert calls == [tmp_path / "jobs" / f"run_slurm_{mlip_name}.sh"]


# ── shellcheck: emitted .sh is clean (skip if the binary isn't on PATH) ─────
def _find_shellcheck() -> str | None:
    found = shutil.which("shellcheck")
    if found:
        return found
    candidate = Path(sys.exec_prefix) / "bin" / "shellcheck"
    return str(candidate) if candidate.is_file() else None


@pytest.mark.parametrize("model", ["MACE", "NequIP", "DPA4"])
def test_emitted_runner_sh_is_shellcheck_clean(tmp_path, model):
    shellcheck = _find_shellcheck()
    if not shellcheck:
        pytest.skip("shellcheck not available on PATH")
    spec = _spec(model)
    mlip_name = spec["version"]
    artifacts = catbench_jobgen.emit(spec, mlip_name, "Example", 1, False, tmp_path, slurm=True)
    for key in ("sh", "slurm"):
        proc = subprocess.run([shellcheck, str(artifacts[key])], capture_output=True, text=True)
        assert proc.returncode == 0, f"{artifacts[key]}: {proc.stdout}\n{proc.stderr}"
