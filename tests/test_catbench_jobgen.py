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


# ── CLI: --calc-file witness, --catbench-version guard, --submit ordering ─────
# (GPU-free; the witness really runs, on ASE's EMT for a 2-atom Cu cell.)
import json  # noqa: E402
import os  # noqa: E402

_CALC_FILE = "from ase.calculators.emt import EMT\n\ndef make_calculator():\n    return EMT()\n"


def _work(tmp_path: Path, tag: str = "demo") -> Path:
    work = tmp_path / "work"
    (work / "raw_data").mkdir(parents=True)
    (work / "raw_data" / f"{tag}_adsorption.json").write_text("{}")
    return work


def test_cli_refuses_missing_dataset_bad_tag_and_bad_calc_num(tmp_path: Path, capsys):
    calc = tmp_path / "calc.py"; calc.write_text(_CALC_FILE)
    assert catbench_jobgen.main(["--tag", "nodata", "--workdir", str(tmp_path), "--calc-file", str(calc), "--python", sys.executable]) == 2
    assert "not found" in capsys.readouterr().err
    work = _work(tmp_path)
    for tag in ("../x", "a/b", "-x"):
        assert catbench_jobgen.main([f"--tag={tag}", "--workdir", str(work), "--calc-file", str(calc), "--python", sys.executable]) == 2
    assert catbench_jobgen.main(["--tag", "demo", "--workdir", str(work), "--calc-file", str(calc), "--python", sys.executable,
                                 "--calc-num", "0"]) == 2
    assert catbench_jobgen.main(["--tag", "demo", "--workdir", str(work), "--calc-file", str(calc), "--python", sys.executable,
                                 "--name", "a/b"]) == 2
    assert not (work / "jobs").exists()


def test_spec_from_calc_file_refuses_bad_files(tmp_path: Path):
    bad = tmp_path / "bad.py"; bad.write_text("def not_the_hook():\n    pass\n")
    with pytest.raises(ValueError, match="make_calculator"):
        catbench_jobgen.spec_from_calc_file(bad, sys.executable, "bad")
    with pytest.raises(ValueError, match="not found"):
        catbench_jobgen.spec_from_calc_file(tmp_path / "missing.py", sys.executable, "x")
    good = tmp_path / "calc.py"; good.write_text(_CALC_FILE)
    with pytest.raises(ValueError, match="interpreter"):
        catbench_jobgen.spec_from_calc_file(good, str(tmp_path / "nopython"), "x")
    spec = catbench_jobgen.spec_from_calc_file(good, sys.executable, "mycalc")
    assert spec["source"] == "calc-file" and len(spec["calc_file_sha256"]) == 64 and spec["env_run"] == {}
    assert spec["inference"] == ["calc = omm_user_calc.make_calculator()"]
    assert any(str(good.resolve()) in ln for ln in spec["imports"])


def test_cli_calc_file_emits_witness_and_guarded_job(tmp_path: Path, capsys):
    work = _work(tmp_path)
    calc = tmp_path / "calc.py"; calc.write_text(_CALC_FILE)
    rc = catbench_jobgen.main(["--tag", "demo", "--workdir", str(work), "--calc-file", str(calc), "--python", sys.executable,
                               "--name", "mycalc", "--catbench-version", "1.1.4", "--slurm"])
    assert rc == 0
    out = capsys.readouterr().out
    jobs = work / "jobs"
    assert "witness :" in out and str(jobs / "run_witness_mycalc.sh") in out
    for f in ("witness_mycalc.py", "run_witness_mycalc.sh", "catbench_mycalc.py", "run_catbench_mycalc.sh",
              "run_slurm_mycalc.sh", "catbench_mycalc.meta.json"):
        assert (jobs / f).is_file(), f
    job = (jobs / "catbench_mycalc.py").read_text()
    ast.parse(job); ast.parse((jobs / "witness_mycalc.py").read_text())
    assert job.index("_cb.__version__ != '1.1.4'") < job.index("from catbench.adsorption import AdsorptionCalculation")
    assert "_sys.exit(3)" in job and 'mlip_name": \'mycalc\'' in job.replace('"mlip_name": ', 'mlip_name": ')
    meta = json.loads((jobs / "catbench_mycalc.meta.json").read_text())
    assert meta["catbench_version"] == "1.1.4" and meta["source"] == "calc-file" and meta["calc_file"] == str(calc.resolve())
    py = str(Path(sys.executable).resolve())                            # spec_from_calc_file resolves the interpreter
    assert meta["python"] == py and len(meta["calc_file_sha256"]) == 64
    # runner unit: same body contract as the registry path
    sh = (jobs / "run_witness_mycalc.sh").read_text().splitlines()
    assert sh[0] == "#!/bin/sh" and f'cd "{work}"' in sh and sh[-1] == f'exec "{py}" "{jobs / "witness_mycalc.py"}"'


def test_version_guard_stops_the_job_under_a_wrong_catbench(tmp_path: Path):
    """Run the emitted job under a stub `catbench` whose __version__ differs:
    it must exit 3 before importing AdsorptionCalculation (never provided)."""
    work = _work(tmp_path)
    calc = tmp_path / "calc.py"; calc.write_text(_CALC_FILE)
    assert catbench_jobgen.main(["--tag", "demo", "--workdir", str(work), "--calc-file", str(calc), "--python", sys.executable,
                                 "--name", "c", "--catbench-version", "1.1.4"]) == 0
    stub = tmp_path / "stub" / "catbench"; stub.mkdir(parents=True)
    (stub / "__init__.py").write_text('__version__ = "1.1.3"\n')
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "stub"))
    proc = subprocess.run(["sh", str(work / "jobs" / "run_catbench_c.sh")], env=env, capture_output=True, text=True)
    assert proc.returncode == 3 and "approved catbench 1.1.4 but this env has 1.1.3" in proc.stderr


def test_witness_runs_and_submit_requires_it(tmp_path: Path, monkeypatch, capsys):
    pytest.importorskip("ase")
    work = _work(tmp_path)
    calc = tmp_path / "calc.py"; calc.write_text(_CALC_FILE)
    dispatched = []
    monkeypatch.setattr(catbench_jobgen, "_default_submit_hook", lambda target: dispatched.append(target) or 0)
    argv = ["--tag", "demo", "--workdir", str(work), "--calc-file", str(calc), "--python", sys.executable, "--name", "c", "--submit"]
    assert catbench_jobgen.main(argv) == 3                              # no witness record yet
    assert "first; the job is not dispatched" in capsys.readouterr().err and dispatched == []
    proc = subprocess.run(["sh", str(work / "jobs" / "run_witness_c.sh")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    rec = json.loads((work / "jobs" / "witness_c.json").read_text())
    assert rec["ok"] and rec["n_atoms"] == 2 and rec["forces_shape"] == [2, 3] and rec["forces_finite"]
    assert rec["calc_file_sha256"] == catbench_jobgen.spec_from_calc_file(calc, sys.executable, "c")["calc_file_sha256"]
    assert catbench_jobgen.main(argv) == 0                              # witness ok => dispatched
    assert dispatched == [work / "jobs" / "run_catbench_c.sh"]
    calc.write_text(_CALC_FILE + "# edited after the witness\n")        # sha256 no longer matches the witness
    assert catbench_jobgen.main(argv) == 3                              # rerun rule: witness_c.py would change
    assert "differ from what these inputs would generate" in capsys.readouterr().err and len(dispatched) == 1
    assert catbench_jobgen.main(argv + ["--regenerate"]) == 3           # regenerated witness .py, stale witness .json
    assert "changed since the witness" in capsys.readouterr().err and len(dispatched) == 1


def test_rerun_keeps_identical_files_and_refuses_silent_overwrite(tmp_path: Path, capsys):
    """A rerun executes the generated files as they are: identical bytes are
    kept (reported `reused`), a changed rendering is refused with every stale
    path named and NOTHING written, and only --regenerate replaces it."""
    work = _work(tmp_path)
    spec = _spec("MACE", "MACE-MPA-0")
    first = catbench_jobgen.emit(spec, "MACE-MPA-0", "demo", 3, False, work, catbench_version="1.1.4")
    assert first["reused"] == []
    before = {k: first[k].read_text() for k in ("py", "sh", "meta")}
    again = catbench_jobgen.emit(spec, "MACE-MPA-0", "demo", 3, False, work, catbench_version="1.1.4")
    assert sorted(again["reused"]) == ["meta", "py", "sh"]
    assert {k: again[k].read_text() for k in before} == before
    with pytest.raises(FileExistsError) as ei:
        catbench_jobgen.emit(spec, "MACE-MPA-0", "demo", 5, False, work, catbench_version="1.1.4")
    msg = str(ei.value)
    assert str(first["py"]) in msg and str(first["meta"]) in msg and str(first["sh"]) not in msg and "--regenerate" in msg
    assert {k: first[k].read_text() for k in before} == before            # nothing written on refusal
    third = catbench_jobgen.emit(spec, "MACE-MPA-0", "demo", 5, False, work, catbench_version="1.1.4", overwrite=True)
    assert third["reused"] == ["sh"] and "calc_num = 5" in first["py"].read_text()
    assert json.loads(first["meta"].read_text())["calc_num"] == 5
    # CLI form of the same rule
    argv = ["--tag", "demo", "--workdir", str(work), "--calc-file", str(_calc(tmp_path)), "--python", sys.executable, "--name", "MACE-MPA-0"]
    assert catbench_jobgen.main(argv) == 3
    assert "[stop] existing job file(s) differ" in capsys.readouterr().err
    assert catbench_jobgen.main(argv + ["--regenerate"]) == 0
    out = capsys.readouterr().out
    assert "(written)" in out and "witness :" in out


def _calc(tmp_path: Path) -> Path:
    calc = tmp_path / "calc.py"
    calc.write_text(_CALC_FILE)
    return calc


def test_witness_fails_on_a_calculator_without_finite_forces(tmp_path: Path):
    pytest.importorskip("ase")
    work = _work(tmp_path)
    calc = tmp_path / "calc.py"
    calc.write_text("import numpy as np\nfrom ase.calculators.calculator import Calculator, all_changes\n"
                    "class Nan(Calculator):\n    implemented_properties = ['energy', 'forces']\n"
                    "    def calculate(self, atoms=None, properties=None, system_changes=all_changes):\n"
                    "        super().calculate(atoms, properties, system_changes)\n"
                    "        self.results = {'energy': float('nan'), 'forces': np.zeros((len(atoms), 3))}\n"
                    "def make_calculator():\n    return Nan()\n")
    assert catbench_jobgen.main(["--tag", "demo", "--workdir", str(work), "--calc-file", str(calc), "--python", sys.executable, "--name", "n"]) == 0
    proc = subprocess.run(["sh", str(work / "jobs" / "run_witness_n.sh")], capture_output=True, text=True)
    assert proc.returncode == 3
    assert json.loads((work / "jobs" / "witness_n.json").read_text())["ok"] is False
    assert catbench_jobgen.witness_passed(work / "jobs" / "witness_n.json", {"calc_file_sha256": "x"})[0] is False


def test_cli_registry_model_path_uses_resolve(tmp_path: Path, monkeypatch):
    work = _work(tmp_path)
    spec = {"python": "/env/bin/python", "imports": ["from x import Calc"], "inference": ["calc = Calc()"],
            "env_run": {"K": "V"}, "version": "Fake-1"}
    import oh_my_mlip
    monkeypatch.setattr(oh_my_mlip, "resolve", lambda model, version=None: spec)
    assert catbench_jobgen.main(["--tag", "demo", "--workdir", str(work), "--model", "Fake", "--catbench-version", "1.1.4", "--d3"]) == 0
    job = (work / "jobs" / "catbench_Fake-1_D3.py").read_text()
    ast.parse(job)
    assert "DispersionCorrection" in job and "'1.1.4'" in job
    assert 'export K="V"' in (work / "jobs" / "run_catbench_Fake-1_D3.sh").read_text()
    assert json.loads((work / "jobs" / "catbench_Fake-1_D3.meta.json").read_text())["source"] == "registry"
