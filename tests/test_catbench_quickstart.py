"""GPU-free tests for run_examples/catbench_quickstart.py's job-file contract.

- `--catbench-version` is passed through to every emitted job (guard line in
  the .py, value in jobs/catbench_<MLIP>.meta.json).
- A rerun executes the job files already on disk: unchanged bytes are reused,
  a changed rendering stops the run (exit 3, nothing overwritten), and only
  `--regenerate` replaces them.
- A TAG missing from raw_data/ is fetched under an installed env (never with
  --no-fetch); --all-versions skips "catbench": false versions; --arch reaches
  resolve().

The example is imported by path (it is a script); resolve()/list_models and
the env-readiness check are monkeypatched so no conda env, subprocess or
dataset is touched (`--emit-only` never executes anything).
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "run_examples" / "catbench_quickstart.py"



def _load_quickstart():
    spec = importlib.util.spec_from_file_location("catbench_quickstart", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prepared(monkeypatch, tmp_path: Path, *, with_data: bool = True):
    qs = _load_quickstart()
    work = tmp_path / "work"
    work.mkdir(parents=True)
    if with_data:
        (work / "raw_data").mkdir()
        (work / "raw_data" / "demo_adsorption.json").write_text("{}")
    monkeypatch.chdir(work)
    # A real registry spec (verbatim resolve() codegen) whose interpreter does
    # not exist: rendering is real, execution is impossible by construction.
    spec = dict(qs.resolve("MACE", "MACE-MPA-0"), python="/nonexistent/envs/mace/bin/python")
    monkeypatch.setattr(qs, "list_models", lambda: ["MACE"])
    monkeypatch.setattr(qs, "_resolve_versions_for", lambda model, pins, **kw: [dict(spec)])
    monkeypatch.setattr(qs, "_env_ready", lambda spec: True)
    return qs, work


def test_catbench_version_passes_through_to_job_and_meta(monkeypatch, tmp_path, capsys):
    qs, work = _prepared(monkeypatch, tmp_path)
    assert qs.main(["demo", "--only", "MACE", "--emit-only", "--catbench-version", "1.1.4"]) == 0
    out = capsys.readouterr().out
    jobs = work / "jobs"
    assert "emitted meta:" in out and "Nothing executed" in out
    meta = json.loads((jobs / "catbench_MACE-MPA-0.meta.json").read_text())
    assert meta["catbench_version"] == "1.1.4" and meta["benchmark"] == "demo" and meta["calc_num"] == 3
    body = (jobs / "catbench_MACE-MPA-0.py").read_text()
    assert "1.1.4" in body and "approved catbench" in body
    assert body.index("approved catbench") < body.index("from catbench.adsorption import AdsorptionCalculation")


def test_rerun_reuses_identical_files_and_never_overwrites_silently(monkeypatch, tmp_path, capsys):
    qs, work = _prepared(monkeypatch, tmp_path)
    argv = ["demo", "--only", "MACE", "--emit-only", "--catbench-version", "1.1.4"]
    assert qs.main(argv) == 0
    capsys.readouterr()
    jobs = work / "jobs"
    snapshot = {p.name: (p.read_text(), p.stat().st_mtime_ns) for p in jobs.iterdir()}

    # Identical inputs: every file is reused, none rewritten.
    assert qs.main(argv) == 0
    out = capsys.readouterr().out
    assert "reused py:" in out and "reused sh:" in out and "reused meta:" in out
    assert not any(f"emitted {k}:" in out for k in ("py", "sh", "meta"))
    assert {p.name: (p.read_text(), p.stat().st_mtime_ns) for p in jobs.iterdir()} == snapshot

    # Changed inputs (different pin): stop, name the stale files, write nothing.
    rc = qs.main(["demo", "--only", "MACE", "--emit-only", "--catbench-version", "1.1.5"])
    err = capsys.readouterr().err
    assert rc == 1                                    # the model run counted as failed => non-zero
    assert "[stop] MACE (MACE-MPA-0)" in err and "--regenerate" in err
    assert str(jobs / "catbench_MACE-MPA-0.py") in err and str(jobs / "catbench_MACE-MPA-0.meta.json") in err
    assert str(jobs / "run_catbench_MACE-MPA-0.sh") not in err   # the runner would not change
    assert {p.name: (p.read_text(), p.stat().st_mtime_ns) for p in jobs.iterdir()} == snapshot

    # Explicit --regenerate replaces exactly the changed files.
    assert qs.main(["demo", "--only", "MACE", "--emit-only", "--catbench-version", "1.1.5", "--regenerate"]) == 0
    out = capsys.readouterr().out
    assert "emitted py:" in out and "emitted meta:" in out and "reused sh:" in out
    assert json.loads((jobs / "catbench_MACE-MPA-0.meta.json").read_text())["catbench_version"] == "1.1.5"
    assert (jobs / "run_catbench_MACE-MPA-0.sh").read_text() == snapshot["run_catbench_MACE-MPA-0.sh"][0]


def test_stale_job_is_not_executed(monkeypatch, tmp_path):
    """Default mode (execute): the refusal happens BEFORE `sh jobs/run_*.sh`."""
    qs, work = _prepared(monkeypatch, tmp_path)
    assert qs.main(["demo", "--only", "MACE", "--emit-only"]) == 0
    ran = []
    monkeypatch.setattr(qs.subprocess, "run", lambda cmd, *a, **k: ran.append(cmd) or type("R", (), {"returncode": 0})())
    assert qs.main(["demo", "--only", "MACE", "--calc-num", "1"]) == 1
    assert ran == []
    assert qs.main(["demo", "--only", "MACE"]) == 0                 # unchanged => executes the existing .sh
    assert ran == [["sh", str(work / "jobs" / "run_catbench_MACE-MPA-0.sh")]]
    assert os.access(work / "jobs" / "run_catbench_MACE-MPA-0.sh", os.X_OK)


def test_missing_tag_is_fetched_under_an_installed_env_unless_no_fetch(monkeypatch, tmp_path):
    qs, work = _prepared(monkeypatch, tmp_path, with_data=False)
    calls = []

    def fake_fetch(tag, python, workdir):
        calls.append((tag, python, Path(workdir).resolve()))
        (workdir / "raw_data").mkdir(exist_ok=True)
        (workdir / "raw_data" / f"{tag}_adsorption.json").write_text("{}")
        return True

    monkeypatch.setattr(qs, "_fetch_tag", fake_fetch)
    with pytest.raises(SystemExit) as stop:
        qs.main(["demo", "--only", "MACE", "--emit-only", "--no-fetch"])
    assert stop.value.code == 2 and calls == []

    assert qs.main(["demo", "--only", "MACE", "--emit-only"]) == 0
    assert calls == [("demo", qs.resolve("MACE")["python"], work.resolve())]
    assert (work / "jobs" / "catbench_MACE-MPA-0.py").is_file()


def test_all_versions_skips_catbench_false_and_passes_arch(monkeypatch):
    qs = _load_quickstart()
    seen = []
    monkeypatch.setattr(qs, "list_versions", lambda model: ["UMA-s-1p2-OMAT", "UMA-s-1p2-OC25"])
    monkeypatch.setattr(qs, "resolve", lambda model, version=None, *, arch=None: seen.append((version, arch)) or {"version": version})

    specs = qs._resolve_versions_for("UMA", {}, arch="sm86", all_versions=True)
    assert [s["version"] for s in specs] == ["UMA-s-1p2-OMAT"]
    assert seen == [("UMA-s-1p2-OMAT", "sm86")]
    assert qs.catbench_excluded("UMA-s-1p2-OC25") and not qs.catbench_excluded("UMA-s-1p2-OMAT")

    # an explicit pin still runs a "catbench": false version (the user named it)
    assert [s["version"] for s in qs._resolve_versions_for("UMA", {"UMA": "UMA-s-1p2-OC25"})] == ["UMA-s-1p2-OC25"]
