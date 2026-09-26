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


def _dataset(n: int) -> dict:
    data = {f"rxn{i:02d}": {"raw": {"star": {"stoi": -1, "ref": f"s{i}"}, "gas": {"stoi": -1, "ref": "g"}},
                            "ref_ads_eng": float(i)} for i in range(n)}
    data["_structures"] = {**{f"s{i}": f"slab{i}" for i in range(n)}, "g": "gas"}
    return data


def test_max_reactions_writes_a_deterministic_subset_with_only_its_structures(monkeypatch, tmp_path, capsys):
    qs, work = _prepared(monkeypatch, tmp_path)
    full = _dataset(12)
    (work / "raw_data" / "demo_adsorption.json").write_text(json.dumps(full))
    assert qs.main(["demo", "--only", "MACE", "--emit-only", "--max-reactions", "3"]) == 0
    assert "first 3 of 12 reactions of demo" in capsys.readouterr().out
    sub = json.loads((work / "raw_data" / "demo_first3_adsorption.json").read_text())
    assert sorted(k for k in sub if not k.startswith("_")) == ["rxn00", "rxn01", "rxn02"]
    assert sub["_structures"] == {"s0": "slab0", "s1": "slab1", "s2": "slab2", "g": "gas"}
    record = json.loads((work / "result" / "omm_dataset.json").read_text())
    assert (record["benchmark"], record["reactions"], record["of"], record["subset"]) == ("demo_first3", 3, 12, True)
    meta = json.loads((work / "jobs" / "catbench_MACE-MPA-0.meta.json").read_text()) \
        if (work / "jobs" / "catbench_MACE-MPA-0.meta.json").exists() else None
    job = (work / "jobs" / "catbench_MACE-MPA-0.py").read_text()
    assert "demo_first3" in job and (meta is None or meta["benchmark"] == "demo_first3")


def test_max_reactions_at_or_above_the_total_runs_the_full_set(monkeypatch, tmp_path, capsys):
    qs, work = _prepared(monkeypatch, tmp_path)
    (work / "raw_data" / "demo_adsorption.json").write_text(json.dumps(_dataset(4)))
    assert qs.main(["demo", "--only", "MACE", "--emit-only", "--max-reactions", "9"]) == 0
    assert "covers all 4 reactions" in capsys.readouterr().out
    assert not (work / "raw_data" / "demo_first9_adsorption.json").exists()
    record = json.loads((work / "result" / "omm_dataset.json").read_text())
    assert (record["benchmark"], record["reactions"], record["of"], record["subset"]) == ("demo", 4, 4, False)


def test_a_subset_never_joins_results_that_carry_no_record(monkeypatch, tmp_path, capsys):
    qs, work = _prepared(monkeypatch, tmp_path)
    (work / "raw_data" / "demo_adsorption.json").write_text(json.dumps(_dataset(12)))
    (work / "result" / "MACE-MPA-0").mkdir(parents=True)          # an earlier full run, before records existed
    with pytest.raises(SystemExit) as info:
        qs.main(["demo", "--only", "MACE", "--emit-only", "--max-reactions", "3"])
    assert info.value.code == 2 and "no dataset record" in capsys.readouterr().err


def test_a_result_folder_never_mixes_two_datasets(monkeypatch, tmp_path, capsys):
    qs, work = _prepared(monkeypatch, tmp_path)
    (work / "raw_data" / "demo_adsorption.json").write_text(json.dumps(_dataset(12)))
    assert qs.main(["demo", "--only", "MACE", "--emit-only", "--max-reactions", "3"]) == 0
    for argv in (["demo", "--only", "MACE", "--emit-only", "--max-reactions", "5"],
                 ["demo", "--only", "MACE", "--emit-only"]):
        with pytest.raises(SystemExit) as info:
            qs.main(argv)
        assert info.value.code == 2
        assert "result/ already holds a run of 'demo_first3'" in capsys.readouterr().err
    assert qs.main(["demo", "--only", "MACE", "--emit-only", "--max-reactions", "3"]) == 0   # same subset reruns


def test_max_reactions_below_one_is_refused(monkeypatch, tmp_path, capsys):
    qs, work = _prepared(monkeypatch, tmp_path)
    assert qs.main(["demo", "--only", "MACE", "--emit-only", "--max-reactions", "0"]) == 2
