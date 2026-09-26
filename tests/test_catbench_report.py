"""GPU-free tests for the approval guards of scripts/catbench_report.py.

Only the parts that run BEFORE the re-exec / `import catbench` are tested
here (the aggregation itself needs a catbench-bearing env and is exercised
by the host sessions): the exact-selection guard on `result/`, the
version-guard flag, and the rerun unit carrying both guards forward.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import catbench_report as cr  # noqa: E402


def _results(root: Path, *names: str) -> Path:
    for n in names:
        (root / "result" / n).mkdir(parents=True, exist_ok=True)
    return root / "result"


def test_check_expected_models_reports_extra_and_missing(tmp_path: Path):
    res = _results(tmp_path, "A", "B", "baseline")
    g = cr.check_expected_models(res, ["A", "B"])
    assert g == {"ok": False, "present": ["A", "B", "baseline"], "extra": ["baseline"], "missing": []}
    g = cr.check_expected_models(res, ["A", "B", "baseline", "C"])
    assert g["ok"] and g["missing"] == ["C"]
    assert cr.check_expected_models(tmp_path / "nope", ["A"])["present"] == []


def test_main_refuses_extra_or_missing_models_before_any_catbench_import(tmp_path: Path, capsys, monkeypatch):
    res = _results(tmp_path, "A", "B", "baseline")
    monkeypatch.setattr(cr, "_reexec_before_catbench_import", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not re-exec")))
    assert cr.main(["--result", str(res), "--out", str(tmp_path / "report"), "--expect-models", "A,B"]) == 3
    assert "outside the approved selection: ['baseline']" in capsys.readouterr().err
    assert cr.main(["--result", str(res), "--out", str(tmp_path / "report"), "--expect-models", "A,B,baseline,C"]) == 3
    assert "without a result/ entry: ['C']" in capsys.readouterr().err
    assert not (tmp_path / "report").exists()


def test_run_report_sh_carries_the_guards_and_cd_contract(tmp_path: Path):
    out = tmp_path / "report"; out.mkdir()
    sh = cr._write_run_report_sh(out, "/env/bin/python", tmp_path / "result", tmp_path,
                                 expected=["A", "B"], catbench_version="1.1.4")
    text = sh.read_text().splitlines()
    assert text[0] == "#!/bin/sh" and "set -eu" in text and f'cd "{tmp_path}"' in text
    last = text[-1]
    assert last.startswith('exec "/env/bin/python"') and '--expect-models "A,B"' in last and '--catbench-version "1.1.4"' in last
    plain = cr._write_run_report_sh(out, "/env/bin/python", tmp_path / "result", tmp_path).read_text()
    assert "--expect-models" not in plain and "--catbench-version" not in plain


def test_capture_excel_wrapper_forwards_every_upstream_argument():
    """The wrapper must pass catbench's own call through untouched. catbench 1.1.4 calls
    _create_excel_output with the gas-shift-corrected twins as keywords; a wrapper with a
    fixed argument list raises TypeError there (killing the whole report), and one that
    swallowed them would silently drop those workbook sheets."""
    seen = {}

    def orig(self, main_data, anomaly_data, mlips, adsorbates, main_data_shifted=None, anomaly_data_shifted=None):
        seen.update(self=self, main_data=main_data, anomaly_data=anomaly_data, mlips=mlips,
                    adsorbates=adsorbates, shifted=main_data_shifted, anomaly_shifted=anomaly_data_shifted)
        return "delegated"

    captured: dict = {}
    wrapper = cr._capture_excel_wrapper(orig, captured)
    analysis = object()
    out = wrapper(analysis, [{"MLIP": "A"}], ["anom"], ["mlips"], ["CO"],
                  main_data_shifted=[{"MLIP": "A-shifted"}], anomaly_data_shifted=["anom-shifted"])
    assert out == "delegated"
    assert captured["main_data"] == [{"MLIP": "A"}]          # rows the table/plot are built from
    assert seen["self"] is analysis and seen["shifted"] == [{"MLIP": "A-shifted"}]
    assert seen["anomaly_shifted"] == ["anom-shifted"] and seen["adsorbates"] == ["CO"]


def test_argparse_accepts_the_recipe_flags():
    a = cr._parse_args(["--result", "r", "--out", "o", "--expect-models", "A,B", "--catbench-version", "1.1.4", "--python", "/p"])
    assert a.expect_models == "A,B" and a.catbench_version == "1.1.4" and a.python == "/p"


def test_mae_table_names_an_empty_class_and_labels_a_subset(tmp_path: Path):
    import catbench_report as cr
    rows = [{"MLIP_name": "M", "MAE_total": 1.2333, "MAE_normal": float("nan"), "MAE_single": 5.305,
             "ADwT": 64.41, "AMDwT": 57.14, "Num_total": 10}]
    record = {"subset": True, "reactions": 10, "of": 45130, "source_tag": "MamunHighT2019",
              "selection": "the first 10 reaction ids of MamunHighT2019 in sorted order"}
    cr._write_mae_table(tmp_path, rows, record)
    md = (tmp_path / "mae_table.md").read_text()
    assert "| M | 1.2333 | 0 reactions in this class | 5.3050 |" in md and "nan" not in md
    assert "Subset: 10 of 45130 reactions of MamunHighT2019" in md and "not the full benchmark" in md
    assert "1.2333,,5.305" in (tmp_path / "mae_table.csv").read_text()   # empty class: empty cell
    cr._write_mae_table(tmp_path, rows, None)
    assert "Subset" not in (tmp_path / "mae_table.md").read_text()


def test_dataset_record_is_read_from_the_result_folder(tmp_path: Path):
    import json
    import catbench_report as cr
    assert cr._load_dataset_record(tmp_path) is None
    (tmp_path / "omm_dataset.json").write_text(json.dumps({"subset": True, "reactions": 3, "of": 9, "source_tag": "x"}))
    assert cr.subset_note(cr._load_dataset_record(tmp_path)).startswith("Subset: 3 of 9 reactions of x")
