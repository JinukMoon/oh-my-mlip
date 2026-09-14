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


def test_argparse_accepts_the_recipe_flags():
    a = cr._parse_args(["--result", "r", "--out", "o", "--expect-models", "A,B", "--catbench-version", "1.1.4", "--python", "/p"])
    assert a.expect_models == "A,B" and a.catbench_version == "1.1.4" and a.python == "/p"
