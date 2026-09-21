"""Messages that must stay off stdout, and job-submission transparency.

The MCP server speaks JSON-RPC on stdout, so the hub's own notices (gated-model
license notice) go to stderr. CatBench's --submit shows the #SBATCH header it
sends, and catbench_report stops with a message on a missing interpreter.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oh_my_mlip import fetch  # noqa: E402


def test_gated_notice_stays_off_stdout(monkeypatch, capsys):
    monkeypatch.setattr(fetch.registry, "resolve",
                        lambda m, version=None, **k: {"model": m, "version": version, "env": "g",
                                                     "gated": True, "license_url": "https://lic"})
    monkeypatch.setattr(fetch, "_resolve_token", lambda: {"source": "env", "env": {}})
    fetch._check_gated("Gated", None)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "GATED" in captured.err


def test_catbench_submit_shows_the_header_it_sends(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import catbench_jobgen

    script = tmp_path / "run_slurm_MACE.sh"
    script.write_text("#!/bin/sh\n#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1\necho hi\n")
    calls = []
    monkeypatch.setattr(catbench_jobgen.subprocess, "run",
                        lambda cmd: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    assert catbench_jobgen._default_submit_hook(script) == 0
    assert calls == [["sbatch", str(script)]]
    err = capsys.readouterr().err
    assert "#SBATCH --partition=gpu" in err and "no --time, --mem" in err


def test_catbench_report_stops_on_a_missing_interpreter(tmp_path):
    (tmp_path / "result").mkdir()
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "catbench_report.py"),
                           "--result", str(tmp_path / "result"), "--out", str(tmp_path / "out"),
                           "--python", str(tmp_path / "missing" / "bin" / "python")],
                          capture_output=True, text=True)
    assert proc.returncode == 2
    assert "is not there" in proc.stderr and "Traceback" not in proc.stderr
