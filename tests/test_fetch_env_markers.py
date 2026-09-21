"""fetch_env (MCP install_model) and the envs install.sh builds.

install.sh marks a finished local build with .omm_ready; fetch_env only knew its
own unpack marker, so it unpacked a distribution tarball over a working local
build. Its gated-model notice also went to stdout, which is the MCP server's
JSON-RPC channel.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oh_my_mlip import fetch  # noqa: E402


def _env(tmp_path, monkeypatch, marker: str | None):
    home = tmp_path / "hub"
    prefix = home / "envs" / "alpha"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").write_text("#!/bin/sh\n")
    if marker:
        (prefix / marker).write_text("ok\n")
    monkeypatch.setattr(fetch.registry, "home", lambda: str(home))
    monkeypatch.setattr(fetch.registry, "resolve",
                        lambda m, version=None, **k: {"model": m, "version": version, "env": "alpha",
                                                     "gated": False})
    return prefix


def test_a_finished_install_sh_build_is_used_not_overwritten(tmp_path, monkeypatch):
    prefix = _env(tmp_path, monkeypatch, fetch.BUILD_SENTINEL_NAME)
    got = fetch.fetch_env("Alpha", probe=False, manifest={"alpha": {"hf_repo": "x/y", "revision": "r"}})
    assert got == str(prefix / "bin" / "python")


def test_an_unfinished_env_is_never_unpacked_over(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch, None)
    with pytest.raises(fetch.FetchError, match="not unpacking over it") as info:
        fetch.fetch_env("Alpha", probe=False, manifest={"alpha": {"hf_repo": "x/y", "revision": "r"}})
    assert "install.sh" in str(info.value)


def test_gated_notice_and_fallback_stay_off_stdout(monkeypatch, capsys):
    monkeypatch.setattr(fetch.registry, "resolve",
                        lambda m, version=None, **k: {"model": m, "version": version, "env": "g",
                                                     "gated": True, "license_url": "https://lic"})
    monkeypatch.setattr(fetch, "_resolve_token", lambda: {"source": "env", "env": {}})
    fetch._check_gated("Gated", None)
    fetch._print_fallback("g", "driver too old")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "GATED" in captured.err and "install.sh" in captured.err


def test_a_failed_conda_unpack_is_a_fetch_error(tmp_path):
    install_dir = tmp_path / "alpha"
    (install_dir / "bin").mkdir(parents=True)
    (install_dir / "bin" / "conda-unpack").write_text("import sys\nsys.stderr.write('bad prefix\\n')\nsys.exit(3)\n")
    py = install_dir / "bin" / "python"
    py.symlink_to(sys.executable)
    with pytest.raises(fetch.FetchError, match="conda-unpack failed") as info:
        fetch._conda_unpack(install_dir)
    assert "bad prefix" in str(info.value) and "install.sh" in str(info.value)


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


def test_fetch_env_refuses_to_start_without_room_for_the_unpacked_env(tmp_path, monkeypatch):
    home = tmp_path / "hub"
    home.mkdir()
    monkeypatch.setattr(fetch.registry, "home", lambda: str(home))
    monkeypatch.setattr(fetch.registry, "resolve",
                        lambda m, version=None, **k: {"model": m, "version": version, "env": "alpha", "gated": False})
    manifest = {"alpha": {"hf_repo": "x/y", "revision": "r", "unpack_size_bytes": 10**18}}
    with pytest.raises(fetch.FetchError, match="needs .* GB but"):
        fetch.fetch_env("Alpha", probe=False, manifest=manifest)
    # the upload placeholder skips the check
    assert fetch._check_free_space("alpha", home / "envs" / "alpha", "TODO-on-upload") is None
