"""Tests for scripts/_setup_common.py — the shared deterministic core.

Pins the three facts the plan makes load-bearing:
  * descendant-PID attribution works on a /proc parent chain (and direct-PID
    matching alone would NOT find the GPU process — oh_my_mlip.provider spawns
    the compute worker as a grandchild of whatever the caller launches, see
    provider.py Worker spawn; this is why the oracle must walk parents);
  * predict_driver_skew implements the SAME comparison as install.sh's
    warn_driver_skew, verified by RUNNING the actual bash function against a
    stubbed nvidia-smi (agreement test), including the early-return fallbacks
    (no +cuNNN pin / no nvidia-smi / unparseable host CUDA => no skew);
  * verify_output_ok parses the single_point witness lines.

GPU-free, stdlib-only.
"""
import importlib.util
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "_setup_common", REPO_ROOT / "scripts" / "_setup_common.py"
)
common = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(common)


# ---------------------------------------------------------------------------
# Fake /proc fixtures
# ---------------------------------------------------------------------------

def _fake_proc(tmp_path: Path, tree: dict[int, tuple[int, str]]) -> Path:
    """tree: pid -> (ppid, comm). Builds /proc/<pid>/{stat,comm,cmdline}."""
    proc = tmp_path / "proc"
    for pid, (ppid, comm) in tree.items():
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} 0 0")
        (d / "comm").write_text(comm)
        (d / "cmdline").write_text(f"/usr/bin/{comm}\x00run\x00")
    return proc


def test_descendant_attribution_walks_grandchild_chain(tmp_path):
    # launcher(100) -> single_point(200) -> env worker(300): the GPU PID (300)
    # is a GRANDCHILD of what the oracle launches (100). Direct equality would
    # miss it; the parent walk must attribute it.
    proc = _fake_proc(
        tmp_path, {100: (1, "python3"), 200: (100, "python3"), 300: (200, "python3")}
    )
    assert common.is_descendant_or_self(300, 100, proc_root=proc)
    assert 300 != 100  # direct-PID matching alone would fail here
    assert not common.is_descendant_or_self(300, 999, proc_root=proc)


def test_descendant_attribution_rejects_unrelated_and_cycles(tmp_path):
    proc = _fake_proc(tmp_path, {50: (1, "python3"), 60: (70, "python3"), 70: (60, "python3")})
    assert not common.is_descendant_or_self(50, 100, proc_root=proc)
    # ppid cycle must terminate, not loop forever
    assert not common.is_descendant_or_self(60, 100, proc_root=proc)


def test_is_python_pid_reads_comm_and_cmdline(tmp_path):
    proc = _fake_proc(tmp_path, {10: (1, "python3"), 11: (1, "nvidia-smi")})
    assert common.is_python_pid(10, proc_root=proc)
    assert not common.is_python_pid(11, proc_root=proc)


# ---------------------------------------------------------------------------
# Driver-skew predicate
# ---------------------------------------------------------------------------

def _home_with_recipe(tmp_path: Path, recipe_line: str | None) -> Path:
    (tmp_path / "envs").mkdir()
    if recipe_line is not None:
        (tmp_path / "envs" / "tace.yml").write_text(
            f"dependencies:\n  - pip:\n    - {recipe_line}\n"
        )
    return tmp_path


def test_predicate_skew_and_reason(tmp_path):
    home = _home_with_recipe(tmp_path, "torch==2.12.1+cu130")
    got = common.predict_driver_skew("tace", home, host_cuda=129)
    assert got["skew"] is True
    assert got["env_cuda"] == 130 and got["host_cuda"] == 129
    assert got["reason"] == "env needs CUDA 13.0, host driver exposes CUDA 12.9"


def test_predicate_no_skew_when_host_new_enough(tmp_path):
    home = _home_with_recipe(tmp_path, "torch==2.7.1+cu126")
    got = common.predict_driver_skew("tace", home, host_cuda=129)
    assert got == {"skew": False, "env_cuda": 126, "host_cuda": 129, "reason": ""}


def test_predicate_early_returns_mirror_install_sh(tmp_path):
    # No +cuNNN pin (CPU-only recipe) => no skew, GPU attempted.
    home = _home_with_recipe(tmp_path, "torch==2.7.1")
    assert common.predict_driver_skew("tace", home, host_cuda=100)["skew"] is False
    # Missing recipe file => no skew (predicate blind, honest GPU attempt).
    assert common.predict_driver_skew("missing", home, host_cuda=100)["skew"] is False


def test_parse_host_cuda():
    text = "| NVIDIA-SMI 576.02  Driver Version: 576.02  CUDA Version: 12.9 |"
    assert common.parse_host_cuda(text) == 129
    assert common.parse_host_cuda("garbage") is None


# ---------------------------------------------------------------------------
# Agreement with install.sh's warn_driver_skew (the single-origin invariant)
# ---------------------------------------------------------------------------

def _run_warn_driver_skew(tmp_path: Path, recipe: Path, host_mm: str | None) -> str:
    """Execute the REAL bash function from install.sh with nvidia-smi stubbed."""
    fn = subprocess.run(
        ["sed", "-n", "/^warn_driver_skew()/,/^}/p", str(REPO_ROOT / "install.sh")],
        stdout=subprocess.PIPE, text=True, check=True,
    ).stdout
    assert fn.startswith("warn_driver_skew()"), "install.sh function extraction failed"
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    # Always stub nvidia-smi (the real one exists on this WSL host, so an
    # absent-stub PATH would leak the real driver into the test). host_mm=None
    # emulates an unusable nvidia-smi: empty output -> unparseable host CUDA
    # -> warn_driver_skew's install.sh:206 early return.
    smi = bindir / "nvidia-smi"
    body = f"echo 'CUDA Version: {host_mm}'" if host_mm is not None else ":"
    smi.write_text(f"#!/bin/sh\n{body}\n")
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, PATH=f"{bindir}:/usr/bin:/bin")
    out = subprocess.run(
        ["bash", "-c", fn + f'\nwarn_driver_skew tace "{recipe}"'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
    )
    return out.stdout


@pytest.mark.parametrize(
    "pin,host_mm,expect_skew",
    [
        ("torch==2.12.1+cu130", "12.9", True),
        ("torch==2.7.1+cu126", "12.9", False),
        ("torch==2.12.1+cu130", "13.0", False),
        ("torch==2.7.1", "12.9", False),        # no pinned wheel early-return
        ("torch==2.12.1+cu130", None, False),    # unparseable host CUDA early-return
    ],
)
def test_predicate_agrees_with_install_sh(tmp_path, monkeypatch, pin, host_mm, expect_skew):
    home = _home_with_recipe(tmp_path, pin)
    recipe = home / "envs" / "tace.yml"
    # host_mm=None emulates an unusable nvidia-smi on BOTH sides: the bash stub
    # prints nothing, and the Python probe must not fall through to the REAL
    # nvidia-smi on this host.
    monkeypatch.setattr(common, "_probe_host_cuda", lambda: None)
    host_cuda = common.parse_host_cuda(f"CUDA Version: {host_mm}") if host_mm else None
    py = common.predict_driver_skew("tace", home, host_cuda=host_cuda)
    bash_out = _run_warn_driver_skew(tmp_path, recipe, host_mm)
    bash_skew = "WARNING driver skew" in bash_out
    assert py["skew"] == bash_skew == expect_skew


# ---------------------------------------------------------------------------
# Witness-output parsing
# ---------------------------------------------------------------------------

def test_stream_process_quiet_keeps_stdout_clean(tmp_path, capsys):
    rc, _elapsed, stdout, _stderr, _gpu = common.stream_process(
        ["/bin/sh", "-c", "echo streamed-witness-line"],
        env=dict(os.environ),
        log_path=tmp_path / "log.txt",
        stderr_path=tmp_path / "err.txt",
        collect=True,
        quiet=True,
    )
    assert rc == 0
    assert "streamed-witness-line" in stdout          # still collected
    assert "streamed-witness-line" in (tmp_path / "log.txt").read_text()  # still logged
    assert capsys.readouterr().out == ""              # but OUR stdout stays clean


def test_verify_output_ok_parses_witness_lines():
    stdout = "energy (eV) : -16.386652\nmax|force| : 0.031245\n"
    assert common.verify_output_ok(stdout, "") == (True, True)
    assert common.verify_output_ok("no numbers here", "") == (False, False)
