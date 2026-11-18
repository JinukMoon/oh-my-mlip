"""A child process that writes to fd 1 while the worker builds or calls the
calculator must not reach the JSONL wire (torch's extension JIT does this)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CHILD = r"""
import os, subprocess, sys
sys.path.insert(0, sys.argv[1])
from oh_my_mlip import _worker
with _worker._stdout_to_stderr():
    print("python-level noise")
    subprocess.run(["sh", "-c", "echo '[1/2] nvcc noise'"], check=True)
    os.write(1, b"raw fd noise\n")
_worker._emit(sys.stdout, {"ready": True})
"""


def test_child_process_output_goes_to_stderr_not_the_wire():
    proc = subprocess.run([sys.executable, "-c", CHILD, str(REPO)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == '{"ready": true}\n'
    for noise in ("python-level noise", "[1/2] nvcc noise", "raw fd noise"):
        assert noise in proc.stderr
