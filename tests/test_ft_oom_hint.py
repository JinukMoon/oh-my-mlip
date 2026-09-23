"""A fine-tune that dies of GPU memory says which knob to turn.

Official batch sizes are set for data-centre GPUs (ORB's is 100); on a 16 GB
card the run ended in a bare torch.OutOfMemoryError traceback. ft_sweep also had
no official way to pass a smaller batch size.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ft_run  # noqa: E402


def _script(tmp_path, body):
    sh = tmp_path / "finetune_T.sh"
    sh.write_text("#!/bin/bash\n" + body)
    return sh


def test_an_oom_failure_names_the_batch_size(tmp_path, capfd, monkeypatch):
    monkeypatch.setattr(ft_run, "_gpu_total", lambda: " on a 16 GB GPU")
    sh = _script(tmp_path, "echo 'step 1' >&2\n"
                           "echo 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 102.00 MiB' >&2\n"
                           "exit 1\n")
    assert ft_run.run_training(sh, batch_size=100) == 1
    err = capfd.readouterr().err
    assert "step 1" in err and "CUDA out of memory" in err  # the trainer's own output still shows
    assert "ran out of GPU memory with batch size 100 on a 16 GB GPU" in err
    assert "--batch-size" in err


def test_other_failures_and_successes_get_no_hint(tmp_path, capfd):
    assert ft_run.run_training(_script(tmp_path, "echo 'KeyError: x' >&2\nexit 3\n"), batch_size=4) == 3
    assert ft_run.run_training(_script(tmp_path, "echo ok >&2\n"), batch_size=4) == 0
    assert "ran out of GPU memory" not in capfd.readouterr().err


def test_ft_sweep_takes_a_batch_size():
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "ft_sweep.py"), "--help"],
                          capture_output=True, text=True)
    assert "--batch-size" in proc.stdout
