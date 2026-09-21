"""Failures a newcomer meets in week one must name the thing that went wrong.

Two of them dropped the only useful detail: a relative --structure resolved
against the clone (the child runs with cwd=home) and came back as a repr with
no filename, and a weight download on a network-less compute node escaped as a
bare URLError naming neither the URL nor the target.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from urllib.error import URLError

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oh_my_mlip import _worker, fetch  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "setup_verify", REPO_ROOT / "scripts" / "setup_verify.py"
)
oracle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oracle)


def test_error_text_keeps_the_filename():
    exc = FileNotFoundError(2, "No such file or directory", "/data/POSCAR")
    assert "/data/POSCAR" in _worker.error_text(exc)
    assert _worker.error_text(exc).startswith("FileNotFoundError:")
    # repr -- what this used to send -- drops the path entirely
    assert "/data/POSCAR" not in repr(exc)


def test_a_missing_structure_is_reported_with_the_resolved_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(oracle, "stream_process", lambda *a, **k: pytest.fail("must not spawn"))
    verdict = oracle.verify_one("MACE", None, "POSCAR", REPO_ROOT, quiet=True, no_local_record=True)
    assert verdict["pass"] is False
    assert str(tmp_path / "POSCAR") in verdict["reason"]      # the caller's cwd, not the clone


def test_a_download_failure_names_the_url_target_and_the_prefetch_command(tmp_path, monkeypatch):
    spec = {
        "model": "Nequix", "version": "Nequix-MP-1", "weights_fetch": "url",
        "weights_source": "https://example.invalid/nequix-mp-1.nqx", "weights_source_url": None,
    }
    target = tmp_path / "models" / "nequix" / "nequix-mp-1.nqx"
    monkeypatch.setattr(fetch, "_download_to_temp",
                        lambda url, directory: (_ for _ in ()).throw(URLError("Network is unreachable")))
    with pytest.raises(fetch.FetchError) as excinfo:
        fetch._materialize_url_weights(spec, [target])
    message = str(excinfo.value)
    assert "https://example.invalid/nequix-mp-1.nqx" in message
    assert str(target) in message
    assert "setup_verify.py Nequix-MP-1" in message
    assert "Network is unreachable" in message
