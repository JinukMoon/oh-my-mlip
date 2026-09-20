"""Weight readiness must check the registry's recorded size, not just existence.

Size > 0 was the whole check, so a 1-byte leftover from an interrupted download
counted as ready and shadowed the real weight -- resolve() then handed the user
a path to a broken file. models.json records weights_size/weights_sha256 for the
variants that have them (AGENTS.md ground rule 5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oh_my_mlip import fetch  # noqa: E402


def test_truncated_weight_is_not_ready(tmp_path: Path):
    weight = tmp_path / "nequix-mp-1.nqx"
    weight.write_bytes(b"x")                       # the interrupted-download case
    assert fetch._target_ready(weight) is True     # unchanged when nothing is recorded
    assert fetch._target_ready(weight, 2842366) is False


def test_exact_size_is_ready_and_empty_never_is(tmp_path: Path):
    weight = tmp_path / "w.pt"
    weight.write_bytes(b"xyz")
    assert fetch._target_ready(weight, 3) is True
    weight.write_bytes(b"")
    assert fetch._target_ready(weight, 0) is False
    assert fetch._target_ready(weight) is False


def test_directory_target_is_unaffected_by_size(tmp_path: Path):
    # extensionless targets are SavedModel-style directories; size does not apply
    target = tmp_path / "GRACE-2L-OAM"
    target.mkdir()
    assert fetch._target_ready(target, 12345) is False   # empty dir
    (target / "saved_model.pb").write_bytes(b"x")
    assert fetch._target_ready(target, 12345) is True


def test_sha256_mismatch_raises(tmp_path: Path):
    weight = tmp_path / "w.pt"
    weight.write_bytes(b"hello")
    with pytest.raises(fetch.FetchError, match="sha256 mismatch"):
        fetch._verify_sha256(str(weight), "0" * 64)
    # the real digest passes, and an unrecorded hash is a no-op
    import hashlib
    fetch._verify_sha256(str(weight), hashlib.sha256(b"hello").hexdigest())
    fetch._verify_sha256(str(weight), None)
