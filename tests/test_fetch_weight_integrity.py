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


def test_directory_holding_only_a_partial_download_is_not_ready(tmp_path: Path):
    """Reported from a real run: a slow GRACE fetch timed out partway, left its
    partial file inside the target directory, and every later run called the
    weights ready while TensorFlow failed to find the SavedModel."""
    target = tmp_path / "GRACE-2L-OAM"
    target.mkdir()
    (target / "tmp.tar.gz").write_bytes(b"x" * 1024)
    (target / ".grace.tar.gz.download").write_bytes(b"x" * 1024)
    assert fetch._target_ready(target) is False
    (target / "saved_model.pb").write_bytes(b"x")
    assert fetch._target_ready(target) is True


# ── the recorded size is the DOWNLOAD's, not a derived file's ────────────────
def test_a_derived_target_is_ready_whatever_its_size(tmp_path: Path, monkeypatch):
    """Regression (reported against c89eda2): DeePMD's inference target is
    frozen-omat24.pth, which scripts/prepare_deepmd_weights.py freezes from the
    downloaded dpa-3.1-3m-ft.pth. models.json records the DOWNLOAD's size, so
    checking it against the frozen file rejected a correct weight forever."""
    home = tmp_path / "hub"
    (home / "scripts").mkdir(parents=True)
    (home / "scripts" / "prepare_deepmd_weights.py").write_text("# derives the target\n")
    target = home / "models" / "deepmd" / "frozen-omat24.pth"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x" * 1234)                 # nothing like the download's 47,176,032 bytes
    spec = {"model": "DeePMD", "version": "DPA-3.1-3M-FT", "env": "deepmd",
            "weights_fetch": "by-name", "weights_size": 47176032, "weights_sha256": "0" * 64}
    monkeypatch.setattr(fetch.registry, "home", lambda: str(home))
    monkeypatch.setattr(fetch, "_inference_weight_targets", lambda s: [target])
    assert fetch._derives_its_target(spec) is True
    assert fetch.ensure_weights("DeePMD", version="DPA-3.1-3M-FT", spec=spec) == [str(target)]


def test_only_targets_that_are_the_download_are_size_checked():
    """Registry-wide guard. A variant is size-checked only when its single
    inference target is the downloaded artifact itself. This set was confirmed
    by comparing each recorded weights_size with the real file on a host that
    had them all (2026-09-21): the six below matched byte for byte, while
    DPA-3.1-3M-FT and PET-OAM-XL are derived by a prepare step and did not. A new
    variant joining this set is a conscious decision, not an accident."""
    import json

    from oh_my_mlip import registry

    checked = set()
    models = json.loads((REPO_ROOT / "models.json").read_text())
    for family, info in models.items():
        if family.startswith("_"):
            continue
        for version in info["versions"]:
            spec = registry.resolve(version)
            targets = fetch.weight_targets(spec)
            if (spec.get("weights_size") and len(targets) == 1
                    and not fetch._looks_like_dir_target(Path(targets[0]))
                    and not fetch._derives_its_target(spec)):
                checked.add(version)
    assert checked == {"Nequix-MP-1", "eSEN-30M-OAM", "EqV3-OMatMPtrjSalex",
                       "EquFlashV2", "EquFlash-v1", "DPA-4.0.1-pro-MPtrj"}
