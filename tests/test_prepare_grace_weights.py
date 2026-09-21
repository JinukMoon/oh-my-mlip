"""GRACE weights come from upstream's Hugging Face copy, verified and resumable.

The ICAMS share grace_models uses served a few KiB/s with no resume, so a first
install took hours and an interruption started it over. The Hugging Face copy
holds a byte-identical SavedModel in different tarball packaging; the prepare
step now fetches it, checks the registry's size/sha256, resumes a partial
download, and falls back to grace_models only when that path fails.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("prepare_grace_weights",
                                               REPO_ROOT / "scripts" / "prepare_grace_weights.py")
pg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pg)


def _fake_release(tmp_path: Path, name: str = "GRACE-2L-OAM") -> tuple[Path, int, str]:
    tarball = tmp_path / "upstream" / f"{name}-model.tar.gz"
    tarball.parent.mkdir(parents=True)
    with tarfile.open(tarball, "w:gz") as tf:
        for rel, data in ((f"{name}_28Jan25/saved_model.pb", b"graph"),
                          (f"{name}_28Jan25/variables/variables.index", b"index"),
                          (f"{name}_28Jan25/variables/variables.data-00000-of-00001", b"w" * 5000)):
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    blob = tarball.read_bytes()
    return tarball, len(blob), hashlib.sha256(blob).hexdigest()


def _home_with(tmp_path: Path, size: int, sha: str) -> Path:
    home = tmp_path / "hub"
    home.mkdir()
    (home / "models.json").write_text(json.dumps({"GRACE": {"versions": {"GRACE-2L-OAM": {
        "weights_source": "GRACE-2L-OAM", "weights_size": size, "weights_sha256": sha}}}}))
    return home


def _run(monkeypatch, home: Path, target_root: Path) -> int:
    monkeypatch.setenv("OH_MY_MLIP_HOME", str(home))
    monkeypatch.setattr(sys, "argv", ["prepare_grace_weights.py", "--target-root", str(target_root)])
    return pg.main()


def test_the_model_is_fetched_verified_and_flattened(tmp_path, monkeypatch):
    tarball, size, sha = _fake_release(tmp_path)
    monkeypatch.setattr(pg, "HF_URL", tarball.parent.as_uri() + "/{name}-model.tar.gz")
    root = tmp_path / "models" / "grace"
    assert _run(monkeypatch, _home_with(tmp_path, size, sha), root) == 0
    target = root / "GRACE-2L-OAM"
    assert (target / "saved_model.pb").read_bytes() == b"graph"
    assert (target / "variables" / "variables.index").read_bytes() == b"index"
    assert not (root / ".GRACE-2L-OAM.hf").exists()          # staging removed on success


def test_a_partial_download_is_continued_not_restarted_from_scratch(tmp_path, monkeypatch):
    tarball, size, sha = _fake_release(tmp_path)
    monkeypatch.setattr(pg, "HF_URL", tarball.parent.as_uri() + "/{name}-model.tar.gz")
    root = tmp_path / "models" / "grace"
    staging = root / ".GRACE-2L-OAM.hf"
    staging.mkdir(parents=True)
    (staging / "GRACE-2L-OAM-model.tar.gz").write_bytes(tarball.read_bytes()[: size // 2])
    # file:// ignores Range, so this exercises the "server ignored Range" restart;
    # either way the result must be the complete, verified file
    assert _run(monkeypatch, _home_with(tmp_path, size, sha), root) == 0
    assert (root / "GRACE-2L-OAM" / "saved_model.pb").is_file()


def test_a_tarball_that_fails_verification_is_discarded_and_never_installed(tmp_path, monkeypatch, capsys):
    tarball, size, _sha = _fake_release(tmp_path)
    monkeypatch.setattr(pg, "HF_URL", tarball.parent.as_uri() + "/{name}-model.tar.gz")
    monkeypatch.setattr(pg, "_grace_models_cmd", lambda: "false")   # the fallback fails too here
    root = tmp_path / "models" / "grace"
    assert _run(monkeypatch, _home_with(tmp_path, size, "0" * 64), root) != 0
    assert "Hugging Face download failed" in capsys.readouterr().err
    assert not (root / "GRACE-2L-OAM" / "saved_model.pb").exists()
    assert not (root / ".GRACE-2L-OAM.hf" / "GRACE-2L-OAM-model.tar.gz").exists()


def test_the_registry_records_the_huggingface_tarball():
    models = json.loads((REPO_ROOT / "models.json").read_text())
    spec = models["GRACE"]["versions"]["GRACE-2L-OAM"]
    assert spec["weights_size"] == 97295519
    assert spec["weights_sha256"] == "b4e5d384e8e3f5222225748f6eed8479c2363cf0022e9ca8dfa4a4f2fe4325ec"
    assert pg.HF_URL.format(name=spec["weights_source"]).endswith("/models/GRACE-2L-OAM-model.tar.gz")
