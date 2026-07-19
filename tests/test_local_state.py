"""Tests for materialize-on-verify (models.local.json, the verified ledger).

Pins the contract:
  * record_local_verified upserts incrementally — a new install adds its
    entry, a re-verification overwrites only its own, others are untouched;
  * registry.resolve() exposes the record as spec["local_verified"] (and
    None when the version has no record);
  * fetch.weight_targets extracts explicit inference weight paths and stays
    empty for arch-pinned / name-based specs.

GPU-free.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oh_my_mlip import fetch, registry  # noqa: E402


def _home(tmp_path: Path) -> Path:
    (tmp_path / "models.json").write_text(json.dumps({
        "_meta": {},
        "Alpha": {
            "env": "alpha",
            "python": "${OH_MY_MLIP_HOME}/envs/alpha/bin/python",
            "import": ["import alpha"],
            "default_version": "A-1",
            "versions": {"A-1": {"gated": False, "inference": ["calc = None"]}},
        },
    }))
    return tmp_path


def _models(home: Path) -> dict:
    return json.loads((home / "models.json").read_text())


def _spec(version: str = "A-1") -> dict:
    return {"model": "Alpha", "version": version, "env": "alpha",
            "python": "/envs/alpha/bin/python"}


VERDICT = {"pass": True, "device": "cuda", "degraded": False,
           "energy_ev": -16.39, "forces_shape": [4, 3]}


def test_record_upserts_incrementally(tmp_path):
    home = _home(tmp_path)
    registry.record_local_verified(_spec("A-1"), VERDICT, ["/w/a.model"], str(home))
    registry.record_local_verified(_spec("B-9"), VERDICT, [], str(home))
    data = registry.load_local_models(str(home))
    assert set(data) == {"A-1", "B-9"}
    # re-verification overwrites its own entry only
    registry.record_local_verified(
        _spec("A-1"), dict(VERDICT, device="cpu", degraded=True), ["/w/a2.model"], str(home)
    )
    data = registry.load_local_models(str(home))
    assert data["A-1"]["device"] == "cpu" and data["A-1"]["weights"] == ["/w/a2.model"]
    assert data["B-9"]["device"] == "cuda"        # untouched
    assert data["A-1"]["verified_at"].endswith("+00:00")


def test_resolve_exposes_local_verified(tmp_path, monkeypatch):
    home = _home(tmp_path)
    monkeypatch.setenv("OH_MY_MLIP_HOME", str(home))
    spec = registry.resolve("Alpha", models=_models(home))
    assert spec["local_verified"] is None
    registry.record_local_verified(spec, VERDICT, ["/w/a.model"], str(home))
    spec2 = registry.resolve("Alpha", models=_models(home))
    assert spec2["local_verified"]["weights"] == ["/w/a.model"]
    assert spec2["local_verified"]["device"] == "cuda"


def test_weight_targets_extraction():
    explicit = {"arch_pinned": False, "inference": [
        "calc = Loader('/home/u/hub/models/grace/GRACE-2L-OAM')",
    ]}
    got = fetch.weight_targets(explicit)
    assert got and got[0].endswith("models/grace/GRACE-2L-OAM")
    assert fetch.weight_targets({"arch_pinned": True, "inference": ["x"]}) == []
    by_name = {"arch_pinned": False, "inference": ["calc = mace_mp(model='medium-mpa-0')"]}
    assert fetch.weight_targets(by_name) == []
