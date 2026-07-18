"""Tests for scripts/setup_sweep.py -- the deterministic batch driver.

Pins the consensus-decided semantics:
  * complete-then-batch-recover: an induced mid-list install failure does NOT
    stop later targets, and shows up as `failed` in the report;
  * the report is generated STRICTLY from the ledger (a truncated ledger
    yields `not_attempted`, never a guess);
  * gated targets without a token are recorded `skipped_gated`;
  * install/verify commands are injected via explicit test-only flags, no
    monkeypatching of subprocess.

GPU-free: fake install/verify shell scripts stand in for the real ones.
"""
import importlib.util
import json
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "setup_sweep", REPO_ROOT / "scripts" / "setup_sweep.py"
)
driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(driver)


def _script(path: Path, body: str) -> str:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _home(tmp_path: Path) -> Path:
    (tmp_path / "models.json").write_text(json.dumps({
        "_meta": {},
        "Alpha": {"env": "alpha", "versions": {"A-1": {"gated": False}}},
        "Beta": {"env": "beta", "versions": {"B-1": {"gated": False}}},
        "Gamma": {"env": "gamma", "versions": {"G-1": {"gated": False}}},
        "Gated": {"env": "gated", "versions": {"X-1": {"gated": True}}},
    }))
    return tmp_path


def _run_sweep(tmp_path, targets, install_body, verify_body):
    home = _home(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    install = _script(tmp_path / "fake_install.sh", install_body)
    verify = _script(tmp_path / "fake_verify.sh", verify_body)
    driver.sweep(targets, home, ledger, [install], [verify])
    lines = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    return home, ledger, lines


def test_mid_list_failure_does_not_stop_sweep(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake-token-for-gating-only")
    # install fails ONLY for env beta; verify always passes.
    home, ledger, lines = _run_sweep(
        tmp_path,
        ["Alpha", "Beta", "Gamma"],
        'if [ "$1" = "beta" ]; then echo "conda solve boom" >&2; exit 1; fi',
        'echo \'{"pass": true, "degraded": false}\'',
    )
    by_target = {t: [l for l in lines if l.get("target") == t] for t in ("Alpha", "Beta", "Gamma")}
    # Beta failed at install, no verify phase for it...
    assert [l["phase"] for l in by_target["Beta"]] == ["install"]
    assert by_target["Beta"][0]["returncode"] == 1
    assert "conda solve boom" in by_target["Beta"][0]["stderr_tail"]
    # ...and Gamma STILL ran to completion after Beta's failure.
    assert [l["phase"] for l in by_target["Gamma"]] == ["install", "verify"]
    rep = driver.report(ledger)
    assert "Beta" in rep and "failed" in rep
    assert driver.target_status(by_target["Gamma"]) == "verified"


def test_gated_without_token_is_skipped(tmp_path, monkeypatch):
    for var in ("HF_TOKEN", "HF_TOKEN_PATH", "OMM_HF_TOKEN_FILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # hide any real hf cache token
    home, ledger, lines = _run_sweep(
        tmp_path,
        ["Gated", "Alpha"],
        "exit 0",
        'echo \'{"pass": true, "degraded": false}\'',
    )
    gated = [l for l in lines if l.get("target") == "Gated"]
    assert [l["phase"] for l in gated] == ["skipped_gated"]
    # The sweep continued past the skip.
    assert driver.target_status([l for l in lines if l.get("target") == "Alpha"]) == "verified"


def test_degraded_verdict_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake-token-for-gating-only")
    home, ledger, lines = _run_sweep(
        tmp_path,
        ["Alpha"],
        "exit 0",
        'echo \'{"pass": true, "degraded": true, "reason": "env needs CUDA 13.0"}\'',
    )
    assert driver.target_status([l for l in lines if l.get("target") == "Alpha"]) == "degraded"


def test_truncated_ledger_yields_not_attempted(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"seq": 0, "phase": "plan", "targets": ["Alpha", "Beta"]}) + "\n"
        + json.dumps({"seq": 1, "target": "Alpha", "env": "alpha", "phase": "install",
                      "returncode": 0, "stderr_tail": "", "verdict": None}) + "\n"
        + json.dumps({"seq": 2, "target": "Alpha", "env": "alpha", "phase": "verify",
                      "returncode": 0, "stderr_tail": "",
                      "verdict": {"pass": True, "degraded": False}}) + "\n"
    )
    rep = driver.report(ledger)
    assert "not_attempted" in rep and "Beta" in rep
    assert "verified" in rep


def test_unknown_target_recorded_not_crashing(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake-token-for-gating-only")
    home, ledger, lines = _run_sweep(
        tmp_path,
        ["Nope", "Alpha"],
        "exit 0",
        'echo \'{"pass": true, "degraded": false}\'',
    )
    nope = [l for l in lines if l.get("target") == "Nope"]
    assert nope[0]["phase"] == "resolve" and nope[0]["returncode"] == 1
    assert driver.target_status([l for l in lines if l.get("target") == "Alpha"]) == "verified"


def test_ledger_runid_monotonic(tmp_path):
    sweep_dir = tmp_path / ".sweep"
    sweep_dir.mkdir()
    (sweep_dir / "setup_sweep_0007.jsonl").write_text("")
    assert driver.next_ledger_path(tmp_path).name == "setup_sweep_0008.jsonl"
