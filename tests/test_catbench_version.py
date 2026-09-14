"""GPU-free, network-free tests for scripts/catbench_version.py.

The version rule as a file: a NEW job proposes the official latest stable
(pre-releases and yanked files excluded), a RERUN reuses the recorded
version and never re-discovers, a failed discovery is never silently
replaced by the pin (only an explicit --offline is), and an env that does
not hold the version is REPORTED, never upgraded. PyPI and the target
interpreters are replaced by injectable fetch/runner fakes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import catbench_version as cv  # noqa: E402


def _pypi(releases: dict) -> bytes:
    return json.dumps({"info": {"version": "9.9.9"}, "releases": releases}).encode()


def _files(when: str, yanked: bool = False) -> list[dict]:
    return [{"upload_time_iso_8601": when, "yanked": yanked}]


def test_discover_latest_picks_highest_stable_non_yanked_release():
    body = _pypi({
        "1.1.3": _files("2026-07-01T00:00:00Z"),
        "1.1.4": _files("2026-09-08T12:00:00Z"),
        "1.1.10": _files("2026-09-10T00:00:00Z", yanked=True),   # yanked: excluded
        "1.2.0rc1": _files("2026-09-11T00:00:00Z"),             # pre-release: excluded
        "1.1.9": [],                                             # no files: excluded
    })
    out = cv.discover_latest(fetch=lambda url: (200, body, None))
    assert out["latest"] == "1.1.4" and out["latest_release_utc"] == "2026-09-08T12:00:00Z"
    assert out["source_url"] == cv.PYPI_URL and out["http_code"] == 200 and out["error"] is None
    assert out["checked_utc"].endswith("+00:00")


def test_discover_latest_orders_numerically_not_lexically():
    body = _pypi({"1.1.9": _files("2026-01-01T00:00:00Z"), "1.1.10": _files("2026-02-01T00:00:00Z")})
    assert cv.discover_latest(fetch=lambda url: (200, body, None))["latest"] == "1.1.10"


def test_discover_latest_records_failures_instead_of_raising():
    out = cv.discover_latest(fetch=lambda url: (0, b"", "URLError: no network"))
    assert out["latest"] is None and out["error"] == "URLError: no network" and out["http_code"] == 0
    out = cv.discover_latest(fetch=lambda url: (200, b"<html>", None))
    assert out["latest"] is None and "unparseable" in out["error"]
    out = cv.discover_latest(fetch=lambda url: (200, _pypi({"1.0.0": _files("x", yanked=True)}), None))
    assert out["latest"] is None and "no stable" in out["error"]


def test_hub_pin_reads_install_sh_default_and_env_override(tmp_path: Path):
    sh = tmp_path / "install.sh"
    sh.write_text('#!/bin/bash\nCATBENCH_PIN="${OMM_CATBENCH_VERSION:-1.1.3}"\n')
    assert cv.hub_pin(sh, env={}) == {"pin": "1.1.3", "source": f"{sh}: CATBENCH_PIN default"}
    assert cv.hub_pin(sh, env={"OMM_CATBENCH_VERSION": "1.1.4"})["pin"] == "1.1.4"
    sh.write_text("#!/bin/bash\n")
    assert cv.hub_pin(sh, env={})["pin"] is None


def test_hub_pin_matches_the_real_install_sh():
    """The regex must keep matching the line install.sh actually carries."""
    pin = cv.hub_pin(env={})
    assert pin["pin"] and cv._STABLE_RE.match(pin["pin"]), pin


def test_installed_version_uses_the_target_interpreter_via_runner(tmp_path: Path):
    py = tmp_path / "python"; py.write_text("")
    seen = []
    def runner(cmd):
        seen.append(cmd)
        return SimpleNamespace(returncode=0, stdout="1.1.3", stderr="")
    assert cv.installed_version(str(py), runner) == {"python": str(py), "installed": "1.1.3", "error": None}
    assert seen[0][0] == str(py) and "catbench.__version__" in seen[0][2]
    failing = lambda cmd: SimpleNamespace(returncode=1, stdout="", stderr="Traceback\nModuleNotFoundError: catbench")
    assert cv.installed_version(str(py), failing)["installed"] is None
    assert cv.installed_version(str(tmp_path / "missing"), runner)["error"] == "interpreter not found"


def test_decide_new_proposes_latest_and_lists_upgrades_and_mismatches():
    envs = [{"python": "/a/python", "installed": "1.1.3"}, {"python": "/b/python", "installed": "1.1.4"},
            {"python": "/c/python", "installed": None}]
    d = cv.decide("new", latest="1.1.4", pin="1.1.3", record_version=None, envs=envs)
    assert d["ok"] and d["approved_version"] == "1.1.4" and d["status"] == "proposed"
    assert d["pin_matches_approved"] is False
    assert d["requires_env_upgrade"] == ["/a/python"] and d["env_mismatch"] == ["/a/python"] and d["env_unknown"] == ["/c/python"]
    assert "SEPARATE step" in d["note"]


def test_decide_new_never_silently_falls_back_to_the_pin():
    d = cv.decide("new", latest=None, pin="1.1.3", record_version=None, envs=[], offline=False, discovery_error="http 503")
    assert d["ok"] is False and d["approved_version"] is None and "http 503" in d["reason"] and "--offline" in d["reason"]
    d = cv.decide("new", latest=None, pin="1.1.3", record_version=None, envs=[], offline=True)
    assert d["ok"] and d["approved_version"] == "1.1.3" and "--offline" in d["basis"]
    d = cv.decide("new", latest=None, pin=None, record_version=None, envs=[], offline=True)
    assert d["ok"] is False


def test_decide_rerun_is_pinned_and_refuses_an_env_that_drifted():
    ok = cv.decide("rerun", latest="1.1.9", pin="1.1.3", record_version="1.1.4",
                   envs=[{"python": "/b/python", "installed": "1.1.4"}])
    assert ok["ok"] and ok["approved_version"] == "1.1.4" and ok["status"] == "recorded"
    drift = cv.decide("rerun", latest=None, pin=None, record_version="1.1.4",
                      envs=[{"python": "/b/python", "installed": "1.1.5"}])
    assert drift["ok"] is False and drift["env_mismatch"] == ["/b/python"] and "never upgraded" in drift["reason"]
    assert cv.decide("rerun", latest=None, pin=None, record_version=None, envs=[])["ok"] is False


def test_build_record_rerun_does_not_discover(tmp_path: Path):
    record = tmp_path / "catbench_version.json"
    record.write_text(json.dumps({"approved_version": "1.1.4"}))
    calls = []
    def fetch(url):
        calls.append(url); return (200, b"{}", None)
    runner = lambda cmd: SimpleNamespace(returncode=0, stdout="1.1.4", stderr="")
    py = tmp_path / "python"; py.write_text("")
    rec = cv.build_record("rerun", pythons=[str(py)], record=record, offline=False, fetch=fetch, runner=runner,
                          install_sh=tmp_path / "no_install.sh")
    assert calls == [] and rec["official"]["latest"] is None
    assert rec["approved_version"] == "1.1.4" and rec["ok"] and rec["record_read"] == str(record)


def test_cli_new_offline_writes_record_and_json(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(cv, "hub_pin", lambda install_sh=None, env=None: {"pin": "1.1.3", "source": "test"})
    out = tmp_path / "rec.json"
    rc = cv.main(["--mode", "new", "--offline", "--out", str(out), "--json"])
    assert rc == 0
    rec = json.loads(out.read_text())
    assert rec == json.loads(capsys.readouterr().out)
    assert rec["approved_version"] == "1.1.3" and rec["official"]["latest"] is None and rec["mode"] == "new"
    assert rec["hub_pin"]["pin"] == "1.1.3" and rec["envs"] == []


def test_cli_new_online_failure_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(cv, "_default_fetch", lambda url, timeout=30.0: (0, b"", "URLError: offline"))
    monkeypatch.setattr(cv, "hub_pin", lambda install_sh=None, env=None: {"pin": "1.1.3", "source": "test"})
    assert cv.main(["--mode", "new"]) == 2
    err = capsys.readouterr().err
    assert "[stop]" in err and "--offline" in err


def test_cli_rerun_without_record_exits_2(capsys):
    assert cv.main(["--mode", "rerun"]) == 2
    captured = capsys.readouterr()
    assert "rerun requires" in captured.out + captured.err
