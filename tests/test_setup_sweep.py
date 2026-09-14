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
import shutil
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


def test_disk_floor_records_skipped_disk_and_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake-token-for-gating-only")
    home = _home(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    install = _script(tmp_path / "fake_install.sh", "exit 0")
    verify = _script(tmp_path / "fake_verify.sh", 'echo \'{"pass": true, "degraded": false}\'')
    # Absurdly high floor: every target must be recorded skipped_disk, none run.
    driver.sweep(["Alpha", "Beta"], home, ledger, [install], [verify],
                 min_free_gb=10**9)
    lines = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    phases = [(l.get("target"), l["phase"]) for l in lines if l.get("target")]
    assert phases == [("Alpha", "skipped_disk"), ("Beta", "skipped_disk")]
    rep = driver.report(ledger)
    assert "skipped_disk: 2" in rep


def test_ledger_runid_monotonic(tmp_path):
    sweep_dir = tmp_path / ".sweep"
    sweep_dir.mkdir()
    (sweep_dir / "setup_sweep_0007.jsonl").write_text("")
    assert driver.next_ledger_path(tmp_path).name == "setup_sweep_0008.jsonl"


# ═══════════════════════════════════════════════════════════════════════════
# --fresh-root cycle (plan G2): whole cycles GPU-free against a throwaway git
# hub, every external step a fake script injected through FreshHooks.
# ═══════════════════════════════════════════════════════════════════════════
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402

_FR_SPEC = importlib.util.spec_from_file_location("fresh_root", REPO_ROOT / "scripts" / "fresh_root.py")
fresh_root = importlib.util.module_from_spec(_FR_SPEC)
_FR_SPEC.loader.exec_module(fresh_root)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout


def _fresh_hub(root: Path) -> Path:
    """Miniature hub with two envs, committed; local state present (and
    therefore hashable as a mandatory attribution target)."""
    repo = root / "hub"
    repo.mkdir()
    # the real package: the isolation check runs the runtime copy's own resolve()
    shutil.copytree(REPO_ROOT / "oh_my_mlip", repo / "oh_my_mlip", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "models.json").write_text(json.dumps({
        "_meta": {},
        "Alpha": {"env": "alpha", "python": "${OH_MY_MLIP_HOME}/envs/alpha/bin/python", "default_version": "A-1",
                  "versions": {"A-1": {"inference": ["calc = L('${OH_MY_MLIP_HOME}/models/alpha/a.pt')"],
                                       "finetune": {"status": "documented", "runnable_as_installed": True}},
                               "A-2": {"inference": ["calc = L('x')"],
                                       "finetune": {"status": "not-supported", "evidence": ["https://up/README"]}}}},
        "Beta": {"env": "beta", "python": "${OH_MY_MLIP_HOME}/envs/beta/bin/python",
                 "versions": {"B-1": {"inference": ["calc = L('y')"],
                                      "finetune": {"status": "documented", "runnable_as_installed": True}}}},
    }))
    (repo / "envs").mkdir()
    for env in ("alpha", "beta"):
        (repo / "envs" / f"{env}.yml").write_text(f"name: {env}\n")
    (repo / "envs" / "_expected.json").write_text("{}\n")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "tool.py").write_text("print('tool')\n")
    (repo / "install.sh").write_text("#!/bin/sh\nexit 0\n")
    (repo / ".gitignore").write_text("envs/*/\nmodels/\n.sweep/\n*.local.json\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / "models.local.json").write_text("{}\n")
    return repo


FAKE_INSTALL = """
env="$1"
mkdir -p "$PWD/envs/$env/bin" && echo "" > "$PWD/envs/$env/bin/python"
echo "installed $env" > "$PWD/work/installed_marker"
[ -n "$FAKE_ADOPT" ] && echo '{"alpha": "/home/user/miniconda3/envs/Alpha"}' > "$PWD/env_map.local.json"
[ -n "$FAKE_MUTATE" ] && echo "mutated" > "$PWD/scripts/tool.py"
[ -n "$FAKE_HUB_WRITE" ] && echo '{"leak": 1}' > "$FAKE_HUB_WRITE/models.local.json"
exit 0
"""
FAKE_VERIFY = 'echo \'{"pass": true, "degraded": false, "gpu_mem_allocated_bytes": 1, "energy_ev": -1.0}\''
FAKE_FT_RUN = """
variant="$1"; shift
while [ $# -gt 0 ]; do case "$1" in --out) out="$2"; shift 2;; *) shift;; esac; done
printf '#!/bin/sh\\n' > "$out/finetune_$variant.sh"; echo w > "$out/$variant.model"
echo '{"schema": "ft_run.json/1", "rematerialize": ["ft_run.py"]}' > "$out/ft_run.json"; exit 0
"""
# a trainer that writes a checkpoint but NO ft_run.json provenance record
FAKE_FT_RUN_NO_PROVENANCE = """
variant="$1"; shift
while [ $# -gt 0 ]; do case "$1" in --out) out="$2"; shift 2;; *) shift;; esac; done
printf '#!/bin/sh\\n' > "$out/finetune_$variant.sh"; echo w > "$out/$variant.model"; exit 0
"""
FAKE_FT_VERIFY = 'echo \'{"pass": true, "energy_ev": -2.0, "forces_shape": [4, 3], "gpu_used": true}\''
FAKE_CONDA = 'echo "{\\"pkgs_dirs\\": [\\"$CONDA_PKGS_DIRS\\"]}"'
# <kind> <prefix> <python>: one answer per inventory kind, with a tokenized
# URL, URL userinfo and an HF-shaped token that the driver must scrub before
# persisting. The token is assembled from fragments so this source holds no
# token-shaped literal (scripts/verify_no_token.py scans the tree).
FAKE_HF_TOKEN = "hf_" + "abcdefghijklm" + "nopqrstuvwxyz"
FAKE_INVENTORY = """
case "$1" in
  conda_explicit) echo "# platform: linux-64"; echo "@EXPLICIT"; echo "https://user:s3cret@conda.example/pkgs/alpha-1.0-0.tar.bz2";;
  pip_freeze) echo "alpha==1.0"; echo "beta @ https://x.example/beta.whl?token=abc123"; echo "# via __TOKEN__" >&2;;
  pip_list) echo '[{"name": "alpha", "version": "1.0"}]'; echo "# token __TOKEN__" >&2;;
  *) echo "unknown kind $1" >&2; exit 3;;
esac
""".replace("__TOKEN__", FAKE_HF_TOKEN)
# a pip_list that is not JSON: the inventory must fail, not accept it
FAKE_INVENTORY_MALFORMED = FAKE_INVENTORY.replace("""echo '[{"name": "alpha", "version": "1.0"}]'""", "echo 'not json'")
# the first kind marks its start inside the owned prefix, then lingers: the
# driver's floor monitor must stop it and abort the cycle like any phase
FAKE_INVENTORY_SLOW = 'touch "$2/inv_started"; sleep 5; echo "alpha==1.0"'


def _hooks(tmp_path, monkeypatch, **overrides) -> "driver.FreshHooks":
    monkeypatch.setenv("HF_TOKEN", "hf_fake-token-for-gating-only")
    proc_root = tmp_path / "proc"
    proc_root.mkdir(exist_ok=True)
    dataset = tmp_path / "demo.traj"
    dataset.write_text("")
    kw = dict(
        install_cmd=[_script(tmp_path / "fake_install.sh", FAKE_INSTALL)],
        verify_cmd=[_script(tmp_path / "fake_verify.sh", FAKE_VERIFY)],
        # the REAL ft_sweep.py driven by fake trainers (the runtime copy of the
        # miniature hub carries no scripts/, so the real script is named directly)
        ft_cmd=[sys.executable, str(REPO_ROOT / "scripts" / "ft_sweep.py")],
        ft_extra_args=["--ft-run-cmd", _script(tmp_path / "fake_ft_run.sh", FAKE_FT_RUN),
                       "--ft-verify-cmd", _script(tmp_path / "fake_ft_verify.sh", FAKE_FT_VERIFY)],
        conda_cmd=[_script(tmp_path / "fake_conda.sh", FAKE_CONDA)],
        inventory_cmd=[_script(tmp_path / "fake_inventory.sh", FAKE_INVENTORY)],
        proc_root=proc_root,
        ft_dataset=dataset,
        min_free_gib=0.0,
        # the miniature hub has no preflight: the fixture stands in with an
        # explicit (zero) operator peak and no reserve, so the gate is the
        # measured addends alone; tests of the gate itself override these
        peak_gib=0.0,
        start_reserve_gib=0.0,
        df_poll_seconds=0.05,
        du_sample_seconds=0.05,
    )
    kw.update(overrides)
    return driver.FreshHooks(**kw)


def _campaign(tmp_path, monkeypatch, targets, **overrides):
    hub = _fresh_hub(tmp_path)
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"patterns": []}))
    hooks = _hooks(tmp_path, monkeypatch, **overrides)
    ledger = driver.fresh_sweep(targets, hub, tmp_path / ".omm_fresh", "c1", allow, hooks)
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines() if ln.strip()]
    return hub, ledger, rows


def _terminal(rows):
    return {(r["env"], r["kind"], r["variant"]): r["state"] for r in rows
            if r.get("kind") and r.get("variant") and r.get("state")}


def _guards(rows, env):
    return {r["phase"]: r["state"] for r in rows if r.get("variant") == "*" and r.get("env") == env}


def test_fresh_cycle_happy_path_evidence_survives_cleanup(tmp_path, monkeypatch):
    hub, ledger, rows = _campaign(tmp_path, monkeypatch, ["alpha", "Beta"])
    assert ledger == hub / ".sweep" / "campaign_c1" / "ledger.jsonl"
    assert _terminal(rows) == {
        ("alpha", "inference", "A-1"): "passed", ("alpha", "inference", "A-2"): "passed",
        ("alpha", "finetune", "A-1"): "passed", ("alpha", "finetune", "A-2"): "unsupported",
        ("beta", "inference", "B-1"): "passed", ("beta", "finetune", "B-1"): "passed",
    }
    for env in ("alpha", "beta"):
        g = _guards(rows, env)
        assert g["snapshot"] == "passed" and g["materialize"] == "passed" and g["routes"] == "passed"
        assert g["isolation_pre"] == "passed" and g["install"] == "passed" and g["isolation_post"] == "passed"
        assert g["sources"] == "passed" and g["attribution"] == "passed" and g["cleanup"] == "cleaned"
    # every row after the snapshot carries the manifest hash, and it is ONE hash per cycle
    shas = {r["manifest_sha256"] for r in rows if r.get("phase") not in ("plan", "snapshot_check")}
    assert None not in shas and len(shas) == 1
    manifest_sha = shas.pop()
    assert (hub / ".sweep" / "campaign_c1" / "snapshot" / manifest_sha[:12] / "manifest.json").is_file()
    # the runtime roots were cleaned, the evidence (ledger + verified copies) was not
    assert not any((tmp_path / ".omm_fresh").iterdir())
    pres = [r for r in rows if r.get("phase") == "preserve"]
    assert [p["state"] for p in pres] == ["passed", "passed"]
    files = pres[0]["evidence"]["files"]
    # the ft ledger, the phase logs (full stdout/stderr) AND the fine-tune
    # checkpoint + rerun script were copied out and re-verified by hash
    assert any(f.endswith("alpha.jsonl") for f in files)
    assert any(f.endswith(".sweep/phases/install.out.log") for f in files)
    assert any(f.endswith("A-1.model") for f in files) and any(f.endswith("finetune_A-1.sh") for f in files)
    # ... and ft_run's provenance record, a REQUIRED artifact of every passed fine-tune row
    assert any(f.endswith("A-1/ft_run.json") for f in files)
    required = set(pres[0]["evidence"]["required_artifacts"])
    assert any(r.endswith("A-1/ft_run.json") for r in required)
    ft_row = next(r for r in rows if r.get("kind") == "finetune" and r.get("variant") == "A-1")
    assert ft_row["evidence"]["ft_run_json"].endswith("A-1/ft_run.json")
    assert len(ft_row["evidence"]["ft_run_json_sha256"]) == 64 and ft_row["evidence"]["ft_run_json_schema"] == "ft_run.json/1"
    assert all(Path(f).is_file() and not Path(f).is_relative_to(tmp_path / ".omm_fresh") for f in files)
    record = json.loads(Path(pres[0]["evidence"]["record"]).read_text())
    assert record["manifest_sha256"] == manifest_sha and record["file_count"] == len(files)
    assert fresh_root.ledger_carries_manifest(ledger, manifest_sha)
    # proof of cleanup is the owned path's absence; df is recorded as a diagnostic
    post = [r for r in rows if r.get("phase") == "post_cleanup"]
    assert all(p["state"] == "passed" and p["evidence"]["runtime_absent"] for p in post)
    assert post[0]["evidence"]["unit"] == "GiB"
    cleaned = next(r for r in rows if r.get("phase") == "cleanup" and r["env"] == "alpha")
    assert cleaned["evidence"]["absent"] and cleaned["evidence"]["preserved"]["file_count"] == len(files)
    # inference verify never records into models.local.json, even inside the root
    verify = next(r for r in rows if r.get("phase") == "verify" and r["variant"] == "A-1")
    assert "--no-local-record" in verify["evidence"]["command"] and "--version" in verify["evidence"]["command"]
    # the real hub was never written: local state hash unchanged, no env dirs
    assert (hub / "models.local.json").read_text() == "{}\n"
    assert not (hub / "envs" / "alpha").exists()
    # install ran INSIDE the runtime root with the routes exported
    install = next(r for r in rows if r.get("phase") == "install" and r["env"] == "alpha")
    assert install["runtime_root"].startswith(str(tmp_path / ".omm_fresh" / "alpha_"))
    budget = next(r for r in rows if r.get("phase") == "budget" and r["env"] == "alpha")
    assert budget["evidence"]["budget_actual"]["peak_bytes"] > 0
    assert budget["evidence"]["unit"] == "GiB" and budget["evidence"]["budget_actual"]["unit"] == "GiB"
    # the miniature hub has no preflight: the fixture's explicit operator peak gated the start, and says so
    assert budget["evidence"]["budget_estimate"] is None and budget["evidence"]["estimate_status"] == "operator_estimate"
    assert budget["evidence"]["peak_estimate_provenance"] == "operator_estimate" and budget["evidence"]["peak_estimate_gib"] == 0.0
    assert "cleaned" in driver.fresh_report(ledger) and "runtime root absent: True" in driver.fresh_report(ledger)


def test_check_runs_before_materialize_and_blocks(tmp_path, monkeypatch):
    hub = _fresh_hub(tmp_path)
    (hub / "scripts" / "rogue.py").write_text("x\n")
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"patterns": []}))
    ledger = driver.fresh_sweep(["alpha"], hub, tmp_path / ".omm_fresh", "c2", allow, _hooks(tmp_path, monkeypatch))
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    check = next(r for r in rows if r.get("phase") == "snapshot_check")
    assert check["state"] == "failed(snapshot:unexpected_untracked)" and "scripts/rogue.py" in check["stderr_tail"]
    assert not any((tmp_path / ".omm_fresh").iterdir())  # nothing materialized
    assert not any(r.get("phase") == "install" for r in rows)


def test_adoption_entry_in_runtime_is_isolation_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_ADOPT", "1")
    hub, ledger, rows = _campaign(tmp_path, monkeypatch, ["alpha"])
    g = _guards(rows, "alpha")
    assert g["isolation_pre"] == "passed" and g["isolation_post"] == "failed(isolation)"
    assert _terminal(rows)[("alpha", "inference", "A-1")] == "failed(isolation)"
    assert not any(r.get("phase") == "verify" for r in rows)  # no compute after the leak
    assert g["cleanup"] == "cleaned"  # evidence durable, so the leaked root is still cleaned


# a verifier that passes the reload but REMOVES the checkpoint afterwards: the
# ledger row is `passed` with a ckpt path that no longer exists at preserve time
FAKE_FT_VERIFY_EATS_CKPT = 'rm -f "$1"; echo \'{"pass": true, "energy_ev": -2.0, "forces_shape": [4, 3], "gpu_used": true}\''


def test_missing_final_artifact_fails_preserve_and_keeps_the_root(tmp_path, monkeypatch):
    """Final artifacts named by fine-tune rows are REQUIRED at preserve: a
    missing checkpoint is failed(preserve:required_missing), cleanup is
    refused as evidence-not-durable and the owned root is retained."""
    hub, ledger, rows = _campaign(
        tmp_path, monkeypatch, ["alpha"],
        ft_extra_args=["--ft-run-cmd", _script(tmp_path / "fake_ft_run.sh", FAKE_FT_RUN),
                       "--ft-verify-cmd", _script(tmp_path / "fake_ft_verify_eats.sh", FAKE_FT_VERIFY_EATS_CKPT)])
    g = _guards(rows, "alpha")
    assert _terminal(rows)[("alpha", "finetune", "A-1")] == "passed"
    assert g["preserve"] == "failed(preserve:required_missing)"
    pres = next(r for r in rows if r.get("phase") == "preserve")
    assert any(p.endswith("A-1.model") for p in pres["evidence"]["required_artifacts"])
    assert any("A-1.model" in m for m in pres["evidence"]["missing"])
    assert g["cleanup"] == "failed(cleanup:evidence_not_durable)" and g["post_cleanup"] == "failed(cleanup:incomplete)"
    roots = list((tmp_path / ".omm_fresh").iterdir())
    assert len(roots) == 1 and (roots[0] / fresh_root.OWNERSHIP_FILE).is_file()
    # the registry still binds the retained root (nothing was cleaned)
    registry = json.loads((hub / ".sweep" / "campaign_c1" / fresh_root.OWNED_ROOTS_FILE).read_text())
    assert registry["roots"][str(roots[0].resolve())]["cleaned_utc"] is None


FAKE_FT_VERIFY_EATS_PROVENANCE = ('rm -f "$(dirname "$1")/ft_run.json"; '
                                  'echo \'{"pass": true, "energy_ev": -2.0, "forces_shape": [4, 3], "gpu_used": true}\'')


def test_ft_run_provenance_record_is_required_and_preserved_before_cleanup(tmp_path, monkeypatch):
    """ft_run's <out>/ft_run.json (rematerialization provenance) is required
    twice: a train step that writes none is failed(no_provenance_record)
    (never a pass), and a passed row whose record is gone by preserve time
    is failed(preserve:required_missing) with the root retained."""
    hub, ledger, rows = _campaign(
        tmp_path, monkeypatch, ["alpha"],
        ft_extra_args=["--ft-run-cmd", _script(tmp_path / "fake_ft_run_noprov.sh", FAKE_FT_RUN_NO_PROVENANCE),
                       "--ft-verify-cmd", _script(tmp_path / "fake_ft_verify.sh", FAKE_FT_VERIFY)])
    assert _terminal(rows)[("alpha", "finetune", "A-1")] == "failed(no_provenance_record)"
    ft_row = next(r for r in rows if r.get("kind") == "finetune" and r.get("variant") == "A-1")
    assert "wrote no ft_run.json" in ft_row["stderr_tail"] and "ft_run_json" not in ft_row["evidence"]
    assert "ckpt" not in ft_row["evidence"]  # refused before checkpoint discovery: no artifact is claimed
    g = _guards(rows, "alpha")
    assert g["preserve"] == "passed" and g["cleanup"] == "cleaned"  # nothing passed, nothing owed
    # a record that existed at train time but is gone at preserve: required, so preserve fails closed
    (tmp_path / "second").mkdir()
    hub, ledger, rows = _campaign(
        tmp_path / "second", monkeypatch, ["alpha"],
        ft_extra_args=["--ft-run-cmd", _script(tmp_path / "second" / "fake_ft_run.sh", FAKE_FT_RUN),
                       "--ft-verify-cmd", _script(tmp_path / "second" / "fake_ft_verify_eats_prov.sh",
                                                  FAKE_FT_VERIFY_EATS_PROVENANCE)])
    assert _terminal(rows)[("alpha", "finetune", "A-1")] == "passed"
    g = _guards(rows, "alpha")
    assert g["preserve"] == "failed(preserve:required_missing)"
    pres = next(r for r in rows if r.get("phase") == "preserve")
    assert any(p.endswith("A-1/ft_run.json") for p in pres["evidence"]["required_artifacts"])
    assert any("ft_run.json" in m for m in pres["evidence"]["missing"])
    assert g["cleanup"] == "failed(cleanup:evidence_not_durable)"
    roots = list((tmp_path / "second" / ".omm_fresh").iterdir())
    assert len(roots) == 1 and (roots[0] / fresh_root.OWNERSHIP_FILE).is_file()


def test_cycle_children_run_with_the_isolated_effective_env(tmp_path, monkeypatch):
    """Inherited cache overrides are dropped, HOME is the in-root campaign
    home, the read-only inputs are path exports, and the route check ran on
    that effective env; the real home received no campaign write."""
    real_home = tmp_path / "realhome"
    (real_home / ".cache" / "huggingface").mkdir(parents=True)
    (real_home / ".cache" / "huggingface" / "token").write_text("hf_not-a-real-token\n")
    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(real_home / ".cache" / "hf"))
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    monkeypatch.delenv("CONDARC", raising=False)
    install = 'env="$1"; mkdir -p "$PWD/envs/$env/bin" && echo "" > "$PWD/envs/$env/bin/python"\n' \
              'mkdir -p "$HOME/.cache/probe" && echo "$TRANSFORMERS_CACHE|$HF_TOKEN_PATH" > "$PWD/work/env_probe.txt"\nexit 0'
    hub, ledger, rows = _campaign(tmp_path, monkeypatch, ["alpha"], cleanup=False,
                                  install_cmd=[_script(tmp_path / "fake_install_env.sh", install)])
    g = _guards(rows, "alpha")
    assert g["routes"] == "passed" and g["install"] == "passed"
    mat = next(r for r in rows if r.get("phase") == "materialize")["evidence"]
    root = Path(next(r for r in rows if r.get("phase") == "install")["runtime_root"])
    assert mat["rejected_inherited"] == ["TRANSFORMERS_CACHE"]
    assert mat["read_only_inputs"] == {"HF_TOKEN_PATH": str(real_home / ".cache" / "huggingface" / "token")}
    assert mat["exports"]["HOME"] == str(root / "home")
    # the child saw the effective env: override gone, token PATH supplied, HOME inside the root
    assert (root / "work" / "env_probe.txt").read_text().strip() == f"|{real_home / '.cache' / 'huggingface' / 'token'}"
    assert (root / "home" / ".cache" / "probe").is_dir()
    assert sorted(p.name for p in real_home.rglob("*") if p.is_dir()) == [".cache", "huggingface"]
    routes = next(r for r in rows if r.get("phase") == "routes")["evidence"]
    assert routes["ok"] and not routes["problems"]


def test_out_of_root_route_aborts_before_install(tmp_path, monkeypatch):
    shared = _script(tmp_path / "shared_conda.sh", 'echo \'{"pkgs_dirs": ["/home/user/miniconda3/pkgs"]}\'')
    hub, ledger, rows = _campaign(tmp_path, monkeypatch, ["alpha"], conda_cmd=[shared])
    g = _guards(rows, "alpha")
    assert g["routes"] == "failed(routes:out_of_root)"
    assert "install" not in g and not any(r.get("phase") == "verify" for r in rows)
    assert all(s == "failed(routes:out_of_root)" for s in _terminal(rows).values())


def test_source_mutation_is_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MUTATE", "1")
    hub, ledger, rows = _campaign(tmp_path, monkeypatch, ["alpha"])
    src = next(r for r in rows if r.get("phase") == "sources")
    assert src["state"] == "failed(source_mutated)" and src["evidence"]["mutated"] == ["scripts/tool.py"]


def test_unattributable_hash_change_pauses_campaign(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_HUB_WRITE", str(tmp_path / "hub"))
    hub, ledger, rows = _campaign(tmp_path, monkeypatch, ["alpha", "beta"])
    attr = next(r for r in rows if r.get("phase") == "attribution")
    assert attr["state"] == "failed(attribution)"
    target = str(hub / "models.local.json")
    ev = attr["evidence"]
    assert target in ev["changed"]
    # the row reports the observation (both hashes) and attributes the change to nobody
    assert ev["changed"][target] == {"before": ev["targets_before"][target], "after": ev["targets_after"][target]}
    assert ev["attribution"] == "unknown" and "writer" not in ev
    assert any(r.get("phase") == "pause" for r in rows)
    assert not any(r.get("env") == "beta" for r in rows)  # campaign paused before beta


def test_skipped_disk_and_preflight_band_are_resource_blocked(tmp_path, monkeypatch):
    hub, ledger, rows = _campaign(tmp_path, monkeypatch, ["alpha"], min_free_gib=10**9)
    assert {r["phase"] for r in rows if r.get("kind")} == {"skipped_disk"}
    assert set(_terminal(rows).values()) == {"resource-blocked"}
    assert not (tmp_path / ".omm_fresh").exists() or not any((tmp_path / ".omm_fresh").iterdir())
    # schema-2 preflight: measured free must clear conservative peak (GiB) + start reserve
    preflight = {"envs": [{"env": "beta", "conservative_disk_peak_gib": 10**9, "conservative_disk_peak_gb": 2 * 10**9}]}
    (tmp_path / "second").mkdir()
    hub, ledger, rows = _campaign(tmp_path / "second", monkeypatch, ["beta"], preflight=preflight, start_reserve_gib=5.0)
    row = next(r for r in rows if r.get("phase") == "budget")
    assert row["state"] == "resource-blocked" and row["evidence"]["additional_free_gib_needed"] > 0
    ev = row["evidence"]
    assert ev["unit"] == "GiB" and ev["peak_estimate_gib"] == 10**9 and ev["gate_gib"] == 10**9 + 5.0
    assert ev["start_reserve_gib"] == 5.0 and ev["estimate_status"] == "provisional"
    assert ev["peak_estimate_provenance"] == "preflight"
    assert ev["home_cold_cache_gib"] == 0.0 and ev["floor_gate_gib"] == ev["min_free_gib"]
    # a decimal-GB-only preflight row is converted once, never mixed
    assert driver.peak_estimate_gib({"conservative_disk_peak_gb": 10.73741824}) == 10.0
    assert driver.peak_estimate_gib({"estimate_confidence": "provisional"}) is None
    # an env WITHOUT an estimate (and no --peak-gib) is resource-blocked before
    # anything is materialized: the measured floor alone is not a budget
    (tmp_path / "third").mkdir()
    hub, ledger, rows = _campaign(tmp_path / "third", monkeypatch, ["alpha"], preflight=preflight, peak_gib=None,
                                  disk_free=lambda p: int(100 * driver.GIB))
    row = next(r for r in rows if r.get("phase") == "budget")
    ev = row["evidence"]
    assert row["state"] == "resource-blocked" and "no conservative peak estimate" in row["stderr_tail"]
    assert ev["estimate_status"] == "unknown" and ev["peak_estimate_provenance"] == "unknown"
    assert ev["peak_estimate_gib"] is None and ev["gate_gib"] is None
    assert set(_terminal(rows).values()) == {"resource-blocked"}
    assert not (tmp_path / "third" / ".omm_fresh").exists() or not any((tmp_path / "third" / ".omm_fresh").iterdir())
    assert not any(r.get("phase") in ("snapshot", "materialize") for r in rows)
    # the operator's explicit conservative peak stands in, with provenance, and enters the gate
    (tmp_path / "fourth").mkdir()
    hub, ledger, rows = _campaign(tmp_path / "fourth", monkeypatch, ["alpha"], preflight=preflight, peak_gib=1.0,
                                  start_reserve_gib=0.5, disk_free=lambda p: int(1.2 * driver.GIB))
    row = next(r for r in rows if r.get("phase") == "budget")
    ev = row["evidence"]
    assert row["state"] == "resource-blocked" and ev["gate_gib"] == 1.5 and ev["peak_estimate_gib"] == 1.0
    assert ev["peak_estimate_provenance"] == "operator_estimate" and ev["estimate_status"] == "operator_estimate"
    assert round(ev["additional_free_gib_needed"], 3) == 0.3
    (tmp_path / "fifth").mkdir()
    hub, ledger, rows = _campaign(tmp_path / "fifth", monkeypatch, ["alpha"], preflight=preflight, peak_gib=1.0,
                                  start_reserve_gib=0.5, disk_free=lambda p: int(2.0 * driver.GIB))
    row = next(r for r in rows if r.get("phase") == "budget")
    assert row["state"] == "passed" and row["evidence"]["peak_estimate_provenance"] == "operator_estimate"
    assert _terminal(rows)[("alpha", "inference", "A-1")] == "passed"


def test_cold_isolated_home_caches_are_measured_into_the_start_gate(tmp_path, monkeypatch):
    """With HOME isolated, the host caches env.sh would have adopted
    ($HOME/.cache/<family>) re-download into the root: their MEASURED host
    size is added to the start gate (peak + cold + reserve) and to the
    measured floor. The host HOME is only read."""
    host_home = tmp_path / "host_home"
    cache = host_home / ".cache" / "alpha"
    cache.mkdir(parents=True)
    with (cache / "weights.bin").open("wb") as fh:  # sparse: apparent size counts, no real disk use
        fh.truncate(256 * 1024 * 1024)
    (host_home / ".cache" / "unrelated").mkdir()
    (host_home / ".cache" / "unrelated" / "big").write_bytes(b"\0" * 1024)
    before = sorted(str(p.relative_to(host_home)) for p in host_home.rglob("*"))
    # zero operator peak: the gate (0.25) clears but the floor 0.4 GiB + 0.25 GiB
    # cold cache > 0.5 GiB measured free => blocked
    hub, ledger, rows = _campaign(tmp_path, monkeypatch, ["alpha"], host_home=host_home,
                                  min_free_gib=0.4, disk_free=lambda p: int(0.5 * driver.GIB))
    row = next(r for r in rows if r.get("phase") == "skipped_disk")
    ev = row["evidence"]
    assert row["state"] == "resource-blocked" and ev["peak_estimate_provenance"] == "operator_estimate"
    assert ev["home_cold_cache_gib"] == 0.25 and round(ev["floor_gate_gib"], 3) == 0.65 and ev["gate_gib"] == 0.25
    assert ev["home_cold_cache"]["dirs"] == {str(cache): 0.25} and "unrelated" not in json.dumps(ev)
    assert round(ev["additional_free_gib_needed"], 3) == 0.15 and "cold HOME caches 0.25" in row["stderr_tail"]
    assert not (tmp_path / ".omm_fresh").exists() or not any((tmp_path / ".omm_fresh").iterdir())
    # known peak: gate = peak + cold + reserve
    preflight = {"envs": [{"env": "alpha", "conservative_disk_peak_gib": 1.0}]}
    (tmp_path / "second").mkdir()
    hub, ledger, rows = _campaign(tmp_path / "second", monkeypatch, ["alpha"], host_home=host_home,
                                  preflight=preflight, start_reserve_gib=0.5, min_free_gib=0.0,
                                  disk_free=lambda p: int(1.6 * driver.GIB))
    row = next(r for r in rows if r.get("phase") == "budget")
    assert row["state"] == "resource-blocked" and row["evidence"]["gate_gib"] == 1.75
    assert round(row["evidence"]["additional_free_gib_needed"], 3) == 0.15
    # enough room: the cycle runs and the isolated home's actual size is measured into budget_actual
    (tmp_path / "third").mkdir()
    hub, ledger, rows = _campaign(tmp_path / "third", monkeypatch, ["alpha"], host_home=host_home,
                                  preflight=preflight, start_reserve_gib=0.5, min_free_gib=0.0,
                                  disk_free=lambda p: int(2.0 * driver.GIB))
    row = next(r for r in rows if r.get("phase") == "budget")
    assert row["state"] == "passed" and row["evidence"]["home_cold_cache_gib"] == 0.25
    assert row["evidence"]["budget_actual"]["isolated_home_gib"] >= 0.0
    assert _terminal(rows)[("alpha", "inference", "A-1")] == "passed"
    # the host HOME received nothing
    assert sorted(str(p.relative_to(host_home)) for p in host_home.rglob("*")) == before
    # the measurement is complete: nothing skipped, not partial
    assert row["evidence"]["home_cold_cache"]["skipped"] == {} and row["evidence"]["home_cold_cache"]["partial"] is False
    assert row["evidence"]["home_cold_cache_provenance"] == "measured_host_home"


def test_partial_cold_cache_measurement_is_never_a_budget(tmp_path, monkeypatch):
    """A known cache path that is a symlink is MEASURED through its recorded
    target (read-only; the deterministic adopted layout symlinks
    ~/.cache/<name> into the hub), but a dangling / non-directory link or
    an unreadable subtree is skipped WITH its reason and marks the
    measurement partial; a partial total is a lower bound, so the cycle is
    resource-blocked before materializing unless the operator supplies an
    explicit conservative estimate (recorded as such). Inside a measured
    tree symlinks are not followed; no path outside the known ones is
    searched."""
    host_home = tmp_path / "host_home"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with (elsewhere / "huge.bin").open("wb") as fh:
        fh.truncate(4 * 1024 ** 3)  # sparse: apparent size, no real disk use
    (elsewhere / "sub").mkdir()
    os.symlink(tmp_path, elsewhere / "sub" / "escape")  # inside the tree: never followed
    (host_home / ".cache").mkdir(parents=True)
    os.symlink(elsewhere, host_home / ".cache" / "alpha")
    # pure function: a resolvable symlinked <name> is measured with provenance, complete
    cold = driver.home_cold_cache(host_home, ["alpha", "Alpha"])
    assert cold["partial"] is False and cold["skipped"] == {} and cold["total_gib"] == 4.0
    assert cold["dirs"] == {str(host_home / ".cache" / "alpha"): 4.0}
    assert cold["resolved"] == {str(host_home / ".cache" / "alpha"):
                                {"resolved_from": str(host_home / ".cache" / "alpha"), "resolved_to": str(elsewhere)}}
    assert "recorded targets" in cold["basis"] and "PARTIAL" not in cold["basis"]
    # a dangling link and a link to a file are skipped, and the answer is partial
    dangling_home = tmp_path / "host_home_dangling"
    (dangling_home / ".cache").mkdir(parents=True)
    os.symlink(tmp_path / "nowhere", dangling_home / ".cache" / "alpha")
    (tmp_path / "afile").write_text("x")
    os.symlink(tmp_path / "afile", dangling_home / ".cache" / "beta")
    cold = driver.home_cold_cache(dangling_home, ["alpha", "beta"])
    assert cold["partial"] is True and cold["dirs"] == {} and cold["resolved"] == {} and "PARTIAL" in cold["basis"]
    assert cold["skipped"][str(dangling_home / ".cache" / "alpha")].startswith("symlink_unresolvable:")
    assert cold["skipped"][str(dangling_home / ".cache" / "beta")] == f"symlink_not_a_dir:{tmp_path / 'afile'}"
    # unreadable subtree
    unreadable_home = tmp_path / "host_home2"
    sub = unreadable_home / ".cache" / "alpha" / "locked"
    sub.mkdir(parents=True)
    (sub / "w.bin").write_bytes(b"\0" * 1024)
    (unreadable_home / ".cache" / "alpha" / "seen.bin").write_bytes(b"\0" * 1024)
    os.chmod(sub, 0)
    try:
        cold = driver.home_cold_cache(unreadable_home, ["alpha"])
    finally:
        os.chmod(sub, 0o755)
    assert cold["partial"] is True and cold["dirs"] == {}
    (reason,) = cold["skipped"].values()
    assert reason.startswith("unreadable:") and "locked" in reason
    # symlinked .cache root: resolved once, then the known names under the target
    linked_home = tmp_path / "host_home3"
    linked_home.mkdir()
    cache_target = tmp_path / "cache_target"
    (cache_target / "alpha").mkdir(parents=True)
    (cache_target / "alpha" / "w.bin").write_bytes(b"\0" * 2048)
    os.symlink(cache_target, linked_home / ".cache")
    cold = driver.home_cold_cache(linked_home, ["alpha"])
    assert cold["partial"] is False and cold["skipped"] == {}
    assert cold["dirs"] == {str(linked_home / ".cache" / "alpha"): round(2048 / driver.GIB, 3)}
    assert cold["resolved"] == {str(linked_home / ".cache"):
                                {"resolved_from": str(linked_home / ".cache"), "resolved_to": str(cache_target)}}
    dangling_root = tmp_path / "host_home5"
    dangling_root.mkdir()
    os.symlink(tmp_path / "nowhere", dangling_root / ".cache")
    cold = driver.home_cold_cache(dangling_root, ["alpha"])
    assert cold["partial"] is True and list(cold["skipped"]) == [str(dangling_root / ".cache")]
    # absent cache dir: a complete answer of zero (nothing to re-download)
    empty_home = tmp_path / "host_home4"
    empty_home.mkdir()
    cold = driver.home_cold_cache(empty_home, ["alpha"])
    assert cold["partial"] is False and cold["skipped"] == {} and cold["total_gib"] == 0.0
    # in a cycle: partial (dangling link) => resource-blocked before anything is materialized
    hub, ledger, rows = _campaign(tmp_path, monkeypatch, ["alpha"], host_home=dangling_home,
                                  disk_free=lambda p: int(100 * driver.GIB))
    row = next(r for r in rows if r.get("phase") == "budget")
    assert row["state"] == "resource-blocked" and "partial measurement is not a budget" in row["stderr_tail"]
    ev = row["evidence"]
    assert ev["home_cold_cache_gib"] is None and ev["home_cold_cache_provenance"] == "unknown(partial_measurement)"
    assert ev["gate_gib"] is None and ev["floor_gate_gib"] is None and "symlink" in json.dumps(ev["home_cold_cache"]["skipped"])
    assert set(_terminal(rows).values()) == {"resource-blocked"}
    assert not (tmp_path / ".omm_fresh").exists() or not any((tmp_path / ".omm_fresh").iterdir())
    # the operator's explicit conservative estimate stands in, with provenance, and enters the gates
    (tmp_path / "second").mkdir()
    hub, ledger, rows = _campaign(tmp_path / "second", monkeypatch, ["alpha"], host_home=dangling_home,
                                  cold_cache_gib=0.3, min_free_gib=0.4, disk_free=lambda p: int(0.65 * driver.GIB))
    row = next(r for r in rows if r.get("phase") == "skipped_disk")
    ev = row["evidence"]
    assert row["state"] == "resource-blocked" and ev["home_cold_cache_gib"] == 0.3
    assert ev["home_cold_cache_provenance"] == "operator_estimate" and round(ev["floor_gate_gib"], 3) == 0.7
    assert ev["home_cold_cache"]["partial"] is True  # the measurement itself is still reported as partial
    (tmp_path / "third").mkdir()
    hub, ledger, rows = _campaign(tmp_path / "third", monkeypatch, ["alpha"], host_home=dangling_home,
                                  cold_cache_gib=0.3, min_free_gib=0.4, disk_free=lambda p: int(1.0 * driver.GIB))
    assert _terminal(rows)[("alpha", "inference", "A-1")] == "passed"
    assert next(r for r in rows if r.get("phase") == "budget")["evidence"]["home_cold_cache_provenance"] == "operator_estimate"
    # a COMPLETE measurement (the resolvable link: 4 GiB) is never replaced by the operator's figure
    (tmp_path / "fourth").mkdir()
    hub, ledger, rows = _campaign(tmp_path / "fourth", monkeypatch, ["alpha"], host_home=host_home,
                                  cold_cache_gib=0.3, min_free_gib=0.4, disk_free=lambda p: int(1.0 * driver.GIB))
    row = next(r for r in rows if r.get("phase") == "budget")
    ev = row["evidence"]
    assert row["state"] == "resource-blocked" and ev["home_cold_cache_gib"] == 4.0 and ev["gate_gib"] == 4.0
    assert ev["home_cold_cache_provenance"] == "measured_host_home" and ev["home_cold_cache"]["partial"] is False


def test_first_run_ft_downloads_are_budgeted_with_provenance_or_block_the_ft_phase(tmp_path, monkeypatch):
    """Artifacts a family's first fine-tune run fetches into the root are
    budgeted from MEASURED host files (provenance recorded); an item with no
    measurable stand-in is `unknown`, and unknown is never a number: the
    fine-tune phase is resource-blocked (inference still runs) unless the
    operator gives an explicit conservative --ft-download-gib."""
    host_home = tmp_path / "host_home"
    probe = host_home / ".alpha_cache"
    probe.mkdir(parents=True)
    with (probe / "pkg.zip").open("wb") as fh:
        fh.truncate(128 * 1024 * 1024)  # 0.125 GiB, sparse (exact at the 3-decimal ledger rounding)
    monkeypatch.setattr(driver, "FIRST_RUN_FT_DOWNLOADS", {
        "Alpha": [{"name": "alpha package (.zip)", "host_probe": ".alpha_cache"},
                  {"name": "alpha fine-tune checkpoint (no host stand-in)", "host_probe": None},
                  {"name": "alpha aux (probe absent)", "host_probe": ".no_such_dir"}]})
    # pure function
    dl = driver.first_run_ft_downloads("Alpha", host_home)
    assert dl["known_gib"] == 0.125 and dl["unknown"] == ["alpha fine-tune checkpoint (no host stand-in)",
                                                          "alpha aux (probe absent)"]
    by_name = {i["name"]: i for i in dl["items"]}
    assert by_name["alpha package (.zip)"]["provenance"] == f"measured_host:{probe}"
    assert by_name["alpha fine-tune checkpoint (no host stand-in)"] == {
        "name": "alpha fine-tune checkpoint (no host stand-in)", "gib": None, "provenance": "unknown", "copies": 1}
    assert "absent" in by_name["alpha aux (probe absent)"]["provenance"]
    assert driver.first_run_ft_downloads("Beta", host_home) == {"unit": "GiB", "items": [], "known_gib": 0.0,
                                                                "unknown": [], "basis": dl["basis"]}
    # a symlinked probe is not followed
    os.symlink(probe, host_home / ".linked")
    monkeypatch.setitem(driver.FIRST_RUN_FT_DOWNLOADS, "Beta", [{"name": "b", "host_probe": ".linked"}])
    assert "symlink" in driver.first_run_ft_downloads("Beta", host_home)["items"][0]["provenance"]
    # in a cycle: unknown item + no estimate => FT rows resource-blocked, inference passes,
    # the known part is still in the budget
    hub, ledger, rows = _campaign(tmp_path, monkeypatch, ["alpha"], host_home=host_home,
                                  disk_free=lambda p: int(100 * driver.GIB))
    term = _terminal(rows)
    assert term[("alpha", "inference", "A-1")] == "passed"
    assert term[("alpha", "finetune", "A-1")] == "resource-blocked" and term[("alpha", "finetune", "A-2")] == "resource-blocked"
    ft_row = next(r for r in rows if r.get("kind") == "finetune" and r.get("variant") == "A-1")
    assert "download size unknown" in ft_row["stderr_tail"] and "no host stand-in" in ft_row["stderr_tail"]
    assert ft_row["evidence"]["first_run_ft_downloads"]["unknown"] == dl["unknown"]
    bud = next(r for r in rows if r.get("phase") == "budget")["evidence"]
    assert bud["first_run_ft_downloads_gib"] == 0.125 and bud["first_run_ft_downloads_provenance"] == "partial(unknown items excluded)"
    assert "floor monitor" in bud["note"]
    # the known part enters the measured floor gate: floor 0.4 + 0.125 > 0.45 free
    (tmp_path / "second").mkdir()
    hub, ledger, rows = _campaign(tmp_path / "second", monkeypatch, ["alpha"], host_home=host_home,
                                  min_free_gib=0.4, disk_free=lambda p: int(0.45 * driver.GIB))
    row = next(r for r in rows if r.get("phase") == "skipped_disk")
    assert row["state"] == "resource-blocked" and round(row["evidence"]["floor_gate_gib"], 3) == 0.525
    assert "first-run FT downloads 0.12" in row["stderr_tail"]
    # the operator's explicit estimate for the unknown items unblocks FT and is added to the gates
    (tmp_path / "third").mkdir()
    hub, ledger, rows = _campaign(tmp_path / "third", monkeypatch, ["alpha"], host_home=host_home,
                                  ft_download_gib=0.5, disk_free=lambda p: int(100 * driver.GIB))
    term = _terminal(rows)
    assert term[("alpha", "finetune", "A-1")] == "passed"
    bud = next(r for r in rows if r.get("phase") == "budget")["evidence"]
    assert bud["first_run_ft_downloads_gib"] == 0.625 and bud["first_run_ft_downloads_provenance"] == "measured_host+operator_estimate"
    # no --ft-dataset: no fine-tune phase, nothing to budget for it
    (tmp_path / "fourth").mkdir()
    hub, ledger, rows = _campaign(tmp_path / "fourth", monkeypatch, ["alpha"], host_home=host_home, ft_dataset=None,
                                  disk_free=lambda p: int(100 * driver.GIB))
    bud = next(r for r in rows if r.get("phase") == "budget")["evidence"]
    assert bud["first_run_ft_downloads_gib"] == 0.0 and "no fine-tune phase" in bud["first_run_ft_downloads"]["basis"]
    # `copies`: the measured stand-in counts once per copy the first run persists (never a guess)
    monkeypatch.setitem(driver.FIRST_RUN_FT_DOWNLOADS, "Gamma",
                        [{"name": "g package x2", "host_probe": ".alpha_cache", "copies": 2}])
    g = driver.first_run_ft_downloads("Gamma", host_home)
    assert g["known_gib"] == 0.25 and g["unknown"] == []
    assert g["items"][0]["copies"] == 2 and g["items"][0]["measured_gib"] == 0.125
    assert g["items"][0]["provenance"] == f"measured_host:{probe} x2"
    # --ft-download-gib is ADDITIVE for every family: a family with no unknown item still gets the
    # operator figure added to its measured part (never replacing it), recorded with its semantics
    monkeypatch.setitem(driver.FIRST_RUN_FT_DOWNLOADS, "Alpha",
                        [{"name": "alpha package (.zip)", "host_probe": ".alpha_cache"}])
    (tmp_path / "fifth").mkdir()
    hub, ledger, rows = _campaign(tmp_path / "fifth", monkeypatch, ["alpha"], host_home=host_home,
                                  ft_download_gib=0.5, disk_free=lambda p: int(100 * driver.GIB))
    bud = next(r for r in rows if r.get("phase") == "budget")["evidence"]
    assert bud["first_run_ft_downloads_gib"] == 0.625 and bud["first_run_ft_downloads_provenance"] == "measured_host+operator_estimate"
    op = bud["first_run_ft_downloads_operator"]
    assert op["gib"] == 0.5 and op["covers_unknown"] == [] and "never replaces a measurement" in op["semantics"]
    assert "--peak-gib does not cover downloads" in op["semantics"]
    assert _terminal(rows)[("alpha", "finetune", "A-1")] == "passed"
    # without the figure, an unknown item is NEVER zero: the FT phase stays blocked and says so
    monkeypatch.setitem(driver.FIRST_RUN_FT_DOWNLOADS, "Alpha",
                        [{"name": "alpha audit not completed -- unknown, not zero", "host_probe": None}])
    (tmp_path / "sixth").mkdir()
    hub, ledger, rows = _campaign(tmp_path / "sixth", monkeypatch, ["alpha"], host_home=host_home,
                                  disk_free=lambda p: int(100 * driver.GIB))
    ft_row = next(r for r in rows if r.get("kind") == "finetune" and r.get("variant") == "A-1")
    assert ft_row["state"] == "resource-blocked" and "unknown is never zero" in ft_row["stderr_tail"]
    bud = next(r for r in rows if r.get("phase") == "budget")["evidence"]
    assert bud["first_run_ft_downloads_operator"] == {"gib": None, "covers_unknown": [],
                                                      "semantics": bud["first_run_ft_downloads_operator"]["semantics"]}
    # the REAL table (restored): deterministic host paths only, never a search; the FT owner's
    # 2026-09-14 audit -- MatterSim/TACE measured, NequIP/TACE persisted twice, the four
    # unaudited families and GRACE/Allegro listed as unknown (never zero)
    monkeypatch.undo()
    real = driver.FIRST_RUN_FT_DOWNLOADS
    for fam, items in real.items():
        for it in items:
            assert it["host_probe"] is None or (not it["host_probe"].startswith("/") and "*" not in it["host_probe"])
    assert set(real) == {"GRACE", "NequIP", "Allegro", "MatterSim", "TACE", "PET", "MACE", "DeePMD", "DPA4"}
    probes = {fam: [(it["host_probe"], it.get("copies", 1)) for it in items] for fam, items in real.items()}
    assert probes["NequIP"] == [(".nequip/model_cache", 2)] and probes["TACE"] == [(".cache/tace", 2)]
    assert probes["MatterSim"] == [(".local/mattersim/pretrained_models", 1)]
    for fam in ("GRACE", "Allegro", "PET", "MACE", "DeePMD", "DPA4"):
        assert probes[fam] == [(None, 1)], fam
        assert driver.first_run_ft_downloads(fam, tmp_path / "empty_home")["unknown"], fam
    for fam in ("PET", "MACE", "DeePMD", "DPA4"):
        assert "unknown, not zero" in real[fam][0]["name"]


FAKE_FT_RUN_BUNDLE = """
variant="$1"; shift
while [ $# -gt 0 ]; do case "$1" in --out) out="$2"; shift 2;; *) shift;; esac; done
printf '#!/bin/sh\\n' > "$out/finetune_$variant.sh"
mkdir -p "$out/seed/0/final_model/variables"
echo pb > "$out/seed/0/final_model/saved_model.pb"
echo idx > "$out/seed/0/final_model/variables/variables.index"
echo data > "$out/seed/0/final_model/variables/variables.data-00000-of-00001"
mkdir -p "$out/seed/0/final_model/assets"
echo '{"schema": "ft_run.json/1"}' > "$out/ft_run.json"
exit 0
"""


def test_saved_model_bundle_is_preserved_whole_not_the_pb_alone(tmp_path, monkeypatch):
    """A checkpoint that is one file of a SavedModel bundle (GRACE) is
    discovered through the shipped ft_run.py's own glob table, hashed as a
    bundle, and EVERY bundle file is a required preserved artifact."""
    hub = _fresh_hub(tmp_path)
    (hub / "scripts" / "ft_run.py").write_text(
        'FAMILY_CHECKPOINT_GLOBS = {\n    "Alpha": ["seed/*/final_model/saved_model.pb"],\n}\n')
    _git(hub, "add", "-A")
    _git(hub, "commit", "-q", "-m", "ft_run globs")
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"patterns": []}))
    hooks = _hooks(tmp_path, monkeypatch,
                   ft_extra_args=["--ft-run-cmd", _script(tmp_path / "fake_ft_run_bundle.sh", FAKE_FT_RUN_BUNDLE),
                                  "--ft-verify-cmd", _script(tmp_path / "fake_ft_verify.sh", FAKE_FT_VERIFY)])
    ledger = driver.fresh_sweep(["alpha"], hub, tmp_path / ".omm_fresh", "cb", allow, hooks)
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines() if ln.strip()]
    assert _terminal(rows)[("alpha", "finetune", "A-1")] == "passed"
    ft = next(r for r in rows if r.get("kind") == "finetune" and r.get("variant") == "A-1")
    ev = ft["evidence"]
    assert ev["ckpt"].endswith("seed/0/final_model/saved_model.pb")
    bundle = ev["ckpt_bundle"]
    assert bundle["dir"].endswith("seed/0/final_model") and bundle["file_count"] == 3
    assert {Path(f["path"]).name for f in bundle["files"]} == {"saved_model.pb", "variables.index",
                                                              "variables.data-00000-of-00001"}
    assert all(len(f["sha256"]) == 64 and f["bytes"] > 0 for f in bundle["files"])
    pres = next(r for r in rows if r.get("phase") == "preserve")
    assert pres["state"] == "passed"
    required = set(pres["evidence"]["required_artifacts"])
    assert {f["path"] for f in bundle["files"]} <= required
    copies = [Path(p) for p in pres["evidence"]["files"]]
    assert {c.name for c in copies} >= {"saved_model.pb", "variables.index", "variables.data-00000-of-00001"}
    assert all(c.is_file() for c in copies)
    # the bundle's directories are mirrored under the copy, the EMPTY assets/ included
    dest = Path(pres["evidence"]["dest"])
    mirrored = {Path(d).relative_to(dest).as_posix().split("seed/0/final_model", 1)[-1] for d in pres["evidence"]["bundle_dirs"]}
    assert mirrored == {"", "/variables", "/assets"}
    assert {Path(d).name for d in bundle["dirs"]} == {"variables", "assets"}
    copied_bundle = next(c for c in copies if c.name == "saved_model.pb").parent
    assert (copied_bundle / "assets").is_dir() and not any((copied_bundle / "assets").iterdir())
    assert (copied_bundle / "variables" / "variables.index").is_file()
    assert _guards(rows, "alpha")["cleanup"] == "cleaned"


RAW_SECRETS = ("user:s3cret@", "token=abc123", FAKE_HF_TOKEN)


def _files_holding(root: Path, needles: tuple, skip: set) -> list[str]:
    hits = []
    for p in root.rglob("*"):
        if not p.is_file() or p.is_symlink() or p.name in skip:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(n in text for n in needles):
            hits.append(str(p))
    return hits


def test_dependency_inventory_is_scrubbed_before_persistence_and_required_for_cleanup(tmp_path, monkeypatch):
    """After install, the owned env's conda/pip inventories and the registry
    resolve() answers are captured into <root>/.sweep/inventory/ as REQUIRED
    preserved artifacts. Credential shapes (URL userinfo, token query
    parameters, HF-shaped tokens) never reach disk: not the inventory files,
    not the phase logs, not the ledger. A malformed inventory fails the phase
    and, being required, keeps preserve -> cleanup closed; a disk floor
    reached during the inventory aborts the cycle like any other phase."""
    hub, ledger, rows = _campaign(tmp_path, monkeypatch, ["alpha"])
    inv = next(r for r in rows if r.get("phase") == "inventory")
    assert inv["state"] == "passed" and inv["returncode"] == 0
    ev = inv["evidence"]
    assert set(ev["files"]) == {"conda_explicit", "pip_freeze", "pip_list", "resolve"}
    assert ev["prefix"].endswith("/envs/alpha") and ev["owned_python"].endswith("/envs/alpha/bin/python")
    assert "dependency-replay" in ev["note"]
    pres = next(r for r in rows if r.get("phase") == "preserve")
    assert pres["state"] == "passed"
    required = set(pres["evidence"]["required_artifacts"])
    assert {f["path"] for f in ev["files"].values()} <= required
    copies = {Path(p).name: Path(p) for p in pres["evidence"]["files"]}
    assert {"alpha.conda_explicit.txt", "alpha.pip_freeze.txt", "alpha.pip_list.json", "alpha.resolve.json"} <= set(copies)
    assert "https://***:***@conda.example/pkgs/alpha-1.0-0.tar.bz2" in copies["alpha.conda_explicit.txt"].read_text()
    assert "beta @ https://x.example/beta.whl?token=***" in copies["alpha.pip_freeze.txt"].read_text()
    assert json.loads(copies["alpha.pip_list.json"].read_text()) == [{"name": "alpha", "version": "1.0"}]
    resolve = json.loads(copies["alpha.resolve.json"].read_text())
    assert resolve["isolation_state"] == "passed" and resolve["python"].endswith("/envs/alpha/bin/python")
    # the registry's resolve() answers for EVERY variant of the runtime copy, under the owned interpreter
    assert {(e["family"], e["version"]) for e in resolve["entries"]} == {("Alpha", "A-1"), ("Alpha", "A-2"), ("Beta", "B-1")}
    assert "resolve()" in resolve["probe"] and all("python" in e for e in resolve["entries"])
    # the scrubbed HF-shaped token went to the fake tool's stderr: the phase log holds the scrubbed form
    err_logs = [p for p in copies.values() if p.name.startswith("inventory.pip_list") and p.name.endswith(".err.log")]
    assert err_logs and "# token hf_***" in err_logs[0].read_text()
    # nothing under the campaign tree holds a raw secret (the fake tool's own source excepted)
    assert _files_holding(tmp_path, RAW_SECRETS, skip={"fake_inventory.sh"}) == []
    assert _guards(rows, "alpha")["cleanup"] == "cleaned"
    # the sha256 recorded for each inventory file is the scrubbed file's
    for f in ev["files"].values():
        assert len(f["sha256"]) == 64 and f["bytes"] > 0
    # a malformed inventory (pip_list not JSON) fails the phase; the missing
    # required file keeps preserve closed and the root retained
    (tmp_path / "bad").mkdir()
    hub, ledger, rows = _campaign(tmp_path / "bad", monkeypatch, ["alpha"],
                                  inventory_cmd=[_script(tmp_path / "bad" / "fake_inventory_bad.sh", FAKE_INVENTORY_MALFORMED)])
    inv = next(r for r in rows if r.get("phase") == "inventory")
    assert inv["state"] == "failed(inventory)" and "pip_list: malformed output" in inv["stderr_tail"]
    assert "pip_list" not in inv["evidence"]["files"] and "resolve" in inv["evidence"]["files"]
    g = _guards(rows, "alpha")
    assert g["preserve"] == "failed(preserve:required_missing)" and g["cleanup"] != "cleaned"
    assert _files_holding(tmp_path / "bad", RAW_SECRETS, skip={"fake_inventory.sh", "fake_inventory_bad.sh"}) == []
    # the disk floor reached while an inventory tool runs stops it and aborts the cycle
    (tmp_path / "floor").mkdir()

    def free(_p, root=tmp_path / "floor" / ".omm_fresh"):
        return int(0.1 * driver.GIB) if any(root.glob("*/envs/alpha/inv_started")) else int(100 * driver.GIB)

    hub, ledger, rows = _campaign(tmp_path / "floor", monkeypatch, ["alpha"], min_free_gib=1.0, disk_free=free,
                                  inventory_cmd=[_script(tmp_path / "floor" / "fake_inventory_slow.sh", FAKE_INVENTORY_SLOW)])
    inv = next(r for r in rows if r.get("phase") == "inventory")
    assert inv["state"] == "failed(inventory)" and "conda_explicit: aborted=disk_floor" in inv["stderr_tail"]
    assert "pip_freeze" not in inv["evidence"]["commands"] or "pip_freeze" not in json.dumps(inv["evidence"]["problems"])
    abort = next(r for r in rows if r.get("phase") == "abort")
    assert abort["state"] == "resource-blocked" and "during inventory" in abort["stderr_tail"]
    assert set(_terminal(rows).values()) == {"resource-blocked"}


FAKE_INSTALL_DROPS_PROC = """
env="$1"
mkdir -p "$PWD/envs/$env/bin" && echo "" > "$PWD/envs/$env/bin/python"
rm -rf "$FAKE_PROC_ROOT"
exit 0
"""


def test_unreadable_proc_at_cleanup_refuses_and_retains_the_root(tmp_path, monkeypatch):
    """The quiescence scan fails closed: when the proc root cannot be read at
    cleanup time the root is retained (failed(cleanup:proc_unreadable)),
    evidence having been preserved as usual."""
    monkeypatch.setenv("FAKE_PROC_ROOT", str(tmp_path / "proc"))
    hub, ledger, rows = _campaign(tmp_path, monkeypatch, ["alpha"],
                                  install_cmd=[_script(tmp_path / "fake_install_drop.sh", FAKE_INSTALL_DROPS_PROC)])
    g = _guards(rows, "alpha")
    assert g["install"] == "passed" and g["preserve"] == "passed"
    assert g["cleanup"] == "failed(cleanup:proc_unreadable)" and g["post_cleanup"] == "failed(cleanup:incomplete)"
    roots = list((tmp_path / ".omm_fresh").iterdir())
    assert len(roots) == 1 and (roots[0] / fresh_root.OWNERSHIP_FILE).is_file()
    # the launched phase sessions were handed to the scan (recorded in the refusal's context)
    cleanup_row = next(r for r in rows if r.get("phase") == "cleanup")
    assert json.loads(cleanup_row["stderr_tail"])["proc_root"] == str(tmp_path / "proc")
    # and the campaign start itself fails closed on an unreadable proc root
    (tmp_path / "second").mkdir()
    hub2 = _fresh_hub(tmp_path / "second")
    allow = tmp_path / "second" / "allow.json"
    allow.write_text(json.dumps({"patterns": []}))
    hooks = _hooks(tmp_path / "second", monkeypatch, proc_root=tmp_path / "second" / "no_proc")
    ledger2 = driver.fresh_sweep(["alpha"], hub2, tmp_path / "second" / ".omm_fresh", "c9", allow, hooks)
    rows2 = [json.loads(ln) for ln in ledger2.read_text().splitlines()]
    assert rows2[-1]["phase"] == "concurrency" and rows2[-1]["state"] == "failed(concurrency:proc_unreadable)"
    assert not (tmp_path / "second" / ".omm_fresh").exists()


def test_measured_disk_floor_mid_phase_stops_the_child_and_still_preserves(tmp_path, monkeypatch):
    """Free space is re-measured WHILE a child runs: when it drops below the
    floor the child's process group is stopped in order, the remaining
    variants are resource-blocked, and the preserve -> cleanup tail runs."""
    hub = _fresh_hub(tmp_path)
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"patterns": []}))
    # an install that would run "forever" (a sleeping subprocess in its group)
    # the sleeper's pid is written into the root (preserved as .txt) so the
    # test can assert on OUR child, not on any other sleep on the host
    slow = _script(tmp_path / "slow_install.sh",
                   'echo "installing $1"; mkdir -p "$PWD/envs/$1/bin" "$PWD/.sweep"; '
                   'sleep 30 & echo $! > "$PWD/.sweep/child_pid.txt"; wait')
    calls = {"n": 0}

    def filling_disk(_path):
        calls["n"] += 1
        return 100 * driver.GIB if calls["n"] < 8 else 1 * driver.GIB  # falls below the 2 GiB floor while install runs
    hooks = _hooks(tmp_path, monkeypatch, install_cmd=[slow], min_free_gib=2.0, disk_free=filling_disk,
                   grace_seconds=1.0)
    ledger = driver.fresh_sweep(["alpha"], hub, tmp_path / ".omm_fresh", "c5", allow, hooks)
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    g = _guards(rows, "alpha")
    install = next(r for r in rows if r.get("phase") == "install")
    assert install["state"] == "failed(install)" and install["evidence"]["aborted"] == "disk_floor"
    assert install["evidence"]["min_free_gib"] == 1.0 and install["returncode"] != 0
    assert g["abort"] == "resource-blocked" and "child group" in next(r for r in rows if r.get("phase") == "abort")["stderr_tail"]
    assert set(_terminal(rows).values()) == {"resource-blocked"}
    assert not any(r.get("phase") == "verify" for r in rows)
    # the child's stdout was streamed to a file that is preserved
    assert g["preserve"] == "passed" and g["cleanup"] == "cleaned" and g["post_cleanup"] == "passed"
    kept = next(r for r in rows if r.get("phase") == "preserve")["evidence"]["files"]
    log = next(f for f in kept if f.endswith("install.out.log"))
    assert Path(log).read_text().startswith("installing alpha")
    assert not any((tmp_path / ".omm_fresh").iterdir())
    # and the sleeping grandchild did not survive the group stop
    child_pid = int(Path(next(f for f in kept if f.endswith("child_pid.txt"))).read_text().strip())
    assert not Path(f"/proc/{child_pid}").exists() or "zombie" in Path(f"/proc/{child_pid}/status").read_text().lower()


def test_stop_signal_preserves_retains_root_and_stops_campaign(tmp_path, monkeypatch):
    hub = _fresh_hub(tmp_path)
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"patterns": []}))
    slow = _script(tmp_path / "slow_install.sh", 'mkdir -p "$PWD/envs/$1/bin"; sleep 30 & wait')
    hooks = _hooks(tmp_path, monkeypatch, install_cmd=[slow], grace_seconds=1.0)
    calls = {"n": 0}

    def tick(_path):  # the driver's cancel flag flips while install runs (as SIGTERM would set it)
        calls["n"] += 1
        if calls["n"] == 5:
            hooks.cancel.set()
        return 100 * driver.GIB
    hooks.disk_free = tick
    ledger = driver.fresh_sweep(["alpha", "beta"], hub, tmp_path / ".omm_fresh", "c6", allow, hooks)
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    g = _guards(rows, "alpha")
    assert g["abort"] == "terminated" and g["preserve"] == "passed" and g["cleanup"] == "retained"
    assert rows[-1]["phase"] == "stop" and rows[-1]["state"] == "terminated"
    assert not any(r.get("env") == "beta" for r in rows)  # campaign stopped, beta never started
    roots = list((tmp_path / ".omm_fresh").iterdir())
    assert len(roots) == 1 and (roots[0] / fresh_root.OWNERSHIP_FILE).is_file()
    assert not (hub / ".sweep" / driver.LOCK_NAME).exists()


def test_stale_lock_from_a_dead_holder_is_replaced(tmp_path, monkeypatch):
    hub = _fresh_hub(tmp_path)
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"patterns": []}))
    hooks = _hooks(tmp_path, monkeypatch)
    (hub / ".sweep").mkdir(exist_ok=True)
    (hub / ".sweep" / driver.LOCK_NAME).write_text(json.dumps({"pid": 999999, "campaign_id": "dead"}))
    ledger = driver.fresh_sweep(["alpha"], hub, tmp_path / ".omm_fresh", "c7", allow, hooks)
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    assert _terminal(rows)[("alpha", "inference", "A-1")] == "passed"
    assert not (hub / ".sweep" / driver.LOCK_NAME).exists()


def test_unapproved_allowlist_file_refuses_the_campaign(tmp_path, monkeypatch):
    hub = _fresh_hub(tmp_path)
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"patterns": ["scripts/*.py"]}))  # wider than APPROVED_NEW_PATHS
    ledger = driver.fresh_sweep(["alpha"], hub, tmp_path / ".omm_fresh", "c8", allow, _hooks(tmp_path, monkeypatch))
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    assert rows[0]["state"] == "failed(snapshot:allowlist_unapproved)" and len(rows) == 1


def test_no_cleanup_retains_root_and_ft_dataset_missing_is_honest(tmp_path, monkeypatch):
    hub, ledger, rows = _campaign(tmp_path, monkeypatch, ["alpha"], cleanup=False, ft_dataset=None)
    g = _guards(rows, "alpha")
    assert g["cleanup"] == "retained"
    roots = list((tmp_path / ".omm_fresh").iterdir())
    assert len(roots) == 1 and (roots[0] / fresh_root.OWNERSHIP_FILE).is_file()
    t = _terminal(rows)
    assert t[("alpha", "inference", "A-1")] == "passed"
    assert t[("alpha", "finetune", "A-1")] == "failed(ft_dataset_missing)"
    # the retained root is still only removable through the guarded path
    with pytest.raises(fresh_root.FreshRootError):
        fresh_root.cleanup(hub, ledger, proc_root=tmp_path / "proc")


def test_concurrency_lock_and_foreign_process_refuse_to_start(tmp_path, monkeypatch):
    hub = _fresh_hub(tmp_path)
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"patterns": []}))
    hooks = _hooks(tmp_path, monkeypatch)
    fake_pid = hooks.proc_root / "777"
    fake_pid.mkdir()
    (fake_pid / "cmdline").write_bytes(b"bash\0./install.sh\0mace\0")
    ledger = driver.fresh_sweep(["alpha"], hub, tmp_path / ".omm_fresh", "c3", allow, hooks)
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    assert rows[-1]["phase"] == "concurrency" and rows[-1]["state"] == "failed(concurrency)"
    assert rows[-1]["evidence"]["processes"][0]["pid"] == 777
    # a pid whose cmdline cannot be read but that is still present cannot be
    # vouched for (failed(concurrency:proc_uninspectable)); one that vanished
    # between listing and reading (dangling entry) exited and is ignored
    (fake_pid / "cmdline").unlink()
    os.symlink(hooks.proc_root / "gone", hooks.proc_root / "778")
    ledger = driver.fresh_sweep(["alpha"], hub, tmp_path / ".omm_fresh", "c3b", allow, hooks)
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    assert rows[-1]["phase"] == "concurrency" and rows[-1]["state"] == "failed(concurrency:proc_uninspectable)"
    assert json.loads(rows[-1]["stderr_tail"])["pids"] == [777]
    assert not (tmp_path / ".omm_fresh").exists()
    (fake_pid / "cmdline").write_bytes(b"sleep\0")
    assert driver.scan_foreign_processes(proc_root=hooks.proc_root) == []  # 778 vanished: ignored
    (hub / ".sweep").mkdir(exist_ok=True)
    (hub / ".sweep" / driver.LOCK_NAME).write_text(json.dumps({"pid": 777, "campaign_id": "other"}))
    ledger = driver.fresh_sweep(["alpha"], hub, tmp_path / ".omm_fresh", "c4", allow, hooks)
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    assert rows[-1]["state"] == "failed(campaign_lock)"
    assert not any(r.get("phase") == "install" for r in rows)


def test_report_cli_detects_campaign_ledger(tmp_path, monkeypatch):
    hub, ledger, rows = _campaign(tmp_path, monkeypatch, ["alpha"])
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "setup_sweep.py"), "report",
                           "--ledger", str(ledger)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          env=dict(os.environ, OH_MY_MLIP_HOME=str(hub)))
    assert proc.returncode == 0, proc.stderr
    assert "fresh-root campaign report" in proc.stdout and "A-1" in proc.stdout
