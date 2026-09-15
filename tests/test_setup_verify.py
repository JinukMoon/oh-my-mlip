"""Tests for scripts/setup_verify.py -- the atomic verify oracle.

Pins the verdict decision table (the whole point of the oracle: the agent
renders this, it never re-judges), the witness-JSON extraction, the registry
env lookup, and the exit-0-iff-pass contract.

GPU-free: verdicts are assembled from injected (skew, returncode, gpu_seen,
witness, stderr) tuples; no model env or GPU is touched.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location(
    "setup_verify", REPO_ROOT / "scripts" / "setup_verify.py"
)
oracle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(oracle)

NO_SKEW = {"skew": False, "env_cuda": 126, "host_cuda": 129, "reason": ""}
SKEW = {
    "skew": True, "env_cuda": 130, "host_cuda": 129,
    "reason": "env needs CUDA 13.0, host driver exposes CUDA 12.9",
}
WITNESS = {"energy_ev": -16.386652, "fmax_ev_a": 0.031, "forces_shape": [4, 3]}


def test_verdict_gpu_pass():
    got = oracle.decide_verdict(NO_SKEW, 0, True, WITNESS, "")
    assert got["pass"] is True
    assert got["device"] == "cuda" and got["degraded"] is False
    assert got["gpu_pid_confirmed"] is True
    assert got["energy_ev"] == WITNESS["energy_ev"]


def test_verdict_gpu_not_used_is_fail():
    got = oracle.decide_verdict(NO_SKEW, 0, False, WITNESS, "")
    assert got["pass"] is False
    assert got["reason"] == "gpu_not_used"


def test_verdict_gpu_mem_witness_passes_without_pid():
    # Some hosts' drivers may hide compute-app PIDs; the system still passes via
    # the worker's realized CUDA allocation when memory usage is confirmed.
    w = dict(WITNESS, gpu_mem_allocated_bytes=200_000_000)
    got = oracle.decide_verdict(NO_SKEW, 0, False, w, "")
    assert got["pass"] is True
    assert got["gpu_pid_confirmed"] is False
    assert got["gpu_mem_bytes"] == 200_000_000


def test_verdict_zero_mem_and_no_pid_fails():
    w = dict(WITNESS, gpu_mem_allocated_bytes=0)
    got = oracle.decide_verdict(NO_SKEW, 0, False, w, "")
    assert got["pass"] is False and got["reason"] == "gpu_not_used"


def test_verdict_skew_cpu_is_degraded_pass():
    got = oracle.decide_verdict(SKEW, 0, False, WITNESS, "")
    assert got["pass"] is True
    assert got["device"] == "cpu" and got["degraded"] is True
    assert got["reason"] == SKEW["reason"]  # computed by predicate, not scraped


def test_verdict_nonzero_exit_is_plain_fail_with_normalized_tail():
    stderr = 'Traceback:\n  File "/home/user/x.py", line 42, in f\nRuntimeError: boom pid 12345\n'
    got = oracle.decide_verdict(NO_SKEW, 1, False, WITNESS, stderr)
    assert got["pass"] is False
    # Normalization comes from setup_guardrail (paths/line-numbers/pids scrubbed).
    assert "/home/user" not in got["reason"] and "12345" not in got["reason"]
    assert "runtimeerror: boom" in got["reason"]  # normalization also lowercases


def test_verdict_missing_witness_is_fail():
    got = oracle.decide_verdict(NO_SKEW, 0, True, None, "")
    assert got["pass"] is False
    assert got["reason"] == "witness_json_missing"


def test_parse_witness_json_takes_last_valid_object():
    stdout = (
        "worker noise {not json}\n"
        '{"other": 1}\n'
        '{"energy_ev": -1.5, "fmax_ev_a": 0.1, "forces_shape": [4, 3]}\n'
    )
    assert oracle.parse_witness_json(stdout)["energy_ev"] == -1.5
    assert oracle.parse_witness_json("no json at all") is None


def test_find_env_accepts_family_and_version_names(tmp_path):
    (tmp_path / "models.json").write_text(json.dumps({
        "_meta": {},
        "MACE": {"env": "mace", "versions": {"MACE-MPA-0": {}}},
    }))
    assert oracle.find_env("mace", tmp_path) == "mace"
    assert oracle.find_env("MACE-MPA-0", tmp_path) == "mace"
    assert oracle.find_env("nope", tmp_path) is None


FAKE_SINGLE_POINT = '''
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--version"); ap.add_argument("--device", default="cuda")
ap.add_argument("--json", action="store_true"); ap.add_argument("--structure")
a = ap.parse_args()
print(json.dumps({"energy_ev": -16.4, "fmax_ev_a": 0.03, "forces_shape": [4, 3],
                  "gpu_mem_allocated_bytes": 123456, "version": a.version}))
'''


def _fake_home(tmp_path: Path, link_package: bool) -> Path:
    """A hub root with a fake run_examples/single_point.py that always
    witnesses a GPU allocation; keyed as MACE so the real registry (read from
    the package's own models.json) can resolve it when the package is linked."""
    (tmp_path / "models.json").write_text(json.dumps({
        "_meta": {},
        "MACE": {"env": "mace", "python": "${OH_MY_MLIP_HOME}/envs/mace/bin/python",
                 "versions": {"MACE-MPA-0": {}, "MACE-MH-1-OMAT": {}}},
    }))
    (tmp_path / "envs").mkdir()
    (tmp_path / "envs" / "mace.yml").write_text("name: mace\n")  # no +cuNNN pin => no skew
    (tmp_path / "run_examples").mkdir()
    (tmp_path / "run_examples" / "single_point.py").write_text(FAKE_SINGLE_POINT)
    if link_package:
        (tmp_path / "oh_my_mlip").symlink_to(REPO_ROOT / "oh_my_mlip")
    return tmp_path


def _run_verify(home: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "setup_verify.py"), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(home),
        env={**__import__("os").environ, "OH_MY_MLIP_HOME": str(home)},
    )


def test_no_local_record_pass_writes_no_models_local_json(tmp_path):
    """G3b regression witness: a PASS under --no-local-record leaves the hub's
    local state byte-for-byte untouched (no models.local.json appears)."""
    home = _fake_home(tmp_path, link_package=True)
    proc = _run_verify(home, "MACE", "--json", "--no-local-record")
    assert proc.returncode == 0, proc.stderr
    verdict = json.loads(proc.stdout.strip().splitlines()[-1])
    assert verdict["pass"] is True and verdict["local_record"] == "skipped(--no-local-record)"
    assert not (home / "models.local.json").exists()


def test_without_flag_pass_records_models_local_json(tmp_path):
    """Contrast for the test above: the default path still materializes."""
    home = _fake_home(tmp_path, link_package=True)
    proc = _run_verify(home, "MACE", "--json")
    assert proc.returncode == 0, proc.stderr
    verdict = json.loads(proc.stdout.strip().splitlines()[-1])
    assert verdict["local_record"] == "recorded", verdict
    assert "MACE-MPA-0" in json.loads((home / "models.local.json").read_text())


def test_all_variants_emits_one_verdict_per_variant_and_summary(tmp_path):
    home = _fake_home(tmp_path, link_package=False)
    proc = _run_verify(home, "MACE", "--all-variants", "--json", "--no-local-record")
    assert proc.returncode == 0, proc.stderr
    lines = [json.loads(ln) for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    per_variant = [ln for ln in lines if "all_variants" not in ln]
    assert [v["version"] for v in per_variant] == ["MACE-MPA-0", "MACE-MH-1-OMAT"]
    assert all(v["pass"] for v in per_variant)
    summary = lines[-1]
    assert summary["all_variants"] is True and summary["pass"] is True
    assert [v["version"] for v in summary["variants"]] == ["MACE-MPA-0", "MACE-MH-1-OMAT"]
    assert not (home / "models.local.json").exists()
    # per-variant logs, so one variant's stderr never overwrites another's
    assert (home / ".sweep" / "verify" / "mace.MACE-MPA-0.log").is_file()
    assert (home / ".sweep" / "verify" / "mace.MACE-MH-1-OMAT.log").is_file()


def test_all_variants_and_version_are_mutually_exclusive(tmp_path):
    home = _fake_home(tmp_path, link_package=False)
    proc = _run_verify(home, "MACE", "--all-variants", "--version", "MACE-MPA-0")
    assert proc.returncode == 2


def test_family_versions_lookup(tmp_path):
    home = _fake_home(tmp_path, link_package=False)
    assert oracle.family_versions("mace", home) == ["MACE-MPA-0", "MACE-MH-1-OMAT"]
    assert oracle.family_versions("MACE-MH-1-OMAT", home) == ["MACE-MPA-0", "MACE-MH-1-OMAT"]
    assert oracle.family_versions("nope", home) == []


def test_exit_code_contract_unknown_model():
    # exit-0-iff-pass: an unknown model must exit nonzero with a JSON verdict.
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "setup_verify.py"),
         "definitely-not-a-model", "--json"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 1
    verdict = json.loads(proc.stdout.strip().splitlines()[-1])
    assert verdict["pass"] is False
    assert "unknown model" in verdict["reason"]
