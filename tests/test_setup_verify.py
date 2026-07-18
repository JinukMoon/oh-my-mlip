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
