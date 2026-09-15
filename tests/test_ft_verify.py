"""GPU-free tests for scripts/ft_verify.py's contract with scripts/ft_sweep.py.

  * `--list-loaders --json` prints one JSON line naming the reloadable
    families (ft_sweep's first capability probe) and agrees with the public
    `loader_families()`;
  * every loader template only BINDS `atoms` + calculator; the E/F forward
    runs inside the backend witness tail (TorchDispatchMode for torch
    families, TF device-placement log for GRACE) -- never behind a
    memory-allocation proxy and never a constant;
  * `judge()` fails closed on every witness shape the sweep must not trust:
    gpu_used without a matching compute-op count, a missing execution record
    (the old allocator-style line), non-finite or mis-shaped forces;
  * `verify()` pins the EXACT variant (`version=`) so a family name does not
    silently reload under the family default's modal (M3);
  * `--device cpu` pins CUDA_VISIBLE_DEVICES="" in the child env.

No model env is launched by default: the child interpreter is stubbed. One
opt-in probe (OMM_RUN_ENV_PROBES=1) executes the real torch witness tail in
an installed family env against a synthetic module -- no checkpoint, no
weights -- to prove the adversarial CPU-forward-plus-CUDA-allocation case
really comes out gpu_used:false.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ft_verify  # noqa: E402
from oh_my_mlip import registry as reg  # noqa: E402

FT_VERIFY = REPO_ROOT / "scripts" / "ft_verify.py"
# the real adoption map, captured before the autouse fixture neutralizes it:
# only the opt-in env probe below needs it (the adopted MACE interpreter)
_REAL_LOCAL_ENV_MAP = reg.local_env_map


@pytest.fixture(autouse=True)
def _neutralize_local_state(monkeypatch):
    monkeypatch.setattr(reg, "local_env_map", lambda *a, **k: {})
    monkeypatch.setattr(reg, "load_local_models", lambda *a, **k: {})


def test_list_loaders_json_matches_public_api():
    proc = subprocess.run([sys.executable, str(FT_VERIFY), "--list-loaders", "--json"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["loaders"] == ft_verify.loader_families()
    assert payload["loaders"] == sorted(ft_verify._LOADER_TEMPLATE)
    assert payload.get("source")


def test_loader_families_covers_the_demo_trio():
    assert {"MACE", "SevenNet", "DeePMD", "DPA4"} <= set(ft_verify.loader_families())


# ── every template: head binds, witness tail computes ────────────────────────
@pytest.mark.parametrize("family", sorted(ft_verify._LOADER_TEMPLATE))
def test_every_template_defers_the_forward_to_a_backend_witness(family):
    # a UMA checkpoint is loaded with the task it was fine-tuned for
    script = ft_verify.build_script(family, "/tmp/fake.ckpt", "cuda", task="omat" if family == "UMA" else None)
    compile(script, f"<{family}>", "exec")  # the inline child script must at least parse
    tail = ft_verify.witness_tail(family)
    assert script.endswith(tail)
    assert script.startswith("DEVICE = 'cuda'\n")
    head = script[: -len(tail)]
    # the head only binds atoms + calculator; the forward belongs to the tail
    assert "atoms.calc = " in head
    assert "get_potential_energy" not in head and "get_forces" not in head
    assert "get_potential_energy()" in tail and "get_forces()" in tail
    # the witness is the backend's own execution record, never an allocation proxy
    assert "memory_allocated" not in script and "get_memory_info" not in script
    if family == "GRACE":
        assert "set_log_device_placement(True)" in tail
        # a saved_model forward runs as one XLA cluster with no per-op placement lines;
        # the profiler's /device:GPU plane events are the execution record for that path
        assert "tf.profiler.experimental.start(_prof_dir)" in tail
        assert 'startswith("/device:GPU:")' in tail
        assert "gpu_count = sum(gpu_ops.values()) + gpu_plane_events" in tail
        assert "gpu_used = bool(_gpus) and gpu_count > 0" in tail
        assert '"backend": "tensorflow"' in tail
    elif family == "Nequix":
        # JAX: CUPTI Compute streams on a /device:GPU process in the jax.profiler trace
        assert "jax.profiler.start_trace(_prof_dir)" in tail
        assert 'startswith("/device:GPU:")' in tail and '"Compute" in' in tail
        assert "gpu_used = bool(_gpus) and compute_streams > 0" in tail
        assert '"backend": "jax"' in tail
    else:
        assert "TorchDispatchMode" in tail
        assert "gpu_used = bool(_w.cuda_ops)" in tail
        assert '"backend": "torch"' in tail
    for key in ('"energy_ev"', '"forces_shape"', '"forces_finite"', '"gpu_used": gpu_used', '"device": DEVICE'):
        assert key in tail, key
    assert "gpu_used = True\n" not in script


def test_torch_witness_excludes_allocation_and_movement_ops_only():
    """The exclusion set is what makes 'unrelated CUDA allocation next to a
    CPU forward' come out false; compute ops must not be in it."""
    tail = ft_verify._TORCH_WITNESS_TAIL
    for op in ("empty", "zeros", "ones", "copy_", "_to_copy", "fill_", "view", "detach"):
        assert f'"{op}"' in tail, op
    for op in ("mm", "addmm", "add", "mul", "sum", "tanh", "index_add_", "scatter_add"):
        assert f'"{op}"' not in tail, op


def test_loader_families_match_ft_run_builders():
    """Every family ft_run.py can build a fine-tune for must be reloadable
    here, and vice versa -- a builder without a loader would train and then
    end failed(ft_verify:no_loader) in the sweep."""
    import ft_run
    assert set(ft_verify.loader_families()) == set(ft_run.BUILDERS)
    assert set(ft_run.FAMILY_CHECKPOINT_GLOBS) == set(ft_run.BUILDERS)


def test_unknown_family_is_a_clean_refusal():
    with pytest.raises(SystemExit) as exc:
        ft_verify.build_script("NotAFamily", "/tmp/x", "cpu")
    assert "supported" in str(exc.value)


def test_sevennet_template_carries_modal_and_device():
    script = ft_verify.build_script("SevenNet", "/tmp/x.pth", "cpu", modal="mpa")
    assert "modal='mpa'" in script
    assert 'device="cpu"' in script


# ── judge(): the verdict re-derives gpu_used from the execution record ───────
def _torch_line(**over) -> dict:
    line = {"energy_ev": -14.9, "forces_shape": [4, 3], "forces_finite": True, "gpu_used": True,
            "device": "cuda",
            "witness": {"backend": "torch", "method": "TorchDispatchMode ...", "ops_total": 60,
                        "cuda_compute_ops": 18, "cuda_op_names": ["addmm", "mm"]}}
    line.update(over)
    return line


def _tf_line(**over) -> dict:
    line = {"energy_ev": -16.39, "forces_shape": [4, 3], "forces_finite": True, "gpu_used": True,
            "device": "cuda",
            "witness": {"backend": "tensorflow", "method": "log_device_placement ...",
                        "tf_gpu_devices": ["/physical_device:GPU:0"], "gpu_compute_ops": 40,
                        "cpu_compute_ops": 3, "gpu_op_names": ["MatMul"]}}
    line.update(over)
    return line


def test_judge_passes_a_consistent_cuda_witness():
    assert ft_verify.judge(_torch_line(), "cuda") == ""
    assert ft_verify.judge(_tf_line(), "cuda") == ""


def test_judge_reads_the_jax_compute_stream_count():
    line = {"energy_ev": -16.38, "forces_shape": [4, 3], "forces_finite": True, "gpu_used": True, "device": "cuda",
            "witness": {"backend": "jax", "gpu_compute_streams": 1, "jax_gpu_devices": ["cuda:0"]}}
    assert ft_verify.judge(line, "cuda") == ""
    line.update(gpu_used=False)
    line["witness"]["gpu_compute_streams"] = 0
    assert ft_verify.judge(line, "cuda") == "gpu_not_used"


def test_judge_cuda_fails_when_no_compute_op_ran_on_the_gpu():
    line = _torch_line(gpu_used=False)
    line["witness"]["cuda_compute_ops"] = 0
    assert ft_verify.judge(line, "cuda") == "gpu_not_used"
    tf = _tf_line(gpu_used=False)
    tf["witness"].update(tf_gpu_devices=[], gpu_compute_ops=0)
    assert ft_verify.judge(tf, "cuda") == "gpu_not_used"


def test_judge_cpu_accepts_a_cpu_forward():
    line = _torch_line(gpu_used=False, device="cpu")
    line["witness"]["cuda_compute_ops"] = 0
    assert ft_verify.judge(line, "cpu") == ""


def test_judge_rejects_a_gpu_used_flag_the_record_does_not_back():
    """gpu_used:true with zero recorded GPU compute ops is not evidence."""
    line = _torch_line()
    line["witness"]["cuda_compute_ops"] = 0
    assert ft_verify.judge(line, "cuda") == "witness_inconsistent"
    line = _torch_line(gpu_used=False)  # count says yes, flag says no
    assert ft_verify.judge(line, "cuda") == "witness_inconsistent"


def test_judge_rejects_the_old_allocation_style_line():
    """A line without the backend execution record (the pre-M1 shape:
    energy + gpu_used only) can no longer pass, on either device."""
    old = {"energy_ev": -1.0, "forces_shape": [4, 3], "forces_finite": True, "gpu_used": True, "device": "cuda"}
    assert ft_verify.judge(old, "cuda") == "witness_record_missing"
    assert ft_verify.judge(dict(old, device="cpu", gpu_used=False), "cpu") == "witness_record_missing"
    bad = _torch_line()
    bad["witness"] = {"backend": "torch"}  # record present, count absent
    assert ft_verify.judge(bad, "cuda") == "witness_record_missing"
    bad = _torch_line()
    bad["witness"]["backend"] = "numpy"
    assert ft_verify.judge(bad, "cuda") == "witness_record_missing"


def test_judge_checks_energy_and_forces():
    assert ft_verify.judge(_torch_line(energy_ev=float("nan")), "cuda") == "non_finite_energy"
    assert ft_verify.judge(_torch_line(energy_ev="-1.0"), "cuda") == "non_finite_energy"
    assert ft_verify.judge(_torch_line(energy_ev=True), "cuda") == "non_finite_energy"
    assert ft_verify.judge(_torch_line(forces_shape=[3, 3]), "cuda") == "bad_forces_shape"
    assert ft_verify.judge(_torch_line(forces_shape=None), "cuda") == "bad_forces_shape"
    assert ft_verify.judge(_torch_line(forces_finite=False), "cuda") == "non_finite_forces"
    line = _torch_line()
    del line["forces_finite"]
    assert ft_verify.judge(line, "cuda") == "non_finite_forces"


# ── verify(): child stubbed, verdict shape + exact-variant resolution ─────────
def _stub_child(monkeypatch, stdout: str, returncode: int = 0, captured: dict | None = None):
    class _Proc:
        def __init__(self):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    def fake_run(cmd, capture_output, text, env):
        if captured is not None:
            captured["cmd"] = cmd
            captured["env"] = env
        return _Proc()

    monkeypatch.setattr(ft_verify.subprocess, "run", fake_run)


def test_cuda_witness_without_gpu_use_fails_closed(monkeypatch):
    line = _torch_line(gpu_used=False)
    line["witness"]["cuda_compute_ops"] = 0
    _stub_child(monkeypatch, json.dumps(line))
    v = ft_verify.verify("MACE", "/tmp/x.model", "cuda")
    assert v["pass"] is False
    assert v["reason"] == "gpu_not_used"
    assert v["gpu_used"] is False and v["device"] == "cuda"
    assert v["witness"]["backend"] == "torch"


def test_cuda_witness_with_gpu_use_passes(monkeypatch):
    _stub_child(monkeypatch, json.dumps(_torch_line()))
    v = ft_verify.verify("MACE", "/tmp/x.model", "cuda")
    assert v["pass"] is True and v["gpu_used"] is True and v["device"] == "cuda"
    assert v["forces_shape"] == [4, 3] and v["forces_finite"] is True
    assert v["version"] == reg.resolve("MACE")["version"]


def test_cpu_run_pins_no_gpu_in_child_env(monkeypatch):
    captured: dict = {}
    line = _torch_line(gpu_used=False, device="cpu")
    line["witness"]["cuda_compute_ops"] = 0
    _stub_child(monkeypatch, json.dumps(line), captured=captured)
    v = ft_verify.verify("MACE", "/tmp/x.model", "cpu")
    assert v["pass"] is True
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert captured["cmd"][0] == reg.resolve("MACE")["python"]


def test_child_env_puts_env_lib_first_and_env_run_wins(tmp_path, monkeypatch):
    (tmp_path / "bin").mkdir()
    (tmp_path / "lib").mkdir()
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib/wsl/lib")
    resolved = {"python": str(tmp_path / "bin" / "python")}
    env = ft_verify._child_env(resolved, "cuda")
    assert env["LD_LIBRARY_PATH"] == f"{tmp_path / 'lib'}:/usr/lib/wsl/lib"
    env = ft_verify._child_env({**resolved, "env_run": {"LD_LIBRARY_PATH": ""}}, "cuda")
    assert env["LD_LIBRARY_PATH"] == ""


def test_non_finite_energy_fails(monkeypatch):
    _stub_child(monkeypatch, json.dumps(_torch_line(energy_ev=float("nan"))))
    v = ft_verify.verify("MACE", "/tmp/x.model", "cuda")
    assert v["pass"] is False and v["reason"] == "non_finite_energy"


def test_verify_pins_the_exact_variant_not_the_family_default(monkeypatch):
    """M3: `version=` must reach reg.resolve and the modal must come from
    THAT variant's inference line, not the family default's."""
    real_resolve = reg.resolve
    calls: list = []

    def fake_resolve(model, version=None, **kw):
        calls.append((model, version))
        r = dict(real_resolve(model, version, **kw))
        if r["version"] == "SevenNet-Omni":
            r["inference"] = ["calc = SevenNetCalculator('7net-omni', modal='omni-only')"]
        return r

    monkeypatch.setattr(reg, "resolve", fake_resolve)
    captured: dict = {}
    _stub_child(monkeypatch, json.dumps(_torch_line()), captured=captured)
    v = ft_verify.verify("SevenNet", "/tmp/x.pth", "cuda", version="SevenNet-Omni")
    assert calls == [("SevenNet", "SevenNet-Omni")]
    assert v["version"] == "SevenNet-Omni" and v["modal"] == "omni-only"
    assert "modal='omni-only'" in captured["cmd"][2]
    # and the family alone still resolves the registry default (documented, not hidden)
    v = ft_verify.verify("SevenNet", "/tmp/x.pth", "cuda")
    assert v["version"] == real_resolve("SevenNet")["version"]


def test_cli_version_flag_reaches_the_verdict(monkeypatch, capsys):
    _stub_child(monkeypatch, json.dumps(_torch_line()))
    monkeypatch.setattr(sys, "argv", ["ft_verify.py", "/tmp/x.pth", "--model", "SevenNet",
                                      "--version", "SevenNet-Omni", "--device", "cuda", "--json"])
    assert ft_verify.main() == 0
    verdict = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert verdict["version"] == "SevenNet-Omni" and verdict["pass"] is True


def test_cli_unknown_version_is_a_clean_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ft_verify.py", "/tmp/x.pth", "--model", "SevenNet",
                                      "--version", "SevenNet-NoSuchVariant", "--json"])
    assert ft_verify.main() == 2
    assert "ft_verify" in capsys.readouterr().err


# ── opt-in: the real torch witness tail in a real family env, no weights ─────
_PROBE_HEAD = '''
import json
import numpy as np
import torch

DEVICE = {device!r}
JUNK = {junk!r}


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 8)
        self.out = torch.nn.Linear(8, 1)

    def forward(self, x):
        return self.out(torch.tanh(self.lin(x))).sum()


class _Atoms:
    """Stands in for ase.Atoms + calculator: E/F via a tiny torch model."""

    def __init__(self):
        self.model = _Model().to(DEVICE)
        self.x = torch.randn(4, 3, device=DEVICE, requires_grad=True)

    def get_potential_energy(self):
        if JUNK:  # unrelated CUDA activity the witness must NOT count
            _a = torch.empty(64, 64, device="cuda")
            _b = torch.zeros(16, device="cuda")
            _c = torch.ones(8, device="cpu").to("cuda")
        return self.model(self.x).item()

    def get_forces(self):
        e = self.model(self.x)
        (g,) = torch.autograd.grad(e, self.x)
        return (-g).detach().cpu().numpy()


atoms = _Atoms()
'''


def _probe_env_python(monkeypatch) -> str | None:
    if os.environ.get("OMM_RUN_ENV_PROBES") != "1":
        return None
    monkeypatch.setattr(reg, "local_env_map", _REAL_LOCAL_ENV_MAP)  # this host's adopted env
    try:
        py = reg.resolve("MACE")["python"]
    except Exception:  # pragma: no cover -- registry without MACE
        return None
    return py if Path(py).exists() else None


@pytest.mark.parametrize("device,junk,expect", [
    ("cpu", False, False), ("cpu", True, False), ("cuda", False, True), ("cuda", True, True),
])
def test_real_torch_witness_tail_adversarial(device, junk, expect, monkeypatch):
    """OMM_RUN_ENV_PROBES=1 only: runs ft_verify's actual torch witness tail
    in the MACE env against a synthetic module. A CPU forward next to
    unrelated CUDA allocations/copies must be gpu_used:false; a CUDA forward
    true -- and the verdict's judge() must agree with the raw record."""
    py = _probe_env_python(monkeypatch)
    if py is None:
        pytest.skip("set OMM_RUN_ENV_PROBES=1 with the MACE env installed to run the real witness tail")
    cuda_ok = subprocess.run([py, "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)"],
                             capture_output=True).returncode == 0
    if (device == "cuda" or junk) and not cuda_ok:
        pytest.skip("no CUDA device visible to the MACE env")
    script = _PROBE_HEAD.format(device=device, junk=junk) + ft_verify.witness_tail("MACE")
    proc = subprocess.run([py, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = ft_verify._last_json_line(proc.stdout)
    assert line is not None
    assert line["gpu_used"] is expect, line
    assert (line["witness"]["cuda_compute_ops"] > 0) is expect
    assert line["forces_shape"] == [4, 3] and line["forces_finite"] is True
    assert ft_verify.judge(line, device) == ("" if expect or device == "cpu" else "gpu_not_used")
    if expect:
        assert {"addmm", "tanh"} & set(line["witness"]["cuda_op_names"])
