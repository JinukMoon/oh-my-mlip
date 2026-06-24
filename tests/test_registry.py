"""No-GPU unit tests for the tiered teacher-provider base.

These MUST pass without a GPU, any conda env, torch, ase weights, or
huggingface_hub. Subprocess interaction is mocked. They verify:
  - list_models() returns the full roster incl. MACE + SevenNet
  - resolve() yields the LOCKED codegen dict with $OH_MY_MLIP_HOME expanded
  - apply_d3 flag plumbs through to the worker command
  - parse_env_run() allowlist: valid token parses; shell/non-allowlisted RAISES
  - Worker routing is by id (not FIFO) against a mocked worker subprocess
"""
import json
import os

import pytest

from oh_my_mlip import (
    list_models,
    parse_env_run,
    resolve,
)
from oh_my_mlip import registry as reg
from oh_my_mlip import provider


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── list_models ──────────────────────────────────────────────────────────────
def test_list_models_includes_mace_and_sevennet():
    models = list_models()
    assert "MACE" in models
    assert "SevenNet" in models
    # full set is carried; sanity-check a few more rows exist
    for extra in ("NequIP", "ORB", "UMA", "CHGNet"):
        assert extra in models
    # _meta must never leak into the model list
    assert not any(m.startswith("_") for m in models)


# ── resolve: codegen dict + $OH_MY_MLIP_HOME expansion ───────────────────────
def test_resolve_mace_codegen_dict():
    spec = resolve("MACE", "MACE-MPA-0")
    assert spec["model"] == "MACE"
    assert spec["version"] == "MACE-MPA-0"
    assert spec["env"] == "mace"
    # interpreter path is expanded to an absolute path under the repo root.
    assert "${OH_MY_MLIP_HOME}" not in spec["python"]
    assert spec["python"].endswith("/envs/mace/bin/python")
    assert spec["python"].startswith(reg.home())
    # imports + inference are present and expanded.
    assert any("mace_mp" in line for line in spec["imports"])
    assert any("calc = mace_mp(" in line for line in spec["inference"])
    assert not any("${OH_MY_MLIP_HOME}" in line for line in spec["inference"])
    # contract keys all present.
    for key in (
        "python", "env", "imports", "inference", "env_run",
        "arch_pinned", "gated", "weights", "validation",
    ):
        assert key in spec
    assert spec["gated"] is False
    assert spec["weights"] == "bundled"
    assert spec["validation"].startswith("validated")
    assert spec["env_run"] == {}


def test_resolve_sevennet_explicit_version():
    spec = resolve("SevenNet", "SevenNet-MF-OMPA")
    assert spec["env"] == "sevennet"
    assert any("SevenNetCalculator" in line for line in spec["imports"])


def test_resolve_default_version_multiversion():
    # Multi-version frameworks declare default_version -> resolve(model, None)
    # must succeed and return that default (not raise).
    mace = resolve("MACE", None)
    assert mace["version"] == "MACE-MPA-0"
    seven = resolve("SevenNet", None)
    assert seven["version"] == "SevenNet-MF-OMPA"


def test_resolve_default_version_matches_registry_field():
    # The selected default must equal the framework's declared default_version.
    models = reg.load_models()
    for fw in ("MACE", "SevenNet"):
        assert resolve(fw, None)["version"] == models[fw]["default_version"]


def test_resolve_ambiguous_without_default_raises(monkeypatch):
    # A multi-version framework with NO default_version must still raise on None.
    models = reg.load_models()
    models = json.loads(json.dumps(models))  # deep copy
    models["MACE"].pop("default_version", None)
    with pytest.raises(reg.RegistryError):
        resolve("MACE", None, models=models)


def test_resolve_bad_default_version_raises():
    models = reg.load_models()
    models = json.loads(json.dumps(models))  # deep copy
    models["MACE"]["default_version"] = "NoSuchVersion"
    with pytest.raises(reg.RegistryError):
        resolve("MACE", None, models=models)


def test_resolve_home_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OH_MY_MLIP_HOME", str(tmp_path))
    spec = resolve("MACE", "MACE-MPA-0")
    assert spec["python"] == str(tmp_path / "envs" / "mace" / "bin" / "python")


def test_resolve_gated_uma():
    spec = resolve("UMA", "UMA-s-1p1-OMAT")
    assert spec["gated"] is True
    assert spec["license_url"] and "huggingface.co" in spec["license_url"]
    assert spec["weights"] == "on-demand-hf"


def test_resolve_arch_pinned_nequip():
    spec86 = resolve("NequIP", "NequIP-OAM-XL", arch="sm86")
    spec89 = resolve("NequIP", "NequIP-OAM-XL", arch="sm89")
    assert spec86["arch_pinned"] is True
    assert any("sm86" in line for line in spec86["inference"])
    assert any("sm89" in line for line in spec89["inference"])
    # default arch is sm89
    spec_default = resolve("NequIP", "NequIP-OAM-XL")
    assert any("sm89" in line for line in spec_default["inference"])


def test_resolve_unknown_raises():
    with pytest.raises(reg.RegistryError):
        resolve("NotAModel")
    with pytest.raises(reg.RegistryError):
        resolve("_meta")


# ── parse_env_run allowlist (security boundary) ──────────────────────────────
def test_parse_env_run_valid_empty_ld_library_path():
    assert parse_env_run('LD_LIBRARY_PATH=""') == {"LD_LIBRARY_PATH": ""}


def test_parse_env_run_valid_multiple():
    out = parse_env_run("OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0")
    assert out == {"OMP_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": "0"}


def test_parse_env_run_empty_and_none():
    assert parse_env_run("") == {}
    assert parse_env_run(None) == {}


@pytest.mark.parametrize(
    "bad",
    [
        "$(rm -rf /)",            # command substitution, not KEY=VALUE
        "FOO=bar; rm x",         # not allow-listed key + shell ;
        "LD_LIBRARY_PATH=$(id)",  # allow-listed key but shell substitution value
        "PATH=/usr/bin",         # PATH not on the allowlist
        "LD_LIBRARY_PATH=a`b`",  # backtick metachar
        "OMP_NUM_THREADS=4|cat",  # pipe metachar
    ],
)
def test_parse_env_run_rejects_unsafe(bad):
    with pytest.raises(reg.RegistryError):
        parse_env_run(bad)


def test_dpa4_env_run_parses_from_registry():
    # DPA4 carries env_run LD_LIBRARY_PATH="" in models.json -> must parse.
    spec = resolve("DPA4", "DPA-4.0.1-pro-MPtrj")
    assert spec["env_run"] == {"LD_LIBRARY_PATH": ""}
    assert spec["env_run_raw"] == 'LD_LIBRARY_PATH=""'


# ── apply_d3 plumbing + Worker id-routing (mocked subprocess) ────────────────
class _FakeStream:
    """A minimal text stream capturing writes / serving canned lines."""

    def __init__(self, lines=None):
        self._lines = list(lines or [])
        self.written = []

    def write(self, s):
        self.written.append(s)

    def flush(self):
        pass

    def close(self):
        pass

    def readline(self):
        return self._lines.pop(0) if self._lines else ""

    def read(self):
        return ""


class _FakeProc:
    """Stand-in for subprocess.Popen that records argv/env and replays lines."""

    def __init__(self, cmd, stdout_lines, env=None, **kw):
        self.cmd = cmd
        self.env = env or {}
        self.stdin = _FakeStream()
        self.stdout = _FakeStream(stdout_lines)
        self.stderr = _FakeStream()
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        self._alive = False
        return 0

    def kill(self):
        self._alive = False


def _make_popen(stdout_lines, captured):
    def _popen(cmd, **kw):
        proc = _FakeProc(cmd, stdout_lines, env=kw.get("env"))
        captured["proc"] = proc
        return proc

    return _popen


def test_worker_apply_d3_flag_in_cmd():
    captured = {}
    handshake = json.dumps({"ready": True, "model": "MACE"}) + "\n"
    w = provider.Worker(
        "MACE", version="MACE-MPA-0", apply_d3=True,
        _popen=_make_popen([handshake], captured),
    )
    w.start()
    cmd = captured["proc"].cmd
    assert "--apply-d3" in cmd
    assert "--model" in cmd and "MACE" in cmd
    assert cmd[0].endswith("/envs/mace/bin/python")
    assert "-m" in cmd and "oh_my_mlip._worker" in cmd


def test_worker_no_d3_flag_absent():
    captured = {}
    handshake = json.dumps({"ready": True, "model": "MACE"}) + "\n"
    w = provider.Worker(
        "MACE", version="MACE-MPA-0", apply_d3=False,
        _popen=_make_popen([handshake], captured),
    )
    w.start()
    assert "--apply-d3" not in captured["proc"].cmd


def test_worker_env_run_applied_as_subprocess_env():
    # DPA4 carries env_run -> the parsed dict must land in the child env.
    captured = {}
    handshake = json.dumps({"ready": True, "model": "DPA4"}) + "\n"
    w = provider.Worker(
        "DPA4", version="DPA-4.0.1-pro-MPtrj",
        _popen=_make_popen([handshake], captured),
    )
    w.start()
    assert captured["proc"].env.get("LD_LIBRARY_PATH") == ""
    assert captured["proc"].env.get("OH_MY_MLIP_HOME")


def test_worker_routes_by_id_not_fifo(monkeypatch):
    # Worker.request sends id=N and must validate the response carries that id.
    captured = {}
    # handshake then a response carrying the SAME id the worker will assign (1).
    resp = json.dumps({"id": 1, "ok": True, "results": {"energy": -3.5}}) + "\n"
    handshake = json.dumps({"ready": True, "model": "MACE"}) + "\n"
    w = provider.Worker(
        "MACE", version="MACE-MPA-0",
        _popen=_make_popen([handshake, resp], captured),
    )
    # avoid importing ase: stub encode_atoms
    monkeypatch.setattr(
        "oh_my_mlip._worker.encode_atoms", lambda atoms: {"stub": True}
    )
    w.start()
    out = w.request(object(), ("energy",))
    assert out == {"id": 1, "ok": True, "results": {"energy": -3.5}}
    # request payload written to stdin carries the id and properties.
    sent = json.loads(captured["proc"].stdin.written[0])
    assert sent["id"] == 1
    assert sent["properties"] == ["energy"]


def test_worker_id_mismatch_raises(monkeypatch):
    captured = {}
    resp = json.dumps({"id": 999, "ok": True, "results": {}}) + "\n"
    handshake = json.dumps({"ready": True}) + "\n"
    w = provider.Worker(
        "MACE", version="MACE-MPA-0",
        _popen=_make_popen([handshake, resp], captured),
    )
    monkeypatch.setattr(
        "oh_my_mlip._worker.encode_atoms", lambda atoms: {"stub": True}
    )
    w.start()
    with pytest.raises(provider.WorkerError):
        w.request(object(), ("energy",))


def test_worker_handshake_failure_raises():
    captured = {}
    handshake = json.dumps({"ready": False, "error": "no cuda"}) + "\n"
    w = provider.Worker(
        "MACE", version="MACE-MPA-0",
        _popen=_make_popen([handshake], captured),
    )
    with pytest.raises(provider.WorkerError):
        w.start()


def test_worker_crash_surfaces_ok_false(monkeypatch):
    # Empty stdout on request => worker died mid-request => ok:false.
    captured = {}
    handshake = json.dumps({"ready": True}) + "\n"
    w = provider.Worker(
        "MACE", version="MACE-MPA-0",
        _popen=_make_popen([handshake], captured),  # no response line queued
    )
    monkeypatch.setattr(
        "oh_my_mlip._worker.encode_atoms", lambda atoms: {"stub": True}
    )
    w.start()
    out = w.request(object(), ("energy",))
    assert out["ok"] is False
    assert "crashed" in out["error"]


def test_worker_version_none_resolves_default():
    # Worker(model) with version=None must resolve the default and spawn.
    for model, want in (("MACE", "MACE-MPA-0"), ("SevenNet", "SevenNet-MF-OMPA")):
        captured = {}
        handshake = json.dumps({"ready": True, "model": model}) + "\n"
        w = provider.Worker(model, _popen=_make_popen([handshake], captured))
        assert w.spec["version"] == want
        w.start()
        assert "--model" in captured["proc"].cmd and model in captured["proc"].cmd


def test_run_version_none_resolves_default(monkeypatch):
    # run(model) with version=None routes through a Worker that resolves default.
    real_worker = provider.Worker
    for model in ("MACE", "SevenNet"):
        captured = {}
        handshake = json.dumps({"ready": True, "model": model}) + "\n"
        resp = json.dumps({"id": 1, "ok": True, "results": {"energy": -1.0}}) + "\n"
        popen = _make_popen([handshake, resp], captured)

        def _worker_with_fake_popen(*a, _popen=popen, **kw):
            kw.setdefault("_popen", _popen)
            return real_worker(*a, **kw)

        monkeypatch.setattr(provider, "Worker", _worker_with_fake_popen)
        monkeypatch.setattr(
            "oh_my_mlip._worker.encode_atoms", lambda atoms: {"stub": True}
        )
        out = provider.run(model, object(), ("energy",))
        assert out == {"energy": -1.0}
        # version=None resolved to the framework's default before spawning.
        assert "--model" in captured["proc"].cmd and model in captured["proc"].cmd


def test_fetch_env_check_gated_resolves_version_none():
    # fetch._check_gated calls resolve(model, version=None); must not raise for
    # multi-version MACE/SevenNet now that defaults exist.
    from oh_my_mlip import fetch

    for model in ("MACE", "SevenNet"):
        spec = fetch._check_gated(model, None)
        assert spec["model"] == model
        assert spec["version"]  # a concrete default version was chosen


def test_package_imports_without_torch_ase():
    # Importing the package must not require torch/ase/huggingface_hub.
    import importlib

    import oh_my_mlip
    import oh_my_mlip.fetch
    import oh_my_mlip.provider
    import oh_my_mlip._worker

    importlib.reload(oh_my_mlip)
    for name in ("list_models", "resolve", "get_calculator", "run",
                 "parse_env_run", "Worker", "WorkerPool"):
        assert hasattr(oh_my_mlip, name)
