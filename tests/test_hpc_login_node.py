"""HPC login nodes (found on a EuroHPC system, 2026-11-24): a per-user task cap,
no GPU on the node that has internet, and GPU nodes without internet.

GPU-free: fake cgroup trees, stub interpreters and stub subprocess runners.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import build_d3  # noqa: E402
import prefetch_weights  # noqa: E402
import prepare_nequip_weights  # noqa: E402
import setup_verify  # noqa: E402
import task_limit  # noqa: E402


def _exe(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)
    return path


def _cgroup(tmp_path: Path, v1: bool, user_max: str, scope_max: str = "max") -> dict:
    uid = os.getuid()
    root = tmp_path / "cg"
    base = root / "pids" if v1 else root
    scope = base / "user.slice" / f"user-{uid}.slice" / "session-1.scope"
    scope.mkdir(parents=True)
    (scope.parent / "pids.max").write_text(user_max + "\n")
    (scope / "pids.max").write_text(scope_max + "\n")
    proc = tmp_path / "proc_cgroup"
    line = (f"11:pids:/user.slice/user-{uid}.slice/session-1.scope" if v1
            else f"0::/user.slice/user-{uid}.slice/session-1.scope")
    proc.write_text(line + "\n")
    return {"OMM_CGROUP_ROOT": str(root), "OMM_PROC_CGROUP": str(proc)}


# ── task limit -> low-parallel settings ──────────────────────────────────────

@pytest.mark.parametrize("v1", [True, False])
def test_a_low_pids_max_turns_on_low_parallel_mode(tmp_path, v1):
    env = _cgroup(tmp_path, v1, "150")
    limit, low, parallel, exports = task_limit.plan(env)
    assert (limit, low, parallel) == (150, True, 2)
    # the pool that deadlocked on that login node, and the ones that hit the cap next
    assert exports["MAMBA_COMPILE_PYC"] == "false"
    for key in ("MAMBA_EXTRACT_THREADS", "CONDA_FETCH_THREADS", "MAX_JOBS", "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS", "CMAKE_BUILD_PARALLEL_LEVEL"):
        assert exports[key] == "2"


def test_the_smallest_limit_on_the_path_wins(tmp_path):
    env = _cgroup(tmp_path, False, "max", scope_max="300")
    assert task_limit.plan(env)[:2] == (300, True)


def test_no_limit_means_no_change(tmp_path):
    env = _cgroup(tmp_path, False, "max")
    assert task_limit.plan(env) == (None, False, 2, {})


def test_a_high_limit_is_not_low(tmp_path):
    assert task_limit.plan(_cgroup(tmp_path, True, "20000"))[1] is False


def test_omm_max_parallel_forces_the_mode_and_a_user_value_wins(tmp_path):
    env = dict(_cgroup(tmp_path, False, "max"), OMM_MAX_PARALLEL="4", MAX_JOBS="1")
    limit, low, parallel, exports = task_limit.plan(env)
    assert low and parallel == 4 and exports["OMP_NUM_THREADS"] == "4"
    assert "MAX_JOBS" not in exports            # already set by the user


def _low_parallel_block() -> str:
    text = (REPO / "install.sh").read_text()
    start = text.index("# ── Low-parallel mode under a per-user task limit")
    end = text.index("TORCHINDUCTOR_COMPILE_THREADS=1)", start) + len("TORCHINDUCTOR_COMPILE_THREADS=1)")
    return text[start:end]


def test_install_sh_exports_the_caps_before_any_build(tmp_path):
    env = dict(os.environ, **_cgroup(tmp_path, True, "150"))
    for key in ("OMP_NUM_THREADS", "MAMBA_COMPILE_PYC", "MAX_JOBS", "OPENBLAS_NUM_THREADS"):
        env.pop(key, None)
    script = (f"OH_MY_MLIP_HOME={REPO}; HOST_PY={sys.executable}\n{_low_parallel_block()}\n"
              'echo "pyc=$MAMBA_COMPILE_PYC jobs=$MAX_JOBS caps=${THREAD_CAPS[*]}"')
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "limits tasks per user (150)" in proc.stdout
    assert "pyc=false jobs=2" in proc.stdout and "OPENBLAS_NUM_THREADS=2" in proc.stdout
    text = (REPO / "install.sh").read_text()
    assert text.index("# ── Low-parallel mode under a per-user task limit") < text.index('"$CONDA_BIN" env create')


def test_install_sh_changes_nothing_without_a_limit(tmp_path):
    env = dict(os.environ, **_cgroup(tmp_path, False, "max"))
    env.pop("OMM_MAX_PARALLEL", None)
    env.pop("MAMBA_COMPILE_PYC", None)
    script = (f"OH_MY_MLIP_HOME={REPO}; HOST_PY={sys.executable}\n{_low_parallel_block()}\n"
              'echo "pyc=${MAMBA_COMPILE_PYC:-unset} caps=${THREAD_CAPS[*]}"')
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    assert "pyc=unset" in proc.stdout and "OMP_NUM_THREADS=4" in proc.stdout
    assert "limits tasks" not in proc.stdout


def test_when_the_limit_cannot_be_read_the_install_is_low_parallel_and_says_so(tmp_path):
    """A helper that cannot run (an old python3) must not leave the host unprotected."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "MAMBA_COMPILE_PYC", "MAX_JOBS", "OMM_MAX_PARALLEL")}
    for py in ("/bin/false", ""):
        script = (f"OH_MY_MLIP_HOME={REPO}; HOST_PY={py}\n{_low_parallel_block()}\n"
                  'echo "pyc=$MAMBA_COMPILE_PYC jobs=$MAX_JOBS caps=${THREAD_CAPS[*]}"')
        proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
        assert proc.returncode == 0, proc.stderr
        assert "could not read this host's task limit" in proc.stderr
        assert "pyc=false jobs=2" in proc.stdout and "OMP_NUM_THREADS=2" in proc.stdout


def test_task_limit_runs_on_python_3_6():
    """Rocky/RHEL 8 login nodes ship python3 = 3.6: no postponed annotations, no X | Y types."""
    import ast
    src = (REPO / "scripts" / "task_limit.py").read_text()
    tree = ast.parse(src, feature_version=(3, 6))
    assert "from __future__ import annotations" not in src
    annotated = [n for n in ast.walk(tree) if isinstance(n, (ast.arg, ast.AnnAssign)) and getattr(n, "annotation", None)]
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.returns]
    assert not annotated and not returns


def _host_python_func() -> str:
    text = (REPO / "install.sh").read_text()
    return text[text.index("host_python() {"):text.index('HOST_PY="$(host_python || true)"')]


def test_helpers_run_on_a_python_3_8_or_newer_not_a_bare_old_python3(tmp_path):
    bin_dir = tmp_path / "bin"
    _exe(bin_dir / "python3", "exit 1\n")              # 3.6: fails the version check
    _exe(bin_dir / "python3.9", "exit 0\n")
    script = f'CONDA_BIN=""; PATH={bin_dir}\n{_host_python_func()}\nhost_python'   # only the fake pythons
    proc = subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert proc.stdout.strip() == str(bin_dir / "python3.9")
    text = (REPO / "install.sh").read_text()
    # no helper script is started with the bare python3 any more
    assert 'python3 "$OH_MY_MLIP_HOME/scripts/' not in text and 'python3 "$prestage"' not in text
    assert text.index('HOST_PY="$(host_python || true)"') < text.index("# ── Low-parallel mode under a per-user task limit")


# ── install on a login node without a GPU (OMM_INSTALL_ARCH) ─────────────────

def test_recipe_choice_on_a_gpu_less_node_with_omm_install_arch(tmp_path):
    """The install path never asks for a GPU: the recipe comes from OMM_INSTALL_ARCH."""
    bin_dir = tmp_path / "bin"
    _exe(bin_dir / "nvidia-smi", "exit 9\n")     # a login node: nothing to report
    env = dict(os.environ, PATH=f"{bin_dir}:/usr/bin:/bin", OMM_INSTALL_ARCH="sm80")
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "select_recipe.py"), "mace", "--json"],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got["recipe"] == "mace" and got["arch"] == "sm80"


def test_install_sh_gpu_free_steps_never_call_setup_verify():
    body = (REPO / "install.sh").read_text()
    assert "scripts/setup_verify.py" not in body[body.index("install_one() {"):]


# ── verify fails at once without a GPU ───────────────────────────────────────

def test_gpu_visible(tmp_path, monkeypatch):
    smi = _exe(tmp_path / "bin" / "nvidia-smi", "")
    monkeypatch.setenv("PATH", f"{smi.parent}:/usr/bin:/bin")
    listed = lambda *a, **k: SimpleNamespace(stdout="GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-x)\n")  # noqa: E731
    empty = lambda *a, **k: SimpleNamespace(stdout="No devices were found\n")  # noqa: E731
    assert setup_verify.gpu_visible(run=listed, env={}) is True
    assert setup_verify.gpu_visible(run=empty, env={}) is False
    assert setup_verify.gpu_visible(run=listed, env={"CUDA_VISIBLE_DEVICES": ""}) is False


def test_verify_without_a_gpu_fails_at_once_with_the_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(setup_verify, "predict_driver_skew", lambda env, home: {"skew": False})
    monkeypatch.setattr(setup_verify, "gpu_visible", lambda: False)
    monkeypatch.setattr(setup_verify, "stream_process",
                        lambda *a, **k: pytest.fail("nothing may run without a GPU"))
    v = setup_verify.verify_one("MACE", None, None, REPO, quiet=True, no_local_record=True)
    assert v["pass"] is False and v["device"] == "cuda"
    assert "no NVIDIA GPU is visible" in v["reason"] and "--gres=gpu:1" in v["reason"]


def test_a_planned_cpu_run_is_not_stopped(monkeypatch):
    """Driver skew -> a planned CPU run: no GPU needed, so the fail-fast does not apply."""
    monkeypatch.setattr(setup_verify, "predict_driver_skew", lambda env, home: {"skew": True})
    monkeypatch.setattr(setup_verify, "gpu_visible", lambda: pytest.fail("not asked on a planned CPU run"))
    called = {}

    def fake_stream(command, **kw):
        called["cmd"] = command
        return 1, 0.0, "", "boom", {"seen": False}

    monkeypatch.setattr(setup_verify, "stream_process", fake_stream)
    monkeypatch.setattr(setup_verify, "check_d3", lambda *a, **k: {"state": "skipped"})
    setup_verify.verify_one("MACE", None, None, REPO, quiet=True, no_local_record=True)
    assert "--device" in called["cmd"]


# ── weights downloaded at install time ───────────────────────────────────────

def test_cpu_lines_switch_only_the_device():
    lines = ["calc = mace_mp(model='medium-mpa-0', device='cuda')",
             'orbff = pretrained.orb(device="cuda:0", precision="float32-high")',
             "calc = X(cpu=False)", "calc = SevenNetCalculator('7net-mf-ompa', modal='mpa')"]
    assert prefetch_weights.cpu_lines(lines) == [
        "calc = mace_mp(model='medium-mpa-0', device='cpu')",
        "orbff = pretrained.orb(device='cpu', precision=\"float32-high\")",
        "calc = X(cpu=True)", "calc = SevenNetCalculator('7net-mf-ompa', modal='mpa')"]


def test_every_variant_of_an_env_is_listed():
    assert prefetch_weights.variants("mace") == [
        ("MACE", "MACE-MPA-0"), ("MACE", "MACE-MH-1-OMAT"), ("MACE", "MACE-MH-1-OC20")]


def test_a_by_name_weight_is_fetched_by_building_on_the_cpu(monkeypatch):
    monkeypatch.setattr(prefetch_weights.fetch, "ensure_weights", lambda *a, **k: [])
    seen = {}

    def run(argv, **kw):
        seen["code"], seen["env"] = argv[2], kw["env"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    state, detail = prefetch_weights.prefetch_one("MACE", "MACE-MPA-0", run=run, env={})
    assert state == "ok"
    assert "device='cpu'" in seen["code"] and "device='cuda'" not in seen["code"]
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == ""


def test_a_cpu_build_failure_is_reported_not_hidden(monkeypatch):
    monkeypatch.setattr(prefetch_weights.fetch, "ensure_weights", lambda *a, **k: [])
    run = lambda argv, **kw: SimpleNamespace(returncode=1, stdout="", stderr="x\nRuntimeError: needs CUDA\n")  # noqa: E731
    state, detail = prefetch_weights.prefetch_one("SevenNet", "SevenNet-MF-OMPA", run=run, env={})
    assert state == "failed" and "RuntimeError: needs CUDA" in detail


def test_gated_without_a_token_and_compiled_models_are_skipped(monkeypatch):
    monkeypatch.setattr(prefetch_weights.fetch, "_resolve_token", lambda env: {"source": "none"})
    no_run = lambda *a, **k: pytest.fail("nothing to build")  # noqa: E731
    assert prefetch_weights.prefetch_one("UMA", "UMA-s-1p2-OMAT", run=no_run, env={})[0] == "skipped"
    assert prefetch_weights.prefetch_one("NequIP", "NequIP-OAM-L", run=no_run, env={})[0] == "skipped"


def test_install_sh_prefetches_before_the_sentinel_and_only_warns():
    text = (REPO / "install.sh").read_text()
    func = text[text.index("prefetch_weights() {"):text.index("build_d3_kernel() {")]
    assert 'CUDA_VISIBLE_DEVICES=""' in func and "|| OMM_PREFETCH_WARNINGS+=" in func
    body = text[text.index("install_one() {"):]
    assert body.index('  prefetch_weights "$env_name" "$prefix"\n') < body.index(': > "$sentinel"')
    # an already-installed env without the record gets it on the next run
    assert '[ -e "$prefix/.omm_prefetch.json" ] || prefetch_weights "$env_name" "$prefix"' in body


# ── NequIP / Allegro: download on the login node, compile on a GPU node ──────

def test_nequip_prepare_without_a_gpu_downloads_and_defers(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(prepare_nequip_weights, "host_arch", lambda: None)
    got = []
    monkeypatch.setattr(prepare_nequip_weights, "fetch_package",
                        lambda version, root: got.append(version) or root / "x.zip")
    rc = prepare_nequip_weights.main(["--target-root", str(tmp_path / "models" / "nequip")])
    assert rc == prepare_nequip_weights.DEFERRED == 4
    assert got == ["NequIP-OAM-XL", "NequIP-OAM-L"]
    assert "compile it on a GPU node" in capsys.readouterr().out


def test_install_sh_accepts_a_deferred_compile():
    body = (REPO / "install.sh").read_text()
    assert 'if [ "$prc" -eq 4 ]; then' in body


# ── D3 kernel: ninja and a stale build lock ──────────────────────────────────

def test_install_sh_adds_ninja_to_an_env_with_nvcc_and_no_ninja(tmp_path):
    text = (REPO / "install.sh").read_text()
    func = text[text.index("OMM_D3_WARNINGS=()"):text.index("install_one() {")]
    home = tmp_path / "hub"
    (home / "scripts").mkdir(parents=True)
    (home / "scripts" / "build_d3.py").write_text("import sys\nsys.exit(0)\n")
    for name, has_ninja in (("a", False), ("b", True)):
        prefix = tmp_path / "envs" / name
        _exe(prefix / "bin" / "nvcc", "exit 0\n")
        _exe(prefix / "bin" / "pip", f'echo "$@" >> {tmp_path}/pip_{name}.log\n')
        if has_ninja:
            _exe(prefix / "bin" / "ninja", "exit 0\n")
        script = f"OH_MY_MLIP_HOME={home}; HOST_PY={sys.executable}; THREAD_CAPS=(env)\n{func}\nbuild_d3_kernel {name} {prefix}\n"
        proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
    assert "install --quiet ninja==1.13.0" in (tmp_path / "pip_a.log").read_text()
    assert not (tmp_path / "pip_b.log").exists()


def test_a_stale_build_lock_is_removed_and_a_fresh_one_kept(tmp_path):
    lock = tmp_path / "lock"
    lock.write_text("")
    now = time.time()
    assert build_d3.clear_stale_lock(str(tmp_path), now=now + 60) == ""
    assert lock.exists()
    msg = build_d3.clear_stale_lock(str(tmp_path), now=now + build_d3.STALE_LOCK_S + 5)
    assert "removed a build lock" in msg and not lock.exists()
    assert build_d3.clear_stale_lock(str(tmp_path)) == ""      # no lock: nothing to say


def test_prebuild_without_ninja_says_so_before_building(tmp_path, monkeypatch):
    prefix = tmp_path / "envs" / "mace"
    for tool in ("python", "nvcc"):
        _exe(prefix / "bin" / tool, "exit 0\n")
    d3 = tmp_path / "site" / "catbench" / "dispersion" / "cuda"
    d3.mkdir(parents=True)
    monkeypatch.setattr(build_d3, "SYSTEM_CUDA", str(tmp_path / "none"))
    monkeypatch.setattr(build_d3, "host_compute_cap", lambda run=None: None)

    def run(argv, **kw):
        if argv[-1] == "--list-gpu-arch":
            return SimpleNamespace(returncode=0, stdout="compute_80 compute_90", stderr="")
        if build_d3.PROBE_MARK in argv[2]:
            out = build_d3.PROBE_MARK + json.dumps({"torch": True, "d3_dir": str(d3)}) + "\n"
            return SimpleNamespace(returncode=0, stdout=out, stderr="")
        if argv[2] == build_d3.CACHE_CODE:
            got = build_d3.CACHE_MARK + json.dumps({"cache_dir": None, "error": "x"}) + "\n"
            return SimpleNamespace(returncode=0, stdout=got, stderr="")
        pytest.fail(f"no build without ninja: {argv}")

    state, detail = build_d3.prebuild(prefix, tmp_path / "d3.log", run=run, base={"PATH": str(tmp_path / "empty")})
    assert state == "failed" and "no ninja" in detail and "pip install ninja==1.13.0" in detail


def test_the_d3_build_caps_blas_threads(tmp_path, monkeypatch):
    prefix = tmp_path / "envs" / "mace"
    _exe(prefix / "bin" / "nvcc", "exit 0\n")
    monkeypatch.setattr(build_d3, "host_compute_cap", lambda run=None: None)
    run = lambda argv, **kw: SimpleNamespace(returncode=0, stdout="compute_80", stderr="")  # noqa: E731
    env, _ = build_d3.build_env(prefix, {"PATH": "/usr/bin"}, run=run)
    assert env["OPENBLAS_NUM_THREADS"] == env["OMP_NUM_THREADS"] == "1"


# ── offline compute nodes: fail network access at once ───────────────────────

def _hooked(code: str, offline: str | None, tmp_path: Path) -> subprocess.CompletedProcess:
    site = tmp_path / "site"
    site.mkdir(exist_ok=True)
    (site / "_omm_offline.py").write_text((REPO / "oh_my_mlip" / "_offline_hook.py").read_text())
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(site)}
    if offline is not None:
        env["OMM_OFFLINE"] = offline
    return subprocess.run([sys.executable, "-c", "import _omm_offline\n" + code],
                          capture_output=True, text=True, env=env, timeout=60)


def test_the_offline_hook_fails_internet_access_at_once_and_keeps_local(tmp_path):
    code = ("import socket, time, os\n"
            "t = time.time()\n"
            "try:\n    socket.create_connection(('8.8.8.8', 443), timeout=20); print('connected')\n"
            "except OSError as e:\n    print('blocked', round(time.time() - t, 1), 'offline mode' in str(e))\n"
            "try:\n    socket.getaddrinfo('huggingface.co', 443); print('resolved')\n"
            "except socket.gaierror:\n    print('no-dns')\n"
            "srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1)\n"
            "socket.create_connection(srv.getsockname(), timeout=5).close(); print('local-ok')\n"
            "print(os.environ.get('HF_HUB_OFFLINE'))\n")
    out = _hooked(code, "1", tmp_path).stdout.split()
    assert out[0] == "blocked" and float(out[1]) < 1.0 and out[2] == "True"
    assert out[3:] == ["no-dns", "local-ok", "1"]


def test_the_offline_hook_does_nothing_unless_asked(tmp_path):
    code = "import socket; print(socket.getaddrinfo.__module__)"
    for value in (None, "0"):
        assert _hooked(code, value, tmp_path).stdout.strip() == "socket"


def test_offline_env_switches_on_only_without_internet(monkeypatch):
    from oh_my_mlip import offline

    def refused(addr, timeout):
        raise OSError("timed out")

    class Ok:
        def close(self):
            pass

    monkeypatch.setattr(offline, "_probe_cache", None)
    got = offline.offline_env({"PATH": "/bin"}, connect=refused)
    assert got["OMM_OFFLINE"] == got["HF_HUB_OFFLINE"] == "1"
    monkeypatch.setattr(offline, "_probe_cache", None)
    assert "OMM_OFFLINE" not in offline.offline_env({}, connect=lambda a, timeout: Ok())
    # the user's choice is never overridden, and no probe runs
    monkeypatch.setattr(offline, "_probe_cache", None)
    assert offline.offline_env({"OMM_OFFLINE": "0"}, connect=refused) == {"OMM_OFFLINE": "0"}
    assert offline._probe_cache is None


def test_workers_get_offline_mode_and_install_sh_installs_the_hook():
    assert "return offline_env(child_env)" in (REPO / "oh_my_mlip" / "provider.py").read_text()
    text = (REPO / "install.sh").read_text()
    func = text[text.index("install_offline_hook() {"):text.index("prefetch_weights() {")]
    assert '"$site/_omm_offline.py"' in func and "import _omm_offline" in func
    body = text[text.index("install_one() {"):]
    assert body.index('  install_offline_hook "$prefix"\n') < body.index('  prefetch_weights "$env_name" "$prefix"\n')
    env_sh = (REPO / "env.sh").read_text()
    assert "export OMM_OFFLINE=1 HF_HUB_OFFLINE=1" in env_sh and 'if [ -z "${OMM_OFFLINE:-}" ]' in env_sh


# ── frameworks that write to ~/.cache whatever the variables say ─────────────

def test_home_caches_become_links_into_the_shared_root(tmp_path):
    home, root = tmp_path / "home", tmp_path / "proj" / "cache"
    (home / ".cache" / "grace").mkdir(parents=True)              # already there: kept, listed
    (home / ".cache" / "tace").parent.mkdir(parents=True, exist_ok=True)
    os.symlink(tmp_path / "elsewhere", home / ".cache" / "tace")  # an existing link: kept
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "OMM_SHARED_CACHE_ROOT": str(root)}
    proc = subprocess.run(["bash", str(REPO / "scripts" / "home_cache_links.sh")],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    link = home / ".cache" / "sevennet"
    assert link.is_symlink() and Path(os.readlink(link)) == root / "home" / ".cache" / "sevennet"
    assert (home / ".local" / "mattersim").is_symlink()
    assert not (home / ".cache" / "grace").is_symlink() and str(home / ".cache" / "grace") in proc.stderr
    assert Path(os.readlink(home / ".cache" / "tace")) == tmp_path / "elsewhere"
    # nothing at all without a shared root
    env.pop("OMM_SHARED_CACHE_ROOT")
    env["HOME"] = str(tmp_path / "home2")
    subprocess.run(["bash", str(REPO / "scripts" / "home_cache_links.sh")], env=env, check=True)
    assert not (tmp_path / "home2").exists()


# ── a failed NequIP / Allegro compile leaves nothing behind ──────────────────

def test_a_failed_compile_removes_its_partial_outputs(monkeypatch, tmp_path, capsys):
    home = tmp_path / "hub"
    root = home / "models" / "nequip"
    root.mkdir(parents=True)
    out_dir = home / "models" / "compiled" / "sm80"
    out_dir.mkdir(parents=True)
    monkeypatch.setattr(prepare_nequip_weights, "host_arch", lambda: "sm80")
    monkeypatch.setattr(prepare_nequip_weights, "fetch_package", lambda version, r: r / "x.zip")

    def fake_run(cmd, env=None):
        out = Path(cmd[2])
        Path(str(out) + ".partial").write_bytes(b"x" * 10)        # what a crashed compile leaves
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(prepare_nequip_weights.subprocess, "run", fake_run)
    assert prepare_nequip_weights.main(["--target-root", str(root)]) == 1
    assert list(out_dir.iterdir()) == []
    assert "removed 1 partial file(s)" in capsys.readouterr().err


# ── offline mode: proxies, cluster-internal names, and never during an install ─

_FAKE_DNS = ("import socket\n"
             "_real = socket.getaddrinfo\n"
             "_table = {'gnx507': '10.11.0.7', 'mirror.site': 'fd00::5', 'proxy.site': '198.51.100.9',\n"
             "          'example.org': '93.184.216.34'}\n"
             "def _fake(host, port, *a, **k):\n"
             "    if host in _table:\n"
             "        fam = socket.AF_INET6 if ':' in _table[host] else socket.AF_INET\n"
             "        return [(fam, socket.SOCK_STREAM, 6, '', (_table[host], port))]\n"
             "    return _real(host, port, *a, **k)\n"
             "socket.getaddrinfo = _fake\n")


def _hooked_with_dns(code: str, tmp_path: Path, extra_env: dict | None = None) -> list[str]:
    site = tmp_path / "site"
    site.mkdir(exist_ok=True)
    (site / "_omm_offline.py").write_text((REPO / "oh_my_mlip" / "_offline_hook.py").read_text())
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(site), "OMM_OFFLINE": "1", **(extra_env or {})}
    proc = subprocess.run([sys.executable, "-c", _FAKE_DNS + "import _omm_offline\n" + code],
                          capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.split()


_RESOLVE = ("def r(h):\n"
            "    try:\n        print(h, socket.getaddrinfo(h, 29500)[0][4][0])\n"
            "    except socket.gaierror:\n        print(h, 'blocked')\n")


def test_cluster_internal_names_and_private_addresses_stay_reachable(tmp_path):
    """DDP rendezvous with MASTER_ADDR=<another compute node>, a site mirror, a private IP."""
    out = _hooked_with_dns(_RESOLVE + "for h in ('gnx507', 'mirror.site', '172.16.4.2', 'example.org', '93.184.216.34'):\n"
                           "    r(h)\n", tmp_path)
    assert out == ["gnx507", "10.11.0.7", "mirror.site", "fd00::5", "172.16.4.2", "172.16.4.2",
                   "example.org", "blocked", "93.184.216.34", "blocked"]


def test_the_proxy_host_stays_reachable(tmp_path):
    code = (_RESOLVE + "r('proxy.site'); r('example.org')\n"
            "s = socket.socket(); s.settimeout(0.2)\n"
            "rc = s.connect_ex(('198.51.100.9', 3128)); print('proxy-connect', rc != __import__('errno').ENETUNREACH)\n")
    out = _hooked_with_dns(code, tmp_path, {"https_proxy": "http://proxy.site:3128"})
    assert out[:4] == ["proxy.site", "198.51.100.9", "example.org", "blocked"]
    assert out[4:] == ["proxy-connect", "True"]          # not refused by the hook (the network may still say no)


def test_a_proxy_site_is_tested_through_its_proxy(monkeypatch):
    from oh_my_mlip import offline
    no_direct = lambda addr, timeout: (_ for _ in ()).throw(OSError("direct blocked"))  # noqa: E731
    asked = []
    monkeypatch.setattr(offline, "_probe_cache", None)
    got = offline.offline_env({"https_proxy": "http://proxy.site:3128"}, connect=no_direct,
                              head=lambda host: asked.append(host) or True)
    assert "OMM_OFFLINE" not in got and asked == ["huggingface.co"]
    env_sh = (REPO / "env.sh").read_text()
    func = env_sh[env_sh.index("omm_internet_ok() {"):env_sh.index('if [ -z "${OMM_OFFLINE:-}" ]; then')]
    assert "curl -sI" in func and "https_proxy" in func


def test_env_sh_through_a_proxy_uses_curl(tmp_path):
    env_sh = (REPO / "env.sh").read_text()
    func = env_sh[env_sh.index("omm_internet_ok() {"):env_sh.index('if [ -z "${OMM_OFFLINE:-}" ]; then')]
    bin_dir = tmp_path / "bin"
    _exe(bin_dir / "curl", f'echo "$@" >> {tmp_path}/curl.log\nexit 0\n')
    script = f"{func}\nomm_internet_ok 3 && echo online"
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          env={"PATH": f"{bin_dir}:/usr/bin:/bin", "https_proxy": "http://proxy.site:3128"})
    assert proc.stdout.strip() == "online"
    assert "-sI -o /dev/null --max-time 3 https://huggingface.co" in (tmp_path / "curl.log").read_text()


def test_install_sh_never_runs_offline():
    text = (REPO / "install.sh").read_text()
    i = text.index('source "$OH_MY_MLIP_HOME/env.sh"\nomm_online=0')
    head = text[i - 600:i]
    assert "export OMM_OFFLINE=0" in head and "unset HF_HUB_OFFLINE" in head
    tail = text[i:i + 900]
    assert "omm_internet_ok 5" in tail and "install on a node" in tail and "exit 1" in tail
    assert tail.index("exit 1") < text[i:].index("THREAD_CAPS=(")
