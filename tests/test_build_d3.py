"""D3 kernel prebuild (scripts/build_d3.py) and where it is wired in.

The first catbench D3 call compiles pair_d3.so with torch cpp_extension. Through
an env's bare interpreter that failed on an sm120 host ("Ninja is required"):
the env's bin/ (ninja, nvcc, the conda compiler) is not on PATH, and nvcc then
picks the system gcc 15. install.sh and adopt_env.py now build the kernel up
front with the env's own tools; setup_verify reports one D3 calculation on its
own; env.sh also finds the active conda env's nvcc.

GPU-free: subprocess calls are faked, fake envs are small shell scripts.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
ENVS = REPO / "envs"
sys.path.insert(0, str(REPO / "scripts"))

import build_d3  # noqa: E402
import setup_verify  # noqa: E402

# the sm120 siblings that exist only to add ninja + a pinned host compiler
D3_SIBLINGS = {"uma.cu128": "14.4.0", "deepmd.cu128": "14.4.0", "equiformer_v3.cu128": "14.4.0",
               "pet.cu128": "14.4.0", "dpa4.cu130": "15.3.0", "matris.cu130": "15.3.0",
               "tace.cu130": "15.3.0"}


def _exe(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _probe_line(d3_dir, torch=True) -> str:
    return build_d3.PROBE_MARK + json.dumps({"torch": torch, "d3_dir": d3_dir}) + "\n"


class FakeRun:
    """Stands in for subprocess.run: answers the probe, nvcc and the build."""

    def __init__(self, d3_dir=None, torch=True, build_rc=0, build_out="", archs="compute_75 compute_120",
                 cache=None, cache_error="", build_into=None):
        self.d3_dir, self.torch, self.build_rc, self.build_out, self.archs = d3_dir, torch, build_rc, build_out, archs
        self.cache, self.cache_error = cache, cache_error
        # where a successful fake build drops pair_d3.so (default: the package folder)
        self.build_into = build_into if build_into is not None else d3_dir
        self.calls = []

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        if argv[-1] == "--list-gpu-arch":
            return SimpleNamespace(returncode=0, stdout=self.archs, stderr="")
        if argv[1:2] == ["-c"] and build_d3.PROBE_MARK in argv[2]:
            d = str(self.d3_dir) if self.d3_dir else None
            return SimpleNamespace(returncode=0, stdout=_probe_line(d, self.torch), stderr="")
        if argv[1:2] == ["-c"] and argv[2] == build_d3.CACHE_CODE:
            got = {"cache_dir": str(self.cache) if self.cache else None, "error": self.cache_error}
            return SimpleNamespace(returncode=0, stdout=build_d3.CACHE_MARK + json.dumps(got) + "\n", stderr="")
        if argv[1:2] == ["-c"] and argv[2] == build_d3.BUILD_CODE:
            if self.build_rc == 0 and self.build_into:
                (Path(self.build_into) / "pair_d3.so").write_bytes(b"\x7fELF")
            return SimpleNamespace(returncode=self.build_rc, stdout="building\n", stderr=self.build_out)
        raise AssertionError(f"unexpected call {argv}")

    def build_call(self):
        return next(kw for argv, kw in self.calls if argv[1:2] == ["-c"] and argv[2] == build_d3.BUILD_CODE)


@pytest.fixture()
def env_prefix(tmp_path: Path, monkeypatch) -> Path:
    """An env with python, nvcc and the conda compiler names; no system CUDA, no GPU."""
    prefix = tmp_path / "envs" / "uma"
    for tool in ("python", "nvcc", "ninja", "x86_64-conda-linux-gnu-gcc", "x86_64-conda-linux-gnu-g++"):
        _exe(prefix / "bin" / tool, "exit 0\n")
    monkeypatch.setattr(build_d3, "SYSTEM_CUDA", str(tmp_path / "no_system_cuda"))
    monkeypatch.setattr(build_d3, "host_compute_cap", lambda run=None: "12.0")
    return prefix


# ── TORCH_CUDA_ARCH_LIST ─────────────────────────────────────────────────────

def test_arch_list_is_env_sh_list_cut_to_what_nvcc_compiles():
    # env.sh 3b: catbench's default list plus this GPU
    assert build_d3.arch_list(None, "12.0", None) == build_d3.DEFAULT_ARCHS + ["12.0"]
    assert build_d3.arch_list(None, "8.9", None) == build_d3.DEFAULT_ARCHS
    # CUDA 13 nvcc has no compute_61/70: those are dropped instead of failing the build
    cu13 = {"75", "80", "86", "89", "90", "100", "120"}
    assert build_d3.arch_list(None, "12.0", cu13) == ["7.5", "8.0", "8.6", "8.9", "9.0", "12.0"]
    # a value already set is kept (and cut the same way)
    assert build_d3.arch_list("7.0;12.0+PTX", None, cu13) == ["12.0+PTX"]
    assert build_d3.arch_list("6.1", None, cu13) == []


def test_nvcc_archs_parses_list_gpu_arch():
    run = FakeRun(archs="compute_75\ncompute_89\ncompute_120\n")
    assert build_d3.nvcc_archs("nvcc", run=run) == {"75", "89", "120"}


# ── the build environment ────────────────────────────────────────────────────

def test_build_env_uses_the_env_own_tools(env_prefix, monkeypatch):
    base = {"PATH": "/usr/bin:/bin", "CUDA_HOME": "/opt/other-cuda", "HOME": "/h"}
    env, why = build_d3.build_env(env_prefix, base, run=FakeRun())
    assert why == ""
    assert env["PATH"].split(os.pathsep)[0] == str(env_prefix / "bin")
    assert env["CUDA_HOME"] == str(env_prefix)          # the env's nvcc wins over a system CUDA_HOME
    assert env["CC"].endswith("x86_64-conda-linux-gnu-gcc") and env["CXX"].endswith("x86_64-conda-linux-gnu-g++")
    assert env["TORCH_CUDA_ARCH_LIST"] == "7.5;12.0"     # default + 12.0, cut to the fake nvcc's list


def test_build_env_falls_back_to_cuda_home_and_leaves_cc_alone(tmp_path, monkeypatch):
    prefix = tmp_path / "env"
    _exe(prefix / "bin" / "python", "exit 0\n")
    system = tmp_path / "cuda"
    _exe(system / "bin" / "nvcc", "exit 0\n")
    monkeypatch.setattr(build_d3, "SYSTEM_CUDA", str(tmp_path / "none"))
    monkeypatch.setattr(build_d3, "host_compute_cap", lambda run=None: None)
    env, _ = build_d3.build_env(prefix, {"PATH": "/usr/bin", "CUDA_HOME": str(system)}, run=FakeRun())
    assert env["CUDA_HOME"] == str(system)
    assert env["PATH"].split(os.pathsep)[:2] == [str(prefix / "bin"), str(system / "bin")]
    assert "CC" not in env and "CXX" not in env


def test_build_env_without_any_nvcc(tmp_path, monkeypatch):
    prefix = tmp_path / "env"
    _exe(prefix / "bin" / "python", "exit 0\n")
    monkeypatch.setattr(build_d3, "SYSTEM_CUDA", str(tmp_path / "none"))
    env, why = build_d3.build_env(prefix, {"PATH": str(tmp_path / "empty")}, run=FakeRun())
    assert env is None and "no nvcc" in why


# ── prebuild states ──────────────────────────────────────────────────────────

def test_prebuild_builds_with_the_env_tools_and_logs(env_prefix, tmp_path):
    d3 = tmp_path / "site" / "catbench" / "dispersion" / "cuda"
    d3.mkdir(parents=True)
    run = FakeRun(d3_dir=d3)
    log = tmp_path / "logs" / "d3.log"
    state, detail = build_d3.prebuild(env_prefix, log, run=run, base={"PATH": "/usr/bin"})
    assert state == "built" and str(log) in detail and "in the catbench package folder" in detail
    env = run.build_call()["env"]
    assert env["CUDA_HOME"] == str(env_prefix) and env["PATH"].startswith(str(env_prefix / "bin"))
    assert "TORCH_CUDA_ARCH_LIST=7.5;12.0" in log.read_text()


def test_prebuild_is_a_no_op_when_the_kernel_exists(env_prefix, tmp_path):
    d3 = tmp_path / "cuda"
    d3.mkdir()
    (d3 / "pair_d3.so").write_bytes(b"\x7fELF")
    run = FakeRun(d3_dir=d3)
    state, detail = build_d3.prebuild(env_prefix, tmp_path / "d3.log", run=run, base={})
    assert state == "present" and detail.endswith("pair_d3.so")
    assert len(run.calls) == 1            # the probe only: no cache lookup, no nvcc query, no build
    assert not (tmp_path / "d3.log").exists()
    assert build_d3.read_marker(env_prefix)["state"] == "present"


@pytest.mark.parametrize("d3_dir,torch", [(None, True), ("x", False)])
def test_prebuild_without_torch_or_catbench_is_not_applicable(env_prefix, tmp_path, d3_dir, torch):
    run = FakeRun(d3_dir=tmp_path if d3_dir else None, torch=torch)
    state, _ = build_d3.prebuild(env_prefix, tmp_path / "d3.log", run=run, base={})
    assert state == "not_applicable" and len(run.calls) == 1


def test_prebuild_failure_names_the_log(env_prefix, tmp_path):
    run = FakeRun(d3_dir=tmp_path, build_rc=1, build_out="RuntimeError: Ninja is required to load C++ extensions\n")
    log = tmp_path / "d3.log"
    state, detail = build_d3.prebuild(env_prefix, log, run=run, base={})
    assert state == "failed" and "Ninja is required" in detail and str(log) in detail
    assert "Ninja is required" in log.read_text()
    marker = build_d3.read_marker(env_prefix)
    assert marker["state"] == "failed" and marker["log"] == str(log)
    assert marker["path"] == str(env_prefix / ".omm_d3")


def test_prebuild_without_nvcc_keeps_d3_off_without_failing(tmp_path, monkeypatch):
    prefix = tmp_path / "env"
    _exe(prefix / "bin" / "python", "exit 0\n")
    monkeypatch.setattr(build_d3, "SYSTEM_CUDA", str(tmp_path / "none"))
    run = FakeRun(d3_dir=tmp_path)
    state, detail = build_d3.prebuild(prefix, tmp_path / "d3.log", run=run, base={"PATH": str(tmp_path)})
    assert state == "no_nvcc" and "MLIP still runs" in detail


def test_cli_exit_codes(env_prefix, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(build_d3, "prebuild", lambda prefix, log: ("failed", "boom (log: x)"))
    assert build_d3.main([str(env_prefix)]) == 1
    assert "d3 kernel: failed: boom" in capsys.readouterr().err
    # an env already in use: a WARNING, the rerun command and the marker, exit 3
    assert build_d3.main([str(env_prefix), "--on-fail", "warn"]) == build_d3.WARN_EXIT == 3
    err = capsys.readouterr().err
    assert err.startswith(f"WARNING: D3 kernel build failed for {env_prefix}: boom (log: x)")
    assert f"Rerun: python3 {REPO / 'scripts' / 'build_d3.py'} {env_prefix}" in err
    assert f"Recorded in {env_prefix / '.omm_d3'}" in err
    for state in ("present", "built", "not_applicable", "no_nvcc"):
        monkeypatch.setattr(build_d3, "prebuild", lambda prefix, log, s=state: (s, "d"))
        assert build_d3.main([str(env_prefix)]) == 0
        assert build_d3.main([str(env_prefix), "--on-fail", "warn"]) == 0


# ── a shared hub: the catbench package folder is not writable ───────────────

def _readonly(monkeypatch, *paths):
    """os.access reports these paths (and only these) as not writable."""
    real = os.access
    ro = {str(p) for p in paths}
    monkeypatch.setattr(build_d3.os, "access",
                        lambda p, mode, real=real: False if (mode & os.W_OK and str(p) in ro) else real(p, mode))


def test_a_kernel_in_the_torch_extensions_cache_is_present(env_prefix, tmp_path, monkeypatch):
    pkg, cache = tmp_path / "pkg", tmp_path / "torch_extensions" / "py311_cu128" / "pair_d3"
    pkg.mkdir()
    cache.mkdir(parents=True)
    (cache / "pair_d3.so").write_bytes(b"")
    _readonly(monkeypatch, pkg)
    run = FakeRun(d3_dir=pkg, cache=cache)
    state, detail = build_d3.prebuild(env_prefix, tmp_path / "d3.log", run=run, base={})
    assert (state, detail) == ("present", str(cache / "pair_d3.so"))
    assert not any(argv[2] == build_d3.BUILD_CODE for argv, _ in run.calls if argv[1:2] == ["-c"])


def test_read_only_package_folder_builds_into_the_cache_and_says_so(env_prefix, tmp_path, monkeypatch):
    pkg, cache = tmp_path / "pkg", tmp_path / "torch_extensions" / "py311_cu128" / "pair_d3"
    pkg.mkdir()
    cache.mkdir(parents=True)
    _readonly(monkeypatch, pkg)
    run = FakeRun(d3_dir=pkg, cache=cache, build_into=cache)
    state, detail = build_d3.prebuild(env_prefix, tmp_path / "d3.log", run=run, base={})
    assert state == "built"
    assert detail.startswith(f"{cache / 'pair_d3.so'}, in the torch extensions cache, because the catbench "
                             f"package folder {pkg} is not writable, for 7.5;12.0")


def test_nothing_writable_is_a_clear_message_not_a_traceback(env_prefix, tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    _readonly(monkeypatch, pkg)
    err = "PermissionError: [Errno 13] Permission denied: '/home/u/.cache/torch_extensions'"
    run = FakeRun(d3_dir=pkg, cache=None, cache_error=err)
    state, detail = build_d3.prebuild(env_prefix, tmp_path / "d3.log", run=run, base={})
    assert state == "failed"
    assert detail == (f"cannot build the D3 kernel: the catbench package folder {pkg} is not writable, and "
                      f"neither is the torch extensions cache ({err}). Ask whoever owns {env_prefix} to run "
                      f"`python3 {REPO / 'scripts' / 'build_d3.py'} {env_prefix}`, or set TORCH_EXTENSIONS_DIR "
                      f"to a directory you can write and run it yourself")
    assert not any(argv[1:2] == ["-c"] and argv[2] == build_d3.BUILD_CODE for argv, _ in run.calls)
    assert "Traceback" not in detail


def test_marker_moves_to_the_user_cache_when_the_prefix_is_read_only(env_prefix, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    _readonly(monkeypatch, env_prefix)
    want = tmp_path / "xdg" / "oh-my-mlip" / "d3" / str(env_prefix.resolve()).strip("/").replace("/", "_") / ".omm_d3"
    assert build_d3.marker_path(env_prefix) == want
    build_d3.prebuild(env_prefix, None, run=FakeRun(d3_dir=None), base={})
    assert build_d3.read_marker(env_prefix)["state"] == "not_applicable"
    assert not (env_prefix / ".omm_d3").exists()


def test_probe_code_runs_and_finds_catbench_without_importing_it(tmp_path):
    """The probe really runs (in this python): finds torch + catbench's D3 module by spec only."""
    site = tmp_path / "site"
    (site / "torch").mkdir(parents=True)
    (site / "torch" / "__init__.py").write_text("raise SystemExit('torch must not be imported')\n")
    d3 = site / "catbench" / "dispersion" / "cuda"
    d3.mkdir(parents=True)
    (site / "catbench" / "__init__.py").write_text("raise SystemExit('catbench must not be imported')\n")
    (d3 / "d3_calculator.py").write_text("")
    env = dict(os.environ, PYTHONPATH=str(site))
    got = build_d3.probe(Path(sys.executable), env=env)
    assert got["applicable"] and got["d3_dir"] == str(d3) and not got["present"]
    (d3 / "pair_d3.so").write_bytes(b"")
    assert build_d3.probe(Path(sys.executable), env=env)["present"]


# ── install.sh and adopt_env.py run it ──────────────────────────────────────

def test_install_sh_prebuilds_d3_and_fails_the_env_on_a_failed_build():
    text = (REPO / "install.sh").read_text()
    func = text[text.index("build_d3_kernel() {"):text.index("install_one() {")]
    assert '"$OH_MY_MLIP_HOME/scripts/build_d3.py" "$prefix" --log "$log" --on-fail "$mode"' in func
    body = text[text.index("install_one() {"):]
    # a fresh build fails the env (before the sentinel); an installed env only warns
    assert body.count('build_d3_kernel "$env_name" "$prefix" || return 1\n') == 1
    assert body.index('build_d3_kernel "$env_name" "$prefix" || return 1\n') < body.index(': > "$sentinel"')
    assert body.count('build_d3_kernel "$env_name" "$prefix" warn || return 1') == 1
    assert "D3 warm-up" not in text


def _run_build_d3_kernel(tmp_path: Path, mode: str, rc: int) -> subprocess.CompletedProcess:
    """install.sh's build_d3_kernel + end-of-run summary, with build_d3.py stubbed."""
    text = (REPO / "install.sh").read_text()
    func = text[text.index("OMM_D3_WARNINGS=()"):text.index("install_one() {")]
    summary = text[text.index('if [ "${#OMM_D3_WARNINGS[@]}" -gt 0 ]; then'):text.index('if [ "$rc" -eq 0 ]; then\n  echo "install.sh: done.')]
    home = tmp_path / "hub"
    _exe(home / "scripts" / "build_d3.py", "")
    (home / "scripts" / "build_d3.py").write_text(f"import sys\nsys.exit({rc})\n")
    script = (f"OH_MY_MLIP_HOME={home}; THREAD_CAPS=(env)\n{func}\n"
              f"build_d3_kernel uma {tmp_path}/envs/uma {mode}; echo \"returned=$?\"\n{summary}")
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_install_sh_fresh_build_fails_on_a_failed_d3_build(tmp_path):
    proc = _run_build_d3_kernel(tmp_path, "error", 1)
    assert "returned=1" in proc.stdout
    assert "D3 kernel build FAILED for 'uma' — no sentinel written" in proc.stderr


def test_install_sh_installed_env_warns_and_keeps_going(tmp_path):
    proc = _run_build_d3_kernel(tmp_path, "warn", 3)
    assert "returned=0" in proc.stdout and "FAILED" not in proc.stderr
    assert "install.sh: WARNING: the D3 kernel could not be built for these already-installed envs" in proc.stderr
    assert f"uma (log: {tmp_path}/envs/uma/.omm_d3_build.log)" in proc.stderr
    # an unexpected error in warn mode (not the WARN exit) still fails
    assert "returned=1" in _run_build_d3_kernel(tmp_path, "warn", 1).stdout


def test_install_status_shows_a_failed_d3_marker(tmp_path):
    text = (REPO / "install.sh").read_text()
    assert "grep -qx 'state=failed' \"$prefix/.omm_d3\"" in text
    assert "D3 kernel build FAILED earlier" in text


def test_install_dry_run_shows_the_d3_build():
    env = {**os.environ, "OMM_INSTALL_ARCH": "sm120", "OH_MY_MLIP_HOME": str(REPO)}
    out = subprocess.run(["bash", str(REPO / "install.sh"), "--dry-run", "uma"], capture_output=True,
                         text=True, cwd=str(REPO), env=env).stdout
    assert f"then: python3 {REPO}/scripts/build_d3.py {REPO}/envs/uma" in out


def _adopt_home(tmp_path: Path) -> Path:
    home = tmp_path / "hub"
    home.mkdir()
    (home / "models.json").write_text(json.dumps({
        "_meta": {},
        "Alpha": {"env": "alpha", "python": "${OH_MY_MLIP_HOME}/envs/alpha/bin/python",
                  "import": ["import alpha"], "default_version": "A-1",
                  "versions": {"A-1": {"gated": False, "inference": ["calc = None"]}}},
    }))
    (home / "envs").mkdir()
    return home


def _adopt(home: Path, prefix: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / "adopt_env.py"), "Alpha", str(prefix)],
                          capture_output=True, text=True,
                          env={"OH_MY_MLIP_HOME": str(home), "PATH": "/usr/bin:/bin",
                               "TORCH_CUDA_ARCH_LIST": "12.0"})


def test_adoption_builds_d3_and_a_failed_build_warns_with_a_marker(tmp_path: Path):
    home = _adopt_home(tmp_path)
    prefix = tmp_path / "alpha_env"
    d3 = tmp_path / "site" / "catbench" / "dispersion" / "cuda"
    d3.mkdir(parents=True)
    probe = _probe_line(str(d3)).strip()
    # imports pass; the probe finds catbench; the D3 build fails like on the sm120 host
    _exe(prefix / "bin" / "python",
         'case "$2" in\n'
         f"  *OMM_D3_PROBE*) echo '{probe}' ;;\n"
         f"  *OMM_D3_CACHE*) echo 'OMM_D3_CACHE {{\"cache_dir\": null, \"error\": \"x\"}}' ;;\n"
         "  *_load*) echo 'RuntimeError: Ninja is required to load C++ extensions' >&2; exit 1 ;;\n"
         "esac\nexit 0\n")
    _exe(prefix / "bin" / "nvcc", 'echo "compute_75 compute_120"\n')
    proc = _adopt(home, prefix)
    assert proc.returncode == 0, proc.stderr                 # the env is in use: warn, do not fail
    assert "WARNING: D3 kernel build failed for alpha" in proc.stderr and "Ninja is required" in proc.stderr
    log = home / "logs" / "d3_alpha.log"
    assert f"python3 {REPO / 'scripts' / 'build_d3.py'} {prefix} --log {log}" in proc.stderr
    assert "TORCH_CUDA_ARCH_LIST=12.0" in log.read_text()
    assert json.loads((home / "env_map.local.json").read_text()) == {"alpha": str(prefix)}  # adoption stands
    marker = (prefix / ".omm_d3").read_text()
    assert "state=failed" in marker and f"log={log}" in marker


def test_adoption_of_an_env_without_catbench_d3_is_not_a_failure(tmp_path: Path):
    home = _adopt_home(tmp_path)
    prefix = tmp_path / "grace_like"
    _exe(prefix / "bin" / "python", "exit 0\n")
    proc = _adopt(home, prefix)
    assert proc.returncode == 0, proc.stderr
    assert "d3 kernel: not_applicable" in proc.stdout


# ── setup_verify: D3 is its own check item ──────────────────────────────────

def _d3_out(e=setup_verify.D3_REFERENCE["energy_ev"], f=setup_verify.D3_REFERENCE["abs_force_sum"]):
    return setup_verify.D3_MARK + json.dumps({"energy_ev": e, "abs_force_sum": f}) + "\n"


def test_judge_d3():
    assert setup_verify.judge_d3(0, "noise\n" + _d3_out(), "")["state"] == "pass"
    off = setup_verify.judge_d3(0, _d3_out(e=-21.2), "")
    assert off["state"] == "fail" and "energy_ev" in off["reason"]
    assert setup_verify.judge_d3(0, _d3_out(f=float("nan")), "")["state"] == "fail"
    crash = setup_verify.judge_d3(1, "", "RuntimeError: Ninja is required to load C++ extensions\n")
    assert crash["state"] == "fail" and "ninja is required" in crash["reason"]


def _spec(python: Path) -> dict:
    return {"python": str(python), "env_run": {"LD_LIBRARY_PATH": ""}}


@pytest.fixture()
def verify_env(tmp_path, monkeypatch):
    python = _exe(tmp_path / "env" / "bin" / "python", "exit 0\n")
    import oh_my_mlip.registry as registry
    monkeypatch.setattr(registry, "resolve", lambda model, version=None: _spec(python))
    return python


class VerifyRun(FakeRun):
    def __init__(self, d3_dir, stdout="", rc=0):
        super().__init__(d3_dir=d3_dir)
        self.out, self.rc = stdout, rc

    def __call__(self, argv, **kw):
        if argv[1:2] == ["-c"] and argv[2] == setup_verify.D3_CHECK_CODE:
            self.calls.append((argv, kw))
            return SimpleNamespace(returncode=self.rc, stdout=self.out, stderr="")
        return super().__call__(argv, **kw)


def test_check_d3_runs_through_the_bare_interpreter(verify_env, tmp_path):
    (tmp_path / "pair_d3.so").write_bytes(b"")
    run = VerifyRun(tmp_path, stdout=_d3_out())
    got = setup_verify.check_d3("X", None, {"skew": False}, run=run)
    assert got["state"] == "pass" and got["energy_ev"] == setup_verify.D3_REFERENCE["energy_ev"]
    env = run.calls[-1][1]["env"]
    assert env["LD_LIBRARY_PATH"] == ""                                   # env_run applied
    assert not env.get("PATH", "").startswith(str(verify_env.parent))    # no env bin injected


def test_check_d3_missing_kernel_is_a_fail_not_a_compile(verify_env, tmp_path):
    run = VerifyRun(tmp_path)
    got = setup_verify.check_d3("X", None, {"skew": False}, run=run)
    assert got["state"] == "fail" and "build_d3.py" in got["reason"] and "is not built" in got["reason"]
    assert all(argv[2] != setup_verify.D3_CHECK_CODE for argv, _ in run.calls)


def test_check_d3_surfaces_a_failed_build_marker(verify_env, tmp_path):
    prefix = verify_env.parent.parent
    build_d3.write_marker(prefix, "failed", "RuntimeError: Ninja is required", tmp_path / "d3.log")
    got = setup_verify.check_d3("X", None, {"skew": False}, run=VerifyRun(tmp_path))
    assert got["state"] == "fail" and "build failed at install/adopt time" in got["reason"]
    assert "Ninja is required" in got["reason"]
    assert got["build"] == {"state": "failed", "log": str(tmp_path / "d3.log"), "path": str(prefix / ".omm_d3")}
    setup_verify._print_verdict({"pass": True, "d3": got}, as_json=False)


def test_check_d3_finds_a_kernel_in_the_torch_extensions_cache(verify_env, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "pair_d3.so").write_bytes(b"")
    run = VerifyRun(tmp_path / "pkg", stdout=_d3_out())
    run.cache = cache
    assert setup_verify.check_d3("X", None, {"skew": False}, run=run)["state"] == "pass"


def test_check_d3_not_applicable_and_skipped(verify_env, tmp_path, monkeypatch):
    got = setup_verify.check_d3("G", None, {"skew": False}, run=VerifyRun(None))
    assert got["state"] == "not_applicable"
    (tmp_path / "pair_d3.so").write_bytes(b"")
    got = setup_verify.check_d3("X", None, {"skew": True, "reason": "driver"}, run=VerifyRun(tmp_path))
    assert got["state"] == "skipped" and "CUDA-only" in got["reason"]
    import oh_my_mlip.registry as registry
    monkeypatch.setattr(registry, "resolve", lambda model, version=None: _spec(tmp_path / "nope" / "python"))
    assert setup_verify.check_d3("X", None, {"skew": False}, run=VerifyRun(tmp_path))["state"] == "skipped"


def test_a_d3_fail_does_not_change_the_ef_verdict(tmp_path):
    """verify_one keeps `pass` for E/F and puts D3 under its own key."""
    text = (REPO / "scripts" / "setup_verify.py").read_text()
    body = text[text.index("def verify_one("):text.index("def _print_verdict(")]
    assert 'verdict["d3"] = check_d3(model, version, skew)' in body
    assert not re.search(r'verdict\["pass"\]\s*=.*d3', body)


# ── env.sh: the active conda env's nvcc ─────────────────────────────────────

def test_env_sh_finds_nvcc_under_conda_prefix(tmp_path):
    if shutil.which("nvcc", path="/usr/bin:/bin"):
        pytest.skip("this host has nvcc on /usr/bin:/bin, which env.sh rightly prefers")
    prefix = tmp_path / "env"
    _exe(prefix / "bin" / "nvcc", "exit 0\n")
    script = (f"export HOME={tmp_path}; export CONDA_PREFIX={prefix}; "
              f"source {REPO}/env.sh >/dev/null 2>&1; printf '%s\\n' \"${{CUDA_HOME:-UNSET}}\"")
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin"}).stdout.strip()
    assert out == str(prefix)
    # an explicit CUDA_HOME still wins
    out = subprocess.run(["bash", "-c", "export CUDA_HOME=/x; " + script], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin"}).stdout.strip()
    assert out == "/x"


# ── the sm120 siblings that add ninja + a pinned compiler ───────────────────

def _deps(text: str) -> tuple[list[str], list[str]]:
    """(conda dependency lines, pip lines) without comments."""
    lines = [ln.split("#", 1)[0].rstrip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln.strip()]
    start = next(i for i, ln in enumerate(lines) if ln.startswith("dependencies:"))
    pip = next(i for i, ln in enumerate(lines) if ln.strip() == "- pip:")
    return lines[start:pip], lines[pip:]


@pytest.mark.parametrize("stem,gxx", sorted(D3_SIBLINGS.items()))
def test_d3_sibling_is_its_base_plus_ninja_and_a_pinned_compiler(stem, gxx):
    env = stem.split(".")[0]
    base, sib = (ENVS / f"{env}.yml").read_text(), (ENVS / f"{stem}.yml").read_text()
    (bconda, bpip), (sconda, spip) = _deps(base), _deps(sib)
    assert spip == bpip
    assert [ln for ln in sconda if ln not in bconda] == [f"  - gxx={gxx}", "  - ninja=1.13.2"]
    assert set(bconda) <= set(sconda)
    assert sib.split("\nname:", 1)[1].split("\ndependencies:", 1)[0] == base.split("\nname:", 1)[1].split("\ndependencies:", 1)[0]
    # gcc 14 for CUDA 12.8's nvcc, gcc 15 for CUDA 13.0's
    nvcc = re.search(r"cuda-nvcc=(\d+\.\d+)", sib).group(1)
    assert {"12.8": "14.4.0", "13.0": "15.3.0"}[nvcc] == gxx
    table = json.loads((ENVS / "_arch_recipes.json").read_text())
    # selected on sm120, and marked candidate until the sibling itself is built and verified there
    assert table["sm120"][env]["recipe"] == stem and table["sm120"][env]["status"] == "candidate"
    assert sib.splitlines()[0].startswith(f"# base + ninja + pinned gxx {gxx} for sm120 (D3 kernel build)")
