#!/usr/bin/env python3
"""build_d3.py -- build catbench's D3 kernel (pair_d3.so) inside one env, once.

catbench compiles ``catbench/dispersion/cuda/pair_d3.so`` with torch's
``cpp_extension.load()`` on the FIRST D3 call. That call usually comes through
the env's interpreter run directly (``resolve()["python"]``), so the env's own
``bin/`` is not on PATH: torch cannot find ``ninja``, ``nvcc`` is not found (or a
system toolkit of another CUDA version is), and nvcc picks the system gcc (gcc
15 on Ubuntu 26.04, which CUDA 12.x refuses). Seen on an sm120 host: the first
D3 call raised "Ninja is required to load C++ extensions".

This script does that build at install / adopt time instead, with the env's
own tools:
  PATH                 = <prefix>/bin first (ninja, nvcc, the conda compiler)
  CUDA_HOME            = <prefix> when <prefix>/bin/nvcc exists (the toolkit the
                         recipe pins to the env's torch CUDA build); otherwise
                         CUDA_HOME / nvcc on PATH / /usr/local/cuda, as env.sh
  CC / CXX             = <prefix>/bin/x86_64-conda-linux-gnu-gcc / -g++ if present
  TORCH_CUDA_ARCH_LIST = env.sh 3b's list (catbench's default 6.1-9.0 plus this
                         GPU's compute capability), or the value already set,
                         reduced to the architectures this nvcc can compile
                         (CUDA 13 dropped everything below 7.5)
pair_d3.so then sits in the package folder and nothing compiles at run time.

Where the kernel goes: catbench builds into its package folder, and only when
that folder is not writable (a shared, read-only hub) into torch's extensions
cache (TORCH_EXTENSIONS_DIR, default ~/.cache/torch_extensions/<py>_<cu>/pair_d3),
where it also looks first at load time. A kernel in either place is `present`;
`built` says which place; when neither is writable the build is not attempted
and the message says what to do (no traceback).

States (printed as ``d3 kernel: <state>: <detail>``):
  present          pair_d3.so already exists: nothing is done (exit 0)
  built            built now (exit 0)
  not_applicable   the env has no torch or no catbench D3 module (GRACE): exit 0
  no_nvcc          no nvcc anywhere: D3 stays off for this env, the MLIP still
                   runs (exit 0; the same degrade install.sh always had)
  failed           no kernel: exit 1 (``--on-fail warn``: a WARNING with the log
                   path and the rerun command, exit 3)

Every run leaves its state in a marker file, ``<prefix>/.omm_d3`` (or, when the
prefix is not writable, ``~/.cache/oh-my-mlip/d3/<prefix path>/.omm_d3``):
``state=``, ``detail=``, ``log=`` lines. setup_verify and ``install.sh --status``
show a ``failed`` marker, so a D3 failure that did not stop an install is not silent.

Usage:
  python3 scripts/build_d3.py <env prefix> [--log <file>] [--on-fail error|warn]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# catbench's own default when TORCH_CUDA_ARCH_LIST is unset (d3_calculator._load)
DEFAULT_ARCHS = ["6.1", "7.0", "7.5", "8.0", "8.6", "8.9", "9.0"]
BUILD_TIMEOUT_S = 1800
SYSTEM_CUDA = "/usr/local/cuda"
LIB_NAME = "pair_d3.so"

PROBE_MARK = "OMM_D3_PROBE "
# Runs inside the env: finds torch and catbench's D3 module without importing them.
PROBE_CODE = f"""
import importlib.util as u, json, os
d = None
try:
    c = u.find_spec("catbench")
    if c is not None and c.submodule_search_locations:
        p = os.path.join(list(c.submodule_search_locations)[0], "dispersion", "cuda")
        if os.path.isfile(os.path.join(p, "d3_calculator.py")):
            d = p
    t = u.find_spec("torch") is not None
except Exception:
    t = False
print({PROBE_MARK!r} + json.dumps({{"torch": t, "d3_dir": d}}))
"""
BUILD_CODE = "from catbench.dispersion.cuda.d3_calculator import _load; _load('pair_d3')"

CACHE_MARK = "OMM_D3_CACHE "
# Runs inside the env (imports torch): the torch-extensions cache directory catbench
# falls back to, by torch's own function. It creates the directory when it can, as
# catbench's _load does before every build; a PermissionError means "not writable".
CACHE_CODE = f"""
import json
try:
    from torch.utils.cpp_extension import _get_build_directory
    d, err = _get_build_directory("pair_d3", verbose=False), ""
except Exception as exc:
    d, err = None, f"{{type(exc).__name__}}: {{exc}}"
print({CACHE_MARK!r} + json.dumps({{"cache_dir": d, "error": err}}))
"""
WARN_EXIT = 3
# torch's cpp_extension guards a build with a FileBaton: a file "lock" in the
# build folder, and every later build waits on it with no timeout. A build that
# was killed (a login node's task cap, a dropped session) leaves it behind, and
# the next build then waited until BUILD_TIMEOUT_S (seen on an HPC login node).
# A real build takes about a minute; a lock older than this is treated as left over.
STALE_LOCK_S = 600


def clear_stale_lock(build_dir: str, now: float | None = None) -> str:
    """Remove build_dir/lock when it is older than STALE_LOCK_S; say what was done."""
    import time

    lock = os.path.join(build_dir, "lock")
    try:
        age = (time.time() if now is None else now) - os.stat(lock).st_mtime
    except OSError:
        return ""
    if age < STALE_LOCK_S:
        return ""
    try:
        os.remove(lock)
    except OSError as exc:
        return f"a build lock {lock} from {age:.0f} s ago could not be removed ({exc}); remove it and rerun"
    return f"removed a build lock left by an earlier build that did not finish ({lock}, {age:.0f} s old)"


def probe(python: Path, run=subprocess.run, env: dict | None = None) -> dict:
    """{"applicable": bool, "d3_dir": str|None, "so": str|None, "present": bool, "error": str}."""
    out = {"applicable": False, "d3_dir": None, "so": None, "present": False, "error": ""}
    try:
        proc = run([str(python), "-c", PROBE_CODE], capture_output=True, text=True,
                   timeout=300, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        out["error"] = f"probe could not run {python}: {exc}"
        return out
    if proc.returncode != 0:
        tail = (proc.stderr.strip().splitlines() or [f"exit {proc.returncode}"])[-1]
        out["error"] = f"probe failed in {python}: {tail}"
        return out
    found = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith(PROBE_MARK):
            try:
                found = json.loads(line[len(PROBE_MARK):])
            except ValueError:
                found = None
    if not found or not found.get("torch") or not found.get("d3_dir"):
        return out
    so = os.path.join(found["d3_dir"], LIB_NAME)
    out.update(applicable=True, d3_dir=found["d3_dir"], so=so, present=os.path.isfile(so))
    return out


def cache_dir(python: Path, run=subprocess.run, env: dict | None = None) -> tuple[str | None, str]:
    """(torch extensions cache dir for pair_d3, error) as torch computes it in this env."""
    try:
        proc = run([str(python), "-c", CACHE_CODE], capture_output=True, text=True, timeout=300, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    for line in (proc.stdout or "").splitlines():
        if line.startswith(CACHE_MARK):
            try:
                got = json.loads(line[len(CACHE_MARK):])
            except ValueError:
                break
            return got.get("cache_dir"), got.get("error") or ""
    tail = ((proc.stderr or "").strip().splitlines() or [f"exit {proc.returncode}"])[-1]
    return None, tail


def locate(python: Path, run=subprocess.run, env: dict | None = None) -> dict:
    """probe() plus the torch extensions cache: `present` / `so` cover both places
    catbench loads from; `cache_dir` / `cache_error` describe the fallback place."""
    found = probe(python, run=run, env=env)
    found.update(cache_dir=None, cache_error="")
    if found["error"] or not found["applicable"] or found["present"]:
        return found
    found["cache_dir"], found["cache_error"] = cache_dir(python, run=run, env=env)
    if found["cache_dir"]:
        cached = os.path.join(found["cache_dir"], LIB_NAME)
        if os.path.isfile(cached):
            found.update(so=cached, present=True)
    return found


def state_dir(prefix: Path) -> Path:
    """Where this env's marker (and default build log) live: the prefix, or a
    per-user cache directory when the prefix is not writable (a shared hub)."""
    if os.access(prefix, os.W_OK):
        return prefix
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return root / "oh-my-mlip" / "d3" / str(prefix.resolve()).strip("/").replace("/", "_")


def marker_path(prefix: Path) -> Path:
    return state_dir(prefix) / ".omm_d3"


def write_marker(prefix: Path, state: str, detail: str, log: Path | None) -> Path | None:
    path = marker_path(prefix)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"state={state}\ndetail={detail}\nlog={log or ''}\n", encoding="utf-8")
    except OSError:
        return None
    return path


def read_marker(prefix: Path) -> dict:
    """{"state", "detail", "log", "path"} from the marker, or {} when there is none."""
    path = marker_path(prefix)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    out = dict(ln.split("=", 1) for ln in lines if "=" in ln)
    out["path"] = str(path)
    return out


def find_cuda_home(prefix: Path, env: dict) -> str | None:
    """The env's own toolkit first, then env.sh's order (CUDA_HOME, nvcc on PATH,
    /usr/local/cuda). Only a directory whose bin/nvcc exists counts."""
    candidates = [str(prefix)]
    if env.get("CUDA_HOME"):
        candidates.append(env["CUDA_HOME"])
    nvcc = shutil.which("nvcc", path=env.get("PATH", ""))
    if nvcc:
        candidates.append(str(Path(nvcc).resolve().parent.parent))
    candidates.append(SYSTEM_CUDA)
    for home in candidates:
        if os.access(os.path.join(home, "bin", "nvcc"), os.X_OK):
            return home
    return None


def host_compute_cap(run=subprocess.run) -> str | None:
    """This GPU's compute capability ("12.0"), as env.sh 3b reads it."""
    smi = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi"
                                         if os.access("/usr/lib/wsl/lib/nvidia-smi", os.X_OK) else None)
    if not smi:
        return None
    try:
        proc = run([smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
                   capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    cc = (proc.stdout or "").strip().splitlines()[0].strip() if (proc.stdout or "").strip() else ""
    return cc if re.fullmatch(r"\d+\.\d+", cc) else None


def nvcc_archs(nvcc: str, run=subprocess.run) -> set[str] | None:
    """Virtual architectures this nvcc compiles ({"75", "120", ...}); None if unknown."""
    try:
        proc = run([nvcc, "--list-gpu-arch"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    found = set(re.findall(r"compute_(\d+)", proc.stdout or ""))
    return found or None


def arch_list(preset: str | None, host_cc: str | None, supported: set[str] | None) -> list[str]:
    """env.sh 3b's TORCH_CUDA_ARCH_LIST, cut to what this nvcc supports."""
    if preset:
        archs = [a for a in re.split(r"[;\s]+", preset.strip()) if a]
    else:
        archs = list(DEFAULT_ARCHS)
        if host_cc and host_cc not in archs:
            archs.append(host_cc)
    if supported is None:
        return archs
    return [a for a in archs if a.split("+")[0].replace(".", "") in supported]


def build_env(prefix: Path, base: dict, run=subprocess.run) -> tuple[dict | None, str]:
    """The environment for the build, or (None, reason) when nvcc is missing."""
    env = dict(base)
    rest = [base["PATH"]] if base.get("PATH") else []
    env["PATH"] = os.pathsep.join([str(prefix / "bin")] + rest)
    cuda_home = find_cuda_home(prefix, env)
    if cuda_home is None:
        return None, f"no nvcc in {prefix}/bin, CUDA_HOME, PATH or {SYSTEM_CUDA}"
    env["CUDA_HOME"] = cuda_home
    if cuda_home != str(prefix):
        env["PATH"] = os.pathsep.join([str(prefix / "bin"), os.path.join(cuda_home, "bin")] + rest)
    gcc = prefix / "bin" / "x86_64-conda-linux-gnu-gcc"
    gxx = prefix / "bin" / "x86_64-conda-linux-gnu-g++"
    if os.access(gcc, os.X_OK) and os.access(gxx, os.X_OK):
        env["CC"], env["CXX"] = str(gcc), str(gxx)
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    archs = arch_list(base.get("TORCH_CUDA_ARCH_LIST"), host_compute_cap(run), nvcc_archs(nvcc, run))
    if not archs:
        return None, f"{nvcc} supports none of the GPU architectures asked for"
    env["TORCH_CUDA_ARCH_LIST"] = ";".join(archs)
    env.setdefault("MAX_JOBS", "4")
    # importing torch/numpy starts one BLAS thread per core (128 on an HPC node),
    # past a login node's task cap; the build needs none of them
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(key, "1")
    env.setdefault("PYTHONUTF8", "1")
    return env, ""


def prebuild(prefix: Path, log: Path | None = None, run=subprocess.run,
             base: dict | None = None) -> tuple[str, str]:
    """(state, detail) for one env; see the module docstring. Leaves the marker."""
    log = log or state_dir(prefix) / ".omm_d3_build.log"
    state, detail, logged = _prebuild(prefix, log, run, dict(os.environ if base is None else base))
    write_marker(prefix, state, detail, log if logged else None)
    return state, detail


def _prebuild(prefix: Path, log: Path, run, base: dict) -> tuple[str, str, bool]:
    python = prefix / "bin" / "python"
    if not os.access(python, os.X_OK):
        return "failed", f"no interpreter at {python}", False
    found = locate(python, run=run, env=base)
    if found["error"]:
        return "failed", found["error"], False
    if not found["applicable"]:
        return "not_applicable", "no torch or no catbench D3 module in this env", False
    if found["present"]:
        return "present", found["so"], False
    if os.access(found["d3_dir"], os.W_OK):
        target, where = found["d3_dir"], "the catbench package folder"
    elif found["cache_dir"] and os.access(found["cache_dir"], os.W_OK):
        target, where = found["cache_dir"], ("the torch extensions cache, because the catbench package "
                                             f"folder {found['d3_dir']} is not writable")
    else:
        cache = found["cache_dir"] or f"the torch extensions cache ({found['cache_error'] or 'unknown'})"
        return "failed", (f"cannot build the D3 kernel: the catbench package folder {found['d3_dir']} is not "
                          f"writable, and neither is {cache}. Ask whoever owns {prefix} to run "
                          f"`python3 {Path(__file__).resolve()} {prefix}`, or set TORCH_EXTENSIONS_DIR to a "
                          f"directory you can write and run it yourself"), False
    env, why = build_env(prefix, base, run=run)
    if env is None:
        return "no_nvcc", f"{why}; D3 stays off for this env (the MLIP still runs)", False
    if shutil.which("ninja", path=env["PATH"]) is None:
        return "failed", (f"no ninja in {prefix}/bin or on PATH; torch builds the kernel with it. Install it into "
                          f"the env ({prefix}/bin/pip install ninja==1.13.0) and rerun"), False
    cleared = clear_stale_lock(target)
    if cleared.startswith("a build lock"):
        return "failed", cleared, False
    if cleared:
        print(f"d3 kernel: {cleared}", flush=True)
    try:
        proc = run([str(python), "-c", BUILD_CODE], capture_output=True, text=True,
                   timeout=BUILD_TIMEOUT_S, env=env)
        rc, text = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        rc, text = 1, f"{exc}\n"
    settings = (f"CUDA_HOME={env['CUDA_HOME']}\nCC={env.get('CC', '')}\nCXX={env.get('CXX', '')}\n"
                f"TORCH_CUDA_ARCH_LIST={env['TORCH_CUDA_ARCH_LIST']}\nPATH={env['PATH']}\n"
                f"target={target}\n\n")
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(settings + text, encoding="utf-8")
        logged, where_log = True, f"log: {log}"
    except OSError as exc:
        logged, where_log = False, f"no log: {log} not writable ({exc})"
    so = os.path.join(target, LIB_NAME)
    if rc != 0 or not os.path.isfile(so):
        tail = [ln for ln in text.splitlines() if ln.strip()][-1:] or [f"exit {rc}, no {so}"]
        return "failed", f"{tail[0]} ({where_log})", logged
    return "built", f"{so}, in {where}, for {env['TORCH_CUDA_ARCH_LIST']} ({where_log})", logged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prefix", type=Path, help="env prefix (contains bin/python)")
    ap.add_argument("--log", type=Path, default=None,
                    help="build log (default: <prefix>/.omm_d3_build.log, or the marker's directory)")
    ap.add_argument("--on-fail", choices=("error", "warn"), default="error",
                    help="error: exit 1 (a fresh install); warn: WARNING + exit 3, for an env "
                         "that is already in use (install.sh re-run, adopt_env.py)")
    args = ap.parse_args(argv)
    prefix = args.prefix.expanduser().absolute()
    state, detail = prebuild(prefix, args.log)
    if state != "failed":
        print(f"d3 kernel: {state}: {detail}")
        return 0
    if args.on_fail == "error":
        print(f"d3 kernel: failed: {detail}", file=sys.stderr)
        return 1
    print(f"WARNING: D3 kernel build failed for {prefix}: {detail}\n"
          f"  The env stays usable; only the D3 correction does not work until this succeeds.\n"
          f"  Rerun: python3 {Path(__file__).resolve()} {prefix}\n"
          f"  Recorded in {marker_path(prefix)} (setup_verify reports it).", file=sys.stderr)
    return WARN_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
