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

States (printed as ``d3 kernel: <state>: <detail>``):
  present          pair_d3.so already exists: nothing is done (exit 0)
  built            built now (exit 0)
  not_applicable   the env has no torch or no catbench D3 module (GRACE): exit 0
  no_nvcc          no nvcc anywhere: D3 stays off for this env, the MLIP still
                   runs (exit 0; the same degrade install.sh always had)
  failed           the build failed: exit 1, with the log path

Usage:
  python3 scripts/build_d3.py <env prefix> [--log <file>]
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
    env.setdefault("PYTHONUTF8", "1")
    return env, ""


def prebuild(prefix: Path, log: Path, run=subprocess.run, base: dict | None = None) -> tuple[str, str]:
    """(state, detail) for one env; see the module docstring for the states."""
    base = dict(os.environ if base is None else base)
    python = prefix / "bin" / "python"
    if not os.access(python, os.X_OK):
        return "failed", f"no interpreter at {python}"
    found = probe(python, run=run, env=base)
    if found["error"]:
        return "failed", found["error"]
    if not found["applicable"]:
        return "not_applicable", "no torch or no catbench D3 module in this env"
    if found["present"]:
        return "present", found["so"]
    env, why = build_env(prefix, base, run=run)
    if env is None:
        return "no_nvcc", f"{why}; D3 stays off for this env (the MLIP still runs)"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = run([str(python), "-c", BUILD_CODE], capture_output=True, text=True,
                   timeout=BUILD_TIMEOUT_S, env=env)
        rc, text = proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        rc, text = 1, f"{exc}\n"
    settings = (f"CUDA_HOME={env['CUDA_HOME']}\nCC={env.get('CC', '')}\nCXX={env.get('CXX', '')}\n"
                f"TORCH_CUDA_ARCH_LIST={env['TORCH_CUDA_ARCH_LIST']}\nPATH={env['PATH']}\n\n")
    log.write_text(settings + text, encoding="utf-8")
    if rc != 0:
        tail = [ln for ln in text.splitlines() if ln.strip()][-1:] or [f"exit {rc}"]
        return "failed", f"{tail[0]} (log: {log})"
    return "built", f"{found['so']} for {env['TORCH_CUDA_ARCH_LIST']} (log: {log})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prefix", type=Path, help="env prefix (contains bin/python)")
    ap.add_argument("--log", type=Path, default=None,
                    help="build log (default: <prefix>/.omm_d3_build.log)")
    args = ap.parse_args(argv)
    prefix = args.prefix.expanduser().absolute()
    log = args.log or prefix / ".omm_d3_build.log"
    state, detail = prebuild(prefix, log)
    print(f"d3 kernel: {state}: {detail}", file=sys.stderr if state == "failed" else sys.stdout)
    return 1 if state == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
