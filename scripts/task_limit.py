#!/usr/bin/env python3
"""task_limit.py -- the per-user task limit of this host, and the low-parallel
settings install.sh exports when it is low.

HPC login nodes often cap tasks (processes + threads) per user with the cgroup
pids controller, e.g. pids.max = 150 on a EuroHPC login node, while `ulimit -u`
still reports a huge number. An install there fans out far past the cap:
mamba's .pyc compile starts os.cpu_count() worker processes (256 on that node)
and deadlocks when fork fails, numpy/OpenBLAS starts one thread per core,
ninja and pip builds run one job per core. Every task over the cap also takes
the user's shell, their agent (Claude Code died with "fork: retry") and new
logins with it.

The limit is the smallest numeric pids.max on the path of this process's cgroup
(v2 "0::<path>" or the v1 "pids" controller line of /proc/self/cgroup), walked
up to the root, plus the user slice user.slice/user-<uid>.slice, and
`ulimit -u`. When it is below LOW_LIMIT, or OMM_MAX_PARALLEL is set, the
settings below cap every pool the install starts to OMM_MAX_PARALLEL (default
2). A value the user already exported wins.

Usage:
  python3 scripts/task_limit.py            # one line: limit=<n|none> low=<0|1> parallel=<n>
  python3 scripts/task_limit.py --exports  # shell `export` lines (empty when not low)

Runs on any Python 3 (3.6 too: it is the one helper whose failure must not
leave a host unprotected; install.sh still falls back to low-parallel mode if it
cannot run). Tests point OMM_CGROUP_ROOT and OMM_PROC_CGROUP at a fake tree.
"""
import argparse
import os
import resource
import shlex
import sys
from pathlib import Path

LOW_LIMIT = 4096        # tasks per user below which the install runs in low-parallel mode
DEFAULT_PARALLEL = 2


def _read_max(path):
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None   # "max" = no limit


def cgroup_paths(proc_cgroup):
    """The pids-controller cgroup paths of this process (v2 and/or v1)."""
    try:
        lines = proc_cgroup.read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == "" or "pids" in controllers.split(","):
            out.append((controllers, path))
    return [f"{'pids' if c else ''}|{p}" for c, p in out]


def cgroup_limit(root, proc_cgroup, uid):
    """Smallest numeric pids.max on this process's cgroup path and the user slice."""
    candidates = []
    for entry in cgroup_paths(proc_cgroup):
        sub, path = entry.split("|", 1)
        base = root / sub if sub else root
        rel = Path(path.lstrip("/"))
        while True:
            candidates.append(base / rel / "pids.max")
            if rel == Path("."):
                break
            rel = rel.parent
    for sub in ("", "pids"):
        candidates.append(root / sub / "user.slice" / f"user-{uid}.slice" / "pids.max")
    values = [v for v in (_read_max(p) for p in candidates) if v is not None]
    return min(values) if values else None


def ulimit_tasks():
    try:
        soft, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    except (OSError, ValueError):
        return None
    return None if soft == resource.RLIM_INFINITY else int(soft)


def task_limit(env=None):
    env = os.environ if env is None else env
    root = Path(env.get("OMM_CGROUP_ROOT", "/sys/fs/cgroup"))
    proc = Path(env.get("OMM_PROC_CGROUP", "/proc/self/cgroup"))
    values = [cgroup_limit(root, proc, os.getuid())]
    if "OMM_CGROUP_ROOT" not in env:       # a fake tree in tests stands alone
        values.append(ulimit_tasks())
    values = [v for v in values if v is not None]
    return min(values) if values else None


def settings(parallel):
    """Everything an install starts that sizes a pool by core count."""
    n = str(parallel)
    return {
        # mamba 2.x compiles .pyc in a ProcessPoolExecutor sized by
        # MAMBA_EXTRACT_THREADS, but passes 0 (= cpu_count) to it whatever is set;
        # only skipping the compile avoids the pool. Python writes __pycache__ on import.
        "MAMBA_COMPILE_PYC": "false",
        "MAMBA_EXTRACT_THREADS": n,
        "MAMBA_DOWNLOAD_THREADS": n,
        "CONDA_FETCH_THREADS": n,
        "CONDA_EXTRACT_THREADS": n,
        "CONDA_VERIFY_THREADS": "1",
        "CONDA_EXECUTE_THREADS": "1",
        "CONDA_REPODATA_THREADS": n,
        "MAX_JOBS": n,                       # torch cpp_extension / ninja
        "CMAKE_BUILD_PARALLEL_LEVEL": n,
        "MAKEFLAGS": f"-j{n}",
        "OMP_NUM_THREADS": n,
        "MKL_NUM_THREADS": n,
        "OPENBLAS_NUM_THREADS": n,
        "NUMEXPR_NUM_THREADS": n,
        "GOTO_NUM_THREADS": n,
        "VECLIB_MAXIMUM_THREADS": n,
        "RAYON_NUM_THREADS": n,
        "TF_NUM_INTRAOP_THREADS": n,
        "TF_NUM_INTEROP_THREADS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "UV_CONCURRENT_DOWNLOADS": n,
        "UV_CONCURRENT_BUILDS": "1",
        "UV_CONCURRENT_INSTALLS": n,
    }


def plan(env=None):
    """(limit, low, parallel, exports) -- exports omit variables already set."""
    env = os.environ if env is None else env
    limit = task_limit(env)
    forced = env.get("OMM_MAX_PARALLEL", "").strip()
    parallel = int(forced) if forced.isdigit() and int(forced) > 0 else DEFAULT_PARALLEL
    low = bool(forced) or (limit is not None and limit < LOW_LIMIT)
    exports = {k: v for k, v in settings(parallel).items() if k not in env} if low else {}
    return limit, low, parallel, exports


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exports", action="store_true", help="print shell export lines")
    args = ap.parse_args(argv)
    limit, low, parallel, exports = plan()
    if args.exports:
        for key, value in exports.items():
            print(f"export {key}={shlex.quote(value)}")
        if low:
            print(f"export OMM_TASK_LIMIT={limit if limit is not None else 'unknown'} OMM_LOW_PARALLEL={parallel}")
        return 0
    print(f"limit={limit if limit is not None else 'none'} low={int(low)} parallel={parallel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
