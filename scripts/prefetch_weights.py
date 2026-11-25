#!/usr/bin/env python3
"""prefetch_weights.py -- download every weight of one env's variants at install
time, so the env also runs on a node without internet.

HPC compute nodes often have no internet (the GPU nodes of some EuroHPC
systems have none), while the login node, where the install runs, has. A variant whose
weight is fetched by name on first use (mace_mp(model='medium-mpa-0'),
SevenNetCalculator('7net-mf-ompa'), ORB, CHGNet, ...) then fails in the GPU job.
install.sh runs this script with the ENV's interpreter right after the env is
built:

  - weights at an explicit $OH_MY_MLIP_HOME/models/... path: oh_my_mlip.fetch
    downloads them (the same code get_calculator uses);
  - by-name weights: the variant's registry import + inference lines run once in
    a child process with the device set to the CPU (device='cuda' -> 'cpu',
    cpu=False -> cpu=True) and CUDA_VISIBLE_DEVICES empty. That builds the
    calculator, which is when the framework downloads its weight; nothing is
    calculated;
  - gated variants without a Hugging Face token, and NequIP / Allegro (their
    per-GPU compile comes from prepare_<env>_weights.py), are skipped and listed.

A variant whose CPU build fails is reported with the error's last line: the
weight may still have been downloaded before the failure (a GPU-only kernel
option, for example), and the GPU verify job on an offline node tells for sure.

Writes <prefix>/.omm_prefetch.json and prints one line per variant.
Exit 0 when nothing failed (skips included), 1 otherwise.

Usage (env interpreter):  <prefix>/bin/python scripts/prefetch_weights.py <env>
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HOME))

from oh_my_mlip import fetch, registry  # noqa: E402

BUILD_TIMEOUT_S = 1800
_DEVICE_RE = re.compile(r"""device\s*=\s*(['"])cuda(:\d+)?\1""")
_CPU_FALSE_RE = re.compile(r"\bcpu\s*=\s*False\b")


def cpu_lines(lines: list[str]) -> list[str]:
    """The inference lines with the GPU device switched to the CPU."""
    return [_CPU_FALSE_RE.sub("cpu=True", _DEVICE_RE.sub("device='cpu'", line)) for line in lines]


def variants(env_name: str, models: dict | None = None) -> list[tuple[str, str]]:
    models = registry.load_models() if models is None else models
    out = []
    for family, info in models.items():
        if family.startswith("_") or not isinstance(info, dict) or info.get("env") != env_name:
            continue
        out.extend((family, version) for version in info.get("versions", {}))
    return out


_DENIED_RE = re.compile(r"Permission denied: '([^']+)'")


def complete_weight_refused(lines: list[str], spec: dict) -> str | None:
    """The path of a weight the framework tried to write again although it is
    complete: the build failed with PermissionError on a file whose size and
    sha256 match the registry (seen with TACE: the finished .pt was read-only)."""
    size, digest = spec.get("weights_size"), spec.get("weights_sha256")
    if not size:
        return None
    for line in reversed(lines):
        match = _DENIED_RE.search(line)
        if not match:
            continue
        path = Path(match.group(1)).expanduser()
        try:
            if path.stat().st_size != int(size):
                return None
        except OSError:
            return None
        if digest:
            import hashlib
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != digest:
                return None
        return str(path)
    return None


def prefetch_one(family: str, version: str, run=subprocess.run, env: dict | None = None) -> tuple[str, str]:
    """(state, detail): state is ok / skipped / failed."""
    env = dict(os.environ if env is None else env)
    spec = registry.resolve(family, version=version)
    if spec.get("arch_pinned"):
        return "skipped", "compiled per GPU by prepare_<env>_weights.py (its package is downloaded there)"
    if spec.get("gated") and fetch._resolve_token(env)["source"] == "none":
        return "skipped", ("gated: no Hugging Face token found, so it cannot be downloaded now; after `hf auth "
                           "login`, run install.sh for this env again (docs/hf_token.md)")
    try:
        fetch.ensure_weights(family, version=version, spec=spec)
    except Exception as exc:  # noqa: BLE001 - reported per variant
        return "failed", f"download: {exc}".splitlines()[0]
    if fetch._inference_weight_targets(spec):
        return "ok", "weights at their registry path"
    code = "\n".join(list(spec["imports"]) + cpu_lines(list(spec["inference"])))
    child = dict(env)
    child.update(spec.get("env_run") or {})
    child["CUDA_VISIBLE_DEVICES"] = ""
    try:
        proc = run([sys.executable, "-c", code], capture_output=True, text=True,
                   timeout=BUILD_TIMEOUT_S, env=child)
    except subprocess.TimeoutExpired:
        return "failed", f"building the calculator on the CPU took over {BUILD_TIMEOUT_S} s"
    if proc.returncode == 0:
        return "ok", "downloaded by the framework (calculator built on the CPU)"
    lines = [ln for ln in ((proc.stderr or "") + (proc.stdout or "")).splitlines() if ln.strip()]
    done = complete_weight_refused(lines, spec)
    if done:
        return "ok", (f"weight already complete at {done} (size and sha256 match the registry); the "
                      f"framework's attempt to write it again was refused, which does not matter")
    return "failed", (f"calculator build on the CPU failed: {lines[-1] if lines else f'exit {proc.returncode}'} "
                      f"(the weight may still be cached; the GPU verify job on an offline node tells)")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    env_name = argv[0]
    found = variants(env_name)
    if not found:
        print(f"prefetch_weights: no registered variant uses env {env_name!r}", file=sys.stderr)
        return 2
    results = {}
    for family, version in found:
        start = time.monotonic()
        state, detail = prefetch_one(family, version)
        results[version] = {"state": state, "detail": detail, "seconds": round(time.monotonic() - start, 1)}
        print(f"  weights {version}: {state}: {detail}", flush=True)
    failed = [v for v, r in results.items() if r["state"] == "failed"]
    prefix = Path(sys.executable).resolve().parent.parent
    try:
        (prefix / ".omm_prefetch.json").write_text(
            json.dumps({"env": env_name, "ok": not failed, "variants": results}, indent=1) + "\n",
            encoding="utf-8")
    except OSError as exc:
        print(f"prefetch_weights: could not write {prefix}/.omm_prefetch.json ({exc})", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
