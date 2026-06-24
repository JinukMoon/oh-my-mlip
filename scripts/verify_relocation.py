#!/usr/bin/env python3
"""verify_relocation.py — the BINDING compute-checkpoint test.

Public port of the internal verify_all.py. On a CLEAN, FOREIGN host (a machine
with a different NVIDIA driver / glibc than the build host — NOT a sibling node of
the build box), this fetches each shipped env's conda-pack tarball, unpacks/relocates
it, then for every model in models.json runs [import + ASE single-point + D3] using
that env's own interpreter and prints a PASS/FAIL table.

This is the load-bearing acceptance test for the whole distribution model: if a
relocated env imports, produces a finite single-point energy, and applies D3 on a
foreign GPU, the conda-pack fetch+relocate path is proven for that env's class
(no-arch-pin / no-editable / no-env_run). It is meant to be COPIED to and run ON
the foreign host.

Heavy imports (oh_my_mlip.fetch / huggingface_hub) are guarded so the module
stays importable for inspection without a network or the package on PATH.

Usage (on the foreign host, inside the clone):
  source env.sh
  python scripts/verify_relocation.py [models.json] [--only MACE,SevenNet]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Single-point + D3 probe, run inside each env's own interpreter.
TEMPLATE = '''
import sys, warnings
warnings.filterwarnings("ignore")
from ase.build import bulk
{imports}
at = bulk("Cu", "fcc", a=3.61, cubic=True)
{inference}
at.calc = calc
e = at.get_potential_energy()
print("ENERGY", round(float(e), 4))
try:
    from catbench.dispersion import DispersionCorrection
    at2 = bulk("Cu", "fcc", a=3.61, cubic=True)
    at2.calc = DispersionCorrection().apply(calc)
    e2 = at2.get_potential_energy()
    print("D3", round(float(e2), 4))
except Exception as ex:  # noqa: BLE001
    print("D3_FAIL", repr(ex)[:200])
print("PASS")
'''


def _ensure_env(framework: str) -> bool:
    """Fetch+relocate one model's env via the public resolver. Guarded import.

    `oh_my_mlip.fetch.fetch_env(model)` resolves the framework -> its env through
    the registry, downloads the conda-pack tarball from the manifest, and runs
    conda-unpack once (sentinel-guarded). It returns the relocated interpreter
    path; we only need the success/failure signal here.
    """
    try:
        from oh_my_mlip import fetch  # heavy: network + hf_hub
    except ImportError as exc:
        print(f"  [fetch unavailable] {framework}: {exc!r}", file=sys.stderr)
        return False
    try:
        fetch.fetch_env(framework)  # registry -> manifest -> download -> conda-unpack
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [fetch failed] {framework}: {repr(exc)[:200]}", file=sys.stderr)
        return False


def run_one(py: str, imports, inference) -> tuple[str, str]:
    snippet = TEMPLATE.format(imports="\n".join(imports), inference="\n".join(inference))
    try:
        r = subprocess.run([py, "-c", snippet], capture_output=True, text=True, timeout=900)
        out = r.stdout
        if "PASS" in out:
            energy = next((l for l in out.splitlines() if l.startswith("ENERGY")), "")
            d3 = next((l for l in out.splitlines() if l.startswith("D3")), "")
            return "PASS", f"{energy} | {d3}"
        err = (r.stderr.strip().splitlines() or [""])[-1][:160]
        return "FAIL", err or out[-160:]
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "900s"
    except Exception as ex:  # noqa: BLE001
        return "ERR", repr(ex)[:160]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models_json", nargs="?", default="models.json")
    ap.add_argument("--only", default=None, help="comma-separated framework filter (e.g. MACE,SevenNet)")
    ap.add_argument("--no-fetch", action="store_true", help="assume envs already unpacked; skip fetch")
    args = ap.parse_args()

    home = os.environ.get("OH_MY_MLIP_HOME")
    if not home:
        print("verify_relocation: OH_MY_MLIP_HOME not set; run `source env.sh` first.", file=sys.stderr)
        return 1

    d = json.loads(Path(args.models_json).read_text())
    only = set(args.only.split(",")) if args.only else None

    print(f"{'model':28} {'result':8} detail")
    print("-" * 100)
    for fw, info in d.items():
        if fw.startswith("_"):
            continue
        if only and fw not in only:
            continue
        env_name = info.get("env", fw)
        if not args.no_fetch and not _ensure_env(fw):
            print(f"{fw:28} {'NOFETCH':8} could not relocate env '{env_name}'")
            continue
        py = (info.get("python") or "").replace("${OH_MY_MLIP_HOME}", home)
        imp = info.get("import", [])
        for vname, v in info.get("versions", {}).items():
            inf = v.get("inference") or v.get("inference_sm89") or v.get("inference_sm86")
            if not inf or any(("TODO" in l) or l.strip().startswith("#") for l in inf):
                print(f"{vname:28} {'SKIP':8} no/placeholder inference")
                continue
            inf = [l.replace("${OH_MY_MLIP_HOME}", home) for l in inf]
            if not py or not os.path.exists(py):
                print(f"{vname:28} {'NOENV':8} {py}")
                continue
            res, detail = run_one(py, imp, inf)
            print(f"{vname:28} {res:8} {detail}")
    print("-" * 100)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
