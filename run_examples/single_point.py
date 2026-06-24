#!/usr/bin/env python3
"""single_point.py — minimal single-point energy/forces with one MLIP.

Uses ONLY the public oh_my_mlip interface. `run()` is the cross-env convenience
path: it spawns the correct env interpreter for the model, computes, and returns
a results dict — you do not need to be inside the model's conda env to call it.

Run:
  source env.sh                       # sets caches + D3/CUDA env, OH_MY_MLIP_HOME
  python run_examples/single_point.py [MODEL]   # MODEL default: MACE

Pass --d3 to apply the D3 dispersion correction (compiles on first use; see
docs/arch_first_run_compile.md). For gated models (e.g. UMA) export HF_TOKEN and
accept the upstream license first (see docs/gated_models.md).
"""
import argparse
import os
import sys
from pathlib import Path

# Import oh_my_mlip by path (it is NOT a pip package).
_HOME = os.environ.get("OH_MY_MLIP_HOME") or str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _HOME)

from ase.build import bulk  # noqa: E402
from oh_my_mlip import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", default="MACE", help="framework key in models.json")
    ap.add_argument("--version", default=None, help="specific version (default: the framework's default_version, e.g. MACE -> MACE-MPA-0)")
    ap.add_argument("--d3", action="store_true", help="apply D3 dispersion correction")
    args = ap.parse_args()

    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    out = run(
        args.model,
        atoms,
        version=args.version,
        properties=("energy", "forces"),
        apply_d3=args.d3,
    )
    print(f"model      : {args.model}{' +D3' if args.d3 else ''}")
    print(f"energy (eV): {out['energy']:.6f}")
    fmax = max((sum(c * c for c in f) ** 0.5 for f in out["forces"]), default=0.0)
    print(f"max|force| : {fmax:.6f} eV/A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
