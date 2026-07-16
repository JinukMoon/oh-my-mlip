#!/usr/bin/env python3
"""single_point.py — minimal single-point energy/forces with one MLIP.

Uses ONLY the public oh_my_mlip interface. `run()` is the cross-env convenience
path: it spawns the correct env interpreter for the model, computes, and returns
a results dict — you do not need to be inside the model's conda env to call it.

This launcher does NOT import ase: the demo structure is passed as a plain
`{symbols, positions, cell, pbc}` spec and built inside the model's worker (which
has ase). So you can run it with any `python` that can import oh_my_mlip — no ase
in the launching interpreter. Pass `--structure PATH` (POSCAR/cif/xyz) to label
your own structure instead; the worker reads the file.

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

from oh_my_mlip import run  # noqa: E402
from oh_my_mlip.provider import WorkerError  # noqa: E402


def _demo_structure() -> dict:
    """A conventional 4-atom fcc Cu cell (a=3.61 A) as an ase-free spec dict.

    Built WITHOUT importing ase so the launcher stays ase-free; the worker turns
    this `{symbols, positions, cell, pbc}` dict into an ase.Atoms.
    """
    a = 3.61
    return {
        "symbols": "Cu4",
        "positions": [
            [0.0, 0.0, 0.0],
            [0.0, a / 2, a / 2],
            [a / 2, 0.0, a / 2],
            [a / 2, a / 2, 0.0],
        ],
        "cell": [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
        "pbc": [True, True, True],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", default="MACE", help="framework key in models.json")
    ap.add_argument("--version", default=None, help="specific version (default: the framework's default_version, e.g. MACE -> MACE-MPA-0)")
    ap.add_argument("--structure", default=None, help="path to a structure file (POSCAR/cif/xyz); read inside the worker. Default: a demo fcc Cu cell")
    ap.add_argument("--d3", action="store_true", help="apply D3 dispersion correction")
    ap.add_argument("--arch", default=None, help="sm86/sm89 for arch-pinned models (NequIP/Allegro); default: host GPU auto-detect")
    args = ap.parse_args()

    atoms = {"file": args.structure} if args.structure else _demo_structure()
    try:
        out = run(
            args.model,
            atoms,
            version=args.version,
            arch=args.arch,
            properties=("energy", "forces"),
            apply_d3=args.d3,
        )
    except WorkerError as exc:
        # Env not materialized (or worker failed to start): print the actionable
        # message, not a raw traceback.
        print(f"[oh-my-mlip] {exc}", file=sys.stderr)
        return 1
    print(f"model      : {args.model}{' +D3' if args.d3 else ''}")
    print(f"energy (eV): {out['energy']:.6f}")
    fmax = max((sum(c * c for c in f) ** 0.5 for f in out["forces"]), default=0.0)
    print(f"max|force| : {fmax:.6f} eV/A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
