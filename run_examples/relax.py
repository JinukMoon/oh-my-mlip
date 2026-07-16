#!/usr/bin/env python3
"""relax.py — relax a structure with one MLIP using the public oh_my_mlip API.

For a relaxation we need a live ASE optimizer attached across many force calls,
so this example uses the persistent `Worker` (one long-lived process for the
model's env) and drives an ASE optimizer against it. This avoids spawning a fresh
subprocess on every optimizer step.

LAUNCHER NEEDS ase: unlike single_point.py, the ASE optimizer (BFGS) runs in the
LAUNCHING interpreter (it drives the worker step by step), so this example must
be run with a python that can import ase, e.g. the toolkit env:
  /home/jumoon/miniconda3/envs/toolkit/bin/python  (or any env with ase).
Only the heavy MLIP framework stays in the worker; ase here is just the optimizer
+ structure I/O. (single_point.py is fully ase-free in the launcher.)

Run:
  source env.sh
  python run_examples/relax.py [MODEL] [--steps 50] [--fmax 0.05] [--d3]

Reads POSCAR from the current directory if present, else relaxes a rattled Cu
cell as a smoke test. For gated models export HF_TOKEN first (docs/gated_models.md).
"""
import argparse
import os
import sys
from pathlib import Path

_HOME = os.environ.get("OH_MY_MLIP_HOME") or str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _HOME)

from ase.build import bulk  # noqa: E402
from ase.calculators.calculator import Calculator, all_changes  # noqa: E402
from ase.io import read  # noqa: E402
from ase.optimize import BFGS  # noqa: E402
from oh_my_mlip import Worker  # noqa: E402
from oh_my_mlip.provider import WorkerError  # noqa: E402


class _WorkerCalculator(Calculator):
    """Thin ASE Calculator that routes every force call to a persistent Worker.

    The public `oh_my_mlip` API exposes the cross-env worker as a JSONL
    request/response surface (`Worker.request(atoms, properties)`), not an ASE
    calculator object — so an optimizer needs this small adapter. Each
    `calculate()` ships the current atoms to the long-lived env process and
    unpacks the returned energy/forces. One subprocess for the whole relaxation.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, worker: Worker, **kwargs):
        super().__init__(**kwargs)
        self._worker = worker

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        resp = self._worker.request(self.atoms, properties=("energy", "forces"))
        if not resp.get("ok"):
            raise RuntimeError(f"worker request failed: {resp.get('error')}")
        results = resp["results"]
        import numpy as np

        self.results["energy"] = float(results["energy"])
        self.results["forces"] = np.asarray(results["forces"], dtype=float)


def _load_atoms():
    poscar = Path("POSCAR")
    if poscar.exists():
        return read(str(poscar))
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    atoms.rattle(stdev=0.1, seed=0)
    return atoms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", default="MACE")
    ap.add_argument("--version", default=None)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--d3", action="store_true")
    ap.add_argument("--arch", default=None, help="sm86/sm89 for arch-pinned models; default: host GPU auto-detect")
    args = ap.parse_args()

    atoms = _load_atoms()

    # One persistent env process backs an ASE calculator adapter for the whole relaxation.
    try:
        with Worker(args.model, version=args.version, apply_d3=args.d3, arch=args.arch) as worker:
            atoms.calc = _WorkerCalculator(worker)
            opt = BFGS(atoms)
            opt.run(fmax=args.fmax, steps=args.steps)
    except WorkerError as exc:
        # Env not materialized (or worker failed to start): print the actionable
        # message, not a raw traceback.
        print(f"[oh-my-mlip] {exc}", file=sys.stderr)
        return 1

    print(f"model       : {args.model}{' +D3' if args.d3 else ''}")
    print(f"final energy: {atoms.get_potential_energy():.6f} eV")
    out = Path("relaxed.extxyz")
    atoms.write(str(out))
    print(f"wrote       : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
