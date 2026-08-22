#!/usr/bin/env python3
"""make_ft_demo_dataset.py -- ~60-frame teacher-labeled demo set for the
fine-tuning pipeline (scripts/ft_dataset.py -> scripts/ft_run.py).

Builds a small, cheap structure set (rattled Cu bulk, a few single-vacancy
and volumetrically-strained variants) and labels every frame with an
installed teacher model through a single persistent `oh_my_mlip.Worker`
(amortizes one env startup across all frames instead of one per call).

Written as a plain ASE ``.traj`` -- DELIBERATELY not any framework-native
format, so the demo actually exercises ft_dataset.py's whole point (one
canonical converter reading an arbitrary ase.io-readable file and re-emitting
per-framework layouts).

Usage:
  python3 run_examples/make_ft_demo_dataset.py --out ft_demo.traj
  python3 run_examples/make_ft_demo_dataset.py --out ft_demo.traj --teacher MACE --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

_HOME = os.environ.get("OH_MY_MLIP_HOME") or str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _HOME)

from oh_my_mlip.provider import Worker, WorkerError  # noqa: E402


def build_structures(n_rattled: int, n_vacancy: int, n_strained: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    base = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((2, 2, 2))  # 32 atoms

    frames = []
    for _ in range(n_rattled):
        a = base.copy()
        a.rattle(stdev=0.03, seed=int(rng.integers(0, 2**31 - 1)))
        frames.append(a)
    for _ in range(n_vacancy):
        a = base.copy()
        del a[int(rng.integers(0, len(a)))]
        a.rattle(stdev=0.02, seed=int(rng.integers(0, 2**31 - 1)))
        frames.append(a)
    for _ in range(n_strained):
        a = base.copy()
        strain = 1.0 + rng.uniform(-0.04, 0.04)
        a.set_cell(a.get_cell() * strain, scale_atoms=True)
        a.rattle(stdev=0.015, seed=int(rng.integers(0, 2**31 - 1)))
        frames.append(a)
    return frames


def label_with_teacher(frames: list, model: str, version: str | None, device: str) -> list:
    worker = Worker(model, version=version, device=device)
    try:
        worker.start()
    except WorkerError as exc:
        raise SystemExit(f"[make_ft_demo_dataset] teacher {model!r} not available: {exc}")

    labeled = []
    try:
        for atoms in frames:
            resp = worker.request(atoms, ("energy", "forces"))
            if not resp.get("ok"):
                raise SystemExit(f"[make_ft_demo_dataset] teacher labeling failed: {resp.get('error')}")
            results = resp["results"]
            a = atoms.copy()
            a.calc = SinglePointCalculator(
                a, energy=float(results["energy"]), forces=np.asarray(results["forces"], dtype=float)
            )
            labeled.append(a)
    finally:
        worker.shutdown()
    return labeled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="ft_demo.traj", type=Path)
    ap.add_argument("--teacher", default="MACE", help="framework key used to label every frame")
    ap.add_argument("--teacher-version", default=None)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--n-rattled", type=int, default=50)
    ap.add_argument("--n-vacancy", type=int, default=5)
    ap.add_argument("--n-strained", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = build_structures(args.n_rattled, args.n_vacancy, args.n_strained, args.seed)
    labeled = label_with_teacher(frames, args.teacher, args.teacher_version, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, labeled)
    print(f"[make_ft_demo_dataset] wrote {len(labeled)} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
