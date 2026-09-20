"""A slab with fixed bottom layers must survive the worker wire format.

_worker.encode_atoms sends Atoms.todict() through json.dumps; constraints came
out as live FixAtoms objects and raised "not JSON serializable: FixAtoms", so
run()/Worker refused the most ordinary structure this tool's readers build.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("ase")
from ase.build import bulk  # noqa: E402
from ase.constraints import FixAtoms  # noqa: E402

from oh_my_mlip import _worker  # noqa: E402


def _slab_with_fixed_bottom():
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True) * (1, 1, 2)
    atoms.set_constraint(FixAtoms(indices=[0, 1]))
    return atoms


def test_constrained_atoms_survive_the_round_trip():
    atoms = _slab_with_fixed_bottom()
    payload = _worker.encode_atoms(atoms)
    json.dumps(payload)                       # the wire is pure JSON, no live objects
    assert payload["constraints"] == [{"name": "FixAtoms", "kwargs": {"indices": [0, 1]}}]

    back = _worker.decode_atoms(payload)
    assert len(back) == len(atoms)
    assert len(back.constraints) == 1
    assert sorted(back.constraints[0].get_indices()) == [0, 1]


def test_unconstrained_atoms_are_unchanged():
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    back = _worker.decode_atoms(_worker.encode_atoms(atoms))
    assert len(back) == len(atoms) and not back.constraints
