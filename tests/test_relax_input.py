"""run_examples/relax.py input contract: an explicit --structure is loaded and
its identity reported; a missing or unreadable explicit input fails (exit 2)
and nothing is substituted; the legacy no-argument demo is labelled as a demo.

No model compute: `Worker` is replaced by a fake whose forces are zero, so
BFGS converges on step 0. The launcher needs ase (relax.py's own requirement).
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RELAX = REPO_ROOT / "run_examples" / "relax.py"

pytest.importorskip("ase")


def _load_module():
    spec = importlib.util.spec_from_file_location("relax_under_test", RELAX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeWorker:
    """Stands in for oh_my_mlip.Worker: no env, no subprocess, zero forces."""

    def __init__(self, model, version=None, apply_d3=False, arch=None):
        self.model = model

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def request(self, atoms, properties=("energy", "forces")):
        return {"ok": True, "results": {"energy": -1.0 * len(atoms),
                                        "forces": np.zeros((len(atoms), 3)).tolist()}}


@pytest.fixture
def relax(monkeypatch):
    mod = _load_module()
    monkeypatch.setattr(mod, "Worker", _FakeWorker)
    return mod


def _run(mod, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["relax.py", *argv])
    return mod.main()


def _write_xyz(path: Path, n: int = 2) -> Path:
    lines = [str(n), "test frame"]
    for i in range(n):
        lines.append(f"Cu {i * 2.5:.3f} 0.0 0.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_explicit_structure_is_loaded_and_identity_reported(relax, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    src = _write_xyz(tmp_path / "input.xyz", n=3)
    rc = _run(relax, monkeypatch, ["MACE", "--structure", str(src), "--steps", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"structure   : {src.resolve()} (explicit)" in out
    assert hashlib.sha256(src.read_bytes()).hexdigest() in out
    assert "atoms       : 3 Cu3" in out
    assert "[demo]" not in out
    assert (tmp_path / "relaxed.extxyz").is_file()


def test_missing_explicit_structure_fails_without_substitute(relax, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    rc = _run(relax, monkeypatch, ["MACE", "--structure", str(tmp_path / "absent.xyz")])
    captured = capsys.readouterr()
    assert rc == 2
    assert "no such file" in captured.err
    assert "structure   :" not in captured.out
    assert not (tmp_path / "relaxed.extxyz").exists()


def test_unreadable_explicit_structure_fails_without_substitute(relax, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    bad = tmp_path / "garbage.xyz"
    bad.write_text("this is not a structure\n", encoding="utf-8")
    rc = _run(relax, monkeypatch, ["MACE", "--structure", str(bad)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "not readable" in captured.err
    assert not (tmp_path / "relaxed.extxyz").exists()


def test_explicit_structure_ignores_cwd_poscar(relax, monkeypatch, tmp_path, capsys):
    """A cwd POSCAR must not shadow an explicit --structure."""
    monkeypatch.chdir(tmp_path)
    five = _write_xyz(tmp_path / "five_atoms.xyz", n=5)
    # legacy-format POSCAR in cwd (2 atoms) — should be ignored
    (tmp_path / "POSCAR").write_text(
        "Cu2\n1.0\n3.6 0 0\n0 3.6 0\n0 0 3.6\nCu\n2\nDirect\n0 0 0\n0.5 0.5 0.5\n", encoding="utf-8")
    rc = _run(relax, monkeypatch, ["MACE", "--structure", str(five), "--steps", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "atoms       : 5 Cu5" in out
    assert "(explicit)" in out


def test_no_argument_uses_cwd_poscar_and_says_so(relax, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "Cu2\n1.0\n3.6 0 0\n0 3.6 0\n0 0 3.6\nCu\n2\nDirect\n0 0 0\n0.5 0.5 0.5\n", encoding="utf-8")
    rc = _run(relax, monkeypatch, ["MACE", "--steps", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"structure   : {poscar.resolve()} (cwd POSCAR)" in out
    assert hashlib.sha256(poscar.read_bytes()).hexdigest() in out
    assert "atoms       : 2 Cu2" in out
    assert "[demo]" not in out


def test_no_argument_no_poscar_is_labelled_demo(relax, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    rc = _run(relax, monkeypatch, ["MACE", "--steps", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "structure   : [demo]" in out
    assert "not a result for any user structure" in out
    assert "sha256" not in out


def test_structure_flag_is_in_the_parser():
    """The contract test promotes --structure on parser evidence; keep it honest."""
    import ast
    tree = ast.parse(RELAX.read_text(encoding="utf-8"))
    flags = {a.value for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "add_argument"
             for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)}
    assert "--structure" in flags
