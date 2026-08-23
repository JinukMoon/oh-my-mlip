"""GPU-free tests for scripts/distill_bootstrap.py (P4.2).

Covers the four failure modes N3/C15 extends the assertion count to, plus the
artifact-content checks the AC2 demo depends on being right before any GPU
compute runs:

  - run_distill.sh carries the PATH pin (F17) to the teacher env's bin;
  - PYTHONPATH carries BOTH the sibling repo D and the work dir (F16), as two
    independently-checkable components;
  - run_distill.sh exports ONTHEFLY_CONFIG pointing at the generated
    config.yaml (F21/N3);
  - rendered work_dir (config.yaml's `work_dir`, which becomes al_loop's own
    working directory) is an ABSOLUTE path (F21/N3);
  - config.yaml parses and carries every key
    onthefly-distill/ontheflydistill/config.py's DEFAULTS declares;
  - omm_teacher.py passes ast.parse and its embedded inference line(s) are
    byte-identical (modulo indentation) to oh_my_mlip.resolve()'s;
  - run_distill.sh is shellcheck-clean and points AT D/scripts/al_loop_local.sh
    (never a copy of it -- no such file is ever written under the work dir);
  - a structure with more than 4 distinct species is refused with an
    actionable error naming pair_nnmtp v1.

Everything here runs against a FAKE onthefly-distill checkout (a temp dir with
just the files distill_bootstrap.py actually reads: scripts/al_loop_local.sh,
scripts/teacher_md.py, config.example.yaml) so this suite never depends on the
real GPL-2.0 sibling repo being present on the machine that runs CI.
"""
from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

pytest.importorskip("ase")  # GPU-free CI has no ase
from ase.build import bulk, molecule
from ase.io import write

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import distill_bootstrap as db  # noqa: E402
from oh_my_mlip import resolve  # noqa: E402

# Every key onthefly-distill's config loader falls back to when absent from
# config.yaml -- see ontheflydistill/config.py's DEFAULTS dict. Mirrored here
# (not imported: D is invoked, never imported) so this test fails the moment
# distill_bootstrap.py's config.example.yaml overlay drops one.
CONFIG_DEFAULT_KEYS = {
    "python_bin": None,
    "work_dir": "./run",
    "system": {"init_structure", "species", "specorder", "masses", "fixed_bottom_n", "force_pbc"},
    "teacher": {"type", "calculator"},
    "student": {
        "n_radial_basis", "n_radial_funcs", "r_max", "hidden_dims", "embed_dim",
        "activation", "zbl", "zbl_r_inner", "zbl_r_outer", "nu_max", "epochs",
    },
    "al_loop": {"target_ps", "no_progress_limit", "max_iter", "seed", "omp_threads"},
    "remote": {"enabled", "host", "base", "slurm_partition", "slurm_gres"},
}


def _find_shellcheck() -> str | None:
    found = shutil.which("shellcheck")
    if found:
        return found
    candidate = Path(sys.exec_prefix) / "bin" / "shellcheck"
    return str(candidate) if candidate.is_file() else None


@pytest.fixture()
def fake_d(tmp_path) -> Path:
    """A minimal onthefly-distill checkout: just enough for check_repo() to
    accept it and for config.example.yaml's overlay to be exercised."""
    d = tmp_path / "onthefly-distill"
    (d / "scripts").mkdir(parents=True)
    (d / "scripts" / "al_loop_local.sh").write_text("#!/bin/bash\necho fake al_loop_local\n")
    (d / "scripts" / "teacher_md.py").write_text("# fake teacher_md.py\n")
    example = textwrap.dedent(
        """\
        lmp_bin: /path/to/lammps/build/lmp
        python_bin: python
        work_dir: ./run

        system:
          init_structure: examples/ptwater_acid/ptwater_acid_init_clean.vasp
          species: [1, 8, 78]
          specorder: [H, O, Pt]
          masses: [1.008, 15.999, 195.084]
          fixed_bottom_n: 32
          force_pbc: true

        teacher:
          type: ase_calculator
          calculator: "mypkg:make_calc"

        student:
          n_radial_basis: 12
          n_radial_funcs: 8
          r_max: 6.0
          hidden_dims: [64, 32]
          embed_dim: 16
          activation: silu
          zbl: true
          zbl_r_inner: 0.5
          zbl_r_outer: 3.0
          nu_max: 2
          epochs: 300

        al_loop:
          target_ps: 100000
          no_progress_limit: 4
          max_iter: 30
          seed: 42
          omp_threads: 8

        remote:
          enabled: false
          host: myserver
          base: /remote/path
          slurm_partition: gpu
          slurm_gres: "gpu:1"
        """
    )
    (d / "config.example.yaml").write_text(example)
    return d


@pytest.fixture()
def cu_structure(tmp_path) -> Path:
    path = tmp_path / "Cu32.vasp"
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2)
    write(path, atoms, format="vasp")
    return path


@pytest.fixture()
def bootstrapped(tmp_path, fake_d, cu_structure):
    work = tmp_path / "work"
    db.main([
        "--teacher", "MACE-MPA-0",
        "--structure", str(cu_structure),
        "--work", str(work),
        "--repo", str(fake_d),
        "--target-ps", "2.5",
    ])
    return work.resolve()


# ── run_distill.sh: the four load-bearing exports (C15/N3) ──────────────────
def test_run_distill_sh_has_path_pin(bootstrapped):
    text = (bootstrapped / "run_distill.sh").read_text()
    teacher_python = resolve("MACE-MPA-0")["python"]
    teacher_bin = str(Path(teacher_python).parent)
    assert f'PATH="{teacher_bin}:$PATH"' in text


def test_run_distill_sh_pythonpath_has_both_components(bootstrapped, fake_d):
    text = (bootstrapped / "run_distill.sh").read_text()
    pythonpath_lines = [ln for ln in text.splitlines() if ln.startswith("PYTHONPATH=")]
    assert len(pythonpath_lines) == 1
    line = pythonpath_lines[0]
    assert str(fake_d.resolve()) in line
    assert str(bootstrapped) in line


def test_run_distill_sh_exports_ontheflyconfig(bootstrapped):
    text = (bootstrapped / "run_distill.sh").read_text()
    assert f'ONTHEFLY_CONFIG="{bootstrapped}/config.yaml"' in text
    assert "export ONTHEFLY_CONFIG" in text


def test_run_distill_sh_points_at_d_never_a_copy(bootstrapped, fake_d):
    text = (bootstrapped / "run_distill.sh").read_text()
    assert f'exec "{fake_d.resolve()}/scripts/al_loop_local.sh"' in text
    assert not (bootstrapped / "al_loop_local.sh").exists()
    assert not any(bootstrapped.glob("*.sh")) or {p.name for p in bootstrapped.glob("*.sh")} == {"run_distill.sh"}


# ── config.yaml: absolute work_dir + full key coverage ───────────────────────
def test_config_work_dir_is_absolute(bootstrapped):
    cfg = yaml.safe_load((bootstrapped / "config.yaml").read_text())
    assert Path(cfg["work_dir"]).is_absolute()
    assert cfg["work_dir"] == str(bootstrapped / "run")


def _assert_keys_present(node, spec):
    if isinstance(spec, set):
        assert spec <= node.keys(), f"missing keys {spec - node.keys()}"
    # scalar defaults (python_bin, work_dir) just need the top-level key, checked by caller


def test_config_yaml_parses_and_carries_every_default_key(bootstrapped):
    cfg = yaml.safe_load((bootstrapped / "config.yaml").read_text())
    for key, spec in CONFIG_DEFAULT_KEYS.items():
        assert key in cfg, f"config.yaml missing top-level key {key!r}"
        if isinstance(spec, set):
            _assert_keys_present(cfg[key], spec)


def test_config_teacher_points_at_omm_teacher(bootstrapped):
    cfg = yaml.safe_load((bootstrapped / "config.yaml").read_text())
    assert cfg["teacher"]["type"] == "ase_calculator"
    assert cfg["teacher"]["calculator"] == "omm_teacher:make_calc"


def test_config_target_ps_from_arg(bootstrapped):
    cfg = yaml.safe_load((bootstrapped / "config.yaml").read_text())
    assert cfg["al_loop"]["target_ps"] == 2.5


# ── omm_teacher.py: ast.parse + byte-identical inference lines ──────────────
def test_omm_teacher_parses_and_matches_resolve(bootstrapped):
    text = (bootstrapped / "omm_teacher.py").read_text()
    ast.parse(text)  # raises on any syntax error
    spec = resolve("MACE-MPA-0")
    for line in spec["inference"]:
        assert line.strip() in text
    for line in spec["imports"]:
        assert line.strip() in text


def test_omm_teacher_defines_zero_arg_make_calc(bootstrapped):
    tree = ast.parse((bootstrapped / "omm_teacher.py").read_text())
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "make_calc"]
    assert len(funcs) == 1
    assert len(funcs[0].args.args) == 0


# ── shellcheck ────────────────────────────────────────────────────────────────
def test_run_distill_sh_is_shellcheck_clean(bootstrapped):
    shellcheck = _find_shellcheck()
    if shellcheck is None:
        pytest.skip("shellcheck not available on PATH")
    proc = subprocess.run([shellcheck, str(bootstrapped / "run_distill.sh")], capture_output=True, text=True)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


# ── >4 species refusal (pair_nnmtp v1 bound, F18) ────────────────────────────
def test_more_than_4_species_is_refused_actionably(tmp_path, fake_d):
    # NaCH3OH-ish blend: 5 distinct elements (H, O, C, N, Na) forced into one cell
    atoms = molecule("CH3OH") + molecule("NH3")
    atoms.set_cell([10, 10, 10])
    atoms.set_pbc(True)
    from ase import Atoms
    atoms += Atoms("Na", positions=[[5, 5, 5]])
    structure = tmp_path / "five_species.xyz"
    write(structure, atoms)

    work = tmp_path / "work_refused"
    with pytest.raises(SystemExit) as exc:
        db.main([
            "--teacher", "MACE-MPA-0",
            "--structure", str(structure),
            "--work", str(work),
            "--repo", str(fake_d),
        ])
    msg = str(exc.value)
    assert "pair_nnmtp" in msg
    assert "v1" in msg
    assert "4" in msg
    # refusal happens before ANYTHING is written -- not even the work dir
    # or its bootstrap.log (AGENTS.md §3D: species is validated before
    # distill_bootstrap.py creates anything on disk).
    assert not work.exists()
    assert not (work / "bootstrap.log").exists()
    assert not (work / "omm_teacher.py").exists()


def test_up_to_4_species_is_accepted(tmp_path, fake_d, cu_structure):
    # sanity: the fixture structure (1 species) must NOT trip the refusal path
    work = tmp_path / "work_ok"
    rc = db.main([
        "--teacher", "MACE-MPA-0",
        "--structure", str(cu_structure),
        "--work", str(work),
        "--repo", str(fake_d),
    ])
    assert rc == 0
    assert (work / "run_distill.sh").exists()


# ── repo-not-found guard ─────────────────────────────────────────────────────
def test_missing_repo_is_a_clean_error(tmp_path, cu_structure):
    with pytest.raises(SystemExit):
        db.main([
            "--teacher", "MACE-MPA-0",
            "--structure", str(cu_structure),
            "--work", str(tmp_path / "work"),
            "--repo", str(tmp_path / "not-a-repo"),
        ])
