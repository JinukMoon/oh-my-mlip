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
    actionable error naming pair_nnmtp v1;
  - every rendered path is shell-quoted (a work/engine/interpreter path with
    spaces, an apostrophe and a literal `$(touch pwned)` runs end to end
    with nothing expanded);
  - the wall-clock supervisor's outcome rule: rc 124 / late 137 = exhausted,
    an early SIGKILL (137) or TERM to the group = engine_exit;
  - acceptance-only flags are refused without --acceptance, never ignored;
  - any ase-readable structure is staged as <work>/structure_init.vasp (the
    only format D's teacher_md.py reads); cell-less or unreadable input is
    refused before anything is written.

Everything here runs against a FAKE onthefly-distill checkout (a temp dir with
just the files distill_bootstrap.py actually reads: scripts/al_loop_local.sh,
scripts/teacher_md.py, config.example.yaml) so this suite never depends on the
real GPL-2.0 sibling repo being present on the machine that runs CI.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("ase")  # GPU-free CI has no ase
from ase.build import bulk, molecule
from ase.io import read, write

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
    assert f'PATH={db._q(teacher_bin)}:$PATH' in text


def test_run_distill_sh_pythonpath_has_both_components(bootstrapped, fake_d):
    text = (bootstrapped / "run_distill.sh").read_text()
    pythonpath_lines = [ln for ln in text.splitlines() if ln.startswith("PYTHONPATH=")]
    assert len(pythonpath_lines) == 1
    line = pythonpath_lines[0]
    assert str(fake_d.resolve()) in line
    assert str(bootstrapped) in line


def test_run_distill_sh_exports_ontheflyconfig(bootstrapped):
    text = (bootstrapped / "run_distill.sh").read_text()
    assert f'ONTHEFLY_CONFIG={db._q(bootstrapped / "config.yaml")}' in text
    assert "export ONTHEFLY_CONFIG" in text


def test_run_distill_sh_points_at_d_never_a_copy(bootstrapped, fake_d):
    text = (bootstrapped / "run_distill.sh").read_text()
    assert f'exec {db._q(fake_d.resolve() / "scripts" / "al_loop_local.sh")}' in text
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


# ── >4 species refusal (pair_nnmtp v1 supports up to 4 elements) ─────────────────
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


# ── --acceptance: approved targets rendered, never defaulted (recipes/distill.md §2) ──
ACC_BASE = ["--mode", "production", "--energy-mae-max", "5", "--force-mae-max", "100",
            "--target-ps", "0.5", "--max-iter", "4", "--no-progress-limit", "2"]


def _acc_main(tmp_path, fake_d, cu_structure, *extra, work_name="work_acc", teacher="MACE-MPA-0"):
    work = tmp_path / work_name
    rc = db.main(["--teacher", teacher, "--structure", str(cu_structure), "--work", str(work),
                  "--repo", str(fake_d), "--acceptance", *extra])
    assert rc == 0
    return work.resolve()


@pytest.fixture()
def accepted(tmp_path, fake_d, cu_structure):
    return _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE, "--pool-steps", "40", "--pool-save-every", "8",
                     "--wallclock-max-h", "6", "--manifest-sha256", "deadbeef")


@pytest.mark.parametrize("drop", ["--mode", "--energy-mae-max", "--force-mae-max", "--target-ps", "--max-iter",
                                  "--no-progress-limit"])
def test_acceptance_refuses_any_missing_target(tmp_path, fake_d, cu_structure, drop, capsys):
    args = list(ACC_BASE)
    i = args.index(drop)
    del args[i:i + 2]
    with pytest.raises(SystemExit) as exc:
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(tmp_path / "w"),
                 "--repo", str(fake_d), "--acceptance", *args])
    assert exc.value.code == 2  # argparse usage error, before anything is written
    assert drop in capsys.readouterr().err
    assert not (tmp_path / "w").exists()


def test_acceptance_fixture_bounds_max_iter(tmp_path, fake_d, cu_structure, capsys):
    args = [a if a != "production" else "fixture" for a in ACC_BASE]  # max_iter 4 > bound 3
    with pytest.raises(SystemExit) as exc:
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(tmp_path / "w"),
                 "--repo", str(fake_d), "--acceptance", *args])
    assert exc.value.code == 2
    assert f"<= {db.FIXTURE_MAX_ITER}" in capsys.readouterr().err
    ok = _acc_main(tmp_path, fake_d, cu_structure, *[a for a in args], "--max-iter", "3")  # last --max-iter wins
    acc = json.loads((ok / "acceptance.json").read_text())
    assert acc["mode"] == "fixture" and acc["budget"]["max_iter"] == 3
    assert acc["budget"]["fixture_max_iter_bound"] == db.FIXTURE_MAX_ITER


@pytest.mark.parametrize("flag,val", [("--energy-mae-max", "0"), ("--force-mae-max", "-1"), ("--target-ps", "0"),
                                      ("--max-iter", "0"), ("--no-progress-limit", "0")])
def test_acceptance_refuses_non_positive_targets(tmp_path, fake_d, cu_structure, flag, val):
    with pytest.raises(SystemExit) as exc:
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(tmp_path / "w"),
                 "--repo", str(fake_d), "--acceptance", *ACC_BASE, flag, val])
    assert exc.value.code == 2


def test_acceptance_heldout_seed_may_not_equal_pool_seed(tmp_path, fake_d, cu_structure, capsys):
    with pytest.raises(SystemExit) as exc:
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(tmp_path / "w"),
                 "--repo", str(fake_d), "--acceptance", *ACC_BASE, "--heldout-seed", str(db.POOL_TEACHER_SEED)])
    assert exc.value.code == 2
    assert "held-out" in capsys.readouterr().err


def test_acceptance_config_al_loop_carries_approved_values(accepted):
    cfg = yaml.safe_load((accepted / "config.yaml").read_text())
    assert cfg["al_loop"]["target_ps"] == 0.5
    assert cfg["al_loop"]["max_iter"] == 4
    assert cfg["al_loop"]["no_progress_limit"] == 2
    assert cfg["al_loop"]["seed"] == 42
    assert cfg["al_loop"]["omp_threads"] == 8  # untouched example key survives


def test_acceptance_seed_flag_reaches_config_and_contract(tmp_path, fake_d, cu_structure):
    work = _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE, "--seed", "7")
    cfg = yaml.safe_load((work / "config.yaml").read_text())
    acc = json.loads((work / "acceptance.json").read_text())
    assert cfg["al_loop"]["seed"] == 7 == acc["budget"]["seed"]


def test_acceptance_run_sh_generates_heldout_outside_run_dir(accepted, fake_d):
    text = (accepted / "run_distill.sh").read_text()
    heldout = accepted / "heldout" / "heldout.extxyz"
    assert f'HELDOUT={db._q(heldout)}' in text
    assert not str(heldout).startswith(str(accepted / "run") + "/")
    raw = accepted / "heldout" / "heldout_md_raw.extxyz"
    teacher_md = db._q(fake_d.resolve() / "scripts" / "teacher_md.py")
    assert (f'TEACHER_SEED={db.DEFAULT_HELDOUT_SEED} python {teacher_md} '
            f'{db._q(raw)} {db.DEFAULT_HELDOUT_STEPS} {db.DEFAULT_HELDOUT_SAVE_EVERY}') in text
    # teacher_md.py's pre-MD frame 0 (the init structure, also the pool's first
    # frame) is dropped by a hub-owned step between the raw MD and $HELDOUT
    assert f'python - {db._q(raw)} "$HELDOUT" <<\'PY\'' in text and "frames[drop:]" in text
    # the pool seeding keeps the flag-given size, and the loop is still D's script
    assert f'python {teacher_md} "$WORK/teacher_md.extxyz" 40 8' in text
    assert 'python -m ontheflydistill.merge_xyz "$WORK/dataset.extxyz" "$WORK/teacher_md.extxyz"' in text
    loop = db._q(fake_d.resolve() / "scripts" / "al_loop_local.sh")
    assert f"exec {loop}" in text
    # 6 h bound => the script re-runs ITSELF under timeout, so seeding, held-out
    # and the loop are all inside the bound (not the loop alone)
    self_launch = (f'OMM_DISTILL_ATTEMPT="$ATTEMPT" {db._q(db._find_gnu_timeout())} -k {db.WALLCLOCK_GRACE_S} '
                   f'21600 /bin/sh "$SELF"')
    assert self_launch in text and f'SELF={db._q(accepted / "run_distill.sh")}' in text
    assert text.index("OMM_DISTILL_ATTEMPT") < text.index("dataset.extxyz") < text.index("HELDOUT=") < text.index(f"exec {loop}")
    # no teacher_md.py invocation sits outside the supervised child's scope:
    # every one (pool seeding, held-out) comes after the supervisor's `fi`
    import re
    invocation = re.compile(r"^\s*(?:TEACHER_SEED=\d+ )?python \S*teacher_md\.py", re.M)
    body = text.split("# (child of the supervisor above", 1)[1]
    assert len(invocation.findall(text)) == len(invocation.findall(body)) == 2


def test_acceptance_json_schema_and_provenance(accepted, fake_d, cu_structure):
    acc = json.loads((accepted / "acceptance.json").read_text())
    assert acc["schema"] == "oh-my-mlip.distill.acceptance/1"
    assert acc["mode"] == "production"
    assert acc["accuracy"]["energy_mae_max_mev_per_atom"] == 5.0
    assert acc["accuracy"]["force_mae_max_mev_per_a"] == 100.0
    assert acc["stability"]["target_ps"] == 0.5
    b = acc["budget"]
    assert {k: b[k] for k in ("max_iter", "no_progress_limit", "seed", "wallclock_max_h", "fixture_max_iter_bound")} == {
        "max_iter": 4, "no_progress_limit": 2, "seed": 42, "wallclock_max_h": 6.0, "fixture_max_iter_bound": None}
    assert b["wallclock_enforcement"]["limit_s"] == 21600 and b["wallclock_enforcement"]["grace_s"] == db.WALLCLOCK_GRACE_S
    assert b["wallclock_enforcement"]["marker"] == str(accepted / ".distill" / "wallclock.json")
    assert b["wallclock_enforcement"]["attempt"] == str(accepted / ".distill" / "attempt.json")
    assert b["wallclock_enforcement"]["timeout_bin"] == db._find_gnu_timeout()
    assert b["wallclock_enforcement"]["covers"] == db.WALLCLOCK_COVERS and "held-out" in db.WALLCLOCK_COVERS
    pool, held = acc["split"]["pool"], acc["split"]["heldout"]
    assert pool == {"teacher_md_seed": db.POOL_TEACHER_SEED, "steps": 40, "save_every": 8,
                    "path": str(accepted / "run" / "dataset.extxyz"),
                    "grows_by": "run/al_iter<K>_labeled.extxyz (teacher-relabelled pre-crash windows)"}
    assert held["source"] == "separate_teacher_md" and held["seed"] == db.DEFAULT_HELDOUT_SEED
    assert held["path"] == str(accepted / "heldout" / "heldout.extxyz")
    assert held["outside_engine_work_dir"] is True
    assert "random" in held["independence_rule"] and "NOT a held-out set" in held["independence_rule"]
    p = acc["provenance"]
    assert p["hub"]["manifest_sha256"] == "deadbeef"
    assert p["engine"]["repo"] == str(fake_d.resolve()) and "GPL" in p["engine"]["licence"]
    assert p["teacher"]["version"] == "MACE-MPA-0"
    assert p["structure"]["sha256"] == db._sha256(cu_structure) and p["structure"]["specorder"] == ["Cu"]
    assert p["structure"]["staged_path"] == str(accepted / "structure_init.vasp")
    assert p["structure"]["staged_sha256"] == db._sha256(accepted / "structure_init.vasp")
    for name in ("config.yaml", "run_distill.sh", "omm_teacher.py", "structure_init.vasp"):
        assert p["generated_files"][name]["sha256"] == db._sha256(accepted / name)
    # the invoked engine files are pinned by hash (never copied)
    assert p["engine"]["files"] == {"scripts/al_loop_local.sh": db._sha256(fake_d / "scripts" / "al_loop_local.sh"),
                                    "scripts/teacher_md.py": db._sha256(fake_d / "scripts" / "teacher_md.py")}
    assert "provenance_drift" in p["identity_rule"] and "not" in p["identity_rule"]
    assert p["engine_facts"]["student_md_temperature_K"] == 300.0
    assert p["lmp_bin"] == db.DEFAULT_LMP_BIN
    assert p["lmp_bin_sha256"] == (db._sha256(Path(db.DEFAULT_LMP_BIN)) if Path(db.DEFAULT_LMP_BIN).is_file() else None)
    assert acc["commands"]["verify"] == f"python3 scripts/distill_verify.py --work {db._q(accepted)} --json"
    # the launch line preserves run_distill.sh's own exit status: no `| tee`
    assert acc["commands"]["run"] == f"cd {db._q(accepted)} && sh run_distill.sh > distill.log 2>&1"
    assert "tee" not in acc["commands"]["run"] and "|" not in acc["commands"]["run"]
    assert acc["commands"]["follow"] == f"tail -f {db._q(accepted / 'distill.log')}"
    ws = acc["witness_scope"]
    assert ".pt" in ws["accuracy_artifact"] and ".bin" in ws["witness_artifact"] and "NOT accuracy" in ws["note"]


def test_acceptance_plan_md_states_every_target(accepted):
    plan = (accepted / "PLAN.md").read_text()
    for needle in ("production", "5.0 meV/atom", "100.0 meV/A", "target_ps` = 0.5", "max_iter` = 4",
                   "no_progress_limit` = 2", "6.0 h = 21600 s, enforced over the whole run", "held-out generation",
                   "unmet(budget:wallclock)", "failed(wallclock_evidence)", "failed(provenance_drift)", "seed 4242",
                   "distill_verify.py", "sh run_distill.sh > distill.log 2>&1", "300.0 K",
                   "rc 137 => exhausted when elapsed_s >= limit_s", "kill_at_or_after_limit", "unknown which",
                   "sigkill_before_limit: source unknown", "own process group", "structure_init.vasp",
                   "NOT accuracy equivalence", "incomplete(lmp_witness_skipped)", "no `| tee`",
                   "failed(supervisor_exit)", "Witness boundary: p p f", "Physical contract", "boundary p p f",
                   "pbc [True, True, True]", "Refused at bootstrap: partial pbc"):
        assert needle in plan, needle
    assert "| tee distill.log" not in plan
    assert "OOM kill of the engine" not in plan  # no proven-source claim for any 137


def test_acceptance_with_user_heldout_file(tmp_path, fake_d, cu_structure):
    frames = [bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2) for _ in range(2)]
    frames[0].rattle(0.05, seed=2)  # neither frame may be the init structure itself
    frames[1].rattle(0.05, seed=3)
    user = tmp_path / "my_frames.extxyz"
    write(user, frames, format="extxyz")
    work = _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE, "--heldout-file", str(user))
    acc = json.loads((work / "acceptance.json").read_text())
    held = acc["split"]["heldout"]
    assert held["source"] == "user_frames_relabelled_by_teacher"
    assert held["input_file"] == str(user.resolve()) and held["input_sha256"] == db._sha256(user)
    assert held["seed"] is None
    assert held["path"] == str(work / "heldout" / "heldout.extxyz")
    text = (work / "run_distill.sh").read_text()
    assert f'python {db._q(fake_d.resolve() / "scripts" / "label.py")} {db._q(user.resolve())} "$HELDOUT"' in text
    assert "TEACHER_SEED" not in text.split("HELDOUT=")[1].split("exec")[0]


def test_acceptance_user_heldout_with_foreign_species_is_refused(tmp_path, fake_d, cu_structure):
    from ase import Atoms
    foreign = Atoms("CuO", positions=[[0, 0, 0], [0, 0, 2]], cell=[10, 10, 10], pbc=True)
    user = tmp_path / "foreign.extxyz"
    write(user, foreign, format="extxyz")
    with pytest.raises(SystemExit, match=r"species \['O'\]"):
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(tmp_path / "w"),
                 "--repo", str(fake_d), "--acceptance", *ACC_BASE, "--heldout-file", str(user)])
    assert not (tmp_path / "w").exists()


def test_acceptance_missing_user_heldout_file_is_a_usage_error(tmp_path, fake_d, cu_structure):
    with pytest.raises(SystemExit) as exc:
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(tmp_path / "w"),
                 "--repo", str(fake_d), "--acceptance", *ACC_BASE, "--heldout-file", str(tmp_path / "nope.xyz")])
    assert exc.value.code == 2


def test_without_acceptance_nothing_new_is_written(bootstrapped):
    assert not (bootstrapped / "acceptance.json").exists()
    assert not (bootstrapped / "PLAN.md").exists()
    assert "HELDOUT" not in (bootstrapped / "run_distill.sh").read_text()
    cfg = yaml.safe_load((bootstrapped / "config.yaml").read_text())
    assert cfg["al_loop"]["max_iter"] == 30  # example values untouched without --acceptance


def test_without_acceptance_target_ps_defaults_to_demo_value(tmp_path, fake_d, cu_structure):
    work = tmp_path / "w_default"
    db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(work), "--repo", str(fake_d)])
    assert yaml.safe_load((work / "config.yaml").read_text())["al_loop"]["target_ps"] == 1.0


# ── --wallclock-max-h: enforced by run_distill.sh, or refused ────────────────
def test_wallclock_is_refused_without_gnu_timeout(tmp_path, fake_d, cu_structure, capsys, monkeypatch):
    monkeypatch.setattr(db, "_find_gnu_timeout", lambda: None)
    with pytest.raises(SystemExit) as exc:
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(tmp_path / "w"),
                 "--repo", str(fake_d), "--acceptance", *ACC_BASE, "--wallclock-max-h", "2"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "cannot be enforced" in err and "Refusing" in err
    assert not (tmp_path / "w").exists()  # an unenforceable bound is never recorded


def test_wallclock_needs_acceptance_and_a_positive_value(tmp_path, fake_d, cu_structure):
    for extra in (["--wallclock-max-h", "2"], ["--acceptance", *ACC_BASE, "--wallclock-max-h", "0"]):
        with pytest.raises(SystemExit) as exc:
            db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(tmp_path / "w"),
                     "--repo", str(fake_d), *extra])
        assert exc.value.code == 2


def test_without_wallclock_the_loop_is_still_execd(tmp_path, fake_d, cu_structure):
    work = _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE)
    text = (work / "run_distill.sh").read_text()
    assert f'exec {db._q(fake_d.resolve() / "scripts" / "al_loop_local.sh")}' in text
    assert "timeout" not in text and "MARKER" not in text and "OMM_DISTILL_ATTEMPT" not in text
    acc = json.loads((work / "acceptance.json").read_text())
    assert acc["budget"]["wallclock_max_h"] is None and acc["budget"]["wallclock_enforcement"] is None
    assert "none stated (unbounded" in (work / "PLAN.md").read_text()


def _fake_loop_that_hangs(fake_d: Path) -> None:
    """al_loop_local.sh stand-in: writes an in-progress status, then parks a
    grandchild `sleep` and waits -- the shape of a training step that never
    finishes inside the budget."""
    (fake_d / "scripts" / "al_loop_local.sh").write_text(textwrap.dedent("""\
        #!/bin/bash
        RUN="$(dirname "$ONTHEFLY_CONFIG")/run"
        echo "round0 training epoch 3" > "$RUN/.al_status"
        sleep 300 &
        echo $! > "$RUN/grandchild.pid"
        wait
        """))
    (fake_d / "scripts" / "al_loop_local.sh").chmod(0o755)


def _prime_for_launch(work: Path) -> None:
    # dataset + held-out already present => run_distill.sh skips both seeding
    # steps and goes straight to the (supervised) hand-off.
    (work / "run").mkdir(exist_ok=True)
    (work / "run" / "dataset.extxyz").write_text("1\nLattice=\"1 0 0 0 1 0 0 0 1\" Properties=species:S:1:pos:R:3\nCu 0 0 0\n")
    (work / "heldout").mkdir(exist_ok=True)
    (work / "heldout" / "heldout.extxyz").write_text("1\nLattice=\"1 0 0 0 1 0 0 0 1\" Properties=species:S:1:pos:R:3\nCu 0.5 0 0\n")


def _run_sh(work: Path, timeout: int = 60) -> subprocess.CompletedProcess:
    # `python` on PATH: run_distill.sh pins the teacher env's bin first; where
    # that env is absent (CI) the test interpreter's dir is the fallback.
    import os
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"}
    return subprocess.run(["sh", str(work / "run_distill.sh")], capture_output=True, text=True, timeout=timeout, env=env)


def _assert_bound_marker(work: Path, *, state: str, rc: int, limit_s: int) -> dict:
    """The supervisor's two records: same attempt id/start time, the sha256
    of the script that ran == the approved one, and the outcome."""
    attempt = json.loads((work / ".distill" / "attempt.json").read_text())
    marker = json.loads((work / ".distill" / "wallclock.json").read_text())
    approved = json.loads((work / "acceptance.json").read_text())["provenance"]["generated_files"]["run_distill.sh"]["sha256"]
    assert attempt["schema"] == db.ATTEMPT_SCHEMA and marker["schema"] == db.WALLCLOCK_SCHEMA
    assert marker["attempt"] == attempt["attempt"] and marker["started_epoch_s"] == attempt["started_epoch_s"]
    assert attempt["run_script_sha256"] == marker["run_script_sha256"] == approved == db._sha256(work / "run_distill.sh")
    assert marker["state"] == state and marker["returncode"] == rc
    assert marker["limit_s"] == attempt["limit_s"] == limit_s and marker["grace_s"] == db.WALLCLOCK_GRACE_S
    assert marker["elapsed_s"] == marker["ended_epoch_s"] - marker["started_epoch_s"] >= 0
    assert marker["covers"] == db.WALLCLOCK_COVERS
    # the supervisor's attribution token is the one the verifier recomputes
    import distill_verify as dv
    assert marker["attribution"] == dv.wallclock_attribution(rc, marker["elapsed_s"], limit_s)
    assert marker["attribution"] in db.WALLCLOCK_ATTRIBUTION
    return marker


@pytest.mark.skipif(db._find_gnu_timeout() is None, reason="GNU coreutils timeout not on PATH")
def test_wallclock_exhaustion_terminates_the_loop_group_and_is_unmet(tmp_path, fake_d, cu_structure):
    import os
    import time
    import distill_verify as dv
    _fake_loop_that_hangs(fake_d)
    work = _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE, "--wallclock-max-h", str(2 / 3600))  # 2 s
    _prime_for_launch(work)
    t0 = time.monotonic()
    proc = _run_sh(work)
    elapsed = time.monotonic() - t0
    assert proc.returncode == 124, proc.stdout + proc.stderr  # GNU timeout: expired, TERM sufficed
    assert elapsed < 30  # nowhere near the 300 s the loop wanted, and no 60 s KILL grace was needed
    assert "wall-clock exhausted" in proc.stdout
    marker = _assert_bound_marker(work, state="exhausted", rc=124, limit_s=2)
    assert 1 <= marker["elapsed_s"] <= 10
    # the grandchild `sleep` went with the process group -- nothing owned is left running
    pid = int((work / "run" / "grandchild.pid").read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, 9)
        pytest.fail(f"grandchild {pid} survived the wall-clock termination")
    # evidence preserved: the engine's own in-progress status is untouched
    assert (work / "run" / ".al_status").read_text().startswith("round0 training")
    # and the verdict is budget exhaustion, not incomplete and never success
    rc = dv.main(["--work", str(work), "--no-lmp-witness"])
    report = json.loads((work / "verify" / "distill_verify.json").read_text())
    assert rc == 1 and (report["state"], report["reason"]) == ("unmet", "budget:wallclock")
    assert report["wallclock"]["exhausted"] is True and report["wallclock"]["elapsed_s"] == marker["elapsed_s"]
    assert report["proposal"]["options"][0]["flag"].startswith("--wallclock-max-h")


@pytest.mark.skipif(db._find_gnu_timeout() is None, reason="GNU coreutils timeout not on PATH")
def test_wallclock_within_bound_records_marker_and_engine_rc(tmp_path, fake_d, cu_structure):
    (fake_d / "scripts" / "al_loop_local.sh").write_text(textwrap.dedent("""\
        #!/bin/bash
        RUN="$(dirname "$ONTHEFLY_CONFIG")/run"
        echo "SUCCESS round0 stable" > "$RUN/.al_status"
        exit 0
        """))
    (fake_d / "scripts" / "al_loop_local.sh").chmod(0o755)
    work = _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE, "--wallclock-max-h", "1")
    _prime_for_launch(work)
    proc = _run_sh(work)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    marker = _assert_bound_marker(work, state="within", rc=0, limit_s=3600)
    assert marker["elapsed_s"] <= 5


@pytest.mark.skipif(db._find_gnu_timeout() is None, reason="GNU coreutils timeout not on PATH")
def test_wallclock_engine_nonzero_exit_is_passed_through(tmp_path, fake_d, cu_structure):
    (fake_d / "scripts" / "al_loop_local.sh").write_text("#!/bin/bash\nexit 3\n")
    (fake_d / "scripts" / "al_loop_local.sh").chmod(0o755)
    work = _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE, "--wallclock-max-h", "1")
    _prime_for_launch(work)
    proc = _run_sh(work)
    assert proc.returncode == 3
    _assert_bound_marker(work, state="engine_exit", rc=3, limit_s=3600)


@pytest.mark.skipif(db._find_gnu_timeout() is None, reason="GNU coreutils timeout not on PATH")
def test_wallclock_new_attempt_removes_the_previous_marker_first(tmp_path, fake_d, cu_structure):
    # attempt 1 finishes within; attempt 2 hangs -> its start record replaces
    # attempt 1's and attempt 1's `within` marker is gone before the loop runs
    (fake_d / "scripts" / "al_loop_local.sh").write_text("#!/bin/bash\nexit 0\n")
    (fake_d / "scripts" / "al_loop_local.sh").chmod(0o755)
    work = _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE, "--wallclock-max-h", str(2 / 3600))
    _prime_for_launch(work)
    assert _run_sh(work).returncode == 0
    first = json.loads((work / ".distill" / "wallclock.json").read_text())
    _fake_loop_that_hangs(fake_d)
    assert _run_sh(work).returncode == 124
    second = json.loads((work / ".distill" / "wallclock.json").read_text())
    attempt = json.loads((work / ".distill" / "attempt.json").read_text())
    assert second["attempt"] == attempt["attempt"] != first["attempt"]
    assert (first["state"], second["state"]) == ("within", "exhausted")


# ── the bound covers the WHOLE run, and the held-out drops teacher_md's frame 0 ──
FAKE_TEACHER_MD = '''\
"""Stand-in for D/scripts/teacher_md.py with its load-bearing behaviour: the
init structure is saved as frame 0 BEFORE the first MD step, for ANY seed;
later frames depend on $TEACHER_SEED. Pure stdlib (runs under any python)."""
import os, sys, time
out, steps, every = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
seed = int(os.environ.get("TEACHER_SEED", "42"))
if os.environ.get("FAKE_TEACHER_MD_HANG"):
    time.sleep(300)
def frame(dx):
    return ('2\\nLattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3:forces:R:3 energy=-1.0 pbc="T T T"\\n'
            f"Cu 0.0 0.0 0.0 0 0 0\\nCu {1.8 + dx:.4f} 1.8 1.8 0 0 0\\n")
with open(out, "w") as fh:
    fh.write(frame(0.0))                              # frame 0: the init structure, seed-independent
    for k in range(steps // every):
        fh.write(frame(0.01 * (k + 1) * seed))        # MD frames: seed-dependent
'''


def _install_fake_teacher_md(fake_d: Path) -> None:
    (fake_d / "scripts" / "teacher_md.py").write_text(FAKE_TEACHER_MD)
    (fake_d / "scripts" / "al_loop_local.sh").write_text(textwrap.dedent("""\
        #!/bin/bash
        RUN="$(dirname "$ONTHEFLY_CONFIG")/run"
        echo "SUCCESS round0 stable" > "$RUN/.al_status"
        """))
    (fake_d / "scripts" / "al_loop_local.sh").chmod(0o755)


def _prime_pool_only(work: Path) -> None:
    # dataset present => pool seeding skipped (the fake D has no merge_xyz);
    # NO held-out => run_distill.sh must generate it
    (work / "run").mkdir(exist_ok=True)
    (work / "run" / "dataset.extxyz").write_text("1\nLattice=\"1 0 0 0 1 0 0 0 1\" Properties=species:S:1:pos:R:3\nCu 0 0 0\n")


def test_generated_heldout_drops_the_shared_frame0(tmp_path, fake_d, cu_structure):
    import distill_verify as dv
    _install_fake_teacher_md(fake_d)
    work = _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE, "--heldout-steps", "30", "--heldout-save-every", "10")
    _prime_pool_only(work)
    proc = _run_sh(work)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "dropped 1 leading init-structure frame(s)" in proc.stdout and "kept 3 of 4" in proc.stdout
    raw = dv.frame_fingerprints(work / "heldout" / "heldout_md_raw.extxyz")
    kept = dv.frame_fingerprints(work / "heldout" / "heldout.extxyz")
    assert len(raw) == 4 and len(kept) == 3 and kept < raw
    # what was dropped is exactly the seed-independent frame 0: a pool MD under
    # the pool's seed starts with the same frame, and nothing else overlaps
    import os
    pool_raw = work / "pool_check.extxyz"
    subprocess.run([sys.executable, str(fake_d / "scripts" / "teacher_md.py"), str(pool_raw), "20", "10"],
                   check=True, env={**os.environ, "TEACHER_SEED": str(db.POOL_TEACHER_SEED)})
    pool = dv.frame_fingerprints(pool_raw)
    assert len(raw & pool) == 1 and not (kept & pool)
    # rerun: the held-out is kept, not regenerated (idempotent like the pool seeding)
    before = (work / "heldout" / "heldout.extxyz").read_bytes()
    assert _run_sh(work).returncode == 0 and (work / "heldout" / "heldout.extxyz").read_bytes() == before


def test_generated_heldout_drops_a_duplicated_init_frame(tmp_path, fake_d, cu_structure):
    # the real teacher_md.py saves the init structure explicitly AND through its
    # step-0 observer call: two identical leading frames. Both must be dropped,
    # otherwise the second copy is shared with the pool.
    import os
    import distill_verify as dv
    _install_fake_teacher_md(fake_d)
    (fake_d / "scripts" / "teacher_md.py").write_text(
        FAKE_TEACHER_MD.replace("    fh.write(frame(0.0))", "    fh.write(frame(0.0))\n    fh.write(frame(0.0))"))
    work = _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE, "--heldout-steps", "30", "--heldout-save-every", "10")
    _prime_pool_only(work)
    proc = _run_sh(work)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "dropped 2 leading init-structure frame(s)" in proc.stdout and "kept 3 of 5" in proc.stdout
    kept = dv.frame_fingerprints(work / "heldout" / "heldout.extxyz")
    pool_raw = work / "pool_check.extxyz"
    subprocess.run([sys.executable, str(fake_d / "scripts" / "teacher_md.py"), str(pool_raw), "20", "10"],
                   check=True, env={**os.environ, "TEACHER_SEED": str(db.POOL_TEACHER_SEED)})
    assert len(kept) == 3 and not (kept & dv.frame_fingerprints(pool_raw))


def test_generated_heldout_with_only_frame0_is_refused_at_run_time(tmp_path, fake_d, cu_structure):
    # bootstrap refuses steps < save_every; a teacher MD that nevertheless
    # yields only frame 0 (e.g. an engine change) must not leave an empty set
    _install_fake_teacher_md(fake_d)
    with pytest.raises(SystemExit) as exc:
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(tmp_path / "w"),
                 "--repo", str(fake_d), "--acceptance", *ACC_BASE, "--heldout-steps", "5", "--heldout-save-every", "10"])
    assert exc.value.code == 2 and not (tmp_path / "w").exists()
    work = _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE, "--heldout-steps", "10", "--heldout-save-every", "10")
    _prime_pool_only(work)
    (fake_d / "scripts" / "teacher_md.py").write_text(FAKE_TEACHER_MD.replace("steps // every", "0"))
    proc = _run_sh(work)
    assert proc.returncode != 0 and "leaves nothing" in proc.stderr
    assert not (work / "heldout" / "heldout.extxyz").exists()
    assert not (work / "run" / ".al_status").exists()  # the loop was never reached


@pytest.mark.parametrize("flag,val", [("--pool-steps", "0"), ("--heldout-save-every", "0")])
def test_step_counts_must_be_positive(tmp_path, fake_d, cu_structure, flag, val):
    with pytest.raises(SystemExit) as exc:
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(tmp_path / "w"),
                 "--repo", str(fake_d), "--acceptance", *ACC_BASE, flag, val])
    assert exc.value.code == 2


def test_user_heldout_containing_the_init_structure_is_refused(tmp_path, fake_d, cu_structure):
    from ase.io import read as ase_read
    init = ase_read(cu_structure)
    other = init.copy()
    other.rattle(0.05, seed=5)
    user = tmp_path / "with_init.extxyz"
    write(user, [other, init], format="extxyz")
    with pytest.raises(SystemExit, match=r"frame\(s\) \[1\] are the init structure"):
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(tmp_path / "w"),
                 "--repo", str(fake_d), "--acceptance", *ACC_BASE, "--heldout-file", str(user)])
    assert not (tmp_path / "w").exists()


@pytest.mark.skipif(db._find_gnu_timeout() is None, reason="GNU coreutils timeout not on PATH")
def test_wallclock_bounds_heldout_generation_not_only_the_loop(tmp_path, fake_d, cu_structure):
    import os
    import distill_verify as dv
    _install_fake_teacher_md(fake_d)
    work = _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE, "--wallclock-max-h", str(2 / 3600))
    _prime_pool_only(work)
    env = {**os.environ, "FAKE_TEACHER_MD_HANG": "1", "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"}
    proc = subprocess.run(["sh", str(work / "run_distill.sh")], capture_output=True, text=True, timeout=60, env=env)
    assert proc.returncode == 124, proc.stdout + proc.stderr
    marker = _assert_bound_marker(work, state="exhausted", rc=124, limit_s=2)
    assert 1 <= marker["elapsed_s"] <= 10
    assert not (work / "heldout" / "heldout.extxyz").exists()  # the hang was in held-out generation
    assert not (work / "run" / ".al_status").exists()          # the loop never started
    rc = dv.main(["--work", str(work), "--no-lmp-witness"])
    report = json.loads((work / "verify" / "distill_verify.json").read_text())
    assert rc == 1 and (report["state"], report["reason"]) == ("unmet", "budget:wallclock")


def test_config_header_names_the_identity_rule(accepted):
    head = (accepted / "config.yaml").read_text().splitlines()[0]
    assert "provenance_drift" in head and "edit directly" not in head


# ── rc 137: an early SIGKILL is an engine/signal failure, never budget expiry ──
@pytest.mark.skipif(db._find_gnu_timeout() is None, reason="GNU coreutils timeout not on PATH")
def test_early_sigkill_of_the_engine_is_engine_exit_not_exhaustion(tmp_path, fake_d, cu_structure):
    import distill_verify as dv
    # the loop (exec'd by the supervised child) is OOM-killed / kill -9'ed
    # seconds in: timeout re-raises KILL -> rc 137, long before the 1 h bound
    (fake_d / "scripts" / "al_loop_local.sh").write_text(textwrap.dedent("""\
        #!/bin/bash
        RUN="$(dirname "$ONTHEFLY_CONFIG")/run"
        echo "round0 training epoch 1" > "$RUN/.al_status"
        kill -9 $$
        """))
    (fake_d / "scripts" / "al_loop_local.sh").chmod(0o755)
    work = _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE, "--wallclock-max-h", "1")
    _prime_for_launch(work)
    proc = _run_sh(work)
    assert proc.returncode == 137, proc.stdout + proc.stderr
    assert "wall-clock engine_exit (sigkill_before_limit): rc=137" in proc.stdout
    marker = _assert_bound_marker(work, state="engine_exit", rc=137, limit_s=3600)
    assert marker["elapsed_s"] < 3600 and marker["attribution"] == "sigkill_before_limit"
    rc = dv.main(["--work", str(work), "--no-lmp-witness"])
    report = json.loads((work / "verify" / "distill_verify.json").read_text())
    # the marker is valid evidence; the run it describes died, so it failed --
    # not "still running", and never a pass whatever .al_status might say
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "supervisor_exit")
    assert "rc=137" in report["detail"] and "source unknown" in report["detail"]
    assert report["wallclock"]["valid"] is True and report["wallclock"]["exhausted"] is False
    assert report["proposal"] is None  # no "longer wall-clock" proposal: the budget was not the problem


@pytest.mark.skipif(db._find_gnu_timeout() is None, reason="GNU coreutils timeout not on PATH")
def test_late_sigkill_is_exhaustion_recorded_as_ambiguous_never_a_proven_source(tmp_path, fake_d, cu_structure):
    import distill_verify as dv
    # rc 137 AFTER the 1 s limit: from rc + elapsed alone this is timeout's
    # KILL after its grace or (as here) an external kill -9 that landed after
    # the deadline. The supervisor must not claim to know which.
    # (TERM is ignored so timeout's expiry at 1 s does not end the run itself;
    # a KILL lands >= 2 s in -- from rc + elapsed indistinguishable from
    # timeout's own KILL after its grace)
    (fake_d / "scripts" / "al_loop_local.sh").write_text(textwrap.dedent("""\
        #!/bin/bash
        trap '' TERM
        RUN="$(dirname "$ONTHEFLY_CONFIG")/run"
        echo "SUCCESS round0 stable" > "$RUN/.al_status"
        while [ "$SECONDS" -lt 2 ]; do sleep 0.2; done
        kill -9 $$
        """))
    (fake_d / "scripts" / "al_loop_local.sh").chmod(0o755)
    work = _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE, "--wallclock-max-h", str(1 / 3600))
    _prime_for_launch(work)
    proc = _run_sh(work)
    assert proc.returncode == 137, proc.stdout + proc.stderr
    assert "wall-clock exhausted (kill_at_or_after_limit): rc=137" in proc.stdout
    marker = _assert_bound_marker(work, state="exhausted", rc=137, limit_s=1)
    assert marker["elapsed_s"] >= 1 and marker["attribution"] == "kill_at_or_after_limit"
    rc = dv.main(["--work", str(work), "--no-lmp-witness"])
    report = json.loads((work / "verify" / "distill_verify.json").read_text())
    assert rc == 1 and (report["state"], report["reason"]) == ("unmet", "budget:wallclock")
    assert "cannot tell which" in report["detail"] and report["heldout_eval"]["ran"] is False


@pytest.mark.skipif(db._find_gnu_timeout() is None, reason="GNU coreutils timeout not on PATH")
def test_term_to_the_supervised_group_is_engine_exit(tmp_path, fake_d, cu_structure):
    # the documented early-stop recipe (PLAN.md process-group note): TERM to
    # the group timeout created (pgid = timeout's pid) ends the whole run and
    # is recorded as engine_exit, never as exhaustion
    import os
    import signal
    import time
    _fake_loop_that_hangs(fake_d)
    work = _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE, "--wallclock-max-h", "1")
    _prime_for_launch(work)
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"}
    proc = subprocess.Popen(["sh", str(work / "run_distill.sh")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env)
    pid_file = work / "run" / "grandchild.pid"
    for _ in range(100):
        if pid_file.is_file() and pid_file.read_text().strip():
            break
        time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("the fake loop never started")
    grandchild = int(pid_file.read_text())
    pgid = os.getpgid(grandchild)
    assert pgid != os.getpgid(proc.pid)  # timeout's own group, not the supervisor's
    os.killpg(pgid, signal.SIGTERM)
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 143, out
    _assert_bound_marker(work, state="engine_exit", rc=143, limit_s=3600)
    for _ in range(50):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(grandchild, 9)
        pytest.fail(f"grandchild {grandchild} survived TERM to its group")


# ── every rendered path is shell-quoted: spaces, apostrophes, $(...) ──────────
HOSTILE = "a b'c$(touch pwned)d"


@pytest.mark.skipif(db._find_gnu_timeout() is None, reason="GNU coreutils timeout not on PATH")
def test_hostile_paths_are_used_literally_end_to_end(tmp_path, fake_d, cu_structure, monkeypatch):
    import shutil as _sh
    import distill_verify as dv
    _install_fake_teacher_md(fake_d)
    hostile_d = tmp_path / f"engine {HOSTILE}"
    _sh.copytree(fake_d, hostile_d)
    struct_dir = tmp_path / f"struct {HOSTILE}"
    struct_dir.mkdir()
    structure = struct_dir / "Cu32.vasp"
    _sh.copy(cu_structure, structure)
    real = resolve("MACE-MPA-0")
    hostile_python = str(tmp_path / f"env {HOSTILE}" / "bin" / "python")
    monkeypatch.setattr(db, "resolve", lambda name: {**real, "python": hostile_python})
    work = tmp_path / f"work {HOSTILE}"
    rc = db.main(["--teacher", "MACE-MPA-0", "--structure", str(structure), "--work", str(work), "--repo",
                  str(hostile_d), "--acceptance", *ACC_BASE, "--wallclock-max-h", "1",
                  "--heldout-steps", "20", "--heldout-save-every", "10"])
    assert rc == 0
    work = work.resolve()
    _prime_pool_only(work)
    proc = _run_sh(work)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "leading init-structure frame(s)" in proc.stdout
    # nothing was expanded: no `pwned` anywhere, nothing redirected into a stray file
    assert not list(tmp_path.rglob("pwned")) and not list(Path.cwd().glob("pwned"))
    assert (work / "heldout" / "heldout.extxyz").is_file() and (work / "run" / ".al_status").is_file()
    _assert_bound_marker(work, state="within", rc=0, limit_s=3600)
    attempt = json.loads((work / ".distill" / "attempt.json").read_text())
    assert attempt["run_script"] == str(work / "run_distill.sh")  # the hostile path, verbatim, as JSON
    text = (work / "run_distill.sh").read_text()
    assert f"PATH={db._q(str(Path(hostile_python).parent))}:$PATH" in text
    # nothing is rendered bare-double-quoted around an interpolated path any more
    for needle in (str(work), str(hostile_d), hostile_python):
        assert f'"{needle}' not in text.replace(json.dumps(str(work / "run_distill.sh")), "")
    # and the verifier reads the same hostile locations back
    rc = dv.main(["--work", str(work), "--no-lmp-witness"])
    report = json.loads((work / "verify" / "distill_verify.json").read_text())
    assert report["wallclock"]["valid"] is True and report["wallclock"]["state"] == "within"
    assert report["contract"]["provenance_drift"] == {}


# ── acceptance-only flags are refused (never ignored) without --acceptance ──
@pytest.mark.parametrize("extra", [["--pool-steps", "999"], ["--pool-save-every", "3"], ["--seed", "7"],
                                   ["--mode", "production"], ["--heldout-steps", "50"], ["--wallclock-max-h", "2"]])
def test_acceptance_only_flag_without_acceptance_is_a_usage_error(tmp_path, fake_d, cu_structure, capsys, extra):
    with pytest.raises(SystemExit) as exc:
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(tmp_path / "w"),
                 "--repo", str(fake_d), *extra])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert extra[0] in err and "--acceptance" in err
    assert not (tmp_path / "w").exists()


def test_pool_flags_are_honoured_and_absent_means_the_default(tmp_path, fake_d, cu_structure):
    plain = tmp_path / "plain"
    db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(plain), "--repo", str(fake_d)])
    assert f'"$WORK/teacher_md.extxyz" {db.TEACHER_MD_STEPS} {db.TEACHER_MD_SAVE_EVERY}' in (plain / "run_distill.sh").read_text()
    work = _acc_main(tmp_path, fake_d, cu_structure, *ACC_BASE, "--pool-steps", "999", "--pool-save-every", "3")
    assert '"$WORK/teacher_md.extxyz" 999 3' in (work / "run_distill.sh").read_text()
    acc = json.loads((work / "acceptance.json").read_text())
    assert (acc["split"]["pool"]["steps"], acc["split"]["pool"]["save_every"]) == (999, 3)


# ── structure staging: D's teacher_md.py reads format="vasp" only ────────────
def test_non_vasp_structure_is_staged_as_vasp_and_the_source_is_untouched(tmp_path, fake_d, cu_structure):
    from ase.io import read as ase_read
    xyz = tmp_path / "cu.xyz"
    write(xyz, bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2), format="extxyz")
    before = db._sha256(xyz)
    work = tmp_path / "w_xyz"
    rc = db.main(["--teacher", "MACE-MPA-0", "--structure", str(xyz), "--work", str(work), "--repo", str(fake_d),
                  "--acceptance", *ACC_BASE])
    assert rc == 0
    work = work.resolve()
    staged = work / "structure_init.vasp"
    assert staged.is_file() and db._sha256(xyz) == before
    cfg = yaml.safe_load((work / "config.yaml").read_text())
    assert cfg["system"]["init_structure"] == str(staged)
    # readable exactly the way teacher_md.py reads it, and the same frame
    assert db.frame_fingerprint(ase_read(str(staged), format="vasp")) == db.frame_fingerprint(ase_read(str(xyz)))
    acc = json.loads((work / "acceptance.json").read_text())
    s = acc["provenance"]["structure"]
    assert s["path"] == str(xyz.resolve()) and s["staged_path"] == str(staged)
    assert s["staged_sha256"] == db._sha256(staged) == acc["provenance"]["generated_files"]["structure_init.vasp"]["sha256"]


def test_structure_without_a_cell_is_refused_before_anything_is_written(tmp_path, fake_d):
    xyz = tmp_path / "nocell.xyz"
    write(xyz, molecule("H2O"), format="xyz")
    with pytest.raises(SystemExit, match="no periodic cell"):
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(xyz), "--work", str(tmp_path / "w"), "--repo", str(fake_d)])
    assert not (tmp_path / "w").exists()


def test_unreadable_structure_is_a_clean_refusal(tmp_path, fake_d):
    bad = tmp_path / "garbage.cif"
    bad.write_text("this is not a structure\n")
    with pytest.raises(SystemExit, match="cannot read it"):
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(bad), "--work", str(tmp_path / "w"), "--repo", str(fake_d)])
    assert not (tmp_path / "w").exists()


# ── F3: a newline in a filename cannot break out of a generated `#` comment ──
def _install_fake_label_py(fake_d: Path) -> None:
    (fake_d / "scripts" / "label.py").write_text(textwrap.dedent('''\
        """Stand-in for D/scripts/label.py: copies the input frames to the output."""
        import shutil, sys
        shutil.copy(sys.argv[1], sys.argv[2])
        '''))


@pytest.mark.skipif(db._find_gnu_timeout() is None, reason="GNU coreutils timeout not on PATH")
def test_newline_in_heldout_filename_is_rendered_inert_end_to_end(tmp_path, fake_d, cu_structure):
    import distill_verify as dv
    _install_fake_teacher_md(fake_d)
    _install_fake_label_py(fake_d)
    # the injected command is relative: the script is run with cwd=tmp_path,
    # so a break-out would create tmp_path/canary_from_comment
    canary = tmp_path / "canary_from_comment"
    frames = [bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2) for _ in range(2)]
    frames[0].rattle(0.05, seed=2)
    frames[1].rattle(0.05, seed=3)
    # a REAL newline in the name: the raw text would end a `#` comment and hand
    # the remainder to the shell as the next command
    hostile_dir = tmp_path / "held\ntouch canary_from_comment\n#"
    hostile_dir.mkdir()
    user = hostile_dir / "frames\ntouch canary_from_comment\n#.extxyz"
    write(user, frames, format="extxyz")
    struct_dir = tmp_path / "struct\ntouch canary_from_comment\n#"
    struct_dir.mkdir()
    structure = struct_dir / "Cu32.vasp"
    shutil.copy(cu_structure, structure)
    work = tmp_path / "w_newline"
    rc = db.main(["--teacher", "MACE-MPA-0", "--structure", str(structure), "--work", str(work), "--repo",
                  str(fake_d), "--acceptance", *ACC_BASE, "--wallclock-max-h", "1", "--heldout-file", str(user)])
    assert rc == 0
    work = work.resolve()
    text = (work / "run_distill.sh").read_text()
    # static: the raw path occurs exactly once, as the single-quoted label.py
    # ARGUMENT (legal sh, inert); the comment carries the JSON-escaped form
    # and no comment line ends inside the path
    quoted_arg = db._q(user.resolve())
    assert quoted_arg in text and quoted_arg.count("\ntouch canary_from_comment\n") == 2  # dir + file name
    assert text.count("\ntouch canary_from_comment\n") == quoted_arg.count("\ntouch canary_from_comment\n")
    assert f"# Held-out set (recipes/distill.md §2 item 4): relabel user frames {db._c(user.resolve())}" in text
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            assert "\n" not in line and "canary_from_comment\n" not in line
    assert subprocess.run(["sh", "-n", str(work / "run_distill.sh")]).returncode == 0
    # dynamic: the script runs to completion (from tmp_path, where the injected
    # `touch` would land) and nothing was created
    import os
    _prime_pool_only(work)
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"}
    proc = subprocess.run(["sh", str(work / "run_distill.sh")], capture_output=True, text=True, timeout=60,
                          env=env, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not canary.exists() and not list(tmp_path.rglob("canary_from_comment"))
    assert not list(Path.cwd().glob("canary_from_comment"))
    assert (work / "heldout" / "heldout.extxyz").is_file() and (work / "run" / ".al_status").is_file()
    _assert_bound_marker(work, state="within", rc=0, limit_s=3600)
    # the proposal carries the path verbatim as data (JSON) and one-line in the plan
    acc = json.loads((work / "acceptance.json").read_text())
    assert acc["split"]["heldout"]["input_file"] == str(user.resolve())
    assert acc["provenance"]["structure"]["path"] == str(structure.resolve())
    plan = (work / "PLAN.md").read_text()
    assert db._c(user.resolve()) in plan and db._c(structure.resolve()) in plan
    assert "\ntouch" not in plan
    rc = dv.main(["--work", str(work), "--no-lmp-witness"])
    report = json.loads((work / "verify" / "distill_verify.json").read_text())
    assert report["contract"]["provenance_drift"] == {} and report["wallclock"]["state"] == "within"


def test_every_generated_comment_escapes_control_characters(tmp_path, fake_d, cu_structure):
    # every host-derived string that can land in a comment or a markdown line
    # goes through _c: one line, JSON-escaped
    for raw in ("a\nb", "a\rb", "tab\there", "quote\"s", "back\\slash", "\x1b[31mred"):
        rendered = db._c(raw)
        assert "\n" not in rendered and "\r" not in rendered and rendered.startswith('"') and rendered.endswith('"')
        assert json.loads(rendered) == raw
    block = db.render_heldout_block(repo_abs="/d", work_dir_abs=tmp_path / "w", heldout_file=Path("x\ny.extxyz"),
                                    heldout_seed=1, heldout_steps=1, heldout_save_every=1)
    comment_lines = [ln for ln in block.splitlines() if ln.lstrip().startswith("#")]
    assert any('x\\ny.extxyz' in ln for ln in comment_lines)  # escaped, on the comment line itself
    # the only raw newline from the path sits inside the single-quoted
    # ARGUMENT (legal sh), never in a comment
    raw_lines = [ln for ln in block.splitlines() if ln.startswith("y.extxyz")]
    assert raw_lines == ["""y.extxyz' "$HELDOUT\""""]


# ── F4: an unreadable / empty user held-out set is refused before any output ──
def test_unparseable_user_heldout_is_refused_before_anything_is_written(tmp_path, fake_d, cu_structure):
    bad = tmp_path / "garbage.extxyz"
    bad.write_text("not a frame set\n\n??\n")
    with pytest.raises(SystemExit) as exc:
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(tmp_path / "w"),
                 "--repo", str(fake_d), "--acceptance", *ACC_BASE, "--heldout-file", str(bad)])
    msg = str(exc.value)
    assert "--heldout-file" in msg and "cannot parse" in msg and "drop the flag" in msg
    assert not (tmp_path / "w").exists()


def test_empty_user_heldout_is_refused_before_anything_is_written(tmp_path, fake_d, cu_structure):
    empty = tmp_path / "empty.extxyz"
    empty.write_text("")
    with pytest.raises(SystemExit, match=r"no frames|cannot parse"):
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(tmp_path / "w"),
                 "--repo", str(fake_d), "--acceptance", *ACC_BASE, "--heldout-file", str(empty)])
    assert not (tmp_path / "w").exists()


# ── F5: scientific identity the VASP staging cannot carry is refused, never converted ──
def _refused(tmp_path, fake_d, atoms, name: str, match: str) -> None:
    path = tmp_path / name
    write(path, atoms)
    with pytest.raises(SystemExit, match=match):
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(path), "--work", str(tmp_path / "w"), "--repo", str(fake_d)])
    assert not (tmp_path / "w").exists()


def test_partially_periodic_structure_is_refused_not_made_periodic(tmp_path, fake_d):
    slab = bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2)
    slab.center(vacuum=6.0, axis=2)
    slab.set_pbc([True, True, False])
    _refused(tmp_path, fake_d, slab, "slab_ttf.extxyz", r"pbc=\[True, True, False\] is not fully periodic")


def test_nonzero_initial_magmoms_are_refused(tmp_path, fake_d):
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2)
    atoms.set_initial_magnetic_moments([1.0] * len(atoms))
    _refused(tmp_path, fake_d, atoms, "magnetic.extxyz", "initial magnetic moments")


def test_nonzero_initial_charges_are_refused(tmp_path, fake_d):
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2)
    atoms.set_initial_charges([0.1] * len(atoms))
    _refused(tmp_path, fake_d, atoms, "charged.extxyz", "initial charges")


def test_constraint_other_than_fixatoms_is_refused(tmp_path, fake_d):
    from ase.constraints import FixBondLength
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2)
    atoms.set_constraint(FixBondLength(0, 1))
    _refused(tmp_path, fake_d, atoms, "bonded.traj", r"constraint\(s\) \['FixBondLengths'\]")  # ase's class name


def test_fixatoms_survive_staging_and_the_identity_is_recorded(tmp_path, fake_d):
    from ase.constraints import FixAtoms
    from ase.io import read as ase_read
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2)
    atoms.set_constraint(FixAtoms(indices=[0, 3, 5]))
    src = tmp_path / "fixed.traj"
    write(src, atoms)
    before = db._sha256(src)
    work = _acc_main(tmp_path, fake_d, src, *ACC_BASE)
    assert db._sha256(src) == before  # the source is never modified
    staged = ase_read(str(work / "structure_init.vasp"), format="vasp")
    assert db._fixed_indices(staged) == [0, 3, 5] and all(staged.pbc)
    assert np.allclose(staged.get_positions(), atoms.get_positions(), atol=1e-8)
    assert np.allclose(np.array(staged.get_cell()), np.array(atoms.get_cell()), atol=1e-8)
    acc = json.loads((work / "acceptance.json").read_text())
    ident = acc["provenance"]["structure"]["identity"]
    assert ident["pbc"] == [True, True, True] and ident["fixed_atoms"] == [0, 3, 5] and ident["n_atoms"] == 32
    assert "selective dynamics" in ident["fixed_atoms_note"] and ident["initial_momenta_present"] is False
    assert "partial pbc" in acc["provenance"]["structure"]["refused_at_bootstrap"]


def test_staging_that_does_not_round_trip_is_refused_and_the_copy_is_retained_not_unlinked(
        tmp_path, fake_d, cu_structure, monkeypatch):
    # if the writer ever stopped reproducing the structure, the run is refused
    # rather than handed a different structure; the rejected copy (created by
    # this invocation) is retained -- the script deletes nothing under --work --
    # and no other output is written; the source is untouched
    real_read = db.ase_read
    src_before = cu_structure.read_bytes()

    def drifted_read(path, *a, **kw):
        out = real_read(path, *a, **kw)
        if kw.get("format") == "vasp":
            out.positions[0, 0] += 0.5
        return out
    monkeypatch.setattr(db, "ase_read", drifted_read)
    work = tmp_path / "w"
    with pytest.raises(SystemExit, match="did not reproduce it .*positions.*retained.*nothing is deleted"):
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(work),
                 "--repo", str(fake_d)])
    assert sorted(os.listdir(work)) == ["bootstrap.log", "structure_init.vasp"]
    assert cu_structure.read_bytes() == src_before
    # the retained copy makes the next bootstrap into this work dir refuse
    with pytest.raises(SystemExit, match="is not empty .*structure_init.vasp"):
        db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(work),
                 "--repo", str(fake_d)])


# ── output ownership: new/empty --work, exclusive creation, nothing deleted ──

def _snapshot(root: Path) -> dict:
    """(relative path -> (bytes, inode, is_symlink, link target)) for every
    entry under root, symlinks not followed -- to prove nothing was touched."""
    out = {}
    for p in sorted(root.rglob("*")):
        st = p.lstat()
        out[str(p.relative_to(root))] = (
            p.read_bytes() if p.is_file() and not p.is_symlink() else None,
            st.st_ino, p.is_symlink(), os.readlink(p) if p.is_symlink() else None,
        )
    return out


def _boot(fake_d, cu_structure, work, *extra):
    return db.main(["--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(work),
                    "--repo", str(fake_d), *extra])


def test_source_equal_to_the_staged_path_is_refused_and_its_bytes_survive(tmp_path, fake_d, cu_structure):
    # the reviewer's measured defect: --structure == <work>/structure_init.vasp
    # used to be overwritten by the staging write (and unlinked on read-back
    # failure); now the non-empty work dir is refused before any write
    work = tmp_path / "w"
    work.mkdir()
    src = work / "structure_init.vasp"
    src.write_bytes(cu_structure.read_bytes())
    before = _snapshot(work)
    with pytest.raises(SystemExit, match=r"is not empty .*--structure .* lives inside it.*Nothing was written"):
        _boot(fake_d, cu_structure=src, work=work)
    assert _snapshot(work) == before
    assert sorted(os.listdir(work)) == ["structure_init.vasp"]


def test_unrelated_preexisting_staged_file_is_never_overwritten(tmp_path, fake_d, cu_structure):
    work = tmp_path / "w"
    work.mkdir()
    (work / "structure_init.vasp").write_text("someone else's POSCAR\n")
    before = _snapshot(work)
    with pytest.raises(SystemExit, match="is not empty .*structure_init.vasp"):
        _boot(fake_d, cu_structure, work)
    assert _snapshot(work) == before


def test_staged_symlink_to_an_external_file_is_neither_followed_nor_replaced(tmp_path, fake_d, cu_structure):
    external = tmp_path / "external" / "precious.vasp"
    external.parent.mkdir()
    external.write_text("external bytes that must survive\n")
    work = tmp_path / "w"
    work.mkdir()
    (work / "structure_init.vasp").symlink_to(external)
    before_ext = external.read_bytes()
    before = _snapshot(work)
    with pytest.raises(SystemExit, match="is not empty"):
        _boot(fake_d, cu_structure, work)
    assert external.read_bytes() == before_ext
    assert _snapshot(work) == before and (work / "structure_init.vasp").is_symlink()
    # and even if the directory gate were bypassed, the write itself refuses a
    # symlink at the output name instead of following it (O_EXCL|O_NOFOLLOW)
    with pytest.raises(SystemExit, match="already exists; bootstrap creates its outputs exclusively"):
        db._write_new(work / "structure_init.vasp", "would be written through the link\n")
    assert external.read_bytes() == before_ext


def test_hardlinked_source_at_the_staged_name_keeps_its_inode_and_bytes(tmp_path, fake_d, cu_structure):
    work = tmp_path / "w"
    work.mkdir()
    linked = work / "structure_init.vasp"
    os.link(cu_structure, linked)  # same inode as the user's source
    src_before = cu_structure.read_bytes()
    ino = cu_structure.stat().st_ino
    with pytest.raises(SystemExit, match="is not empty"):
        _boot(fake_d, cu_structure, work)
    assert cu_structure.read_bytes() == src_before and linked.read_bytes() == src_before
    assert cu_structure.stat().st_ino == ino == linked.stat().st_ino and cu_structure.stat().st_nlink == 2


def test_second_bootstrap_into_the_same_work_dir_is_refused_without_regenerating(bootstrapped, fake_d, cu_structure):
    before = _snapshot(bootstrapped)
    with pytest.raises(SystemExit, match=r"is not empty \(acceptance.json|is not empty \(bootstrap.log"):
        _boot(fake_d, cu_structure, bootstrapped, "--target-ps", "9.9")
    assert _snapshot(bootstrapped) == before  # same bytes AND same inodes: nothing regenerated
    assert "9.9" not in (bootstrapped / "config.yaml").read_text()


@pytest.mark.parametrize("name", ["config.yaml", "acceptance.json", "run_distill.sh", "PLAN.md", "omm_teacher.py",
                                  "bootstrap.log", ".distill"])
def test_preexisting_proposal_output_is_refused_before_any_write(tmp_path, fake_d, cu_structure, name):
    work = tmp_path / "w"
    work.mkdir()
    if name == ".distill":
        (work / name).mkdir()
        (work / name / "attempt.json").write_text("{}")
    else:
        (work / name).write_text("operator's own content\n")
    before = _snapshot(work)
    with pytest.raises(SystemExit, match="is not empty .*Nothing was written"):
        _boot(fake_d, cu_structure, work, "--acceptance", "--mode", "fixture", "--energy-mae-max", "5",
              "--force-mae-max", "50", "--target-ps", "0.5", "--max-iter", "1", "--no-progress-limit", "1")
    assert _snapshot(work) == before


def test_heldout_file_inside_the_work_dir_is_refused_untouched(tmp_path, fake_d, cu_structure):
    from ase.io import write as ase_write
    work = tmp_path / "w"
    work.mkdir()
    held = work / "heldout.extxyz"
    at = read(cu_structure)
    at.rattle(0.2, seed=7)
    ase_write(held, [at], format="extxyz")
    before = _snapshot(work)
    with pytest.raises(SystemExit, match=r"--heldout-file .* lives inside it"):
        _boot(fake_d, cu_structure, work, "--acceptance", "--mode", "fixture", "--energy-mae-max", "5",
              "--force-mae-max", "50", "--target-ps", "0.5", "--max-iter", "1", "--no-progress-limit", "1",
              "--heldout-file", str(held))
    assert _snapshot(work) == before


def test_work_that_is_a_file_is_refused(tmp_path, fake_d, cu_structure):
    work = tmp_path / "w"
    work.write_text("not a directory\n")
    with pytest.raises(SystemExit, match="exists and is not a directory; nothing was written"):
        _boot(fake_d, cu_structure, work)
    assert work.read_text() == "not a directory\n"


def test_empty_existing_work_dir_is_accepted_and_every_output_is_a_fresh_regular_file(tmp_path, fake_d, cu_structure):
    work = tmp_path / "w"
    work.mkdir()
    assert _boot(fake_d, cu_structure, work) == 0
    for name in ("bootstrap.log", "structure_init.vasp", "omm_teacher.py", "config.yaml", "run_distill.sh"):
        p = work / name
        assert p.is_file() and not p.is_symlink()
    assert os.access(work / "run_distill.sh", os.X_OK)
    assert (work / "run_distill.sh").stat().st_mode & 0o777 == 0o755 & ~_umask()
    assert set(os.listdir(work)) <= set(db.BOOTSTRAP_OUTPUTS)


def _umask() -> int:
    cur = os.umask(0)
    os.umask(cur)
    return cur


def test_acceptance_and_plan_disclose_the_work_dir_policy(accepted):
    acc = json.loads((accepted / "acceptance.json").read_text())
    assert acc["provenance"]["work_dir_policy"] == db.WORK_DIR_POLICY
    assert "nothing preexisting is overwritten, regenerated or deleted" in db.WORK_DIR_POLICY
    assert "new or empty" in db.WORK_DIR_POLICY and "O_EXCL" in db.WORK_DIR_POLICY
    assert "Work directory ownership: --work must be new or empty" in (accepted / "PLAN.md").read_text()
    assert "held open for the whole bootstrap" in db.WORK_DIR_POLICY


# ── bootstrap.log: created exclusively once, then written through the retained stream ──
#
# The reviewer's residual: bootstrap.log was the one output opened by name
# without O_EXCL, so an object planted there between the emptiness check and
# a write was appended to (hardlink) or blocked the open (FIFO). Both windows
# are modelled deterministically below by planting the object from inside a
# wrapped bootstrap step -- the harness, never the bootstrap, does the planting.

def _plant(kind: str, at: Path, external: Path) -> None:
    if kind == "hardlink":
        os.link(external, at)
    elif kind == "symlink":
        at.symlink_to(external)
    elif kind == "dangling":
        at.symlink_to(at.parent / "points-nowhere")
    elif kind == "fifo":
        os.mkfifo(at)
    else:  # pragma: no cover
        raise ValueError(kind)


def _external_collector(tmp_path: Path) -> Path:
    external = tmp_path / "external" / "collector.log"
    external.parent.mkdir()
    external.write_bytes(b"external bytes that must survive\n")
    return external


@pytest.mark.parametrize("kind", ["hardlink", "symlink", "dangling", "fifo"])
def test_object_planted_at_bootstrap_log_after_the_gate_passes_is_refused_not_written_or_opened(
        tmp_path, fake_d, cu_structure, monkeypatch, kind):
    # check->first-write window: refuse_unowned_work passes on the empty dir,
    # then something appears at bootstrap.log before the exclusive create
    external = _external_collector(tmp_path)
    work = tmp_path / "w"
    real_gate = db.refuse_unowned_work

    def gate_then_plant(w, **kw):
        real_gate(w, **kw)
        w.mkdir(parents=True, exist_ok=True)
        _plant(kind, w / "bootstrap.log", external)
    monkeypatch.setattr(db, "refuse_unowned_work", gate_then_plant)
    t0 = time.monotonic()
    with pytest.raises(SystemExit, match="bootstrap.log already exists; bootstrap creates its outputs exclusively"):
        _boot(fake_d, cu_structure, work)
    assert time.monotonic() - t0 < 10  # a FIFO fails the exclusive create at once; it is never opened
    assert external.read_bytes() == b"external bytes that must survive\n"
    assert sorted(os.listdir(work)) == ["bootstrap.log"]  # the planted object; nothing else was written
    planted = work / "bootstrap.log"
    if kind == "hardlink":
        assert planted.stat().st_ino == external.stat().st_ino and external.stat().st_nlink == 2
    elif kind == "fifo":
        assert stat.S_ISFIFO(planted.lstat().st_mode)
    else:
        assert planted.is_symlink()  # not replaced by a regular file, not followed


@pytest.mark.parametrize("kind", ["hardlink", "symlink", "fifo"])
def test_bootstrap_log_replaced_between_two_log_lines_is_never_written_to_or_opened(
        tmp_path, fake_d, cu_structure, monkeypatch, kind):
    # between-call window: the log already exists and later lines must go to
    # the descriptor created by this invocation, not to whatever now sits at
    # the pathname -- the harness moves the real log aside and plants an object
    external = _external_collector(tmp_path)
    work = tmp_path / "w"
    moved = tmp_path / "moved-aside.log"
    real_stage = db.stage_structure

    def stage_then_replace(atoms, w):
        out = real_stage(atoms, w)
        os.rename(w / "bootstrap.log", moved)
        _plant(kind, w / "bootstrap.log", external)
        return out
    monkeypatch.setattr(db, "stage_structure", stage_then_replace)
    t0 = time.monotonic()
    assert _boot(fake_d, cu_structure, work) == 0
    assert time.monotonic() - t0 < 30  # a FIFO at the name is never opened, so nothing blocks
    text = moved.read_text()
    assert "step 1/4" in text and "step 4/4" in text and text.rstrip("\n").endswith(db.run_command(work))
    assert external.read_bytes() == b"external bytes that must survive\n"  # the hardlink/symlink target: untouched
    planted = work / "bootstrap.log"
    if kind == "hardlink":
        assert planted.stat().st_ino == external.stat().st_ino and external.stat().st_nlink == 2
    elif kind == "fifo":
        assert stat.S_ISFIFO(planted.lstat().st_mode)
    else:
        assert planted.is_symlink()
    # the planted object is left where it is: nothing under --work is deleted
    for name in ("structure_init.vasp", "omm_teacher.py", "config.yaml", "run_distill.sh"):
        assert (work / name).is_file() and not (work / name).is_symlink()


def test_bootstrap_log_is_created_once_exclusively_written_repeatedly_and_closed_on_exit(tmp_path, capsys):
    work = tmp_path / "w"
    work.mkdir()
    with db.BootstrapLog(work) as log:
        db._log(log, "one")
        db._log(log, "two")
        db._log(None, "stdout only")
        assert not log._stream.closed
        # normal repeated logging: the same stream, flushed after every line
        assert (work / "bootstrap.log").read_text() == "[distill_bootstrap] one\n[distill_bootstrap] two\n"
    assert log._stream.closed
    assert capsys.readouterr().out.splitlines() == ["[distill_bootstrap] one", "[distill_bootstrap] two",
                                                     "[distill_bootstrap] stdout only"]
    # a second BootstrapLog at the same name is the refusal -- never a reopen, never an append
    with pytest.raises(SystemExit, match="bootstrap.log already exists"):
        db.BootstrapLog(work)
    assert (work / "bootstrap.log").read_text() == "[distill_bootstrap] one\n[distill_bootstrap] two\n"


def test_bootstrap_log_stream_is_closed_and_retained_when_the_bootstrap_refuses(tmp_path, fake_d, cu_structure,
                                                                                monkeypatch):
    # failure retention still holds with the retained stream: the refusal
    # propagates through the context manager, the stream is closed, the log
    # and the rejected copy stay on disk
    created = []

    class Recording(db.BootstrapLog):
        def __init__(self, w):
            super().__init__(w)
            created.append(self)
    monkeypatch.setattr(db, "BootstrapLog", Recording)
    real_stage = db.stage_structure

    def stage_then_refuse(atoms, w):
        real_stage(atoms, w)
        raise SystemExit("simulated staging refusal")
    monkeypatch.setattr(db, "stage_structure", stage_then_refuse)
    work = tmp_path / "w"
    with pytest.raises(SystemExit, match="simulated staging refusal"):
        _boot(fake_d, cu_structure, work)
    assert len(created) == 1 and created[0]._stream.closed  # one log, created once, closed on the refusal
    assert sorted(os.listdir(work)) == ["bootstrap.log", "structure_init.vasp"]
    assert (work / "bootstrap.log").read_text().startswith("[distill_bootstrap] step 1/4: read structure ")


def test_acceptance_records_the_engines_boundary_contract(accepted):
    acc = json.loads((accepted / "acceptance.json").read_text())
    ef = acc["provenance"]["engine_facts"]
    assert "force_pbc" in ef["labeling_pbc"] and "[T, T, T]" in ef["labeling_pbc"]
    assert ef["student_md_boundary"].startswith("p p f") and "student_md_lammps.py" in ef["student_md_boundary"]
    assert "disclosed, not reconciled" in ef["boundary_note"] and "fully periodic" in ef["input_requirement"]
    assert acc["witness_scope"]["witness_boundary"].startswith("p p f")
    assert acc["budget"]["wallclock_enforcement"]["attribution"] == db.WALLCLOCK_ATTRIBUTION
    assert "unknown which" in acc["budget"]["wallclock_enforcement"]["rc_rule"]
