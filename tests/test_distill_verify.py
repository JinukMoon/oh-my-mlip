"""GPU-free tests for scripts/distill_verify.py (recipes/distill.md §5 oracle).

Every run artifact here is FAKE: a work dir rendered by
scripts/distill_bootstrap.py --acceptance against a fake onthefly-distill
checkout, then populated with hand-written `run/.al_status`,
`al_iter<K>_labeled.extxyz`, `model_scratch<N>.{pt,bin}`, `run_scratch<N>/`
and `heldout/heldout.extxyz`. The teacher-env interpreter is a shell stub
that writes whatever metrics the test dictates; `lmp` is a shell stub that
prints a `Step PotEng` thermo table. No torch, no LAMMPS, no GPU, no
network, nothing outside tmp_path.

What is asserted, per verdict class of the docstring in distill_verify.py:

  passed / unmet(accuracy) / unmet(stability:stalled) / unmet(budget:*) /
  failed(engine_failed|engine_error) / failed(teacher_cannot_relabel) /
  incomplete / failed(heldout_leak) x3 (shared frames, inside run/, same
  seed) / failed(heldout_missing) / failed(heldout_eval) /
  failed(config_drift) / failed(provenance_drift) (judged BEFORE any run
  directory is derived from the live config) / failed(wallclock_evidence)
  / unmet(budget:wallclock) / failed(malformed_evidence) / fixture passed
  vs unmet(fixture:no_relabel) / failed(lmp_witness) for a crashing lmp vs
  incomplete(lmp_witness_skipped) for --no-lmp-witness; the rc 137 rule
  (early = engine_exit, at/after the limit = exhaustion); the artifact-scope
  payload; plus the report JSON, the ledger row, the --json line and exit
  codes.
"""
from __future__ import annotations

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

pytest.importorskip("ase")
from ase.build import bulk
from ase.io import read as ase_read, write as ase_write

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import distill_bootstrap as db  # noqa: E402
import distill_verify as dv  # noqa: E402

TOOLKIT = ["--energy-mae-max", "5", "--force-mae-max", "100"]


# ── fakes ────────────────────────────────────────────────────────────────────
def _write_exec(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture()
def fake_d(tmp_path) -> Path:
    d = tmp_path / "onthefly-distill"
    (d / "scripts").mkdir(parents=True)
    (d / "scripts" / "al_loop_local.sh").write_text("#!/bin/bash\necho fake al_loop_local\n")
    (d / "scripts" / "teacher_md.py").write_text("# fake teacher_md.py\n")
    (d / "config.example.yaml").write_text(textwrap.dedent("""\
        lmp_bin: /path/to/lmp
        python_bin: python
        work_dir: ./run
        system:
          init_structure: x.vasp
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
          host: h
          base: /b
          slurm_partition: gpu
          slurm_gres: "gpu:1"
        """))
    return d


@pytest.fixture()
def fake_python(tmp_path) -> Path:
    """Stands in for the teacher env's interpreter: writes $FAKE_METRICS to
    the --out path (or exits $FAKE_PY_RC without writing)."""
    return _write_exec(tmp_path / "fake_python", """\
        #!/bin/sh
        out=""; prev=""
        for a in "$@"; do
            if [ "$prev" = "--out" ]; then out="$a"; fi
            prev="$a"
        done
        echo "fake heldout_eval: $*"
        if [ "${FAKE_PY_RC:-0}" != "0" ]; then echo "Traceback: fake failure"; exit "$FAKE_PY_RC"; fi
        printf '%s' "$FAKE_METRICS" > "$out"
        """)


@pytest.fixture()
def fake_lmp(tmp_path) -> Path:
    """Stands in for lmp: prints a Step/PotEng table, or $FAKE_LMP_ERROR."""
    return _write_exec(tmp_path / "fake_lmp", """\
        #!/bin/sh
        echo "LAMMPS (fake)"
        echo "args: $*"
        if [ -n "$FAKE_LMP_ERROR" ]; then echo "ERROR: $FAKE_LMP_ERROR"; exit 1; fi
        echo "   Step         PotEng"
        echo "      0  ${FAKE_LMP_PE:--112.345}"
        echo "Loop time of 0.001 on 1 procs for 0 steps"
        """)


def _frames(seed: int, n: int, *, stdev: float = 0.05):
    rng = np.random.default_rng(seed)
    base = bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2)
    out = []
    for _ in range(n):
        a = base.copy()
        a.positions += rng.normal(0, stdev, a.positions.shape)
        a.calc = None
        a.info["energy"] = float(rng.normal(-3.5 * len(a), 0.1))
        a.arrays["forces"] = rng.normal(0, 0.5, a.positions.shape)
        out.append(a)
    return out


@pytest.fixture()
def cu_structure(tmp_path) -> Path:
    path = tmp_path / "Cu32.vasp"
    ase_write(path, bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2), format="vasp")
    return path


class FakeRun:
    """A bootstrapped acceptance work dir plus knobs to fake the engine's run."""

    def __init__(self, work: Path, fake_python: Path, fake_lmp: Path):
        self.work = work
        self.run = work / "run"
        self.fake_python = fake_python
        self.fake_lmp = fake_lmp
        self.acc = json.loads((work / "acceptance.json").read_text())
        self.heldout = Path(self.acc["split"]["heldout"]["path"])

    # engine artifacts ------------------------------------------------------
    def pool(self, n_dataset: int = 6, relabel_rounds: int = 1, n_per_round: int = 3) -> None:
        self.run.mkdir(parents=True, exist_ok=True)
        ase_write(self.run / "dataset.extxyz", _frames(42, n_dataset), format="extxyz")
        for k in range(relabel_rounds):
            ase_write(self.run / f"al_iter{k}_labeled.extxyz", _frames(1000 + k, n_per_round), format="extxyz")

    def models(self, *rounds: int, md: bool = True) -> None:
        self.run.mkdir(parents=True, exist_ok=True)
        for r in rounds:
            (self.run / f"model_scratch{r}.pt").write_bytes(b"fake torch checkpoint")
            (self.run / f"model_scratch{r}.bin").write_bytes(b"fake nnmtp v1 export")
            (self.run / f"train_scratch{r}.log").write_text(f"epoch 1 ...\nbest F_MAE: {80.0 + r} meV/A\n")
            if md:
                d = self.run / f"run_scratch{r}"
                d.mkdir(exist_ok=True)
                (d / "structure_init.data").write_text("fake LAMMPS data\n")
                (d / "md.log").write_text("Step Temp\n0 300\n")
                (d / "failure.json").write_text(json.dumps(
                    {"status": "stable_in_dump", "last_ps": 0.5, "reached_target": True}))

    def status(self, line: str) -> None:
        self.run.mkdir(parents=True, exist_ok=True)
        (self.run / ".al_status").write_text(line + "\n")

    def heldout_frames(self, frames=None) -> None:
        self.heldout.parent.mkdir(parents=True, exist_ok=True)
        ase_write(self.heldout, _frames(4242, 4) if frames is None else frames, format="extxyz")

    def happy(self, *, status: str = "SUCCESS round1 stable", relabel_rounds: int = 1, rounds=(0, 1)) -> "FakeRun":
        self.pool(relabel_rounds=relabel_rounds)
        self.models(*rounds)
        self.status(status)
        self.heldout_frames()
        return self

    # verify invocation -------------------------------------------------------
    def verify(self, monkeypatch, *, e: float = 2.0, f: float = 50.0, py_rc: int = 0,
               lmp_error: str | None = None, extra: list[str] | None = None) -> tuple[int, dict]:
        monkeypatch.setenv("FAKE_METRICS", json.dumps({
            "energy_mae_mev_per_atom": e, "force_mae_mev_per_a": f, "n_frames": 4,
            "device": "cpu", "versions": {"python": "3.x", "torch": "fake", "numpy": np.__version__}}))
        monkeypatch.setenv("FAKE_PY_RC", str(py_rc))
        if lmp_error:
            monkeypatch.setenv("FAKE_LMP_ERROR", lmp_error)
        else:
            monkeypatch.delenv("FAKE_LMP_ERROR", raising=False)
        argv = ["--work", str(self.work), "--python", str(self.fake_python), "--lmp-bin", str(self.fake_lmp),
                "--json", *(extra or [])]
        rc = dv.main(argv)
        report = json.loads((self.work / "verify" / "distill_verify.json").read_text())
        return rc, report


def _bootstrap(tmp_path, fake_d, cu_structure, fake_python, fake_lmp, *, mode: str, extra=()) -> FakeRun:
    work = tmp_path / f"work_{mode}"
    rc = db.main([
        "--teacher", "MACE-MPA-0", "--structure", str(cu_structure), "--work", str(work),
        "--repo", str(fake_d), "--lmp-bin", str(fake_lmp),
        "--acceptance", "--mode", mode, *TOOLKIT, "--target-ps", "0.5",
        "--max-iter", "2", "--no-progress-limit", "2", "--pool-steps", "20", "--pool-save-every", "5",
        *extra,
    ])
    assert rc == 0
    return FakeRun(work.resolve(), fake_python, fake_lmp)


@pytest.fixture()
def prod(tmp_path, fake_d, cu_structure, fake_python, fake_lmp) -> FakeRun:
    return _bootstrap(tmp_path, fake_d, cu_structure, fake_python, fake_lmp, mode="production")


@pytest.fixture()
def fixture_run(tmp_path, fake_d, cu_structure, fake_python, fake_lmp) -> FakeRun:
    return _bootstrap(tmp_path, fake_d, cu_structure, fake_python, fake_lmp, mode="fixture")


# ── production: the four outcomes of a terminal SUCCESS ─────────────────────
def test_production_passed(prod, monkeypatch, capsys):
    prod.happy()
    rc, report = prod.verify(monkeypatch)
    assert rc == 0
    assert (report["state"], report["reason"][:7]) == ("passed", "SUCCESS")
    assert report["accuracy"]["within_thresholds"] is True
    assert report["accuracy"]["energy_mae_mev_per_atom"] == 2.0
    assert report["lmp_witness"]["ok"] is True and report["lmp_witness"]["pe_ev"] == pytest.approx(-112.345)
    assert report["heldout"]["leak"] is False and report["heldout"]["n_overlap"] == 0
    assert report["final_model"]["round"] == 1
    assert report["final_model"]["train_log_best_f_mae_mev_per_a"] == 81.0  # cited, not gating
    # the --json line is the last stdout line and carries the state
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(last)["state"] == "passed"


def test_production_stable_but_inaccurate_is_unmet_accuracy_with_proposal(prod, monkeypatch):
    prod.happy()
    rc, report = prod.verify(monkeypatch, e=2.0, f=250.0)
    assert rc == 1
    assert (report["state"], report["reason"]) == ("unmet", "accuracy")
    assert report["accuracy"]["within_thresholds"] is False
    prop = report["proposal"]
    assert prop["needs_approval"] is True
    flags = " ".join(o["flag"] for o in prop["options"])
    assert "--pool-steps 40" in flags and "--max-iter 4" in flags
    assert "unmet" not in report["reason"]  # state carries unmet, reason the class


def test_production_round0_success_is_legitimate(prod, monkeypatch):
    prod.pool(relabel_rounds=0)
    prod.models(0)
    prod.status("SUCCESS round0 stable")
    prod.heldout_frames()
    rc, report = prod.verify(monkeypatch)
    assert rc == 0 and report["state"] == "passed"
    assert report["relabel"]["n_relabel_rounds"] == 0


def test_production_stalled_is_unmet_stability(prod, monkeypatch):
    prod.happy(status="STOPPED no_progress round1 best=0.12")
    rc, report = prod.verify(monkeypatch)
    assert rc == 1
    assert (report["state"], report["reason"]) == ("unmet", "stability:stalled")
    assert report["proposal"]["needs_approval"] is True
    assert report["stability"]["kind"] == "stalled"


@pytest.mark.parametrize("line,reason", [
    ("STOPPED backstop", "budget:backstop"),
    ("STOPPED label_fail round1", "budget:label_fail"),
])
def test_production_budget_stops_are_unmet(prod, monkeypatch, line, reason):
    prod.happy(status=line)
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("unmet", reason)


# ── engine failures and non-terminal states ─────────────────────────────────
@pytest.mark.parametrize("line,reason", [
    ("FAILED train round1", "engine_failed"),
    ("FAILED md oneshot", "engine_failed"),
    ("ERROR no_dataset", "engine_error"),
])
def test_engine_failed_is_failed(prod, monkeypatch, line, reason):
    prod.happy(status=line)
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", reason)
    assert report["heldout_eval"]["ran"] is False  # nothing evaluated on a failed run


def test_oneshot_branch_is_failed_teacher_cannot_relabel(prod, monkeypatch):
    prod.happy(status="DONE oneshot model_scratch0.bin")
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "teacher_cannot_relabel")
    assert "can_relabel" in report["detail"]


@pytest.mark.parametrize("line", ["round1 training ...", "round0 student-MD 0.3ps", None])
def test_running_or_missing_status_is_incomplete(prod, monkeypatch, line):
    prod.happy()
    if line is None:
        (prod.run / ".al_status").unlink()
    else:
        prod.status(line)
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("incomplete", "loop_not_terminal")
    assert report["stability"]["terminal"] is False


# ── held-out integrity ──────────────────────────────────────────────────────
def test_heldout_sharing_a_pool_frame_is_a_leak(prod, monkeypatch):
    prod.happy()
    leaked = ase_read(str(prod.run / "al_iter0_labeled.extxyz"), index=":")[1]
    prod.heldout_frames(_frames(4242, 3) + [leaked])
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "heldout_leak")
    assert report["heldout"]["n_overlap"] == 1 and report["heldout"]["leak"] is True
    assert report["heldout_eval"]["ran"] is False  # never evaluated on a leaked set


def test_heldout_from_the_trainer_split_would_leak_too(prod, monkeypatch):
    # the trainer's "validation split" is a subset of dataset.extxyz -- copying
    # it out as a held-out set trips the same fingerprint check
    prod.happy()
    val = ase_read(str(prod.run / "dataset.extxyz"), index=":")[:2]
    prod.heldout_frames(val)
    rc, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "heldout_leak")
    assert report["heldout"]["n_overlap"] == 2


def test_heldout_inside_engine_work_dir_is_a_leak(prod, monkeypatch):
    prod.happy()
    inside = prod.run / "heldout.extxyz"
    shutil.copy(prod.heldout, inside)
    prod.acc["split"]["heldout"]["path"] = str(inside)
    (prod.work / "acceptance.json").write_text(json.dumps(prod.acc))
    rc, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "heldout_leak")
    assert report["heldout"]["outside_engine_work_dir"] is False
    assert "inside" in report["detail"]


def test_heldout_seed_equal_to_pool_seed_is_a_leak(prod, monkeypatch):
    prod.happy()
    prod.acc["split"]["heldout"]["seed"] = prod.acc["split"]["pool"]["teacher_md_seed"]
    (prod.work / "acceptance.json").write_text(json.dumps(prod.acc))
    rc, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "heldout_leak")
    assert report["heldout"]["seed_distinct"] is False


def test_heldout_missing_is_failed(prod, monkeypatch):
    prod.pool(); prod.models(0, 1); prod.status("SUCCESS round1 stable")
    rc, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "heldout_missing")


def test_heldout_eval_crash_is_failed(prod, monkeypatch):
    prod.happy()
    rc, report = prod.verify(monkeypatch, py_rc=3)
    assert (report["state"], report["reason"]) == ("failed", "heldout_eval")
    assert report["heldout_eval"]["returncode"] == 3
    assert "Traceback" in (prod.work / "verify" / "heldout_eval.log").read_text()


def test_heldout_eval_runs_under_teacher_env_with_engine_on_pythonpath(prod, monkeypatch):
    prod.happy()
    _, report = prod.verify(monkeypatch)
    ev = report["heldout_eval"]
    assert ev["python"] == str(prod.fake_python)
    assert ev["command"][1] == str(prod.work / "verify" / "heldout_eval.py")
    assert ev["command"][ev["command"].index("--model-pt") + 1] == str(prod.run / "model_scratch1.pt")
    assert ev["command"][ev["command"].index("--specorder") + 1:] == ["Cu"]
    script = (prod.work / "verify" / "heldout_eval.py").read_text()
    assert "from ontheflydistill import common" in script
    assert "common.FMAX_FILTER" in script  # every held-out frame counts
    assert "model_state_dict" in script


# ── contract drift ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("key,value", [("max_iter", 9), ("target_ps", 5.0), ("seed", 7), ("no_progress_limit", 8)])
def test_config_al_loop_edited_past_approval_is_config_drift(prod, monkeypatch, key, value):
    prod.happy()
    cfg = yaml.safe_load((prod.work / "config.yaml").read_text())
    cfg["al_loop"][key] = value
    (prod.work / "config.yaml").write_text(yaml.safe_dump(cfg))
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "config_drift")
    assert key in report["contract"]["al_loop_drift"]
    assert report["contract"]["config_edited_since_bootstrap"] is True
    assert report["heldout_eval"]["ran"] is False


def test_missing_acceptance_json_refuses_to_judge(prod, monkeypatch):
    prod.happy()
    (prod.work / "acceptance.json").unlink()
    with pytest.raises(SystemExit, match="acceptance.json missing"):
        dv.main(["--work", str(prod.work), "--python", str(prod.fake_python), "--lmp-bin", str(prod.fake_lmp)])


# ── fixture mode ────────────────────────────────────────────────────────────
def test_fixture_passed_needs_a_real_relabel_round(fixture_run, monkeypatch):
    fixture_run.happy(relabel_rounds=1, rounds=(0, 1))
    rc, report = fixture_run.verify(monkeypatch, e=50.0, f=900.0)  # far above thresholds: reported, not gating
    assert rc == 0 and report["state"] == "passed"
    assert report["accuracy"]["within_thresholds"] is False
    assert report["relabel"]["n_relabel_rounds"] == 1 and report["relabel"]["retrain_after_relabel"] is True


def test_fixture_round0_success_is_unmet_no_relabel(fixture_run, monkeypatch):
    fixture_run.pool(relabel_rounds=0)
    fixture_run.models(0)
    fixture_run.status("SUCCESS round0 stable")
    fixture_run.heldout_frames()
    rc, report = fixture_run.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("unmet", "fixture:no_relabel")


def test_fixture_empty_relabel_file_does_not_count(fixture_run, monkeypatch):
    fixture_run.happy(relabel_rounds=0, rounds=(0, 1))
    (fixture_run.run / "al_iter0_labeled.extxyz").write_text("")  # engine wrote an empty window
    rc, report = fixture_run.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("unmet", "fixture:no_relabel")
    assert report["relabel"]["n_relabel_rounds"] == 0


def test_fixture_relabel_without_retrain_does_not_count(fixture_run, monkeypatch):
    fixture_run.happy(relabel_rounds=1, rounds=(0,), status="STOPPED backstop")
    rc, report = fixture_run.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("unmet", "fixture:no_relabel")
    assert report["relabel"]["retrain_after_relabel"] is False


def test_fixture_terminal_stop_with_relabel_still_passes(fixture_run, monkeypatch):
    # a fixture exercises the loop; it need not reach SUCCESS
    fixture_run.happy(relabel_rounds=2, rounds=(0, 1, 2), status="STOPPED backstop")
    rc, report = fixture_run.verify(monkeypatch)
    assert rc == 0 and report["state"] == "passed"
    assert report["final_model"]["round"] == 2


# ── lmp witness ─────────────────────────────────────────────────────────────
def test_lmp_witness_crash_is_failed(prod, monkeypatch):
    prod.happy()
    rc, report = prod.verify(monkeypatch, lmp_error="Unknown pair style nnmtp")
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "lmp_witness")
    w = report["lmp_witness"]
    assert w["ran"] is True and w["returncode"] == 1 and w["log_error"] is True and w["ok"] is False
    assert "Unknown pair style" in (prod.work / "verify" / "lmp_witness" / "witness.log").read_text()


def test_lmp_witness_non_finite_pe_is_failed(prod, monkeypatch):
    prod.happy()
    monkeypatch.setenv("FAKE_LMP_PE", "nan")
    rc, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "lmp_witness")
    assert report["lmp_witness"]["ok"] is False


def test_no_lmp_witness_is_incomplete_not_failed_not_passed(prod, monkeypatch):
    # a deliberately skipped witness is missing evidence: the honest state
    prod.happy()
    rc, report = prod.verify(monkeypatch, extra=["--no-lmp-witness"])
    assert rc == 1 and (report["state"], report["reason"]) == ("incomplete", "lmp_witness_skipped")
    assert report["lmp_witness"]["skipped"] is True and report["lmp_witness"]["ran"] is False
    assert "--no-lmp-witness" in report["detail"]
    assert report["artifacts"]["witness_state"] == "skipped"
    row = json.loads((prod.work / ".distill" / "ledger.jsonl").read_text().splitlines()[-1])
    assert row["state"] == "incomplete" and row["returncode"] == 1


def test_no_lmp_witness_does_not_hide_an_earlier_verdict(prod, monkeypatch):
    prod.happy(status="STOPPED backstop")
    _, report = prod.verify(monkeypatch, extra=["--no-lmp-witness"])
    assert (report["state"], report["reason"]) == ("unmet", "budget:backstop")


def test_fixture_no_lmp_witness_is_incomplete_too(fixture_run, monkeypatch):
    fixture_run.happy(relabel_rounds=1, rounds=(0, 1))
    rc, report = fixture_run.verify(monkeypatch, extra=["--no-lmp-witness"])
    assert rc == 1 and (report["state"], report["reason"]) == ("incomplete", "lmp_witness_skipped")


def test_report_and_ledger_carry_the_artifact_scope(prod, monkeypatch):
    # accuracy is measured on the .pt; the witness only proves the .bin runs
    # and stays finite -- the row must say so, machine-readably
    prod.happy()
    _, report = prod.verify(monkeypatch)
    a = report["artifacts"]
    assert a["accuracy_artifact"] == str(prod.run / "model_scratch1.pt")
    assert a["witness_artifact"] == str(prod.run / "model_scratch1.bin")
    assert "NOT accuracy equivalence" in a["witness_scope"] and "finiteness" in a["witness_scope"]
    assert ".pt" in a["accuracy_scope"] and a["witness_state"] == "ok"
    row = json.loads((prod.work / ".distill" / "ledger.jsonl").read_text().splitlines()[-1])
    assert row["verdict"]["artifacts"] == a
    _, report = prod.verify(monkeypatch, lmp_error="boom")
    assert report["artifacts"]["witness_state"] == "failed"


def test_lmp_witness_input_names_final_bin_and_specorder(prod, monkeypatch):
    prod.happy()
    prod.verify(monkeypatch)
    text = (prod.work / "verify" / "lmp_witness" / "witness.in").read_text()
    assert "pair_style      nnmtp" in text
    assert f"pair_coeff      * * {prod.run / 'model_scratch1.bin'} Cu" in text
    assert f"read_data       {prod.run / 'run_scratch1' / 'structure_init.data'}" in text
    assert "run             0" in text
    assert "mass            1 63.546" in text


def test_lmp_witness_needs_structure_init_data(prod, monkeypatch):
    prod.happy()
    (prod.run / "run_scratch1" / "structure_init.data").unlink()
    rc, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "lmp_witness")
    assert "structure_init.data" in report["lmp_witness"]["error"]


# ── report + ledger + CLI surface ───────────────────────────────────────────
def test_report_and_ledger_row_shape(prod, monkeypatch):
    prod.happy()
    rc, report = prod.verify(monkeypatch, extra=["--manifest-sha256", "abc123", "--campaign-id", "camp-1"])
    assert report["schema"] == "oh-my-mlip.distill.verify/1"
    assert report["mode"] == "production" and report["manifest_sha256"] == "abc123"
    assert report["budget"] == prod.acc["budget"]
    assert report["provenance"]["engine"]["repo"] == prod.acc["provenance"]["engine"]["repo"]
    assert report["heldout"]["sha256"] == dv._sha256(prod.heldout)

    rows = [json.loads(l) for l in (prod.work / ".distill" / "ledger.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "distill" and row["phase"] == "verify" and row["state"] == "passed"
    assert row["returncode"] == 0 and row["manifest_sha256"] == "abc123" and row["campaign_id"] == "camp-1"
    assert row["evidence"]["final_bin"] == str(prod.run / "model_scratch1.bin")
    assert row["evidence"]["heldout_sha256"] == report["heldout"]["sha256"]
    assert row["evidence"]["n_relabel_rounds"] == 1
    assert row["teacher"] == row["variant"] == prod.acc["provenance"]["teacher"]["version"]
    assert row["env"] == prod.acc["provenance"]["teacher"]["env"] and row["family"] == "MACE"
    assert row["verdict"]["degraded"] is False and row["stderr_tail"] is None
    assert row["evidence"]["lmp_bin_sha256"] == dv._sha256(prod.fake_lmp) == report["lmp_witness"]["lmp_bin_sha256"]


def test_ledger_row_for_a_non_pass_is_not_passed(prod, monkeypatch):
    prod.happy(status="STOPPED no_progress round1 best=0.2")
    prod.verify(monkeypatch)
    row = json.loads((prod.work / ".distill" / "ledger.jsonl").read_text().splitlines()[-1])
    assert row["state"] == "unmet" and row["verdict"]["reason"] == "stability:stalled" and row["returncode"] == 1
    assert row["stderr_tail"].startswith("stability:stalled:")


def test_ledger_row_reads_as_not_passed_in_evidence_report(prod, monkeypatch):
    # the shared five-state accounting (scripts/evidence_report.py judge_row):
    # anything but "passed" is INCOMPLETE, and a passed row without a manifest
    # is INCOMPLETE(no_manifest) -- the distill row must key and judge there
    import evidence_report as er
    prod.happy(status="STOPPED backstop")
    prod.verify(monkeypatch, extra=["--manifest-sha256", "abc"])
    prod.verify(monkeypatch)  # a second, passed-looking run but without a manifest
    prod.happy(); prod.verify(monkeypatch, extra=["--manifest-sha256", "abc"])
    rows = [json.loads(l) for l in (prod.work / ".distill" / "ledger.jsonl").read_text().splitlines()]
    assert [r["state"] for r in rows] == ["unmet", "unmet", "passed"]
    fresh, _ = er.terminal_rows(rows)
    assert ("distill", "MACE-MPA-0") in fresh
    assert er.judge_row(rows[0], required=True, allow_degraded=False) == (False, "INCOMPLETE unmet")
    assert er.judge_row(dict(rows[2], manifest_sha256=None), required=True, allow_degraded=False)[0] is False
    assert er.judge_row(rows[2], required=True, allow_degraded=False) == (True, "passed")


def test_custom_ledger_path_and_repeat_appends(prod, monkeypatch, tmp_path):
    prod.happy()
    ledger = tmp_path / "campaign.jsonl"
    prod.verify(monkeypatch, extra=["--ledger", str(ledger)])
    prod.verify(monkeypatch, extra=["--ledger", str(ledger)], f=999.0)
    rows = [json.loads(l) for l in ledger.read_text().splitlines()]
    assert [r["state"] for r in rows] == ["passed", "unmet"]
    assert not (prod.work / ".distill").exists()


def test_verify_log_lists_the_seven_steps(prod, monkeypatch):
    prod.happy()
    prod.verify(monkeypatch)
    log = (prod.work / "verify" / "verify.log").read_text()
    for step in ("step 1/7", "step 2/7", "step 3/7", "step 4/7", "step 5/7", "step 6/7", "step 7/7"):
        assert step in log


def test_cli_subprocess_exit_code_and_json_line(prod, monkeypatch):
    prod.happy()
    env = dict(os.environ, FAKE_METRICS=json.dumps({"energy_mae_mev_per_atom": 1.0, "force_mae_mev_per_a": 10.0,
                                                    "n_frames": 4}), FAKE_PY_RC="0")
    env.pop("FAKE_LMP_ERROR", None)
    proc = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "distill_verify.py"), "--work", str(prod.work),
                           "--python", str(prod.fake_python), "--lmp-bin", str(prod.fake_lmp), "--json"],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    last = json.loads(proc.stdout.strip().splitlines()[-1])
    assert last["state"] == "passed" and last["report"] == str(prod.work / "verify" / "distill_verify.json")


def test_bootstrap_verify_command_matches_cli(prod):
    cmd = prod.acc["commands"]["verify"]
    assert cmd == f"python3 scripts/distill_verify.py --work {prod.work} --json"
    assert dv.parse_args(["--work", str(prod.work), "--json"]).json is True


# ── wall-clock marker (written by run_distill.sh's supervised launch) ───────
def _with_wallclock(run: FakeRun, marker_state: str | None, *, hours: float = 1.0, attempt: bool = True,
                    marker_edit: dict | None = None, attempt_id: str = "20260913T000000Z-4242") -> None:
    """Approve a bound and fake what run_distill.sh's supervisor writes: a start
    record and (unless marker_state is None) an outcome marker bound to it --
    same attempt id/start, approved limit/grace, the approved script's sha256,
    and a window around the engine's .al_status write time. `marker_edit`
    tampers with the marker afterwards (fabrication tests)."""
    run.acc["budget"]["wallclock_max_h"] = hours
    run.acc["budget"]["wallclock_enforcement"] = db.wallclock_plan(hours, "/usr/bin/timeout", run.work)
    (run.work / "acceptance.json").write_text(json.dumps(run.acc))
    limit_s = int(hours * 3600)
    status = run.run / ".al_status"
    now = int(status.stat().st_mtime) if status.is_file() else int(time.time())
    script_sha = run.acc["provenance"]["generated_files"]["run_distill.sh"]["sha256"]
    d = run.work / ".distill"
    d.mkdir(exist_ok=True)
    rc = {"exhausted": 124, "within": 0, "engine_exit": 2, None: None}[marker_state]
    elapsed = limit_s + 1 if marker_state == "exhausted" else 12
    started = now - elapsed + 1
    if attempt:
        (d / "attempt.json").write_text(json.dumps({
            "schema": db.ATTEMPT_SCHEMA, "attempt": attempt_id, "started_utc": "x", "started_epoch_s": started,
            "limit_s": limit_s, "grace_s": 60, "run_script": str(run.work / "run_distill.sh"),
            "run_script_sha256": script_sha, "supervisor_pid": 1, "covers": db.WALLCLOCK_COVERS}))
    if marker_state is not None:
        marker = {"schema": db.WALLCLOCK_SCHEMA, "attempt": attempt_id, "state": marker_state, "returncode": rc,
                  "limit_h": hours, "limit_s": limit_s, "grace_s": 60, "started_utc": "x", "started_epoch_s": started,
                  "ended_utc": "y", "ended_epoch_s": started + elapsed, "elapsed_s": elapsed,
                  "run_script_sha256": script_sha, "covers": db.WALLCLOCK_COVERS}
        marker.update(marker_edit or {})
        # the supervisor's attribution for the FINAL rc/elapsed (an edit may
        # set its own to fake a disagreeing token)
        marker.setdefault("attribution", dv.wallclock_attribution(marker["returncode"], marker["elapsed_s"], limit_s)
                          if isinstance(marker.get("returncode"), int) else None)
        (d / "wallclock.json").write_text(json.dumps(marker))


def test_wallclock_exhausted_beats_a_running_status(prod, monkeypatch):
    prod.happy(status="round1 training")
    _with_wallclock(prod, "exhausted")
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("unmet", "budget:wallclock")
    assert report["heldout_eval"]["ran"] is False
    assert report["proposal"]["options"][0] == {"change": "longer wall-clock bound", "flag": "--wallclock-max-h 2.0"}


def test_wallclock_exhausted_beats_even_a_success_line(prod, monkeypatch):
    # a SUCCESS written in the same second the group was signalled is not a pass
    prod.happy()
    _with_wallclock(prod, "exhausted")
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("unmet", "budget:wallclock")
    assert "not a pass under an exhausted budget" in report["detail"]
    row = json.loads((prod.work / ".distill" / "ledger.jsonl").read_text().splitlines()[-1])
    assert row["state"] == "unmet" and row["evidence"]["wallclock"] == "exhausted"


def test_wallclock_within_does_not_change_the_verdict(prod, monkeypatch):
    prod.happy()
    _with_wallclock(prod, "within")
    rc, report = prod.verify(monkeypatch)
    assert rc == 0 and report["state"] == "passed"
    assert report["wallclock"] == {**report["wallclock"], "enforced": True, "state": "within", "exhausted": False}


def test_wallclock_marker_absent_while_running_is_incomplete(prod, monkeypatch):
    # the one legitimate no-marker case: the attempt started and is still going
    prod.happy(status="round0 training")
    _with_wallclock(prod, None)
    rc, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("incomplete", "loop_not_terminal")
    assert report["wallclock"] == {**report["wallclock"], "enforced": True, "state": None, "pending": True, "valid": True}


# ── wall-clock evidence is demanded, not merely read (G1/G2/G3 + B3) ─────────
def test_wallclock_marker_absent_with_a_terminal_status_cannot_pass(prod, monkeypatch):
    prod.happy()
    _with_wallclock(prod, None)  # attempt started, SUCCESS on disk, no outcome marker
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "wallclock_evidence")
    assert "no outcome marker" in report["detail"] and report["heldout_eval"]["ran"] is False


def test_wallclock_without_a_start_record_cannot_pass(prod, monkeypatch):
    # SUCCESS reached by running al_loop_local.sh directly: no supervisor, no bound
    prod.happy()
    _with_wallclock(prod, None, attempt=False)
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "wallclock_evidence")
    assert "not launched through run_distill.sh" in report["detail"]
    row = json.loads((prod.work / ".distill" / "ledger.jsonl").read_text().splitlines()[-1])
    assert row["state"] == "failed" and row["evidence"]["wallclock_valid"] is False


@pytest.mark.parametrize("text", ["", "{not json", "[]", '{"schema": "oh-my-mlip.distill.wallclock/1", "state": "within"}'])
def test_wallclock_corrupt_or_old_schema_marker_cannot_pass(prod, monkeypatch, text):
    prod.happy()
    _with_wallclock(prod, "within")
    (prod.work / ".distill" / "wallclock.json").write_text(text)
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "wallclock_evidence")


@pytest.mark.parametrize("edit,needle", [
    ({"attempt": "20260101T000000Z-1"}, "stale marker"),                 # marker from another attempt
    ({"limit_s": 7200}, "limit/grace"),                                  # not the approved bound
    ({"run_script_sha256": "0" * 64}, "not the approved run_distill.sh"),  # a different script ran
    ({"returncode": 124}, "says within but rc=124"),                     # state/rc inconsistent
    ({"elapsed_s": 99999}, "!= ended - started"),                        # timestamps inconsistent
    ({"started_epoch_s": 1, "ended_epoch_s": 13}, "outside this attempt's window"),  # .al_status not this attempt's
])
def test_wallclock_fabricated_within_marker_cannot_pass(prod, monkeypatch, edit, needle):
    prod.happy()
    _with_wallclock(prod, "within", marker_edit=edit)
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "wallclock_evidence"), report["detail"]
    assert needle in report["detail"]


def test_wallclock_stale_marker_over_a_newer_status_cannot_pass(prod, monkeypatch):
    # attempt 1 ended `within` an hour ago; the engine's status was (re)written
    # since, by something the supervisor did not bound
    prod.happy()
    _with_wallclock(prod, "within")
    status = prod.run / ".al_status"
    status.write_text("SUCCESS round1 stable\n")
    os.utime(status, (time.time() + 600, time.time() + 600))
    rc, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "wallclock_evidence")
    assert "outside this attempt's window" in report["detail"]


def test_wallclock_exhausted_marker_that_is_unbound_is_failed_not_unmet(prod, monkeypatch):
    prod.happy()
    _with_wallclock(prod, "exhausted", marker_edit={"attempt": "other"})
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "wallclock_evidence")
    assert report["wallclock"]["exhausted"] is False and report["proposal"] is None


def test_wallclock_stated_without_an_enforcement_plan_cannot_pass(prod, monkeypatch):
    prod.happy()
    prod.acc["budget"]["wallclock_max_h"] = 2.0  # hand-edited proposal: bound claimed, nothing enforces it
    (prod.work / "acceptance.json").write_text(json.dumps(prod.acc))
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "wallclock_evidence")


# ── provenance identity (G6 + B3): every recorded sha256 must still hold ─────
def _touch(path: Path) -> None:
    path.write_text(path.read_text() + "\n# edited after approval\n")


@pytest.mark.parametrize("what", ["run_distill.sh", "omm_teacher.py", "structure", "engine:scripts/al_loop_local.sh",
                                  "engine:scripts/teacher_md.py"])
def test_post_approval_edit_of_any_recorded_file_is_provenance_drift(prod, monkeypatch, what):
    prod.happy()
    p = prod.acc["provenance"]
    if what == "structure":  # a different (still readable) structure under the approved path
        moved = ase_read(p["structure"]["path"])
        moved.rattle(0.1, seed=9)
        ase_write(p["structure"]["path"], moved, format="vasp")
    else:
        _touch(Path(p["engine"]["repo"]) / what.split(":", 1)[1] if what.startswith("engine:") else prod.work / what)
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "provenance_drift")
    assert list(report["contract"]["provenance_drift"]) == [what]
    assert report["contract"]["provenance_drift"][what]["recorded"] != report["contract"]["provenance_drift"][what]["actual"]
    assert report["heldout_eval"]["ran"] is False
    row = json.loads((prod.work / ".distill" / "ledger.jsonl").read_text().splitlines()[-1])
    assert row["evidence"]["provenance_drift"] == [what]


def test_config_edit_outside_al_loop_is_provenance_drift(prod, monkeypatch):
    # G6: teacher/system/student edits that leave al_loop.* intact used to pass
    prod.happy()
    cfg = yaml.safe_load((prod.work / "config.yaml").read_text())
    cfg["teacher"]["calculator"] = "somewhere_else:make_calc"
    (prod.work / "config.yaml").write_text(yaml.safe_dump(cfg))
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "provenance_drift")
    assert list(report["contract"]["provenance_drift"]) == ["config.yaml"]
    assert report["contract"]["config_edited_since_bootstrap"] is True and report["contract"]["al_loop_drift"] == {}


def test_al_loop_drift_is_reported_as_config_drift_before_provenance(prod, monkeypatch):
    prod.happy()
    cfg = yaml.safe_load((prod.work / "config.yaml").read_text())
    cfg["al_loop"]["max_iter"] = 99
    (prod.work / "config.yaml").write_text(yaml.safe_dump(cfg))
    _touch(prod.work / "omm_teacher.py")
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "config_drift")
    assert set(report["contract"]["provenance_drift"]) == {"config.yaml", "omm_teacher.py"}


def test_missing_recorded_file_is_provenance_drift(prod, monkeypatch):
    prod.happy()
    (prod.work / "omm_teacher.py").unlink()
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "provenance_drift")
    assert report["contract"]["provenance_drift"]["omm_teacher.py"]["actual"] is None


def test_provenance_drift_outranks_wallclock_evidence(prod, monkeypatch):
    prod.happy()
    _with_wallclock(prod, "exhausted")
    _touch(prod.work / "run_distill.sh")
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "provenance_drift")


# ── B1: teacher_md.py's pre-MD frame 0 is the init structure, for every seed ──
def _teacher_md_like(init, seed: int, n: int):
    """What D/scripts/teacher_md.py actually writes: save_frame() once before
    dyn.run() (the init structure, seed-independent), then seeded MD frames."""
    frames = [init.copy()]
    frames[0].info["energy"] = -3.5 * len(init)
    frames[0].arrays["forces"] = np.zeros(init.positions.shape)
    return frames + _frames(seed, n)


def test_heldout_keeping_teacher_md_frame0_leaks_the_pool_start(prod, monkeypatch, cu_structure):
    init = ase_read(cu_structure)
    prod.pool()
    ase_write(prod.run / "dataset.extxyz", _teacher_md_like(init, db.POOL_TEACHER_SEED, 5), format="extxyz")
    prod.models(0, 1)
    prod.status("SUCCESS round1 stable")
    prod.heldout_frames(_teacher_md_like(init, db.DEFAULT_HELDOUT_SEED, 4))  # frame 0 not dropped
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "heldout_leak")
    assert report["heldout"]["contains_init_frame"] is True and report["heldout"]["n_overlap"] == 1
    assert "init structure" in report["detail"]


def test_heldout_with_frame0_dropped_passes(prod, monkeypatch, cu_structure):
    init = ase_read(cu_structure)
    prod.pool()
    ase_write(prod.run / "dataset.extxyz", _teacher_md_like(init, db.POOL_TEACHER_SEED, 5), format="extxyz")
    prod.models(0, 1)
    prod.status("SUCCESS round1 stable")
    prod.heldout_frames(_teacher_md_like(init, db.DEFAULT_HELDOUT_SEED, 4)[1:])  # what run_distill.sh keeps
    rc, report = prod.verify(monkeypatch)
    assert rc == 0 and report["state"] == "passed"
    assert report["heldout"]["contains_init_frame"] is False and report["heldout"]["n_overlap"] == 0


def test_heldout_init_frame_is_a_leak_even_before_the_pool_file_exists(prod, monkeypatch, cu_structure):
    # no dataset.extxyz on disk to intersect with: the init structure itself is the evidence
    init = ase_read(cu_structure)
    prod.models(0)
    prod.status("SUCCESS round0 stable")
    prod.heldout_frames([init] + _frames(4242, 3))
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "heldout_leak")
    assert report["heldout"]["contains_init_frame"] is True and report["heldout"]["n_pool_frames"] == 0


def test_heldout_with_zero_frames_is_missing(prod, monkeypatch):
    prod.happy()
    prod.heldout.write_text("\n")
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "heldout_missing")


def test_config_drift_still_outranks_wallclock(prod, monkeypatch):
    prod.happy()
    _with_wallclock(prod, "exhausted")
    cfg = yaml.safe_load((prod.work / "config.yaml").read_text())
    cfg["al_loop"]["max_iter"] = 99
    (prod.work / "config.yaml").write_text(yaml.safe_dump(cfg))
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "config_drift")


# ── drift stops the verifier BEFORE any run directory is derived (G6) ────────
def test_drift_is_judged_before_the_run_dir_is_inspected(prod, monkeypatch):
    prod.happy()
    _touch(prod.work / "omm_teacher.py")
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "provenance_drift")
    assert report["stability"]["kind"] == "not_inspected" and report["stability"]["al_status"] is None
    assert report["final_model"]["pt"] is None and report["heldout"]["exists"] is False
    assert report["wallclock"]["valid"] is False and "not inspected" in report["wallclock"]["problems"][0]
    log = (prod.work / "verify" / "verify.log").read_text()
    assert "step 2/7" not in log and "step 3/7" not in log and "step 4/7" not in log
    assert "stopping before any run directory" in log


def test_config_work_dir_redirect_is_drift_and_the_other_dir_is_never_read(prod, tmp_path, monkeypatch):
    # G6 worst case: config.yaml repointed at a previous, finished run's directory
    prod.happy()
    other = tmp_path / "other_finished_run"
    shutil.copytree(prod.run, other)
    (other / ".al_status").write_text("SUCCESS round1 stable\n")
    shutil.rmtree(prod.run)  # THIS attempt produced nothing
    cfg = yaml.safe_load((prod.work / "config.yaml").read_text())
    cfg["work_dir"] = str(other)
    (prod.work / "config.yaml").write_text(yaml.safe_dump(cfg))
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "provenance_drift")
    assert list(report["contract"]["provenance_drift"]) == ["config.yaml"]
    assert report["stability"]["al_status"] is None and report["heldout_eval"]["ran"] is False
    assert str(other) not in (prod.work / "verify" / "verify.log").read_text()


# ── foreign al_iter*_labeled files: a verdict, never a crash ─────────────────
@pytest.mark.parametrize("name", ["al_iterX_labeled.extxyz", "al_iter_labeled.extxyz", "al_iter1b_labeled.extxyz"])
def test_digitless_relabel_file_is_malformed_evidence_not_a_crash(prod, monkeypatch, name):
    prod.happy()
    ase_write(prod.run / name, _frames(77, 2), format="extxyz")  # the trainer's glob would train on this
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "malformed_evidence")
    assert report["relabel"]["malformed_files"] == [str(prod.run / name)]
    assert report["relabel"]["n_relabel_rounds"] == 1  # the genuine al_iter0 still counted
    assert name in report["detail"] and report["heldout_eval"]["ran"] is False
    row = json.loads((prod.work / ".distill" / "ledger.jsonl").read_text().splitlines()[-1])
    assert row["evidence"]["malformed_files"] == [str(prod.run / name)]


def test_relabel_rounds_sorts_numerically_not_lexically(prod, monkeypatch):
    prod.happy(relabel_rounds=0)
    for k in (2, 10, 1):
        ase_write(prod.run / f"al_iter{k}_labeled.extxyz", _frames(100 + k, 1), format="extxyz")
    _, report = prod.verify(monkeypatch)
    assert [Path(f["path"]).name for f in report["relabel"]["labeled_files"]] == [
        "al_iter1_labeled.extxyz", "al_iter2_labeled.extxyz", "al_iter10_labeled.extxyz"]
    assert report["relabel"]["malformed_files"] == [] and report["state"] == "passed"


# ── rc 137: early = SIGKILL of the engine (engine_exit), at/after the limit = expiry ──
def test_marker_engine_exit_rc137_before_the_limit_is_valid_evidence_and_fails_the_run(prod, monkeypatch):
    # 12 s into a 3600 s bound: a genuine engine_exit marker (accepted as
    # evidence, sigkill_before_limit) -- and the run it describes did not
    # complete, so SUCCESS on disk is not this attempt's pass
    prod.happy()
    _with_wallclock(prod, "engine_exit", marker_edit={"returncode": 137})
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "supervisor_exit")
    assert report["wallclock"] == {**report["wallclock"], "valid": True, "state": "engine_exit", "exhausted": False,
                                   "returncode": 137, "attribution": "sigkill_before_limit"}
    assert "source unknown" in report["wallclock"]["attribution_note"]


def test_marker_exhausted_rc137_before_the_limit_is_rejected(prod, monkeypatch):
    # a SIGKILLed engine relabelled as budget exhaustion: not accepted
    prod.happy()
    _with_wallclock(prod, "exhausted", marker_edit={"returncode": 137, "elapsed_s": 12})
    marker = json.loads((prod.work / ".distill" / "wallclock.json").read_text())
    marker["ended_epoch_s"] = marker["started_epoch_s"] + 12
    (prod.work / ".distill" / "wallclock.json").write_text(json.dumps(marker))
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "wallclock_evidence")
    assert "SIGKILL of the engine (source unknown)" in report["detail"] and report["wallclock"]["exhausted"] is False
    assert "OOM kill of the engine" not in report["detail"]  # never a proven source


def test_marker_engine_exit_rc137_at_the_limit_is_rejected(prod, monkeypatch):
    # rc 137 at/after the limit is exhaustion (timeout's KILL after the grace
    # or an external SIGKILL after the deadline -- unknown which); a marker
    # calling it engine_exit is inconsistent
    prod.happy()
    _with_wallclock(prod, "exhausted", marker_edit={"returncode": 137, "state": "engine_exit"})
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "wallclock_evidence")
    assert "unknown which" in report["detail"] and "exhausted, not engine_exit" in report["detail"]


def test_marker_engine_exit_rc124_is_rejected(prod, monkeypatch):
    prod.happy()
    _with_wallclock(prod, "engine_exit", marker_edit={"returncode": 124})
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "wallclock_evidence")
    assert "rc=124 is GNU timeout's expiry code" in report["detail"]


def test_marker_exhausted_rc137_after_the_limit_is_budget_exhaustion_recorded_ambiguous(prod, monkeypatch):
    prod.happy()
    _with_wallclock(prod, "exhausted", marker_edit={"returncode": 137})  # limit_s + 1 s
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("unmet", "budget:wallclock")
    assert report["wallclock"]["attribution"] == "kill_at_or_after_limit"
    # the verdict says so explicitly and asserts no source
    assert "kill_at_or_after_limit" in report["detail"] and "cannot tell which" in report["detail"]
    assert "OOM" not in report["detail"].split("cannot tell which")[0]
    row = json.loads((prod.work / ".distill" / "ledger.jsonl").read_text().splitlines()[-1])
    assert row["evidence"]["wallclock_attribution"] == "kill_at_or_after_limit"


def test_marker_rc137_exactly_at_the_limit_is_exhaustion_not_tightened_to_limit_plus_grace(prod, monkeypatch):
    # the supervisor's rule is elapsed_s >= limit_s in whole seconds; a KILL
    # after the grace measured at exactly the limit (or anywhere below limit +
    # grace) is still exhaustion -- the verifier mirrors it, no tightening
    prod.happy()
    _with_wallclock(prod, "exhausted", marker_edit={"returncode": 137, "elapsed_s": 3600})
    marker = json.loads((prod.work / ".distill" / "wallclock.json").read_text())
    marker["ended_epoch_s"] = marker["started_epoch_s"] + 3600
    (prod.work / ".distill" / "wallclock.json").write_text(json.dumps(marker))
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("unmet", "budget:wallclock")
    assert dv.wallclock_attribution(137, 3600, 3600) == "kill_at_or_after_limit"
    assert dv.wallclock_attribution(137, 3599, 3600) == "sigkill_before_limit"
    assert dv.wallclock_attribution(137, 61, 2) == "kill_at_or_after_limit"  # the measured real timeout shape


@pytest.mark.parametrize("edit,needle", [
    ({"attribution": "timeout_term"}, "!= 'clean' recomputed"),            # token disagrees with rc/elapsed
    ({"attribution": None}, "!= 'clean' recomputed"),                      # token missing
    ({"attribution": "proven_oom"}, "!= 'clean' recomputed"),              # token outside the table
])
def test_marker_attribution_must_match_rc_and_elapsed(prod, monkeypatch, edit, needle):
    prod.happy()
    _with_wallclock(prod, "within", marker_edit=edit)
    rc, report = prod.verify(monkeypatch)
    assert rc == 1 and (report["state"], report["reason"]) == ("failed", "wallclock_evidence")
    assert needle in report["detail"]


# ── F1: a valid engine_exit marker gates BEFORE any evaluation subprocess and before PASS ──
@pytest.mark.parametrize("rc", [1, 137, 143])
@pytest.mark.parametrize("status_line", ["SUCCESS round1 stable", None])
def test_engine_exit_marker_fails_the_run_before_metrics_or_witness(prod, monkeypatch, rc, status_line):
    prod.happy()
    if status_line is None:
        (prod.run / ".al_status").unlink()
    else:
        prod.status(status_line)
    _with_wallclock(prod, "engine_exit", marker_edit={"returncode": rc})  # 12 s into a 3600 s bound
    exit_code, report = prod.verify(monkeypatch)
    assert exit_code == 1 and (report["state"], report["reason"]) == ("failed", "supervisor_exit")
    assert f"rc={rc}" in report["detail"] and report["wallclock"]["valid"] is True
    assert report["wallclock"]["attribution"] == ("sigkill_before_limit" if rc == 137 else "exit_nonzero")
    # no evaluation subprocess was launched: no heldout_eval log, no witness dir
    assert report["heldout_eval"]["ran"] is False and report["lmp_witness"]["ran"] is False
    assert not (prod.work / "verify" / "heldout_eval.log").exists()
    assert not (prod.work / "verify" / "lmp_witness").exists()
    assert report["artifacts"]["witness_state"] == "not_run"
    row = json.loads((prod.work / ".distill" / "ledger.jsonl").read_text().splitlines()[-1])
    assert row["state"] == "failed" and row["returncode"] == 1 and row["evidence"]["wallclock"] == "engine_exit"


def test_engine_exit_yields_to_exhaustion_and_to_drift(prod, monkeypatch):
    # timeout precedence: an exhausted marker is unmet(budget:wallclock), not
    # supervisor_exit; and any drift outranks both
    prod.happy()
    _with_wallclock(prod, "exhausted")
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("unmet", "budget:wallclock")
    _with_wallclock(prod, "engine_exit")
    _touch(prod.work / "omm_teacher.py")
    _, report = prod.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "provenance_drift")
    assert report["heldout_eval"]["ran"] is False


def test_artifacts_disclose_the_witness_boundary(prod, monkeypatch):
    prod.happy()
    _, report = prod.verify(monkeypatch)
    a = report["artifacts"]
    assert a["witness_boundary"] == "p p f" and "student_md_lammps.py" in a["witness_boundary_source"]
    assert "[T, T, T]" in a["accuracy_pbc"] and "force_pbc" in a["accuracy_pbc"]
    assert "disclosed, not reconciled" in a["boundary_note"]
    text = (prod.work / "verify" / "lmp_witness" / "witness.in").read_text()
    assert "boundary        p p f" in text and "student_md_lammps.py" in text


# ── the staged structure is what teacher_md.py read: the init-frame check uses it ──
def test_init_frame_check_uses_the_staged_structure(tmp_path, fake_d, fake_python, fake_lmp, monkeypatch):
    xyz = tmp_path / "cu.xyz"
    init = bulk("Cu", "fcc", a=3.61, cubic=True) * (2, 2, 2)
    ase_write(xyz, init, format="extxyz")
    run = _bootstrap(tmp_path, fake_d, xyz, fake_python, fake_lmp, mode="production")
    staged = run.work / "structure_init.vasp"
    assert run.acc["provenance"]["structure"]["staged_path"] == str(staged)
    run.pool(); run.models(0, 1); run.status("SUCCESS round1 stable")
    run.heldout_frames([ase_read(str(staged), format="vasp")] + _frames(4242, 3))
    _, report = run.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "heldout_leak")
    assert report["heldout"]["init_structure"] == str(staged) and report["heldout"]["contains_init_frame"] is True
    _touch(staged)  # and the staged copy is part of the identity
    _, report = run.verify(monkeypatch)
    assert (report["state"], report["reason"]) == ("failed", "provenance_drift")
    assert list(report["contract"]["provenance_drift"]) == ["structure_init.vasp"]


# ── pure helpers ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("line,kind,rnd", [
    ("SUCCESS round0 stable", "success", 0),
    ("SUCCESS round3 stable (0.50 ps)", "success", 3),
    ("STOPPED no_progress round2 best=0.1", "stalled", 2),
    ("STOPPED backstop", "backstop", None),
    ("STOPPED label_fail round1", "label_fail", 1),
    ("FAILED train round0", "failed", None),
    ("DONE oneshot model_scratch0.bin", "oneshot", None),
    ("ERROR no_dataset", "error", None),
    ("round2 student-MD", "running", None),
])
def test_classify_status_lines(tmp_path, line, kind, rnd):
    (tmp_path / ".al_status").write_text(line + "\n")
    out = dv.classify_status(tmp_path, dv.Log(tmp_path / "v.log"))
    assert (out["kind"], out["round"]) == (kind, rnd)
    assert out["terminal"] is (kind != "running")


def test_frame_fingerprints_survive_rewrite_but_not_real_moves(tmp_path):
    # The rounding exists to make a frame that was copied through a different
    # writer (relabel -> extxyz -> extxyz) hash identically; it is NOT a
    # tolerance for noise (a coordinate near a 1e-4 boundary flips).
    a = _frames(1, 1)[0]
    ase_write(tmp_path / "a.extxyz", [a], format="extxyz")
    rewritten = ase_read(str(tmp_path / "a.extxyz"))  # round-trip through the text format
    rewritten.calc = None
    rewritten.info["energy"] = -1.0  # labels differ, geometry identical -> same frame
    c = a.copy(); c.positions[0, 0] += 0.01  # a real move: different frame
    ase_write(tmp_path / "bc.extxyz", [rewritten, c], format="extxyz")
    fa = dv.frame_fingerprints(tmp_path / "a.extxyz")
    fbc = dv.frame_fingerprints(tmp_path / "bc.extxyz")
    assert len(fa) == 1 and len(fbc) == 2
    assert len(fa & fbc) == 1


def test_parse_pe_reads_first_row_after_header():
    text = "LAMMPS\n   Step         PotEng\n      0  -112.5\n      1  -113\nLoop time"
    assert dv._parse_pe(text) == -112.5
    assert dv._parse_pe("no table here") is None
