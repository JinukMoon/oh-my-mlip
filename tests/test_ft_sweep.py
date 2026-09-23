"""Tests for scripts/ft_sweep.py -- the per-variant fine-tune driver.

Pins the five-state vocabulary from executed evidence: fake ft_run /
ft_verify scripts stand in for the real trainers (explicit --ft-run-cmd /
--ft-verify-cmd injection, no monkeypatching); every state is produced by a
fake, and a stub/emitted config never passes (no checkpoint => no pass).
GPU-free.
"""
import importlib.util
import json
import os
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location("ft_sweep", REPO_ROOT / "scripts" / "ft_sweep.py")
fts = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fts)


def _script(path: Path, body: str) -> str:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _home(tmp_path: Path) -> Path:
    (tmp_path / "models.json").write_text(json.dumps({
        "_meta": {},
        "Fam": {"env": "fam", "python": "x", "versions": {
            "F-ok": {"finetune": {"status": "documented", "runnable_as_installed": True}},
            "F-train-fails": {"finetune": {"status": "documented", "runnable_as_installed": True}},
            "F-exit3": {"finetune": {"status": "documented", "runnable_as_installed": False, "blockers": ["pip install z"]}},
            "F-no-ckpt": {"finetune": {"status": "documented", "runnable_as_installed": True}},
            "F-reload-fails": {"finetune": {"status": "documented", "runnable_as_installed": True}},
            "F-exit4": {"finetune": {"status": "documented", "runnable_as_installed": True}},
            "F-exit5": {"finetune": {"status": "documented", "runnable_as_installed": True}},
            "F-no-prov": {"finetune": {"status": "documented", "runnable_as_installed": True}},
            "F-no-witness": {"finetune": {"status": "documented", "runnable_as_installed": True}},
            "F-gated": {"gated": True, "finetune": {"status": "documented", "runnable_as_installed": False}},
            "F-unsupported": {"finetune": {"status": "not-supported", "reason": "no train code",
                                           "evidence": ["https://example.org/repo/README.md"]}},
            "F-uncited": {"finetune": {"status": "not-supported", "reason": "no train code", "evidence": []}},
            "F-excavate": {"finetune": {"status": "code-excavation-needed", "reason": "no path"}},
            "F-excavate-audited": {"finetune": {"status": "code-excavation-needed", "reason": "no path"}},
        }},
        "Other": {"env": "other", "python": "x", "versions": {"O-1": {"finetune": {"status": "documented"}}}},
    }))
    return tmp_path


FT_RUN = """
variant="$1"; shift
out=""
while [ $# -gt 0 ]; do case "$1" in --out) out="$2"; shift 2;; *) shift;; esac; done
case "$variant" in
  F-train-fails) echo "CUDA OOM during epoch 1" >&2; exit 1;;
  F-exit3) echo "blocker: pip install z" >&2; exit 3;;
  F-exit4) echo "no real builder for Fam (implementation gap)" >&2; exit 4;;
  F-exit5) echo "seed unhonoured: explicit --seed on a data-split-only family without --allow-partial-seed" >&2; exit 5;;
  F-no-ckpt) printf '#!/bin/sh\\n' > "$out/finetune_$variant.sh"; echo '{"schema": "ft_run.json/1"}' > "$out/ft_run.json"; exit 0;;
  F-no-prov) printf '#!/bin/sh\\n' > "$out/finetune_$variant.sh"; echo "weights" > "$out/$variant.model"; exit 0;;
  *) printf '#!/bin/sh\\n' > "$out/finetune_$variant.sh"; echo "weights" > "$out/$variant.model";
     echo '{"schema": "ft_run.json/1", "rematerialize": ["ft_run.py", "'"$variant"'"]}' > "$out/ft_run.json"; exit 0;;
esac
"""

FT_VERIFY = """
ckpt="$1"
case "$ckpt" in
  *F-reload-fails*) echo '{"pass": false, "reason": "non_finite_energy", "version": "F-reload-fails"}'; exit 1;;
  *F-no-witness*) echo '{"pass": true, "energy_ev": -3.5, "forces_shape": [4, 3], "device": "cuda"}'; exit 0;;
  *) echo '{"pass": true, "energy_ev": -3.5, "forces_shape": [4, 3], "gpu_used": true, "device": "cuda", "reason": "", "version": "F-resolved"}'; exit 0;;
esac
"""

# ft_run rendering its self-labelled generic stub (and leaving a file behind)
FT_RUN_STUB = """
variant="$1"; shift
out=""
while [ $# -gt 0 ]; do case "$1" in --out) out="$2"; shift 2;; *) shift;; esac; done
printf '#!/bin/sh\\n' > "$out/finetune_$variant.sh"
echo '{"_generic_stub": "best-effort ft_run.py rendering -- this family is not modeled"}' > "$out/finetune_config.json"
echo "weights" > "$out/$variant.model"; echo '{"schema": "ft_run.json/1"}' > "$out/ft_run.json"; exit 0
"""

# ft_run that exits 0 with its provenance record but writes NO checkpoint
# (a stale checkpoint from an earlier attempt is already there)
FT_RUN_NOOP = """
variant="$1"; shift
out=""
while [ $# -gt 0 ]; do case "$1" in --out) out="$2"; shift 2;; *) shift;; esac; done
echo '{"schema": "ft_run.json/1"}' > "$out/ft_run.json"; exit 0
"""


def _run(tmp_path, monkeypatch, audit=None, token=True):
    if token:
        monkeypatch.setenv("HF_TOKEN", "hf_fake-token-for-gating-only")
    else:
        for var in ("HF_TOKEN", "HF_TOKEN_PATH", "OMM_HF_TOKEN_FILE"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
    home = _home(tmp_path)
    dataset = tmp_path / "demo.traj"
    dataset.write_text("")
    ledger = tmp_path / "ft.jsonl"
    rows = fts.sweep("fam", home, ledger, dataset, tmp_path / "out", audit=audit or {},
                     campaign_id="c1", manifest_sha256="abc123", epochs=1, device="cuda",
                     ft_run_cmd=[_script(tmp_path / "fake_ft_run.sh", FT_RUN)],
                     ft_verify_cmd=[_script(tmp_path / "fake_ft_verify.sh", FT_VERIFY)])
    lines = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    return {r["variant"]: r for r in rows}, lines


def test_every_state_from_executed_evidence(tmp_path, monkeypatch):
    audit = {"F-excavate-audited": {"supported": False,
                                    "citation": "docs/finetune.md:432 -- no checkpoint-consuming train path"}}
    states, lines = _run(tmp_path, monkeypatch, audit=audit)
    got = {v: r["state"] for v, r in states.items()}
    assert got == {
        "F-ok": "passed",
        "F-train-fails": "failed(train)",
        "F-exit3": "failed(not_runnable_as_installed)",
        "F-no-ckpt": "failed(no_checkpoint)",
        "F-reload-fails": "failed(reload)",
        "F-exit4": "failed(no_builder)",  # ft_run rc 4: implementation gap, not a training failure
        "F-exit5": "failed(seed_unhonoured)",  # ft_run rc 5: refusal before any write, not a training failure
        "F-no-prov": "failed(no_provenance_record)",  # trained, but wrote no ft_run.json: never a pass
        "F-no-witness": "failed(ft_verify:no_gpu_witness)",  # cuda verify without a measured gpu_used
        "F-gated": "passed",  # token present => gate is open; the fake trains it
        "F-unsupported": "unsupported",
        "F-uncited": "failed(uncited_unsupported)",
        "F-excavate": "failed(audit_missing)",
        "F-excavate-audited": "unsupported",
    }
    # the passed row carries the real chain's evidence: rerun .sh, ckpt, verdict, provenance record
    ev = states["F-ok"]["evidence"]
    assert ev["sh"].endswith("finetune_F-ok.sh") and ev["ckpt"].endswith("F-ok.model")
    assert ev["verdict"]["forces_shape"] == [4, 3]
    assert ev["builder"] == "real" and ev["gpu_witness"] is True and ev["device"] == "cuda"
    assert len(ev["ckpt_sha256"]) == 64 and ev["ckpt_bytes"] > 0
    assert ev["ft_run_json"].endswith("F-ok/ft_run.json") and len(ev["ft_run_json_sha256"]) == 64
    assert ev["ft_run_json_schema"] == "ft_run.json/1"
    # ft_verify's reason/version are recorded VERBATIM (None when not reported)
    assert ev["ft_verify_reason"] == "" and ev["ft_verify_version"] == "F-resolved"
    rf = states["F-reload-fails"]
    assert rf["evidence"]["ft_verify_reason"] == "non_finite_energy" and rf["evidence"]["ft_verify_version"] == "F-reload-fails"
    assert rf["stderr_tail"] == "verifier_failed (ft_verify reason: non_finite_energy)"
    assert states["F-no-witness"]["evidence"]["ft_verify_reason"] is None
    assert states["F-no-witness"]["evidence"]["ft_verify_version"] is None
    # rc 5 is a refusal: no artifacts claimed, nothing written, the sweep passed no --seed
    e5 = states["F-exit5"]
    assert e5["returncode"] == 5 and "seed unhonoured" in e5["stderr_tail"] and e5["evidence"]["sh"] is None
    assert "--seed" not in e5["evidence"]["command"] and "--allow-partial-seed" not in e5["evidence"]["command"]
    assert not any((tmp_path / "out" / "F-exit5").iterdir())
    # the provenance-less train step is refused BEFORE checkpoint discovery
    np_ = states["F-no-prov"]
    assert "wrote no ft_run.json" in np_["stderr_tail"] and "ckpt" not in np_["evidence"] and np_["returncode"] == 0
    # the verifier is handed the VARIANT key, so variant-specific reload details resolve
    verify_row = next(r for r in lines if r["variant"] == "F-ok" and r["phase"] == "verify")
    assert verify_row["evidence"]["verdict"]["pass"] is True
    plan = lines[0]
    assert plan["phase"] == "plan" and plan["unit"] == "GiB" and plan["ft_verify_loaders_source"].startswith("injected")
    # unsupported rows carry their citation; the audited one cites the audit
    assert states["F-unsupported"]["evidence"]["citation"] == "https://example.org/repo/README.md"
    assert "docs/finetune.md:432" in states["F-excavate-audited"]["evidence"]["citation"]
    # every row carries the campaign identity and the manifest hash
    assert all(r["manifest_sha256"] == "abc123" and r["campaign_id"] == "c1" for r in lines)
    assert all(r["kind"] == "finetune" for r in lines)
    # the other env's variant was never touched
    assert "O-1" not in states
    # stderr of the failed train is preserved
    assert "CUDA OOM" in states["F-train-fails"]["stderr_tail"]
    assert states["F-exit4"]["returncode"] == 4 and "no real builder" in states["F-exit4"]["stderr_tail"]
    # rc 4 is the "no real builder" contract: the row carries no train
    # artifacts (no rerun script, no checkpoint) and nothing was emitted
    ev4 = states["F-exit4"]["evidence"]
    assert ev4["sh"] is None and "ckpt" not in ev4 and "builder" not in ev4
    assert not any((tmp_path / "out" / "F-exit4").iterdir())
    # the unwitnessed cuda row records the witness as absent (None), never as a pass
    nw = states["F-no-witness"]
    assert nw["evidence"]["gpu_witness"] is None and nw["evidence"]["device"] == "cuda"
    assert nw["stderr_tail"] == "no_gpu_witness"


FT_RUN_BUNDLE = """
variant="$1"; shift
out=""
while [ $# -gt 0 ]; do case "$1" in --out) out="$2"; shift 2;; *) shift;; esac; done
printf '#!/bin/sh\\n' > "$out/finetune_$variant.sh"
mkdir -p "$out/seed/0/final_model/variables"
echo pb > "$out/seed/0/final_model/saved_model.pb"
echo idx > "$out/seed/0/final_model/variables/variables.index"
echo data > "$out/seed/0/final_model/variables/variables.data-00000-of-00001"
echo '{"schema": "ft_run.json/1"}' > "$out/ft_run.json"
exit 0
"""


def test_checkpoint_globs_come_from_the_shipped_ft_run_and_bundles_are_hashed_whole(tmp_path, monkeypatch):
    """Discovery rules are the SHIPPED ft_run.py's FAMILY_CHECKPOINT_GLOBS
    (parsed, never imported), recorded in the plan row with their source;
    the local table is only the fallback and says so. A SavedModel witness
    (saved_model.pb) is hashed with every file of its bundle directory."""
    monkeypatch.setenv("HF_TOKEN", "hf_fake-token-for-gating-only")
    home = _home(tmp_path)
    (home / "scripts").mkdir()
    (home / "scripts" / "ft_run.py").write_text(
        "import sys  # noqa\n"
        'FAMILY_CHECKPOINT_GLOBS = {\n'
        '    "MACE": ["*.model"],\n'
        '    "Fam": ["seed/*/final_model/saved_model.pb"],\n'
        '}\n'
        'def main():\n    raise SystemExit("never imported by ft_sweep")\n')
    table = fts.family_checkpoint_globs(home)
    assert table["source"] == f"{home / 'scripts' / 'ft_run.py'}:FAMILY_CHECKPOINT_GLOBS"
    assert table["globs"] == {"MACE": ["{version}.model", "*.model"], "Fam": ["seed/*/final_model/saved_model.pb"]}
    dataset = tmp_path / "demo.traj"
    dataset.write_text("")
    ledger = tmp_path / "ft.jsonl"
    rows = fts.sweep("fam", home, ledger, dataset, tmp_path / "out", audit={}, campaign_id="c1",
                     manifest_sha256="abc123", epochs=1, device="cuda",
                     ft_run_cmd=[_script(tmp_path / "fake_ft_run.sh", FT_RUN_BUNDLE)],
                     ft_verify_cmd=[_script(tmp_path / "fake_ft_verify.sh", FT_VERIFY)])
    lines = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    plan = lines[0]
    assert plan["ckpt_globs_source"].endswith("ft_run.py:FAMILY_CHECKPOINT_GLOBS")
    assert plan["ckpt_globs"]["Fam"] == ["seed/*/final_model/saved_model.pb"]
    ok = next(r for r in rows if r["variant"] == "F-ok")
    assert ok["state"] == "passed" and ok["evidence"]["ckpt"].endswith("seed/0/final_model/saved_model.pb")
    bundle = ok["evidence"]["ckpt_bundle"]
    assert bundle["file_count"] == 3 and bundle["dir"].endswith("seed/0/final_model")
    assert {Path(f["path"]).name for f in bundle["files"]} == {"saved_model.pb", "variables.index",
                                                              "variables.data-00000-of-00001"}
    assert all(len(f["sha256"]) == 64 for f in bundle["files"]) and bundle["bytes"] == sum(f["bytes"] for f in bundle["files"])
    # fallback: a shipped ft_run.py without the table, or no ft_run.py at all
    (home / "scripts" / "ft_run.py").write_text("X = 1\n")
    fb = fts.family_checkpoint_globs(home)
    assert fb["globs"] == fts.CKPT_GLOBS and fb["source"].startswith("ft_sweep.CKPT_GLOBS")
    assert fts.family_checkpoint_globs(tmp_path / "nowhere")["source"].startswith("ft_sweep.CKPT_GLOBS")
    # a malformed table is not trusted
    (home / "scripts" / "ft_run.py").write_text('FAMILY_CHECKPOINT_GLOBS = {"MACE": "*.model"}\n')
    assert fts.family_checkpoint_globs(home)["source"].startswith("ft_sweep.CKPT_GLOBS")


# Expected checkpoint per family under a correctly shaped <out> (the FT
# owner's report, 2026-09-14, matching the SHIPPED ft_run.FAMILY_CHECKPOINT_GLOBS).
EXPECTED_CKPT = {
    "MACE": "MACE-OMAT-0.model", "SevenNet": "checkpoint_best.pth", "DeePMD": "model.ckpt-100.pt",
    "DPA4": "model.ckpt-100.pt", "GRACE": "seed/1/final_model/saved_model.pb", "PET": "model-ft.pt",
    "NequIP": "checkpoints/last.ckpt", "Allegro": "checkpoints/last.ckpt",
    "MatterSim": "results/best_model.pth", "TACE": "checkpoints_epoch/last.ckpt",
    "CHGNet": "chgnet_ft/bestE_epoch10_e1_f1.pth.tar",
    "UMA": "runs/ft/checkpoints/final/inference_ckpt.pt",
    "fairchemv1": "runs/checkpoints/ft/checkpoint.pt",
    "EquFlash": "runs/checkpoints/checkpoint_last.pt",
    "Nequix": "wandb/offline-run-20260915_120000-abc/files/checkpoint.nqx",
    "ORB": "ckpts/checkpoint_epoch49.ckpt",
    "EquiformerV3": "runs/checkpoints/ft/checkpoint.pt",
}
# Files that ALSO exist in the same tree and must not be chosen
DECOYS = {
    "NequIP": ["checkpoints/best.ckpt"], "Allegro": ["checkpoints/best.ckpt"],
    "MatterSim": ["results/last_model.pth"], "TACE": ["checkpoints_epoch/TACE-3.ckpt"],
    "CHGNet": ["chgnet_ft/epoch5_e2_f2.pth.tar"], "GRACE": ["seed/1/final_model/variables/variables.index",
                                                          "seed/1/final_model/variables/variables.data-00000-of-00001"],
}


def _synthetic_out(root: Path, family: str, version: str) -> Path:
    out = root / family
    for rel in [EXPECTED_CKPT[family].replace("MACE-OMAT-0", version)] + DECOYS.get(family, []):
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rel)
    if family == "GRACE":
        (out / "seed/1/final_model/assets").mkdir()  # legal, empty
    return out


def test_checkpoint_matrix_against_the_shipped_ft_run_table_single_source(tmp_path):
    """B1 regression (ft review): discovery runs against the SHIPPED
    ft_run.py's FAMILY_CHECKPOINT_GLOBS -- one table, every BUILDERS family
    -- and finds exactly the expected artifact for all eleven families on
    synthetic trees (touch/mkdir only: no env, no download, no training).
    The generic glob list is pinned as the negative: a family without a
    table entry finds NOTHING for GRACE and CHGNet (the reviewer's
    reproduction of the pre-fix fallback), so a "just use a generic glob"
    refactor re-breaks visibly. The local fallback table must EQUAL the
    shipped table whenever ft_run.py is readable (staleness pin)."""
    import importlib.util as _ilu
    import sys as _sys
    spec = _ilu.spec_from_file_location("ft_run_under_test", REPO_ROOT / "scripts" / "ft_run.py")
    ft_run = _ilu.module_from_spec(spec)
    _sys.modules["ft_run_under_test"] = ft_run  # dataclass string annotations look the module up
    spec.loader.exec_module(ft_run)
    table = fts.family_checkpoint_globs(REPO_ROOT)
    globs = table["globs"]
    # single source: the table ft_sweep resolves against IS ft_run's (the
    # version-named MACE preference in front only when ft_run lacks it), over
    # exactly the builder families
    assert table["source"] == f"{REPO_ROOT / 'scripts' / 'ft_run.py'}:FAMILY_CHECKPOINT_GLOBS"
    assert set(globs) == set(ft_run.FAMILY_CHECKPOINT_GLOBS) == set(ft_run.BUILDERS)
    for fam, rules in ft_run.FAMILY_CHECKPOINT_GLOBS.items():
        assert globs[fam] == [g for g in fts.CKPT_GLOB_PREFIX.get(fam, []) if g not in rules] + rules, fam
    assert set(EXPECTED_CKPT) == set(ft_run.BUILDERS)
    version = "MACE-OMAT-0"
    for fam, rel in EXPECTED_CKPT.items():
        out = _synthetic_out(tmp_path / "trees", fam, version)
        found = fts.find_checkpoint(out, fam, version, globs=globs)
        assert found is not None and found == out / rel, (fam, found)
    # the generic list (a family with no table entry) is the pinned negative
    assert fts.find_checkpoint(tmp_path / "trees" / "GRACE", "GRACE", version, globs={}) is None
    assert fts.find_checkpoint(tmp_path / "trees" / "CHGNet", "CHGNet", version, globs={}) is None
    # staleness pin: the local fallback IS the shipped table, entry for entry (a
    # drift of the frozen FT table fails here, not silently behind "*.pt")
    assert fts.CKPT_GLOBS == ft_run.FAMILY_CHECKPOINT_GLOBS
    for fam, rel in EXPECTED_CKPT.items():
        assert fts.find_checkpoint(tmp_path / "trees" / fam, fam, version) == tmp_path / "trees" / fam / rel, fam
    # CHGNet suffix: *.pth does not match the .pth.tar artifact, ft_run's pattern does
    chg = tmp_path / "trees" / "CHGNet"
    assert list(chg.rglob("*.pth")) == [] and [p.name for p in chg.rglob("chgnet_ft/bestE_*.pth.tar")] == ["bestE_epoch10_e1_f1.pth.tar"]
    # determinism (M4): first matching glob wins, never mtime order -- best.ckpt
    # newer than last.ckpt, then the other way round, same answer
    for fam in ("NequIP", "Allegro"):
        out = tmp_path / "trees" / fam
        last, best = out / "checkpoints" / "last.ckpt", out / "checkpoints" / "best.ckpt"
        for newer, older in ((best, last), (last, best)):
            os.utime(older, (1_700_000_000, 1_700_000_000))
            os.utime(newer, (1_700_000_100, 1_700_000_100))
            assert fts.find_checkpoint(out, fam, version, globs=globs) == last, (fam, newer.name)
    # the GRACE SavedModel is a directory-shaped artifact: its whole tree is the
    # bundle (files hashed, directories -- the empty assets/ too -- listed)
    grace = tmp_path / "trees" / "GRACE"
    ckpt = fts.find_checkpoint(grace, "GRACE", version, globs=globs)
    assert ckpt.name in fts.CKPT_BUNDLE_MARKERS
    bundle_files = sorted(p.relative_to(ckpt.parent).as_posix() for p in ckpt.parent.rglob("*") if p.is_file())
    assert bundle_files == ["saved_model.pb", "variables/variables.data-00000-of-00001", "variables/variables.index"]
    assert (ckpt.parent / "assets").is_dir()


def test_shipped_ft_run_glob_table_agrees_with_the_fine_tune_owner_report():
    """Integration pin against the REAL scripts/ft_run.py: the exported table
    covers every family the FT owner reported (2026-09-14) with the reported
    first-choice glob, and GRACE's witness is the SavedModel .pb (whose whole
    bundle is preserved). Glob CORRECTNESS against real trainers is the FT
    owner's integration gate, not asserted here."""
    table = fts.family_checkpoint_globs(REPO_ROOT)
    assert table["source"] == f"{REPO_ROOT / 'scripts' / 'ft_run.py'}:FAMILY_CHECKPOINT_GLOBS"
    globs = table["globs"]
    first = {
        "MACE": "{version}.model", "SevenNet": "checkpoint_best.pth", "DeePMD": "model.ckpt-*.pt",
        "DPA4": "model.ckpt-*.pt", "GRACE": "seed/*/final_model/saved_model.pb", "PET": "model-ft.pt",
        "NequIP": "checkpoints/last.ckpt", "Allegro": "checkpoints/last.ckpt",
        "MatterSim": "results/best_model.pth", "TACE": "checkpoints_epoch/last.ckpt",
        "CHGNet": "chgnet_ft/bestE_*.pth.tar",
    }
    assert set(first) <= set(globs), sorted(set(first) - set(globs))
    assert {fam: globs[fam][0] for fam in first} == first
    assert Path(first["GRACE"]).name in fts.CKPT_BUNDLE_MARKERS


def test_gated_without_token_is_access_blocked(tmp_path, monkeypatch):
    """M8: without a token SOURCE (presence only, contents never read) a
    gated variant is recorded as access prerequisite missing/unverified --
    nothing is fetched, so no remote denial is observed or claimed."""
    states, _ = _run(tmp_path, monkeypatch, token=False)
    row = states["F-gated"]
    assert row["state"] == "access-blocked"
    assert row["evidence"]["reason"].startswith("access prerequisite missing/unverified")
    assert "no remote denial observed" in row["evidence"]["reason"]
    assert row["evidence"]["access"] == {"prerequisite": "hf_token_source", "status": "missing/unverified",
                                         "checked": "presence_only", "remote_denial_observed": False, "fetched": False}
    assert not (tmp_path / "out" / "F-gated").exists()  # nothing attempted, nothing fetched
    assert states["F-ok"]["state"] == "passed"


def test_fallback_table_finds_the_real_deepmd_checkpoint_never_the_symlink_or_a_foundation_decoy(tmp_path):
    """Degraded path (shipped table unreadable): the DeePMD/DPA4 fallback
    entry is model.ckpt-*.pt -- the real files -- so it cannot pick the
    `model.ckpt.pt` SYMLINK (rejected by the discovery filter) nor a
    foundation `.pt` lying in the tree, and with only those present it
    finds nothing rather than the decoy."""
    for fam in ("DeePMD", "DPA4"):
        out = tmp_path / fam
        out.mkdir()
        (out / "DPA-3.1-3M.pt").write_text("foundation decoy")  # the pretrained model ft_run points at
        (out / "model.ckpt-50.pt").write_text("step 50")
        (out / "model.ckpt-100.pt").write_text("step 100")
        os.symlink(out / "model.ckpt-100.pt", out / "model.ckpt.pt")
        os.utime(out / "model.ckpt-50.pt", (1_700_000_000, 1_700_000_000))
        os.utime(out / "model.ckpt-100.pt", (1_700_000_100, 1_700_000_100))
        os.utime(out / "DPA-3.1-3M.pt", (1_700_000_200, 1_700_000_200))  # newest, still never chosen
        found = fts.find_checkpoint(out, fam, "V")  # globs=None: the fallback table
        assert found == out / "model.ckpt-100.pt", (fam, found)
        assert not found.is_symlink()
        # only the symlink and the decoy: nothing, not the decoy
        (out / "model.ckpt-50.pt").unlink()
        (out / "model.ckpt-100.pt").unlink()
        assert fts.find_checkpoint(out, fam, "V") is None, fam
    assert fts.CKPT_GLOBS["DeePMD"] == fts.CKPT_GLOBS["DPA4"] == ["model.ckpt-*.pt"]
    # the fallback never declares the shipped table's source
    assert fts.family_checkpoint_globs(tmp_path / "nowhere")["source"].startswith("ft_sweep.CKPT_GLOBS")


def test_audit_found_supported_is_attempted(tmp_path, monkeypatch):
    audit = {"F-excavate": {"supported": True, "citation": "upstream/train.py:10"}}
    states, _ = _run(tmp_path, monkeypatch, audit=audit)
    assert states["F-excavate"]["state"] == "passed"
    assert states["F-excavate"]["evidence"]["citation"] == "upstream/train.py:10"


FAKE_FT_VERIFY_MODULE = '''
"""stand-in for the shipped scripts/ft_verify.py: its loader table is the capability."""
_LOADER_TEMPLATE = {"Fam": "...", "Ghost": "..."}
'''


def test_loader_capability_is_queried_from_the_shipped_ft_verify(tmp_path, monkeypatch):
    """A family that trains but that the SHIPPED ft_verify cannot reload is
    an honest implementation-gap failure, not a pass and not `unsupported`.
    The capability is read from the script itself (loader_families() or its
    _LOADER_TEMPLATE), never from a hard-coded family list."""
    monkeypatch.setenv("HF_TOKEN", "hf_fake-token-for-gating-only")
    home = _home(tmp_path)
    (home / "scripts").mkdir()
    (home / "scripts" / "ft_verify.py").write_text(FAKE_FT_VERIFY_MODULE)
    caps = fts.loader_families(home)
    assert caps["loaders"] == {"Fam", "Ghost"} and caps["source"] == "_LOADER_TEMPLATE"
    # a public loader_families() wins over the private table
    (home / "scripts" / "ft_verify.py").write_text(FAKE_FT_VERIFY_MODULE + "\ndef loader_families():\n    return ['Fam']\n")
    assert fts.loader_families(home) == {"loaders": {"Fam"}, "source": "loader_families()"}
    # --list-loaders --json, once the verifier ships it, is preferred over importing
    (home / "scripts" / "ft_verify.py").write_text(
        'import json, sys\nif "--list-loaders" in sys.argv:\n    print(json.dumps({"loaders": ["Other"], "source": "cli"}))\n')
    assert fts.loader_families(home) == {"loaders": {"Other"}, "source": "cli"}
    dataset = tmp_path / "demo.traj"
    dataset.write_text("")
    # "Fam" trains fine but the shipped verifier (now: only Other) cannot reload it
    rows = fts.sweep("fam", home, tmp_path / "o.jsonl", dataset, tmp_path / "out", audit={},
                     campaign_id="c", manifest_sha256="m",
                     ft_run_cmd=[_script(tmp_path / "ft_run.sh", FT_RUN)], ft_verify_cmd=None)
    ok = next(r for r in rows if r["variant"] == "F-ok")
    assert ok["state"] == "failed(ft_verify:no_loader)" and "loaders: ['Other']" in ok["stderr_tail"]
    plan = json.loads((tmp_path / "o.jsonl").read_text().splitlines()[0])
    assert plan["ft_verify_loaders"] == ["Other"] and plan["ft_verify_loaders_source"] == "cli"
    # no verifier at all => empty capability, everything fails closed
    (home / "scripts" / "ft_verify.py").unlink()
    assert fts.loader_families(home)["loaders"] == set()


def test_generic_stub_rendering_never_passes(tmp_path, monkeypatch):
    """Defensive guard (the upstream stub builder was removed; rc 4 is the
    live contract): a rendering that labels itself `_generic_stub` is
    failed(generic_stub) even though it exited 0 and left a
    checkpoint-looking file behind -- never a pass on a stub's leftovers."""
    monkeypatch.setenv("HF_TOKEN", "hf_fake-token-for-gating-only")
    home = _home(tmp_path)
    dataset = tmp_path / "demo.traj"
    dataset.write_text("")
    rows = fts.sweep("other", home, tmp_path / "s.jsonl", dataset, tmp_path / "out", audit={},
                     campaign_id="c", manifest_sha256="m",
                     ft_run_cmd=[_script(tmp_path / "ft_run_stub.sh", FT_RUN_STUB)],
                     ft_verify_cmd=[_script(tmp_path / "ft_verify.sh", FT_VERIFY)])
    assert rows[0]["state"] == "failed(generic_stub)" and rows[0]["evidence"]["builder"] == "generic_stub"
    assert "ckpt" not in rows[0]["evidence"]


def test_stale_checkpoint_from_an_earlier_attempt_is_not_this_runs_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake-token-for-gating-only")
    home = _home(tmp_path)
    dataset = tmp_path / "demo.traj"
    dataset.write_text("")
    stale = tmp_path / "out" / "O-1" / "O-1.model"
    stale.parent.mkdir(parents=True)
    stale.write_text("left over from yesterday")
    import os
    old = 1_600_000_000
    os.utime(stale, (old, old))
    rows = fts.sweep("other", home, tmp_path / "st.jsonl", dataset, tmp_path / "out", audit={},
                     campaign_id="c", manifest_sha256="m",
                     ft_run_cmd=[_script(tmp_path / "ft_run_noop.sh", FT_RUN_NOOP)],
                     ft_verify_cmd=[_script(tmp_path / "ft_verify.sh", FT_VERIFY)])
    assert rows[0]["state"] == "failed(no_checkpoint)" and "pre-existing files ignored" in rows[0]["stderr_tail"]


def test_witness_contract_requires_finite_energy_forces_and_gpu_agreement():
    ok = {"pass": True, "energy_ev": -1.0, "forces_shape": [4, 3], "gpu_used": True}
    assert fts.witness_ok(ok, "cuda") == (True, None)
    # a cuda witness WITHOUT a measured gpu_used is a failure, never a pass
    absent = {k: v for k, v in ok.items() if k != "gpu_used"}
    assert fts.witness_ok(absent, "cuda") == (False, "no_gpu_witness")
    assert fts.witness_ok(absent, "cpu") == (True, None)
    assert fts.witness_ok(dict(ok, gpu_used=False), "cuda") == (False, "gpu_not_used")
    assert fts.witness_ok(dict(ok, gpu_used="yes"), "cuda") == (False, "gpu_not_used")  # only the literal True
    assert fts.witness_ok(dict(ok, device="cpu"), "cuda") == (False, "gpu_not_used")
    assert fts.witness_ok(dict(ok, energy_ev=True), "cuda") == (False, "non_finite_energy")
    assert fts.witness_ok(dict(ok, energy_ev=float("nan")), "cuda") == (False, "non_finite_energy")
    assert fts.witness_ok(dict(ok, forces_shape=None), "cuda") == (False, "forces_missing")
    assert fts.witness_ok({"pass": False}, "cuda") == (False, "verifier_failed")
    assert fts.witness_ok(None, "cpu") == (False, "verifier_failed")


def test_disk_floor_is_resource_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake-token-for-gating-only")
    home = _home(tmp_path)
    dataset = tmp_path / "demo.traj"
    dataset.write_text("")
    rows = fts.sweep("fam", home, tmp_path / "d.jsonl", dataset, tmp_path / "out", audit={},
                     campaign_id="c", manifest_sha256="m", min_free_gib=10**9,
                     ft_run_cmd=[_script(tmp_path / "ft_run.sh", FT_RUN)],
                     ft_verify_cmd=[_script(tmp_path / "ft_verify.sh", FT_VERIFY)])
    attempted = [r for r in rows if r["phase"] == "skipped_disk"]
    assert attempted and all(r["state"] == "resource-blocked" for r in attempted)
    assert attempted[0]["evidence"]["unit"] == "GiB" and "GiB floor" in attempted[0]["stderr_tail"]
    # classification-only rows (unsupported etc.) are still recorded, not disk-blocked
    assert {r["state"] for r in rows if r["variant"] == "F-unsupported"} == {"unsupported"}


def test_classify_is_pure_and_distinguishes_blocked_from_unsupported():
    ft_doc = {"gated": True, "finetune": {"status": "documented"}}
    assert fts.classify("V", ft_doc, {}, token_missing=True)["state"] == "access-blocked"
    assert fts.classify("V", ft_doc, {}, token_missing=False)["action"] == "attempt"
    ns = {"finetune": {"status": "not-supported", "evidence": ["u"]}}
    assert fts.classify("V", ns, {}, token_missing=True) == {
        "action": "record", "state": "unsupported", "citation": "u", "reason": "not-supported"}


def test_batch_size_reaches_every_ft_run_call(tmp_path, monkeypatch):
    """ORB's official batch size (100) does not fit a 16 GB GPU; the sweep had
    no way to pass a smaller one except a test-only command override."""
    monkeypatch.setenv("HF_TOKEN", "hf_fake-token-for-gating-only")
    home = _home(tmp_path)
    dataset = tmp_path / "demo.traj"
    dataset.write_text("")
    argv_log = tmp_path / "argv.log"
    recorder = FT_RUN.replace('variant="$1"; shift', f'echo "$@" >> {argv_log}\nvariant="$1"; shift', 1)
    fts.sweep("fam", home, tmp_path / "ft.jsonl", dataset, tmp_path / "out", audit={},
              campaign_id="c1", manifest_sha256=None, epochs=1, device="cuda",
              ft_run_cmd=[_script(tmp_path / "fake_ft_run.sh", recorder)],
              ft_verify_cmd=[_script(tmp_path / "fake_ft_verify.sh", FT_VERIFY)], batch_size=4)
    calls = argv_log.read_text().splitlines()
    assert calls and all(line.endswith("--batch-size 4") for line in calls)
