"""GPU-free tests for scripts/catbench_vasp_stage.py (temp fixtures only).

The property under test: upstream `vasp_preprocessing` deletes every file it
does not keep, so it must only ever see a COPY. These tests build a small
VASP-shaped tree, stage it, and assert the originals are byte-identical
afterwards, that nothing but CONTCAR/OSZICAR was copied by default, and that
every unsafe destination / symlink / missing-DFT-result case is REFUSED
rather than worked around. No catbench import, no network, no compute.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import catbench_vasp_stage as st  # noqa: E402

OSZ_OK = "       N       E                     dE             d eps       ncg     rms          rms(c)\n" \
         "DAV:   1    -0.1E+03   -0.1E+03  -0.3E+03  1  0.1E+02\n" \
         "   1 F= -.10E+03 E0= -.10000000E+03  d E =-.1E-05\n"
OSZ_NO_E0 = "DAV:   1    -0.1E+03   -0.1E+03  -0.3E+03  1  0.1E+02\n"


def _pair(d: Path, oszicar: str = OSZ_OK, extra: bool = True) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "CONTCAR").write_text(f"contcar of {d.name}\n1.0\n")
    (d / "OSZICAR").write_text(oszicar)
    if extra:
        (d / "INCAR").write_text("ENCUT = 400\n")      # upstream would delete this on the tree it sees
        (d / "OUTCAR").write_text("big output\n" * 50)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    src = tmp_path / "originals"
    _pair(src / "gas" / "H2gas")
    _pair(src / "Pt111" / "slab")
    _pair(src / "Pt111" / "H" / "site_0")
    _pair(src / "Pt111" / "H" / "site_1")
    _pair(src / "Pt111" / "OH" / "site_0")
    (src / "README.txt").write_text("user notes\n")
    return src


COEFF = {"H": {"slab": -1, "adslab": 1, "H2gas": -0.5}, "OH": {"slab": -1, "adslab": 1, "H2gas": -0.5}}


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _listing(root: Path) -> set[str]:
    """Every entry under root, links included, without following them: a
    before/after pair proves a refused call wrote nothing anywhere."""
    out = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for n in list(dirnames) + filenames:
            out.add(os.path.relpath(os.path.join(dirpath, n), root))
    return out


def _outside_dest(root: Path, dest: Path) -> set[str]:
    """Everything under root that is NOT inside the owned destination: a refusal
    may leave the dest exactly as it was, but must never write anywhere else."""
    return {e for e in _listing(root) if e != dest.name and not e.startswith(dest.name + os.sep)}


def _assert_nothing_staged(dest: Path, dataset_name: str = "pt_h") -> None:
    """A refused stage never reaches the upstream unit: `stage` itself executes
    nothing, so the evidence is that its rerunnable unit, its owner marker and
    its temp dir were never created."""
    for name in ("run_stage.sh", "stage_preprocess.py", st.OWNER_MARKER, "originals.sha256",
                 "originals.inventory.json", "stage_record.json", dataset_name, f".{dataset_name}.staging.tmp"):
        assert not (dest / name).exists(), name


# ── scan / proposal ───────────────────────────────────────────────────────────
def test_scan_lists_pairs_half_pairs_and_symlinks_without_following(tree: Path):
    _pair(tree / "Pt111" / "O" / "site_0", extra=False)
    (tree / "Pt111" / "O" / "site_0" / "OSZICAR").unlink()             # half-pair: DFT result missing
    outside = tree.parent / "outside"
    _pair(outside / "Cu111" / "slab")
    os.symlink(outside, tree / "linked_tree")                           # dir symlink
    os.symlink(outside / "Cu111" / "slab" / "CONTCAR", tree / "Pt111" / "CONTCAR_link")
    sc = st.scan(tree)
    dirs = {p["dir"] for p in sc["pairs"]}
    assert dirs == {"gas/H2gas", "Pt111/slab", "Pt111/H/site_0", "Pt111/H/site_1", "Pt111/OH/site_0"}
    assert sc["incomplete"] == [{"dir": "Pt111/O/site_0", "present": ["CONTCAR"], "missing": ["OSZICAR"]}]
    assert sc["symlinks"] == ["Pt111/CONTCAR_link", "linked_tree"]
    assert not any(p["dir"].startswith("linked_tree") for p in sc["pairs"])   # never descended through the link
    assert all(p["oszicar_has_E0"] for p in sc["pairs"])


def test_proposal_reports_missing_slab_and_asks_for_gas_terms(tree: Path):
    _pair(tree / "Au111" / "H" / "site_0")                              # a system with no slab
    prop = st.propose_mapping(st.scan(tree))
    assert set(prop["reactions"]) == {"H", "OH"}
    assert prop["reactions"]["H"]["coeff"] == {"slab": -1, "adslab": 1}
    assert prop["reactions"]["H"]["gas_terms"] == "NEEDS_CONFIRMATION"
    assert prop["reactions"]["H"]["gas_candidates"] == ["H2gas"]
    assert sorted(prop["reactions"]["H"]["systems"]) == ["Au111", "Pt111"]
    assert {"system": "Au111", "issue": "missing_slab"}.items() <= [p for p in prop["problems"] if p.get("system")][0].items()
    assert prop["status"] == "needs_confirmation"
    assert any("confirm gas terms" in q for q in prop["ask"])


def test_proposal_flags_gas_dir_without_gas_suffix_and_oszicar_without_e0(tree: Path):
    _pair(tree / "gas" / "H2")                                          # upstream rejects keys not ending in "gas"
    _pair(tree / "Pt111" / "N" / "site_0", oszicar=OSZ_NO_E0)
    prop = st.propose_mapping(st.scan(tree))
    issues = {(p.get("gas") or p.get("dir")): p["issue"] for p in prop["problems"]}
    assert issues["H2"] == "gas_key_without_gas_suffix"
    assert issues["Pt111/N/site_0"] == "oszicar_without_E0"
    assert prop["reactions"]["N"]["gas_candidates"] == ["H2gas"]        # the malformed one is not a candidate


# ── coefficient validation ───────────────────────────────────────────────────
def test_validate_coeff_rejects_unknown_gas_missing_keys_and_non_finite(tree: Path):
    prop = st.propose_mapping(st.scan(tree))
    assert st.validate_coeff(COEFF, prop) == []
    errs = st.validate_coeff({"H": {"slab": -1, "adslab": 1, "COgas": -1}}, prop)
    assert any("COgas" in e and "no gas/COgas" in e for e in errs)
    errs = st.validate_coeff({"H": {"adslab": 1}}, prop)
    assert any("missing required keys ['slab']" in e for e in errs)
    errs = st.validate_coeff({"H": {"slab": True, "adslab": float("nan"), "H2gas": "x"}}, prop)
    assert sum("not a finite number" in e for e in errs) == 3
    errs = st.validate_coeff({"CO": {"slab": -1, "adslab": 1}}, prop)
    assert any("no scanned reaction directory named 'CO'" in e for e in errs)
    assert st.validate_coeff({}, prop) == ["coeff_setting must be a non-empty dict"]


# ── stage: the copy-only boundary ────────────────────────────────────────────
def test_stage_copies_only_needed_files_and_leaves_originals_untouched(tree: Path, tmp_path: Path):
    before = _snapshot(tree)
    dest = tmp_path / "stage"
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3", catbench_version="1.1.4")
    assert rec["ok"], rec
    assert _snapshot(tree) == before                                    # originals byte-identical
    copied = sorted(str(p.relative_to(dest / "pt_h")) for p in (dest / "pt_h").rglob("*") if p.is_file())
    assert copied == sorted(f"{d}/{f}" for d in ("gas/H2gas", "Pt111/slab", "Pt111/H/site_0", "Pt111/H/site_1", "Pt111/OH/site_0")
                            for f in ("CONTCAR", "OSZICAR"))
    assert not any(p.is_symlink() for p in (dest / "pt_h").rglob("*"))  # real copies, never links
    assert rec["copied_files"] == 10 and rec["hashed_files"] == 10 and rec["copied_subset"] == ["CONTCAR", "OSZICAR"]
    # evidence files
    manifest = st.read_manifest(dest / "originals.sha256")
    assert len(manifest) == 10 and all(k.startswith("./") for k in manifest)
    inv = json.loads((dest / "originals.inventory.json").read_text())
    assert "./README.txt" in inv and "./Pt111/slab/OUTCAR" in inv       # every original inventoried, not just copied ones
    assert json.loads((dest / "coeff_setting.json").read_text()) == COEFF
    assert json.loads((dest / st.OWNER_MARKER).read_text())["source"] == str(tree.resolve())
    # the rerunnable unit
    pre = (dest / "stage_preprocess.py").read_text()
    ast.parse(pre)
    assert "vasp_preprocessing('pt_h', coeff_setting)" in pre
    assert "'1.1.4'" in pre and "sys.exit(3)" in pre                   # version guard
    assert "load_catbench_json" in pre
    sh = (dest / "run_stage.sh").read_text()
    assert sh.splitlines()[0] == "#!/bin/sh" and "set -eu" in sh
    assert f"cd {dest.resolve()}" in sh and str(dest / "stage_preprocess.py") in sh
    assert os.access(dest / "run_stage.sh", os.X_OK)
    if shutil.which("sh"):
        assert subprocess.run(["sh", "-n", str(dest / "run_stage.sh")]).returncode == 0
    if shutil.which("sha256sum"):                                       # manifest is sha256sum -c compatible
        assert subprocess.run(["sha256sum", "-c", "--quiet", str(dest / "originals.sha256")], cwd=tree).returncode == 0


def test_stage_is_deterministic_and_repeatable_into_its_own_destination(tree: Path, tmp_path: Path):
    dest = tmp_path / "stage"
    assert st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")["ok"]
    first = {k: v for k, v in _snapshot(dest).items() if k in ("stage_preprocess.py", "run_stage.sh", "originals.sha256", "coeff_setting.json")}
    assert st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")["ok"]     # owner marker => accepted
    second = {k: v for k, v in _snapshot(dest).items() if k in first}
    assert first == second


def test_restage_after_source_lost_files_is_refused_and_a_new_dest_is_exact(tree: Path, tmp_path: Path):
    """Reviewer defect: a re-stage after files vanished from the source kept the
    stale copies (manifest 6 vs 10 staged files). Now: refused, nothing
    written or removed; a new owned --dest holds exactly the planned set.
    This script never replaces a staging (no --replace-staged exists)."""
    dest = tmp_path / "stage"
    assert st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")["ok"]
    (dest / "raw_data").mkdir(); (dest / "raw_data" / "pt_h_adsorption.json").write_text("{}")   # a previous run's outputs
    (dest / "stage_result.json").write_text("{}")
    shutil.rmtree(tree / "Pt111" / "H" / "site_1"); shutil.rmtree(tree / "Pt111" / "OH")            # source shrinks: 10 -> 6
    before_src, before_dest = _snapshot(tree), _snapshot(dest)
    coeff_h = {"H": {"slab": -1, "adslab": 1, "H2gas": -0.5}}
    rec = st.stage(tree, dest, "pt_h", coeff_h, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "staged_tree_differs"
    assert rec["staged_diff"]["extras"] == ["./Pt111/H/site_1/CONTCAR", "./Pt111/H/site_1/OSZICAR",
                                             "./Pt111/OH/site_0/CONTCAR", "./Pt111/OH/site_0/OSZICAR"]
    assert rec["staged_diff"]["missing"] == [] and "new owned --dest" in rec["detail"] and "never replaces" in rec["detail"]
    assert sorted(Path(p).name for p in rec["previous_outputs"]) == ["raw_data", "stage_result.json"]
    assert _snapshot(dest) == before_dest and _snapshot(tree) == before_src                     # nothing written anywhere
    assert "replace_staged" not in st.stage.__code__.co_varnames                                 # the destructive path is gone
    dest2 = tmp_path / "stage2"
    rec = st.stage(tree, dest2, "pt_h", coeff_h, "/usr/bin/python3")
    assert rec["ok"], rec
    assert rec["copied_files"] == 6 and rec["hashed_files"] == 6 and rec["staged_tree_exact"] is True and rec["reused_identical_staging"] is False
    staged = sorted(str(p.relative_to(dest2 / "pt_h")) for p in (dest2 / "pt_h").rglob("*") if p.is_file())
    assert staged == sorted(f"{d}/{f}" for d in ("gas/H2gas", "Pt111/slab", "Pt111/H/site_0") for f in ("CONTCAR", "OSZICAR"))
    assert len(st.read_manifest(dest2 / "originals.sha256")) == 6 == len(staged)               # manifest == staged set
    assert not (dest2 / ".pt_h.staging.tmp").exists() and _snapshot(tree) == before_src         # originals untouched
    assert st.verify(tree, dest2)["ok"] and _snapshot(dest) == before_dest                      # old dest still untouched
    # an identical re-stage (no diff) is accepted, copies nothing, keeps previous outputs
    (dest2 / "stage_result.json").write_text("{}")
    rec = st.stage(tree, dest2, "pt_h", coeff_h, "/usr/bin/python3")
    assert rec["ok"] and rec["reused_identical_staging"] is True and rec["copied_files"] == 0
    assert rec["previous_outputs_kept"] == [str(dest2 / "stage_result.json")] and (dest2 / "stage_result.json").is_file()


def test_stage_validates_exact_staged_path_set_including_extras(tree: Path, tmp_path: Path):
    dest = tmp_path / "stage"
    assert st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")["ok"]
    (dest / "pt_h" / "Pt111" / "slab" / "stray.txt").write_text("not from the source\n")        # extra in the copy tree
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "staged_tree_differs" and rec["staged_diff"]["extras"] == ["./Pt111/slab/stray.txt"]
    (dest / "pt_h" / "Pt111" / "slab" / "CONTCAR").write_text("edited copy\n")                  # changed copy
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["reason"] == "staged_tree_differs" and rec["staged_diff"]["changed"] == ["./Pt111/slab/CONTCAR"]
    assert (dest / "pt_h" / "Pt111" / "slab" / "stray.txt").is_file()                          # refusal removed nothing
    assert st._diff_trees(st.read_manifest(dest / "originals.sha256"), st._tree_hashes(dest / "pt_h"))["extras"] == ["./Pt111/slab/stray.txt"]


def test_stage_refuses_leftover_tmp_stale_outputs_and_symlink_escapes_under_dest(tree: Path, tmp_path: Path):
    """The paths this script would create/rename under dest are checked before
    anything happens: a pre-existing `.<name>.staging.tmp` is never removed,
    outputs without their copy tree are never overwritten, and a copy tree /
    raw_data / tmp path that is (or passes through) a symlink is refused."""
    dest = tmp_path / "stage"
    assert st.main(["scan", "--source", str(tree), "--out", str(dest)]) == 0                     # owned, empty of stagings
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir(); (elsewhere / "keep.txt").write_text("user data\n")
    # 1. an arbitrary existing tmp dir with our name: refused, untouched
    (dest / ".pt_h.staging.tmp").mkdir(); (dest / ".pt_h.staging.tmp" / "x").write_text("someone's file\n")
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "staging_tmp_exists" and (dest / ".pt_h.staging.tmp" / "x").is_file()
    shutil.rmtree(dest / ".pt_h.staging.tmp")
    # 2. previous outputs without a copy tree: refused, kept
    (dest / "stage_result.json").write_text("{}")
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "stale_outputs_without_copy_tree" and (dest / "stage_result.json").is_file()
    (dest / "stage_result.json").unlink()
    # 3. symlinks under dest at any of the paths this script touches: refused before any write
    for name in ("pt_h", ".pt_h.staging.tmp", "raw_data", "stage_result.json"):
        os.symlink(elsewhere, dest / name)
        before = _snapshot(elsewhere)
        rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
        assert rec["ok"] is False and rec["reason"] == "destination_refused", name
        assert any("symlink" in e and name in e for e in rec["errors"]) and _snapshot(elsewhere) == before, name
        assert not (dest / "originals.sha256").exists()
        (dest / name).unlink()
    # 4. a copy tree under a symlinked parent dir is caught the same way
    assert st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")["ok"]
    assert (elsewhere / "keep.txt").read_text() == "user data\n"


def test_cli_restage_refusal_exits_3_and_replace_flag_does_not_exist(tree: Path, tmp_path: Path, capsys):
    dest = tmp_path / "stage"; coeff = tmp_path / "coeff.json"; coeff.write_text(json.dumps(COEFF))
    base = ["stage", "--source", str(tree), "--dest", str(dest), "--dataset-name", "pt_h", "--coeff", str(coeff), "--python", "/usr/bin/python3"]
    assert st.main(base) == 0; capsys.readouterr()
    shutil.rmtree(tree / "Pt111" / "OH"); coeff.write_text(json.dumps({"H": COEFF["H"]}))
    assert st.main(base) == 3 and json.loads(capsys.readouterr().out)["reason"] == "staged_tree_differs"
    with pytest.raises(SystemExit) as exc:                                                    # no destructive flag is parsed
        st.main(base + ["--replace-staged"])
    assert exc.value.code == 2 and "unrecognized arguments: --replace-staged" in capsys.readouterr().err
    assert len(st.read_manifest(dest / "originals.sha256")) == 10                             # the first staging is intact


def test_stage_all_files_copies_everything(tree: Path, tmp_path: Path):
    rec = st.stage(tree, tmp_path / "stage", "pt_h", COEFF, "/usr/bin/python3", all_files=True)
    assert rec["ok"] and rec["copied_subset"] == "ALL"
    assert (tmp_path / "stage" / "pt_h" / "README.txt").is_file()
    assert (tmp_path / "stage" / "pt_h" / "Pt111" / "slab" / "OUTCAR").is_file()


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", ".hidden", "-x"])
def test_stage_refuses_dataset_names_that_are_not_plain(tree: Path, tmp_path: Path, bad: str):
    rec = st.stage(tree, tmp_path / "stage", bad, COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "dataset_name_invalid"
    assert not (tmp_path / "stage").exists()


def test_stage_refuses_destination_equal_inside_or_containing_source(tree: Path, tmp_path: Path):
    for dest in (tree, tree / "stage", tmp_path):
        rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
        assert rec["ok"] is False and rec["reason"] == "destination_refused", dest
    assert not (tree / "stage").exists()


def test_stage_refuses_non_owned_and_foreign_owned_destinations(tree: Path, tmp_path: Path):
    dest = tmp_path / "stage"
    dest.mkdir(); (dest / "precious.txt").write_text("user file\n")
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "destination_refused"
    assert any("does not own" in e for e in rec["errors"])
    # a marker forged for ANOTHER source is not ownership for this one
    (dest / st.OWNER_MARKER).write_text(json.dumps({"created_by": "scripts/catbench_vasp_stage.py", "source": "/elsewhere"}))
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and any("refusing to mix stagings" in e for e in rec["errors"])
    assert (dest / "precious.txt").read_text() == "user file\n"


def test_stage_refuses_symlinks_inside_and_at_the_root(tree: Path, tmp_path: Path):
    os.symlink(tree / "Pt111" / "slab", tree / "Pt111" / "slab_link")
    rec = st.stage(tree, tmp_path / "stage", "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "symlinks_in_source" and rec["symlinks"] == ["Pt111/slab_link"]
    (tree / "Pt111" / "slab_link").unlink()
    link_root = tmp_path / "src_link"
    os.symlink(tree, link_root)
    rec = st.stage(link_root, tmp_path / "stage", "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "destination_refused"
    assert any("passes through a symlink" in e for e in rec["errors"])
    dest_link = tmp_path / "dest_link"
    os.symlink(tmp_path / "real_dest", dest_link)
    rec = st.stage(tree, dest_link, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and any("destination" in e and "symlink" in e for e in rec["errors"])


def test_stage_refuses_missing_dft_results_it_would_need(tree: Path, tmp_path: Path):
    _pair(tree / "Au111" / "H" / "site_0")                              # H is used, Au111 has no slab
    rec = st.stage(tree, tmp_path / "stage", "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "missing_dft_results"
    assert [p["issue"] for p in rec["problems"]] == ["missing_slab"]
    assert not (tmp_path / "stage").exists()
    shutil.rmtree(tree / "Au111")
    (tree / "Pt111" / "slab" / "OSZICAR").write_text(OSZ_NO_E0)
    rec = st.stage(tree, tmp_path / "stage", "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and [p["issue"] for p in rec["problems"]] == ["oszicar_without_E0"]


def test_stage_reports_but_does_not_block_on_an_unused_incomplete_pair(tree: Path, tmp_path: Path):
    d = tree / "Pt111" / "O" / "site_0"; d.mkdir(parents=True); (d / "CONTCAR").write_text("x\n")
    rec = st.stage(tree, tmp_path / "stage", "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] and [p["issue"] for p in rec["problems"]] == ["incomplete_pair"]


def test_stage_refuses_invalid_coeff_before_touching_the_destination(tree: Path, tmp_path: Path):
    rec = st.stage(tree, tmp_path / "stage", "pt_h", {"H": {"slab": -1, "adslab": 1, "COgas": -1}}, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "coeff_setting_invalid"
    assert not (tmp_path / "stage").exists()


# ── mount boundaries (synthetic mountinfo only; nothing is ever mounted) ─────
_MOUNTINFO_BASE = ["28 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw",
                   "23 28 0:5 / /proc rw,nosuid,nodev,noexec shared:2 - proc proc rw",
                   "25 28 0:22 / /sys rw,nosuid,nodev,noexec shared:3 - sysfs sysfs rw"]


def _mountinfo_with(*points: str) -> str:
    """A mountinfo text naming `points` as mount points, plus the ordinary
    ancestor mounts every machine has."""
    lines = list(_MOUNTINFO_BASE)
    for i, p in enumerate(points):
        esc = p.replace("\\", "\\134").replace(" ", "\\040")
        lines.append(f"{100 + i} 28 8:1 /bind{i} {esc} rw,relatime shared:{10 + i} - ext4 /dev/sda1 rw")
    return "\n".join(lines) + "\n"


@pytest.fixture
def fake_mountinfo(monkeypatch):
    """Replaces the mountinfo reader; a same-device bind mount is exactly what
    these synthetic entries describe (major:minor stays 8:1 throughout)."""
    def _set(*points: str):
        monkeypatch.setattr(st, "read_mountinfo", lambda: _mountinfo_with(*points))
    _set()
    return _set


def test_parse_mountpoints_reads_field_5_past_optional_fields_and_unescapes(fake_mountinfo):
    text = ("28 1 8:1 / / rw - ext4 /dev/sda1 rw\n"
            "36 28 8:1 /a /mnt/my\\040dir rw,relatime shared:2 master:3 - ext4 /dev/sda1 rw\n")
    assert st.parse_mountpoints(text) == ["/", "/mnt/my dir"]
    for bad in ("not a mountinfo line at all\n", "36 35 98:0 / /mnt rw,noatime\n",
                "36 35 98:0 / relative/mount rw - ext4 /dev/x rw\n"):
        with pytest.raises(ValueError):
            st.parse_mountpoints(bad)


def test_stage_refuses_a_same_device_bind_mount_nested_under_the_source(tree: Path, tmp_path: Path, fake_mountinfo):
    """st_dev is identical across a bind mount, so only mountinfo sees this one."""
    mnt = str((tree / "Pt111" / "H").resolve())
    fake_mountinfo(mnt)
    before = _listing(tmp_path)
    dest = tmp_path / "stage"
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "mounted_tree_unsupported"
    assert rec["mounts"] == [mnt] and any(f"holds the mountpoint {mnt}" in e for e in rec["errors"])
    assert all("unsupported by this staging recipe" in e for e in rec["errors"])
    assert not any("invalid" in e or "corrupt" in e for e in rec["errors"])   # never implies the data is bad
    assert not dest.exists() and _listing(tmp_path) == before


def test_stage_refuses_a_bind_mount_nested_under_the_destination(tree: Path, tmp_path: Path, fake_mountinfo):
    dest = tmp_path / "stage"
    assert st.main(["scan", "--source", str(tree), "--out", str(dest)]) == 0
    (dest / "coeff_setting.json").write_text(json.dumps(COEFF))
    fake_mountinfo(str((dest / "pt_h").resolve()))                # a mount where the copy tree would go
    before, before_all = _snapshot(dest), _listing(tmp_path)
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "mounted_tree_unsupported"
    assert rec["mounts"] == [str((dest / "pt_h").resolve())]
    assert _snapshot(dest) == before and _listing(tmp_path) == before_all
    assert not (dest / "run_stage.sh").exists() and not (dest / "originals.sha256").exists()


def test_stage_refuses_when_a_selected_root_is_itself_a_mountpoint(tree: Path, tmp_path: Path, fake_mountinfo):
    dest = tmp_path / "stage"
    for mounted in (str(tree.resolve()), str(dest.resolve())):
        fake_mountinfo(mounted)
        before = _listing(tmp_path)
        rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
        assert rec["ok"] is False and rec["reason"] == "mounted_tree_unsupported", mounted
        assert any("is itself a mountpoint" in e for e in rec["errors"]), mounted
        assert not dest.exists() and _listing(tmp_path) == before, mounted


def test_stage_accepts_ordinary_ancestor_mounts(tree: Path, tmp_path: Path, fake_mountinfo):
    """/ itself and a workspace living on a mounted filesystem are the normal
    case: only a mount AT or UNDER a selected root is refused."""
    fake_mountinfo("/", "/tmp", str(tmp_path.parent), str(tmp_path))
    rec = st.stage(tree, tmp_path / "stage", "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"], rec
    assert rec["copied_files"] == 10


def test_stage_fails_closed_when_mountinfo_is_unreadable_or_malformed(tree: Path, tmp_path: Path, monkeypatch):
    def boom():
        raise OSError(13, "Permission denied")

    dest = tmp_path / "stage"
    for reader in (boom, lambda: "not a mountinfo line at all\n",
                   lambda: "36 35 98:0 / /mnt rw,noatime\n",
                   lambda: "36 35 98:0 / relative/mount rw - ext4 /dev/x rw\n"):
        monkeypatch.setattr(st, "read_mountinfo", reader)
        before = _listing(tmp_path)
        rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
        assert rec["ok"] is False and rec["reason"] == "mountinfo_unavailable", reader
        assert "mountinfo" in rec["detail"] and "refusing" in rec["detail"]
        assert not dest.exists() and _listing(tmp_path) == before


def test_cli_scan_out_is_refused_on_a_nested_mount_and_exits_3(tree: Path, tmp_path: Path, capsys, fake_mountinfo):
    fake_mountinfo(str((tree / "gas").resolve()))
    out = tmp_path / "s1"
    assert st.main(["scan", "--source", str(tree), "--out", str(out)]) == 3
    assert json.loads(capsys.readouterr().out)["reason"] == "mounted_tree_unsupported"
    assert not out.exists()


# ── symlinks nested inside an existing staged copy ───────────────────────────
def test_stage_refuses_a_symlink_nested_inside_an_existing_copy_tree(tree: Path, tmp_path: Path):
    """A link under the staged copy is REFUSED, never skipped: upstream deletes
    inside the tree it is handed, so a link would reach out of the copy."""
    dest = tmp_path / "stage"
    assert st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")["ok"]
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir(); (elsewhere / "keep.txt").write_text("user data\n")
    for link in (dest / "pt_h" / "Pt111" / "linked_dir", dest / "pt_h" / "gas" / "H2gas" / "CONTCAR_link"):
        os.symlink(elsewhere if link.name.endswith("dir") else elsewhere / "keep.txt", link)
        before, before_all = _snapshot(dest), _listing(tmp_path)
        rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
        assert rec["ok"] is False and rec["reason"] == "symlinks_in_staged_tree", link
        assert rec["symlinks"] == [str(link)], link                     # the reason names the link path
        assert _snapshot(dest) == before and _listing(tmp_path) == before_all, link
        assert not (dest / ".pt_h.staging.tmp").exists()
        assert (elsewhere / "keep.txt").read_text() == "user data\n"
        link.unlink()


# ── generated outputs: never written over ────────────────────────────────────
@pytest.mark.parametrize("name", list(st.GENERATED_OUTPUTS))
def test_stage_refuses_a_preexisting_symlink_at_a_generated_output(tree: Path, tmp_path: Path, name: str):
    dest = tmp_path / "stage"
    assert st.main(["scan", "--source", str(tree), "--out", str(dest)]) == 0
    elsewhere = tmp_path / "elsewhere"; elsewhere.mkdir(); (elsewhere / "keep.txt").write_text("user data\n")
    target = dest / name
    if target.exists():
        target.unlink()
    os.symlink(elsewhere / "keep.txt", target)
    before, before_all = _snapshot(elsewhere), _listing(tmp_path)
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "destination_refused", name
    assert any(name in e for e in rec["errors"]), (name, rec["errors"])
    assert _snapshot(elsewhere) == before and _listing(tmp_path) == before_all, name
    # nothing of the rerunnable unit was produced (the planted entry is the test's own)
    assert all(n == name or not (dest / n).exists() for n in ("run_stage.sh", "stage_preprocess.py")), name
    assert not (dest / "pt_h").exists() and not (dest / ".pt_h.staging.tmp").exists()


@pytest.mark.parametrize("name", list(st.GENERATED_OUTPUTS))
def test_stage_refuses_unrelated_preexisting_content_at_a_generated_output(tree: Path, tmp_path: Path, name: str):
    """A file this script did not produce is never written over — not the two
    generated scripts (no generated-by marker), not the manifest, not the JSON."""
    dest = tmp_path / "stage"
    assert st.main(["scan", "--source", str(tree), "--out", str(dest)]) == 0
    target = dest / name
    if target.exists():
        target.unlink()
    target.write_text("the user's own file, not this script's\n")
    before_all = _listing(tmp_path)
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False, name
    # the owner marker is additionally the ownership proof, so it refuses one step earlier
    assert rec["reason"] == ("destination_refused" if name == st.OWNER_MARKER else "unrelated_destination_content"), (name, rec)
    assert any(name in e for e in rec["errors"]), (name, rec["errors"])
    assert target.read_text() == "the user's own file, not this script's\n", name
    assert _listing(tmp_path) == before_all, name
    assert all(n == name or not (dest / n).exists() for n in ("run_stage.sh", "stage_preprocess.py")), name
    assert not (dest / "pt_h").exists() and not (dest / ".pt_h.staging.tmp").exists()


def test_stage_refuses_a_directory_where_a_generated_file_goes(tree: Path, tmp_path: Path):
    dest = tmp_path / "stage"
    assert st.main(["scan", "--source", str(tree), "--out", str(dest)]) == 0
    (dest / "run_stage.sh").mkdir()
    before_all = _listing(tmp_path)
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "unrelated_destination_content"
    assert any("is not a regular file" in e for e in rec["errors"])
    assert _listing(tmp_path) == before_all and (dest / "run_stage.sh").is_dir()


def test_stage_keeps_its_own_previous_outputs_on_an_identical_restage(tree: Path, tmp_path: Path):
    """The ownership check must not fire on this script's own recorded outputs:
    an identical re-stage of an owned dest still reuses the staging and keeps them."""
    dest = tmp_path / "stage"
    assert st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")["ok"]
    (dest / "raw_data").mkdir(); (dest / "raw_data" / "pt_h_adsorption.json").write_text("{}")
    (dest / "stage_result.json").write_text("{}")
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] and rec["reused_identical_staging"] is True and rec["copied_files"] == 0
    assert sorted(Path(p).name for p in rec["previous_outputs_kept"]) == ["raw_data", "stage_result.json"]
    assert (dest / "raw_data" / "pt_h_adsorption.json").is_file()


# ── ownership is a record, not a shape ───────────────────────────────────────
def _plausible(name: str, tree: Path) -> str:
    """Content a user could plausibly have put there: valid JSON, a well-formed
    `sha256sum -c` manifest, scripts carrying the generated-by marker, a marker
    naming this very source. None of it is evidence of who wrote the file."""
    return {
        "coeff_setting.json": json.dumps(COEFF, indent=2) + "\n",
        "mapping_proposal.json": json.dumps({"reactions": {}, "problems": [], "status": "mine"}, indent=2) + "\n",
        "originals.sha256": "0" * 64 + "  ./gas/H2gas/CONTCAR\n",
        "originals.inventory.json": json.dumps({"./README.txt": {"size": 1, "mtime_ns": 1}}, indent=2) + "\n",
        "stage_preprocess.py": st._GEN + "\nprint('my own conversion')\n",
        "run_stage.sh": "#!/bin/sh\n" + st._GEN + "\nset -eu\necho mine\n",
        "stage_record.json": json.dumps({"ok": True, "dest": "somewhere else"}, indent=2) + "\n",
        st.OWNER_MARKER: json.dumps({"created_by": "scripts/catbench_vasp_stage.py",
                                     "source": str(tree.resolve())}, indent=2) + "\n",
    }[name]


@pytest.mark.parametrize("name", list(st.GENERATED_OUTPUTS))
def test_stage_refuses_user_content_that_merely_looks_like_its_own_output(tree: Path, tmp_path: Path, name: str):
    """Shape is not ownership. Valid JSON, a well-formed manifest, a script with
    the generated-by marker, even a hand-written owner marker naming this source:
    without a recorded sha256 for that path none of it may be written over."""
    dest = tmp_path / "stage"
    assert st.main(["scan", "--source", str(tree), "--out", str(dest)]) == 0
    target = dest / name
    if target.exists():
        target.unlink()
    target.write_text(_plausible(name, tree))
    before_all, before_bytes = _listing(tmp_path), target.read_bytes()
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False, name
    # the marker is the ownership evidence itself, so a forged one refuses one step earlier
    assert rec["reason"] == ("destination_refused" if name == st.OWNER_MARKER else "unrelated_destination_content"), (name, rec)
    assert any(name in e for e in rec["errors"]), (name, rec["errors"])
    assert target.read_bytes() == before_bytes and _listing(tmp_path) == before_all, name
    assert all(n == name or not (dest / n).exists() for n in ("run_stage.sh", "stage_preprocess.py")), name
    assert not (dest / "pt_h").exists() and not (dest / ".pt_h.staging.tmp").exists(), name


def test_stage_refuses_an_owner_marker_that_records_no_output_hashes(tree: Path, tmp_path: Path):
    """The recorded hashes are the whole proof: a marker without them owns nothing."""
    dest = tmp_path / "stage"
    assert st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")["ok"]
    marker = dest / st.OWNER_MARKER
    stripped = json.loads(marker.read_text()); stripped.pop("outputs")
    marker.write_text(json.dumps(stripped, indent=2) + "\n")
    before = _snapshot(dest)
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "destination_refused"
    assert any("no recorded output hashes" in e for e in rec["errors"]), rec["errors"]
    assert _snapshot(dest) == before


def test_identical_restage_keeps_recorded_outputs_without_rewriting_them(tree: Path, tmp_path: Path):
    """Proven ours by recorded hash => kept as they are: same inode, same mtime,
    no atomic replace either. The owner record covers every generated output."""
    dest = tmp_path / "stage"
    assert st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")["ok"]
    stamp = {n: (os.stat(dest / n).st_ino, os.stat(dest / n).st_mtime_ns) for n in st.GENERATED_OUTPUTS}
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] and rec["reused_identical_staging"] is True and rec["copied_files"] == 0
    assert {n: (os.stat(dest / n).st_ino, os.stat(dest / n).st_mtime_ns) for n in st.GENERATED_OUTPUTS} == stamp
    recorded = json.loads((dest / st.OWNER_MARKER).read_text())["outputs"]
    assert sorted(recorded) == sorted(st.RECORDED_OUTPUTS)
    assert all(recorded[n] == st.sha256_file(dest / n) for n in st.RECORDED_OUTPUTS)


def test_stage_refuses_an_owned_output_whose_hash_drifted(tree: Path, tmp_path: Path):
    """Ours once, edited since: the recorded hash no longer matches, so it is
    treated exactly like a stranger's file and never written over."""
    dest = tmp_path / "stage"
    assert st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")["ok"]
    (dest / "run_stage.sh").write_text("#!/bin/sh\n" + st._GEN + "\necho edited by hand\n")
    before = _snapshot(dest)
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "unrelated_destination_content"
    assert rec["paths"] == [str(dest / "run_stage.sh")]
    assert any("recorded writing" in e for e in rec["errors"]), rec["errors"]
    assert _snapshot(dest) == before


def test_owned_outputs_that_would_change_are_refused_never_overwritten(tree: Path, tmp_path: Path, capsys):
    """The other half of the rule: outputs proven ours may be KEPT, but if this
    call would put different content there it refuses instead of overwriting."""
    dest = tmp_path / "stage"
    assert st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")["ok"]
    before = _snapshot(dest)
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3.12")        # a different interpreter => new run_stage.sh
    assert rec["ok"] is False and rec["reason"] == "owned_outputs_differ"
    assert rec["paths"] == [str(dest / "run_stage.sh")] and "never overwrites" in rec["detail"]
    assert _snapshot(dest) == before
    # the same rule guards `scan --out`: an unchanged source re-scans (outputs kept), a changed one refuses
    out = tmp_path / "scanned"
    assert st.main(["scan", "--source", str(tree), "--out", str(out)]) == 0; capsys.readouterr()
    stamp = {n: os.stat(out / n).st_mtime_ns for n in ("scan.json", "mapping_proposal.json")}
    assert st.main(["scan", "--source", str(tree), "--out", str(out)]) == 0; capsys.readouterr()
    assert {n: os.stat(out / n).st_mtime_ns for n in stamp} == stamp        # kept, not rewritten
    _pair(tree / "Pt111" / "CO" / "site_0")                                 # the proposal would change now
    assert st.main(["scan", "--source", str(tree), "--out", str(out)]) == 3
    assert json.loads(capsys.readouterr().out)["reason"] == "owned_outputs_differ"


# ── check-to-write injection: the exclusive create must fail ─────────────────
def _inject(monkeypatch, match, plant) -> dict:
    """Plant something at the path the script is about to create, in the instant
    between its ownership check and the create itself. Both create points are
    module-level seams, so nothing has to be raced for real."""
    real_open, real_mkdir, planted = st.open_exclusive, st.mkdir_exclusive, {}

    def open_exclusive(path, mode=0o644):
        if not planted and match(Path(path)):
            planted["path"] = Path(path); plant(Path(path))
        return real_open(path, mode)

    def mkdir_exclusive(path):
        if not planted and match(Path(path)):
            planted["path"] = Path(path); plant(Path(path))
        return real_mkdir(path)

    monkeypatch.setattr(st, "open_exclusive", open_exclusive)
    monkeypatch.setattr(st, "mkdir_exclusive", mkdir_exclusive)
    return planted


@pytest.mark.parametrize("target", list(st.RECORDED_OUTPUTS) + [st.OWNER_MARKER, ".pt_h.staging.tmp", "CONTCAR"])
def test_check_to_write_injection_fails_the_exclusive_create(tree: Path, tmp_path: Path, monkeypatch, target: str):
    """A file or directory appearing after the ownership check FAILS the create:
    nothing is written over, the temp tree already on disk is retained rather
    than removed, the originals are only ever read, and nothing outside the
    destination is touched."""
    dest = tmp_path / "stage"
    is_dir = target.endswith(".staging.tmp")
    planted = _inject(monkeypatch,
                      lambda p: p.name == target and (target != "CONTCAR" or ".pt_h.staging.tmp" in p.parts),
                      lambda p: p.mkdir() if is_dir else p.write_text("planted between check and write\n"))
    before_src, before_outside = _snapshot(tree), _outside_dest(tmp_path, dest)
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "exclusive_create_failed", (target, rec)
    assert rec["path"] == str(planted["path"]) and "FileExistsError" in rec["error"], rec
    assert ("retained_tmp" in rec) == (dest / ".pt_h.staging.tmp").exists(), rec
    if is_dir:
        assert (dest / target).is_dir() and not list((dest / target).iterdir())      # planted dir kept, not filled
    else:
        assert planted["path"].read_text() == "planted between check and write\n", target
    assert _snapshot(tree) == before_src and _outside_dest(tmp_path, dest) == before_outside, target


def test_a_symlink_planted_at_a_generated_output_fails_the_create(tree: Path, tmp_path: Path, monkeypatch):
    """O_NOFOLLOW plus O_EXCL: a link planted at an output path is neither
    followed nor replaced, so its target keeps the user's bytes."""
    dest = tmp_path / "stage"
    victim = tmp_path / "victim.txt"; victim.write_text("user data\n")
    planted = _inject(monkeypatch, lambda p: p.name == "run_stage.sh", lambda p: os.symlink(victim, p))
    before_src = _snapshot(tree)
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "exclusive_create_failed"
    assert rec["path"] == str(planted["path"]) == str(dest / "run_stage.sh") and "Error" in rec["error"], rec
    assert (dest / "run_stage.sh").is_symlink() and victim.read_text() == "user data\n"
    assert _snapshot(tree) == before_src


# ── a copy that fails part way is retained, never cleaned up ─────────────────
def test_stage_retains_the_temp_tree_when_a_copy_fails(tree: Path, tmp_path: Path, monkeypatch):
    dest = tmp_path / "stage"
    tmp_root = dest / ".pt_h.staging.tmp"
    real, calls = st.copy_file, {"n": 0}

    def flaky(src: Path, dst: Path) -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError(28, "No space left on device")
        real(src, dst)

    monkeypatch.setattr(st, "copy_file", flaky)
    before_src = _snapshot(tree)
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "copy_failed"
    assert rec["retained_tmp"] == str(tmp_root) and "No space left on device" in rec["error"]
    assert rec["copied_files"] == 2
    assert tmp_root.is_dir() and _listing(tmp_root)                       # the partial copy is RETAINED
    assert _snapshot(tree) == before_src                                  # originals only read
    assert not (dest / "pt_h").exists()                                   # nothing renamed into place
    for gone in ("run_stage.sh", "stage_preprocess.py", "originals.sha256", "stage_record.json"):
        assert not (dest / gone).exists(), gone                           # the rerunnable unit was never written
    # a rerun refuses to reuse the retained tree and still removes nothing
    monkeypatch.setattr(st, "copy_file", real)
    kept = _listing(tmp_root)
    rec2 = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec2["ok"] is False and rec2["reason"] == "staging_tmp_exists"
    assert _listing(tmp_root) == kept and not (dest / "pt_h").exists()
    assert _snapshot(tree) == before_src


def test_stage_retains_the_temp_tree_when_the_rename_fails(tree: Path, tmp_path: Path, monkeypatch):
    dest = tmp_path / "stage"
    real_rename = Path.rename

    def no_rename(self, target):
        if self.name.endswith(".staging.tmp"):
            raise OSError(18, "Invalid cross-device link")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", no_rename)
    rec = st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] is False and rec["reason"] == "copy_failed"
    assert rec["retained_tmp"] == str(dest / ".pt_h.staging.tmp")
    assert (dest / ".pt_h.staging.tmp").is_dir() and not (dest / "pt_h").exists()
    assert not (dest / "run_stage.sh").exists()


# ── verify: originals unchanged is an item, not an assumption ────────────────
def test_verify_detects_hash_change_new_file_and_new_symlink(tree: Path, tmp_path: Path):
    dest = tmp_path / "stage"
    assert st.stage(tree, dest, "pt_h", COEFF, "/usr/bin/python3")["ok"]
    v = st.verify(tree, dest)
    assert v["ok"] and v["hashes"]["checked"] == 10 and not any(v["inventory_diff"].values()) and v["symlinks_now"] == []
    (tree / "Pt111" / "slab" / "OSZICAR").write_text(OSZ_OK + "extra\n")
    v = st.verify(tree, dest)
    assert not v["ok"] and v["hashes"]["changed"] == ["./Pt111/slab/OSZICAR"]
    assert v["inventory_diff"]["changed"] == ["./Pt111/slab/OSZICAR"]
    (tree / "Pt111" / "slab" / "OSZICAR").write_text(OSZ_OK)
    (tree / "Pt111" / "slab" / "INCAR").unlink()                        # an uncopied original vanished: inventory catches it
    v = st.verify(tree, dest)
    assert v["hashes"]["unchanged"] and v["inventory_diff"]["missing"] == ["./Pt111/slab/INCAR"] and not v["ok"]
    os.symlink(tmp_path, tree / "late_link")
    assert st.verify(tree, dest)["symlinks_now"] == ["late_link"]
    assert st.verify(tree, tmp_path / "nowhere")["ok"] is False        # no manifest => not verified


# ── CLI ───────────────────────────────────────────────────────────────────────
def test_cli_scan_then_stage_then_verify(tree: Path, tmp_path: Path, capsys):
    stage_dir = tmp_path / "stage"
    assert st.main(["scan", "--source", str(tree), "--out", str(stage_dir)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["pairs"] == 5 and out["reactions"] == ["H", "OH"] and out["gas_references"] == ["H2gas"]
    assert (stage_dir / "mapping_proposal.json").is_file() and (stage_dir / st.OWNER_MARKER).is_file()
    confirmed = tmp_path / "coeff_setting.json"                        # the confirmed file lives OUTSIDE the dest:
    confirmed.write_text(json.dumps(COEFF))                            # <dest>/coeff_setting.json is written by `stage`
    assert st.main(["stage", "--source", str(tree), "--dest", str(stage_dir), "--dataset-name", "pt_h",
                    "--coeff", str(confirmed), "--python", "/usr/bin/python3",
                    "--catbench-version", "1.1.4"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["ok"] and rec["catbench_version"] == "1.1.4" and rec["expected_output"].endswith("raw_data/pt_h_adsorption.json")
    assert st.main(["verify", "--source", str(tree), "--dest", str(stage_dir)]) == 0


def test_cli_scan_refuses_out_inside_source_and_exits_3_on_symlinks(tree: Path, tmp_path: Path, capsys):
    assert st.main(["scan", "--source", str(tree), "--out", str(tree / "stage")]) == 3
    assert json.loads(capsys.readouterr().out)["reason"] == "out_refused"
    assert not (tree / "stage").exists()
    os.symlink(tmp_path, tree / "lnk")
    assert st.main(["scan", "--source", str(tree), "--out", str(tmp_path / "s2")]) == 3
    assert json.loads(capsys.readouterr().out)["symlinks"] == ["lnk"]


def test_cli_stage_failure_exits_3(tree: Path, tmp_path: Path, capsys):
    coeff = tmp_path / "c.json"; coeff.write_text(json.dumps(COEFF))
    assert st.main(["stage", "--source", str(tree), "--dest", str(tree / "inside"), "--dataset-name", "x",
                    "--coeff", str(coeff), "--python", "/usr/bin/python3"]) == 3
    assert json.loads(capsys.readouterr().out)["reason"] == "destination_refused"


# ── the two ways a staged dataset goes quietly wrong ─────────────────────────
NSW_RELAX = "   NSW    =    200    number of steps for IOM\n"
ACCURACY = " reached required accuracy - stopping structural energy minimisation\n"


def test_an_unconverged_relaxation_is_reported_and_does_not_block(tree: Path, tmp_path: Path):
    """An unfinished relaxation's last ionic step still parses as an energy, so
    it became a reference energy with no signal anywhere."""
    (tree / "Pt111" / "H" / "site_0" / "OUTCAR").write_text(NSW_RELAX + "ionic step\n" * 40)
    proposal = st.propose_mapping(st.scan(tree))
    flagged = [p for p in proposal["problems"] if p["issue"] == "unconverged_relaxation"]
    assert [p["dir"] for p in flagged] == ["Pt111/H/site_0"]
    # reported, not blocking: the user may still want the run staged deliberately
    assert "unconverged_relaxation" not in st.BLOCKING_ISSUES
    rec = st.stage(tree, tmp_path / "dest", "pt_h", COEFF, "/usr/bin/python3")
    assert rec["ok"] and any(p["issue"] == "unconverged_relaxation" for p in rec["problems"])


def test_a_converged_relaxation_and_a_single_point_are_not_flagged(tree: Path):
    (tree / "Pt111" / "slab" / "OUTCAR").write_text(NSW_RELAX + "ionic step\n" + ACCURACY)
    (tree / "Pt111" / "H" / "site_1" / "OUTCAR").write_text("   NSW    =      0    number of steps for IOM\n")
    proposal = st.propose_mapping(st.scan(tree))
    assert not [p for p in proposal["problems"] if p["issue"] == "unconverged_relaxation"]


def test_a_positive_gas_coefficient_is_warned_about(tree: Path, tmp_path: Path):
    """Upstream sums energy_ref * stoi and the convention subtracts the gas
    terms, so +0.5 where -0.5 was meant shifts every adsorption energy."""
    flipped = {"H": {"slab": -1, "adslab": 1, "H2gas": 0.5},
               "OH": {"slab": -1, "adslab": 1, "H2gas": -0.5}}
    warnings = st.coeff_warnings(flipped)
    assert len(warnings) == 1
    assert "H2gas" in warnings[0] and "-0.5" in warnings[0] and warnings[0].startswith("H:")
    assert st.coeff_warnings(COEFF) == []          # the correct signs say nothing
    # a released gas is positive by right: H2O -> OH* + 1/2 H2 (the how-to's example)
    assert st.coeff_warnings({"OH": {"slab": -1, "adslab": 1, "H2Ogas": -1, "H2gas": 0.5}}) == []
    rec = st.stage(tree, tmp_path / "dest", "pt_h", flipped, "/usr/bin/python3")
    assert rec["ok"] and rec["warnings"] == warnings
