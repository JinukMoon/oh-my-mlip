"""Network-free tests for scripts/catbench_datasets.py.

Two layers stay apart: upstream README rows (each cited) and user-supplied
hints (labelled). A target with no hint match yields `ask`, never a guess;
no target yields `needs_target`. `--confirm` sets `zenodo_size_confirmed`
only when the live Zenodo listing carries the file at a size that agrees
with the README figure; `leaderboard_alias_verified` is always False (the
alias is README-count inference, nothing verifies it); unreachable sources
are recorded, not raised.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import catbench_datasets as cd  # noqa: E402


def test_catalogue_rows_carry_citation_and_labelled_hint():
    rows = cd.catalogue()
    assert {r["name"] for r in rows} == set(cd.UPSTREAM)
    for r in rows:
        assert r["citation"] == cd._CITATION and r["get_benchmark_arg"] == r["name"]
        assert r["zenodo_size_confirmed"] is False and r["leaderboard_alias_verified"] is False and r["confirmation"] is None
        assert "confirmed" not in r and "inferred from README reaction counts" in r["leaderboard_alias_note"]
        assert r["description"] and r["size_mb"] is not None
    by = {r["name"]: r for r in rows}
    assert by["MamunHighT2019"]["user_hint"] and by["KHLOHC_origin"]["user_hint"] is None   # no hint => None, not invented
    assert by["FG_dataset"]["leaderboard_id"] == "FG" and by["OC20-Dense"]["leaderboard_id"] is None


def test_recommend_asks_when_no_target_or_no_mapping():
    r = cd.recommend(None)
    assert r["needs_target"] and r["candidates"] == [] and "surfaces, adsorbates" in r["ask"]
    assert cd.recommend("   ")["needs_target"]
    r = cd.recommend("single-atom Fe-N-C electrocatalyst for ORR")
    assert not r["needs_target"] and r["candidates"] == [] and "do not guess" in r["ask"]


def test_recommend_returns_matching_candidates_not_a_ranking():
    r = cd.recommend("CO and H on Pt-Ni bimetallic alloy surfaces")
    names = [c["name"] for c in r["candidates"]]
    assert names == ["MamunHighT2019"] and r["ask"] is None
    assert r["candidates"][0]["matched_keywords"] == ["alloy", "bimetallic"]
    assert r["candidates"][0]["citation"] == cd._CITATION
    r = cd.recommend("large organic molecules on metal oxide")
    assert {c["name"] for c in r["candidates"]} == {"FG_dataset", "ComerGeneralized2024"}
    assert "rules of thumb" in r["note"]
    assert cd.recommend("oxides")["candidates"] == []        # whole-word match: 'oxide' != 'oxides'


def _fake_fetch(zenodo_files, meta_datasets, fail=None):
    def fetch(url):
        if fail and url in fail:
            return (0, b"", "URLError: down")
        if url == cd.ZENODO_CONCEPT_URL:
            return (200, json.dumps({"files": zenodo_files}).encode(), None)
        if url == cd.LEADERBOARD_META_URL:
            return (200, json.dumps({"datasets": meta_datasets}).encode(), None)
        return (404, b"", "not found")
    return fetch


def test_confirm_marks_only_name_and_size_agreement():
    zfiles = [{"key": "FG_dataset_adsorption.json", "size": 9_200_000, "checksum": "md5:x"},
              {"key": "BM_dataset_adsorption.json", "size": 5_000_000, "checksum": "md5:y"},   # README says 0.3 MB
              {"key": "unrelated.txt", "size": 1}]
    meta = [{"id": "FG", "reaction_count": 2651}]
    rows = cd.confirm(cd.catalogue(), fetch=_fake_fetch(zfiles, meta))
    by = {r["name"]: r for r in rows}
    assert by["FG_dataset"]["zenodo_size_confirmed"] is True
    assert by["FG_dataset"]["confirmation"]["zenodo"]["size_matches_readme"] is True
    lbc = by["FG_dataset"]["confirmation"]["leaderboard"]
    assert lbc["url"] == cd.LEADERBOARD_META_URL and lbc["listed"] is True and lbc["reaction_count"] == 2651 and lbc["error"] is None
    assert "not evidence" in lbc["note"]
    assert by["BM_dataset"]["zenodo_size_confirmed"] is False and by["BM_dataset"]["confirmation"]["zenodo"]["size_matches_readme"] is False
    assert by["MamunHighT2019"]["zenodo_size_confirmed"] is False and by["MamunHighT2019"]["confirmation"]["zenodo"]["listed"] is False
    assert by["OC20-Dense"]["confirmation"]["leaderboard"]["listed"] is False
    for r in rows:                                            # a correct size never promotes the alias (reviewer finding 4)
        assert r["leaderboard_alias_verified"] is False and "confirmed" not in r
    assert cd.confirmation_words(by["FG_dataset"]) == "zenodo size confirmed; leaderboard alias 'FG' NOT verified"


def test_wrong_alias_with_correct_zenodo_size_is_not_confirmed_as_identity(monkeypatch):
    """Finding 4: the Zenodo size check says nothing about the leaderboard alias.
    With a deliberately wrong alias and a correct size, the size flag is true
    and the identity flag is still false — no single 'confirmed' word exists."""
    monkeypatch.setitem(cd.UPSTREAM, "FG_dataset", dict(cd.UPSTREAM["FG_dataset"], leaderboard_id="WRONG-ID"))
    zfiles = [{"key": "FG_dataset_adsorption.json", "size": 9_000_000}]
    row = {r["name"]: r for r in cd.confirm(cd.catalogue(), fetch=_fake_fetch(zfiles, [{"id": "WRONG-ID", "reaction_count": 1}]))}["FG_dataset"]
    assert row["zenodo_size_confirmed"] is True and row["leaderboard_alias_verified"] is False
    assert row["confirmation"]["leaderboard"]["listed"] is True            # the string exists on meta.json ...
    assert "NOT verified" in cd.confirmation_words(row) and "'WRONG-ID'" in cd.confirmation_words(row)   # ... and is still not identity


def test_confirm_records_unreachable_sources():
    rows = cd.confirm(cd.catalogue(), fetch=_fake_fetch([], [], fail={cd.ZENODO_CONCEPT_URL, cd.LEADERBOARD_META_URL}))
    for r in rows:
        assert r["zenodo_size_confirmed"] is False and r["leaderboard_alias_verified"] is False
        assert r["confirmation"]["zenodo"]["error"] == "URLError: down"
        assert r["confirmation"]["leaderboard"]["error"] == "URLError: down"


def test_cli_list_target_and_ask_exit_codes(capsys):
    assert cd.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "NOT live-confirmed" in out and "leaderboard alias 'FG' NOT verified" in out and cd._CITATION in out and "user hint:" in out
    assert cd.main(["--target", "bimetallic alloys"]) == 0
    out = capsys.readouterr().out
    assert "candidate: MamunHighT2019" in out and "NOT verified" in out
    assert cd.main(["--target", "something unmapped"]) == 3
    assert "[ask]" in capsys.readouterr().out
    assert cd.main(["--target", "something unmapped", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ask"]
    assert cd.main([]) == 3                                   # no target, no --list => needs_target
    assert "No dataset is proposed" in capsys.readouterr().out


# ── --check / --fetch: the repo-owned replacements for check_dataset.py /
# fetch_dataset.py. Format rules are exercised in-process with an injected
# loader; the CLI is exercised under a stub `catbench` package on PYTHONPATH
# (toolkit interpreter has no catbench => the wrong-interpreter path is real).
import os          # noqa: E402
import subprocess  # noqa: E402

SCRIPT = REPO_ROOT / "scripts" / "catbench_datasets.py"


def _entry(energy_ref=-1.0, stoi=1.0, atoms=(1, 2), **extra):
    d = {"atoms": list(atoms), "stoi": stoi, "energy_ref": energy_ref}
    d.update(extra)
    return d


def _good_data():
    return {"Pt111_CO": {"raw": {"star": _entry(), "COstar": _entry(-2.0, 1.0), "COgas": {"atoms": [1, 2], "stoi": -1.0}}},
            "Pt111_H": {"raw": {"star": _entry(), "Hstar": _entry(-1.5), "H2gas": {"atoms": [1, 2], "stoi": -0.5}}}}


def test_check_accepts_known_format_and_reports_sha(tmp_path):
    f = tmp_path / "demo_adsorption.json"
    f.write_text("{}")                          # the loader is injected; bytes only feed the sha256
    rec = cd.check_adsorption_json(f, load=lambda p: _good_data())
    assert rec["ok"] and rec["problems"] == []
    assert rec["reactions"] == 2 and rec["structures"] == 6 and rec["gas_keys"] == ["COgas", "H2gas"]
    assert rec["sha256"] == cd._sha256(f) and rec["size_bytes"] == 2


def test_check_names_every_format_problem(tmp_path):
    f = tmp_path / "x.json"; f.write_text("{}")
    data = _good_data()
    del data["Pt111_CO"]["raw"]["star"]                                  # no slab
    data["Pt111_H"]["raw"]["Hstar"]["energy_ref"] = float("nan")         # non-finite reference
    data["Pt111_H"]["raw"]["H2gas"].pop("stoi")                          # gas without stoichiometry
    data["Pt111_H"]["raw"]["Hstar"]["atoms"] = None                      # no structure
    data["broken"] = {"nope": 1}                                         # no raw
    data["only_slab"] = {"raw": {"star": _entry(stoi=True)}}             # bool is not a number; no adslab
    rec = cd.check_adsorption_json(f, load=lambda p: data)
    assert not rec["ok"]
    for needle in ("Pt111_CO: no exact 'star'", "Pt111_H/Hstar: 'energy_ref' missing or not a finite number",
                   "Pt111_H/H2gas: 'stoi' missing", "Pt111_H/Hstar: no 'atoms'", "broken: no 'raw'",
                   "only_slab: no '<X>star'", "only_slab/star: 'stoi' missing"):
        assert any(needle in p for p in rec["problems"]), (needle, rec["problems"])
    assert not cd.check_adsorption_json(tmp_path / "missing.json", load=lambda p: {})["ok"]
    rec = cd.check_adsorption_json(f, load=lambda p: (_ for _ in ()).throw(ValueError("Dangling structure ref")))
    assert rec["problems"] == ["loader failed: ValueError: Dangling structure ref"]
    assert cd.check_adsorption_json(f, load=lambda p: {})["problems"] == ["top level is not a non-empty object of reactions"]


def test_fetch_writes_provenance_and_never_replaces_existing(tmp_path):
    calls = []

    def fake_get_benchmark(name):
        calls.append((name, os.getcwd()))
        p = Path("raw_data") / f"{name}_adsorption.json"          # upstream: cwd-relative
        p.parent.mkdir(exist_ok=True)
        p.write_text('{"a": 1}')
        return str(p)

    work = tmp_path / "work"
    rec = cd.fetch_dataset("FG_dataset", work, get_benchmark=fake_get_benchmark, catbench_version="1.1.4")
    assert calls == [("FG_dataset", str(work))] and os.getcwd() != str(work)
    assert rec["fetched"] and not rec["pre_existing"] and rec["provenance_written"]
    prov = json.loads((work / "raw_data" / "FG_dataset_adsorption.provenance.json").read_text())
    assert prov["sha256"] == rec["sha256"] == cd._sha256(work / "raw_data" / "FG_dataset_adsorption.json")
    assert prov["catbench_version"] == "1.1.4" and prov["size_bytes"] == 8 and "provenance" not in prov
    # second call, file unchanged: upstream is not even called, provenance kept (hash still matches)
    rec2 = cd.fetch_dataset("FG_dataset", work, get_benchmark=fake_get_benchmark)
    assert len(calls) == 1 and rec2["pre_existing"] and not rec2["fetched"] and not rec2["provenance_written"]
    assert "kept" in rec2["provenance_note"] and "error" not in rec2 and rec2["sha256"] == rec["sha256"]
    # file edited after it was recorded: explicit drift error, stale record NOT kept silently, NOT rewritten
    (work / "raw_data" / "FG_dataset_adsorption.json").write_text('{"a": 1, "edited": true}')
    rec3 = cd.fetch_dataset("FG_dataset", work, get_benchmark=fake_get_benchmark)
    assert len(calls) == 1 and rec3["pre_existing"] and "provenance_note" not in rec3 and not rec3["provenance_written"]
    assert rec3["error"].startswith("provenance drift:") and rec3["provenance_drift"] == {"recorded_sha256": prov["sha256"], "current_sha256": rec3["sha256"]}
    assert rec3["sha256"] != prov["sha256"]
    assert json.loads((work / "raw_data" / "FG_dataset_adsorption.provenance.json").read_text()) == prov   # untouched
    # unreadable provenance is an error too, never overwritten
    (work / "raw_data" / "FG_dataset_adsorption.provenance.json").write_text("{not json")
    rec4 = cd.fetch_dataset("FG_dataset", work, get_benchmark=fake_get_benchmark)
    assert "unreadable" in rec4["error"] and (work / "raw_data" / "FG_dataset_adsorption.provenance.json").read_text() == "{not json"
    with pytest.raises(ValueError):
        cd.fetch_dataset("../escape", work, get_benchmark=fake_get_benchmark)
    rec3 = cd.fetch_dataset("Nothing", work, get_benchmark=lambda name: None)
    assert rec3["error"].startswith("get_benchmark('Nothing') returned without writing") and rec3["sha256"] is None


def _stub_catbench(root: Path, version: str = "1.1.4") -> dict:
    """Minimal `catbench` package: __version__, load_catbench_json (rehydrates
    'atoms_json' into a sized placeholder), get_benchmark (writes cwd/raw_data)."""
    pkg = root / "stub" / "catbench"
    (pkg / "utils").mkdir(parents=True)
    (pkg / "adsorption" / "data").mkdir(parents=True)
    (pkg / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (pkg / "utils" / "__init__.py").write_text("")
    (pkg / "utils" / "data_utils.py").write_text(
        "import json\n"
        "def load_catbench_json(path):\n"
        "    data = json.load(open(path))\n"
        "    for rxn in data.values():\n"
        "        for sv in rxn.get('raw', {}).values():\n"
        "            if 'atoms_json' in sv:\n"
        "                sv['atoms'] = [0] * int(sv.pop('atoms_json'))\n"
        "    return data\n")
    (pkg / "adsorption" / "__init__.py").write_text("")
    (pkg / "adsorption" / "data" / "__init__.py").write_text("")
    (pkg / "adsorption" / "data" / "zenodo.py").write_text(
        "import os, json\n"
        "def get_benchmark(name, overwrite=False, verify=True):\n"
        "    os.makedirs('raw_data', exist_ok=True)\n"
        "    p = os.path.join(os.getcwd(), 'raw_data', name + '_adsorption.json')\n"
        "    json.dump({'r': {'raw': {'star': {'atoms_json': 2, 'stoi': 1, 'energy_ref': -1.0},\n"
        "                             'Hstar': {'atoms_json': 3, 'stoi': 1, 'energy_ref': -2.0}}}}, open(p, 'w'))\n"
        "    return p\n")
    return dict(os.environ, PYTHONPATH=str(root / "stub"))


def _cli(args, env=None, cwd=None):
    return subprocess.run([sys.executable, str(SCRIPT), *args], env=env, cwd=cwd, capture_output=True, text=True)


def test_cli_check_and_fetch_under_stub_catbench(tmp_path):
    env = _stub_catbench(tmp_path)
    work = tmp_path / "work"; work.mkdir()
    # --fetch under the pinned version: file + provenance land under --workdir/raw_data
    p = _cli(["--fetch", "FG_dataset", "--workdir", str(work), "--catbench-version", "1.1.4", "--json"], env=env, cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    rec = json.loads(p.stdout)
    assert rec["fetched"] and rec["catbench_version"] == "1.1.4" and rec["provenance_written"]
    assert (work / "raw_data" / "FG_dataset_adsorption.provenance.json").is_file()
    # rerun on the unchanged file: 0, kept; after the JSON drifts: 3 with an explicit [stop]
    p = _cli(["--fetch", "FG_dataset", "--workdir", str(work), "--catbench-version", "1.1.4"], env=env, cwd=tmp_path)
    assert p.returncode == 0 and "(kept)" in p.stdout
    (work / "raw_data" / "FG_dataset_adsorption.json").write_text('{"drifted": 1}')
    p = _cli(["--fetch", "FG_dataset", "--workdir", str(work), "--catbench-version", "1.1.4"], env=env, cwd=tmp_path)
    assert p.returncode == 3 and "[stop] provenance drift:" in p.stderr
    (work / "raw_data" / "FG_dataset_adsorption.json").unlink()
    (work / "raw_data" / "FG_dataset_adsorption.provenance.json").unlink()
    p = _cli(["--fetch", "FG_dataset", "--workdir", str(work), "--catbench-version", "1.1.4", "--json"], env=env, cwd=tmp_path)
    assert p.returncode == 0 and json.loads(p.stdout)["fetched"]
    # --check on what was fetched, recorded to a file
    record = tmp_path / "check.json"
    p = _cli(["--check", str(work / "raw_data" / "FG_dataset_adsorption.json"), "--record", str(record), "--catbench-version", "1.1.4"], env=env)
    assert p.returncode == 0, p.stderr
    assert "1 reactions, 2 structures" in p.stdout and json.loads(record.read_text())["ok"]
    assert json.loads(record.read_text())["sha256"] == rec["sha256"]
    # --check exits 3 on a format problem, naming it
    bad = work / "raw_data" / "bad_adsorption.json"
    bad.write_text(json.dumps({"r": {"raw": {"star": {"atoms_json": 2, "stoi": 1, "energy_ref": -1.0}}}}))
    p = _cli(["--check", str(bad)], env=env)
    assert p.returncode == 3 and "no '<X>star'" in p.stdout
    # version guard: stop (3) before doing anything; wrong interpreter (no catbench): 2
    p = _cli(["--check", str(bad), "--catbench-version", "1.1.3"], env=env)
    assert p.returncode == 3 and "approved catbench 1.1.3 but this env has 1.1.4" in p.stderr
    p = _cli(["--fetch", "BM_dataset", "--workdir", str(work), "--catbench-version", "1.1.3"], env=env)
    assert p.returncode == 3 and not (work / "raw_data" / "BM_dataset_adsorption.json").exists()
    # --check and --fetch are exclusive; the recommendation modes are untouched
    assert _cli(["--check", str(bad), "--fetch", "x"], env=env).returncode == 2
    assert _cli(["--target", "bimetallic alloys"]).returncode == 0


def test_wrong_interpreter_without_catbench_exits_2(monkeypatch, tmp_path, capsys):
    """`sys.modules['catbench'] = None` makes `import catbench` raise ImportError
    in this process — the toolkit interpreter itself happens to carry an old
    catbench, so the wrong-interpreter path is exercised in-process."""
    monkeypatch.setitem(sys.modules, "catbench", None)
    mod, why = cd.import_catbench()
    assert mod is None and "catbench-bearing env interpreter" in why
    f = tmp_path / "x_adsorption.json"; f.write_text("{}")
    assert cd.main(["--check", str(f)]) == 2
    assert "catbench-bearing env interpreter" in capsys.readouterr().err
    assert cd.main(["--fetch", "FG_dataset", "--workdir", str(tmp_path)]) == 2
    assert not (tmp_path / "raw_data").exists()
