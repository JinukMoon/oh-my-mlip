"""Network-free tests for scripts/catbench_leaderboard.py.

Official values are FETCHED (through an injectable fetch) and never
recomputed; every snapshot records URL / HTTP code / UTC time; an
unreachable endpoint, an absent dataset, a changed schema or a page that
answers for another id makes every row "no comparable official value".
`compare` keeps the user's model order and never claims a VERIFIED
comparison: "yes" only on machine-read signals (the tag IS the official id
string — a name match, not content identity; equal reaction count; D3
published per entry or as a `_D3` name marker), "conditional" whenever any
signal is an operator declaration (`--official-id`, cited
`--official-conditions`), "no" otherwise (README-count alias alone,
page-level `has_d3` alone, differing count/D3, non-finite value). The
status+basis label travels with the number into comparison.json and the
markdown table. It refuses a `result/` that holds a model outside the
selection.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import catbench_leaderboard as lb  # noqa: E402

META = {"datasets": [{"id": "FG", "reaction_count": 2651}, {"id": "MamunHighT2019", "reaction_count": 45130}],
        "mlips": [], "mlip_name_mapping": {"SevenNet-MF-OMPA": "7net-mf-ompa"}}


def _entry(mae, n=2651):
    return {"MAE_total_eV": mae, "MAE_normal_eV": mae - 0.01, "MAE_single_eV": mae + 0.01,
            "ADwT_pct": 50.0, "AMDwT_pct": 40.0, "num_total": n, "original_name": "x"}


FG_DOC = {"id": "FG", "reaction_count": 2651, "last_updated": "2026-08-27T23:32:40Z", "has_d3": False,
          "results": {"MACE-MPA-0": _entry(0.20), "7net-mf-ompa": _entry(0.15), "UMA-S": _entry(0.10, n=2600)}}


def _fetch(meta=META, docs=None, fail=()):
    docs = docs if docs is not None else {"FG": FG_DOC}
    def fetch(url):
        if url in fail:
            return (0, b"", "URLError: down")
        if url == lb.META_URL:
            return (200, json.dumps(meta).encode(), None)
        for lid, doc in docs.items():
            if url == lb.DATASET_URL.format(id=lid):
                return (200, (doc if isinstance(doc, bytes) else json.dumps(doc).encode()), None)
        return (404, b"<html>", "HTTPError: 404")
    return fetch


EVIDENCE = {"MACE-MPA-0": {"d3": False, "source": "user-cited: CatBench paper SI table (example citation)"},
            "7net-mf-ompa": {"d3": False, "source": "user-cited: CatBench paper SI table (example citation)"}}


def test_leaderboard_id_maps_zenodo_stems_and_passes_exact_ids():
    assert lb.leaderboard_id("FG_dataset", META) == "FG"
    assert lb.leaderboard_id("FG", META) == "FG"
    assert lb.leaderboard_id("MamunHighT2019", META) == "MamunHighT2019"
    assert lb.leaderboard_id("OC20-Dense", META) is None            # not on the leaderboard
    assert lb.leaderboard_id("my_vasp_tag", META) is None
    assert lb.leaderboard_id("BM_dataset", None) == "BM"            # without meta: the known alias only
    assert lb.leaderboard_id_source("FG_dataset", META) == ("FG", "alias")     # located by inference ...
    assert lb.leaderboard_id_source("FG", META) == ("FG", "exact")            # ... vs the id itself
    assert lb.leaderboard_id_source("OC20-Dense", META) == (None, None)


def test_check_schema_flags_missing_keys_bad_entries_and_id_mismatch():
    assert lb.check_schema(FG_DOC, "FG")["ok"]
    s = lb.check_schema({"id": "FG", "reaction_count": 1, "results": {"m": {"MAE_total_eV": 1}}}, "FG")
    assert not s["ok"] and s["missing_result_keys"][0]["model"] == "m" and "ADwT_pct" in s["missing_result_keys"][0]["missing"]
    assert lb.check_schema({"id": "FG", "results": {}}, "FG")["missing_dataset_keys"] == ["reaction_count", "results (empty or not an object)"]
    s = lb.check_schema(FG_DOC, "BM")
    assert not s["ok"] and "requested 'BM'" in s["id_mismatch"]
    assert not lb.check_schema(["not", "a", "dict"], "FG")["ok"]


def test_snapshot_available_records_provenance(tmp_path: Path):
    rec = lb.snapshot("FG_dataset", tmp_path, fetch=_fetch())
    assert rec["available"] and rec["leaderboard_id"] == "FG" and rec["reason"] is None
    assert rec["leaderboard_id_source"] == "alias" and "README-count alias" in rec["leaderboard_id_note"]
    assert lb.snapshot("FG", tmp_path / "exact", fetch=_fetch())["leaderboard_id_source"] == "exact"
    assert [a["url"] for a in rec["attempts"]] == [lb.META_URL, lb.DATASET_URL.format(id="FG")]
    assert all(a["http_code"] == 200 and a["utc"].endswith("+00:00") for a in rec["attempts"])
    assert rec["official_reaction_count"] == 2651 and rec["official_has_d3"] is False and "per-model" in rec["official_has_d3_note"]
    assert rec["models_listed"] == ["7net-mf-ompa", "MACE-MPA-0", "UMA-S"]
    assert len(rec["dataset_body_sha256"]) == 64
    assert (tmp_path / "leaderboard_meta.json").is_file() and (tmp_path / "leaderboard_FG.json").is_file()
    side = json.loads((tmp_path / "leaderboard_snapshot.meta.json").read_text())
    assert side["mlip_name_mapping"] == META["mlip_name_mapping"]


@pytest.mark.parametrize("dataset,fail,docs,reason", [
    ("FG_dataset", (lb.META_URL,), None, "meta_unreachable"),
    ("OC20-Dense", (), None, "not_on_leaderboard"),
    ("my_vasp_tag", (), None, "not_on_leaderboard"),
    ("FG_dataset", (lb.DATASET_URL.format(id="FG"),), None, "dataset_unreachable"),
    ("FG_dataset", (), {"FG": b"{not json"}, "unparseable"),
    ("FG_dataset", (), {"FG": {"id": "FG", "reaction_count": 1, "results": {"m": {}}}}, "schema_mismatch"),
    ("FG_dataset", (), {"FG": dict(FG_DOC, id="BM")}, "schema_mismatch"),
])
def test_snapshot_unavailable_cases_are_recorded_not_raised(tmp_path, dataset, fail, docs, reason):
    rec = lb.snapshot(dataset, tmp_path, fetch=_fetch(docs=docs, fail=fail))
    assert rec["available"] is False and rec["reason"].startswith(reason)
    assert rec["attempts"] and all("url" in a and "utc" in a and "http_code" in a for a in rec["attempts"])
    assert (tmp_path / "leaderboard_snapshot.meta.json").is_file()


def _hub_report(report: Path, rows: dict[str, tuple[float, int]]) -> None:
    report.mkdir(parents=True, exist_ok=True)
    with (report / "mae_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["MLIP_name", "MAE_total", "MAE_normal", "MAE_single", "ADwT", "AMDwT", "Num_total"])
        w.writeheader()
        for name, (mae, n) in rows.items():
            w.writerow({"MLIP_name": name, "MAE_total": mae, "MAE_normal": mae, "MAE_single": mae, "ADwT": 1, "AMDwT": 1, "Num_total": n})


def test_compare_keeps_user_order_and_marks_comparability_honestly(tmp_path: Path):
    snap, report, result = tmp_path / "lb", tmp_path / "report", tmp_path / "result"
    lb.snapshot("FG_dataset", snap, fetch=_fetch())
    _hub_report(report, {"MACE-MPA-0": (0.21, 2651), "SevenNet-MF-OMPA": (0.16, 2651), "UMA-S": (0.11, 2651),
                         "MACE-MPA-0_D3": (0.19, 2651)})
    for m in ("MACE-MPA-0", "SevenNet-MF-OMPA", "UMA-S", "MACE-MPA-0_D3", "MyOwnCalc"):
        (result / m).mkdir(parents=True)
    models = ["UMA-S", "MyOwnCalc", "MACE-MPA-0_D3", "SevenNet-MF-OMPA", "MACE-MPA-0"]   # deliberately not sorted
    conditions = {"dataset": "FG_dataset", "official_id": "FG", "catbench_version": "1.1.4", "d3": False, "calc_num": 3}
    cmp = lb.compare(snap, report, models, conditions, result, official_conditions=EVIDENCE)
    assert [r["model"] for r in cmp["rows"]] == models                    # user order kept
    ident = cmp["dataset_identity"]                                       # declared != verified
    assert ident["status"] == "declared" and ident["content_verified"] is False and "NOT verified" in ident["how"]
    assert cmp["leaderboard_id_source"] == "alias" and cmp["official_conditions_evidence"] == ["7net-mf-ompa", "MACE-MPA-0"]
    assert "NOT verified" in cmp["official_conditions_note"] and cmp["verified_comparable_rows"] == 0
    by = {r["model"]: r for r in cmp["rows"]}
    exact = by["MACE-MPA-0"]
    assert exact["comparable"] is False and exact["conditional"] is True   # never "yes" on a declaration
    assert exact["comparison"]["status"] == "conditional_declared" and exact["comparison"]["verified"] is False
    assert len(exact["comparison"]["basis"]) == 2                          # both declarations named as the basis
    assert exact["official_name"] == "MACE-MPA-0" and exact["official"]["MAE_total"] == 0.20
    assert exact["discrepancies"] == [] and "not published" in exact["caveats"][0] and "NOT verified" in exact["caveats"][1]
    assert exact["official_d3"]["value"] is False and exact["official_d3"]["kind"] == lb.D3_DECLARED
    assert EVIDENCE["MACE-MPA-0"]["source"] in exact["official_d3"]["source"] and "not verified" in exact["official_d3"]["source"]
    mapped = by["SevenNet-MF-OMPA"]
    assert mapped["conditional"] and not mapped["comparable"] and mapped["official_name"] == "7net-mf-ompa"
    assert any("mlip_name_mapping" in d for d in mapped["discrepancies"])
    d3 = by["MACE-MPA-0_D3"]
    assert d3["comparable"] is False and d3["conditional"] is False and d3["official_name"] == "MACE-MPA-0"
    assert any(d.startswith("D3 differs: hub on vs official off") for d in d3["discrepancies"])
    count = by["UMA-S"]                                                   # count differs AND no D3 evidence for UMA-S
    assert count["comparable"] is False and any("reaction count differs" in d for d in count["discrepancies"])
    assert count["official_d3"]["value"] is None and any("official D3 condition not evidenced for 'UMA-S'" in d for d in count["discrepancies"])
    own = by["MyOwnCalc"]
    assert own["official"] == lb.NO_VALUE and own["comparable"] is False
    assert any("not listed on the official FG page" in d and lb.META_URL in d for d in own["discrepancies"])
    assert cmp["result_guard"]["ok"] and cmp["result_guard"]["extra"] == []
    md = lb.render_markdown(cmp)
    assert md.index("| UMA-S ") < md.index("| MACE-MPA-0 |") and "NOT a ranking" in md and "result/ guard: OK" in md
    assert "| conditional (operator-declared, unverified: dataset identity declared by operator" in md and "| no |" in md
    assert "| yes" not in md and "caveat:" in md and "Dataset identity: DECLARED" in md
    assert "Comparison basis:" in md and "no row is verified comparable" in md and "never verified" in md   # header + column


def test_declared_vs_verified_travels_with_every_number(tmp_path: Path):
    """Every consumer of a row (JSON row, markdown cell, CLI label) carries the
    comparison status AND its basis; no consumer can print 'yes' bare, and the
    strongest possible row still says content identity was not measured."""
    yes = _one_row(tmp_path, dict(FG_DOC, results={"MACE-MPA-0": dict(_entry(0.2), d3=False)}), "FG")
    assert yes["comparable"] is True and yes["comparison"]["status"] == "comparable_by_name_match"
    assert yes["comparison"]["verified"] is False and "content identity not measured" in yes["comparison"]["label"]
    assert yes["comparison"]["label"].startswith("yes (basis: name-matched official id")
    assert yes["comparison"]["basis"] == ["dataset tag equals the official id string 'FG' (a name match; content identity not measured)",
                                          "reaction counts equal", "official D3 published: official entry field 'd3'"]
    assert lb.comparability_label(yes) == yes["comparison"]["label"]
    assert any(c.startswith("dataset identity:") and "content identity not measured" in c for c in yes["caveats"])
    # a cited evidence file is a declaration: it never lifts a row to "yes", even with a name-matched id
    cited = _one_row(tmp_path, FG_DOC, "FG", evidence=EVIDENCE)
    assert cited["comparable"] is False and cited["conditional"] is True
    assert cited["comparison"]["status"] == "conditional_declared" and cited["official_d3"]["kind"] == lb.D3_DECLARED
    assert cited["comparison"]["basis"] == ["official D3 " + cited["official_d3"]["source"]]
    assert "--official-conditions" in cited["comparison"]["label"] and "unverified" in cited["comparison"]["label"]
    assert any(c.startswith("official D3: operator-declared") for c in cited["caveats"])
    # the markdown cell prints the same label, never a bare word
    snap, report = tmp_path / "lb_md", tmp_path / "report"
    lb.snapshot("FG", snap, fetch=_fetch(docs={"FG": dict(FG_DOC, results={"MACE-MPA-0": dict(_entry(0.2), d3=False), "7net-mf-ompa": _entry(0.15)})}))
    _hub_report(report, {"MACE-MPA-0": (0.21, 2651), "SevenNet-MF-OMPA": (0.16, 2651)})
    cmp = lb.compare(snap, report, ["MACE-MPA-0", "SevenNet-MF-OMPA"], {"dataset": "FG", "d3": False}, official_conditions=EVIDENCE)
    md = lb.render_markdown(cmp)
    for r in cmp["rows"]:
        assert f"| {r['comparison']['label']} |" in md
    assert "| yes |" not in md and "| conditional |" not in md
    assert md.count("content identity not measured") >= 2                  # header (identity line) + the 'yes' cell


def _one_row(tmp_path: Path, doc: dict, dataset: str, model: str = "MACE-MPA-0", d3: bool = False,
             official_id: str | None = None, evidence: dict | None = None) -> dict:
    snap, report = tmp_path / f"lb_{len(list(tmp_path.iterdir()))}", tmp_path / "report"
    lb.snapshot(dataset, snap, fetch=_fetch(docs={"FG": doc}))
    _hub_report(report, {model: (0.21, 2651)})
    return lb.compare(snap, report, [model], {"dataset": dataset, "official_id": official_id, "d3": d3},
                      official_conditions=evidence)["rows"][0]


def test_alias_only_dataset_identity_fails_closed(tmp_path: Path):
    """The README-count alias FG_dataset -> FG locates the page; it never makes a row comparable."""
    row = _one_row(tmp_path, FG_DOC, "FG_dataset", evidence=EVIDENCE)
    assert row["official_name"] == "MACE-MPA-0" and row["official"]["MAE_total"] == 0.20   # value shown ...
    assert row["comparable"] is False and row["conditional"] is False                        # ... but not comparable
    assert any("dataset identity not evidenced" in d and "README-count alias" in d and "--official-id FG" in d
               for d in row["discrepancies"])
    # the id string itself + PUBLISHED D3 -> "yes", labelled as a name match (content identity not measured)
    published = dict(FG_DOC, results={"MACE-MPA-0": dict(_entry(0.2), d3=False)})
    exact = _one_row(tmp_path, published, "FG")
    assert exact["comparable"] is True and exact["conditional"] is False
    assert lb.comparability_label(exact).startswith("yes (basis: name-matched official id") and "not measured" in lb.comparability_label(exact)
    # the explicit declaration is recorded but unverified -> "conditional", never "yes"
    declared = _one_row(tmp_path, published, "FG_dataset", official_id="FG")
    assert declared["comparable"] is False and declared["conditional"] is True
    assert lb.comparability_label(declared).startswith("conditional (operator-declared, unverified") and "--official-id" in lb.comparability_label(declared)
    assert declared["discrepancies"] == [] and any("NOT verified" in c for c in declared["caveats"])
    # a declaration that names another id is a discrepancy, not a pass
    wrong = _one_row(tmp_path, published, "FG_dataset", official_id="BM")
    assert wrong["comparable"] is False and wrong["conditional"] is False
    assert any("--official-id 'BM' does not equal" in d for d in wrong["discrepancies"])
    # a declaration never rescues a row with another hard discrepancy (count differs here)
    doc = dict(FG_DOC, results={"MACE-MPA-0": dict(_entry(0.2, n=2600), d3=False)})
    row = _one_row(tmp_path, doc, "FG_dataset", official_id="FG")
    assert row["comparable"] is False and row["conditional"] is False and any("reaction count differs" in d for d in row["discrepancies"])


def test_page_level_has_d3_alone_never_evidences_per_model_d3(tmp_path: Path):
    for page_flag in (False, True, None):
        doc = dict(FG_DOC); doc.pop("has_d3", None)
        if page_flag is not None:
            doc["has_d3"] = page_flag
        row = _one_row(tmp_path, doc, "FG")                              # identity exact, hub D3 off, no evidence
        assert row["comparable"] is False and row["official_d3"]["value"] is None
        assert any(f"official D3 condition not evidenced for 'MACE-MPA-0' (page-level has_d3={page_flag!r} only" in d
                   for d in row["discrepancies"])
    # published paths (boolean per-entry field, _D3 name marker) -> "yes"; a cited evidence file -> "conditional"
    field_doc = dict(FG_DOC, has_d3=True, results={"MACE-MPA-0": dict(_entry(0.2), d3=False)})
    row = _one_row(tmp_path, field_doc, "FG")
    assert row["comparable"] is True and row["official_d3"] == {"value": False, "source": "official entry field 'd3'", "kind": lb.D3_PUBLISHED}
    marker_doc = dict(FG_DOC, has_d3=True, results={"MACE-MPA-0_D3": _entry(0.2)})
    row = _one_row(tmp_path, marker_doc, "FG", model="MACE-MPA-0_D3", d3=True)
    assert row["comparable"] is True and row["official_d3"]["source"].endswith("_D3 marker") and row["official_d3"]["kind"] == lb.D3_PUBLISHED
    row = _one_row(tmp_path, field_doc, "FG", model="MACE-MPA-0_D3", d3=True)   # published off vs hub on
    assert row["comparable"] is False and any(d.startswith("D3 differs: hub on vs official off") for d in row["discrepancies"])
    row = _one_row(tmp_path, FG_DOC, "FG", evidence=EVIDENCE)
    assert row["comparable"] is False and row["conditional"] is True and row["official_d3"]["kind"] == lb.D3_DECLARED
    assert row["official_d3"]["source"].startswith("operator-declared via --official-conditions, citing: ")


def test_official_conditions_file_requires_boolean_d3_and_a_source(tmp_path: Path):
    good = tmp_path / "good.json"; good.write_text(json.dumps(EVIDENCE))
    assert lb.load_official_conditions(good) == EVIDENCE
    for bad in ({}, [], {"MACE-MPA-0": {"d3": "no", "source": "x"}}, {"MACE-MPA-0": {"d3": False}},
                {"MACE-MPA-0": {"d3": False, "source": "  "}}):
        p = tmp_path / "bad.json"; p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            lb.load_official_conditions(p)


def test_compare_unavailable_snapshot_gives_no_value_rows_with_attempted_url(tmp_path: Path):
    snap, report = tmp_path / "lb", tmp_path / "report"
    lb.snapshot("FG_dataset", snap, fetch=_fetch(fail=(lb.META_URL,)))
    _hub_report(report, {"MACE-MPA-0": (0.21, 2651)})
    cmp = lb.compare(snap, report, ["MACE-MPA-0", "Other"], {"dataset": "FG_dataset"})
    assert cmp["available"] is False and cmp["reason"] == "meta_unreachable"
    for r in cmp["rows"]:
        assert r["official"] == lb.NO_VALUE and r["comparable"] is False
        assert any("meta_unreachable" in d and lb.META_URL in d for d in r["discrepancies"])
    assert any("hub result missing" in d for d in cmp["rows"][1]["discrepancies"])
    assert cmp["result_guard"] is None and "NOT CHECKED" in lb.render_markdown(cmp)


def test_compare_never_marks_comparable_with_non_numeric_official_values(tmp_path: Path):
    doc = dict(FG_DOC, results={"MACE-MPA-0": dict(_entry(0.2), MAE_total_eV="n/a")})
    snap, report = tmp_path / "lb", tmp_path / "report"
    lb.snapshot("FG_dataset", snap, fetch=_fetch(docs={"FG": doc}))
    _hub_report(report, {"MACE-MPA-0": (0.21, 2651)})
    row = lb.compare(snap, report, ["MACE-MPA-0"], {"dataset": "FG"})["rows"][0]
    assert row["comparable"] is False and row["official"]["MAE_total"] is None
    assert any("not finite" in d for d in row["discrepancies"])


def test_result_guard_refuses_models_outside_the_selection(tmp_path: Path):
    for m in ("A", "B", "baseline_X"):
        (tmp_path / "result" / m).mkdir(parents=True)
    g = lb.assert_result_only_selected(tmp_path / "result", ["A", "B"])
    assert not g["ok"] and g["extra"] == ["baseline_X"] and "baseline rerun" in g["note"]
    g = lb.assert_result_only_selected(tmp_path / "result", ["A", "B", "baseline_X", "C"])
    assert g["ok"] and g["missing"] == ["C"]
    assert lb.assert_result_only_selected(tmp_path / "none", ["A"])["present"] == []


def test_cli_snapshot_and_compare_write_files_and_exit_codes(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(lb, "_default_fetch", lambda url, timeout=60.0: _fetch()(url))
    snap, report, result = tmp_path / "lb", tmp_path / "report", tmp_path / "result"
    assert lb.main(["snapshot", "--dataset", "FG_dataset", "--out", str(snap)]) == 0
    assert json.loads(capsys.readouterr().out)["available"] is True
    _hub_report(report, {"MACE-MPA-0": (0.21, 2651)})
    (result / "MACE-MPA-0").mkdir(parents=True)
    evidence = tmp_path / "official_conditions.json"; evidence.write_text(json.dumps(EVIDENCE))
    base = ["compare", "--snapshot", str(snap), "--report", str(report), "--result", str(result),
            "--models", "MACE-MPA-0", "--dataset", "FG_dataset", "--catbench-version", "1.1.4", "--d3", "0", "--calc-num", "3"]
    assert lb.main(base) == 0                                            # alias-located, no evidence: runs, but ...
    assert json.loads((snap / "comparison.json").read_text())["rows"][0]["comparable"] is False   # ... fails closed
    argv = base + ["--official-id", "FG", "--official-conditions", str(evidence)]
    assert lb.main(argv) == 0
    cmp = json.loads((snap / "comparison.json").read_text())
    row = cmp["rows"][0]                                                 # declared identity: conditional, not "yes"
    assert row["comparable"] is False and row["conditional"] is True and cmp["dataset_identity"]["status"] == "declared"
    assert row["comparison"]["status"] == "conditional_declared" and row["comparison"]["verified"] is False
    assert cmp["conditions"]["catbench_version"] == "1.1.4"
    assert cmp["conditions"]["official_id"] == "FG" and len(cmp["conditions"]["official_conditions_sha256"]) == 64
    md = (snap / "comparison.md").read_text()
    assert md.startswith("# CatBench comparison") and "Dataset identity: DECLARED" in md and "unverified" in md
    assert f"| {row['comparison']['label']} |" in md and md in capsys.readouterr().out    # file == stdout, same label
    exact_argv = [a if a != "FG_dataset" else "FG" for a in base] + ["--official-conditions", str(evidence)]
    assert lb.main(exact_argv) == 0                                      # exact id + cited (declared) D3: still conditional
    row = json.loads((snap / "comparison.json").read_text())["rows"][0]
    assert row["comparable"] is False and row["conditional"] is True and row["official_d3"]["kind"] == "declared"
    capsys.readouterr()
    (tmp_path / "uncited.json").write_text(json.dumps({"MACE-MPA-0": {"d3": False}}))
    assert lb.main(base + ["--official-id", "FG", "--official-conditions", str(tmp_path / "uncited.json")]) == 2
    assert "rejected" in capsys.readouterr().err
    (result / "SomeBaseline").mkdir()
    assert lb.main(argv) == 3                                           # baseline rerun in result/ => refused
    assert "result/ guard: FAILED" in capsys.readouterr().out
