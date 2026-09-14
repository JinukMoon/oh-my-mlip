#!/usr/bin/env python3
"""catbench_leaderboard.py — official leaderboard values at request time, and
an honest comparison against the hub's own `report/mae_table.csv`.

Rules (recipes/catbench.md §2 "Comparison rule" / §5):
  * Official values are FETCHED, never recomputed; no published uMLIP is
    rerun as a baseline. `assert_result_only_selected` proves `result/`
    holds only the user's models.
  * Every snapshot records URL, HTTP code and UTC time. When the endpoint is
    unreachable, the dataset is not on the leaderboard, or the schema is not
    the one inspected on 2026-09-14, the record says so and every comparison
    row becomes "no comparable official value" with the attempted URL/time.
  * Unlike results are never presented as a ranking: rows keep the user's
    model order, and every difference in dataset id, D3, reaction count or
    unknown official conditions is listed as a discrepancy on the row.
  * Fail closed, and DECLARED is never VERIFIED. The hub never measures
    whether the local dataset's contents are the official page's dataset,
    and it cannot check the conditions behind a published number. So:
      - "yes" rows rest only on MACHINE-READ signals: the user's tag is the
        official id string itself (a name match, not content identity),
        the reaction counts are equal, and the official entry's D3 is
        published (a boolean per-entry field or a `_D3` name marker). The
        row's label spells that basis out; it never claims verified
        content identity.
      - "conditional" rows rest on OPERATOR DECLARATIONS the hub cannot
        verify: `--official-id <id>` (the operator declares the tag is that
        page's dataset) and/or `--official-conditions` (a cited evidence
        file for per-entry D3). Cited is not verified; such rows are never
        "yes".
      - everything else is "no": the README-count alias table
        (catbench_datasets.UPSTREAM) only locates a page; the page-level
        `has_d3` flag never states per-model D3; a differing count, D3 or
        non-finite value is a discrepancy.
    The status and basis travel with the number: every row in
    comparison.json carries `comparison` {status, basis, verified: false,
    label}, and comparison.md prints the same label in the table cell.

Endpoints (live-inspected 2026-09-14; schema recorded in EXPECTED_*):
  https://catbench.org/data/meta.json                 datasets[], mlips[], mlip_name_mapping{}
  https://catbench.org/data/datasets/<id>.json        results{<model>: {MAE_total_eV, ...}}

Usage:
  python3 scripts/catbench_leaderboard.py snapshot --dataset FG_dataset --out ./leaderboard
  python3 scripts/catbench_leaderboard.py compare  --snapshot ./leaderboard --report ./report \\
          --result ./result --models MACE-MPA-0,SevenNet-MF-OMPA --dataset FG_dataset --official-id FG \\
          --official-conditions ./official_conditions.json --catbench-version 1.1.4 --d3 0 --calc-num 3

  official_conditions.json: {"<official entry name>": {"d3": false, "source": "<citation/URL>"}, ...}
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catbench_datasets import UPSTREAM  # noqa: E402  (Zenodo name -> leaderboard id)

META_URL = "https://catbench.org/data/meta.json"
DATASET_URL = "https://catbench.org/data/datasets/{id}.json"
EXPECTED_DATASET_KEYS = ("id", "reaction_count", "results")
EXPECTED_RESULT_KEYS = ("MAE_total_eV", "MAE_normal_eV", "MAE_single_eV", "ADwT_pct", "AMDwT_pct", "num_total")
# hub report column -> official field (catbench_report._COLUMNS vs datasets/<id>.json results)
FIELD_MAP = {"MAE_total": "MAE_total_eV", "MAE_normal": "MAE_normal_eV", "MAE_single": "MAE_single_eV",
             "ADwT": "ADwT_pct", "AMDwT": "AMDwT_pct", "Num_total": "num_total"}
NO_VALUE = "no comparable official value"


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _default_fetch(url: str, timeout: float = 60.0) -> tuple[int, bytes, str | None]:
    req = urllib.request.Request(url, headers={"User-Agent": "oh-my-mlip/catbench_leaderboard"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), None
    except Exception as exc:  # noqa: BLE001
        return getattr(exc, "code", 0) or 0, b"", f"{type(exc).__name__}: {exc}"


ALIAS_NOTE = ("id inferred from the hub's README-count alias table (catbench_datasets.UPSTREAM); "
              "the alias locates the page but does not evidence that the local dataset is this page's dataset")


def leaderboard_id_source(dataset: str, meta: dict | None = None) -> tuple[str | None, str | None]:
    """User tag / Zenodo name -> (leaderboard id, how it was obtained).
    "exact": the tag is itself a leaderboard id; "alias": mapped through
    catbench_datasets.UPSTREAM (an inference, never identity evidence)."""
    ids = {d.get("id") for d in (meta or {}).get("datasets", [])} if meta else set()
    if dataset in ids:
        return dataset, "exact"
    up = UPSTREAM.get(dataset)
    if up and up.get("leaderboard_id"):
        lid = up["leaderboard_id"]
        return (lid, "alias") if (not ids or lid in ids) else (None, None)
    return (dataset, "exact") if not ids else (None, None)


def leaderboard_id(dataset: str, meta: dict | None = None) -> str | None:
    return leaderboard_id_source(dataset, meta)[0]


_D3_ENTRY_FIELDS = ("d3", "D3", "has_d3", "dispersion")
# where a per-entry D3 condition came from: read off the official JSON, or declared by the operator
D3_PUBLISHED, D3_DECLARED = "published", "declared"


def official_d3(name: str, entry: dict, evidence: dict | None = None) -> tuple[bool | None, str, str | None]:
    """Per-entry D3 condition of an official value: (value, source, kind).
    kind D3_PUBLISHED = read off the official JSON (boolean field / `_D3`
    name); D3_DECLARED = from the operator's cited evidence file (cited is
    not verified); None = not evidenced (the page-level `has_d3` flag is not
    per-model evidence — BM: has_d3=true with no _D3 names, 2026-09-14)."""
    for key in _D3_ENTRY_FIELDS:
        if isinstance(entry.get(key), bool):
            return entry[key], f"official entry field {key!r}", D3_PUBLISHED
    if name.endswith("_D3"):
        return True, "official entry name carries the _D3 marker", D3_PUBLISHED
    ev = (evidence or {}).get(name)
    if ev is not None:
        return bool(ev["d3"]), f"operator-declared via --official-conditions, citing: {ev['source']} (not verified by the hub)", D3_DECLARED
    return None, "the endpoint publishes only the page-level has_d3 flag, which does not state per-model D3", None


def load_official_conditions(path: Path) -> dict:
    """Evidence file for official per-entry conditions:
    {"<official name>": {"d3": bool, "source": "<non-empty citation>"}}.
    Anything else is a ValueError — an uncited or non-boolean entry is not evidence."""
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc, dict) or not doc:
        raise ValueError(f"{path}: expected a non-empty JSON object keyed by official entry name")
    for name, ev in doc.items():
        if not isinstance(ev, dict) or not isinstance(ev.get("d3"), bool):
            raise ValueError(f"{path}: entry {name!r} needs a boolean 'd3'")
        if not isinstance(ev.get("source"), str) or not ev["source"].strip():
            raise ValueError(f"{path}: entry {name!r} needs a non-empty 'source' (citation/URL) — uncited values are not evidence")
    return doc


def check_schema(doc, expected_id: str | None = None) -> dict:
    """The shape inspected on 2026-09-14: a dict with id/reaction_count/results,
    every results entry a dict carrying the six metric fields. `expected_id`
    must equal `doc["id"]` (the page answered for another dataset otherwise)."""
    if not isinstance(doc, dict):
        return {"ok": False, "missing_dataset_keys": ["<document is not a JSON object>"], "missing_result_keys": [], "id_mismatch": None}
    missing = [k for k in EXPECTED_DATASET_KEYS if k not in doc]
    res = doc.get("results")
    result_missing = []
    if isinstance(res, dict) and res:
        for name, entry in res.items():
            bad = [k for k in EXPECTED_RESULT_KEYS if not isinstance(entry, dict) or k not in entry]
            if bad:
                result_missing.append({"model": name, "missing": bad})
    elif "results" in doc:
        missing.append("results (empty or not an object)")
    id_mismatch = (expected_id is not None and doc.get("id") != expected_id)
    return {"ok": not missing and not result_missing and not id_mismatch, "missing_dataset_keys": missing,
            "missing_result_keys": result_missing,
            "id_mismatch": f"requested {expected_id!r}, document says {doc.get('id')!r}" if id_mismatch else None}


def _finite(v) -> bool:
    try:
        return v is not None and not isinstance(v, bool) and float(v) == float(v) and abs(float(v)) != float("inf")
    except (TypeError, ValueError):
        return False


def snapshot(dataset: str, out_dir: Path, fetch=None) -> dict:
    """Fetch meta + dataset JSON into out_dir; write the sidecar record."""
    fetch = fetch or _default_fetch
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    attempts, meta = [], None
    code, body, err = fetch(META_URL)
    attempts.append({"url": META_URL, "http_code": code, "utc": utc_now(), "error": err})
    if code == 200 and body:
        try:
            meta = json.loads(body)
            (out_dir / "leaderboard_meta.json").write_bytes(body)
        except ValueError as exc:
            attempts[-1]["error"] = f"unparseable meta.json: {exc}"
    rec = {"dataset_requested": dataset, "leaderboard_id": None, "available": False, "reason": None,
           "attempts": attempts, "official_last_updated": None, "schema": None, "metric_fields": list(EXPECTED_RESULT_KEYS),
           "models_listed": [], "mlip_name_mapping": (meta or {}).get("mlip_name_mapping", {}), "snapshot_utc": utc_now()}
    if meta is None:
        rec["reason"] = "meta_unreachable"
        _write_sidecar(out_dir, rec); return rec
    lid, source = leaderboard_id_source(dataset, meta)
    rec["leaderboard_id"] = lid
    rec["leaderboard_id_source"] = source
    rec["leaderboard_id_note"] = ALIAS_NOTE if source == "alias" else None
    if lid is None:
        rec["reason"] = "not_on_leaderboard"
        _write_sidecar(out_dir, rec); return rec
    url = DATASET_URL.format(id=lid)
    code, body, err = fetch(url)
    attempts.append({"url": url, "http_code": code, "utc": utc_now(), "error": err})
    if code != 200 or not body:
        rec["reason"] = "dataset_unreachable"
        _write_sidecar(out_dir, rec); return rec
    try:
        doc = json.loads(body)
    except ValueError as exc:
        rec["reason"] = f"unparseable dataset JSON: {exc}"
        _write_sidecar(out_dir, rec); return rec
    (out_dir / f"leaderboard_{lid}.json").write_bytes(body)
    rec["dataset_body_sha256"] = hashlib.sha256(body).hexdigest()
    schema = check_schema(doc, expected_id=lid)
    rec["schema"] = schema
    if isinstance(doc, dict):
        rec["official_last_updated"] = doc.get("last_updated")
        rec["official_reaction_count"] = doc.get("reaction_count")
        rec["official_has_d3"] = doc.get("has_d3")
        rec["official_has_d3_note"] = ("has_d3 is a page-level availability flag; the endpoint does not state per-model "
                                       "whether a listed value used D3 (BM: has_d3=true, no _D3 names — inspected 2026-09-14)")
        rec["models_listed"] = sorted(doc["results"].keys()) if isinstance(doc.get("results"), dict) else []
    if not schema["ok"]:
        rec["reason"] = "schema_mismatch"
    else:
        rec["available"] = True
    _write_sidecar(out_dir, rec)
    return rec


def _write_sidecar(out_dir: Path, rec: dict) -> None:
    (out_dir / "leaderboard_snapshot.meta.json").write_text(json.dumps(rec, indent=2) + "\n")


# ── result/ contains only the selected models (no baseline reruns) ──────────
def list_result_models(result_dir: Path) -> list[str]:
    result_dir = Path(result_dir)
    return sorted(p.name for p in result_dir.iterdir() if p.is_dir()) if result_dir.is_dir() else []


def assert_result_only_selected(result_dir: Path, selected: list[str]) -> dict:
    present = list_result_models(result_dir)
    extra = sorted(set(present) - set(selected))
    missing = sorted(set(selected) - set(present))
    return {"ok": not extra, "present": present, "extra": extra, "missing": missing,
            "note": "extra entries would mean a model outside the approved selection was run (baseline rerun) — refused" if extra else None}


# ── compare ──────────────────────────────────────────────────────────────────
def _read_hub_rows(report_dir: Path) -> dict[str, dict]:
    path = Path(report_dir) / "mae_table.csv"
    if not path.is_file():
        return {}
    with path.open() as fh:
        return {row["MLIP_name"]: row for row in csv.DictReader(fh)}


def match_official(model: str, results: dict, mapping: dict) -> tuple[str | None, list[str]]:
    """Exact name, then `_D3`-stripped (D3 discrepancy noted), then meta.json's
    mlip_name_mapping. None when nothing matches."""
    notes = []
    if model in results:
        return model, notes
    base = model[:-3] if model.endswith("_D3") else model
    if base != model:
        notes.append("hub run used D3; official entry name carries no D3 marker")
        if base in results:
            return base, notes
    mapped = mapping.get(base)
    if mapped and mapped in results:
        notes.append(f"matched via leaderboard mlip_name_mapping {base!r} -> {mapped!r}")
        return mapped, notes
    return None, notes


CONTENT_NOT_MEASURED = ("the hub never measures whether the local dataset's contents are the official page's dataset; "
                        "a name match plus equal reaction count is the strongest signal it has")


def dataset_identity(conditions: dict, rec: dict) -> dict:
    """Is the local dataset the official page's dataset? The hub can only
    compare NAMES, never contents (`content_verified` is always False):
    status "name_match" — the user's tag is the official id string itself
                          (machine-checked string equality; not content identity);
    status "declared"   — the operator passed `--official-id` equal to the page
                          id: a recorded declaration the hub cannot verify;
                          rows become "conditional", never "yes";
    status "unresolved" — alias-located, absent, or contradicted.
    The README-count alias that located the page is never identity evidence."""
    lid = rec.get("leaderboard_id")
    tag, declared = conditions.get("dataset"), conditions.get("official_id")
    base = {"content_verified": False, "content_note": CONTENT_NOT_MEASURED}
    if lid is None:
        return {"status": "unresolved", "basis": None, "how": "no official page", **base}
    if declared is not None:
        if declared == lid:
            return {"status": "declared", "basis": "operator declaration",
                    "how": f"operator declared --official-id {lid!r} for dataset tag {tag!r}: recorded, NOT verified by the hub", **base}
        return {"status": "unresolved", "basis": None,
                "how": f"--official-id {declared!r} does not equal the snapshot's official id {lid!r}", **base}
    if tag == lid:
        return {"status": "name_match", "basis": "machine-checked name match",
                "how": f"dataset tag equals the official id string {lid!r} (a name match; content identity not measured)", **base}
    if tag is None:
        return {"status": "unresolved", "basis": None,
                "how": "no --dataset/--official-id given; identity with the official page not stated", **base}
    if leaderboard_id(tag, None) == lid:
        return {"status": "unresolved", "basis": None,
                "how": f"dataset tag {tag!r} was mapped to official id {lid!r} by the README-count alias only; "
                       f"pass --official-id {lid} to record your declaration (rows then read 'conditional', never 'yes')", **base}
    return {"status": "unresolved", "basis": None, "how": f"dataset tag {tag!r} vs official id {lid!r} (name/subset may differ)", **base}


COMPARISON_BASIS_NOTE = (
    "'yes' rows rest on machine-read signals only (official id name match, equal reaction count, published per-entry D3) "
    "and do not claim verified dataset content identity; 'conditional' rows rest on operator declarations "
    "(--official-id / --official-conditions) the hub cannot verify; no row is verified comparable, and nothing here is a ranking")


def _comparison(row: dict, identity: dict, d3_kind: str | None, d3_source: str, clean: bool) -> dict:
    """The per-row comparison status WITH its basis, so a consumer that
    prints the number cannot drop the qualifier. `verified` is always False."""
    if not clean:
        return {"status": "not_comparable", "basis": [], "verified": False, "label": "no"}
    declared = []
    if identity["status"] == "declared":
        declared.append(f"dataset identity declared by operator ({identity['how']})")
    if d3_kind == D3_DECLARED:
        declared.append(f"official D3 {d3_source}")
    if identity["status"] == "name_match" and d3_kind == D3_PUBLISHED and not declared:
        basis = [identity["how"], "reaction counts equal", f"official D3 published: {d3_source}"]
        return {"status": "comparable_by_name_match", "basis": basis, "verified": False,
                "label": f"yes (basis: name-matched official id, equal reaction count, published D3 [{d3_source}]; "
                         "dataset content identity not measured)"}
    return {"status": "conditional_declared", "basis": declared, "verified": False,
            "label": "conditional (operator-declared, unverified: " + "; ".join(declared) + ")"}


def comparability_label(row: dict) -> str:
    return (row.get("comparison") or {}).get("label") or "no"


def compare(snapshot_dir: Path, report_dir: Path, models: list[str], conditions: dict,
            result_dir: Path | None = None, official_conditions: dict | None = None) -> dict:
    snapshot_dir = Path(snapshot_dir)
    side = snapshot_dir / "leaderboard_snapshot.meta.json"
    rec = json.loads(side.read_text()) if side.is_file() else {"available": False, "reason": "no snapshot", "attempts": []}
    attempted = [{"url": a["url"], "utc": a["utc"], "http_code": a["http_code"]} for a in rec.get("attempts", [])]
    official = {}
    if rec.get("available"):
        official = json.loads((snapshot_dir / f"leaderboard_{rec['leaderboard_id']}.json").read_text()).get("results", {})
    hub = _read_hub_rows(report_dir)
    guard = assert_result_only_selected(result_dir, models) if result_dir else None
    identity = dataset_identity(conditions, rec)
    page_d3 = rec.get("official_has_d3")
    rows = []
    for model in models:                     # user order — deliberately NOT sorted by any metric
        h = hub.get(model)
        row = {"model": model, "hub": {k: h.get(k) for k in FIELD_MAP} if h else None,
               "official": None, "official_name": None, "discrepancies": [], "comparable": False, "conditional": False,
               "comparison": {"status": "not_comparable", "basis": [], "verified": False, "label": "no"}}
        if h is None:
            row["discrepancies"].append("hub result missing from report/mae_table.csv")
        if not rec.get("available"):
            row["official"] = NO_VALUE
            row["discrepancies"].append(f"{rec.get('reason')}; attempted {attempted}")
            rows.append(row); continue
        name, notes = match_official(model, official, rec.get("mlip_name_mapping", {}))
        row["discrepancies"].extend(notes)
        if name is None:
            row["official"] = NO_VALUE
            row["discrepancies"].append(f"model not listed on the official {rec['leaderboard_id']} page; attempted {attempted}")
            rows.append(row); continue
        o = official[name]
        row["official_name"] = name
        row["official"] = {k: (o.get(v) if _finite(o.get(v)) else None) for k, v in FIELD_MAP.items()}
        nonnum = [v for v in FIELD_MAP.values() if not _finite(o.get(v))]
        hard = []                                 # any hard discrepancy => not comparable
        if nonnum:
            hard.append(f"official fields not finite numbers: {nonnum}")
        if identity["status"] == "unresolved":
            hard.append(f"dataset identity not evidenced: {identity['how']}")
        if h and str(h.get("Num_total")) != str(o.get("num_total")):
            hard.append(f"reaction count differs: hub {h.get('Num_total')} vs official {o.get('num_total')} (subset)")
        hub_d3 = bool(conditions.get("d3")) or model.endswith("_D3")
        if conditions.get("d3") and not model.endswith("_D3"):
            hard.append("hub conditions say D3 on, but the model name carries no _D3 suffix")
        od3, od3_source, od3_kind = official_d3(name, o, official_conditions)
        row["official_d3"] = {"value": od3, "source": od3_source, "kind": od3_kind}
        if od3 is None:
            hard.append(f"official D3 condition not evidenced for {name!r} (page-level has_d3={page_d3!r} only; "
                        "supply --official-conditions with a cited source — the row then reads 'conditional', never 'yes')")
        elif od3 != hub_d3:
            hard.append(f"D3 differs: hub {'on' if hub_d3 else 'off'} vs official {'on' if od3 else 'off'} ({od3_source})")
        row["discrepancies"].extend(hard)
        row["caveats"] = ["official catbench version, calc_num and relaxation settings are not published by the endpoint; "
                          f"hub ran catbench {conditions.get('catbench_version')} calc_num={conditions.get('calc_num')}",
                          f"dataset identity: {identity['how']}"]
        if od3_kind == D3_DECLARED:
            row["caveats"].append(f"official D3: {od3_source}")
        # "yes" = hub row present, no hard discrepancy, AND every signal machine-read (name-matched id + published D3);
        # even then the label says content identity was not measured. Any operator declaration (--official-id or a cited
        # --official-conditions entry) makes the row "conditional": recorded, never verified, never "yes".
        clean = h is not None and not hard
        row["comparison"] = _comparison(row, identity, od3_kind, od3_source, clean)
        row["comparable"] = row["comparison"]["status"] == "comparable_by_name_match"
        row["conditional"] = row["comparison"]["status"] == "conditional_declared"
        rows.append(row)
    return {"dataset": conditions.get("dataset"), "leaderboard_id": rec.get("leaderboard_id"), "available": rec.get("available"),
            "leaderboard_id_source": rec.get("leaderboard_id_source"), "dataset_identity": identity,
            "official_conditions_evidence": sorted(official_conditions) if official_conditions else None,
            "official_conditions_note": ("operator-declared, cited, NOT verified by the hub" if official_conditions else None),
            "comparison_basis_note": COMPARISON_BASIS_NOTE, "verified_comparable_rows": 0,
            "reason": rec.get("reason"), "attempted": attempted, "official_last_updated": rec.get("official_last_updated"),
            "conditions": conditions, "result_guard": guard, "rows": rows, "compared_utc": utc_now(),
            "not_a_ranking": "rows are in the user's model order; unlike conditions are listed per row, never ranked"}


def render_markdown(cmp: dict) -> str:
    lines = [f"# CatBench comparison — {cmp.get('dataset')} (official id: {cmp.get('leaderboard_id')})", "",
             f"Official source: catbench.org, last_updated {cmp.get('official_last_updated')}; "
             f"fetched {[(a['url'], a['utc']) for a in cmp['attempted']]}",
             f"Conditions (hub): {json.dumps(cmp['conditions'])}",
             f"Dataset identity: {str((cmp.get('dataset_identity') or {}).get('status', 'unresolved')).upper()} — "
             f"{(cmp.get('dataset_identity') or {}).get('how')} (content identity NOT measured by the hub)",
             f"Comparison basis: {cmp.get('comparison_basis_note') or COMPARISON_BASIS_NOTE}",
             f"NOT a ranking — {cmp['not_a_ranking']}", ""]
    g = cmp.get("result_guard")
    if g:
        lines.append(f"result/ guard: {'OK' if g['ok'] else 'FAILED'} present={g['present']} extra={g['extra']}")
    else:
        lines.append("result/ guard: NOT CHECKED (no --result given)")
    lines.append("")
    lines += ["| model | hub MAE_total | hub MAE_normal | official name | official MAE_total | official MAE_normal | comparable (status + basis; never verified) | discrepancies / caveats |",
              "|---|---|---|---|---|---|---|---|"]
    for r in cmp["rows"]:
        h, o = r.get("hub") or {}, r.get("official")
        if isinstance(o, dict):
            ocell = (r["official_name"], _fmt(o.get("MAE_total")), _fmt(o.get("MAE_normal")))
        else:
            ocell = ("—", o or NO_VALUE, "—")
        notes = list(r["discrepancies"]) + [f"caveat: {c}" for c in r.get("caveats", [])]
        lines.append(f"| {r['model']} | {_fmt(h.get('MAE_total'))} | {_fmt(h.get('MAE_normal'))} | {ocell[0]} | {ocell[1]} | {ocell[2]} | "
                     f"{comparability_label(r)} | " + "; ".join(notes) + " |")
    return "\n".join(lines) + "\n"


def _fmt(v) -> str:
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "—" if v in (None, "") else str(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot"); s.add_argument("--dataset", required=True); s.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--snapshot", required=True); c.add_argument("--report", required=True)
    c.add_argument("--result", default=None, help="result/ dir: assert it holds only --models")
    c.add_argument("--models", required=True, help="comma-separated hub mlip_names, in the order to print")
    c.add_argument("--dataset", default=None); c.add_argument("--catbench-version", default=None)
    c.add_argument("--official-id", default=None,
                   help="operator DECLARATION that --dataset is this official leaderboard id (recorded, not verified): rows become "
                        "'conditional', never 'yes'; without it an alias-located page yields 'no' rows")
    c.add_argument("--official-conditions", default=None,
                   help="operator DECLARATION file {official name: {d3: bool, source: str}} for per-entry official D3; cited is not "
                        "verified, so rows relying on it read 'conditional', never 'yes'")
    c.add_argument("--d3", type=int, default=0); c.add_argument("--calc-num", type=int, default=None)
    c.add_argument("--out", default=None, help="directory for comparison.{json,md} (default: --snapshot)")
    args = ap.parse_args(argv)

    if args.cmd == "snapshot":
        rec = snapshot(args.dataset, Path(args.out))
        print(json.dumps({k: rec[k] for k in ("dataset_requested", "leaderboard_id", "leaderboard_id_source", "available", "reason", "attempts")}, indent=2))
        return 0
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    conditions = {"dataset": args.dataset, "official_id": args.official_id, "catbench_version": args.catbench_version,
                  "d3": bool(args.d3), "calc_num": args.calc_num}
    evidence = None
    if args.official_conditions:
        try:
            evidence = load_official_conditions(Path(args.official_conditions))
        except (OSError, ValueError) as exc:
            print(f"[stop] --official-conditions rejected: {exc}", file=sys.stderr)
            return 2
        conditions["official_conditions_file"] = str(Path(args.official_conditions).resolve())
        conditions["official_conditions_sha256"] = hashlib.sha256(Path(args.official_conditions).read_bytes()).hexdigest()
    cmp = compare(Path(args.snapshot), Path(args.report), models, conditions, Path(args.result) if args.result else None, evidence)
    out = Path(args.out or args.snapshot); out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(cmp, indent=2) + "\n")
    (out / "comparison.md").write_text(render_markdown(cmp))
    print(render_markdown(cmp))
    return 0 if (cmp["result_guard"] is None or cmp["result_guard"]["ok"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
