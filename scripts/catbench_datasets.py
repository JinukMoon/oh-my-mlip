#!/usr/bin/env python3
"""catbench_datasets.py — dataset recommendation from upstream-confirmed metadata.

Two layers, kept apart on purpose (recipes/catbench.md §1):

  * `UPSTREAM`  — description/size per featured dataset, quoted from the
    catbench README table of Zenodo-hosted datasets (catbench 1.1.4,
    README.md "Featured datasets (hosted on Zenodo)"). Cited per row.
  * `USER_HINT` — the science-to-dataset rules of thumb the USER supplied in
    the interview. They are labelled as such; they are never presented as
    upstream fact.

`--confirm` re-reads the live sources upstream itself uses (the Zenodo
concept record `get_benchmark` resolves, and the leaderboard `meta.json`).
It answers two SEPARATE questions with two separate flags, because one
boolean must not cover two claims:
  * `zenodo_size_confirmed`   — Zenodo lists `<name>_adsorption.json` at a
    size that agrees with the README figure (what the row's size/description
    asserts). Offline: false, and the recipe says "not live-confirmed".
  * `leaderboard_alias_verified` — ALWAYS false. The `leaderboard_id` in
    UPSTREAM is inferred from README reaction counts; nothing here (or on the
    endpoints) verifies that the Zenodo file and the leaderboard page are the
    same dataset. `confirmation.leaderboard.listed` only says the id string
    exists on meta.json.

Recommendation is keyword matching against the hints. No match ⇒ `ask`
(the question to put to the user), never a guess and never a fixed
"representative suite". No target given ⇒ `needs_target`.

Two more modes replace the hand-written `check_dataset.py` / `fetch_dataset.py`
bodies (recipes/catbench.md §1 A/B). Both import catbench lazily, so they run
under a catbench-bearing env interpreter (`<env>/bin/python`), and both stop
(exit 3) when `--catbench-version` names a different release than the env has:

  <env>/bin/python scripts/catbench_datasets.py --check raw_data/<tag>_adsorption.json [--record out.json]
  <env>/bin/python scripts/catbench_datasets.py --fetch <name> [--workdir <work>] [--catbench-version V]

`--check` validates the known adsorption input format (below) and reports the
file's sha256; `--fetch` calls upstream `get_benchmark(name)` (writes
`<work>/raw_data/<name>_adsorption.json`, never overwrites an existing file)
and writes `<work>/raw_data/<name>_adsorption.provenance.json` (name, catbench
version, sha256, size, whether the file pre-existed).

Usage:
  python3 scripts/catbench_datasets.py --list [--confirm] [--json]
  python3 scripts/catbench_datasets.py --target "CO on Pt-Ni alloys" [--json]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path

ZENODO_CONCEPT_URL = "https://zenodo.org/api/records/17157085"   # catbench zenodo.py:_ZENODO_CONCEPT_ID
LEADERBOARD_META_URL = "https://catbench.org/data/meta.json"
_CITATION = "catbench 1.1.4 README.md, 'Featured datasets (hosted on Zenodo)' table"
_SUFFIX = "_adsorption.json"

# name = Zenodo file stem (what get_benchmark(name) takes); leaderboard_id =
# the id catbench.org uses for the same dataset (None = not on the leaderboard
# meta.json as of the 2026-09-14 inspection). size_mb is the README figure.
UPSTREAM: dict[str, dict] = {
    "MamunHighT2019": {"size_mb": 95, "leaderboard_id": "MamunHighT2019",
                       "description": "45,130 small-molecule adsorptions on 2,035 bimetallic alloys"},
    "FG_dataset": {"size_mb": 9, "leaderboard_id": "FG",
                   "description": "2,651 C1-C10 organic molecules on transition metals"},
    "BM_dataset": {"size_mb": 0.3, "leaderboard_id": "BM",
                   "description": "32 industrial large molecules (biomass, polyurethane, plastics)"},
    "ComerGeneralized2024": {"size_mb": 1, "leaderboard_id": "ComerGeneralized2024",
                            "description": "325 adsorptions on metal oxide surfaces"},
    "GameNetOx_oxide": {"size_mb": 11, "leaderboard_id": "Game-Net-Ox",
                        "description": "987 adsorptions on metal-oxide surfaces"},
    "KHLOHC_origin": {"size_mb": 6, "leaderboard_id": "KHLOHC",
                      "description": "Liquid organic hydrogen carrier adsorption (fine-tuning)"},
    "OC20-Dense": {"size_mb": 397, "leaderboard_id": None,
                   "description": "65,073 dense adsorption configurations (Open Catalyst 2020)"},
}

# User-supplied rules of thumb (interview, 2026-09-13). keywords are matched
# case-insensitively against the user's target text.
USER_HINT: dict[str, dict] = {
    "MamunHighT2019": {"hint": "binary metals / bimetallic alloys with small adsorbates",
                       "keywords": ["alloy", "bimetallic", "binary metal", "small adsorbate", "small molecule"]},
    "FG_dataset": {"hint": "larger (functional-group) molecules on metal surfaces",
                   "keywords": ["large molecule", "organic", "functional group", "c1-c10", "hydrocarbon"]},
    "BM_dataset": {"hint": "larger molecular systems (industrial molecules)",
                   "keywords": ["biomass", "polymer", "plastic", "polyurethane", "industrial molecule", "big molecule"]},
    "ComerGeneralized2024": {"hint": "metal oxides",
                             "keywords": ["metal oxide", "oxide surface", "oxide"]},
    "OC20-Dense": {"hint": "adsorption configurations / relaxation search",
                   "keywords": ["configuration", "site search", "relaxation search", "dense", "oc20"]},
}


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _default_fetch(url: str, timeout: float = 60.0) -> tuple[int, bytes, str | None]:
    req = urllib.request.Request(url, headers={"User-Agent": "oh-my-mlip/catbench_datasets"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), None
    except Exception as exc:  # noqa: BLE001
        return getattr(exc, "code", 0) or 0, b"", f"{type(exc).__name__}: {exc}"


def catalogue() -> list[dict]:
    """Static rows: upstream description (cited) + user hint (labelled)."""
    rows = []
    for name, up in UPSTREAM.items():
        hint = USER_HINT.get(name)
        rows.append({
            "name": name, "get_benchmark_arg": name, **up, "citation": _CITATION,
            "user_hint": hint["hint"] if hint else None,
            "zenodo_size_confirmed": False, "leaderboard_alias_verified": False,
            "leaderboard_alias_note": ALIAS_NOTE, "confirmation": None,
        })
    return rows


ALIAS_NOTE = ("leaderboard_id is inferred from README reaction counts (2026-09-14 inspection); no source verifies that the "
              "Zenodo file and the leaderboard page hold the same dataset, so it is never marked verified")


def confirm(rows: list[dict], fetch=None) -> list[dict]:
    """Live confirmation against the sources upstream reads. Sets
    `zenodo_size_confirmed` only when Zenodo lists the file AND its size
    agrees with the README figure within 15%; attaches the leaderboard
    reaction_count when the id string exists there. `leaderboard_alias_verified`
    stays False: listing an id is not dataset identity. Unreachable sources
    are recorded, not raised."""
    fetch = fetch or _default_fetch
    when = utc_now()
    code, body, err = fetch(ZENODO_CONCEPT_URL)
    zfiles: dict[str, dict] = {}
    zerr = err
    if code == 200 and body:
        try:
            for f in json.loads(body).get("files", []):
                key = f.get("key", "")
                if key.endswith(_SUFFIX):
                    zfiles[key[: -len(_SUFFIX)]] = {"size": f.get("size"), "checksum": f.get("checksum")}
        except ValueError as exc:
            zerr = f"unparseable zenodo JSON: {exc}"
    else:
        zerr = zerr or f"http {code}"
    lcode, lbody, lerr = fetch(LEADERBOARD_META_URL)
    lcounts: dict[str, int] = {}
    if lcode == 200 and lbody:
        try:
            for d in json.loads(lbody).get("datasets", []):
                lcounts[d["id"]] = d.get("reaction_count")
        except (ValueError, KeyError) as exc:
            lerr = f"unparseable meta.json: {exc}"
    else:
        lerr = lerr or f"http {lcode}"

    for row in rows:
        z = zfiles.get(row["name"])
        size_ok = None
        if z and z.get("size") and row.get("size_mb"):
            size_ok = abs(z["size"] / 1e6 - row["size_mb"]) <= 0.15 * max(row["size_mb"], 1.0)
        lid = row.get("leaderboard_id")
        row["confirmation"] = {
            "utc": when,
            "zenodo": {"url": ZENODO_CONCEPT_URL, "listed": z is not None,
                       "size_bytes": z.get("size") if z else None, "size_matches_readme": size_ok,
                       "error": zerr},
            "leaderboard": {"url": LEADERBOARD_META_URL, "listed": lid in lcounts if lid else False,
                            "reaction_count": lcounts.get(lid) if lid else None, "error": lerr,
                            "note": "listed = the id string exists on meta.json; not evidence that it is this Zenodo dataset"},
        }
        row["zenodo_size_confirmed"] = bool(z) and size_ok is True
        row["leaderboard_alias_verified"] = False          # nothing checks the alias; never promoted
    return rows


def confirmation_words(row: dict) -> str:
    """The two claims, spelled out separately for every printed row."""
    size = "zenodo size confirmed" if row.get("zenodo_size_confirmed") else "NOT live-confirmed"
    return f"{size}; leaderboard alias {row.get('leaderboard_id')!r} NOT verified"


def recommend(target: str | None, rows: list[dict] | None = None) -> dict:
    """Keyword match against USER_HINT. Empty target ⇒ ask for it; no match ⇒
    ask, never guess. Matches are unordered candidates, not a ranking."""
    rows = rows if rows is not None else catalogue()
    by_name = {r["name"]: r for r in rows}
    if not target or not target.strip():
        return {"needs_target": True, "candidates": [],
                "ask": "What surfaces, adsorbates and chemistry do you want to test? "
                       "(No dataset is proposed until the scientific target is known.)"}
    text = target.lower()
    cands = []
    for name, hint in USER_HINT.items():
        hits = [k for k in hint["keywords"] if re.search(r"\b" + re.escape(k) + r"\b", text)]
        if hits:
            r = dict(by_name[name])
            r["matched_keywords"] = hits
            cands.append(r)
    if not cands:
        return {"needs_target": False, "candidates": [],
                "ask": (f"No user-supplied mapping covers the target {target!r}. Ask which dataset "
                        "(or which chemistry class) is intended; do not guess and do not propose a fixed suite.")}
    return {"needs_target": False, "candidates": cands, "ask": None,
            "note": "candidates come from user-supplied rules of thumb; each carries the upstream README citation, "
                    "a `zenodo_size_confirmed` flag from --confirm and `leaderboard_alias_verified` (always false)"}


# ── dataset file modes: --check / --fetch (need a catbench-bearing env) ───────
# Known adsorption input format, as upstream AdsorptionCalculation reads it
# (catbench 1.1.4 adsorption/calculation/calculation.py): one entry per
# reaction, `raw` holding the exact key "star" (slab), at least one "<X>star"
# key (adslab), optional "<G>gas" keys; every non-gas entry carries `atoms`,
# finite `energy_ref` and finite `stoi`; gas entries carry `atoms` and `stoi`.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def import_catbench(expected_version: str | None = None):
    """`(module, None)` or `(None, reason)`. The version guard mirrors the one
    inside every emitted job: a mismatch is reported, never upgraded here."""
    try:
        import catbench  # noqa: WPS433 (lazy on purpose: toolkit envs lack it)
    except ImportError as exc:
        return None, f"catbench is not importable here ({exc}); run this under a catbench-bearing env interpreter"
    have = getattr(catbench, "__version__", None)
    if expected_version is not None and have != expected_version:
        return None, f"approved catbench {expected_version} but this env has {have}"
    return catbench, None


def check_adsorption_json(path: Path, load=None) -> dict:
    """Validate `path` against the known adsorption input format. `load`
    defaults to upstream `load_catbench_json` (which rehydrates ASE Atoms);
    it is injectable so the format rules are testable without catbench.
    Returns a record with `ok`, counts, sha256 and every problem found."""
    path = Path(path)
    rec = {"path": str(path), "ok": False, "problems": [], "reactions": 0, "structures": 0,
           "gas_keys": [], "sha256": None, "size_bytes": None}
    if not path.is_file():
        rec["problems"].append(f"not a file: {path}")
        return rec
    rec["sha256"] = _sha256(path)
    rec["size_bytes"] = path.stat().st_size
    if load is None:
        from catbench.utils.data_utils import load_catbench_json
        load = load_catbench_json
    try:
        data = load(str(path))
    except Exception as exc:  # noqa: BLE001 — the loader's own message is the evidence
        rec["problems"].append(f"loader failed: {type(exc).__name__}: {exc}")
        return rec
    if not isinstance(data, dict) or not data:
        rec["problems"].append("top level is not a non-empty object of reactions")
        return rec
    gas_keys: set[str] = set()
    for rxn, entry in data.items():
        raw = entry.get("raw") if isinstance(entry, dict) else None
        if not isinstance(raw, dict) or not raw:
            rec["problems"].append(f"{rxn}: no 'raw' structures")
            continue
        rec["reactions"] += 1
        rec["structures"] += len(raw)
        if "star" not in raw:
            rec["problems"].append(f"{rxn}: no exact 'star' (slab) entry")
        if not any("star" in k and k != "star" for k in raw):
            rec["problems"].append(f"{rxn}: no '<X>star' (adslab) entry")
        for key, sv in raw.items():
            if not isinstance(sv, dict):
                rec["problems"].append(f"{rxn}/{key}: not an object")
                continue
            is_gas = "gas" in key
            if is_gas:
                gas_keys.add(key)
            atoms = sv.get("atoms")
            if atoms is None or not hasattr(atoms, "__len__"):
                rec["problems"].append(f"{rxn}/{key}: no 'atoms' structure")
            if not _finite_number(sv.get("stoi")):
                rec["problems"].append(f"{rxn}/{key}: 'stoi' missing or not a finite number")
            if not is_gas and not _finite_number(sv.get("energy_ref")):
                rec["problems"].append(f"{rxn}/{key}: 'energy_ref' missing or not a finite number")
    rec["gas_keys"] = sorted(gas_keys)
    rec["ok"] = not rec["problems"]
    return rec


def _zenodo_entry(name: str) -> dict | None:
    """Upstream's own Zenodo listing for `name` ({url, md5, size}), or None when
    `name` is not on the record or this catbench does not expose the listing."""
    try:
        from catbench.adsorption.data.zenodo import _zenodo_latest_files
    except ImportError:
        return None
    try:
        return _zenodo_latest_files().get(name)
    except Exception:  # the listing itself is a network call; the fallback is get_benchmark
        return None


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_leaderboard_copy(name: str, target: Path, entry: dict) -> bool:
    """Try catbench.org's gzip copy of `name` first: upstream serves the same
    datasets there (catbench zenodo.py, source 2), and it is several times
    smaller and far faster than Zenodo. It is accepted only when the unpacked
    file has exactly the size and md5 Zenodo lists; otherwise False, and the
    caller downloads from Zenodo."""
    import gzip
    import shutil
    if not entry.get("md5"):
        return False
    try:
        from catbench.adsorption.data.zenodo import _LEADERBOARD_BASE as base
    except ImportError:
        base = "https://catbench.org/benchmark"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from oh_my_mlip._download import download_resumable
    gz = target.with_name(target.name + ".gz.part")
    part = target.with_name(target.name + ".cdn.part")     # never the Zenodo resume file
    url = f"{base}/{name}.json.gz"
    print(f"Downloading {name} from the catbench.org copy ({url}), checked against Zenodo's md5", flush=True)
    try:
        download_resumable(url, gz, label=f"{name} (catbench.org)")
        with gzip.open(gz, "rb") as src, part.open("wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
    except Exception as exc:
        print(f"  catbench.org copy not usable ({exc.__class__.__name__}: {exc}); using Zenodo", flush=True)
        gz.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
        return False
    gz.unlink(missing_ok=True)
    size = entry.get("size")
    if (size and part.stat().st_size != size) or _md5(part) != entry["md5"]:
        print("  catbench.org copy differs from the Zenodo file (size or md5); using Zenodo", flush=True)
        part.unlink()
        return False
    print(f"MD5 verified against Zenodo: {entry['md5']}", flush=True)
    os.replace(part, target)
    target.with_name(target.name + ".part").unlink(missing_ok=True)   # an earlier Zenodo attempt is moot
    return True


def download_zenodo(name: str, target: Path, entry: dict) -> None:
    """Fetch a Zenodo benchmark file with the hub's resumable downloader and
    verify upstream's size and md5 before the file appears at `target`.

    catbench's own downloader restarts from zero and deletes the partial file
    when a connection drops, which on a slow link to Zenodo can fail a 95 MB
    file every time. Here the partial file `<target>.part` survives a failure
    and the next run resumes it."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from oh_my_mlip._download import download_resumable
    part = target.with_name(target.name + ".part")
    size = entry.get("size")
    print(f"Downloading {name} from Zenodo ({(size or 0) / 1e6:.1f} MB), resumable: {part}", flush=True)
    # Zenodo drops long transfers often; every attempt resumes the kept bytes,
    # so a generous count costs nothing but a few waits
    download_resumable(entry["url"], part, label=name, size=size, attempts=20)
    if size and part.stat().st_size != size:
        raise RuntimeError(f"{part} holds {part.stat().st_size} bytes, Zenodo lists {size}; "
                           f"delete it and fetch again")
    if entry.get("md5"):
        got = _md5(part)
        if got != entry["md5"]:
            part.unlink()
            raise RuntimeError(f"md5 of the downloaded {name} is {got}, Zenodo lists {entry['md5']}; "
                               f"the file was removed, fetch again")
        print(f"MD5 verified: {entry['md5']}", flush=True)
    os.replace(part, target)


def fetch_dataset(name: str, workdir: Path, get_benchmark=None, catbench_version: str | None = None,
                  zenodo_entry=None, leaderboard_copy=None) -> dict:
    """Fetch `name` through upstream `get_benchmark` into `<workdir>/raw_data/`
    and write the provenance record next to it. A file that already exists is
    never replaced (upstream's own default) and is reported as pre-existing;
    an existing provenance record is kept only if its sha256 still matches
    the file — otherwise the drift is an explicit error (never a silently
    stale record)."""
    workdir = Path(workdir).resolve()
    if not _NAME_RE.match(name or ""):
        raise ValueError(f"dataset name {name!r} is not a plain file-name stem")
    target = workdir / "raw_data" / f"{name}{_SUFFIX}"
    prov_path = workdir / "raw_data" / f"{name}_adsorption.provenance.json"
    pre_existing = target.is_file()
    real = get_benchmark is None
    if real:
        from catbench.adsorption.data.zenodo import get_benchmark as _gb
        get_benchmark = _gb
        zenodo_entry = _zenodo_entry
        leaderboard_copy = download_leaderboard_copy
    if leaderboard_copy is None:
        leaderboard_copy = lambda name, target, entry: False
    failure = None
    source = None
    if not pre_existing:
        workdir.mkdir(parents=True, exist_ok=True)
        entry = zenodo_entry(name) if zenodo_entry else None
        try:
            if entry and entry.get("url"):
                target.parent.mkdir(parents=True, exist_ok=True)
                if leaderboard_copy(name, target, entry):
                    source = "catbench.org copy, size and md5 equal to catbench's Zenodo listing"
                else:
                    source = "zenodo (resumable download; url, size and md5 from catbench's Zenodo listing)"
                    download_zenodo(name, target, entry)
            else:
                source = "catbench get_benchmark"
                cwd = os.getcwd()
                os.chdir(workdir)                       # upstream resolves raw_data/ against cwd
                try:
                    get_benchmark(name)
                finally:
                    os.chdir(cwd)
        except Exception as exc:  # network failures surface as a message, not a traceback
            failure = f"{exc.__class__.__name__}: {exc}"
    rec = {
        "dataset": name, "get_benchmark_arg": name, "path": str(target), "utc": utc_now(),
        "pre_existing": pre_existing, "fetched": (not pre_existing) and target.is_file(),
        "catbench_version": catbench_version,
        "sha256": _sha256(target) if target.is_file() else None,
        "size_bytes": target.stat().st_size if target.is_file() else None,
        "provenance": str(prov_path), "provenance_written": False,
    }
    if source:
        rec["source"] = source
    if not target.is_file():
        if failure:
            part = target.with_name(target.name + ".part")
            kept = (f" {part.stat().st_size / 1e6:.1f} MB are kept in {part}; running the same command again "
                    f"resumes from there." if part.is_file() else " Run the same command again to retry.")
            why = ("the connection kept dropping" if "did not finish after" in failure else failure)
            rec["error"] = f"downloading {name} failed ({why}).{kept}"
        else:
            rec["error"] = f"get_benchmark({name!r}) returned without writing {target}"
        return rec
    if prov_path.exists():
        try:
            recorded = json.loads(prov_path.read_text()).get("sha256")
        except (ValueError, AttributeError, OSError) as exc:
            recorded = None
            rec["error"] = f"existing provenance record {prov_path} is unreadable ({exc}); not rewritten"
        if "error" not in rec and recorded != rec["sha256"]:
            rec["provenance_drift"] = {"recorded_sha256": recorded, "current_sha256": rec["sha256"]}
            rec["error"] = (f"provenance drift: {prov_path} records sha256 {recorded}, but {target} now hashes to "
                            f"{rec['sha256']}; the file changed after it was recorded. Not rewritten — move the stale "
                            "record and the file aside (or use a fresh --workdir) and fetch again")
        elif "error" not in rec:
            rec["provenance_note"] = "existing provenance record kept (sha256 matches the file)"
    else:
        prov_path.write_text(json.dumps({k: v for k, v in rec.items() if k not in ("provenance", "provenance_written")},
                                        indent=2, sort_keys=True) + "\n")
        rec["provenance_written"] = True
    return rec


def _run_file_mode(args) -> int:
    """--check / --fetch: import catbench (guarded), then act. Exit 2 = wrong
    interpreter (no catbench), 3 = version mismatch or a failed check."""
    catbench, why = import_catbench(args.catbench_version)
    if catbench is None:
        print(f"[stop] {why}", file=sys.stderr)
        return 3 if "approved catbench" in why else 2
    have = getattr(catbench, "__version__", None)
    if args.check:
        rec = check_adsorption_json(Path(args.check))
        rec["catbench_version"] = have
        if args.record:
            Path(args.record).write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
        if args.json:
            print(json.dumps(rec, indent=2, sort_keys=True))
        else:
            print(f"{rec['path']}: {rec['reactions']} reactions, {rec['structures']} structures, "
                  f"gas keys {rec['gas_keys']}, sha256 {rec['sha256']}, catbench {have}")
            for p in rec["problems"]:
                print(f"  problem: {p}")
        return 0 if rec["ok"] else 3
    try:
        rec = fetch_dataset(args.fetch, Path(args.workdir), catbench_version=have)
    except ValueError as exc:
        print(f"[stop] {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(rec, indent=2, sort_keys=True))
    elif rec.get("error") and not rec.get("sha256"):
        pass                                     # the [stop] line below says what happened
    else:
        state = "pre-existing (not replaced)" if rec["pre_existing"] else ("fetched" if rec["fetched"] else "MISSING")
        print(f"{rec['path']}: {state}; sha256 {rec['sha256']}; catbench {have}; provenance {rec['provenance']}"
              + (" (written)" if rec["provenance_written"] else " (kept)"))
    if rec.get("error"):
        print(f"[stop] {rec['error']}", file=sys.stderr)
        return 3
    return 0 if rec.get("sha256") else 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the catalogue")
    ap.add_argument("--target", default=None, help="the user's scientific target (free text)")
    ap.add_argument("--confirm", action="store_true", help="live-confirm rows against Zenodo + leaderboard meta")
    ap.add_argument("--json", action="store_true")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", default=None, metavar="JSON",
                      help="validate an existing raw_data/<tag>_adsorption.json (needs catbench in this interpreter)")
    mode.add_argument("--fetch", default=None, metavar="NAME",
                      help="get_benchmark(NAME) into --workdir/raw_data/ + provenance record (needs catbench)")
    ap.add_argument("--workdir", default=".", help="benchmark workdir for --fetch (default: cwd)")
    ap.add_argument("--catbench-version", default=None, help="approved catbench version; stop if the env differs")
    ap.add_argument("--record", default=None, help="with --check: also write the check record to this JSON path")
    args = ap.parse_args(argv)

    if args.check or args.fetch:
        return _run_file_mode(args)

    rows = catalogue()
    if args.confirm:
        rows = confirm(rows)
    if args.target is not None or not args.list:
        out = recommend(args.target, rows)
    else:
        out = {"catalogue": rows}
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    if "catalogue" in out:
        for r in rows:
            print(f"{r['name']:<22} {r['size_mb']:>6} MB  {r['description']}  [{confirmation_words(r)}; {r['citation']}]")
            if r["user_hint"]:
                print(f"{'':<22}         user hint: {r['user_hint']}")
        return 0
    if out.get("ask"):
        print(f"[ask] {out['ask']}")
        return 3
    for c in out["candidates"]:
        print(f"candidate: {c['name']} — {c['description']} (matched {c['matched_keywords']}; "
              f"{confirmation_words(c)}; {c['citation']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
