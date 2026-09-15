#!/usr/bin/env python3
"""evidence_report.py -- verdict layers over a fresh-root sweep ledger.

Reads ONLY the campaign ledger written by `setup_sweep.py --fresh-root`
(kind=inference rows from setup_verify, kind=finetune rows from ft_sweep,
`*` guard rows for snapshot/routes/isolation/sources/attribution/budget/
cleanup) and prints three layers:

  2. Ledger accounting   every required (env, kind, variant) has exactly one
                         terminal state, an evidence record and a
                         manifest_sha256; anything else is INCOMPLETE.
  3. Scientific completeness (--strict)
                         inference: every registered variant `passed`
                         (a degraded CPU pass is NOT a GPU proof and fails
                         strict unless --allow-degraded);
                         finetune: every variant of the DYNAMIC required set
                         `passed` -- baseline = models.json `finetune.status`
                         starting with `documented`; plus any not-supported /
                         code-excavation-needed candidate the per-run support
                         audit (--audit) marks `supported: true`. Printed as
                         `passed/required`; every excluded candidate is listed
                         with its citation.
                         `unsupported` is valid ONLY with a non-empty citation
                         (an uncited row is failed(uncited_unsupported)) and is
                         distinct from access-blocked / resource-blocked /
                         failed(*); any blocked or failed required row =>
                         INCOMPLETE, nonzero exit.
                         Rows tagged `adopted-regression` never count toward
                         fresh coverage; they are listed separately.

--bundle DIR writes evidence_bundle.md + evidence_bundle.json (table with
INCOMPLETE marks, guard rows, budget history vs estimate, cleanup records) --
it works on a paused/terminated campaign exactly as on a finished one (the
bundle exists in either case; the strict verdict is what changes).

Exit code: --strict => 0 iff complete; otherwise 0 whenever the ledger parsed.

Usage:
  python3 scripts/evidence_report.py --ledger .sweep/campaign_X/ledger.jsonl
  python3 scripts/evidence_report.py --ledger L --strict --audit audit.json --bundle .sweep/campaign_X/bundle
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup_common import resolve_home, utc_now  # noqa: E402

DOCUMENTED_PREFIX = "documented"
SOURCE_DERIVED = "source-derived (hub builder)"   # a hub builder for a path read from the installed source
CANDIDATE_STATUSES = ("not-supported", "code-excavation-needed")
ADOPTED_TAG = "adopted-regression"


def load_rows(ledger: Path) -> tuple[list[dict], int]:
    """(rows, malformed_line_count). A malformed line is counted, never
    silently dropped: a ledger that cannot be fully read cannot be complete."""
    rows = []
    malformed = 0
    for ln in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            malformed += 1
    return rows, malformed


def registry_variants(models: dict) -> list[dict]:
    out = []
    for family, spec in models.items():
        if family.startswith("_"):
            continue
        for version, v in (spec.get("versions") or {}).items():
            ft = (v or {}).get("finetune") or {}
            out.append({"env": spec.get("env"), "family": family, "variant": version,
                        "ft_status": ft.get("status") or "", "ft_evidence": list(ft.get("evidence") or []),
                        "ft_reason": ft.get("reason") or "", "gated": bool((v or {}).get("gated"))})
    return out


def load_audit(path: Path | None) -> dict:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("variants", data)


def required_sets(models: dict, audit: dict) -> dict:
    """Dynamic required sets Returns
    {inference: [...], finetune: [...], excluded: [{variant, status, citation}]}"""
    variants = registry_variants(models)
    inference = [v["variant"] for v in variants]
    finetune, excluded = [], []
    for v in variants:
        entry = audit.get(v["variant"]) or {}
        if v["ft_status"].startswith(DOCUMENTED_PREFIX) or v["ft_status"] == SOURCE_DERIVED:
            finetune.append(v["variant"])
        elif v["ft_status"] in CANDIDATE_STATUSES and entry.get("supported"):
            finetune.append(v["variant"])
        else:
            citation = entry.get("citation") or "; ".join(v["ft_evidence"]) or ""
            excluded.append({"variant": v["variant"], "env": v["env"], "status": v["ft_status"],
                             "citation": citation, "audited": bool(entry),
                             "reason": entry.get("reason") or v["ft_reason"]})
    return {"inference": inference, "finetune": finetune, "excluded": excluded, "variants": variants}


def terminal_rows(rows: list[dict]) -> tuple[dict, dict]:
    """{(kind, variant): last row with a state} for fresh rows, and the same
    for adopted-regression rows (kept apart)."""
    fresh: dict[tuple, dict] = {}
    adopted: dict[tuple, dict] = {}
    for r in rows:
        if not (r.get("kind") and r.get("variant") and r.get("state")):
            continue
        key = (r["kind"], r["variant"])
        if r.get("tag") == ADOPTED_TAG:
            adopted[key] = r
        else:
            fresh[key] = r
    return fresh, adopted


def judge_row(row: dict | None, *, required: bool, allow_degraded: bool) -> tuple[bool, str]:
    """(counts_as_passed, display_state) for one (kind, variant)."""
    if row is None:
        return False, "INCOMPLETE(missing)"
    state = row.get("state") or "INCOMPLETE(no_state)"
    if not row.get("manifest_sha256"):
        return False, f"INCOMPLETE(no_manifest) {state}"
    if row.get("evidence") is None and row.get("verdict") is None and state == "passed":
        return False, f"INCOMPLETE(no_evidence) {state}"
    if state == "unsupported":
        citation = (row.get("evidence") or {}).get("citation")
        if not citation:
            return False, "failed(uncited_unsupported)"
        if required:
            return False, "INCOMPLETE(unsupported_on_required_row)"
        return True, "unsupported"
    if state == "passed":
        ev = row.get("evidence") or {}
        degraded = bool(ev.get("degraded")) or bool((row.get("verdict") or {}).get("degraded"))
        if degraded and not allow_degraded:
            return False, "INCOMPLETE(degraded_cpu_pass)"
        # a fine-tune pass is scientific evidence only on cuda AND with the
        # verifier's own measured witness (ft_sweep records `gpu_witness`:
        # True / False / None). A cpu pass or an unwitnessed cuda pass is not
        # a GPU proof (display-only --allow-degraded may count it).
        if row.get("kind") == "finetune" and not allow_degraded:
            if ev.get("device") != "cuda":
                return False, f"INCOMPLETE(ft_not_cuda:{ev.get('device') or 'unknown'})"
            if ev.get("gpu_witness") is not True:
                return False, "INCOMPLETE(ft_gpu_unwitnessed)"
        return True, "passed" + (" (degraded)" if degraded else "")
    return False, f"INCOMPLETE {state}"


def evaluate(rows: list[dict], models: dict, audit: dict, *, allow_degraded: bool, malformed: int = 0) -> dict:
    req = required_sets(models, audit)
    fresh, adopted = terminal_rows(rows)
    variant_env = {v["variant"]: v["env"] for v in req["variants"]}
    table = []
    complete = True
    counts = {"inference": {"passed": 0, "required": len(req["inference"])},
              "finetune": {"passed": 0, "required": len(req["finetune"])}}
    for kind in ("inference", "finetune"):
        required = set(req[kind])
        for v in req["inference"]:
            row = fresh.get((kind, v))
            is_required = v in required
            ok, display = judge_row(row, required=is_required, allow_degraded=allow_degraded)
            if is_required:
                counts[kind]["passed"] += int(ok)
                complete &= ok
            elif row is not None and not ok:
                complete = False  # e.g. an uncited unsupported claim on an excluded row
            table.append({"env": variant_env.get(v), "kind": kind, "variant": v, "required": is_required,
                          "state": display, "raw_state": (row or {}).get("state"),
                          "manifest_sha256": (row or {}).get("manifest_sha256"),
                          "evidence": (row or {}).get("evidence"), "verdict": (row or {}).get("verdict"),
                          "seq": (row or {}).get("seq")})
    # An exclusion from the required set is only valid with a citation:
    # audit citation, models.json evidence, or the ledger's own cited
    # `unsupported` row. A candidate with none of those means the support
    # audit never happened for it -- the required set is not final => incomplete.
    uncited_exclusions = []
    for ex in req["excluded"]:
        if not ex["citation"]:
            row = fresh.get(("finetune", ex["variant"])) or {}
            if row.get("state") == "unsupported" and (row.get("evidence") or {}).get("citation"):
                ex["citation"] = row["evidence"]["citation"]
        if not ex["citation"]:
            uncited_exclusions.append(ex["variant"])
            complete = False
    guards = [r for r in rows if r.get("variant") == "*" and r.get("state")]
    failed_guards = [g for g in guards if not (g["state"] in ("passed", "cleaned", "retained"))]
    # a failed guard (snapshot, routes, isolation, source hash, attribution,
    # preserve, cleanup, disk floor ...) invalidates the cycle it belongs to:
    # a variant row that happened to pass inside a leaky or unverified root
    # is not evidence. Campaign-level refusals (concurrency/lock/plan) count too.
    campaign_failures = [r for r in rows if r.get("variant") is None and r.get("state")
                         and r.get("phase") in ("concurrency", "plan", "stop", "resolve")]
    budgets = [r for r in rows if r.get("phase") == "budget"]
    paused = any(r.get("phase") == "pause" for r in rows)
    terminated = any(r.get("phase") == "stop" for r in rows)
    campaign_ids = sorted({str(r.get("campaign_id")) for r in rows if r.get("campaign_id") is not None})
    manifests = sorted({r.get("manifest_sha256") for r in rows if r.get("manifest_sha256")})
    blockers = []
    # one campaign = one candidate tree: rows carrying different manifest
    # hashes (AAA inference + BBB fine-tune) are evidence about two different
    # trees and can never add up to one complete verdict.
    if len(manifests) > 1:
        blockers.append(f"rows carry {len(manifests)} different manifest_sha256 values (mixed candidate trees)")
    if failed_guards:
        blockers.append(f"{len(failed_guards)} failed guard row(s)")
    if campaign_failures:
        blockers.append(f"{len(campaign_failures)} campaign-level refusal(s)")
    if malformed:
        blockers.append(f"{malformed} malformed ledger line(s)")
    if len(campaign_ids) > 1:
        blockers.append(f"rows from {len(campaign_ids)} campaigns mixed in one ledger")
    if paused:
        blockers.append("campaign paused")
    if terminated:
        blockers.append("campaign terminated")
    return {"complete": complete and not blockers, "paused": paused, "terminated": terminated, "counts": counts,
            "excluded": req["excluded"], "uncited_exclusions": uncited_exclusions,
            "table": table, "guards": guards, "failed_guards": failed_guards,
            "campaign_failures": campaign_failures, "malformed_lines": malformed,
            "campaign_ids": campaign_ids, "blockers": blockers,
            "budgets": budgets, "adopted_regression": [dict(r, key=list(k)) for k, r in adopted.items()],
            "manifests": manifests}


def render_text(result: dict, ledger: Path) -> str:
    out = [f"evidence report -- ledger: {ledger}", f"  manifests: {', '.join(m[:12] for m in result['manifests']) or '-'}"]
    for row in result["table"]:
        mark = "" if row["required"] else " (not required)"
        out.append(f"  {row['env'] or '-':<12} {row['kind']:<9} {row['variant']:<24} {row['state']}{mark}")
    c = result["counts"]
    out.append(f"  inference {c['inference']['passed']}/{c['inference']['required']} passed")
    out.append(f"  finetune  {c['finetune']['passed']}/{c['finetune']['required']} passed"
               f" (required = documented baseline + audited-supported candidates)")
    for ex in result["excluded"]:
        cite = ex["citation"] or "NO CITATION (campaign audit missing -> required set not final)"
        out.append(f"    excluded {ex['variant']:<24} {ex['status']:<24} cite: {cite}")
    for g in result["failed_guards"]:
        out.append(f"  GUARD [{g.get('env')}] {g.get('phase')}: {g['state']}")
    for g in result["campaign_failures"]:
        out.append(f"  CAMPAIGN {g.get('phase')}: {g['state']}")
    for b in result["budgets"]:
        ev = b.get("evidence") or {}
        est = ev.get("peak_estimate_gib")
        act = ev.get("budget_actual") or {}
        out.append(f"  budget [{b.get('env')}] estimate {est if est is not None else 'unknown'} GiB "
                   f"vs actual peak {act.get('peak_gib')} GiB, wall {act.get('wall_seconds')} s")
    if result["adopted_regression"]:
        out.append(f"  adopted-regression rows (not fresh coverage): {len(result['adopted_regression'])}")
    if result["paused"]:
        out.append("  CAMPAIGN PAUSED -- unattributable change to a mandatory-hash target")
    for b in result["blockers"]:
        out.append(f"  BLOCKER: {b}")
    out.append("  VERDICT: " + ("COMPLETE" if result["complete"] else "INCOMPLETE"))
    return "\n".join(out)


def render_markdown(result: dict, ledger: Path) -> str:
    out = ["# oh-my-mlip fresh-cycle evidence bundle", "",
           f"- ledger: `{ledger}`", f"- generated: {utc_now()}",
           f"- verdict: **{'COMPLETE' if result['complete'] else 'INCOMPLETE'}**"
           + (" (campaign paused)" if result["paused"] else ""),
           f"- manifests: {', '.join(f'`{m}`' for m in result['manifests']) or '-'}", "",
           "| env | kind | variant | required | state | manifest |", "|---|---|---|---|---|---|"]
    for row in result["table"]:
        sha = (row["manifest_sha256"] or "")[:12]
        out.append(f"| {row['env'] or '-'} | {row['kind']} | `{row['variant']}` | {'yes' if row['required'] else 'no'} "
                   f"| {row['state']} | `{sha}` |")
    c = result["counts"]
    out += ["", f"- inference: {c['inference']['passed']}/{c['inference']['required']} passed",
            f"- finetune: {c['finetune']['passed']}/{c['finetune']['required']} passed (dynamic required set)"]
    if result["excluded"]:
        out += ["", "## Excluded from the fine-tune required set", "", "| variant | status | citation |", "|---|---|---|"]
        for ex in result["excluded"]:
            out.append(f"| `{ex['variant']}` | {ex['status']} | {ex['citation'] or 'NO CITATION'} |")
    if result["blockers"]:
        out += ["", "## Blockers", ""] + [f"- {b}" for b in result["blockers"]]
    if result["failed_guards"] or result["campaign_failures"]:
        out += ["", "## Guard failures", ""]
        for g in result["failed_guards"]:
            out.append(f"- [{g.get('env')}] {g.get('phase')}: `{g['state']}` -- {g.get('stderr_tail') or ''}")
        for g in result["campaign_failures"]:
            out.append(f"- campaign {g.get('phase')}: `{g['state']}` -- {g.get('stderr_tail') or ''}")
    if result["budgets"]:
        out += ["", "## Budget: estimate vs actual (GiB)", "",
                "| env | estimate peak GiB | actual peak GiB | wall s | aborted |", "|---|---|---|---|---|"]
        for b in result["budgets"]:
            ev = b.get("evidence") or {}
            est = ev.get("peak_estimate_gib")
            act = ev.get("budget_actual") or {}
            out.append(f"| {b.get('env')} | {est if est is not None else 'unknown'} | {act.get('peak_gib')} "
                       f"| {act.get('wall_seconds')} | {act.get('aborted') or '-'} |")
    cleanups = [g for g in result["guards"] if g.get("phase") in ("preserve", "cleanup", "post_cleanup")]
    if cleanups:
        out += ["", "## Preserve / cleanup", ""]
        for g in cleanups:
            ev = g.get("evidence") or {}
            extra = ""
            if g.get("phase") == "post_cleanup":
                extra = f" runtime_absent={ev.get('runtime_absent')} free_after={ev.get('free_gib_after_cleanup')} GiB (diagnostic)"
            elif g.get("phase") == "preserve" and ev:
                extra = f" {ev.get('file_count')} files, {ev.get('bytes')} bytes -> {ev.get('dest')}"
            out.append(f"- [{g.get('env')}] {g.get('phase')} `{g['state']}` {g.get('runtime_root') or ''}{extra}")
    if result["adopted_regression"]:
        out += ["", "## Adopted-path regression rows (NOT fresh coverage)", ""]
        for r in result["adopted_regression"]:
            out.append(f"- {r['key'][0]} `{r['key'][1]}`: {r.get('state')}")
    return "\n".join(out) + "\n"


def write_bundle(result: dict, ledger: Path, bundle_dir: Path) -> dict:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    md = bundle_dir / "evidence_bundle.md"
    js = bundle_dir / "evidence_bundle.json"
    md.write_text(render_markdown(result, ledger), encoding="utf-8")
    payload = {k: v for k, v in result.items() if k != "guards"}
    payload["ledger"] = str(ledger)
    payload["generated_utc"] = utc_now()
    js.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {"markdown": str(md), "json": str(js)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger", required=True, type=Path)
    ap.add_argument("--models", default=None, help="models.json (default: $OH_MY_MLIP_HOME/models.json)")
    ap.add_argument("--audit", default=None, help="FT support audit JSON for this run")
    ap.add_argument("--strict", action="store_true", help="exit nonzero unless every required row passed")
    ap.add_argument("--allow-degraded", action="store_true",
                    help="DISPLAY ONLY: count a degraded CPU pass / unwitnessed GPU as passed; refused with --strict")
    ap.add_argument("--bundle", default=None, help="write evidence_bundle.{md,json} into this dir")
    args = ap.parse_args(argv)
    if args.strict and args.allow_degraded:
        ap.error("--allow-degraded cannot satisfy --strict: a degraded or unwitnessed pass is not scientific proof")

    models_path = Path(args.models) if args.models else resolve_home() / "models.json"
    models = json.loads(models_path.read_text(encoding="utf-8"))
    rows, malformed = load_rows(args.ledger)
    result = evaluate(rows, models, load_audit(Path(args.audit) if args.audit else None),
                      allow_degraded=args.allow_degraded, malformed=malformed)
    print(render_text(result, args.ledger))
    if args.bundle:
        written = write_bundle(result, args.ledger, Path(args.bundle))
        print(f"  bundle: {written['markdown']}")
    if args.strict:
        return 0 if result["complete"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
