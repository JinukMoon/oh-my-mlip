"""Tests for scripts/evidence_report.py -- the Part 5 verdict layers.

Pins: the DYNAMIC fine-tune required set (documented baseline + audited
supported candidates, exclusions printed with citations); `unsupported`
valid only with a citation (strict rejects an uncited row); blocked/failed
required rows => INCOMPLETE nonzero; a degraded CPU pass is not GPU proof;
adopted-regression rows never count as fresh coverage; the bundle is written
for an incomplete/paused campaign too. GPU-free, pure ledger processing.
"""
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location("evidence_report", REPO_ROOT / "scripts" / "evidence_report.py")
er = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(er)

MODELS = {
    "_meta": {},
    "A": {"env": "a", "python": "x", "versions": {
        "A-1": {"finetune": {"status": "documented"}},
        "A-2": {"finetune": {"status": "documented (inherited)"}},
    }},
    "B": {"env": "b", "python": "x", "versions": {
        "B-ns": {"finetune": {"status": "not-supported", "evidence": ["https://up/README.md"]}},
        "B-exc": {"finetune": {"status": "code-excavation-needed", "reason": "no path"}},
    }},
}
SHA = "f" * 64


def _row(kind, variant, state, env="a", **extra):
    base = {"seq": 1, "campaign_id": "c", "env": env, "kind": kind, "variant": variant, "state": state,
            "manifest_sha256": SHA, "evidence": {"log": "x"}}
    if kind == "finetune" and state == "passed" and "evidence" not in extra:
        base["evidence"] = {"log": "x", "device": "cuda", "gpu_witness": True}
    base.update(extra)
    return base


def _eval(rows, audit=None, allow_degraded=False, malformed=0):
    return er.evaluate(rows, MODELS, audit or {}, allow_degraded=allow_degraded, malformed=malformed)


def _ledger(tmp_path, rows) -> Path:
    p = tmp_path / "ledger.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _complete_rows():
    return [
        _row("inference", "A-1", "passed"), _row("inference", "A-2", "passed"),
        _row("inference", "B-ns", "passed", env="b"), _row("inference", "B-exc", "passed", env="b"),
        _row("finetune", "A-1", "passed"), _row("finetune", "A-2", "passed"),
        _row("finetune", "B-ns", "unsupported", env="b", evidence={"citation": "https://up/README.md"}),
        _row("finetune", "B-exc", "unsupported", env="b", evidence={"citation": "docs/finetune.md:432"}),
    ]


def test_required_sets_are_dynamic_with_cited_exclusions():
    req = er.required_sets(MODELS, {})
    assert req["inference"] == ["A-1", "A-2", "B-ns", "B-exc"]
    assert req["finetune"] == ["A-1", "A-2"]
    excluded = {e["variant"]: e for e in req["excluded"]}
    assert excluded["B-ns"]["citation"] == "https://up/README.md"
    assert excluded["B-exc"]["citation"] == "" and excluded["B-exc"]["audited"] is False
    # a candidate the campaign audit finds supported JOINS the required set
    req = er.required_sets(MODELS, {"B-exc": {"supported": True, "citation": "up/train.py:1"}})
    assert req["finetune"] == ["A-1", "A-2", "B-exc"]
    assert [e["variant"] for e in req["excluded"]] == ["B-ns"]


def test_complete_campaign_is_strict_zero(tmp_path, capsys):
    ledger = _ledger(tmp_path, _complete_rows())
    models = tmp_path / "models.json"
    models.write_text(json.dumps(MODELS))
    rc = er.main(["--ledger", str(ledger), "--models", str(models), "--strict"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "inference 4/4 passed" in out and "finetune  2/2 passed" in out
    assert "excluded B-ns" in out and "https://up/README.md" in out
    assert "VERDICT: COMPLETE" in out


def test_strict_rejects_uncited_unsupported_row(tmp_path):
    rows = _complete_rows()
    rows[6] = _row("finetune", "B-ns", "unsupported", env="b", evidence={"citation": ""})
    result = er.evaluate(rows, MODELS, {}, allow_degraded=False)
    assert not result["complete"]
    states = {(t["kind"], t["variant"]): t["state"] for t in result["table"]}
    assert states[("finetune", "B-ns")] == "failed(uncited_unsupported)"


def test_blocked_or_failed_required_row_is_incomplete_nonzero(tmp_path, capsys):
    models = tmp_path / "models.json"
    models.write_text(json.dumps(MODELS))
    for bad in ("access-blocked", "resource-blocked", "failed(train)"):
        rows = _complete_rows()
        rows[4] = _row("finetune", "A-1", bad)
        ledger = _ledger(tmp_path, rows)
        rc = er.main(["--ledger", str(ledger), "--models", str(models), "--strict"])
        out = capsys.readouterr().out
        assert rc == 1, bad
        assert f"INCOMPLETE {bad}" in out and "finetune  1/2 passed" in out


def test_missing_row_and_missing_manifest_are_incomplete():
    rows = _complete_rows()[:-1]  # B-exc finetune row absent: not required, fine once the audit cites it
    audit = {"B-exc": {"supported": False, "citation": "docs/finetune.md:432"}}
    result = er.evaluate(rows, MODELS, audit, allow_degraded=False)
    assert result["complete"]
    rows = [r for r in _complete_rows() if not (r["kind"] == "inference" and r["variant"] == "A-2")]
    result = er.evaluate(rows, MODELS, {}, allow_degraded=False)
    assert not result["complete"]
    assert result["counts"]["inference"] == {"passed": 3, "required": 4}
    rows = _complete_rows()
    rows[0]["manifest_sha256"] = None
    result = er.evaluate(rows, MODELS, {}, allow_degraded=False)
    assert not result["complete"]
    assert any(t["state"].startswith("INCOMPLETE(no_manifest)") for t in result["table"])


def test_uncited_exclusion_means_required_set_not_final():
    """A code-excavation-needed candidate with no audit entry, no models.json
    evidence and no cited ledger row cannot be excluded silently."""
    rows = [r for r in _complete_rows() if not (r["kind"] == "finetune" and r["variant"] == "B-exc")]
    result = er.evaluate(rows, MODELS, {}, allow_degraded=False)
    assert not result["complete"] and result["uncited_exclusions"] == ["B-exc"]
    # the campaign audit supplies the citation -> complete again
    audit = {"B-exc": {"supported": False, "citation": "docs/finetune.md:432"}}
    result = er.evaluate(rows, MODELS, audit, allow_degraded=False)
    assert result["complete"] and result["uncited_exclusions"] == []
    # so does the ledger's own cited unsupported row (the default fixture)
    result = er.evaluate(_complete_rows(), MODELS, {}, allow_degraded=False)
    assert result["complete"]


def test_unsupported_on_a_required_row_never_passes():
    rows = _complete_rows()
    rows[4] = _row("finetune", "A-1", "unsupported", evidence={"citation": "made-up"})
    result = er.evaluate(rows, MODELS, {}, allow_degraded=False)
    assert not result["complete"]


def test_degraded_cpu_pass_is_not_gpu_proof_unless_allowed():
    rows = _complete_rows()
    rows[0] = _row("inference", "A-1", "passed", evidence={"degraded": True}, verdict={"pass": True, "degraded": True})
    assert not er.evaluate(rows, MODELS, {}, allow_degraded=False)["complete"]
    assert er.evaluate(rows, MODELS, {}, allow_degraded=True)["complete"]


def test_adopted_regression_rows_never_count_as_fresh_coverage():
    rows = [dict(r, tag="adopted-regression") for r in _complete_rows()]
    result = er.evaluate(rows, MODELS, {}, allow_degraded=False)
    assert not result["complete"]
    assert result["counts"]["inference"]["passed"] == 0
    assert len(result["adopted_regression"]) == 8


def test_bundle_written_for_paused_incomplete_campaign(tmp_path, capsys):
    rows = _complete_rows()[:3] + [
        {"seq": 9, "campaign_id": "c", "env": "b", "variant": "*", "phase": "attribution",
         "state": "failed(attribution)", "manifest_sha256": SHA, "stderr_tail": "changed"},
        {"seq": 10, "campaign_id": "c", "phase": "pause", "state": "paused"},
        {"seq": 11, "campaign_id": "c", "env": "a", "variant": "*", "phase": "budget", "state": "passed",
         "evidence": {"unit": "GiB", "peak_estimate_gib": 18.16,
                      "budget_actual": {"peak_gib": 6.7, "wall_seconds": 600.0, "aborted": None}}},
        {"seq": 12, "campaign_id": "c", "env": "a", "variant": "*", "phase": "post_cleanup", "state": "passed",
         "evidence": {"runtime_absent": True, "free_gib_after_cleanup": 11.9}},
    ]
    ledger = _ledger(tmp_path, rows)
    models = tmp_path / "models.json"
    models.write_text(json.dumps(MODELS))
    bundle = tmp_path / "bundle"
    rc = er.main(["--ledger", str(ledger), "--models", str(models), "--strict", "--bundle", str(bundle)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "CAMPAIGN PAUSED" in out and "GUARD [b] attribution: failed(attribution)" in out
    assert "estimate 18.16 GiB vs actual peak 6.7 GiB" in out
    assert "BLOCKER: 1 failed guard row(s)" in out and "BLOCKER: campaign paused" in out
    md = (bundle / "evidence_bundle.md").read_text()
    assert "**INCOMPLETE**" in md and "INCOMPLETE(missing)" in md
    assert "| a | 18.16 | 6.7 | 600.0 | - |" in md
    assert "`failed(attribution)`" in md and "runtime_absent=True" in md
    js = json.loads((bundle / "evidence_bundle.json").read_text())
    assert js["complete"] is False and js["paused"] is True
    assert js["counts"]["finetune"]["required"] == 2


def test_failed_guard_alone_makes_a_fully_passed_campaign_incomplete():
    """A cycle whose isolation/source/preserve/cleanup guard failed produced
    its variant rows inside an unverified root: they are not evidence."""
    rows = _complete_rows()
    assert _eval(rows)["complete"]
    for phase, state in (("isolation_post", "failed(isolation)"), ("sources", "failed(source_mutated)"),
                         ("preserve", "failed(preserve:copy_mismatch)"),
                         ("cleanup", "failed(cleanup:evidence_not_durable)"), ("abort", "resource-blocked")):
        guard = {"seq": 50, "campaign_id": "c", "env": "a", "variant": "*", "phase": phase, "state": state,
                 "manifest_sha256": SHA}
        result = _eval(rows + [guard])
        assert not result["complete"], phase
        assert result["failed_guards"][0]["phase"] == phase and "failed guard" in result["blockers"][0]
    # a campaign-level refusal (lock / foreign process / terminated) blocks too
    stop = {"seq": 51, "campaign_id": "c", "phase": "concurrency", "state": "failed(campaign_lock)"}
    assert not _eval(rows + [stop])["complete"]
    term = {"seq": 52, "campaign_id": "c", "phase": "stop", "state": "terminated", "env": "a"}
    assert _eval(rows + [term])["terminated"] and not _eval(rows + [term])["complete"]


def test_malformed_lines_and_mixed_campaigns_are_blockers(tmp_path):
    rows = _complete_rows()
    assert not _eval(rows, malformed=1)["complete"]
    foreign = dict(_row("inference", "A-1", "passed"), campaign_id="other")
    result = _eval(rows + [foreign])
    assert not result["complete"] and result["campaign_ids"] == ["c", "other"]
    # load_rows counts, never silently drops
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps(rows[0]) + "\n{broken json\n[1, 2]\n")
    loaded, malformed = er.load_rows(ledger)
    assert len(loaded) == 1 and malformed == 2


def test_finetune_pass_needs_cuda_and_a_true_gpu_witness_unless_allow_degraded():
    """A fine-tune `passed` row is scientific evidence only with device=cuda
    AND the verifier's measured gpu_witness True: an unwitnessed cuda pass, a
    False witness and a CPU pass are all INCOMPLETE under strict."""
    rows = _complete_rows()
    others = [r for r in rows if not (r["kind"] == "finetune" and r["variant"] == "A-1")]
    for ev, expect in (({"device": "cuda", "gpu_witness": None}, "INCOMPLETE(ft_gpu_unwitnessed)"),
                       ({"device": "cuda"}, "INCOMPLETE(ft_gpu_unwitnessed)"),
                       ({"device": "cuda", "gpu_witness": False}, "INCOMPLETE(ft_gpu_unwitnessed)"),
                       ({"device": "cuda", "gpu_witness": "true"}, "INCOMPLETE(ft_gpu_unwitnessed)"),
                       ({"device": "cpu", "gpu_witness": "n/a(cpu)"}, "INCOMPLETE(ft_not_cuda:cpu)"),
                       ({"gpu_witness": True}, "INCOMPLETE(ft_not_cuda:unknown)")):
        bad = _row("finetune", "A-1", "passed", evidence=ev)
        result = _eval(others + [bad])
        row = next(t for t in result["table"] if t["kind"] == "finetune" and t["variant"] == "A-1")
        assert row["state"] == expect and not result["complete"], ev
        assert result["counts"]["finetune"]["passed"] == 1
        # display-only --allow-degraded counts it; --strict refuses that flag (see below)
        assert _eval(others + [bad], allow_degraded=True)["complete"]
    good = _row("finetune", "A-1", "passed", evidence={"device": "cuda", "gpu_witness": True})
    assert _eval(others + [good])["complete"]


def test_mixed_manifest_hashes_are_a_blocker():
    """AAA inference + BBB fine-tune rows are evidence about two different
    candidate trees: every row may pass and the campaign is still INCOMPLETE."""
    rows = _complete_rows()
    for r in rows:
        if r["kind"] == "finetune":
            r["manifest_sha256"] = "b" * 64
    result = _eval(rows)
    assert not result["complete"] and result["manifests"] == ["b" * 64, SHA]
    assert any("different manifest_sha256" in b for b in result["blockers"])
    # every per-row judgement still passed: the blocker alone decides
    assert result["counts"] == {"inference": {"passed": 4, "required": 4}, "finetune": {"passed": 2, "required": 2}}
    assert _eval(_complete_rows())["complete"]


def test_strict_refuses_allow_degraded(tmp_path, capsys):
    ledger = _ledger(tmp_path, _complete_rows())
    models = tmp_path / "models.json"
    models.write_text(json.dumps(MODELS))
    with pytest.raises(SystemExit) as exc:
        er.main(["--ledger", str(ledger), "--models", str(models), "--strict", "--allow-degraded"])
    assert exc.value.code == 2
    assert "cannot satisfy --strict" in capsys.readouterr().err
    # without --strict it is a display option and the run still exits 0
    assert er.main(["--ledger", str(ledger), "--models", str(models), "--allow-degraded"]) == 0
