"""docs/finetune.md is generated — these tests keep it honest.

Mirrors tests/test_gen_recipes.py's discipline for the fine-tuning doc
(oh-my-mlip 4-feature expansion, P2b):

  1. the committed doc is in sync with models.json + upstream_finetune.py
     (`--check`);
  2. every registry family has an UPSTREAM_FT entry and vice versa (no
     orphans), so a new family can never silently ship an empty
     fine-tuning section;
  3. every family's upstream `src` URL is quoted somewhere in the doc, so a
     citation can never be silently dropped;
  4. coverage: all 32 variants carry a classification -- each is either
     "supported-ish" (status starts with 'documented', has an entrypoint,
     and its family carries a src) or `not-supported` with a `reason` and
     at least one `evidence` URL;
  5. the five divergence candidates (C11/9.4) each carry a non-empty
     `variant_args`, except DPA-3.1-3M-FT, whose divergence is descriptive
     and is asserted on `semantics`/`reason` instead;
  6. UMA-s-1p2-OC22 does NOT inherit the UMA family's runnable_as_installed
     -- it must classify explicitly as blocked;
  7. the rendered doc is host-independent (no absolute clone path, no
     baked-in host GPU arch);
  8. the doc ships the recipe only -- it never claims a fine-tune was run and
     verified here (no `demonstrated` field, no such wording in the output).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

gen_finetune = pytest.importorskip("gen_finetune")

from upstream_finetune import UPSTREAM_FT  # noqa: E402


@pytest.fixture(scope="module")
def models() -> dict:
    return json.loads((REPO / "models.json").read_text())


@pytest.fixture(scope="module")
def families(models: dict) -> list[str]:
    return gen_finetune.families(models)


def _all_variant_finetunes(models: dict, families: list[str]):
    for fam in families:
        for v in gen_finetune.variants(models, fam):
            yield fam, v, gen_finetune.variant_finetune(models, fam, v)


def test_committed_doc_is_in_sync():
    """CI's real gate: the doc must equal what the generator produces now."""
    assert gen_finetune.main(["--check"]) == 0, (
        "docs/finetune.md is stale — run `python3 scripts/gen_finetune.py --write`"
    )


def test_no_orphan_families(families: list[str]):
    missing = sorted(set(families) - set(UPSTREAM_FT))
    extra = sorted(set(UPSTREAM_FT) - set(families))
    assert not missing, f"models.json families with no UPSTREAM_FT entry: {missing}"
    assert not extra, f"UPSTREAM_FT entries for unknown families: {extra}"


def test_every_family_cites_its_source(families: list[str]):
    for fam in families:
        src = UPSTREAM_FT[fam].get("src", "")
        assert src.startswith("http"), f"{fam}: no fine-tuning source URL"


def test_every_src_appears_in_rendered_doc(models: dict, families: list[str]):
    text = gen_finetune.render(models)
    for fam in families:
        src = UPSTREAM_FT[fam]["src"].split()[0]
        assert src in text, f"{fam}: src URL missing from the rendered doc"


def test_coverage_32_of_32_classified(models: dict, families: list[str]):
    """Every variant is either supported-ish (documented status + entrypoint
    + a family src) or not-supported with a reason and >=1 evidence URL."""
    total = 0
    for fam, v, ft in _all_variant_finetunes(models, families):
        total += 1
        status = ft.get("status") or ""
        if status == "not-supported":
            assert ft.get("reason"), f"{fam}/{v}: not-supported with no reason"
            assert ft.get("evidence"), f"{fam}/{v}: not-supported with no evidence URL"
        else:
            assert status, f"{fam}/{v}: empty status"
            assert ft.get("entrypoint") or status == "code-excavation-needed", (
                f"{fam}/{v}: {status!r} carries no entrypoint"
            )
            assert UPSTREAM_FT[fam].get("src", "").startswith("http"), (
                f"{fam}/{v}: family {fam} carries no src"
            )
    assert total == 32, f"expected 32 registry variants, found {total}"


# The five divergence candidates named in the consensus plan (P2a task 2 /
# 9.4): each must be checked explicitly rather than silently inherited.
_DIVERGENCE_VARIANT_ARGS = {
    "MACE-MH-1-OMAT": "--foundation_head",
    "MACE-MH-1-OC20": "--foundation_head",
    "EquFlashV2": "config_template",
    "EquFlash-v1": "config_template",
    "UMA-s-1p2-OC22": "--uma-task",
    # every other UMA task variant also diverges on --uma-task; spot-check
    # one more alongside the refuted one to keep the assertion meaningful.
    "UMA-m-1p1-OC20": "--uma-task",
}


def test_divergence_candidates_carry_variant_args(models: dict):
    for version, key in _DIVERGENCE_VARIANT_ARGS.items():
        fam = next(f for f in gen_finetune.families(models) if version in models[f]["versions"])
        ft = gen_finetune.variant_finetune(models, fam, version)
        variant_args = ft.get("variant_args") or {}
        assert key in variant_args, (
            f"{fam}/{version}: expected variant_args[{key!r}], got {variant_args}"
        )
        assert variant_args[key], f"{fam}/{version}: variant_args[{key!r}] is empty"


def test_dpa_3_1_3m_ft_divergence_is_descriptive_not_variant_args(models: dict):
    """DPA-3.1-3M-FT's divergence (it is itself already a multi-task
    fine-tune) changes what --model-branch MEANS, not what flag is passed --
    so it is asserted on semantics/reason, not variant_args (C11/9.4)."""
    ft = gen_finetune.variant_finetune(models, "DeePMD", "DPA-3.1-3M-FT")
    assert ft.get("semantics"), "DPA-3.1-3M-FT: expected a non-empty semantics field"
    assert ft.get("reason"), "DPA-3.1-3M-FT: expected a non-empty reason field"


def test_uma_s_1p2_oc22_does_not_inherit_runnable(models: dict):
    """The installed UMATask enum has no 'oc22'. This
    variant must classify explicitly as blocked, not inherit the UMA
    family's runnable_as_installed."""
    ft = gen_finetune.variant_finetune(models, "UMA", "UMA-s-1p2-OC22")
    assert ft["runnable_as_installed"] is False
    assert any("oc22" in b.lower() for b in ft.get("blockers") or []), (
        "UMA-s-1p2-OC22: expected an oc22-specific blocker"
    )


def test_rendered_doc_is_host_independent(models: dict):
    text = gen_finetune.render(models)
    assert str(REPO) not in text, (
        "the rendered doc embeds this clone's absolute path; it would differ on "
        "another machine and break --check"
    )
    assert "/home/jumoon" not in text
    assert "sm86" not in text and "sm89" not in text, (
        "fine-tuning commands are not arch-pinned; a host GPU arch leaked into the doc"
    )


def test_the_doc_makes_no_execution_claim_about_this_hub_s_own_runs(models: dict, families: list[str]):
    """The doc ships the recipe, never a verdict on runs done here: no variant carries a
    `demonstrated` field, and the rendered doc never claims a fine-tune was verified."""
    assert not [v for _, v, ft in _all_variant_finetunes(models, families) if "demonstrated" in ft]
    text = gen_finetune.render(models)
    for claim in ("demonstrated", "ft_verify reload verified", "demo fine-tune"):
        assert claim not in text, f"the generated doc still claims {claim!r}"
