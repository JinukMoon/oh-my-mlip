"""docs/recipes.md is generated — these tests keep it honest.

Four properties, all GPU-free and hermetic:

  1. the committed doc is in sync with models.json (`--check`);
  2. every registry family has an upstream recipe and vice versa (no orphans),
     so adding a model to models.json cannot silently ship an empty recipe;
  3. the hard drift guard actually fires when a curated digest stops matching
     the registry — the guard that failed to exist is worse than no guard;
  4. the rendered doc is host-independent (no absolute clone path leaks in),
     which is what makes --check meaningful on a different machine.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

gen_recipes = pytest.importorskip("gen_recipes")
pytest.importorskip("yaml")

from upstream_recipes import UPSTREAM  # noqa: E402


@pytest.fixture(scope="module")
def models() -> dict:
    return json.loads((REPO / "models.json").read_text())


@pytest.fixture(scope="module")
def families(models: dict) -> list[str]:
    return [k for k in models if not k.startswith("_")]


def test_committed_doc_is_in_sync():
    """CI's real gate: the doc must equal what the generator produces now."""
    assert gen_recipes.main(["--check"]) == 0, (
        "docs/recipes.md is stale — run `python3 scripts/gen_recipes.py --write`"
    )


def test_no_orphan_recipes(families: list[str]):
    missing = sorted(set(families) - set(UPSTREAM))
    extra = sorted(set(UPSTREAM) - set(families))
    assert not missing, f"models.json families with no upstream recipe: {missing}"
    assert not extra, f"upstream_recipes entries for unknown families: {extra}"


def test_every_recipe_cites_its_sources(families: list[str]):
    for fam in families:
        u = UPSTREAM[fam]
        assert u.get("src", "").startswith("http"), f"{fam}: no install source URL"
        assert u.get("fetch_src", "").startswith("http"), f"{fam}: no weights source URL"
        assert u.get("pip"), f"{fam}: no install commands"
        assert u.get("fetch"), f"{fam}: no weight-acquisition commands"


def test_hard_drift_guard_fires_on_hardcoded_digest(models: dict):
    """EquFlash types its digests literally into the recipe. If the registry
    stops recording one, generation must FAIL rather than emit a sha256sum line
    that would reject the correct file."""
    fam = "EquFlash"
    literals = {h for ln in UPSTREAM[fam]["fetch"]
                for h in __import__("re").findall(r"\b[0-9a-f]{64}\b", ln)}
    assert literals, "this test assumes EquFlash hardcodes its digests"

    mutated = copy.deepcopy(models)
    for ver in mutated[fam]["versions"].values():
        ver["weights_sha256"] = "0" * 64

    with pytest.raises(SystemExit) as excinfo:
        gen_recipes.render(mutated)
    assert "DRIFT" in str(excinfo.value)


def test_placeholder_recipes_cannot_drift(models: dict):
    """Families that use the {sha} placeholder are sync-by-construction: change
    the registry and the rendered command changes with it (no guard needed)."""
    fam, variant = "Nequix", "Nequix-MP-1"
    assert any("{sha}" in ln for ln in UPSTREAM[fam]["fetch"])
    mutated = copy.deepcopy(models)
    mutated[fam]["versions"][variant]["weights_sha256"] = "a" * 64
    body = "\n".join(gen_recipes.weights_block(fam, mutated))
    assert "a" * 64 in body
    assert models[fam]["versions"][variant]["weights_sha256"] not in body


def test_soft_guard_surfaces_unquoted_fingerprints(models: dict):
    """A digest recorded in the registry but absent from the curated recipe text
    must still reach the doc — this is the hole that let the 2026-08-17
    ORB/GRACE/DeePMD fingerprints go unrendered."""
    fam = "ORB"  # by-name fetch: no curl line quotes the digest
    sha = models[fam]["versions"]["ORB-v3"].get("weights_sha256")
    if not sha:
        pytest.skip("ORB has no recorded fingerprint in this registry")
    body = "\n".join(gen_recipes.weights_block(fam, models))
    assert sha in body, "recorded fingerprint missing from the rendered recipe"


def test_rendered_doc_is_host_independent(models: dict):
    text = gen_recipes.render(models)
    assert str(REPO) not in text, (
        "the rendered doc embeds this clone's absolute path; it would differ on "
        "another machine and break --check"
    )


def test_rendered_doc_does_not_bake_in_the_host_gpu_arch(models: dict):
    """Compile paths must stay generic. resolve() substitutes the running host's
    arch, which would both mislead the reader and make --check fail on a machine
    with a different GPU."""
    import re as _re

    text = gen_recipes.render(models)
    leaked = sorted(set(_re.findall(r"/models/compiled/(sm\d+)/", text)))
    assert not leaked, f"host GPU arch baked into compile paths: {leaked}"
    assert "${OMM_ARCH}" in text, "arch placeholder missing from the compile paths"
