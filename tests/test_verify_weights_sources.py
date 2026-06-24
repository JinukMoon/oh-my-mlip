"""Offline tests for scripts/verify_weights_sources.py.

Network is fully mocked, so no network/GPU/conda/torch/ase is needed. Pins the
status mapping:

  - url 200/3xx              -> pass
  - url 404                  -> fail
  - url timeout/5xx/conn (0) -> unknown
  - gated-hf (exists/401/403)-> token-required  (never pass, never fail)
  - by-name + card url 200   -> pass; 404 -> fail; none -> unknown
  - bundled                  -> n/a-bundled
  - exit code 0 when no fail, 1 when any fail
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_weights_sources as w  # noqa: E402


def _vinfo(**kw):
    base = {"weights": "on-demand-hf", "weights_fetch": "url",
            "weights_source": "https://example.com/w.bin",
            "weights_source_url": None}
    base.update(kw)
    return base


def _verify(vinfo, fetcher):
    return w._verify_version("FW", "v1", vinfo, fetcher)


def const_fetcher(code):
    return lambda method, url: code


# ── url ──────────────────────────────────────────────────────────────────────

def test_url_200_pass():
    r = _verify(_vinfo(weights_fetch="url"), const_fetcher(200))
    assert r.status == w.PASS


def test_url_302_pass():
    r = _verify(_vinfo(weights_fetch="url"), const_fetcher(302))
    assert r.status == w.PASS


def test_url_404_fail():
    r = _verify(_vinfo(weights_fetch="url"), const_fetcher(404))
    assert r.status == w.FAIL


def test_url_timeout_unknown():
    r = _verify(_vinfo(weights_fetch="url"), const_fetcher(0))
    assert r.status == w.UNKNOWN


def test_url_403_unknown_not_fail():
    r = _verify(_vinfo(weights_fetch="url"), const_fetcher(403))
    assert r.status == w.UNKNOWN


def test_url_prefers_source_url_over_source():
    seen = {}

    def fetcher(method, url):
        seen["url"] = url
        return 200

    _verify(_vinfo(weights_fetch="url", weights_source="https://a/x",
                   weights_source_url="https://b/y"), fetcher)
    assert seen["url"] == "https://b/y"


# ── gated-hf (never pass, never fail) ────────────────────────────────────────

def test_gated_hf_exists_token_required():
    r = _verify(_vinfo(weights="on-demand-hf", weights_fetch="gated-hf",
                       weights_source="facebook/UMA"), const_fetcher(200))
    assert r.status == w.TOKEN_REQUIRED


def test_gated_hf_401_token_required():
    r = _verify(_vinfo(weights="on-demand-hf", weights_fetch="gated-hf",
                       weights_source="facebook/UMA"), const_fetcher(401))
    assert r.status == w.TOKEN_REQUIRED


def test_gated_hf_403_token_required():
    r = _verify(_vinfo(weights="on-demand-hf", weights_fetch="gated-hf",
                       weights_source="facebook/UMA"), const_fetcher(403))
    assert r.status == w.TOKEN_REQUIRED


def test_gated_hf_never_fails_even_on_404():
    """Even an inconclusive/absent probe keeps gated weights token-required."""
    r = _verify(_vinfo(weights="on-demand-hf", weights_fetch="gated-hf",
                       weights_source="facebook/UMA"), const_fetcher(404))
    assert r.status == w.TOKEN_REQUIRED
    assert r.status not in (w.FAIL, w.PASS)


# ── by-name ──────────────────────────────────────────────────────────────────

def test_by_name_card_200_pass():
    r = _verify(_vinfo(weights="on-demand-hf", weights_fetch="by-name",
                       weights_source="some-name",
                       weights_source_url="https://card/x"), const_fetcher(200))
    assert r.status == w.PASS


def test_by_name_card_404_fail():
    r = _verify(_vinfo(weights="on-demand-hf", weights_fetch="by-name",
                       weights_source="some-name",
                       weights_source_url="https://card/x"), const_fetcher(404))
    assert r.status == w.FAIL


def test_by_name_no_card_unknown():
    r = _verify(_vinfo(weights="on-demand-hf", weights_fetch="by-name",
                       weights_source="some-name",
                       weights_source_url=None), const_fetcher(200))
    assert r.status == w.UNKNOWN


# ── bundled ──────────────────────────────────────────────────────────────────

def test_bundled_na():
    r = _verify(_vinfo(weights="bundled", weights_fetch="by-name",
                       weights_source="7net-mf-ompa", weights_source_url=None),
                const_fetcher(404))
    assert r.status == w.NA_BUNDLED


def test_bundled_na_even_with_card():
    r = _verify(_vinfo(weights="bundled", weights_fetch="by-name",
                       weights_source="7net-mf-ompa",
                       weights_source_url="https://card/x"), const_fetcher(200))
    assert r.status == w.NA_BUNDLED


# ── hf_repo_exists fallback ──────────────────────────────────────────────────

def test_hf_repo_exists_http_fallback_when_hub_absent(monkeypatch):
    """When huggingface_hub import fails, fall back to HTTP HEAD."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "huggingface_hub":
            raise ImportError("no hf hub")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    exists, code = w.hf_repo_exists("facebook/UMA", const_fetcher(200))
    assert exists is True and code == 200


def test_hf_repo_exists_401_is_not_exists_but_code_kept(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "huggingface_hub":
            raise ImportError("no hf hub")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    exists, code = w.hf_repo_exists("gated/repo", const_fetcher(401))
    assert exists is False and code == 401


# ── exit codes over the full registry ────────────────────────────────────────

def _load_models():
    import json
    return json.loads((REPO_ROOT / "models.json").read_text(encoding="utf-8"))


def test_verify_all_no_fail_with_permissive_fetcher():
    """All-200 fetcher => no model fails; gated stays token-required, never fail
    or pass; the real UMA gated entries are never failures."""
    results = w.verify_all(_load_models(), const_fetcher(200))
    assert all(r.status != w.FAIL for r in results)
    gated = [r for r in results if r.weights_fetch == "gated-hf"]
    assert gated, "expected at least one gated-hf model (UMA)"
    assert all(r.status == w.TOKEN_REQUIRED for r in gated)


def test_one_url_404_produces_a_fail_but_gated_safe():
    models = _load_models()
    results = w.verify_all(models, const_fetcher(404))
    # url-fetch models become fail on 404; gated must remain token-required.
    assert any(r.status == w.FAIL for r in results)
    gated = [r for r in results if r.weights_fetch == "gated-hf"]
    assert all(r.status == w.TOKEN_REQUIRED for r in gated)


def test_main_exit_zero(monkeypatch, capsys):
    monkeypatch.setattr(w, "real_fetcher", lambda method, url, **k: 200)
    rc = w.main([])
    assert rc == 0


def test_main_exit_one_on_fail(monkeypatch):
    monkeypatch.setattr(w, "real_fetcher", lambda method, url, **k: 404)
    rc = w.main([])
    assert rc == 1
