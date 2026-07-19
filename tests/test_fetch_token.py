"""Tests for oh_my_mlip.fetch._resolve_token — HF token precedence + leak-safety.

Precedence: HF_TOKEN > HF_TOKEN_PATH/HF-cache > OMM_HF_TOKEN_FILE. When the
oh-my-mlip convenience var (OMM_HF_TOKEN_FILE) is used, the child-visible env
must expose HF_TOKEN_PATH (a path) — never an inlined token literal — and no
hf_<...> token literal may leak into any returned/logged string.
"""
from __future__ import annotations

import re

from oh_my_mlip import fetch

# Assemble a fake token-shaped literal from fragments (so this test file itself
# contains no matching literal for verify_no_token).
FAKE_TOKEN = "hf" + "_" + "Z" * 24
LEAK_PAT = re.compile("hf" + "_" + "[A-Za-z0-9]" + "{20,}")


def test_hf_token_wins():
    env = {
        "HF_TOKEN": FAKE_TOKEN,
        "HF_TOKEN_PATH": "/x/path",
        "OMM_HF_TOKEN_FILE": "/y/file",
    }
    out = fetch._resolve_token(env)
    assert out["source"] == "HF_TOKEN"
    assert out["env"] == {}


def test_hf_token_path_beats_omm():
    env = {"HF_TOKEN_PATH": "/x/path", "OMM_HF_TOKEN_FILE": "/y/file"}
    out = fetch._resolve_token(env)
    assert out["source"] == "HF_TOKEN_PATH"
    assert out["env"] == {}


def test_omm_exports_hf_token_path_not_literal():
    token_file = "/path/outside/repo/token"
    env = {"OMM_HF_TOKEN_FILE": token_file}
    out = fetch._resolve_token(env)
    assert out["source"] == "OMM_HF_TOKEN_FILE"
    # Child-visible env exposes the PATH under the standard HF var...
    assert out["env"] == {"HF_TOKEN_PATH": token_file}
    # ...and exposes no HF_TOKEN literal.
    assert "HF_TOKEN" not in out["env"]


def test_none_when_nothing_set():
    out = fetch._resolve_token({})
    assert out["source"] == "none"
    assert out["env"] == {}


def test_no_token_literal_leaks_in_any_case():
    """No precedence case may surface an hf_<20+> literal in the returned dict
    (the resolver works with paths, never with token contents)."""
    cases = [
        {"OMM_HF_TOKEN_FILE": "/path/outside/repo/token"},
        {"HF_TOKEN_PATH": "/path/outside/repo/token"},
        {},
    ]
    for env in cases:
        out = fetch._resolve_token(env)
        blob = repr(out)
        assert not LEAK_PAT.search(blob), f"token literal leaked: {blob!r}"


def test_check_gated_exports_path_for_child(monkeypatch):
    """When a gated model is fetched via OMM_HF_TOKEN_FILE, _check_gated must
    export HF_TOKEN_PATH into the process env (for child loaders) without ever
    setting an HF_TOKEN literal."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    monkeypatch.setenv("OMM_HF_TOKEN_FILE", "/path/outside/repo/token")
    # Hide any real `huggingface-cli login` cache on the test host (the
    # hf_cache precedence step would otherwise win over OMM_HF_TOKEN_FILE).
    monkeypatch.setenv("HOME", "/nonexistent-test-home")

    fake_spec = {
        "gated": True,
        "license_url": "https://huggingface.co/facebook/UMA",
        "env": "uma",
    }
    monkeypatch.setattr(fetch.registry, "resolve", lambda model, version=None: fake_spec)

    spec = fetch._check_gated("UMA", None)
    assert spec is fake_spec
    import os

    assert os.environ.get("HF_TOKEN_PATH") == "/path/outside/repo/token"
    assert not os.environ.get("HF_TOKEN")


def test_check_gated_raises_actionable_when_no_token(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    monkeypatch.delenv("OMM_HF_TOKEN_FILE", raising=False)
    monkeypatch.setenv("HOME", "/nonexistent-test-home")  # hide real hf cache

    fake_spec = {
        "gated": True,
        "license_url": "https://huggingface.co/facebook/UMA",
        "env": "uma",
    }
    monkeypatch.setattr(fetch.registry, "resolve", lambda model, version=None: fake_spec)

    try:
        fetch._check_gated("UMA", None)
        assert False, "expected GatedError"
    except fetch.GatedError as exc:
        msg = str(exc)
        assert "docs/hf_token.md" in msg
        assert "https://huggingface.co/facebook/UMA" in msg
