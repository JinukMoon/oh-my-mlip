"""Tests for scripts/verify_no_token.py — the working-tree token-leak scanner.

Covers: (a) a clean tree passes; (b) an UNTRACKED file holding a fake token
trips the verifier (working-tree scan, not diff/tracked-only); (c) the
verifier's own source does not self-trip; plus redaction of the matched value.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "verify_no_token.py"

# Assemble a fake token-shaped literal from fragments so THIS test file does not
# itself contain a matching literal (which would make the repo scan fail).
FAKE_TOKEN = "hf" + "_" + "A" * 24


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_no_token", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clean_tree_passes(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.md").write_text("use `hf_...` as a placeholder\n")
    mod = _load_verifier()
    assert mod.scan(tmp_path) == []
    assert mod.main([str(tmp_path)]) == 0


def test_untracked_file_trips(tmp_path, capsys):
    # An UNTRACKED note (not in git) must still be caught: working-tree scan.
    (tmp_path / "notes.txt").write_text(f"my token is {FAKE_TOKEN} oops\n")
    mod = _load_verifier()
    hits = mod.scan(tmp_path)
    assert len(hits) == 1
    assert hits[0][0].name == "notes.txt"

    rc = mod.main([str(tmp_path)])
    assert rc != 0
    err = capsys.readouterr().err
    # The token VALUE must be redacted, not echoed.
    assert FAKE_TOKEN not in err
    assert "hf_<redacted>" in err
    assert "notes.txt" in err


def test_verifier_does_not_self_trip():
    """Scanning the real repo (which includes scripts/verify_no_token.py and the
    docs) must pass: the pattern is built from fragments, docs use `hf_...`."""
    mod = _load_verifier()
    hits = mod.scan(REPO_ROOT)
    assert hits == [], f"unexpected token-shaped literals: {hits}"
    assert mod.main([str(REPO_ROOT)]) == 0


def test_pattern_matches_real_shape():
    mod = _load_verifier()
    assert mod.PAT.search(FAKE_TOKEN)
    # The safe placeholder form must NOT match.
    assert not mod.PAT.search("hf_...")
