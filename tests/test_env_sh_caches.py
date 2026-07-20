"""Tests for env.sh weight-cache semantics (framework-native by default).

Pins the 2026-07-20 design decision:
  * DEFAULT: env.sh must not redirect any download cache (no HF_HOME /
    FAIRCHEM_CACHE_DIR / TORCH_HOME / CACHED_PATH_CACHE_ROOT exports) — the
    always-on redirect forked users' existing caches (13 GB UMA re-download)
    and moved the HF login-token lookup, turning gated fetches anonymous;
  * a user's own pre-set HF_HOME is left untouched;
  * OMM_SHARED_CACHE_ROOT opts back in EXPLICITLY (shared /TGM-style hubs),
    rooting every cache under it and preserving the standard HF login via
    HF_TOKEN_PATH (a path export, never a token value).
"""
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _source_and_env(tmp_home: Path, extra: str = "") -> dict:
    script = (
        f"export HOME={tmp_home}; {extra} "
        f"source {REPO_ROOT}/env.sh >/dev/null 2>&1; "
        "printf 'HF_HOME=%s\\n' \"${HF_HOME:-UNSET}\"; "
        "printf 'FAIRCHEM_CACHE_DIR=%s\\n' \"${FAIRCHEM_CACHE_DIR:-UNSET}\"; "
        "printf 'TORCH_HOME=%s\\n' \"${TORCH_HOME:-UNSET}\"; "
        "printf 'CACHED_PATH_CACHE_ROOT=%s\\n' \"${CACHED_PATH_CACHE_ROOT:-UNSET}\"; "
        "printf 'HF_TOKEN_PATH=%s\\n' \"${HF_TOKEN_PATH:-UNSET}\"; "
    )
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin"},
    ).stdout
    return dict(line.split("=", 1) for line in out.strip().splitlines())


def test_default_no_cache_redirection(tmp_path):
    got = _source_and_env(tmp_path)
    assert got["HF_HOME"] == "UNSET"
    assert got["FAIRCHEM_CACHE_DIR"] == "UNSET"
    assert got["TORCH_HOME"] == "UNSET"
    assert got["CACHED_PATH_CACHE_ROOT"] == "UNSET"


def test_user_preset_hf_home_untouched(tmp_path):
    got = _source_and_env(tmp_path, extra="export HF_HOME=/my/own/hf;")
    assert got["HF_HOME"] == "/my/own/hf"


def test_shared_root_opt_in_redirects_all(tmp_path):
    got = _source_and_env(
        tmp_path, extra="export OMM_SHARED_CACHE_ROOT=/shared/omm;"
    )
    assert got["HF_HOME"] == "/shared/omm/hf"
    assert got["FAIRCHEM_CACHE_DIR"] == "/shared/omm/fairchem"
    assert got["TORCH_HOME"] == "/shared/omm/torch"
    assert got["CACHED_PATH_CACHE_ROOT"] == "/shared/omm/cached_path"


def test_shared_root_preserves_hf_login(tmp_path):
    token = tmp_path / ".cache" / "huggingface" / "token"
    token.parent.mkdir(parents=True)
    token.write_text("not-a-real-token\n")
    got = _source_and_env(
        tmp_path, extra="export OMM_SHARED_CACHE_ROOT=/shared/omm;"
    )
    assert got["HF_TOKEN_PATH"] == str(token)
    # without a login file, nothing is exported
    got2 = _source_and_env(tmp_path / "nohome", extra="export OMM_SHARED_CACHE_ROOT=/s;")
    assert got2["HF_TOKEN_PATH"] == "UNSET"
