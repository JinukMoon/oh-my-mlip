"""Regression guard for figshare download-URL normalization in fetch.py.

`www.figshare.com` / `figshare.com` return an HTTP 202 "preparing file" /
bot-block interstitial (content-length 0) for both `/files/<id>` and
`/ndownloader/files/<id>` paths that never resolves to 200 on some hosts. The
dedicated `ndownloader.figshare.com` subdomain 302-redirects straight to the
signed S3 object. `_normalise_download_url` must always rewrite figshare file
URLs to that subdomain so registry-driven weight fetches (dpa4, eqnorm,
equflash, alphanet, ...) actually download instead of saving a 0-byte file.

No GPU, conda, torch, or network required.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FETCH_PY = REPO_ROOT / "oh_my_mlip" / "fetch.py"


def _load():
    spec = importlib.util.spec_from_file_location("_omm_fetch", FETCH_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_www_ndownloader_path_rewritten_to_subdomain():
    mod = _load()
    out = mod._normalise_download_url("https://figshare.com/ndownloader/files/65469204")
    assert out == "https://ndownloader.figshare.com/files/65469204", out


def test_www_files_path_rewritten_to_subdomain():
    mod = _load()
    out = mod._normalise_download_url("https://figshare.com/files/55429685")
    assert out == "https://ndownloader.figshare.com/files/55429685", out


def test_www_host_variant_rewritten():
    mod = _load()
    out = mod._normalise_download_url("https://www.figshare.com/files/123")
    assert out == "https://ndownloader.figshare.com/files/123", out


def test_subdomain_url_left_untouched():
    mod = _load()
    url = "https://ndownloader.figshare.com/files/999"
    assert mod._normalise_download_url(url) == url


def test_non_figshare_url_left_untouched():
    mod = _load()
    url = "https://example.com/weights/model.pt"
    assert mod._normalise_download_url(url) == url


def test_subdomain_counts_as_direct_download():
    mod = _load()
    assert mod._looks_like_direct_download("https://ndownloader.figshare.com/files/65469204")
