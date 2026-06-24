#!/usr/bin/env python3
"""verify_weights_sources.py — model weights reachability lint (GPU-free).

For every model version in ``models.json`` this reads ``weights_fetch`` /
``weights_source`` (and optional ``weights_source_url``) and probes whether the
weights are reachable WITHOUT downloading them, installing torch/ase, or needing
a GPU. Each version is classified:

  - ``url``        -> HTTP HEAD the ``weights_source`` URL (10s timeout, 1 retry):
                      200/3xx => ``pass``; 404 => ``fail``;
                      timeout / 5xx / connection error => ``unknown``.
  - ``gated-hf``   -> check the HF repo exists (huggingface_hub repo-exists if
                      installed, else HTTP HEAD ``huggingface.co/<repo>``):
                      exists OR 401/403 => ``token-required``
                      (NEVER ``fail``, NEVER ``pass`` — gated weights like UMA
                      must never be a build failure).
  - ``by-name``    -> if a ``weights_source_url`` card URL is present, HEAD it
                      (200/3xx => ``pass``, 404 => ``fail``, else ``unknown``);
                      otherwise ``unknown`` (name is resolved at runtime).
  - ``bundled``    -> ``n/a-bundled`` (weights ship inside the package).

A per-model table is printed. The script exits non-zero ONLY when at least one
version is ``fail``. ``token-required`` / ``unknown`` / ``n/a-bundled`` never
cause a non-zero exit.

``huggingface_hub`` is lazily imported (no hard new dependency); HEAD requests
use stdlib ``urllib``. Network is injectable via a ``fetcher`` for offline tests.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_JSON = _REPO_ROOT / "models.json"

HF_HOST = "https://huggingface.co"

# A fetcher takes (method, url) -> status_code. status_code conventions:
#   2xx/3xx/4xx  : the real HTTP status
#   0            : timeout / connection error / 5xx-after-retry (=> "unknown")
Fetcher = Callable[[str, str], int]

# Status constants.
PASS = "pass"
FAIL = "fail"
UNKNOWN = "unknown"
TOKEN_REQUIRED = "token-required"
NA_BUNDLED = "n/a-bundled"


@dataclass
class WeightResult:
    framework: str
    version: str
    weights_fetch: str
    status: str
    detail: str = ""


def real_fetcher(method: str, url: str, timeout: float = 10.0, retries: int = 1) -> int:
    """Default network fetcher (stdlib urllib). Returns an HTTP status code, or 0
    for timeout / connection error / 5xx (after one retry)."""
    req = urllib.request.Request(url, method=method)
    req.add_header("User-Agent", "oh-my-mlip-verify-weights/1.0")
    attempts = retries + 1
    last = 0
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status
        except urllib.error.HTTPError as exc:
            if exc.code >= 500:
                last = 0  # treat 5xx as retryable/unknown
                continue
            return exc.code
        except Exception:
            last = 0
            continue
    return last


def hf_repo_exists(repo: str, fetcher: Fetcher) -> "tuple[bool, int]":
    """Best-effort existence check for an HF repo.

    Tries ``huggingface_hub`` (lazy import) first; on any failure falls back to a
    plain HTTP HEAD of ``huggingface.co/<repo>``. Returns (exists, http_code)
    where http_code is the HEAD status (or 0 when the hub path answered / failed
    without an HTTP code).
    """
    try:  # lazy import — never a hard dependency
        from huggingface_hub import repo_exists  # type: ignore

        try:
            return bool(repo_exists(repo)), 0
        except Exception:
            # 401/403 from a gated repo can surface as an exception; fall through
            # to the HTTP HEAD which can distinguish gated (401/403) from absent.
            pass
    except Exception:
        pass

    code = fetcher("HEAD", f"{HF_HOST}/{repo}")
    return (code != 0 and code < 400), code


def _verify_version(
    framework: str,
    version: str,
    vinfo: dict,
    fetcher: Fetcher,
) -> WeightResult:
    fetch = vinfo.get("weights_fetch", "")
    source = vinfo.get("weights_source", "")
    source_url = vinfo.get("weights_source_url")

    def mk(status: str, detail: str = "") -> WeightResult:
        return WeightResult(framework, version, fetch, status, detail)

    # bundled weights (some entries carry weights == 'bundled' with by-name).
    if vinfo.get("weights") == "bundled" and fetch != "url":
        if fetch == "by-name" and source_url:
            code = fetcher("HEAD", source_url)
            if code and code < 400:
                return mk(NA_BUNDLED, f"bundled; card {code}")
            return mk(NA_BUNDLED, "bundled in package")
        return mk(NA_BUNDLED, "bundled in package")

    if fetch == "url":
        url = source_url or source
        if not (isinstance(url, str) and url.startswith(("http://", "https://"))):
            return mk(UNKNOWN, "no usable url")
        code = fetcher("HEAD", url)
        if code == 0:
            return mk(UNKNOWN, "timeout/5xx/conn-error")
        if code == 404:
            return mk(FAIL, "HTTP 404")
        if code < 400:
            return mk(PASS, f"HTTP {code}")
        # other 4xx (401/403 etc.) — reachable but access-restricted, not a 404.
        return mk(UNKNOWN, f"HTTP {code}")

    if fetch == "gated-hf":
        # gated weights must NEVER fail and NEVER pass.
        repo = source
        exists, code = hf_repo_exists(repo, fetcher)
        if exists or code in (401, 403):
            return mk(TOKEN_REQUIRED, f"gated HF repo {repo}")
        # Even if the existence probe is inconclusive, gated stays token-required.
        return mk(TOKEN_REQUIRED, f"gated HF repo {repo} (unverified)")

    if fetch == "by-name":
        if source_url and isinstance(source_url, str) and source_url.startswith("http"):
            code = fetcher("HEAD", source_url)
            if code == 0:
                return mk(UNKNOWN, "card timeout/conn-error")
            if code == 404:
                return mk(FAIL, "card HTTP 404")
            if code < 400:
                return mk(PASS, f"card HTTP {code}")
            return mk(UNKNOWN, f"card HTTP {code}")
        return mk(UNKNOWN, "by-name; resolved at runtime")

    return mk(UNKNOWN, f"unhandled weights_fetch={fetch!r}")


def verify_all(models: dict, fetcher: Fetcher) -> list[WeightResult]:
    results: list[WeightResult] = []
    for framework, info in models.items():
        if framework.startswith("_"):
            continue
        for version, vinfo in (info.get("versions") or {}).items():
            results.append(_verify_version(framework, version, vinfo, fetcher))
    return results


def _print_table(results: list[WeightResult]) -> None:
    fw_w = max((len(r.framework) for r in results), default=9)
    ver_w = max((len(r.version) for r in results), default=12)
    fetch_w = max((len(r.weights_fetch) for r in results), default=11)
    hdr = (
        f"{'framework'.ljust(fw_w)}  {'version'.ljust(ver_w)}  "
        f"{'fetch'.ljust(fetch_w)}  {'status'.ljust(14)}  detail"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r.framework.ljust(fw_w)}  {r.version.ljust(ver_w)}  "
            f"{r.weights_fetch.ljust(fetch_w)}  {r.status.ljust(14)}  {r.detail}"
        )


def _load_models(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models-json",
        default=str(MODELS_JSON),
        help="path to models.json (default: <repo>/models.json)",
    )
    args = ap.parse_args(argv)

    models = _load_models(Path(args.models_json))
    results = verify_all(models, real_fetcher)
    _print_table(results)

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    n_fail = counts.get(FAIL, 0)
    if n_fail:
        print(f"\n{n_fail} weight source(s) FAILED ({summary}).", file=sys.stderr)
        return 1
    print(f"\nNo failures ({summary}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
