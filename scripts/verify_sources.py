#!/usr/bin/env python3
"""verify_sources.py — recipe wheel/source reachability lint (GPU-free).

Walks every `envs/*.yml` recipe, parses the ``pip:`` block, and verifies that
each install source is actually reachable / resolvable WITHOUT installing
anything (no torch, no ase, no GPU). It classifies each pip line:

  - ``torch==V+cuNNN``     -> the wheel index ``download.pytorch.org/whl/cuNNN/torch/``
                              is fetched and the version string is looked for.
  - ``--extra-index-url`` / ``--find-links`` -> the index URL is HEAD/GET reachable.
  - ``pkg==ver`` (PyPI)    -> ``pypi.org/pypi/<pkg>/json`` exists; the package is
                              flagged ``sdist-only`` when no wheel is published
                              (e.g. openequivariance, which compiles at install).
  - ``pkg @ git+https://...@<sha>`` -> the repo URL (not the sha) is HEAD reachable.
  - ``catbench==ver``      -> verified on PyPI like any other package.

A ``+local`` version segment (e.g. ``torch_scatter==2.1.2+pt29cu126``) marks a
package served from the torch extra-index, not PyPI; PyPI has no such version, so
those are reported as ``extra-index`` (resolved-by-index, not failed).

HARD FAIL conditions:
  - An unresolved local path or ``file://`` URL in an actual install line fails
    the recipe REGARDLESS of build status.
  - A 404 (package/wheel genuinely absent) fails the recipe — unless that recipe
    is ``candidate``, where unresolved lines are expected (reported, not failed).

Per-recipe status is one of ``ok | candidate | fail``. The script exits non-zero
ONLY when at least one recipe is ``fail``.

Network access is injectable: pass ``--mock`` to run fully offline with a static
in-process fetcher (every URL "exists"), or import :func:`verify_recipe` /
:func:`verify_all` and pass your own ``fetcher`` for testing. No heavy imports.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
ENVS_DIR = _REPO_ROOT / "envs"

PYTORCH_WHL_HOST = "https://download.pytorch.org/whl"

# A fetcher takes (method, url) and returns (status_code, body_text).
# status_code is an int (HTTP status, or 0 for a network/timeout/conn error).
# body_text may be "" for HEAD requests or errors.
Fetcher = Callable[[str, str], "tuple[int, str]"]

# Line-classification regexes.
_TORCH_RE = re.compile(r"^torch==(?P<ver>[0-9][^+\s]*)\+cu(?P<cu>\d+)$")
_GIT_RE = re.compile(r"@\s*git\+(?P<url>https?://\S+?)(?:@(?P<sha>[0-9A-Za-z._-]+))?$")
_PINNED_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<ver>[^+\s]+)(?P<local>\+\S+)?$"
)


@dataclass
class LineResult:
    raw: str
    kind: str  # torch | index | pypi | git | local | unknown
    status: str  # ok | fail | sdist-only | extra-index | expected | skipped
    detail: str = ""


@dataclass
class RecipeResult:
    name: str
    path: Path
    build_status: str  # clean | candidate | unknown
    status: str = "ok"  # ok | candidate | fail
    lines: list[LineResult] = field(default_factory=list)


# ── network ──────────────────────────────────────────────────────────────────

def real_fetcher(method: str, url: str, timeout: float = 15.0) -> "tuple[int, str]":
    """Default network fetcher using stdlib urllib. Returns (status, body).

    A network/timeout/connection error returns (0, "") so callers can treat it
    distinctly from an HTTP 404.
    """
    req = urllib.request.Request(url, method=method)
    req.add_header("User-Agent", "oh-my-mlip-verify-sources/1.0")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = ""
            if method == "GET":
                raw = resp.read()
                body = raw.decode("utf-8", errors="replace")
            return resp.status, body
    except urllib.error.HTTPError as exc:  # 4xx/5xx
        body = ""
        try:
            if method == "GET":
                body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return exc.code, body
    except Exception:
        return 0, ""


def mock_fetcher(method: str, url: str) -> "tuple[int, str]":
    """Offline fetcher: every URL "exists". PyPI JSON returns a body that always
    advertises at least one wheel so packages resolve as wheel-backed."""
    if "pypi.org/pypi/" in url and url.endswith("/json"):
        return 200, json.dumps(
            {"urls": [{"packagetype": "bdist_wheel", "filename": "x-1-py3-none-any.whl"}]}
        )
    if "download.pytorch.org/whl/" in url:
        # Empty body => the torch spot-check takes its "index reachable, can't
        # disprove the version" branch and reports ok (offline, no real listing).
        return 200, ""
    return 200, ""


# ── recipe parsing ───────────────────────────────────────────────────────────

def _read_build_status(text: str) -> str:
    """Extract ``# build_status: <x>`` from the recipe header; default unknown."""
    m = re.search(r"^#\s*build_status:\s*(\w+)", text, re.MULTILINE)
    return m.group(1).strip() if m else "unknown"


def _pip_lines(doc: dict) -> list[str]:
    """Return the raw strings inside the recipe's ``pip:`` list."""
    for dep in doc.get("dependencies", []) or []:
        if isinstance(dep, dict) and "pip" in dep:
            return [str(x) for x in (dep["pip"] or [])]
    return []


# ── per-line verification ────────────────────────────────────────────────────

def _pypi_url(pkg: str) -> str:
    return f"https://pypi.org/pypi/{pkg}/json"


def _has_wheel(body: str) -> bool:
    """True if a PyPI JSON body advertises at least one bdist_wheel."""
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        # Can't tell -> don't flag sdist-only on an unparsable body.
        return True
    urls = data.get("urls") or []
    if any(u.get("packagetype") == "bdist_wheel" for u in urls):
        return True
    releases = data.get("releases") or {}
    for files in releases.values():
        if any(f.get("packagetype") == "bdist_wheel" for f in (files or [])):
            return True
    return False


def _verify_line(line: str, fetcher: Fetcher) -> LineResult:
    s = line.strip()

    # HARD FAIL: file:// or an unresolved local path in an install line.
    if s.startswith("file://") or "file://" in s:
        return LineResult(s, "local", "fail", "file:// URL in install line")
    # bare local path install (e.g. "- /home/.../pkg" or "./pkg")
    if (s.startswith("/") or s.startswith("./") or s.startswith("../")) and "://" not in s:
        return LineResult(s, "local", "fail", "local path in install line")

    # index flags
    if s.startswith("--extra-index-url") or s.startswith("--find-links"):
        parts = s.split(None, 1)
        if len(parts) < 2:
            return LineResult(s, "index", "fail", "index flag without URL")
        url = parts[1].strip()
        if url.startswith("file://") or url.startswith("/"):
            return LineResult(s, "local", "fail", "local index path")
        code, _ = fetcher("GET", url)
        if code and code < 400:
            return LineResult(s, "index", "ok", f"index {code}")
        if code == 0:
            return LineResult(s, "index", "fail", "index unreachable")
        return LineResult(s, "index", "fail", f"index HTTP {code}")

    if s == "--no-deps" or s.startswith("--"):
        return LineResult(s, "unknown", "skipped", "pip flag")

    # torch==V+cuNNN -> pytorch wheel index
    m = _TORCH_RE.match(s)
    if m:
        cu = m.group("cu")
        ver = m.group("ver")
        url = f"{PYTORCH_WHL_HOST}/cu{cu}/torch/"
        code, body = fetcher("GET", url)
        if code == 0:
            return LineResult(s, "torch", "fail", f"wheel index cu{cu} unreachable")
        if code >= 400:
            return LineResult(s, "torch", "fail", f"wheel index cu{cu} HTTP {code}")
        # Spot-check: the version should appear somewhere on the index page.
        if ver in body or not body:
            return LineResult(s, "torch", "ok", f"cu{cu} index lists {ver}")
        return LineResult(s, "torch", "fail", f"{ver} absent from cu{cu} index")

    # pkg @ git+https://...@sha -> HEAD the repo URL (not the sha)
    gm = _GIT_RE.search(s)
    if gm:
        url = gm.group("url")
        if url.endswith(".git"):
            head = url[: -len(".git")]
        else:
            head = url
        code, _ = fetcher("HEAD", head)
        if code and code < 400:
            return LineResult(s, "git", "ok", f"repo {code}")
        if code == 0:
            return LineResult(s, "git", "fail", "git repo unreachable")
        return LineResult(s, "git", "fail", f"git repo HTTP {code}")

    # pkg==ver  (optionally +local)
    pm = _PINNED_RE.match(s)
    if pm:
        name = pm.group("name")
        local = pm.group("local")
        if local:
            # +local versions are served from the torch extra-index, not PyPI.
            return LineResult(s, "pypi", "extra-index", f"{name} served via extra-index")
        url = _pypi_url(name)
        code, body = fetcher("GET", url)
        if code == 0:
            return LineResult(s, "pypi", "fail", f"{name} PyPI unreachable")
        if code == 404:
            return LineResult(s, "pypi", "fail", f"{name} not on PyPI (404)")
        if code >= 400:
            return LineResult(s, "pypi", "fail", f"{name} PyPI HTTP {code}")
        if not _has_wheel(body):
            return LineResult(s, "pypi", "sdist-only", f"{name} is sdist-only")
        return LineResult(s, "pypi", "ok", f"{name} on PyPI (wheel)")

    return LineResult(s, "unknown", "skipped", "unrecognized pip line")


def verify_recipe(path: Path, fetcher: Fetcher) -> RecipeResult:
    text = path.read_text(encoding="utf-8")
    build_status = _read_build_status(text)
    doc = yaml.safe_load(text) or {}
    name = doc.get("name", path.stem)

    result = RecipeResult(name=name, path=path, build_status=build_status)

    is_candidate = build_status == "candidate"
    failed = False

    for raw in _pip_lines(doc):
        lr = _verify_line(raw, fetcher)
        result.lines.append(lr)

        if lr.status == "fail":
            # file:// in an install line ALWAYS fails, even for candidates.
            if lr.kind == "local":
                failed = True
            elif is_candidate:
                # Candidates may legitimately have unresolved lines; report,
                # don't fail.
                lr.status = "expected"
            else:
                failed = True

    if failed:
        result.status = "fail"
    elif is_candidate:
        result.status = "candidate"
    else:
        result.status = "ok"
    return result


def verify_all(envs_dir: Path, fetcher: Fetcher) -> list[RecipeResult]:
    return [
        verify_recipe(p, fetcher)
        for p in sorted(envs_dir.glob("*.yml"))
    ]


# ── reporting / CLI ──────────────────────────────────────────────────────────

def _print_report(results: list[RecipeResult]) -> None:
    width = max((len(r.name) for r in results), default=8)
    print(f"{'recipe'.ljust(width)}  status     details")
    print(f"{'-' * width}  ---------  -------")
    for r in results:
        notes: list[str] = []
        for lr in r.lines:
            if lr.status in {"fail", "sdist-only", "expected"}:
                notes.append(f"{lr.status}:{lr.detail}")
        note = "; ".join(notes) if notes else "all resolvable lines reachable"
        print(f"{r.name.ljust(width)}  {r.status.ljust(9)}  {note}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mock",
        action="store_true",
        help="run fully offline with a static fetcher (every URL exists)",
    )
    ap.add_argument(
        "--envs-dir",
        default=str(ENVS_DIR),
        help="directory of *.yml recipes (default: <repo>/envs)",
    )
    args = ap.parse_args(argv)

    fetcher: Fetcher = mock_fetcher if args.mock else real_fetcher
    results = verify_all(Path(args.envs_dir), fetcher)
    _print_report(results)

    n_fail = sum(1 for r in results if r.status == "fail")
    if n_fail:
        print(f"\n{n_fail} recipe(s) FAILED source verification.", file=sys.stderr)
        return 1
    print(
        f"\nAll {len(results)} recipe(s) OK "
        f"({sum(1 for r in results if r.status == 'candidate')} candidate)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
