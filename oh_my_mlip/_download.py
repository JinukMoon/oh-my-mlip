"""Resumable HTTP download: the one download loop the hub and its weight scripts share.

Pure stdlib and free of hub imports: the prepare/prestage scripts load this file
directly under a model env's interpreter, where the hub itself is not installed.

Weight hosts can be slow (a few KiB/s), stall without closing, or drop
connections that stay open for long (seen after ~10 minutes even on Hugging
Face). So every read has a timeout, reads return whatever has arrived, and an
interrupted transfer is continued with an HTTP Range request -- several times
within one call, and across calls because the partial file is left in place.
Progress goes to stderr: stdout may be a JSON-RPC channel.
"""
from __future__ import annotations

import sys
import time
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class DownloadError(RuntimeError):
    """The transfer did not complete; the partial file is kept for a resume."""


def download_resumable(url: str, part: Path, *, label: str, size: int | None = None,
                       timeout: float = 60, attempts: int = 5) -> None:
    """Grow `part` until it holds all of `url`.

    Complete means `size` bytes when the caller knows the size, otherwise the
    length the server declares. Raises DownloadError after `attempts` failed
    tries. 5xx and 429 answers are retried (honouring Retry-After up to two
    minutes); other HTTP errors, such as 403 or 404, propagate at once."""
    part.parent.mkdir(parents=True, exist_ok=True)
    cause = ""
    for attempt in range(1, attempts + 1):
        wait = min(10, 2 * attempt)
        try:
            if _once(url, part, label, size, timeout):
                return
            cause = "the connection closed before the end of the file"
        except HTTPError as exc:
            if exc.code == 416 and part.exists():
                part.unlink()  # the partial file no longer fits what the server holds
                cause = "the server rejected the resume offset"
                continue
            if exc.code != 429 and exc.code < 500:
                raise  # 401/403/404...: retrying cannot help
            # 5xx and 429 are the server being busy for a while (a gateway
            # timeout from a loaded weight host is common); wait and retry.
            cause = f"HTTP {exc.code} {exc.reason}"
            wait = max(wait, _retry_after(exc))
        except (OSError, HTTPException) as exc:  # timeouts, resets, IncompleteRead
            cause = f"{exc.__class__.__name__}: {exc}"
        if attempt < attempts:
            have = part.stat().st_size if part.exists() else 0
            print(f"[oh-my-mlip] {label}: interrupted at {have / 1e6:.1f} MB ({cause}); "
                  f"retrying in {wait:.0f} s (attempt {attempt + 1} of {attempts})",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
    have = part.stat().st_size if part.exists() else 0
    raise DownloadError(
        f"download of {url} did not finish after {attempts} attempts ({cause}); "
        f"{have} bytes are kept at {part} and the next attempt resumes them"
    )


def _retry_after(exc: HTTPError, cap: float = 120) -> float:
    """Seconds the server asked us to wait (Retry-After in seconds), capped."""
    value = exc.headers.get("Retry-After") if exc.headers is not None else None
    try:
        return min(cap, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _once(url: str, part: Path, label: str, size: int | None, timeout: float) -> bool:
    """One request appended to `part`; True when `part` is now complete."""
    have = part.stat().st_size if part.exists() else 0
    if size and have > size:
        part.unlink()
        have = 0
    if size and have == size:
        return True
    headers = {"User-Agent": "oh-my-mlip/0.1"}
    if have:
        headers["Range"] = f"bytes={have}-"
    with urlopen(Request(url, headers=headers), timeout=timeout) as response:
        if have and getattr(response, "status", None) != 206:
            have = 0  # the server ignored Range: start over
        declared = response.headers.get("Content-Length")
        total = size or (have + int(declared) if declared and declared.isdigit() else None)
        if have:
            print(f"[oh-my-mlip] {label}: resuming at {have / 1e6:.1f} MB", file=sys.stderr, flush=True)
        done, last = have, time.monotonic()
        with open(part, "ab" if have else "wb") as fh:
            while chunk := response.read1(1 << 16):
                fh.write(chunk)
                done += len(chunk)
                if time.monotonic() - last > 10:
                    shown = f" / {total / 1e6:.1f}" if total else ""
                    print(f"[oh-my-mlip] {label}: {done / 1e6:.1f}{shown} MB", file=sys.stderr, flush=True)
                    last = time.monotonic()
    got = part.stat().st_size
    if total is None:
        return True
    if got > total:
        part.unlink()  # not the file we expected; start over on the next attempt
        return False
    return got == total
