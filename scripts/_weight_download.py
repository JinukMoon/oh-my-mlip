"""Verified, resumable weight downloads for the prepare/prestage scripts.

Pure stdlib: these scripts run under a model env's interpreter, before anything
of the hub is importable there.

Why one helper: each script used to write straight to the final path and treat
"exists and non-empty" as done. http.client returns b"" rather than raising when
a connection closes early inside a sized read, so a cut-off download looked
complete and every rerun reused it -- a 900 MB stump of a 2.9 GB PET checkpoint
then failed at export forever. Here a file only becomes the final path after its
size (and sha256, when the caller pins one) matches, and a partial file is
kept beside it so the next run resumes instead of starting over.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen


def sha256_of(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_complete(path: Path, size: int | None) -> bool:
    """Reuse test for an existing download: the recorded size when there is one
    (cheap on multi-GB files), otherwise merely non-empty."""
    if not path.is_file():
        return False
    return path.stat().st_size == size if size else path.stat().st_size > 0


def download(url: str, dest: Path, *, size: int | None = None, sha256: str | None = None,
             md5: str | None = None, label: str = "weights") -> None:
    """Download `url` to `dest` only if it arrives whole.

    The bytes land in `.<name>.download` beside `dest`; a rerun continues it with
    an HTTP Range request. The file is renamed onto `dest` only when its size
    matches `size` (or the server's declared length) and its sha256 / md5 match
    when given; a file that fails the hash is deleted, never resumed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.parent / f".{dest.name}.download"
    have = part.stat().st_size if part.exists() else 0
    if size and have > size:
        part.unlink()
        have = 0
    total = size
    if not (size and have == size):
        headers = {"User-Agent": "oh-my-mlip/0.1"}
        if have:
            headers["Range"] = f"bytes={have}-"
        with urlopen(Request(url, headers=headers), timeout=60) as resp:
            status = getattr(resp, "status", None)
            if have and status != 206:                     # the server ignored Range: start over
                have = 0
            if total is None:
                declared = resp.headers.get("Content-Length") if hasattr(resp, "headers") else None
                if declared and declared.isdigit():
                    total = have + int(declared)
            done, last = have, time.time()
            with open(part, "ab" if have else "wb") as fh:
                while chunk := resp.read1(1 << 16):
                    fh.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 5:
                        shown = f" / {total / 1e6:.1f}" if total else ""
                        print(f"[{label}] {done / 1e6:.1f}{shown} MB", flush=True)
                        last = time.time()
    got = part.stat().st_size
    if total and got != total:
        raise RuntimeError(f"{url}: received {got} of {total} bytes; the partial file is kept "
                           f"at {part} and the next run resumes it")
    if got == 0:
        part.unlink()
        raise RuntimeError(f"{url}: downloaded nothing")
    for algorithm, expected in (("sha256", sha256), ("md5", md5)):
        if not expected:
            continue
        digest = sha256_of(part, algorithm)
        if digest != expected.lower():
            part.unlink()
            raise RuntimeError(f"{url}: {algorithm} {digest} does not match the recorded {expected}")
    os.replace(part, dest)
