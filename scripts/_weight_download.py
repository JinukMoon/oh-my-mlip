"""Verified, resumable weight downloads for the prepare/prestage scripts.

Pure stdlib: these scripts run under a model env's interpreter, before anything
of the hub is importable there.

Why one helper: each script used to write straight to the final path and treat
"exists and non-empty" as done. http.client returns b"" rather than raising when
a connection closes early inside a sized read, so a cut-off download looked
complete and every rerun reused it -- a 900 MB stump of a 2.9 GB PET checkpoint
then failed at export forever. Here a file only becomes the final path after its
size (and hash, when the caller pins one) matches, and a partial file is kept
beside it so the transfer resumes instead of starting over.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path


def _load_download_core():
    # The hub package is not installed in a model env, so load the shared download
    # loop from its file rather than through `import oh_my_mlip` (whose __init__
    # would pull in the rest of the hub).
    path = Path(__file__).resolve().parent.parent / "oh_my_mlip" / "_download.py"
    spec = importlib.util.spec_from_file_location("_omm_download", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_download = _load_download_core()
DownloadError = _download.DownloadError
download_resumable = _download.download_resumable


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
             md5: str | None = None, label: str = "weights",
             min_rate: float | None = None) -> None:
    """Download `url` to `dest` only if it arrives whole.

    The bytes land in `.<name>.download` beside `dest` and are continued with an
    HTTP Range request -- retried within this call and resumed by the next run
    (oh_my_mlip/_download.py). The file is renamed onto `dest` only when its size
    matches `size` (or the server's declared length) and its sha256 / md5 match
    when given; a file that fails a hash is deleted, never resumed."""
    part = dest.parent / f".{dest.name}.download"
    _download.download_resumable(url, part, label=label, size=size, min_rate=min_rate)
    if part.stat().st_size == 0:
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


def download_first_available(sources: list[tuple[str, str]], dest: Path, *,
                             size: int | None = None, sha256: str | None = None,
                             md5: str | None = None, label: str = "weights",
                             min_rate: float | None = None) -> str:
    """Download `dest` from the first of `sources` ((name, url) pairs, official
    host first) that delivers it; returns the name of the source used.

    Every source is held to the same size and hash, so a fallback mirror can only
    ever supply the identical file. A source that fails after its retries, or
    (except the last) stays below `min_rate`, hands over to the next one, which
    resumes the same partial file with a Range request."""
    failures = []
    for index, (name, url) in enumerate(sources):
        last = index == len(sources) - 1
        try:
            download(url, dest, size=size, sha256=sha256, md5=md5, label=label,
                     min_rate=None if last else min_rate)
        except Exception as exc:  # noqa: BLE001 - every failure moves to the next source
            failures.append(f"{name}: {exc}")
            if last:
                raise DownloadError(f"{label}: no source delivered {dest.name}: "
                                    + "; ".join(failures)) from exc
            print(f"[oh-my-mlip] {label}: {name} failed ({exc}); trying {sources[index + 1][0]}",
                  file=sys.stderr, flush=True)
            continue
        print(f"[oh-my-mlip] {label}: {dest.name} downloaded from {name} ({url})",
              file=sys.stderr, flush=True)
        return name
    raise DownloadError(f"{label}: no download source given")
