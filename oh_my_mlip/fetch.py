"""oh_my_mlip.fetch — env tarball resolver (conda-pack distribution path).

Reads ``dist_manifest.json`` to map an env -> its conda-pack tarball on the
Hugging Face Hub, downloads it into the shared cache, runs ``conda-unpack``
once (guarded by a sentinel file), and returns the relocated interpreter path.

Honesty / safety behaviour:
  * gated-aware  : if the model is ``gated`` in models.json, require ``HF_TOKEN``
                   and print the ``license_url`` before any download. Gated
                   weights are NEVER redistributed by this repo.
  * CUDA probe   : after unpack, probe ``torch.cuda`` against the manifest's
                   ``min_driver_version``; on failure, print the EXACT
                   ``install.sh`` fallback command (not a raw traceback).
  * TODO marker  : manifest entries still carrying ``TODO-on-upload`` are
                   treated as not-yet-publishable (raises a clear error).

``huggingface_hub`` and ``torch`` are imported lazily INSIDE functions so this
module imports on a host without them (the unit tests rely on that).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from oh_my_mlip import registry

__all__ = [
    "FetchError",
    "GatedError",
    "cache_root",
    "ensure_weights",
    "env_install_dir",
    "fetch_env",
    "interpreter_path",
]

TODO_MARKER = "TODO-on-upload"
SENTINEL_NAME = ".oh-my-mlip-unpacked"
_MODEL_PATH_RE = re.compile(r"['\"]([^'\"]*/models/[^'\"]+)['\"]")
_DIRECT_URL_MARKERS = ("/resolve/", "/raw/", "/ndownloader/", "/api/records/")
_DIRECT_URL_SUFFIXES = (
    ".ckpt",
    ".model",
    ".nqx",
    ".pt",
    ".pt2",
    ".pth",
    ".tar",
    ".tar.bz2",
    ".tar.gz",
    ".tar.xz",
    ".tbz2",
    ".tgz",
    ".zip",
)


class FetchError(RuntimeError):
    """Raised on a missing/placeholder manifest entry or a failed unpack."""


class GatedError(FetchError):
    """Raised when a gated model is requested without HF_TOKEN."""


# ── model-weight materialization ─────────────────────────────────────────────
def ensure_weights(
    model: str,
    version: str | None = None,
    *,
    spec: dict | None = None,
) -> list[str]:
    """Ensure explicit local weight paths referenced by inference code exist.

    Name-based loaders such as MACE/ORB/Eqnorm that do not reference a local
    ``$OH_MY_MLIP_HOME/models/...`` path are left to their upstream cache logic.
    URL/gated-HF weights with explicit inference paths are downloaded into those
    paths before the calculator constructor runs.
    """
    resolved = spec if spec is not None else registry.resolve(model, version=version)

    # NequIP/Allegro inference points at per-GPU compiled .pt2 outputs, not the
    # downloadable source checkpoint zip. Those are produced by the compile path.
    if resolved.get("arch_pinned"):
        return []

    targets = _inference_weight_targets(resolved)
    if not targets:
        return []
    if all(_target_ready(path) for path in targets):
        return [str(path) for path in targets]

    fetch_mode = resolved.get("weights_fetch")
    if fetch_mode == "url":
        _materialize_url_weights(resolved, targets)
    elif fetch_mode == "gated-hf":
        _materialize_gated_hf_weights(resolved, targets)
    elif fetch_mode == "by-name":
        _materialize_by_name_weights(resolved, targets)
    else:
        raise FetchError(
            f"{resolved['model']}/{resolved['version']}: unsupported "
            f"weights_fetch={fetch_mode!r}"
        )

    missing = [str(path) for path in targets if not _target_ready(path)]
    if missing:
        raise FetchError(
            f"{resolved['model']}/{resolved['version']}: weight materialization "
            f"did not create expected path(s): {missing}"
        )
    return [str(path) for path in targets]


def _inference_weight_targets(spec: dict) -> list[Path]:
    """Extract absolute ``.../models/...`` paths from registry inference code."""
    out: list[Path] = []
    seen: set[str] = set()
    for line in spec.get("inference", []):
        for raw in _MODEL_PATH_RE.findall(line):
            path = Path(raw)
            key = str(path)
            if key not in seen:
                seen.add(key)
                out.append(path)
    return out


def _looks_like_dir_target(path: Path) -> bool:
    return path.suffix == ""


def _target_ready(path: Path) -> bool:
    if _looks_like_dir_target(path):
        return path.is_dir() and any(path.iterdir())
    return path.is_file() and path.stat().st_size > 0


def _target_root(targets: list[Path]) -> Path:
    parents = [path if _looks_like_dir_target(path) else path.parent for path in targets]
    return Path(os.path.commonpath([str(path) for path in parents]))


def _materialize_url_weights(spec: dict, targets: list[Path]) -> None:
    url = _select_download_url(spec)
    if not url:
        raise FetchError(
            f"{spec['model']}/{spec['version']}: weights_fetch='url' but no "
            "download URL is recorded"
        )
    root = _target_root(targets)
    root.mkdir(parents=True, exist_ok=True)
    local = _download_to_temp(url, root)
    try:
        _install_downloaded_artifact(local, targets, root)
    finally:
        local.unlink(missing_ok=True)


def _materialize_gated_hf_weights(spec: dict, targets: list[Path]) -> None:
    _check_gated(spec["model"], spec.get("version"))
    if len(targets) != 1 or _looks_like_dir_target(targets[0]):
        raise FetchError(
            f"{spec['model']}/{spec['version']}: gated-hf materialization needs "
            "one file target in inference"
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FetchError(
            "huggingface_hub is required to fetch gated HF weights "
            "(pip install huggingface_hub)"
        ) from exc

    target = targets[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id=spec["weights_source"],
        filename=_hf_filename_for_target(spec, target),
        cache_dir=str(cache_root()),
        token=os.environ.get("HF_TOKEN") or None,
    )
    _place_file(Path(local), target, copy=True)


def _materialize_by_name_weights(spec: dict, targets: list[Path]) -> None:
    command = spec.get("weights_fetch_command")
    if not command:
        # No explicit local target can be materialized generically. Upstream
        # loaders without local inference paths are intentionally handled by the
        # framework's own cache path and never reach this branch.
        raise FetchError(
            f"{spec['model']}/{spec['version']}: weights_fetch='by-name' has "
            "local inference path(s), but no weights_fetch_command is recorded"
        )
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise FetchError(
            f"{spec['model']}/{spec['version']}: weights_fetch_command must be "
            "a list of argv strings"
        )

    root = _target_root(targets)
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(spec.get("env_run", {}))
    env.setdefault("OH_MY_MLIP_HOME", registry.home())
    env["PATH"] = str(Path(sys.executable).resolve().parent) + os.pathsep + env.get(
        "PATH", ""
    )
    cache_env = spec.get("weights_cache_env")
    if cache_env:
        env[cache_env] = str(root)

    fmt = {
        "weights_source": spec.get("weights_source") or "",
        "target_root": str(root),
        "OH_MY_MLIP_HOME": registry.home(),
    }

    def _subst(part: str) -> str:
        # support both ${OH_MY_MLIP_HOME} env-style and {target_root} format-style tokens
        part = part.replace("${OH_MY_MLIP_HOME}", registry.home())
        return part.format(**fmt)

    cmd = [_subst(part) for part in command]
    # argv[0] of "python3"/"python" must resolve to THIS env's interpreter, not PATH
    if cmd and cmd[0] in ("python3", "python"):
        cmd[0] = sys.executable
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise FetchError(
            f"{spec['model']}/{spec['version']}: weight fetch command failed "
            f"with exit code {proc.returncode}: {' '.join(cmd)}\n{detail}"
        )


def _select_download_url(spec: dict) -> str | None:
    urls = [
        value
        for value in (spec.get("weights_source_url"), spec.get("weights_source"))
        if isinstance(value, str) and value.startswith(("http://", "https://"))
    ]
    if not urls:
        return None
    for url in urls:
        if _looks_like_direct_download(url):
            return _normalise_download_url(url)
    return _normalise_download_url(urls[0])


def _looks_like_direct_download(url: str) -> bool:
    lowered = url.lower()
    parsed = urlparse(lowered)
    if parsed.netloc == "ndownloader.figshare.com":
        return True
    return any(marker in lowered for marker in _DIRECT_URL_MARKERS) or lowered.endswith(
        _DIRECT_URL_SUFFIXES
    )


def _normalise_download_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    # www.figshare.com (and bare figshare.com) return an HTTP 202 "preparing
    # file" / bot-block interstitial (content-length 0) for BOTH /files/<id> and
    # /ndownloader/files/<id> that never resolves to 200 on some hosts. The
    # dedicated ndownloader.figshare.com subdomain instead 302-redirects straight
    # to the signed S3 object, so always route figshare file downloads through it.
    if host in ("figshare.com", "www.figshare.com"):
        match = re.search(r"/files/(\d+)", parsed.path)
        if match:
            return f"{parsed.scheme}://ndownloader.figshare.com/files/{match.group(1)}"
    return url


def _download_to_temp(url: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    parsed_name = Path(urlparse(url).path).name or "weights"
    tmp = directory / f".{parsed_name}.download"
    request = Request(url, headers={"User-Agent": "oh-my-mlip/0.1"})
    with urlopen(request) as response, open(tmp, "wb") as fh:
        shutil.copyfileobj(response, fh)
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise FetchError(f"downloaded empty weight artifact from {url}")
    return tmp


def _install_downloaded_artifact(local: Path, targets: list[Path], root: Path) -> None:
    file_targets = [path for path in targets if not _looks_like_dir_target(path)]
    # A PyTorch checkpoint (.pt/.pth) is itself a zip container (data.pkl + data/),
    # but it must be placed verbatim at the single-file target, NOT extracted. Only
    # treat the artifact as an archive-to-unpack when it is a real archive AND we are
    # not simply moving one checkpoint file into one file target.
    if _is_archive(local) and not (file_targets and _is_torch_checkpoint(local)):
        _extract_archive(local, root)
        _fill_targets_from_tree(targets, root)
        return
    if file_targets:
        _place_file(local, file_targets[0], copy=False)


def _is_torch_checkpoint(path: Path) -> bool:
    """A torch.save() artifact is a zip whose entries live under a top dir and
    include a ``data.pkl`` pickle (and usually a ``data/`` tensor blob dir). Such
    files load via torch.load and must never be unpacked onto disk."""
    import zipfile

    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except zipfile.BadZipFile:
        return False
    return any(n.endswith("data.pkl") for n in names)


def _is_archive(path: Path) -> bool:
    import tarfile
    import zipfile

    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def _extract_archive(path: Path, dest: Path) -> None:
    import tarfile
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for member in zf.infolist():
                target = (dest / member.filename).resolve()
                if not str(target).startswith(str(dest.resolve())):
                    raise FetchError(f"unsafe archive member: {member.filename}")
            zf.extractall(dest)
        return

    with tarfile.open(path, "r:*") as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # pragma: no cover - older python
            tf.extractall(dest)


def _fill_targets_from_tree(targets: list[Path], root: Path) -> None:
    for target in targets:
        if _target_ready(target):
            continue
        matches = [path for path in root.rglob(target.name) if path != target]
        if not matches:
            continue
        source = matches[0]
        if source.is_dir():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        elif source.is_file() and not _looks_like_dir_target(target):
            _place_file(source, target, copy=True)


def _place_file(source: Path, target: Path, *, copy: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(source, target)
    else:
        shutil.move(str(source), str(target))


def _hf_filename_for_target(spec: dict, target: Path) -> str:
    source_url = spec.get("weights_source_url")
    if isinstance(source_url, str) and "/resolve/" in source_url:
        return source_url.split("/resolve/", 1)[1].split("/", 1)[1]
    return target.name


# ── HF token resolution ──────────────────────────────────────────────────────
def _resolve_token(env: dict | None = None) -> dict:
    """Resolve how the user's Hugging Face token is made available, WITHOUT ever
    reading or returning the token's value.

    Precedence (most-explicit wins):
      1. ``HF_TOKEN``          — token already in the environment (standard HF).
      2. ``HF_TOKEN_PATH`` / HF cache — standard ``huggingface_hub`` resolution
         (a token file path, or ``huggingface-cli login`` having written one).
      3. ``OMM_HF_TOKEN_FILE`` — oh-my-mlip convenience: a path to a file holding
         the token. We DO NOT read it; instead we export it as ``HF_TOKEN_PATH``
         so third-party loaders (``huggingface_hub``) resolve it the standard
         way for child processes.

    Returns a dict describing the resolution that is safe to log:
      ``{"source": <which>, "env": <child-visible env additions>}``.
    The ``env`` mapping NEVER contains a token literal — at most a path under
    ``HF_TOKEN_PATH``. ``source`` is one of ``"HF_TOKEN"``,
    ``"HF_TOKEN_PATH"``, ``"OMM_HF_TOKEN_FILE"``, or ``"none"``.
    """
    src = os.environ if env is None else env
    out: dict = {"source": "none", "env": {}}

    if src.get("HF_TOKEN"):
        out["source"] = "HF_TOKEN"
        return out

    if src.get("HF_TOKEN_PATH"):
        out["source"] = "HF_TOKEN_PATH"
        return out

    omm = src.get("OMM_HF_TOKEN_FILE")
    if omm:
        # Re-export as the standard HF variable for child processes; do NOT read
        # the file contents (no token literal ever enters this process's state).
        out["source"] = "OMM_HF_TOKEN_FILE"
        out["env"] = {"HF_TOKEN_PATH": omm}
        return out

    return out


# ── cache / layout ───────────────────────────────────────────────────────────
def cache_root() -> Path:
    """Shared download+unpack cache: $OH_MY_MLIP_HOME/cache or ~/.cache/oh-my-mlip."""
    home = os.environ.get("OH_MY_MLIP_HOME")
    if home:
        root = Path(home) / "cache"
    else:
        root = Path.home() / ".cache" / "oh-my-mlip"
    return root


def env_install_dir(env: str) -> Path:
    """Where the relocated env lives after unpack: $OH_MY_MLIP_HOME/envs/<env>."""
    return Path(registry.home()) / "envs" / env


def interpreter_path(env: str) -> Path:
    return env_install_dir(env) / "bin" / "python"


# ── manifest lookup ──────────────────────────────────────────────────────────
def _manifest_entry(env: str, manifest: dict | None = None) -> dict:
    data = manifest if manifest is not None else registry.load_manifest()
    if env not in data or env.startswith("_"):
        raise FetchError(f"no dist_manifest.json entry for env {env!r}")
    entry = data[env]
    placeholders = [k for k, v in entry.items() if v == TODO_MARKER]
    if placeholders:
        raise FetchError(
            f"env {env!r} is not yet publishable: manifest fields "
            f"{placeholders} still carry the {TODO_MARKER!r} marker. "
            f"Use install.sh to build this env locally instead."
        )
    return entry


# ── gated gate ───────────────────────────────────────────────────────────────
def _check_gated(model: str, version: str | None) -> dict:
    spec = registry.resolve(model, version=version)
    if spec["gated"]:
        license_url = spec.get("license_url") or "(see model card)"
        print(
            f"[oh-my-mlip] {model} is GATED. You must accept the upstream "
            f"license before download:\n    {license_url}\n"
            f"  This repo never redistributes gated weights; they are fetched "
            f"with YOUR Hugging Face token.",
            flush=True,
        )
        resolved = _resolve_token()
        if resolved["source"] == "none":
            raise GatedError(
                f"{model} is gated: authenticate with Hugging Face (after "
                f"accepting {license_url}) and retry. Run "
                f"`huggingface-cli login`, or set HF_TOKEN_PATH / "
                f"OMM_HF_TOKEN_FILE to a token file outside the repo. "
                f"See docs/hf_token.md."
            )
        # If the token was provided via the oh-my-mlip convenience var, export it
        # as the standard HF_TOKEN_PATH so child loaders resolve it the normal
        # way — never inline the token value into the environment.
        for key, val in resolved["env"].items():
            os.environ.setdefault(key, val)
    return spec


# ── fallback message ─────────────────────────────────────────────────────────
def _install_fallback_cmd(env: str) -> str:
    return f'bash "$OH_MY_MLIP_HOME/install.sh" {env}'


def _print_fallback(env: str, reason: str) -> None:
    print(
        f"[oh-my-mlip] cannot use the prebuilt {env} tarball on this host:\n"
        f"    {reason}\n"
        f"  Fall back to a local build:\n"
        f"    {_install_fallback_cmd(env)}",
        flush=True,
    )


# ── CUDA capability probe ────────────────────────────────────────────────────
def _probe_cuda(env: str, min_driver_version, python_exe: Path) -> bool:
    """Probe torch.cuda inside the unpacked env. Returns True if usable; on
    failure prints the install.sh fallback command (not a raw traceback).

    When ``min_driver_version`` is recorded (not the ``TODO-on-upload``
    placeholder / None), the host's CUDA driver version (from
    ``torch._C._cuda_getDriverVersion``) is compared against it and a host below
    the minimum is rejected with the fallback message. A placeholder skips the
    comparison."""
    probe = (
        "import json,sys\n"
        "try:\n"
        "    import torch\n"
        "    info={'available':torch.cuda.is_available(),"
        "'driver':getattr(torch._C,'_cuda_getDriverVersion',lambda:None)()}\n"
        "    print('PROBE_OK '+json.dumps(info))\n"
        "except Exception as e:\n"
        "    print('PROBE_FAIL '+repr(e))\n"
    )
    try:
        out = subprocess.run(
            [str(python_exe), "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        _print_fallback(env, f"could not run CUDA probe: {exc!r}")
        return False

    line = (out.stdout or out.stderr).strip().splitlines()
    line = line[-1] if line else ""
    if not line.startswith("PROBE_OK"):
        _print_fallback(env, f"torch import/CUDA probe failed: {line or out.stderr.strip()}")
        return False
    import json as _json

    info = _json.loads(line[len("PROBE_OK ") :])
    if not info.get("available"):
        _print_fallback(
            env,
            f"torch.cuda not available (host driver may be below the env's "
            f"min_driver_version={min_driver_version}).",
        )
        return False

    # Compare the host driver against the env's minimum. The placeholder
    # 'TODO-on-upload' (and a missing/None value) means "not yet recorded" -> we
    # skip the comparison rather than block. A recorded minimum is enforced.
    if min_driver_version not in (None, "", TODO_MARKER):
        host_driver = info.get("driver")
        try:
            min_req = int(min_driver_version)
        except (TypeError, ValueError):
            min_req = None
        if min_req is not None and host_driver is not None and host_driver < min_req:
            _print_fallback(
                env,
                f"host CUDA driver version {host_driver} is below the env's "
                f"min_driver_version={min_req}.",
            )
            return False
    return True


# ── main resolver ────────────────────────────────────────────────────────────
def fetch_env(
    model: str,
    version: str | None = None,
    *,
    probe: bool = True,
    manifest: dict | None = None,
) -> str:
    """Resolve, download, and unpack the conda-pack env for ``model``.

    Steps:
      1. gated check (require HF_TOKEN + print license_url for gated models).
      2. manifest lookup (reject TODO-on-upload placeholders).
      3. hf_hub_download the tarball (revision-pinned) into the cache.
      4. unpack + conda-unpack ONCE, guarded by a sentinel file.
      5. optional CUDA probe against min_driver_version; on fail print the
         exact install.sh fallback command.

    Returns the absolute relocated interpreter path. Heavy deps (huggingface_hub,
    torch) are imported lazily here so the module imports without them.
    """
    spec = _check_gated(model, version)
    env = spec["env"]
    entry = _manifest_entry(env, manifest)

    install_dir = env_install_dir(env)
    sentinel = install_dir / SENTINEL_NAME
    py = interpreter_path(env)

    if sentinel.exists() and py.exists():
        # Already unpacked; just (optionally) re-probe.
        if probe:
            _probe_cuda(env, entry.get("min_driver_version"), py)
        return str(py)

    # 3) download the tarball (lazy import).
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FetchError(
            "huggingface_hub is required to fetch env tarballs "
            "(pip install huggingface_hub), or build locally with install.sh"
        ) from exc

    cache = cache_root()
    cache.mkdir(parents=True, exist_ok=True)
    # Pass an explicit HF_TOKEN if present; otherwise let huggingface_hub do its
    # own standard resolution (HF_TOKEN_PATH / the HF cache login). We never read
    # a token file ourselves, so no token literal enters this process's state.
    token = os.environ.get("HF_TOKEN") or None
    tarball_name = entry.get("tarball") or f"{env}.tar.gz"
    local_tar = hf_hub_download(
        repo_id=entry["hf_repo"],
        filename=tarball_name,
        revision=entry["revision"],
        cache_dir=str(cache),
        token=token,
    )

    _verify_sha256(local_tar, entry.get("sha256"))

    # 4) unpack + conda-unpack once.
    install_dir.mkdir(parents=True, exist_ok=True)
    _extract_tarball(local_tar, install_dir)
    _conda_unpack(install_dir)
    sentinel.write_text("ok\n", encoding="utf-8")

    if not py.exists():
        raise FetchError(f"unpack finished but interpreter missing: {py}")

    # 5) CUDA probe.
    if probe:
        _probe_cuda(env, entry.get("min_driver_version"), py)
    return str(py)


def _verify_sha256(path: str, expected) -> None:
    if not expected or expected == TODO_MARKER:
        return
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected:
        raise FetchError(f"sha256 mismatch for {path}: expected {expected}, got {got}")


def _extract_tarball(tar_path: str, dest: Path) -> None:
    import tarfile

    with tarfile.open(tar_path, "r:*") as tf:
        # Python 3.12 supports the 'data' filter to reject unsafe members.
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # pragma: no cover - older python
            tf.extractall(dest)


def _conda_unpack(install_dir: Path) -> None:
    """Run the env's bundled conda-unpack to fix up relocation.

    A conda-pack tarball MUST ship the ``bin/conda-unpack`` shim. If it is
    absent the env is not relocatable and must not be treated as ready, so we
    raise ``FetchError`` (the caller will not write the ready sentinel)."""
    unpack = install_dir / "bin" / "conda-unpack"
    if not unpack.exists():
        raise FetchError(
            f"conda-unpack shim missing under {install_dir}; the tarball is not "
            f"a relocatable conda-pack env. Build locally with install.sh instead."
        )
    subprocess.run([str(unpack)], check=True, cwd=str(install_dir))
