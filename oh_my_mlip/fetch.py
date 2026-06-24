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
import subprocess
from pathlib import Path

from oh_my_mlip import registry

__all__ = [
    "FetchError",
    "GatedError",
    "cache_root",
    "env_install_dir",
    "fetch_env",
    "interpreter_path",
]

TODO_MARKER = "TODO-on-upload"
SENTINEL_NAME = ".oh-my-mlip-unpacked"


class FetchError(RuntimeError):
    """Raised on a missing/placeholder manifest entry or a failed unpack."""


class GatedError(FetchError):
    """Raised when a gated model is requested without HF_TOKEN."""


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
        if not os.environ.get("HF_TOKEN"):
            raise GatedError(
                f"{model} is gated: set HF_TOKEN (after accepting "
                f"{license_url}) and retry."
            )
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
    token = os.environ.get("HF_TOKEN")
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
