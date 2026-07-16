"""oh_my_mlip.registry — load + validate models.json / dist_manifest.json and
resolve a model into a machine-readable codegen dict (Layer 1 of the tiered
teacher-provider interface).

This is the LOCKED `resolve()` contract:

    resolve(model, version=None) -> {
        python, env, imports, inference, env_run,
        arch_pinned, gated, weights, validation,
    }

`${OH_MY_MLIP_HOME}` placeholders are expanded to the absolute clone root
(read from the OH_MY_MLIP_HOME env var, defaulting to the repo root that
contains this package). `env_run` strings are parsed into a dict through a
strict key=value allowlist (a security boundary — see `parse_env_run`).

No heavy / framework imports here: this module imports cleanly on a host with
no torch / ase / conda env present (the unit tests rely on that).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

__all__ = [
    "RegistryError",
    "home",
    "models_json_path",
    "dist_manifest_path",
    "load_models",
    "load_manifest",
    "list_models",
    "list_versions",
    "resolve",
    "detect_host_arch",
    "parse_env_run",
    "ENV_RUN_ALLOWLIST",
]


class RegistryError(Exception):
    """Raised on malformed registry data or a disallowed env_run token."""

# ── Host GPU arch detection (stdlib-only; no torch) ─────────────────────────
# `resolve()` uses this when the caller does not pass an explicit arch for an
# arch-pinned model. Beta-test finding (RTX A4500 / sm86 host, 2026-07): the
# launcher spawned workers with --model only, so resolve() silently defaulted
# to sm89 and the sm86 host could never verify Allegro/NequIP through the
# public launcher. Detecting the host compute capability here makes every
# layer (resolve / get_calculator / Worker / run) host-correct by construction.
_HOST_ARCH_UNSET = object()
_host_arch_cache: Any = _HOST_ARCH_UNSET


def detect_host_arch() -> str | None:
    """Return the host GPU arch as ``"sm<major><minor>"`` (e.g. ``sm86``,
    ``sm89``) via ``nvidia-smi --query-gpu=compute_cap``, or ``None`` when no
    GPU / no nvidia-smi / unparseable output. stdlib-only and cached for the
    process lifetime (this module must import and run on GPU-less hosts)."""
    global _host_arch_cache
    if _host_arch_cache is not _HOST_ARCH_UNSET:
        return _host_arch_cache
    arch: str | None = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().splitlines()
        if out:
            m = re.match(r"^(\d+)\.(\d+)$", out[0].strip())
            if m:
                arch = f"sm{m.group(1)}{m.group(2)}"
    except Exception:  # noqa: BLE001 - any failure means "unknown host arch"
        arch = None
    _host_arch_cache = arch
    return arch



# ── repo-root / file location ────────────────────────────────────────────────
# This file lives at <repo>/oh_my_mlip/registry.py, so the repo root is two
# parents up. OH_MY_MLIP_HOME (env var) overrides it when set.
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent


def home() -> str:
    """Return the resolved OH_MY_MLIP_HOME (absolute clone root).

    Order: OH_MY_MLIP_HOME env var -> repo root containing this package.
    """
    env = os.environ.get("OH_MY_MLIP_HOME")
    if env:
        return str(Path(env).expanduser())
    return str(_REPO_ROOT)


def _expand(value: Any, home_path: str) -> Any:
    """Expand ${OH_MY_MLIP_HOME} / $OH_MY_MLIP_HOME in strings (recursing into
    lists). Other strings pass through untouched."""
    if isinstance(value, str):
        return value.replace("${OH_MY_MLIP_HOME}", home_path).replace(
            "$OH_MY_MLIP_HOME", home_path
        )
    if isinstance(value, list):
        return [_expand(v, home_path) for v in value]
    return value


def models_json_path() -> Path:
    return _REPO_ROOT / "models.json"


def dist_manifest_path() -> Path:
    return _REPO_ROOT / "dist_manifest.json"


def _load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError as exc:
        raise RegistryError(f"registry file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistryError(f"{path}: top-level JSON must be an object")
    return data


def load_models(path: str | os.PathLike | None = None) -> dict:
    """Load + lightly validate models.json. Returns the raw dict (incl _meta)."""
    p = Path(path) if path is not None else models_json_path()
    data = _load_json(p)
    for fw, info in data.items():
        if fw.startswith("_"):
            continue
        if not isinstance(info, dict):
            raise RegistryError(f"models.json: '{fw}' must be an object")
        if "env" not in info or "python" not in info:
            raise RegistryError(f"models.json: '{fw}' missing 'env'/'python'")
        if "versions" not in info or not isinstance(info["versions"], dict):
            raise RegistryError(f"models.json: '{fw}' missing 'versions' object")
    return data


def load_manifest(path: str | os.PathLike | None = None) -> dict:
    """Load + lightly validate dist_manifest.json. Returns the raw dict."""
    p = Path(path) if path is not None else dist_manifest_path()
    data = _load_json(p)
    for env, entry in data.items():
        if env.startswith("_"):
            continue
        if not isinstance(entry, dict) or "env" not in entry or "hf_repo" not in entry:
            raise RegistryError(
                f"dist_manifest.json: '{env}' must carry 'env' + 'hf_repo'"
            )
    return data


def list_models(models: dict | None = None) -> list[str]:
    """Return the framework names in registry order (excludes the _meta block)."""
    data = models if models is not None else load_models()
    return [k for k in data.keys() if not k.startswith("_")]


def list_versions(model: str, models: dict | None = None) -> list[str]:
    """Return the version keys for a framework."""
    data = models if models is not None else load_models()
    if model not in data or model.startswith("_"):
        raise RegistryError(f"unknown model: {model!r}")
    return list(data[model].get("versions", {}).keys())


def _resolve_family_and_version_name(
    model: str,
    version: str | None,
    data: dict,
) -> tuple[str, str | None]:
    """Normalize family-name or version-name input to ``(family, version)``."""
    if model.startswith("_"):
        raise RegistryError(
            f"unknown model: {model!r} (known: {list_models(data)})"
        )
    if model in data:
        return model, version

    matches: list[str] = []
    for family, info in data.items():
        if family.startswith("_") or not isinstance(info, dict):
            continue
        versions = info.get("versions", {})
        if isinstance(versions, dict) and model in versions:
            matches.append(family)

    if not matches:
        raise RegistryError(
            f"unknown model: {model!r} (known: {list_models(data)})"
        )
    if len(matches) > 1:
        raise RegistryError(
            f"ambiguous version name {model!r}; found under families {matches}; "
            f"pass the family name and version=... explicitly"
        )
    if version is not None and version != model:
        raise RegistryError(
            f"{model!r} is a version name for family {matches[0]!r}; "
            f"do not also pass version={version!r}"
        )
    return matches[0], model


# ── env_run key=value allowlist (security boundary) ──────────────────────────
# models.json is code-equivalent / trusted (PR-gated), but env_run must be
# constrained: it is applied as a subprocess env, NEVER shell-interpolated.
# Only bare KEY=VALUE tokens whose KEY is on this allowlist are accepted.
ENV_RUN_ALLOWLIST: frozenset[str] = frozenset(
    {
        "LD_LIBRARY_PATH",
        "PYTORCH_CUDA_ALLOC_CONF",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_LAUNCH_BLOCKING",
        "PYTHONUTF8",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
    }
)

# A permitted token is exactly KEY=VALUE where KEY is an allowed env-var name
# and VALUE contains no shell metacharacters. VALUE may be empty or quoted.
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHELL_META_RE = re.compile(r"[`$&|;<>(){}\n\r*?!\\]")


def parse_env_run(s: str | None) -> dict[str, str]:
    """Parse an `env_run` prefix string into an env-var dict.

    Accepts whitespace-separated ``KEY=VALUE`` tokens. Each KEY must be on
    ``ENV_RUN_ALLOWLIST`` and each VALUE must be free of shell metacharacters.
    Surrounding single/double quotes on the VALUE are stripped (so
    ``LD_LIBRARY_PATH=""`` -> ``{"LD_LIBRARY_PATH": ""}``).

    Raises ``RegistryError`` for ANY token that is not a bare allow-listed
    KEY=VALUE (e.g. ``$(rm -rf /)`` or ``FOO=bar; rm x``). There is NO raw
    shell interpolation — this is the trust boundary for the registry.
    """
    if not s:
        return {}
    if not isinstance(s, str):
        raise RegistryError(f"env_run must be a string, got {type(s).__name__}")

    out: dict[str, str] = {}
    for token in s.split():
        if "=" not in token:
            raise RegistryError(
                f"env_run token is not KEY=VALUE: {token!r} "
                f"(raw shell interpolation is forbidden)"
            )
        key, value = token.split("=", 1)
        if not _KEY_RE.match(key):
            raise RegistryError(f"env_run key is not a valid identifier: {key!r}")
        if key not in ENV_RUN_ALLOWLIST:
            raise RegistryError(
                f"env_run key {key!r} is not on the allowlist "
                f"{sorted(ENV_RUN_ALLOWLIST)}"
            )
        # Strip matching surrounding quotes, then reject shell metacharacters.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if _SHELL_META_RE.search(value):
            raise RegistryError(
                f"env_run value for {key!r} contains shell metacharacters: {value!r}"
            )
        out[key] = value
    return out


# ── resolve (Layer 1) ────────────────────────────────────────────────────────
def resolve(
    model: str,
    version: str | None = None,
    *,
    arch: str | None = None,
    models: dict | None = None,
) -> dict:
    """Resolve a model (+ optional version) into the LOCKED codegen dict.

    ``model`` may be either a framework/family key (for example ``"MACE"``)
    or a version key found under a family's ``versions`` object (for example
    ``"MACE-MPA-0"``). Version-key inputs are normalized to their owning
    family before resolution.

    Returns::

        {
          "model": <framework>,
          "version": <version key>,
          "env": <conda env name>,
          "python": <absolute interpreter path, $OH_MY_MLIP_HOME expanded>,
          "imports": [<import line>, ...],          # $OH_MY_MLIP_HOME expanded
          "inference": [<inference line>, ...],     # $OH_MY_MLIP_HOME expanded
          "env_run": {<KEY>: <VALUE>, ...},         # parsed + allowlisted
          "env_run_raw": <original env_run string or "">,
          "arch_pinned": bool,
          "gated": bool,
          "license_url": str | None,
          "weights": str,
          "weights_fetch": str,
          "weights_source": str,
          "weights_source_url": str | None,
          "validation": str,
          "note": str | None,
        }

    `version=None` selects the framework's ``default_version`` if declared,
    else its sole version when there is only one, else raises ``RegistryError``
    (genuinely ambiguous: several versions and no default). For arch-pinned
    models pass ``arch`` (``"sm86"``/``"sm89"``) to pick the matching
    ``inference_<arch>`` block; when omitted the host GPU arch is auto-detected
    (``detect_host_arch()``, via nvidia-smi compute_cap), falling back to
    ``sm89`` on GPU-less hosts.
    """
    data = models if models is not None else load_models()
    home_path = home()

    model, version = _resolve_family_and_version_name(model, version, data)
    info = data[model]
    versions = info.get("versions", {})
    if not versions:
        raise RegistryError(f"model {model!r} has no versions")

    if version is None:
        default_version = info.get("default_version")
        if default_version is not None:
            if default_version not in versions:
                raise RegistryError(
                    f"model {model!r} declares default_version "
                    f"{default_version!r}, which is not one of {list(versions)}"
                )
            version = default_version
        elif len(versions) == 1:
            version = next(iter(versions))
        else:
            raise RegistryError(
                f"model {model!r} has multiple versions {list(versions)} and no "
                f"'default_version'; pass version=..."
            )
    if version not in versions:
        raise RegistryError(
            f"unknown version {version!r} for {model!r}; "
            f"available: {list(versions)}"
        )
    vinfo = versions[version]

    arch_pinned = bool(vinfo.get("arch_pinned", False))
    # inference: arch-specific block for arch-pinned models, else plain. The
    # arch actually used is returned as spec["arch"] so callers (Worker) can
    # propagate it explicitly to the in-env worker process.
    use_arch: str | None = None
    if arch_pinned:
        use_arch = arch or detect_host_arch() or "sm89"
        inference = vinfo.get(f"inference_{use_arch}") or vinfo.get("inference")
        if not inference:
            raise RegistryError(
                f"{model}/{version}: no inference for arch {use_arch!r}"
            )
        # ${OMM_ARCH} template: a generic `inference` block may parameterize the
        # compiled-artifact path on the arch token, so ANY host arch (sm86, sm89,
        # sm120, ...) resolves without a hand-added inference_<arch> block — the
        # per-arch .pt2 is then materialized by the first-run compile flow
        # (nequip-compile targets whatever GPU is present). Explicit
        # inference_<arch> blocks still win when present (host-verified paths).
        inference = [line.replace("${OMM_ARCH}", use_arch) for line in inference]
    else:
        inference = vinfo.get("inference")
        if not inference:
            raise RegistryError(f"{model}/{version}: missing 'inference'")

    python = _expand(info["python"], home_path)
    imports = _expand(list(info.get("import", [])), home_path)
    inference = _expand(list(inference), home_path)
    env_run_raw = info.get("env_run", "") or ""
    env_run = parse_env_run(env_run_raw)

    return {
        "model": model,
        "version": version,
        "env": info["env"],
        "python": python,
        "imports": imports,
        "inference": inference,
        "env_run": env_run,
        "env_run_raw": env_run_raw,
        "arch_pinned": arch_pinned,
        "arch": use_arch,
        "gated": bool(vinfo.get("gated", False)),
        "license_url": vinfo.get("license_url"),
        "weights": vinfo.get("weights", "bundled"),
        "weights_fetch": vinfo.get("weights_fetch", "by-name"),
        "weights_source": vinfo.get("weights_source"),
        "weights_source_url": vinfo.get("weights_source_url"),
        "weights_sha256": vinfo.get("weights_sha256"),
        "weights_size": vinfo.get("weights_size"),
        "weights_fetch_command": vinfo.get("weights_fetch_command"),
        "weights_cache_env": vinfo.get("weights_cache_env"),
        "validation": vinfo.get("validation", "unknown"),
        "note": vinfo.get("note") or info.get("note"),
    }
