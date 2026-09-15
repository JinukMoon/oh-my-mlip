#!/usr/bin/env python3
"""fresh_root.py -- content-addressed working-tree snapshot + disposable
runtime roots for fresh isolated validation cycles.

The candidate code under test is the WORKING TREE (uncommitted edits
included), never `git archive HEAD` -- so every cycle snapshots the tree
content-addressed, materializes a writable runtime copy under a disposable
parent, and later proves that copy's source files were not mutated and that
the real hub / user envs were never written.

Subcommands (each prints ONE JSON object on stdout; exit 0 iff `ok`):

  allowlist   write the approved new-path allowlist (generated from
              APPROVED_NEW_PATHS -- a new untracked file is NOT auto-added
              here; add it to APPROVED_NEW_PATHS explicitly).
  snapshot    build the manifest {path: sha256} from tracked paths (read from
              the working tree) + allowlisted new paths; fail closed on
              unexpected untracked code, tracked deletions, symlink escape.
              --check performs every check and writes nothing.
  materialize extract a snapshot tarball into an EMPTY runtime root, create
              the designated writable dirs, write the ownership record, verify
              the extracted source hashes, and print the cache-route exports.
  seed-cache  COPY (never symlink/move) a declared subset of native caches
              into the runtime root; record every seeded file with its hash.
  verify      --routes: every cache/tmp/pkgs route resolves inside the runtime
              root and `conda config --show pkgs_dirs` agrees;
              --isolation: the runtime copy's OWN `oh_my_mlip.registry.resolve()`
              (run with OH_MY_MLIP_HOME = the root -- never a re-implementation
              of its path logic) yields, for every variant, an interpreter and
              weight paths inside the root, and reports no adoption entry;
              --sources: re-hash the runtime copy against the manifest
              (drift => failed(source_mutated)).
  preserve    COPY every log / ledger / rerun script / config plus the named
              final artifacts (fine-tune checkpoints, ...) out of the root,
              re-read each copy and compare sha256 + size, write
              preserved.json next to the copies. Every named artifact is
              REQUIRED (a missing one fails preserve; nothing is skipped
              silently). Cleanup requires this record.
  cleanup     remove ONE owned runtime root -- bounded to a symlink-free
              canonical path whose direct parent equals the canonical
              --allowed-parent AND bound in the campaign-external
              owned_roots.json (written at materialize; the in-root marker is
              never trusted alone); refuses anything containing `.git`, the
              hub itself, roots still in use (any live process with cwd /
              exe / open fd / mapped file inside the root, or in a launched
              --owned-group session; an unreadable /proc refuses, so does a
              still-present pid nothing could be read of, and so does a pid
              whose root use is unreadable and not excluded by the root's
              0700 permission -- the scan reports quiescence "partial",
              never claims "nobody"), roots with a mount point at or below
              them (checked BEFORE removal), roots
              whose evidence ledger (outside the root) does not carry this
              root's manifest_sha256,
              and roots whose preserved.json copies do not re-verify. Proof of
              cleanup is the owned path's ABSENCE (df is a diagnostic only).

Child environment: `build_cycle_env` produces the EFFECTIVE env every child
runs with -- inherited cache overrides dropped, every route (HOME included:
<root>/home) inside the root, the user's read-only inputs (HF token file,
condarc) supplied as PATH exports. `verify --routes` is run on that env.

Failure classes are stable literal strings, e.g.
`failed(snapshot:unexpected_untracked)`, `failed(source_mutated)`,
`failed(isolation)`, `failed(routes:out_of_root)`, `failed(cleanup:not_owned)`.

Nothing here installs, runs models, or touches the real hub's local state;
the only writes are the campaign snapshot dir, the runtime root, and (for
cleanup) the removal of a root that passed every guard.
"""
from __future__ import annotations

import argparse
import ctypes
import fnmatch
import hashlib
import io
import json
import errno
import os
import re
import shutil
import stat as stat_module
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup_common import utc_now  # noqa: E402

OWNER_TAG = "oh-my-mlip/fresh_root"
OWNERSHIP_FILE = ".omm_owned.json"
ENV_EXPORT_FILE = ".omm_fresh_env.sh"
MANIFEST_FILE = "manifest.json"
TARBALL_FILE = "snapshot.tar"
PRESERVED_FILE = "preserved.json"
# Every free-space number this harness records is GiB (2^30), matching the
# preflight's schema-2 `*_gib` fields; decimal-GB inputs are converted once.
GIB = 1024 ** 3
GB_PER_GIB = 1.073741824
# Files copied out of a runtime root by `preserve`: ledgers, logs, rerun
# scripts, configs -- never datasets, never the env tree. Final artifacts
# (checkpoints, ...) are named explicitly by the caller.
PRESERVE_SUFFIXES = {".jsonl", ".log", ".sh", ".json", ".yaml", ".yml", ".txt", ".csv", ".md"}
PRESERVE_DIRS = (".sweep", ".ft", ".catbench", ".distill")
# Campaign-external ownership registry (one per campaign dir, OUTSIDE every
# root): cleanup refuses a root that is not bound here -- the in-root marker
# alone is forgeable by anything that can write inside the root.
OWNED_ROOTS_FILE = "owned_roots.json"

# Fail closed: an untracked, non-ignored path with one of these
# suffixes outside the allowlist aborts the snapshot (it is candidate code
# the snapshot would otherwise silently omit).
CODE_SUFFIXES = {".py", ".sh", ".md", ".json", ".yml", ".yaml"}

# Exclude: never enters a snapshot whether tracked or not. Local
# state, weights, materialized envs, ledgers, build products.
EXCLUDE_TOP_DIRS = {
    ".git", "models", ".sweep", "dist", ".omc", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".gjc_closure_artifacts", "cache", "jobs", "result", "report",
    ".omm_fresh",
}
EXCLUDE_FILES = {"env_map.local.json", "models.local.json"}
EXCLUDE_ANY_DIR_NAMES = {"__pycache__"}
# Secret / runtime patterns applied to UNTRACKED paths only. Tracked paths are
# public git content by definition (the repo's .gitignore already whitelists
# the token-free docs/hf_token.md, tests/test_fetch_token.py, ...).
EXCLUDE_UNTRACKED_GLOBS = [".reloc_gate*", "*token*", ".env", "*.log", "*.pyc"]

# New (not yet tracked) paths a snapshot may include: globs for new source
# families plus exact implementation/test paths (exact paths, never a widened
# glob). The allowlist FILE is generated from this constant; any other
# untracked code path fails the snapshot until it is added here.
APPROVED_NEW_PATHS = [
    "recipes/*.md",
    "scripts/fresh_root.py",
    "scripts/ft_sweep.py",
    "scripts/catbench_vasp_stage.py",
    "scripts/catbench_datasets.py",
    "scripts/catbench_leaderboard.py",
    "scripts/catbench_version.py",
    "scripts/distill_verify.py",
    "scripts/evidence_report.py",
    # NequIP/Allegro per-arch .pt2 compile step run by install.sh (exact paths).
    "scripts/prepare_nequip_weights.py",
    "scripts/prepare_allegro_weights.py",
    # Exact-replay env locks (conda @EXPLICIT + pip --no-deps) generated from a
    # passing fresh-root campaign inventory, and their generator.
    "scripts/gen_env_lock.py",
    "envs/locks/*.txt",
    "tests/test_fresh_root.py",
    "tests/test_ft_sweep.py",
    "tests/test_catbench_vasp_stage.py",
    "tests/test_catbench_datasets.py",
    "tests/test_catbench_leaderboard.py",
    "tests/test_catbench_version.py",
    "tests/test_distill_verify.py",
    "tests/test_evidence_report.py",
    # Recipe-contract test (exact path, no blanket allow).
    "tests/test_recipe_contract.py",
    # Exact paths (no glob widening): explicit-input relax test, ft_verify
    # loader test, CatBench report test.
    "tests/test_relax_input.py",
    "tests/test_ft_verify.py",
    "tests/test_catbench_report.py",
    # CatBench quickstart passthrough test (exact path).
    "tests/test_catbench_quickstart.py",
    "docs/host_sessions/*.md",
]

# Materialize: designated writable outputs inside the runtime copy.
WRITABLE_DIRS = ["envs", "models", ".sweep", ".ft", ".catbench", ".distill", "work", "cache", "tmp", "home",
                 "cache/conda_pkgs", "cache/pip", "cache/shared", "cache/torch_ext", "cache/xdg",
                 "cache/triton", "cache/nv", "cache/inductor"]

# Route verification: every one of these must resolve inside the
# runtime root before any compute. They are checked on the EFFECTIVE child
# environment (what the children are actually spawned with), not on the
# export list alone.
#
# HOME is routed to <root>/home: env.sh (sections 4/5) and several upstream
# packages (MatRIS/TACE/eqnorm `os.mkdir("$HOME/.cache/<m>")`, conda's
# ~/.conda/environments.txt, huggingface_hub / torch defaults) write to
# literal `$HOME/...` paths that XDG_CACHE_HOME does NOT redirect --
# XDG_CACHE_HOME only covers XDG-aware libraries. The only things the isolated
# HOME loses are read-only user inputs, re-supplied explicitly as PATH exports
# (READ_ONLY_INPUT_VARS), never copied into the root.
ROUTE_VARS = [
    "OH_MY_MLIP_HOME", "OMM_SHARED_CACHE_ROOT", "HF_HOME", "HF_HUB_CACHE", "FAIRCHEM_CACHE_DIR",
    "TORCH_HOME", "CACHED_PATH_CACHE_ROOT", "PIP_CACHE_DIR", "CONDA_PKGS_DIRS", "CONDA_ENVS_DIRS",
    "TMPDIR", "XDG_CACHE_HOME", "HOME", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR",
]
# Inherited variables that would silently re-route a cache OUT of the root
# (huggingface_hub honours HUGGINGFACE_HUB_CACHE / TRANSFORMERS_CACHE over
# HF_HOME; conda honours CONDA_ENVS_PATH; OMM_HOME is the legacy hub
# override). They are dropped from the child environment and their presence
# in an effective environment is a route failure.
REJECT_INHERITED_VARS = [
    "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_ASSETS_CACHE", "HF_DATASETS_CACHE",
    "CONDA_ENVS_PATH", "OMM_HOME",
]
# Approved READ-ONLY user inputs that the isolated HOME would otherwise hide:
# each is a path export pointing at the user's own file (never a copy). They
# may -- must -- lie outside the root; verify_routes only requires that a set
# one exists as a regular file.
READ_ONLY_INPUT_VARS = ["HF_TOKEN_PATH", "CONDARC"]
EXTRA_EXPORT_VARS: list[str] = []


class FreshRootError(Exception):
    """Carries the failure class (`failed(<class>)`) plus detail."""

    def __init__(self, state: str, detail=None):
        super().__init__(state)
        self.state = state
        self.detail = detail


# ---------------------------------------------------------------------------
# hashing helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def aggregate_manifest_sha256(files: dict[str, str]) -> str:
    """sha256 over the sorted `path  sha` lines -- the value every ledger row
    of a cycle carries."""
    lines = "".join(f"{p}  {files[p]}\n" for p in sorted(files))
    return sha256_text(lines)


def hash_or_absent(path: Path) -> str:
    """Targeted-hash helper for the attribution guard: the sha256
    of a file, or the literal 'absent' so a file appearing later is detected
    exactly like one changing."""
    if path.is_file():
        return sha256_file(path)
    return "absent"


def disk_free_bytes(path: Path) -> int:
    """Measured free bytes on the filesystem holding `path` (walks up to the
    nearest existing ancestor). The ONLY source of any capacity decision --
    never a planned or projected figure."""
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


# ---------------------------------------------------------------------------
# git plumbing (repo injectable; every call is read-only)
# ---------------------------------------------------------------------------

def _git_z(repo: Path, *args: str) -> list[str]:
    proc = subprocess.run(["git", "-C", str(repo), *args, "-z"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise FreshRootError("failed(snapshot:git)", proc.stderr.strip())
    return [p for p in proc.stdout.split("\0") if p]


def git_head(repo: Path) -> str:
    proc = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def git_tracked(repo: Path) -> list[str]:
    return _git_z(repo, "ls-files")


def git_untracked(repo: Path) -> list[str]:
    return _git_z(repo, "ls-files", "--others", "--exclude-standard")


def git_deleted(repo: Path) -> list[str]:
    return _git_z(repo, "ls-files", "--deleted")


def git_dirty_tracked(repo: Path) -> list[str]:
    dirty = set(_git_z(repo, "diff", "--name-only"))
    dirty |= set(_git_z(repo, "diff", "--cached", "--name-only"))
    return sorted(dirty)


# ---------------------------------------------------------------------------
# allowlist
# ---------------------------------------------------------------------------

def default_state_dir(repo: Path) -> Path:
    """`.omc/state` of the WORKSPACE (repo parent) when it exists -- where the
    preflight writes omm-e2e-preflight.json -- else the repo's own (gitignored) .omc."""
    ws = repo.parent / ".omc" / "state"
    if ws.is_dir():
        return ws
    return repo / ".omc" / "state"


def default_allowlist_path(repo: Path) -> Path:
    return default_state_dir(repo) / "omm-e2e-allowlist.json"


def render_allowlist() -> dict:
    return {
        "schema_version": 1,
        "source": "omm-end-to-end-recipes-consensus.md v4 Part 3.1 (approved new files)",
        "generated_utc": utc_now(),
        "generator": "scripts/fresh_root.py allowlist",
        "patterns": list(APPROVED_NEW_PATHS),
        "note": ("Untracked paths matching these globs enter a snapshot; any other "
                 "untracked code path fails the snapshot closed. Additions need scope "
                 "approval -- edit APPROVED_NEW_PATHS, never this file by hand."),
    }


def load_allowlist(path: Path | None) -> list[str]:
    """Patterns from the allowlist FILE, which must be a subset of the
    approved constant: a hand-edited file cannot widen the snapshot scope."""
    if path is None or not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    patterns = data.get("patterns") or []
    if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
        raise FreshRootError("failed(snapshot:allowlist_invalid)", str(path))
    unapproved = sorted(set(patterns) - set(APPROVED_NEW_PATHS))
    if unapproved:
        raise FreshRootError("failed(snapshot:allowlist_unapproved)", unapproved)
    return patterns


def allowlisted(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(rel, pat) for pat in patterns)


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------

def _excluded(rel: str, *, tracked: bool) -> bool:
    parts = rel.split("/")
    if parts[0] in EXCLUDE_TOP_DIRS:
        return True
    if any(p in EXCLUDE_ANY_DIR_NAMES for p in parts):
        return True
    if parts[-1] in EXCLUDE_FILES:
        return True
    # materialized env dirs: envs/<env>/... (tracked recipes are flat files
    # directly under envs/ and stay in). envs/locks/ is the exception: small
    # text exact-replay locks (scripts/gen_env_lock.py), not a built prefix.
    if parts[0] == "envs" and len(parts) > 2 and parts[1] != "locks":
        return True
    if not tracked:
        name = parts[-1]
        if any(fnmatch.fnmatchcase(name, g) for g in EXCLUDE_UNTRACKED_GLOBS):
            return True
    return False


def _is_code(rel: str) -> bool:
    return Path(rel).suffix.lower() in CODE_SUFFIXES


def _symlink_escapes(repo: Path, rel: str) -> bool:
    path = repo / rel
    if not path.is_symlink():
        return False
    target = path.resolve()
    try:
        target.relative_to(repo.resolve())
        return False
    except ValueError:
        return True


def build_manifest(repo: Path, allowlist: list[str]) -> dict:
    """The whole fail-closed decision, as data. Raises FreshRootError with the
    failure class; returns the manifest dict on success."""
    repo = repo.resolve()
    tracked = git_tracked(repo)
    deleted = git_deleted(repo)
    if deleted:
        raise FreshRootError("failed(snapshot:tracked_deleted)", sorted(deleted))

    untracked = git_untracked(repo)
    included: dict[str, str] = {}
    included_untracked: list[str] = []
    excluded_tracked: list[str] = []
    omitted_untracked_noncode: list[str] = []
    unexpected: list[str] = []
    escapes: list[str] = []

    for rel in tracked:
        if _excluded(rel, tracked=True):
            excluded_tracked.append(rel)
            continue
        if _symlink_escapes(repo, rel):
            escapes.append(rel)
            continue
        if not (repo / rel).is_file():
            # a tracked directory entry / submodule pointer -- nothing to hash
            continue
        included[rel] = sha256_file(repo / rel)

    for rel in untracked:
        if _excluded(rel, tracked=False):
            continue
        if allowlisted(rel, allowlist):
            if _symlink_escapes(repo, rel):
                escapes.append(rel)
                continue
            if (repo / rel).is_file():
                included[rel] = sha256_file(repo / rel)
                included_untracked.append(rel)
            continue
        if _is_code(rel):
            unexpected.append(rel)
        else:
            omitted_untracked_noncode.append(rel)

    if escapes:
        raise FreshRootError("failed(snapshot:symlink_escape)", sorted(escapes))
    if unexpected:
        raise FreshRootError("failed(snapshot:unexpected_untracked)", sorted(unexpected))

    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "repo": str(repo),
        "git_head": git_head(repo),
        "dirty_tracked_paths": git_dirty_tracked(repo),
        "included_untracked": sorted(included_untracked),
        "excluded_tracked": sorted(excluded_tracked),
        "omitted_untracked_noncode": sorted(omitted_untracked_noncode),
        "allowlist_patterns": list(allowlist),
        "file_count": len(included),
        "files": dict(sorted(included.items())),
        "manifest_sha256": aggregate_manifest_sha256(included),
    }


def write_snapshot(repo: Path, manifest: dict, campaign_dir: Path) -> dict:
    """Immutable canonical snapshot: tarball + manifest under
    <campaign_dir>/snapshot/<sha12>/, files chmod 0444. A second
    call with the same manifest is a no-op (content-addressed)."""
    repo = repo.resolve()
    sha = manifest["manifest_sha256"]
    snap_dir = campaign_dir / "snapshot" / sha[:12]
    tar_path = snap_dir / TARBALL_FILE
    man_path = snap_dir / MANIFEST_FILE
    if tar_path.exists() and man_path.exists():
        existing = json.loads(man_path.read_text(encoding="utf-8"))
        if existing.get("manifest_sha256") == sha:
            # reuse only a tarball whose member list is exactly the manifest
            with tarfile.open(tar_path, "r") as tar:
                names = sorted(m.name for m in tar.getmembers())
            if names != sorted(manifest["files"]):
                raise FreshRootError("failed(snapshot:tarball_mismatch)", str(tar_path))
            return {"snapshot_dir": str(snap_dir), "tarball": str(tar_path),
                    "manifest": str(man_path), "reused": True}
    snap_dir.mkdir(parents=True, exist_ok=True)
    tmp_tar = snap_dir / (TARBALL_FILE + ".tmp")
    with tarfile.open(tmp_tar, "w") as tar:
        for rel in sorted(manifest["files"]):
            src = repo / rel
            # the bytes that go into the tarball are the bytes that were
            # hashed: a file edited between build_manifest and here is a
            # different candidate, not a silent substitution.
            data = src.read_bytes()
            if hashlib.sha256(data).hexdigest() != manifest["files"][rel]:
                tmp_tar.unlink(missing_ok=True)
                raise FreshRootError("failed(snapshot:changed_during_write)", rel)
            info = tar.gettarinfo(str(src), arcname=rel)
            info.size = len(data)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    os.replace(tmp_tar, tar_path)
    man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for p in (tar_path, man_path):
        os.chmod(p, 0o444)
    return {"snapshot_dir": str(snap_dir), "tarball": str(tar_path),
            "manifest": str(man_path), "reused": False}


def load_manifest(snapshot_dir: Path) -> dict:
    man_path = snapshot_dir / MANIFEST_FILE
    if not man_path.is_file():
        raise FreshRootError("failed(materialize:manifest_missing)", str(man_path))
    return json.loads(man_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------

def route_exports(runtime: Path) -> dict[str, str]:
    """The env-var routing that pins every cache/tmp/pkgs write inside the
    runtime root. OMM_SHARED_CACHE_ROOT drives env.sh's own
    HF_HOME/FAIRCHEM_CACHE_DIR/TORCH_HOME/CACHED_PATH_CACHE_ROOT redirect; the
    same four are exported explicitly so a caller that does not source env.sh
    is routed identically."""
    runtime = runtime.resolve()
    shared = runtime / "cache" / "shared"
    return {
        "OH_MY_MLIP_HOME": str(runtime),
        "OMM_SHARED_CACHE_ROOT": str(shared),
        "HF_HOME": str(shared / "hf"),
        "HF_HUB_CACHE": str(shared / "hf" / "hub"),
        "FAIRCHEM_CACHE_DIR": str(shared / "fairchem"),
        "TORCH_HOME": str(shared / "torch"),
        "CACHED_PATH_CACHE_ROOT": str(shared / "cached_path"),
        "PIP_CACHE_DIR": str(runtime / "cache" / "pip"),
        "CONDA_PKGS_DIRS": str(runtime / "cache" / "conda_pkgs"),
        "CONDA_ENVS_DIRS": str(runtime / "envs"),
        "TMPDIR": str(runtime / "tmp"),
        "XDG_CACHE_HOME": str(runtime / "cache" / "xdg"),
        "HOME": str(runtime / "home"),
        "TORCH_EXTENSIONS_DIR": str(runtime / "cache" / "torch_ext"),
        "TRITON_CACHE_DIR": str(runtime / "cache" / "triton"),
        "CUDA_CACHE_PATH": str(runtime / "cache" / "nv"),
        "TORCHINDUCTOR_CACHE_DIR": str(runtime / "cache" / "inductor"),
    }


def read_only_inputs(base_env: dict[str, str]) -> dict[str, str]:
    """The approved read-only user inputs the isolated HOME hides, as PATH
    exports to the user's own files: an already-set HF_TOKEN_PATH / CONDARC
    is respected; otherwise the standard `hf auth login` token file
    and ~/.condarc of the REAL home (from `base_env`) are pointed at when they
    exist. Nothing is read or copied."""
    out: dict[str, str] = {}
    real_home = base_env.get("HOME")
    defaults = {}
    if real_home:
        defaults = {"HF_TOKEN_PATH": Path(real_home) / ".cache" / "huggingface" / "token",
                    "CONDARC": Path(real_home) / ".condarc"}
    for var in READ_ONLY_INPUT_VARS:
        current = base_env.get(var)
        if current:
            out[var] = current
        elif var in defaults and defaults[var].is_file():
            out[var] = str(defaults[var])
    return out


def build_cycle_env(runtime: Path, base_env: dict[str, str] | None = None) -> tuple[dict[str, str], dict]:
    """The EFFECTIVE environment every child of a cycle is spawned with:
    `base_env` minus REJECT_INHERITED_VARS, plus route_exports (HOME included)
    and the read-only inputs. Returns (env, report) where report lists what
    was dropped and which read-only inputs were supplied -- verify_routes is
    run on exactly this env, not on the export list."""
    base = dict(os.environ if base_env is None else base_env)
    rejected = sorted(v for v in REJECT_INHERITED_VARS if v in base)
    for v in rejected:
        base.pop(v, None)
    exports = route_exports(runtime)
    inputs = read_only_inputs(base)
    env = {**base, **exports, **inputs}
    return env, {"rejected_inherited": rejected, "read_only_inputs": inputs, "exports": exports}


def render_env_export(exports: dict[str, str]) -> str:
    lines = ["#!/bin/sh", "# generated by scripts/fresh_root.py materialize -- source before any command in this runtime root",
             "# HOME is the ISOLATED campaign home; read-only user inputs are re-supplied as path exports.",
             "unset " + " ".join(REJECT_INHERITED_VARS)]
    for k, v in exports.items():
        lines.append(f'export {k}="{v}"')
    return "\n".join(lines) + "\n"


def _load_registry(path: Path) -> dict:
    if not path.is_file():
        return {"owner": OWNER_TAG, "roots": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FreshRootError("failed(ownership:registry_invalid)", f"{path}: {exc}")
    if not isinstance(data, dict) or data.get("owner") != OWNER_TAG or not isinstance(data.get("roots"), dict):
        raise FreshRootError("failed(ownership:registry_invalid)", str(path))
    return data


def register_owned_root(registry: Path, runtime: Path, record: dict) -> dict:
    """Bind a materialized root in the campaign-external registry: canonical
    root, canonical parent, manifest_sha256, campaign_id. The registry must
    lie outside the root; an existing entry for the same canonical root is a
    refusal (a root is materialized once)."""
    registry = Path(registry).resolve()
    runtime = Path(runtime).resolve()
    if _inside(registry, runtime):
        raise FreshRootError("failed(ownership:registry_inside_runtime)", str(registry))
    data = _load_registry(registry)
    key = str(runtime)
    if key in data["roots"] and data["roots"][key].get("cleaned_utc") is None:
        raise FreshRootError("failed(ownership:already_registered)", key)
    entry = {"runtime_root": key, "runtime_parent": str(runtime.parent),
             "manifest_sha256": record["manifest_sha256"], "campaign_id": record["campaign_id"],
             "snapshot_dir": record.get("snapshot_dir"), "created_utc": record.get("created_utc"),
             "creator_pid": record.get("creator_pid"), "cleaned_utc": None}
    data["roots"][key] = entry
    registry.parent.mkdir(parents=True, exist_ok=True)
    tmp = registry.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, registry)
    return entry


def registered_entry(registry: Path, runtime: Path) -> dict | None:
    data = _load_registry(Path(registry).resolve())
    return data["roots"].get(str(Path(runtime).resolve()))


def materialize(snapshot_dir: Path, runtime: Path, *, campaign_id: str,
                repo: Path | None = None, allowlist: list[str] | None = None,
                allow_stale: bool = False, owned_registry: Path | None = None) -> dict:
    """Extract the canonical snapshot into an empty runtime root.

    Before extracting, when `repo` is given the working tree is re-checked
    (`snapshot --check` semantics, mandatory before EVERY materialization) and
    its manifest_sha256 must equal the snapshot's -- otherwise the snapshot no
    longer represents the candidate code and materialization is refused
    (`--allow-stale` records the mismatch instead, for re-materialization
    from a pinned manifest).

    `owned_registry` (default: <snapshot_dir>/../../owned_roots.json, i.e. the
    campaign dir) is the campaign-external ownership binding cleanup later
    requires; it must lie outside the root."""
    snapshot_dir = Path(snapshot_dir).resolve()
    runtime = Path(runtime).resolve()
    manifest = load_manifest(snapshot_dir)
    registry = Path(owned_registry).resolve() if owned_registry else default_registry_path(snapshot_dir)
    if _inside(registry, runtime):
        raise FreshRootError("failed(ownership:registry_inside_runtime)", str(registry))
    tar_path = snapshot_dir / TARBALL_FILE
    if not tar_path.is_file():
        raise FreshRootError("failed(materialize:tarball_missing)", str(tar_path))

    stale = None
    if repo is not None:
        current = build_manifest(Path(repo), allowlist or [])
        if current["manifest_sha256"] != manifest["manifest_sha256"]:
            stale = {"snapshot": manifest["manifest_sha256"], "working_tree": current["manifest_sha256"]}
            if not allow_stale:
                raise FreshRootError("failed(materialize:stale_snapshot)", stale)

    if runtime.exists() and any(runtime.iterdir()):
        raise FreshRootError("failed(materialize:runtime_not_empty)", str(runtime))
    if repo is not None and runtime == Path(repo).resolve():
        raise FreshRootError("failed(materialize:runtime_is_repo)", str(runtime))
    # the root is PRIVATE (0700, owned by this uid): the only way cleanup's
    # quiescence scan can later prove that a ptrace-opaque process of
    # another uid cannot reach it. Created that way, then verified -- an
    # existing (empty) directory is never chmod'ed, only checked.
    runtime.parent.mkdir(parents=True, exist_ok=True)
    if not runtime.exists():
        os.mkdir(runtime, 0o700)
    root_mode = root_privacy(runtime)
    if not root_mode["private"]:
        raise FreshRootError("failed(materialize:root_not_private)", root_mode)

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk() or member.name.startswith("/") or ".." in Path(member.name).parts:
                raise FreshRootError("failed(materialize:unsafe_member)", member.name)
        tar.extractall(runtime, filter="data")

    for rel in WRITABLE_DIRS:
        (runtime / rel).mkdir(parents=True, exist_ok=True)

    drift = source_drift(runtime, manifest)
    if drift["mutated"] or drift["missing"]:
        raise FreshRootError("failed(materialize:extract_mismatch)", drift)

    exports = route_exports(runtime)
    (runtime / ENV_EXPORT_FILE).write_text(render_env_export(exports), encoding="utf-8")
    record = {
        "owner": OWNER_TAG,
        "campaign_id": campaign_id,
        "runtime_root": str(runtime),
        "runtime_parent": str(runtime.parent),
        "snapshot_dir": str(snapshot_dir),
        "manifest_sha256": manifest["manifest_sha256"],
        "git_head": manifest.get("git_head"),
        "created_utc": utc_now(),
        "creator_pid": os.getpid(),
        "stale_snapshot": stale,
        "writable_dirs": list(WRITABLE_DIRS),
        "owned_registry": str(registry),
        "root_mode": root_mode["mode"],
        "root_uid": root_mode["uid"],
    }
    (runtime / OWNERSHIP_FILE).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    entry = register_owned_root(registry, runtime, record)
    return {"runtime_root": str(runtime), "manifest_sha256": manifest["manifest_sha256"],
            "file_count": manifest["file_count"], "exports": exports,
            "env_file": str(runtime / ENV_EXPORT_FILE), "ownership": record,
            "owned_registry": str(registry), "registry_entry": entry}


def default_registry_path(snapshot_dir: Path) -> Path:
    """<campaign_dir>/owned_roots.json for a snapshot written by
    write_snapshot (<campaign_dir>/snapshot/<sha12>/)."""
    snapshot_dir = Path(snapshot_dir).resolve()
    campaign_dir = snapshot_dir.parents[1] if snapshot_dir.parent.name == "snapshot" else snapshot_dir.parent
    return campaign_dir / OWNED_ROOTS_FILE


def read_ownership(runtime: Path) -> dict | None:
    path = Path(runtime) / OWNERSHIP_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# seed-cache
# ---------------------------------------------------------------------------

def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def seed_cache(runtime: Path, items: dict[str, str]) -> dict:
    """COPY declared native-cache subsets into the runtime root.

    items: {dst_relative_to_runtime: src_absolute}. Sources are read-only
    inputs -- copied, never symlinked or moved; a source inside the runtime,
    a destination outside it, or an existing destination refuses (no clobber).
    Every seeded file is recorded with its sha256."""
    runtime = Path(runtime).resolve()
    if read_ownership(runtime) is None:
        raise FreshRootError("failed(seed:not_owned)", str(runtime))
    seeded: list[dict] = []
    for dst_rel, src in items.items():
        src_p = Path(src)
        dst_p = (runtime / dst_rel)
        if not _inside(dst_p, runtime):
            raise FreshRootError("failed(seed:dst_outside_runtime)", dst_rel)
        if _inside(src_p, runtime):
            raise FreshRootError("failed(seed:src_inside_runtime)", str(src_p))
        if not src_p.exists():
            raise FreshRootError("failed(seed:src_missing)", str(src_p))
        if dst_p.exists() or dst_p.is_symlink():
            raise FreshRootError("failed(seed:dst_exists)", str(dst_p))
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        if src_p.is_dir():
            shutil.copytree(src_p, dst_p, symlinks=False)
            files = [p for p in dst_p.rglob("*") if p.is_file()]
        else:
            shutil.copy2(src_p, dst_p)
            files = [dst_p]
        for f in files:
            seeded.append({"src": str(src_p), "dst": str(f),
                           "sha256": sha256_file(f), "bytes": f.stat().st_size})
    record_dir = runtime / ".sweep" / "seed"
    record_dir.mkdir(parents=True, exist_ok=True)
    record = {"at": utc_now(), "weights_source": "reused-native-cache" if seeded else "fresh-download",
              "seeded": seeded}
    (record_dir / "seeded.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


# ---------------------------------------------------------------------------
# verify: routes / isolation / sources
# ---------------------------------------------------------------------------

def verify_routes(runtime: Path, env: dict[str, str], *, conda_cmd: list[str] | None = None) -> dict:
    runtime = Path(runtime).resolve()
    problems: dict[str, str] = {}
    for var in ROUTE_VARS:
        value = env.get(var)
        if not value:
            problems[var] = "unset"
            continue
        # CONDA_PKGS_DIRS may be a comma-separated list; every entry must be inside.
        values = value.split(",") if var in ("CONDA_PKGS_DIRS", "CONDA_ENVS_DIRS") else [value]
        for v in values:
            if not _inside(Path(v.strip()), runtime):
                problems[var] = f"outside runtime root: {v}"
    # an inherited override that huggingface_hub / conda would honour over the
    # routed variable re-routes writes out of the root: must be ABSENT.
    for var in REJECT_INHERITED_VARS:
        if env.get(var) is not None:
            problems[var] = f"inherited override present (must be unset): {env[var]}"
    # approved read-only inputs: path exports to existing regular files only
    for var in READ_ONLY_INPUT_VARS:
        value = env.get(var)
        if value and not Path(value).is_file():
            problems[var] = f"read-only input is not a regular file: {value}"
    conda_pkgs = "unchecked"
    cmd = conda_cmd if conda_cmd is not None else (["conda"] if shutil.which("conda") else None)
    if cmd is None:
        problems["conda"] = "conda not on PATH -- pkgs_dirs cannot be asserted"
    else:
        try:
            proc = subprocess.run([*cmd, "config", "--show", "pkgs_dirs", "--json"],
                                  env=dict(os.environ, **env), stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, timeout=120)
            shown = json.loads(proc.stdout or "{}").get("pkgs_dirs") if proc.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            shown = None
            problems["conda"] = f"conda config failed: {exc}"
        expected = [p.strip() for p in (env.get("CONDA_PKGS_DIRS") or "").split(",") if p.strip()]
        conda_pkgs = shown
        if shown is None:
            problems.setdefault("conda", "conda config --show pkgs_dirs returned nothing")
        elif [str(Path(p)) for p in shown] != [str(Path(p)) for p in expected]:
            problems["conda"] = f"pkgs_dirs {shown} != CONDA_PKGS_DIRS {expected}"
    ok = not problems
    return {"ok": ok, "state": "passed" if ok else "failed(routes:out_of_root)",
            "runtime_root": str(runtime), "problems": problems, "conda_pkgs_dirs": conda_pkgs}


# Executed INSIDE the runtime root by the runtime copy's own package: the
# registry under test resolves every variant exactly as a user script would
# (OH_MY_MLIP_HOME = the root). No path logic is re-implemented here -- the
# user forbids a parallel resolver; this only reports what resolve() says.
_ISOLATION_PROBE = r'''
import json, os, sys
root = os.environ["OH_MY_MLIP_HOME"]
sys.path.insert(0, root)
from oh_my_mlip import registry, fetch
out = {"home": registry.home(), "models_json": str(registry.models_json_path()),
       "local_env_map": registry.local_env_map(), "entries": []}
data = registry.load_models()
for family, info in data.items():
    if family.startswith("_"):
        continue
    for version, v in (info.get("versions") or {}).items():
        archs = [None]
        if (v or {}).get("arch_pinned"):
            archs = sorted(k[len("inference_"):] for k in v if k.startswith("inference_")) or [None]
        for arch in archs:
            e = {"family": family, "version": version, "arch": arch}
            try:
                spec = registry.resolve(family, version, arch=arch, models=data)
                e["python"] = spec["python"]
                e["weights"] = [str(p) for p in fetch._inference_weight_targets(spec)]
                e["local_verified"] = spec.get("local_verified") is not None
            except Exception as exc:  # noqa: BLE001 -- reported verbatim, judged by the caller
                e["error"] = f"{type(exc).__name__}: {exc}"
            out["entries"].append(e)
print(json.dumps(out))
'''


def verify_isolation(runtime: Path, *, python: str | None = None) -> dict:
    """Run the runtime copy's OWN `oh_my_mlip.registry.resolve()` for every
    variant with OH_MY_MLIP_HOME = the root and judge its answers: the
    package root and models.json must be the runtime's, no adoption entry may
    exist, and every interpreter / referenced weight path must lie inside the
    root. A resolve() error (an adoption entry pointing at a dead prefix, a
    missing inference block, ...) is itself an isolation failure."""
    runtime = Path(runtime).resolve()
    problems: list[str] = []
    adoption = runtime / "env_map.local.json"
    if adoption.exists():
        try:
            entries = json.loads(adoption.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            entries = "unparseable"
        problems.append(f"adoption entry present in runtime root: {entries}")
    env = dict(os.environ, OH_MY_MLIP_HOME=str(runtime), PYTHONPATH=str(runtime))
    env.pop("OMM_HOME", None)
    try:
        proc = subprocess.run([python or sys.executable, "-c", _ISOLATION_PROBE], cwd=str(runtime), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        problems.append(f"registry probe could not run: {exc}")
        return {"ok": False, "state": "failed(isolation)", "runtime_root": str(runtime), "problems": problems}
    report = None
    if proc.returncode == 0:
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("{"):
                try:
                    report = json.loads(line)
                except json.JSONDecodeError:
                    pass
                break
    if report is None:
        problems.append("registry probe failed (rc=%d): %s" % (proc.returncode, proc.stderr.strip()[-800:]))
        return {"ok": False, "state": "failed(isolation)", "runtime_root": str(runtime), "problems": problems}

    if not _inside(Path(report["home"]), runtime) or Path(report["home"]).resolve() != runtime:
        problems.append(f"registry.home() is not the runtime root: {report['home']}")
    if not _inside(Path(report["models_json"]), runtime):
        problems.append(f"registry.models_json_path() outside runtime root: {report['models_json']}")
    if report["local_env_map"] and not adoption.exists():
        problems.append(f"registry.local_env_map() reports adoption entries: {report['local_env_map']}")
    for e in report["entries"]:
        tag = f"{e['family']}/{e['version']}" + (f"[{e['arch']}]" if e.get("arch") else "")
        if "error" in e:
            problems.append(f"{tag}: resolve() failed: {e['error']}")
            continue
        if not _inside(Path(e["python"]), runtime):
            problems.append(f"{tag}: interpreter route outside runtime root: {e['python']}")
        for w in e["weights"]:
            if not _inside(Path(w), runtime):
                problems.append(f"{tag}: weight route outside runtime root: {w}")
    ok = not problems
    return {"ok": ok, "state": "passed" if ok else "failed(isolation)", "runtime_root": str(runtime),
            "problems": problems, "resolved": len(report["entries"]), "entries": report["entries"],
            "probe": "oh_my_mlip.registry.resolve() inside the runtime root"}


def source_drift(runtime: Path, manifest: dict) -> dict:
    runtime = Path(runtime).resolve()
    mutated, missing = [], []
    for rel, sha in manifest["files"].items():
        p = runtime / rel
        if not p.is_file():
            missing.append(rel)
        elif sha256_file(p) != sha:
            mutated.append(rel)
    return {"mutated": sorted(mutated), "missing": sorted(missing)}


def verify_sources(runtime: Path, manifest: dict | None = None) -> dict:
    runtime = Path(runtime).resolve()
    if manifest is None:
        record = read_ownership(runtime)
        if record is None:
            raise FreshRootError("failed(verify:not_owned)", str(runtime))
        manifest = load_manifest(Path(record["snapshot_dir"]))
    drift = source_drift(runtime, manifest)
    ok = not drift["mutated"] and not drift["missing"]
    return {"ok": ok, "state": "passed" if ok else "failed(source_mutated)",
            "runtime_root": str(runtime), "manifest_sha256": manifest["manifest_sha256"], **drift}


# ---------------------------------------------------------------------------
# preserve (durable evidence BEFORE cleanup)
# ---------------------------------------------------------------------------

def _copy_verified(src: Path, dst: Path) -> dict:
    """copy2 + re-read the destination: its sha256 and size must equal the
    source's, and the copy must be non-empty unless the source was."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    src_sha, dst_sha = sha256_file(src), sha256_file(dst)
    src_size, dst_size = src.stat().st_size, dst.stat().st_size
    if src_sha != dst_sha or src_size != dst_size:
        raise FreshRootError("failed(preserve:copy_mismatch)", {"src": str(src), "dst": str(dst)})
    return {"src": str(src), "dst": str(dst), "sha256": dst_sha, "bytes": dst_size}


def preserve_artifacts(runtime: Path, dest: Path, *, extra_files: list[str] | None = None,
                       required: list[str] | None = None) -> dict:
    """Copy the root's ledgers/logs/scripts/configs (PRESERVE_DIRS x
    PRESERVE_SUFFIXES) plus every explicitly named final artifact into
    `dest` (outside the root), verifying each copy by re-reading it. EVERY
    named artifact is required: `extra_files` and `required` (absolute or
    root-relative) are the same list under two names, and a named file that
    is missing, a symlink, or outside the root is
    failed(preserve:required_missing) -- there is no silently-skipped
    optional artifact. Writes <dest>/preserved.json carrying the root's
    manifest_sha256; cleanup later re-verifies every listed copy before
    removing anything."""
    runtime = Path(runtime).resolve()
    dest = Path(dest).resolve()
    record = read_ownership(runtime)
    if record is None:
        raise FreshRootError("failed(preserve:not_owned)", str(runtime))
    if _inside(dest, runtime):
        raise FreshRootError("failed(preserve:dest_inside_runtime)", str(dest))
    wanted: dict[Path, Path] = {}
    for sub in PRESERVE_DIRS:
        root = runtime / sub
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.is_file() and not p.is_symlink() and p.suffix in PRESERVE_SUFFIXES:
                    wanted[p] = dest / p.relative_to(runtime)
    missing_required = []
    for raw in list(extra_files or []) + list(required or []):
        p = Path(raw)
        p = p if p.is_absolute() else runtime / p
        if not p.is_file() or p.is_symlink() or not _inside(p, runtime):
            missing_required.append(str(raw))
            continue
        wanted[p] = dest / p.resolve().relative_to(runtime)
    if missing_required:
        raise FreshRootError("failed(preserve:required_missing)", missing_required)
    files = [_copy_verified(src, dst) for src, dst in sorted(wanted.items())]
    payload = {"owner": OWNER_TAG, "runtime_root": str(runtime), "campaign_id": record.get("campaign_id"),
               "manifest_sha256": record.get("manifest_sha256"), "dest": str(dest), "at": utc_now(),
               "file_count": len(files), "bytes": sum(f["bytes"] for f in files), "files": files}
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / PRESERVED_FILE
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    payload["record"] = str(out)
    return payload


def verify_preserved(record_path: Path, runtime: Path) -> dict:
    """Re-read a preserved.json and every copy it lists (outside the root):
    all must exist with the recorded sha256 and size."""
    record_path = Path(record_path).resolve()
    runtime = Path(runtime).resolve()
    if not record_path.is_file():
        raise FreshRootError("failed(cleanup:no_preservation_record)", str(record_path))
    if _inside(record_path, runtime):
        raise FreshRootError("failed(cleanup:evidence_inside_runtime)", str(record_path))
    try:
        data = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FreshRootError("failed(cleanup:preservation_record_invalid)", str(exc))
    if data.get("owner") != OWNER_TAG or Path(data.get("runtime_root", "")).resolve() != runtime:
        raise FreshRootError("failed(cleanup:preservation_record_mismatch)",
                             {"record": data.get("runtime_root"), "path": str(runtime)})
    bad = []
    for f in data.get("files") or []:
        dst = Path(f["dst"])
        if _inside(dst, runtime) or not dst.is_file() or dst.stat().st_size != f["bytes"] or sha256_file(dst) != f["sha256"]:
            bad.append(str(dst))
    if bad:
        raise FreshRootError("failed(cleanup:evidence_not_durable)", {"unverified_copies": bad})
    return data


# ---------------------------------------------------------------------------
# cleanup (guarded)
# ---------------------------------------------------------------------------

def _proc_groups(entry: Path) -> tuple[int | None, int | None]:
    """(pgrp, session) of one /proc/<pid> from its world-readable `stat`;
    (None, None) when unreadable. `comm` may hold spaces/parens, so the
    numeric fields are taken after the last ')'."""
    try:
        stat = (entry / "stat").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    tail = stat.rsplit(")", 1)[-1].split()
    try:
        return int(tail[2]), int(tail[3])
    except (IndexError, ValueError):
        return None, None


# CapEff bits that let a process ignore directory permissions: with either,
# a 0700 root of another uid is still reachable (capabilities(7)).
CAP_DAC_OVERRIDE = 1
CAP_DAC_READ_SEARCH = 2
_DAC_CAP_MASK = (1 << CAP_DAC_OVERRIDE) | (1 << CAP_DAC_READ_SEARCH)


def root_privacy(root: Path) -> dict:
    """Mode/uid of the runtime root and whether it is PRIVATE to this uid
    (0700, owned by os.getuid()): the precondition for excluding a
    ptrace-opaque process of another uid from the quiescence scan."""
    st = os.lstat(root)
    mode = stat_module.S_IMODE(st.st_mode)
    return {"path": str(root), "mode": f"{mode:04o}", "uid": st.st_uid,
            "private": stat_module.S_ISDIR(st.st_mode) and mode == 0o700 and st.st_uid == os.getuid()}


def _proc_identity(entry: Path) -> dict:
    """uids (real/effective/saved/fs) and CapEff of one /proc/<pid> from its
    world-readable `status`; empty fields when unreadable or unparsable."""
    out: dict = {"uids": None, "cap_eff": None}
    try:
        text = (entry / "status").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        if line.startswith("Uid:"):
            try:
                uids = [int(x) for x in line.split()[1:5]]
            except ValueError:
                continue
            if len(uids) == 4:  # real/effective/saved/fs -- anything shorter is unparsable
                out["uids"] = uids
        elif line.startswith("CapEff:"):
            try:
                out["cap_eff"] = int(line.split()[1], 16)
            except (IndexError, ValueError):
                pass
    return out


def _unknown_use_census(pid: int, entry: Path, root_private: bool) -> dict:
    """Why a stat-readable but use-opaque pid can or cannot be excluded.
    Excluded ONLY when the root is private AND every uid of the process is
    neither 0 nor ours AND its effective capabilities carry neither
    CAP_DAC_OVERRIDE nor CAP_DAC_READ_SEARCH: such a process provably cannot
    traverse a 0700 directory of another uid. Everything else (uid 0, our
    own uid but ptrace-opaque, unreadable identity) stays UNKNOWN -- no
    userspace fact rules its use of the root out."""
    ident = _proc_identity(entry)
    me = os.getuid()
    uids = ident["uids"]
    cap = ident["cap_eff"]
    row = {"pid": pid, "uids": uids, "cap_eff": None if cap is None else f"{cap:016x}"}
    if uids is None or cap is None:
        row["reason"] = "identity_unreadable"
    elif 0 in uids:
        row["reason"] = "uid_0"
    elif me in uids:
        row["reason"] = "same_uid_ptrace_opaque"
    elif cap & _DAC_CAP_MASK:
        row["reason"] = "dac_capability"
    elif not root_private:
        row["reason"] = "root_not_private"
    else:
        row["reason"] = "excluded_by_perm"
    return row


def _pids_using(root: Path, proc_root: Path = Path("/proc"),
                owned_groups: list[int] | None = None, *, root_private: bool = False) -> dict:
    """Quiescence scan before removal. A live process counts as USING the
    root when its cwd / exe / root link resolves inside it, when any open fd
    (/proc/<pid>/fd/*) or mapped file (/proc/<pid>/maps) lies inside it, or
    when it belongs to a process group / session the campaign launched
    (`owned_groups`: the session ids of the monitored phase children, whose
    descendants may outlive them).

    Two readabilities are kept apart. `stat` (world-readable) proves GROUP
    IDENTITY only; it never vouches for root use. A pid counts as `inspected`
    for root use only when at least one USE probe (cwd/exe/root readlink,
    fd/ listing, maps) was readable. A pid with no readable use probe is
    classified: its /proc entry gone -> `vanished` (exited during the scan;
    ignored, an exited pid uses nothing); nothing readable at all, entry
    still present -> `uninspectable`; stat readable but every use probe
    denied (ptrace-opaque: uid 0, services, our own opaque helpers) ->
    `unknown_use` with a per-pid census. Of those, `excluded_by_perm`
    collects the pids that provably cannot reach a PRIVATE root (see
    `_unknown_use_census`); the rest remain unknown. `quiescence` is
    "total" only when no pid is uninspectable or unknown -- a partial scan
    is reported as partial, never as "nobody". Fails CLOSED on an unreadable
    proc root (failed(cleanup:proc_unreadable)). The remover's own pid is
    skipped."""
    proc_root = Path(proc_root)
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise FreshRootError("failed(cleanup:proc_unreadable)",
                             {"proc_root": str(proc_root), "error": str(exc)})
    groups = {int(g) for g in (owned_groups or [])}
    users: list[dict] = []
    uninspectable: list[int] = []
    vanished: list[int] = []
    unknown_use: list[dict] = []
    excluded: list[int] = []
    inspected = 0
    me = os.getpid()
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        pid = int(entry.name)
        identity_readable = False
        use_readable = 0
        hit: str | None = None
        pgrp, session = _proc_groups(entry)
        if pgrp is not None:
            identity_readable = True
            if pgrp in groups or session in groups:
                hit = f"group:{pgrp if pgrp in groups else session}"
        for link in ("cwd", "exe", "root"):
            try:
                target = Path(os.readlink(entry / link))
            except OSError:
                continue
            use_readable += 1
            if hit is None and _inside(target, root):
                hit = link
        try:
            fds = list((entry / "fd").iterdir())
            use_readable += 1
        except OSError:
            fds = []
        for fd in fds:
            try:
                target = Path(os.readlink(fd))
            except OSError:
                continue
            if hit is None and _inside(target, root):
                hit = f"fd:{fd.name}"
                break
        try:
            maps = (entry / "maps").read_text(encoding="utf-8", errors="replace")
            use_readable += 1
        except OSError:
            maps = ""
        if hit is None:
            for line in maps.splitlines():
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[5].startswith("/") and _inside(Path(parts[5]), root):
                    hit = "maps"
                    break
        if hit is not None:
            # an owned-group member or a positive use hit blocks regardless
            # of how much else was readable
            users.append({"pid": pid, "via": hit})
            if use_readable:
                inspected += 1
            continue
        if use_readable == 0:
            # nothing of its root use could be read: exited during the scan,
            # genuinely opaque, or identity-only -- only the first is safe
            # to ignore
            try:
                os.stat(entry)
            except FileNotFoundError:
                vanished.append(pid)  # its /proc entry is gone: the process exited
                continue
            except OSError as exc:
                if exc.errno == errno.ESRCH:
                    vanished.append(pid)
                    continue
                uninspectable.append(pid)  # still there, but not even stat-able: never "gone"
                continue
            if not identity_readable:
                uninspectable.append(pid)
                continue
            census = _unknown_use_census(pid, entry, root_private)
            if census["reason"] == "excluded_by_perm":
                excluded.append(pid)
            else:
                unknown_use.append(census)
            continue
        inspected += 1
    quiescence = "total" if not (uninspectable or unknown_use) else "partial"
    users.sort(key=lambda u: u["pid"])
    unknown_use.sort(key=lambda c: c["pid"])
    return {"users": users, "inspected": inspected, "uninspectable": sorted(uninspectable),
            "vanished": sorted(vanished), "unknown_use": unknown_use, "excluded_by_perm": sorted(excluded),
            "quiescence": quiescence,
            "root_private": root_private, "owned_groups": sorted(groups), "proc_root": str(proc_root)}


def _unescape_mountinfo(field: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def _mount_points_at_or_below(root: Path, mountinfo: Path) -> set[str]:
    """Mount points at or below `root` per the kernel mount table (field 5 =
    mount point, octal-escaped; catches bind mounts on the same device). An
    unreadable table fails CLOSED (failed(cleanup:mounts_unreadable))."""
    try:
        text = Path(mountinfo).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise FreshRootError("failed(cleanup:mounts_unreadable)", {"mountinfo": str(mountinfo), "error": str(exc)})
    found: set[str] = set()
    for line in text.splitlines():
        parts = line.split(" ")
        if len(parts) < 5:
            continue
        point = Path(_unescape_mountinfo(parts[4]))
        if point == root or _inside(point, root):
            found.add(str(point))
    return found


# --- handle-based removal primitives (Linux openat2 + *at operations) -----
# openat2(2) has the same syscall number on every Linux architecture (it was
# added in 5.6, after syscall-number unification).
_OPENAT2_NR = 437
_AT_FDCWD = -100
_RESOLVE_NO_XDEV, _RESOLVE_NO_MAGICLINKS, _RESOLVE_NO_SYMLINKS, _RESOLVE_BENEATH = 0x01, 0x02, 0x04, 0x08
_DIR_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_CHILD_RESOLVE = _RESOLVE_BENEATH | _RESOLVE_NO_SYMLINKS | _RESOLVE_NO_XDEV | _RESOLVE_NO_MAGICLINKS
_ABS_RESOLVE = _RESOLVE_NO_SYMLINKS | _RESOLVE_NO_MAGICLINKS
_UNSUPPORTED_ERRNOS = (errno.ENOSYS, errno.EINVAL, errno.E2BIG, errno.EPERM)


class _OpenHow(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint64), ("mode", ctypes.c_uint64), ("resolve", ctypes.c_uint64)]


def _openat2(dirfd: int, name: bytes, resolve: int) -> int:
    """openat2(dirfd, name, {O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC, resolve}).
    The kernel refuses, atomically at open time, what the pathname remover
    could not: RESOLVE_NO_SYMLINKS (any symlink component, the leaf included),
    RESOLVE_NO_XDEV (any mount point, bind mounts included: EXDEV),
    RESOLVE_BENEATH (no escape above dirfd). Raises OSError; ENOSYS/EINVAL/
    E2BIG mean the primitive is not available (caller refuses, no fallback)."""
    if sys.platform != "linux":
        raise OSError(errno.ENOSYS, "openat2 is Linux-only", name)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    how = _OpenHow(_DIR_OPEN_FLAGS, 0, resolve)
    fd = libc.syscall(ctypes.c_long(_OPENAT2_NR), ctypes.c_int(dirfd), ctypes.c_char_p(name),
                      ctypes.byref(how), ctypes.c_size_t(ctypes.sizeof(how)))
    if fd < 0:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e), name)
    return int(fd)


def _open_abs_dir(path: Path) -> int:
    """Directory handle for an absolute path with no symlink component
    (RESOLVE_NO_SYMLINKS): the one pathname resolution the remover performs."""
    return _openat2(_AT_FDCWD, os.fsencode(str(path)), _ABS_RESOLVE)


def _open_child_dir(dfd: int, name: str) -> int:
    """Handle for ONE child directory of an open directory handle: single
    component, no symlink, no mount crossing (EXDEV), no escape."""
    return _openat2(dfd, os.fsencode(name), _CHILD_RESOLVE)


def _stat_at(dfd: int, name: str) -> os.stat_result:
    return os.stat(name, dir_fd=dfd, follow_symlinks=False)


def _listdir_fd(dfd: int) -> list[str]:
    return os.listdir(dfd)


def _unlink_at(dfd: int, name: str) -> None:
    os.unlink(name, dir_fd=dfd)  # unlinkat: never follows; EISDIR on a directory, EBUSY on a mount point


def _rmdir_at(dfd: int, name: str) -> None:
    os.rmdir(name, dir_fd=dfd)  # unlinkat(AT_REMOVEDIR): EBUSY if mounted over, ENOTEMPTY if refilled


_REMOVE_NOTE = ("root retained; the entries counted in removed_files/removed_dirs are gone, everything else "
                "in the tree is still there. Every unlink/rmdir was a single-component operation relative to a "
                "verified no-follow directory handle, so the remover resolved no pathname outside the root and "
                "entered no mount point (bind mounts included) and no symlink; a mount placed over a directory "
                "after its handle was opened is not entered either (the handle stays bound to the underlying "
                "directory) and its rmdir fails EBUSY. NOT covered: content moved (renamed) INTO the root by a "
                "writer with access to the 0700 root before it was inspected is indistinguishable from owned "
                "content and is removed.")


def _stopped(reason: str, at: str, phase: str, acct: dict, exc: OSError | None = None) -> FreshRootError:
    detail = {"at": at, "reason": reason, "phase": phase,
              "errno": errno.errorcode.get(exc.errno, str(exc.errno)) if exc is not None and exc.errno else None,
              "removed": acct["removed_files"] + acct["removed_dirs"],
              "removed_files": acct["removed_files"], "removed_dirs": acct["removed_dirs"],
              "note": _REMOVE_NOTE}
    return FreshRootError("failed(cleanup:remove_stopped)", detail)


def _walk_handle(dfd: int, rel: str, root_dev: int, remove: bool, counts: dict, acct: dict,
                 skip: frozenset[str] = frozenset()) -> None:
    """Depth-first over an open directory handle. Every child directory is
    opened with `_open_child_dir` (atomic no-symlink/no-xdev/beneath check)
    and its handle identity (st_dev, st_ino) must equal the lstat that
    preceded the open; a mismatch, EXDEV, ENOTDIR/ELOOP or any other OSError
    stops the walk (failed(cleanup:remove_stopped)). With remove=False nothing
    is touched (verification pass); with remove=True files are unlinked and
    directories rmdir-ed by name relative to their parent handle, accounted
    in `acct` as they go. `skip` names top-level entries left for the caller
    (the ownership marker, removed last so a stopped removal leaves a root
    that is still recognisably owned)."""
    phase = "remove" if remove else "verify"
    try:
        names = sorted(n for n in _listdir_fd(dfd) if n not in skip)
    except OSError as exc:
        raise _stopped("unreadable_during_listing", rel, phase, acct, exc) from None
    for name in names:
        path = f"{rel}/{name}"
        try:
            st = _stat_at(dfd, name)
        except OSError as exc:
            raise _stopped("unreadable_during_listing", path, phase, acct, exc) from None
        if stat_module.S_ISDIR(st.st_mode):
            try:
                cfd = _open_child_dir(dfd, name)
            except OSError as exc:
                if exc.errno == errno.EXDEV:
                    raise _stopped("mount_or_device_boundary", path, phase, acct, exc) from None
                if exc.errno in (errno.ENOTDIR, errno.ELOOP, errno.ENOENT):
                    raise _stopped("entry_changed_during_removal", path, phase, acct, exc) from None
                if exc.errno in _UNSUPPORTED_ERRNOS:
                    raise FreshRootError("failed(cleanup:remove_unsupported)",
                                         {"primitive": "openat2", "at": path,
                                          "errno": errno.errorcode.get(exc.errno, str(exc.errno)),
                                          "removed": acct["removed_files"] + acct["removed_dirs"],
                                          "note": "no pathname fallback; root retained"}) from None
                raise _stopped(f"os_error: {exc.strerror or exc}", path, phase, acct, exc) from None
            try:
                cst = os.fstat(cfd)
                if (cst.st_dev, cst.st_ino) != (st.st_dev, st.st_ino):
                    raise _stopped("identity_changed", path, phase, acct)
                if cst.st_dev != root_dev:
                    raise _stopped("mount_or_device_boundary", path, phase, acct)
                _walk_handle(cfd, path, root_dev, remove, counts, acct)
            finally:
                os.close(cfd)
            if remove:
                try:
                    _rmdir_at(dfd, name)
                except OSError as exc:
                    raise _stopped(f"os_error: {exc.strerror or exc}", path, phase, acct, exc) from None
                acct["removed_dirs"] += 1
            counts["dirs"] += 1
        else:
            if remove:
                try:
                    _unlink_at(dfd, name)  # a symlink is unlinked as a link; its target is never touched
                except OSError as exc:
                    raise _stopped(f"os_error: {exc.strerror or exc}", path, phase, acct, exc) from None
                acct["removed_files"] += 1
            counts["files"] += 1


def _remove_tree_bounded(root: Path, mountinfo: Path) -> dict:
    """Removal of ONE owned root through verified directory handles, never
    through pathnames. The mount table is re-read first (unreadable ->
    failed(cleanup:mounts_unreadable); a mount at/below the root ->
    failed(cleanup:remove_stopped) with nothing removed). The root's parent
    is opened by absolute path with RESOLVE_NO_SYMLINKS, the root as its
    child with RESOLVE_BENEATH|NO_SYMLINKS|NO_XDEV, and its handle identity
    must match the root's lstat. A verification walk (nothing touched) then
    opens every directory the same way; any boundary or change stops with
    nothing removed. Only then does the removal walk run, unlinking files and
    rmdir-ing directories by single-component name relative to their parent
    handle; a stop during it retains the root with exact partial-removal
    accounting (removed_files/removed_dirs), see `_REMOVE_NOTE` for what that
    guarantees and what it does not. Where openat2 is unavailable the whole
    removal is refused (failed(cleanup:remove_unsupported)); there is no
    pathname fallback. Point-in-time checks (mount table, verification walk)
    remain point-in-time; the guarantees against a change DURING removal come
    from the kernel's per-open resolution and the handles, as stated in the
    note, and nothing more is claimed."""
    root = Path(root)
    root_st = os.lstat(root)
    root_dev = root_st.st_dev
    mounts = _mount_points_at_or_below(root, mountinfo)
    acct = {"removed_files": 0, "removed_dirs": 0}
    if mounts:
        raise FreshRootError("failed(cleanup:remove_stopped)",
                             {"at": sorted(mounts), "reason": "mount_point_appeared", "phase": "precheck",
                              "removed": 0, "removed_files": 0, "removed_dirs": 0})
    try:
        pfd = _open_abs_dir(root.parent)
    except OSError as exc:
        if exc.errno in _UNSUPPORTED_ERRNOS:
            raise FreshRootError("failed(cleanup:remove_unsupported)",
                                 {"primitive": "openat2", "at": str(root.parent),
                                  "errno": errno.errorcode.get(exc.errno, str(exc.errno)),
                                  "removed": 0, "note": "no pathname fallback; root retained"}) from None
        raise _stopped(f"os_error: {exc.strerror or exc}", str(root.parent), "open", acct, exc) from None
    try:
        try:
            rfd = _open_child_dir(pfd, root.name)
        except OSError as exc:
            reason = "mount_or_device_boundary" if exc.errno == errno.EXDEV else f"os_error: {exc.strerror or exc}"
            raise _stopped(reason, ".", "open", acct, exc) from None
        try:
            rst = os.fstat(rfd)
            if (rst.st_dev, rst.st_ino) != (root_st.st_dev, root_st.st_ino):
                raise _stopped("identity_changed", ".", "open", acct)
            marker = frozenset({OWNERSHIP_FILE})
            counts = {"files": 0, "dirs": 0}
            _walk_handle(rfd, ".", root_dev, False, counts, acct, marker)   # verify: nothing touched
            seen = {"files": 0, "dirs": 0}
            _walk_handle(rfd, ".", root_dev, True, seen, acct, marker)      # remove, by handle
            try:  # the ownership marker goes last: a stopped removal leaves an owned, retryable root
                _unlink_at(rfd, OWNERSHIP_FILE)
            except OSError as exc:
                raise _stopped(f"os_error: {exc.strerror or exc}", f"./{OWNERSHIP_FILE}", "remove", acct, exc) from None
            acct["removed_files"] += 1
            seen["files"] += 1
            counts["files"] += 1
        finally:
            os.close(rfd)
        try:
            _rmdir_at(pfd, root.name)
        except OSError as exc:
            raise _stopped(f"os_error: {exc.strerror or exc}", ".", "remove", acct, exc) from None
        acct["removed_dirs"] += 1
    finally:
        os.close(pfd)
    return {"files": seen["files"], "dirs": seen["dirs"] + 1,
            "removed": acct["removed_files"] + acct["removed_dirs"],
            "removed_files": acct["removed_files"], "removed_dirs": acct["removed_dirs"],
            "verified": counts,
            "primitive": "openat2(RESOLVE_BENEATH|NO_SYMLINKS|NO_XDEV|NO_MAGICLINKS) + unlinkat/rmdir relative to handles"}


def mounts_inside(root: Path, mountinfo: Path = Path("/proc/self/mountinfo")) -> dict:
    """Mount points at or below `root`, found BEFORE any removal: from the
    kernel's mount table (`mountinfo`, field 5 = mount point, octal-escaped;
    catches bind mounts on the same device) and from a device-boundary walk
    (an entry whose st_dev differs from the root's). An unreadable mount
    table fails CLOSED (failed(cleanup:mounts_unreadable)): removal must not
    recurse into something whose mount status is unknown."""
    root = Path(root)
    found = _mount_points_at_or_below(root, mountinfo)
    root_dev = os.lstat(root).st_dev
    unreadable: list[str] = []

    def _onerror(exc: OSError) -> None:
        unreadable.append(f"{getattr(exc, 'filename', None) or root}: {exc.strerror or exc}")

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=_onerror):
        keep = []
        for name in dirnames:
            p = Path(dirpath) / name
            try:
                st = os.lstat(p)
            except OSError as exc:
                _onerror(exc)
                continue
            if st.st_dev != root_dev or str(p) in found:
                found.add(str(p))  # a mount point: recorded, never descended into
            else:
                keep.append(name)
        dirnames[:] = keep
        for name in filenames:
            p = Path(dirpath) / name
            try:
                st = os.lstat(p)
            except OSError as exc:
                _onerror(exc)
                continue
            if st.st_dev != root_dev:
                found.add(str(p))
    if unreadable:
        # a subtree whose mount status could not be established is not known
        # to be mount-free: fail closed like an unreadable mount table
        raise FreshRootError("failed(cleanup:mounts_unreadable)",
                             {"mountinfo": str(mountinfo), "unreadable": unreadable[:20], "mounts": sorted(found)})
    return {"mountinfo": str(mountinfo), "mounts": sorted(found)}


def ledger_carries_manifest(ledger: Path, manifest_sha256: str) -> bool:
    if not ledger.is_file():
        return False
    for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("manifest_sha256") == manifest_sha256:
            return True
    return False


def _canonical_no_symlink(path: Path, klass: str) -> Path:
    """Absolute, `..`-free path that must already BE its own realpath: any
    symlink component (the leaf included) is refused -- a symlinked root or
    parent could point the removal at something that is not the owned copy."""
    absolute = Path(os.path.normpath(os.path.abspath(str(path))))
    real = Path(os.path.realpath(str(absolute)))
    if real != absolute or absolute.is_symlink():
        raise FreshRootError(klass, {"given": str(path), "absolute": str(absolute), "realpath": str(real)})
    return real


def cleanup(runtime: Path, evidence_ledger: Path, *, preserved: Path | None = None,
            owned_registry: Path | None = None, allowed_parent: Path | None = None,
            protected: list[Path] | None = None, proc_root: Path = Path("/proc"),
            owned_groups: list[int] | None = None,
            mountinfo: Path = Path("/proc/self/mountinfo")) -> dict:
    """Remove ONE owned runtime root after every guard passes. Guards are
    fail-closed and each has its own failure class so a refusal is auditable.

    Bounded removal: the root must be a symlink-free canonical path whose
    DIRECT parent equals the canonical `allowed_parent` (the campaign's
    runtime parent), and it must be bound in the campaign-external
    `owned_registry` (written at materialize, outside every root) with the
    same canonical root, parent, manifest_sha256 and campaign_id as the
    in-root marker -- the marker alone is never trusted.
    Durable evidence = the campaign ledger OUTSIDE the root carrying this
    root's manifest_sha256 AND a
    preserved.json (`preserve`) whose every copy re-verifies by hash -- a
    ledger row alone is not proof that the logs and final artifacts made it
    out. Mounts: any mount point at or below the root (kernel mount table +
    device-boundary walk) refuses BEFORE removal (failed(cleanup:mount_inside_root));
    an unreadable mount table refuses too. Quiescence: no live process may
    touch the root (cwd/exe/root, open fds, mapped files) or belong to a
    launched `owned_groups` session; an unreadable proc root refuses, a
    still-present pid nothing could be read of refuses
    (failed(cleanup:proc_uninspectable)), and so does a pid whose root use
    could not be inspected and cannot be excluded by the root's 0700
    permission (failed(cleanup:proc_unknown_use): uid 0, our own
    ptrace-opaque uid, DAC-capable or unidentifiable processes -- the scan
    reports its census and quiescence "partial"; it never claims total
    quiescence it did not establish); a pid that exited during the scan is
    ignored. Removal itself (`_remove_tree_bounded`) never resolves a
    pathname below the root: the mount table is re-read, the parent and root
    are opened as no-symlink handles, every directory is opened from its
    parent handle with openat2(RESOLVE_BENEATH|NO_SYMLINKS|NO_XDEV) and its
    identity checked against the preceding lstat, a verification walk runs
    with nothing touched, and only then are entries unlinked/rmdir-ed by
    single-component name relative to their parent handle. Any boundary,
    swap or error stops (failed(cleanup:remove_stopped)) with exact
    partial-removal accounting; where openat2 is unavailable the removal is
    refused outright (failed(cleanup:remove_unsupported)), never done by
    pathname. THREAT BOUNDARY (see `_REMOVE_NOTE`): a symlink or mount point
    (bind mounts included) is refused at open time by the kernel; a mount
    placed over a directory whose handle is already open is not entered and
    its rmdir fails; a directory swapped between its lstat and its open is
    caught by identity. NOT covered: content moved into the 0700 root by a
    writer with access to it before it is inspected is removed as owned
    content; the pre-checks stay point-in-time, and partial removal before a
    stop is real -- "root retained" never means "contents retained". Proof
    of removal is the owned path's absence afterwards; the df
    numbers in the result are diagnostics, never a pass/fail signal -- and a
    completed removal is not evidence that nothing was using the tree, which
    is why the scan comes first."""
    given = Path(runtime)
    if not given.is_dir():
        raise FreshRootError("failed(cleanup:not_a_dir)", str(given))
    runtime = _canonical_no_symlink(given, "failed(cleanup:symlink_root)")
    ledger = Path(evidence_ledger).resolve()
    if runtime == Path("/") or runtime == Path.home().resolve() or runtime.parent == runtime:
        raise FreshRootError("failed(cleanup:refused_path)", str(runtime))
    if allowed_parent is None:
        raise FreshRootError("failed(cleanup:no_allowed_parent)",
                             "cleanup is bounded to one canonical runtime parent; none given")
    parent = _canonical_no_symlink(Path(allowed_parent), "failed(cleanup:symlink_parent)")
    if runtime.parent != parent:
        raise FreshRootError("failed(cleanup:outside_allowed_parent)",
                             {"root_parent": str(runtime.parent), "allowed_parent": str(parent)})
    record = read_ownership(runtime)
    if record is None or record.get("owner") != OWNER_TAG:
        raise FreshRootError("failed(cleanup:not_owned)", str(runtime))
    if Path(record.get("runtime_root", "")).resolve() != runtime:
        raise FreshRootError("failed(cleanup:ownership_mismatch)",
                             {"record": record.get("runtime_root"), "path": str(runtime)})
    if owned_registry is None:
        raise FreshRootError("failed(cleanup:no_ownership_registry)",
                             "cleanup needs the campaign-external owned_roots.json written at materialize")
    registry = Path(owned_registry).resolve()
    if _inside(registry, runtime):
        raise FreshRootError("failed(cleanup:registry_inside_runtime)", str(registry))
    if not registry.is_file():
        raise FreshRootError("failed(cleanup:not_registered)", {"registry": str(registry), "root": str(runtime)})
    entry = registered_entry(registry, runtime)
    if entry is None:
        raise FreshRootError("failed(cleanup:not_registered)", {"registry": str(registry), "root": str(runtime)})
    expected = {"runtime_root": str(runtime), "runtime_parent": str(parent),
                "manifest_sha256": record.get("manifest_sha256"), "campaign_id": record.get("campaign_id")}
    mismatch = {k: {"registry": entry.get(k), "root": v} for k, v in expected.items() if entry.get(k) != v}
    if mismatch or entry.get("cleaned_utc") is not None:
        raise FreshRootError("failed(cleanup:registry_mismatch)",
                             mismatch or {"cleaned_utc": entry.get("cleaned_utc")})
    if (runtime / ".git").exists():
        raise FreshRootError("failed(cleanup:git_repo)", str(runtime))
    for prot in (protected or []):
        prot = Path(prot).resolve()
        if runtime == prot or _inside(prot, runtime):
            raise FreshRootError("failed(cleanup:protected_path)", str(prot))
    if _inside(ledger, runtime):
        raise FreshRootError("failed(cleanup:evidence_inside_runtime)", str(ledger))
    if not ledger_carries_manifest(ledger, record["manifest_sha256"]):
        raise FreshRootError("failed(cleanup:evidence_not_durable)",
                             {"ledger": str(ledger), "manifest_sha256": record["manifest_sha256"]})
    if preserved is None:
        raise FreshRootError("failed(cleanup:no_preservation_record)",
                             "cleanup needs the preserved.json written by `preserve` (outside the root)")
    kept = verify_preserved(preserved, runtime)
    if kept.get("manifest_sha256") != record["manifest_sha256"]:
        raise FreshRootError("failed(cleanup:preservation_record_mismatch)",
                             {"preserved": kept.get("manifest_sha256"), "root": record["manifest_sha256"]})
    # mount points are refused BEFORE any recursion: rmtree crossing into a
    # mounted filesystem would delete data that is not this root's
    mounts = mounts_inside(runtime, mountinfo)
    if mounts["mounts"]:
        raise FreshRootError("failed(cleanup:mount_inside_root)", mounts)
    privacy = root_privacy(runtime)
    scan = _pids_using(runtime, proc_root, owned_groups, root_private=privacy["private"])
    if scan["users"]:
        raise FreshRootError("failed(cleanup:in_use)", scan["users"])
    if scan["uninspectable"]:
        # still-present pids we could not read anything of: not "nobody",
        # so the root is retained (an exited-during-scan pid is `vanished`
        # and does not block -- see _pids_using)
        raise FreshRootError("failed(cleanup:proc_uninspectable)",
                             {"pids": scan["uninspectable"], "vanished": scan["vanished"],
                              "proc_root": scan["proc_root"]})
    if scan["unknown_use"]:
        # identity readable, root use NOT inspectable, no permission-based
        # exclusion: the scan is partial and says so; the root is retained
        # with the census until a justified exclusion exists (none does for
        # uid 0 or a ptrace-opaque process of our own uid)
        raise FreshRootError("failed(cleanup:proc_unknown_use)",
                             {"pids": [c["pid"] for c in scan["unknown_use"]], "census": scan["unknown_use"],
                              "excluded_by_perm": scan["excluded_by_perm"], "root": privacy,
                              "quiescence": scan["quiescence"], "proc_root": scan["proc_root"],
                              "note": "root use of these pids could not be read and cannot be ruled out; "
                                      "no total quiescence is claimed"})
    size = du_bytes(runtime)
    children = sorted(p.name for p in runtime.iterdir())
    df_before = disk_free_bytes(runtime.parent)
    # handle-based removal: mount table re-read, verified no-follow directory
    # handles, single-component *at operations, kernel-refused symlink/mount
    # crossing; any stop retains the root with partial-removal accounting
    # (failed(cleanup:remove_stopped)); no pathname fallback where openat2 is
    # missing (failed(cleanup:remove_unsupported))
    removal = _remove_tree_bounded(runtime, mountinfo)
    if runtime.exists():
        surviving = sorted(str(p) for p in runtime.rglob("*"))[:50]
        raise FreshRootError("failed(cleanup:incomplete)", {"surviving": surviving or [str(runtime)]})
    df_after = disk_free_bytes(runtime.parent)
    try:  # audit trail in the external registry (best effort; absence is the proof)
        data = _load_registry(registry)
        data["roots"][str(runtime)]["cleaned_utc"] = utc_now()
        registry.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, KeyError, FreshRootError):
        pass
    result = {"ok": True, "state": "cleaned", "removed": str(runtime), "absent": not runtime.exists(),
              "allowed_parent": str(parent), "owned_registry": str(registry),
              "removed_children": children, "bytes_freed": size, "removal": removal,
              "process_scan": {k: v for k, v in scan.items() if k != "users"},
              "preserved": {"record": str(Path(preserved).resolve()), "file_count": kept.get("file_count"),
                            "bytes": kept.get("bytes")},
              "df_diagnostic_gib": {"before": round(df_before / GIB, 3), "after": round(df_after / GIB, 3),
                                    "note": "level/delta recorded for diagnosis only; proof of cleanup is `absent`"},
              "manifest_sha256": record["manifest_sha256"], "campaign_id": record.get("campaign_id"),
              "at": utc_now()}
    note = ledger.parent / f"cleanup_{runtime.name}.json"
    try:
        note.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        result["record"] = str(note)
    except OSError:
        pass
    return result


def du_bytes(root: Path, errors: list[str] | None = None) -> int:
    """Hardlink-aware apparent usage of one tree: each inode counted once
    -- the same answer `du -s` gives on one root.
    Symlinks are never followed. Anything that could not be listed or
    stat'ed is skipped AND, when `errors` is given, appended to it -- a
    caller that needs a COMPLETE measurement (a budget) must treat a
    non-empty `errors` as a partial answer, never as a small one."""
    seen: set[tuple[int, int]] = set()
    total = 0

    def _onerror(exc: OSError) -> None:
        if errors is not None:
            errors.append(f"{getattr(exc, 'filename', None) or root}: {exc.strerror or exc}")

    for dirpath, dirnames, filenames in os.walk(root, onerror=_onerror):
        for name in filenames:
            p = Path(dirpath) / name
            try:
                st = p.lstat()
            except OSError as exc:
                _onerror(exc)
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += st.st_size
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _emit(obj: dict) -> int:
    print(json.dumps(obj, sort_keys=True))
    return 0 if obj.get("ok") else 1


def _fail(exc: FreshRootError) -> int:
    return _emit({"ok": False, "state": exc.state, "detail": exc.detail})


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        key, _, value = line[len("export "):].partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]),
                    help="hub clone whose working tree is snapshotted (default: this clone)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("allowlist", help="write the approved new-path allowlist")
    p.add_argument("--out", default=None)

    p = sub.add_parser("snapshot", help="manifest + canonical tarball (or --check only)")
    p.add_argument("--check", action="store_true", help="run every check, write nothing")
    p.add_argument("--allowlist", default=None)
    p.add_argument("--campaign-dir", default=None, help="default: <repo>/.sweep/campaign_<id>")
    p.add_argument("--campaign-id", default=None)

    p = sub.add_parser("materialize", help="extract a snapshot into an empty runtime root")
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--runtime", required=True)
    p.add_argument("--campaign-id", required=True)
    p.add_argument("--allowlist", default=None)
    p.add_argument("--allow-stale", action="store_true",
                   help="record (instead of refuse) a working tree that no longer matches the snapshot; "
                        "the working-tree recheck itself always runs")
    p.add_argument("--owned-registry", default=None,
                   help="campaign-external owned_roots.json (default: <campaign_dir>/owned_roots.json)")

    p = sub.add_parser("seed-cache", help="copy declared native-cache subsets into the runtime")
    p.add_argument("--runtime", required=True)
    p.add_argument("--item", action="append", default=[], metavar="DST_REL=SRC_ABS")
    p.add_argument("--spec", default=None, help="JSON file {dst_rel: src_abs}")

    p = sub.add_parser("verify", help="routes / isolation / sources checks")
    p.add_argument("--runtime", required=True)
    p.add_argument("--routes", action="store_true")
    p.add_argument("--isolation", action="store_true")
    p.add_argument("--sources", action="store_true")
    p.add_argument("--env-file", default=None, help="sh export file to read routes from (default: current environment)")
    p.add_argument("--conda-cmd", default=None, help="TEST ONLY: replacement for `conda`")
    p.add_argument("--python", default=None, help="interpreter that runs the isolation probe (default: this one)")

    p = sub.add_parser("preserve", help="copy logs + named final artifacts out of a runtime root, verified by hash")
    p.add_argument("--runtime", required=True)
    p.add_argument("--dest", required=True, help="destination dir OUTSIDE the root (e.g. the campaign dir)")
    p.add_argument("--file", action="append", default=[],
                   help="named final artifact to preserve; REQUIRED to exist (repeatable)")
    p.add_argument("--require", action="append", default=[],
                   help="same as --file (kept for readability; every named artifact is required)")

    p = sub.add_parser("cleanup", help="remove one owned runtime root (guarded)")
    p.add_argument("--runtime", required=True)
    p.add_argument("--evidence-ledger", required=True)
    p.add_argument("--preserved", required=True, help="preserved.json written by `preserve`")
    p.add_argument("--owned-registry", required=True, help="campaign-external owned_roots.json written at materialize")
    p.add_argument("--allowed-parent", required=True, help="the campaign's canonical runtime parent; the root's direct parent must equal it")
    p.add_argument("--protect", action="append", default=[])
    p.add_argument("--owned-group", action="append", type=int, default=[],
                   help="session/process-group id the campaign launched; any live member refuses cleanup (repeatable)")

    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()

    try:
        if args.cmd == "allowlist":
            out = Path(args.out) if args.out else default_allowlist_path(repo)
            out.parent.mkdir(parents=True, exist_ok=True)
            data = render_allowlist()
            out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
            return _emit({"ok": True, "allowlist": str(out), "patterns": data["patterns"]})

        if args.cmd == "snapshot":
            allow_path = Path(args.allowlist) if args.allowlist else default_allowlist_path(repo)
            patterns = load_allowlist(allow_path)
            manifest = build_manifest(repo, patterns)
            summary = {k: v for k, v in manifest.items() if k != "files"}
            summary["allowlist"] = str(allow_path) if allow_path.is_file() else None
            if args.check:
                return _emit({"ok": True, "state": "check_passed", **summary})
            cid = args.campaign_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            campaign_dir = Path(args.campaign_dir) if args.campaign_dir else repo / ".sweep" / f"campaign_{cid}"
            written = write_snapshot(repo, manifest, campaign_dir)
            return _emit({"ok": True, "state": "snapshot_written", "campaign_id": cid, **summary, **written})

        if args.cmd == "materialize":
            patterns = load_allowlist(Path(args.allowlist) if args.allowlist else default_allowlist_path(repo))
            result = materialize(Path(args.snapshot_dir), Path(args.runtime), campaign_id=args.campaign_id,
                                 repo=repo, allowlist=patterns, allow_stale=args.allow_stale,
                                 owned_registry=Path(args.owned_registry) if args.owned_registry else None)
            return _emit({"ok": True, "state": "materialized", **result})

        if args.cmd == "seed-cache":
            items: dict[str, str] = {}
            if args.spec:
                items.update(json.loads(Path(args.spec).read_text(encoding="utf-8")))
            for item in args.item:
                dst, _, src = item.partition("=")
                items[dst] = src
            record = seed_cache(Path(args.runtime), items)
            return _emit({"ok": True, "state": "seeded", **record})

        if args.cmd == "verify":
            runtime = Path(args.runtime)
            results: dict[str, dict] = {}
            if args.routes:
                env = parse_env_file(Path(args.env_file)) if args.env_file else dict(os.environ)
                results["routes"] = verify_routes(runtime, env, conda_cmd=args.conda_cmd.split() if args.conda_cmd else None)
            if args.isolation:
                results["isolation"] = verify_isolation(runtime, python=args.python)
            if args.sources:
                results["sources"] = verify_sources(runtime)
            if not results:
                return _emit({"ok": False, "state": "failed(verify:nothing_requested)"})
            ok = all(r["ok"] for r in results.values())
            state = "passed" if ok else next(r["state"] for r in results.values() if not r["ok"])
            return _emit({"ok": ok, "state": state, "checks": results})

        if args.cmd == "preserve":
            record = preserve_artifacts(Path(args.runtime), Path(args.dest), extra_files=args.file,
                                        required=args.require)
            return _emit({"ok": True, "state": "preserved", **{k: v for k, v in record.items() if k != "files"},
                          "file_count": record["file_count"]})

        if args.cmd == "cleanup":
            result = cleanup(Path(args.runtime), Path(args.evidence_ledger), preserved=Path(args.preserved),
                             owned_registry=Path(args.owned_registry), allowed_parent=Path(args.allowed_parent),
                             protected=[repo, *map(Path, args.protect)], owned_groups=args.owned_group)
            return _emit(result)
    except FreshRootError as exc:
        return _fail(exc)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
