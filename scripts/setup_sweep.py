#!/usr/bin/env python3
"""setup_sweep.py -- deterministic batch driver for `setup all` sweeps.

Complete-then-batch-recover (the consensus decision): the driver NEVER stops
on a failed target. Every phase of every target is appended to a JSONL ledger,
and the final report is generated STRICTLY from that ledger -- never from
anyone's memory of what happened. Agent recovery reasoning runs only after
the sweep completes, over the ledger's `failed` entries.

Per target (sequential -- parallel conda solves corrupt the error signal):
  1. gated bookkeeping: a gated model with NO HF token available is recorded
     `skipped_gated` (the approval gate upstream should already have excluded
     it; double bookkeeping is intentional honesty). With a token it proceeds.
  2. `./install.sh <env>`   -> ledger line (phase "install")
  3. `setup_verify.py <target> --json` -> ledger line (phase "verify",
     verdict embedded). setup_verify owns ALL verify judgment; this driver
     records returncodes and verdicts, it never re-judges.

The sweep itself always exits 0 (completion is the contract; per-target facts
live in the ledger -- same convention as setup_survey/setup_guardrail).

Usage:
  python3 scripts/setup_sweep.py --targets MACE,SevenNet,ORB
  python3 scripts/setup_sweep.py report [--ledger PATH]
  python3 scripts/setup_sweep.py --fresh-root --targets mace --ft-dataset ft_demo.traj \\
      [--preflight ../.omc/state/omm-e2e-preflight.json] [--campaign-id ID]

Test-only: --install-cmd / --verify-cmd replace the real commands (explicit
injection, no monkeypatching); the target/env is appended as the last arg.

--fresh-root (plan G2, ADR-OMM-05): one DISPOSABLE runtime root per env,
serialized. Per target env the cycle is, in this exact order, each step a
ledger row carrying the snapshot's `manifest_sha256`:

  start gate: MEASURED free space (GiB) >= G0 conservative peak + start
  reserve, and >= the mid-cycle floor (a planned/projected figure is never
  used) -> snapshot --check (fail-closed, before EVERY materialization) ->
  snapshot -> materialize (W/.omm_fresh/<env>_<runid>) -> seed-cache (copy
  only) -> routes verify (every cache/tmp/pkgs route inside the root, conda
  pkgs_dirs asserted) -> isolation pre-check (the runtime copy's OWN
  registry.resolve()) -> install.sh <env> (inside the root, its own process
  group, free space re-measured while it runs) -> isolation post-check ->
  setup_verify per variant --no-local-record (kind=inference rows) ->
  ft_sweep.py (kind=finetune rows) -> source-hash verify (drift =>
  failed(source_mutated)) -> targeted hashes of the real hub's local state +
  the user env's conda-meta/history before/after (change =>
  failed(attribution), campaign PAUSES) -> budget row (du peak, wall time vs
  the G0 band, GiB) -> PRESERVE: verified copies of every log/ledger/config
  and every final artifact (fine-tune checkpoints, ...) into the campaign dir
  -> guarded owned cleanup (needs the preserved record; refusals are
  recorded, never forced) -> post-cleanup row (owned path absent? + df).

Start-gate addends (GiB, each with provenance in the budget row): the cold
isolated-HOME caches this family re-downloads (MEASURED on the host at the
known `~/.cache/<name>` paths, symlinks never followed; a symlinked or
unreadable path makes the measurement PARTIAL, which is a lower bound and
never a budget -- the cycle is resource-blocked unless --cold-cache-gib
supplies an explicit conservative estimate) and the first-run fine-tune
downloads (FIRST_RUN_FT_DOWNLOADS: measured at deterministic host cache
paths, times the copies the first run persists, or `unknown` -- a family
whose fetch audit is incomplete is unknown, never zero; an unknown item
resource-blocks the fine-tune phase unless --ft-download-gib supplies an
explicit conservative bound. --ft-download-gib works for every family and
is ADDITIVE: it is added to the measured known part, never replaces it,
and is the bound for the unknown items; --peak-gib never covers
downloads). The peak itself comes from the G0 preflight or, for an env it
does not estimate, ONLY from --peak-gib (an explicit conservative figure):
without one the cycle is resource-blocked, because the floor monitor stops
a filling disk but is not proof that the install/FT peak fits.
The fine-tune rows' checkpoint, rerun script and ft_run.json provenance
record are REQUIRED preserved artifacts.

After install, the OWNED env's dependency inventory (conda list --explicit,
pip freeze, pip list --format=json, the registry's resolve() answers under
the owned interpreter) is written to <root>/.sweep/inventory/ with every
output line scrubbed of credential shapes BEFORE it is persisted (phase
logs included); the files are REQUIRED preserved artifacts. Inventories
record what was installed; they do not close the dependency-replay gap.

Between phases and while any child runs, the measured free space is
re-checked against --min-free-gib: below it the child's process group is
stopped in order (SIGTERM, grace, SIGKILL), the remaining variants are
recorded resource-blocked, and the preserve -> cleanup tail still runs.
SIGTERM/SIGINT to the driver do the same, then retain the root and stop.

The campaign ledger lives OUTSIDE every runtime root
(<hub>/.sweep/campaign_<id>/ledger.jsonl), so evidence survives cleanup by
construction, and cleanup itself refuses until that ledger carries the root's
manifest_sha256 AND the preserved copies re-verify by hash. Real hub and
user envs are never written: the only writable prefixes are the runtime root
and the campaign dir.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fresh_root  # noqa: E402
from _setup_common import load_local_env_map, resolve_home, utc_now  # noqa: E402
from setup_survey import token_source  # noqa: E402
from setup_verify import family_versions, find_env  # noqa: E402

STDERR_TAIL_CHARS = 2000


def next_ledger_path(home: Path) -> Path:
    sweep_dir = home / ".sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(sweep_dir.glob("setup_sweep_*.jsonl"))
    if existing:
        last = existing[-1].stem.rsplit("_", 1)[-1]
        run_id = int(last) + 1 if last.isdigit() else len(existing) + 1
    else:
        run_id = 1
    return sweep_dir / f"setup_sweep_{run_id:04d}.jsonl"


def latest_ledger_path(home: Path) -> Path | None:
    existing = sorted((home / ".sweep").glob("setup_sweep_*.jsonl"))
    return existing[-1] if existing else None


def append(ledger: Path, record: dict) -> None:
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def is_gated(model: str, home: Path) -> bool:
    registry = json.loads((home / "models.json").read_text())
    want = model.lower()
    for family, spec in registry.items():
        if family.startswith("_"):
            continue
        versions = spec.get("versions") or {}
        if family.lower() == want:
            return any(bool(v.get("gated")) for v in versions.values())
        for version, v in versions.items():
            if version.lower() == want:
                return bool(v.get("gated"))
    return False


def run_phase(command: list[str], cwd: Path, env: dict | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr[-STDERR_TAIL_CHARS:]


def last_json_line(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def sweep(targets: list[str], home: Path, ledger: Path,
          install_cmd: list[str] | None, verify_cmd: list[str] | None,
          min_free_gb: float = 10.0) -> None:
    append(ledger, {"seq": 0, "phase": "plan", "targets": targets, "at": utc_now()})
    token_missing = token_source() == "none"
    seq = 0
    for idx, target in enumerate(targets):
        # Disk floor precheck: below min_free_gb every further install is a
        # guaranteed noisy failure, so record THIS and all remaining targets
        # as skipped_disk (honest ledger, no silent truncation) and stop.
        free_gb = shutil.disk_usage(home).free / 1024**3
        if free_gb < min_free_gb:
            for remaining in targets[idx:]:
                seq += 1
                append(ledger, {"seq": seq, "target": remaining,
                                "env": find_env(remaining, home),
                                "phase": "skipped_disk", "returncode": None,
                                "stderr_tail": f"free disk {free_gb:.1f} GB < {min_free_gb:.1f} GB floor",
                                "verdict": None})
            break
        env_name = find_env(target, home)
        if env_name is None:
            seq += 1
            append(ledger, {"seq": seq, "target": target, "env": None,
                            "phase": "resolve", "returncode": 1,
                            "stderr_tail": f"unknown model: {target}", "verdict": None})
            continue
        if is_gated(target, home) and token_missing:
            seq += 1
            append(ledger, {"seq": seq, "target": target, "env": env_name,
                            "phase": "skipped_gated", "returncode": None,
                            "stderr_tail": "", "verdict": None})
            continue

        install = (install_cmd or [str(home / "install.sh")]) + [env_name]
        rc, _out, err = run_phase(install, home)
        seq += 1
        append(ledger, {"seq": seq, "target": target, "env": env_name,
                        "phase": "install", "returncode": rc,
                        "stderr_tail": err, "verdict": None})
        if rc != 0:
            continue  # NEVER stops the sweep; this target is simply failed

        verify = (verify_cmd or [sys.executable,
                                 str(home / "scripts" / "setup_verify.py")]) + [target, "--json"]
        rc, out, err = run_phase(verify, home)
        seq += 1
        append(ledger, {"seq": seq, "target": target, "env": env_name,
                        "phase": "verify", "returncode": rc,
                        "stderr_tail": err, "verdict": last_json_line(out)})


# ---------------------------------------------------------------------------
# --fresh-root: disposable-root cycle (plan G2)
# ---------------------------------------------------------------------------

LOCK_NAME = "campaign.lock"
FOREIGN_PROCESS_MARKERS = ("install.sh", "ft_run.py")
GIB = fresh_root.GIB


class CycleAbort(Exception):
    """Raised inside a cycle to stop compute with a ledger state; the cycle's
    preserve -> cleanup tail still runs."""

    def __init__(self, state: str, detail: str = ""):
        super().__init__(state)
        self.state, self.detail = state, detail


class FreshHooks:
    """Every external dependency of a cycle, explicitly injectable (the test
    suite runs whole cycles GPU-free against a throwaway git repo). A plain
    class, not a dataclass: the tests load this module through
    importlib.spec_from_file_location without registering it in sys.modules,
    which the dataclass decorator cannot tolerate under postponed annotations.

    Disk numbers are GiB (2^30) throughout, matching the G0 preflight's
    schema-2 `*_gib` fields. `min_free_gib` is the MEASURED mid-cycle floor
    (checked before every phase and while every child runs); `start_reserve_gib`
    is the margin above an env's conservative peak estimate that the measured
    free space must clear before the cycle starts."""

    def __init__(self, *, install_cmd: list[str] | None = None, verify_cmd: list[str] | None = None,
                 ft_cmd: list[str] | None = None, ft_extra_args: list[str] | None = None,
                 conda_cmd: list[str] | None = None, proc_root: Path = Path("/proc"),
                 seed_items: dict | None = None, preflight: dict | None = None,
                 ft_dataset: Path | None = None, ft_audit: Path | None = None, cleanup: bool = True,
                 min_free_gib: float = 10.0, start_reserve_gib: float = 15.0,
                 df_poll_seconds: float = 15.0, du_sample_seconds: float = 30.0,
                 grace_seconds: float = 20.0, probe_python: str | None = None,
                 disk_free=None, host_home: Path | None = None,
                 cold_cache_gib: float | None = None, ft_download_gib: float | None = None,
                 peak_gib: float | None = None, inventory_cmd: list[str] | None = None):
        self.install_cmd = install_cmd
        self.verify_cmd = verify_cmd
        self.ft_cmd = ft_cmd
        self.ft_extra_args = list(ft_extra_args or [])
        self.conda_cmd = conda_cmd
        self.proc_root = proc_root
        self.seed_items = dict(seed_items or {})
        self.preflight = preflight
        self.ft_dataset = ft_dataset
        self.ft_audit = ft_audit
        self.cleanup = cleanup
        self.min_free_gib = min_free_gib
        self.start_reserve_gib = start_reserve_gib
        self.df_poll_seconds = df_poll_seconds
        self.du_sample_seconds = du_sample_seconds
        self.grace_seconds = grace_seconds
        self.probe_python = probe_python
        # TEST ONLY: replacement for the conda/pip inventory tools (<kind> <prefix> <python>)
        self.inventory_cmd = inventory_cmd
        # measured free bytes for a path (TEST ONLY override: simulate a filling disk)
        self.disk_free = disk_free or fresh_root.disk_free_bytes
        # the user's real HOME: only MEASURED there (cold-cache accounting),
        # never written; the children run with HOME inside the root
        self.host_home = Path(host_home) if host_home else Path.home()
        # operator-supplied CONSERVATIVE estimates (GiB) that stand in ONLY
        # when the corresponding measurement cannot be complete: a partial
        # cold-cache measurement, or a first-run FT download of unknown size.
        # Without them such a cycle / FT phase is resource-blocked. They are
        # recorded with provenance "operator_estimate" and never replace a
        # complete measurement.
        self.cold_cache_gib = None if cold_cache_gib is None else float(cold_cache_gib)
        self.ft_download_gib = None if ft_download_gib is None else float(ft_download_gib)
        # operator's CONSERVATIVE install/FT peak (GiB) for an env the G0
        # preflight gives no estimate for; without it such a cycle is
        # resource-blocked -- the measured floor alone is not a budget
        self.peak_gib = None if peak_gib is None else float(peak_gib)
        self.cancel = threading.Event()  # set by SIGTERM/SIGINT -> orderly stop


class CampaignLedger:
    """Append-only JSONL outside every runtime root; `seq` resumes from the
    last row so a paused campaign continues in the same file. Every append is
    flushed + fsynced: a row that was written survives a kill."""

    def __init__(self, path: Path, campaign_id: str):
        self.path = path
        self.campaign_id = campaign_id
        self.seq = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        self.seq = max(self.seq, int(json.loads(line).get("seq", 0)))
                    except (ValueError, json.JSONDecodeError):
                        continue

    def append(self, **row) -> dict:
        self.seq += 1
        record = {"seq": self.seq, "at": utc_now(), "campaign_id": self.campaign_id}
        record.update(row)
        for key in ("env", "family", "variant", "kind", "phase", "state", "returncode",
                    "stderr_tail", "verdict", "evidence", "manifest_sha256", "runtime_root"):
            record.setdefault(key, None)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return record


def env_family(env: str, home: Path) -> str | None:
    registry = json.loads((home / "models.json").read_text(encoding="utf-8"))
    for family, spec in registry.items():
        if not family.startswith("_") and spec.get("env") == env:
            return family
    return None


def resolve_target_env(target: str, home: Path) -> str | None:
    """Accept an env name (mace) or a model/version name (MACE, MACE-MPA-0)."""
    if env_family(target, home):
        return target
    return find_env(target, home)


def scan_foreign_processes(markers=FOREIGN_PROCESS_MARKERS, proc_root: Path = Path("/proc")) -> list[dict]:
    """Other install/finetune processes on the host (Part 3.4 concurrency).
    Fails closed: an unreadable proc root cannot vouch that nobody else is
    installing (failed(concurrency:proc_unreadable)); a pid whose cmdline
    could not be read is re-checked -- gone means it exited during the scan
    (ignored), still present means we cannot say what it is
    (failed(concurrency:proc_uninspectable))."""
    found = []
    uninspectable = []
    me = os.getpid()
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise fresh_root.FreshRootError("failed(concurrency:proc_unreadable)",
                                        {"proc_root": str(proc_root), "error": str(exc)})
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            try:
                os.stat(entry)
            except OSError:
                continue  # vanished: exited between listing and reading
            uninspectable.append(int(entry.name))
            continue
        if any(m in cmdline for m in markers):
            found.append({"pid": int(entry.name), "cmdline": cmdline.strip()[:200]})
    if uninspectable:
        raise fresh_root.FreshRootError("failed(concurrency:proc_uninspectable)",
                                        {"pids": sorted(uninspectable), "proc_root": str(proc_root)})
    return found


def acquire_lock(sweep_dir: Path, campaign_id: str, proc_root: Path = Path("/proc")) -> Path:
    """Atomic (O_EXCL) campaign lock. A lock whose holder pid is dead is
    stale and replaced; a live holder refuses with failed(campaign_lock)."""
    sweep_dir.mkdir(parents=True, exist_ok=True)
    lock = sweep_dir / LOCK_NAME
    payload = json.dumps({"pid": os.getpid(), "campaign_id": campaign_id, "at": utc_now()}) + "\n"
    for _attempt in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                holder = json.loads(lock.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                holder = {}
            pid = holder.get("pid")
            if pid and pid != os.getpid() and (proc_root / str(pid)).exists():
                raise fresh_root.FreshRootError("failed(campaign_lock)", holder)
            try:
                lock.unlink()  # stale: holder gone
            except OSError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        return lock
    raise fresh_root.FreshRootError("failed(campaign_lock)", "could not replace a stale lock")


def mandatory_hash_targets(hub: Path, env: str) -> dict[str, str]:
    """Part 3.4 mandatory evidence: the real hub's local state files and the
    user env's conda-meta/history for the env under test, hashed (or
    'absent') -- compared before/after every cycle."""
    paths = [hub / "env_map.local.json", hub / "models.local.json", hub / "envs" / "_expected.json"]
    adopted = load_local_env_map(hub).get(env)
    if adopted:
        paths.append(Path(adopted) / "conda-meta" / "history")
    return {str(p): fresh_root.hash_or_absent(p) for p in paths}


def preflight_row(preflight: dict | None, env: str) -> dict | None:
    if not preflight:
        return None
    for row in preflight.get("envs") or []:
        if row.get("env") == env:
            return row
    return None


def peak_estimate_gib(estimate: dict | None) -> float | None:
    """The G0 conservative peak in GiB: schema-2 `conservative_disk_peak_gib`,
    else a decimal-GB field converted once, else None (= UNKNOWN; an unknown
    estimate never becomes a bound, it only means the start gate cannot be
    computed and the measured floor alone applies)."""
    if not estimate:
        return None
    if estimate.get("conservative_disk_peak_gib") is not None:
        return float(estimate["conservative_disk_peak_gib"])
    if estimate.get("conservative_disk_peak_gb") is not None:
        return float(estimate["conservative_disk_peak_gb"]) / fresh_root.GB_PER_GIB
    return None


def home_cold_cache(host_home: Path, names: list[str]) -> dict:
    """Host HOME caches env.sh adopts through its literal `$HOME/.cache/<name>`
    writes (matris/tace/eqnorm-style symlinks into the hub's models/). The
    isolated campaign HOME starts cold, so a fresh root re-downloads them:
    their MEASURED size on the host is the provisional addend to the start
    gate and to the measured floor (Part 3.5). Read-only measurement of the
    real HOME; nothing there is touched.

    Deterministic, known paths only: `<host_home>/.cache/<name>` for the
    given names -- no search for weights anywhere else. A symlinked `.cache`
    or `<name>` is MEASURED through its resolved target (read-only du; the
    no-symlink rule guards escape and deletion, not measurement) with the
    resolution recorded under `resolved` as provenance; a dangling or
    unresolvable link is `skipped`. Inside a measured tree symlinks are not
    followed. An unreadable subtree is `skipped` with its errors. Anything
    skipped makes the result `partial`: a partial measurement is NOT a small
    one, and the caller must not use `total_gib` as an adequate preflight
    figure without an explicit conservative estimate from the operator (or
    must resource-block)."""
    dirs: dict[str, float] = {}
    skipped: dict[str, str] = {}
    resolved: dict[str, dict] = {}

    def resolve_dir(p: Path) -> Path | None:
        """The directory to measure for `p`: itself, or its symlink target
        (recorded); None (skipped) when dangling / not a directory."""
        if not p.is_symlink():
            return p if p.is_dir() else None
        try:
            target = p.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            skipped[str(p)] = f"symlink_unresolvable:{exc}"
            return None
        if not target.is_dir():
            skipped[str(p)] = f"symlink_not_a_dir:{target}"
            return None
        resolved[str(p)] = {"resolved_from": str(p), "resolved_to": str(target)}
        return target

    cache_root = Path(host_home) / ".cache"
    cache_dir = resolve_dir(cache_root)
    if cache_dir is None and str(cache_root) not in skipped:
        return {"unit": "GiB", "dirs": {}, "skipped": {}, "resolved": {}, "partial": False, "total_gib": 0.0,
                "basis": "no <host_home>/.cache on the host: nothing to re-download"}
    for name in dict.fromkeys(n for n in names if n):
        if cache_dir is None:
            break
        p = cache_root / name
        if not (cache_dir / name).is_symlink() and not (cache_dir / name).exists():
            continue  # absent: nothing to re-download, a complete answer of 0
        target = resolve_dir(cache_dir / name)
        if target is None:
            continue
        errors: list[str] = []
        size = fresh_root.du_bytes(target, errors)
        if errors:
            skipped[str(p)] = "unreadable:" + "; ".join(errors[:3])
            continue
        dirs[str(p)] = round(size / GIB, 3)
    return {"unit": "GiB", "dirs": dirs, "skipped": skipped, "resolved": resolved, "partial": bool(skipped),
            "total_gib": round(sum(dirs.values()), 3),
            "basis": "measured size of the host HOME caches this family would have adopted via env.sh; "
                     "the isolated campaign HOME starts cold, so they re-download into the root"
                     + ("; symlinked entries measured through their recorded targets" if resolved else "")
                     + ("; PARTIAL: some paths were skipped, total_gib is a lower bound" if skipped else "")}


# First-run fine-tune downloads: artifacts a family's FIRST ft_run fetches
# INTO the isolated root that the seeded inference weights do not cover
# (reported by the FT owner from the builders' prestage steps, 2026-09-14).
# Each entry names a deterministic host path to MEASURE (read-only) as the
# size estimate, or None when nothing on this host can stand in for it: an
# unknown size never becomes a number -- the fine-tune phase is
# resource-blocked unless the operator supplies an explicit conservative
# estimate (--ft-download-gib, recorded with provenance "operator_estimate").
# Sizes are MEASURED at deterministic upstream/registry cache paths under the
# host HOME (never a filesystem search); `copies` is how many copies the
# first run persists in the root (the FT owner's 2026-09-14 audit: nequip
# 0.17.1 keeps the package under NEQUIP_CACHE_DIR/model_cache AND the
# prestage copies it to $OH_MY_MLIP_HOME/models/<env>/; TACE lands both as
# ~/.cache/tace/<model>.pt and as an HF blob of the same size). A family the
# audit did not complete is listed with an `unknown` item -- never zero.
FIRST_RUN_FT_DOWNLOADS: dict[str, list[dict]] = {
    "GRACE": [{"name": "GRACE fine-tune checkpoint (dict format; `grace_models checkpoint <model>` -> "
                       "$GRACE_CACHE/checkpoints/<model>; tar.gz downloaded then extracted: peak = archive + "
                       "extracted, size not measured)", "host_probe": None}],
    "NequIP": [{"name": "NequIP-OAM <Model>.nequip.zip packages (nequip.net:mir-group/<Model>:0.1), "
                        "x2: model_cache + prestage copy under models/",
                "host_probe": ".nequip/model_cache", "copies": 2}],
    "Allegro": [{"name": "Allegro-OAM-L .nequip.zip package (not cached on this host; nequip 0.15.0: 1x persisted "
                         "+ 1x transient)", "host_probe": None}],
    "MatterSim": [{"name": "MatterSim-v1.0.0-*.pth pretrained weights (~/.local/mattersim/pretrained_models)",
                   "host_probe": ".local/mattersim/pretrained_models", "copies": 1}],
    "TACE": [{"name": "TACE-OAM-L.pt (~/.cache/tace) x2: cache file + HF blob of the same size",
              "host_probe": ".cache/tace", "copies": 2}],
    "PET": [{"name": "PET additional first-run fetches: audit NOT completed (FT owner 2026-09-14) -- unknown, not zero",
             "host_probe": None}],
    "MACE": [{"name": "MACE additional first-run fetches: audit NOT completed (FT owner 2026-09-14) -- unknown, not zero",
              "host_probe": None}],
    "DeePMD": [{"name": "DeePMD additional first-run fetches: audit NOT completed (FT owner 2026-09-14) -- unknown, not zero",
                "host_probe": None}],
    "DPA4": [{"name": "DPA4 additional first-run fetches: audit NOT completed (FT owner 2026-09-14) -- unknown, not zero",
              "host_probe": None}],
}


def first_run_ft_downloads(family: str, host_home: Path) -> dict:
    """Budget addend for the artifacts the first fine-tune run of `family`
    downloads into the root. Sizes come only from measuring the named host
    path (provenance "measured_host:<path>"); a missing/symlinked/unreadable
    probe or a None probe is `unknown` -- listed, never estimated here."""
    items = []
    unknown = []
    for spec in FIRST_RUN_FT_DOWNLOADS.get(family or "", []):
        copies = int(spec.get("copies", 1))
        item = {"name": spec["name"], "gib": None, "provenance": "unknown", "copies": copies}
        probe = spec.get("host_probe")
        if probe:
            p = Path(host_home) / probe
            if p.is_dir() and not p.is_symlink():
                errors: list[str] = []
                size = fresh_root.du_bytes(p, errors)
                if not errors:
                    # the measured stand-in times the copies the first run
                    # persists in the root (never a per-file guess)
                    item.update(gib=round(size * copies / GIB, 3), measured_gib=round(size / GIB, 3),
                                provenance=f"measured_host:{p}" + (f" x{copies}" if copies != 1 else ""))
                else:
                    item["provenance"] = f"unknown (host probe unreadable: {p})"
            elif p.is_symlink():
                item["provenance"] = f"unknown (host probe is a symlink, not followed: {p})"
            else:
                item["provenance"] = f"unknown (host probe absent: {p})"
        if item["gib"] is None:
            unknown.append(item["name"])
        items.append(item)
    return {"unit": "GiB", "items": items, "known_gib": round(sum(i["gib"] for i in items if i["gib"] is not None), 3),
            "unknown": unknown,
            "basis": "artifacts the first ft_run fetches into the root beyond the seeded weights; "
                     "sizes only from measured host files at deterministic cache paths (x copies persisted), "
                     "otherwise unknown -- never zero"}


class DuSampler:
    """Peak hardlink-aware usage of one root, sampled on a thread (Part 3.5
    `budget_actual`)."""

    def __init__(self, root: Path, interval: float):
        self.root, self.interval = root, interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        try:
            self.peak = max(self.peak, fresh_root.du_bytes(self.root))
        except OSError:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def start(self) -> "DuSampler":
        self._thread.start()
        return self

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        self._sample()
        return self.peak


def _terminate_group(proc: subprocess.Popen, grace: float) -> str:
    """SIGTERM the child's whole process group, wait `grace`, then SIGKILL
    whatever is left. Returns how it ended."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "exited"
    try:
        proc.wait(timeout=grace)
        return "terminated"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        return "killed"


def _pump_sanitized(stream, fh, sanitize) -> None:
    """Reader thread: every line of a child's pipe is scrubbed BEFORE it is
    written to the log file, so the raw text never reaches disk."""
    for line in iter(stream.readline, ""):
        fh.write(sanitize(line))
        fh.flush()
    stream.close()


def run_phase_monitored(command: list[str], cwd: Path, env: dict | None, *, log_stem: Path,
                        hooks: FreshHooks, df_path: Path, launched: list[int] | None = None,
                        sanitize=None) -> dict:
    """Run one phase in its own session/process group with stdout/stderr
    streamed to <log_stem>.out.log / .err.log (full logs, preserved later).
    While it runs, the MEASURED free space at `df_path` is re-read every
    hooks.df_poll_seconds: below hooks.min_free_gib the group is stopped
    (SIGTERM, grace, SIGKILL) and the phase is reported aborted=disk_floor;
    hooks.cancel (SIGTERM/SIGINT to the driver) stops it the same way with
    aborted=terminated. The child's session id (= its pid) is appended to
    `launched` so cleanup can refuse while any descendant of a phase is
    still alive. With `sanitize` (a str -> str scrubber) the child's output
    goes through pipes and every line is scrubbed BEFORE it is persisted:
    nothing unscrubbed is written to the log files, and the returned
    stdout/stderr are read back from those scrubbed files."""
    log_stem.parent.mkdir(parents=True, exist_ok=True)
    out_path, err_path = Path(f"{log_stem}.out.log"), Path(f"{log_stem}.err.log")
    floor = hooks.min_free_gib * GIB
    min_free = None
    aborted = None
    with out_path.open("w", encoding="utf-8") as out_fh, err_path.open("w", encoding="utf-8") as err_fh:
        if sanitize is None:
            proc = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=out_fh, stderr=err_fh,
                                    start_new_session=True)
            pumps: list[threading.Thread] = []
        else:
            proc = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    start_new_session=True, text=True, encoding="utf-8", errors="replace")
            pumps = [threading.Thread(target=_pump_sanitized, args=(proc.stdout, out_fh, sanitize), daemon=True),
                     threading.Thread(target=_pump_sanitized, args=(proc.stderr, err_fh, sanitize), daemon=True)]
            for t in pumps:
                t.start()
        if launched is not None:
            launched.append(proc.pid)
        while True:
            try:
                proc.wait(timeout=hooks.df_poll_seconds)
                break
            except subprocess.TimeoutExpired:
                pass
            free = hooks.disk_free(df_path)
            min_free = free if min_free is None else min(min_free, free)
            if free < floor:
                aborted = "disk_floor"
                ended = _terminate_group(proc, hooks.grace_seconds)
                break
            if hooks.cancel.is_set():
                aborted = "terminated"
                ended = _terminate_group(proc, hooks.grace_seconds)
                break
        for t in pumps:  # the pipes close when the group is gone; drain them fully
            t.join()
    free = hooks.disk_free(df_path)
    min_free = free if min_free is None else min(min_free, free)
    stdout = out_path.read_text(encoding="utf-8", errors="replace")
    stderr = err_path.read_text(encoding="utf-8", errors="replace")
    return {"rc": proc.returncode, "stdout": stdout, "stderr_tail": stderr[-STDERR_TAIL_CHARS:],
            "aborted": aborted, "ended": ended if aborted else "exited", "pgid": proc.pid,
            "min_free_gib": round(min_free / GIB, 3), "out_log": str(out_path), "err_log": str(err_path)}


# Credential shapes scrubbed from every inventory file and recorded command
# before persistence: userinfo in URLs, token-style query parameters, HF /
# GitHub / generic bearer tokens.
_SECRET_PATTERNS = [
    (re.compile(r"(?i)(https?://)[^/\s@:]+:[^/\s@]+@"), r"\1***:***@"),
    (re.compile(r"(?i)([?&](?:token|access_token|api_key|apikey|key|auth|password|secret)=)[^&\s]+"), r"\1***"),
    (re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"), "hf_***"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"), "gh*_***"),
    (re.compile(r"(?i)(authorization:\s*bearer\s+)\S+"), r"\1***"),
]


def sanitize_secrets(text: str) -> str:
    for pat, rep in _SECRET_PATTERNS:
        text = pat.sub(rep, text)
    return text


INVENTORY_KINDS = ("conda_explicit", "pip_freeze", "pip_list")


def dependency_inventory(runtime: Path, env: str, cycle_env: dict, hooks: "FreshHooks", *, logs: Path,
                         launched: list[int], df_path: Path) -> dict:
    """Per-owned-env dependency evidence, captured UNDER THE OWNED INTERPRETER
    after install: `conda list --explicit -p <prefix>`, `pip freeze`,
    `pip list --format=json`, and the runtime registry's resolve() answers
    for every variant (the isolation probe's report, verbatim). Written to
    <root>/.sweep/inventory/ (sanitized; preserved before cleanup, every file
    REQUIRED). The owned prefix is <root>/envs/<env> -- the only place
    install.sh creates an env inside a root. Inventories record what was
    installed; they do not by themselves close the dependency-replay gap
    (an exact replay procedure is still required), and the row says so."""
    inv_dir = runtime / ".sweep" / "inventory"
    inv_dir.mkdir(parents=True, exist_ok=True)
    prefix = runtime / "envs" / env
    owned_python = prefix / "bin" / "python"
    files = {k: inv_dir / f"{env}.{k}.{'json' if k in ('pip_list', 'resolve') else 'txt'}"
             for k in INVENTORY_KINDS + ("resolve",)}
    required = [str(f) for f in files.values()]
    evidence: dict = {"prefix": str(prefix), "owned_python": str(owned_python), "files": {},
                      "commands": {}, "problems": [],
                      "note": "inventories record the installed set; they do not close the dependency-replay "
                              "gap -- an exact replay procedure is still required"}
    if not prefix.is_dir():
        evidence["problems"].append(f"owned prefix missing: {prefix}")
        return {"state": "failed(inventory:prefix_missing)", "returncode": 1, "required": required,
                "stderr_tail": evidence["problems"][-1], "evidence": evidence}
    commands = {
        "conda_explicit": list(hooks.conda_cmd or ["conda"]) + ["list", "--explicit", "-p", str(prefix)],
        "pip_freeze": [str(owned_python), "-m", "pip", "freeze"],
        "pip_list": [str(owned_python), "-m", "pip", "list", "--format=json"],
    }
    if hooks.inventory_cmd:  # TEST ONLY: one fake tool answering every kind
        commands = {k: list(hooks.inventory_cmd) + [k, str(prefix), str(owned_python)] for k in commands}
    rc_total = 0
    aborted: dict | None = None
    for kind, cmd in commands.items():
        evidence["commands"][kind] = [sanitize_secrets(c) for c in cmd]
        try:
            # every output line is scrubbed BEFORE it reaches the phase logs
            res = run_phase_monitored(cmd, runtime, cycle_env, log_stem=logs / f"inventory.{kind}", hooks=hooks,
                                      df_path=df_path, launched=launched, sanitize=sanitize_secrets)
        except OSError as exc:  # a tool that cannot even start (no interpreter in the prefix, ...)
            rc_total += 1
            evidence["problems"].append(sanitize_secrets(f"{kind}: could not run {cmd[0]}: {exc}"))
            continue
        if res["aborted"]:
            # disk floor / driver stop: the same outcome as any other phase
            # (the caller raises CycleAbort); the remaining kinds are not run
            rc_total += 1
            aborted = res
            evidence["problems"].append(f"{kind}: aborted={res['aborted']} child group {res['ended']}")
            break
        text = res["stdout"]  # already scrubbed at the pipe
        if res["rc"] == 0 and text.strip() and _inventory_well_formed(kind, text):
            files[kind].write_text(text, encoding="utf-8")
            evidence["files"][kind] = {"path": str(files[kind]), "sha256": fresh_root.sha256_file(files[kind]),
                                       "bytes": files[kind].stat().st_size}
        else:
            rc_total += 1
            why = "empty output" if res["rc"] == 0 and not text.strip() else \
                ("malformed output" if res["rc"] == 0 else f"rc={res['rc']}")
            evidence["problems"].append(f"{kind}: {why} {res['stderr_tail'][-300:]}".rstrip())
    # registry resolve() answers under the owned interpreter (no re-implemented path logic)
    if aborted is None:
        iso = fresh_root.verify_isolation(runtime, python=str(owned_python) if not hooks.inventory_cmd else hooks.probe_python)
        report = {"python": str(owned_python), "isolation_state": iso.get("state"), "problems": iso.get("problems"),
                  "entries": iso.get("entries"), "probe": iso.get("probe")}
        files["resolve"].write_text(sanitize_secrets(json.dumps(report, indent=1, sort_keys=True)) + "\n", encoding="utf-8")
        evidence["files"]["resolve"] = {"path": str(files["resolve"]), "sha256": fresh_root.sha256_file(files["resolve"]),
                                        "bytes": files["resolve"].stat().st_size}
        if not iso.get("ok"):
            rc_total += 1
            evidence["problems"].append(sanitize_secrets(f"resolve: {iso.get('state')} {json.dumps(iso.get('problems'))[:300]}"))
    state = "passed" if rc_total == 0 else "failed(inventory)"
    return {"state": state, "returncode": rc_total, "required": required, "aborted": aborted,
            "stderr_tail": "; ".join(evidence["problems"])[:STDERR_TAIL_CHARS] if evidence["problems"] else None,
            "evidence": evidence}


def _inventory_well_formed(kind: str, text: str) -> bool:
    """Shape check per inventory kind: pip_list must be a JSON list of
    objects; the text kinds need at least one non-comment line."""
    if kind == "pip_list":
        try:
            data = json.loads(text)
        except ValueError:
            return False
        return isinstance(data, list) and all(isinstance(d, dict) for d in data)
    return any(ln.strip() and not ln.startswith("#") for ln in text.splitlines())


def fresh_cycle(env: str, hub: Path, runtime_parent: Path, campaign_dir: Path, ledger: CampaignLedger,
                allowlist: list[str], hooks: FreshHooks) -> bool:
    """One disposable-root cycle for `env`. Returns True when the campaign
    must PAUSE (unattributable change to a mandatory-hash target) or STOP
    (driver terminated)."""
    family = env_family(env, hub)
    variants = family_versions(family, hub) if family else []
    run_id = f"{env}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    runtime = runtime_parent / run_id
    base = {"env": env, "family": family, "runtime_root": str(runtime)}
    started = time.monotonic()
    df_path = runtime_parent if runtime_parent.exists() else hub

    def rows_for_all(phase: str, state: str, detail: str, **extra) -> None:
        for kind in ("inference", "finetune"):
            for v in variants:
                ledger.append(variant=v, kind=kind, phase=phase, state=state, stderr_tail=detail, **base, **extra)

    def free_gib() -> float:
        return hooks.disk_free(df_path) / GIB

    def floor_check(phase: str) -> None:
        """Measured mid-cycle floor between phases (Part 3.5)."""
        free = free_gib()
        if free < hooks.min_free_gib:
            raise CycleAbort("resource-blocked",
                             f"before {phase}: free {free:.2f} GiB < {hooks.min_free_gib:.2f} GiB floor (measured)")
        if hooks.cancel.is_set():
            raise CycleAbort("terminated", f"driver received a stop signal before {phase}")

    def phase_result(res: dict, phase: str) -> None:
        if res["aborted"] == "disk_floor":
            raise CycleAbort("resource-blocked",
                             f"during {phase}: free fell to {res['min_free_gib']} GiB < {hooks.min_free_gib:.2f} GiB "
                             f"floor (measured); child group {res['ended']}")
        if res["aborted"] == "terminated":
            raise CycleAbort("terminated", f"during {phase}: driver stop signal; child group {res['ended']}")

    if not family or not variants:
        ledger.append(variant="*", phase="resolve", state="failed(resolve)", returncode=1,
                      stderr_tail=f"no registry family lives in env {env!r}", **base)
        return False

    # --- start gate (Part 3.5): MEASURED free space vs the G0 conservative
    # peak (GiB) + the cold isolated-HOME caches this family re-downloads
    # (measured on the host) + first-run FT downloads + this harness's start
    # reserve. A peak the preflight does not estimate is taken ONLY from the
    # operator's explicit conservative --peak-gib; otherwise the cycle is
    # resource-blocked before anything is materialized: the mid-cycle floor
    # monitor stops a filling disk, it is not proof that the peak fits.
    free_start = free_gib()
    estimate = preflight_row(hooks.preflight, env)
    peak_gib = peak_estimate_gib(estimate)
    if peak_gib is not None:
        peak_provenance = "preflight"
    elif hooks.peak_gib is not None:
        peak_gib, peak_provenance = hooks.peak_gib, "operator_estimate"
    else:
        peak_provenance = "unknown"
    cold = home_cold_cache(hooks.host_home, [env, family, family.lower()])
    # a PARTIAL cold-cache measurement is a lower bound, never a budget: it
    # is replaced by the operator's explicit conservative estimate or the
    # cycle is resource-blocked before anything is materialized
    if cold["partial"] and hooks.cold_cache_gib is not None:
        cold_gib, cold_provenance = hooks.cold_cache_gib, "operator_estimate"
    elif cold["partial"]:
        cold_gib, cold_provenance = None, "unknown(partial_measurement)"
    else:
        cold_gib, cold_provenance = cold["total_gib"], "measured_host_home"
    # first-run FT downloads (known part measured on the host; unknown part
    # either operator-estimated or the FT phase is blocked, below)
    ftdl = first_run_ft_downloads(family, hooks.host_home) if hooks.ft_dataset is not None else \
        {"unit": "GiB", "items": [], "known_gib": 0.0, "unknown": [], "basis": "no --ft-dataset: no fine-tune phase"}
    # --ft-download-gib semantics (every family, additive, conservative): the
    # operator figure is ADDED to the measured known part -- it never replaces
    # a measurement, so nothing measured is double-counted against it -- and
    # it is the explicit bound for the family's unknown items; a family with
    # unknown items and no figure keeps its fine-tune phase blocked (unknown
    # is never zero). --peak-gib never covers downloads.
    op_gib = hooks.ft_download_gib
    if op_gib is not None:
        ftdl_gib = ftdl["known_gib"] + op_gib
        ftdl_provenance = ("measured_host+operator_estimate" if ftdl["items"]
                           else "operator_estimate")
        ft_block_reason = None
    elif ftdl["unknown"]:
        ftdl_gib, ftdl_provenance = ftdl["known_gib"], "partial(unknown items excluded)"
        ft_block_reason = ("first-run fine-tune download size unknown for: " + "; ".join(ftdl["unknown"])
                           + " (no --ft-download-gib estimate given; unknown is never zero)")
    else:
        ftdl_gib, ftdl_provenance = ftdl["known_gib"], "measured_host"
        ft_block_reason = None
    ftdl_operator = {"gib": op_gib, "covers_unknown": list(ftdl["unknown"]) if op_gib is not None else [],
                     "semantics": "added to the measured known part; the explicit bound for unknown items; "
                                  "never replaces a measurement; --peak-gib does not cover downloads"}
    addend = None if cold_gib is None else cold_gib + ftdl_gib
    gate_gib = None if (peak_gib is None or addend is None) else peak_gib + addend + hooks.start_reserve_gib
    floor_gate_gib = None if addend is None else hooks.min_free_gib + addend
    budget_evidence = {"unit": "GiB", "free_gib": round(free_start, 3), "peak_estimate_gib": peak_gib,
                       "peak_estimate_provenance": peak_provenance,
                       "estimate_status": "unknown" if peak_gib is None else
                       (estimate or {}).get("estimate_confidence", "provisional" if peak_provenance == "preflight"
                                            else "operator_estimate"),
                       "home_cold_cache": cold, "home_cold_cache_gib": cold_gib, "home_cold_cache_provenance": cold_provenance,
                       "first_run_ft_downloads": ftdl, "first_run_ft_downloads_gib": round(ftdl_gib, 3),
                       "first_run_ft_downloads_provenance": ftdl_provenance,
                       "first_run_ft_downloads_operator": ftdl_operator,
                       "start_reserve_gib": hooks.start_reserve_gib, "gate_gib": gate_gib,
                       "min_free_gib": hooks.min_free_gib, "floor_gate_gib": floor_gate_gib,
                       "budget_estimate": estimate,
                       "note": "the start gate needs a peak estimate (preflight or --peak-gib): the mid-cycle floor "
                               "monitor stops a filling disk, it is not proof that the install/FT peak fits"}
    if peak_gib is None:
        rows_for_all("budget", "resource-blocked",
                     f"no conservative peak estimate for env {env!r} (preflight has none, no --peak-gib given); "
                     f"the measured floor alone is not a budget",
                     evidence=budget_evidence)
        return False
    if cold_gib is None:
        rows_for_all("budget", "resource-blocked",
                     f"cold HOME cache size could not be measured completely ({json.dumps(cold['skipped'])}); "
                     f"a partial measurement is not a budget -- give --cold-cache-gib (conservative) or fix the path",
                     evidence=budget_evidence)
        return False
    if free_start < gate_gib:
        rows_for_all("budget", "resource-blocked",
                     f"measured free {free_start:.2f} GiB < start gate {gate_gib:.2f} GiB "
                     f"(peak estimate {peak_gib:.2f} + cold HOME caches {cold_gib:.2f} "
                     f"+ first-run FT downloads {ftdl_gib:.2f} + reserve {hooks.start_reserve_gib:.2f})",
                     evidence={**budget_evidence, "additional_free_gib_needed": round(gate_gib - free_start, 3)})
        return False
    if free_start < floor_gate_gib:
        rows_for_all("skipped_disk", "resource-blocked",
                     f"measured free {free_start:.2f} GiB < {floor_gate_gib:.2f} GiB "
                     f"(floor {hooks.min_free_gib:.2f} + cold HOME caches {cold_gib:.2f} + first-run FT downloads {ftdl_gib:.2f})",
                     evidence={**budget_evidence, "additional_free_gib_needed": round(floor_gate_gib - free_start, 3)})
        return False

    # --- 1. snapshot --check (mandatory before every materialization) + 2. snapshot
    try:
        manifest = fresh_root.build_manifest(hub, allowlist)
        written = fresh_root.write_snapshot(hub, manifest, campaign_dir)
    except fresh_root.FreshRootError as exc:
        ledger.append(variant="*", phase="snapshot_check", state=exc.state, returncode=1,
                      stderr_tail=json.dumps(exc.detail), **base)
        return False
    sha = manifest["manifest_sha256"]
    base["manifest_sha256"] = sha
    ledger.append(variant="*", phase="snapshot", state="passed", returncode=0,
                  evidence={**written, "git_head": manifest["git_head"],
                            "dirty_tracked_paths": manifest["dirty_tracked_paths"],
                            "included_untracked": manifest["included_untracked"],
                            "file_count": manifest["file_count"]}, **base)

    # --- 3. mandatory hashes BEFORE
    hashes_before = mandatory_hash_targets(hub, env)

    # --- 4. materialize (re-checks the working tree itself)
    owned_registry = campaign_dir / fresh_root.OWNED_ROOTS_FILE
    try:
        mat = fresh_root.materialize(Path(written["snapshot_dir"]), runtime, campaign_id=ledger.campaign_id,
                                     repo=hub, allowlist=allowlist, owned_registry=owned_registry)
    except fresh_root.FreshRootError as exc:
        ledger.append(variant="*", phase="materialize", state=exc.state, returncode=1,
                      stderr_tail=json.dumps(exc.detail), **base)
        return False
    # the EFFECTIVE child environment: inherited cache overrides dropped,
    # every route (HOME included) inside the root, read-only inputs as paths
    cycle_env, env_report = fresh_root.build_cycle_env(runtime, os.environ)
    exports = env_report["exports"]
    ledger.append(variant="*", phase="materialize", state="passed", returncode=0,
                  evidence={"env_file": mat["env_file"], "exports": exports, "owned_registry": mat["owned_registry"],
                            "rejected_inherited": env_report["rejected_inherited"],
                            "read_only_inputs": env_report["read_only_inputs"]}, **base)
    sampler = DuSampler(runtime, hooks.du_sample_seconds).start()
    logs = runtime / ".sweep" / "phases"
    final_artifacts: list[str] = []
    missing_final: list[str] = []
    bundle_dirs: list[str] = []  # checkpoint-bundle directories to mirror under the preserved copy
    launched_groups: list[int] = []  # session ids of every phase child (cleanup quiescence)
    aborted: str | None = None
    terminated = False
    peak = 0

    try:
        # --- 5. seed-cache (copy only)
        if hooks.seed_items:
            try:
                seeded = fresh_root.seed_cache(runtime, hooks.seed_items)
                ledger.append(variant="*", phase="seed_cache", state="passed", returncode=0,
                              evidence={"weights_source": seeded["weights_source"],
                                        "seeded": seeded["seeded"]}, **base)
            except fresh_root.FreshRootError as exc:
                ledger.append(variant="*", phase="seed_cache", state=exc.state, returncode=1,
                              stderr_tail=json.dumps(exc.detail), **base)
                raise CycleAbort(exc.state, json.dumps(exc.detail))
        else:
            ledger.append(variant="*", phase="seed_cache", state="passed", returncode=0,
                          evidence={"weights_source": "fresh-download", "seeded": []}, **base)

        # --- 6. routes + 7. isolation pre-check (abort BEFORE any compute)
        routes = fresh_root.verify_routes(runtime, cycle_env, conda_cmd=hooks.conda_cmd)
        ledger.append(variant="*", phase="routes", state=routes["state"],
                      returncode=0 if routes["ok"] else 1, evidence=routes, **base)
        if not routes["ok"]:
            raise CycleAbort(routes["state"], json.dumps(routes["problems"]))
        iso = fresh_root.verify_isolation(runtime, python=hooks.probe_python)
        ledger.append(variant="*", phase="isolation_pre", state=iso["state"],
                      returncode=0 if iso["ok"] else 1, evidence=iso, **base)
        if not iso["ok"]:
            raise CycleAbort(iso["state"], json.dumps(iso["problems"]))

        # --- 8. install (monitored: measured floor + orderly stop)
        floor_check("install")
        install = (hooks.install_cmd or [str(runtime / "install.sh")]) + [env]
        res = run_phase_monitored(install, runtime, cycle_env, log_stem=logs / "install", hooks=hooks,
                                  df_path=df_path, launched=launched_groups)
        ledger.append(variant="*", phase="install",
                      state="passed" if res["rc"] == 0 and not res["aborted"] else "failed(install)",
                      returncode=res["rc"], stderr_tail=res["stderr_tail"],
                      evidence={"command": install, "out_log": res["out_log"], "err_log": res["err_log"],
                                "min_free_gib": res["min_free_gib"], "aborted": res["aborted"]}, **base)
        phase_result(res, "install")
        if res["rc"] != 0:
            raise CycleAbort("failed(install)", res["stderr_tail"])

        # --- 9. isolation post-install (the runtime's own resolve(), again)
        iso = fresh_root.verify_isolation(runtime, python=hooks.probe_python)
        ledger.append(variant="*", phase="isolation_post", state=iso["state"],
                      returncode=0 if iso["ok"] else 1, evidence=iso, **base)
        if not iso["ok"]:
            raise CycleAbort(iso["state"], json.dumps(iso["problems"]))

        # --- 9b. dependency inventory of the OWNED env (evidence for the
        # dependency-replay gap; preserved before cleanup). Failure does not
        # abort the cycle, but the missing files keep preserve -> cleanup closed.
        floor_check("inventory")
        inv = dependency_inventory(runtime, env, cycle_env, hooks, logs=logs, launched=launched_groups, df_path=df_path)
        final_artifacts.extend(inv["required"])
        ledger.append(variant="*", phase="inventory", state=inv["state"], returncode=inv["returncode"],
                      stderr_tail=inv["stderr_tail"], evidence=inv["evidence"], **base)
        if inv["aborted"]:
            phase_result(inv["aborted"], "inventory")

        # --- 10. inference verify per variant (explicit --version, never
        # recording into models.local.json even inside the disposable root)
        for v in variants:
            floor_check(f"verify {v}")
            verify = (hooks.verify_cmd or [sys.executable, str(runtime / "scripts" / "setup_verify.py")]) \
                + [family, "--version", v, "--json", "--no-local-record"]
            res = run_phase_monitored(verify, runtime, cycle_env, log_stem=logs / f"verify.{v}", hooks=hooks,
                                      df_path=df_path, launched=launched_groups)
            verdict = last_json_line(res["stdout"])
            passed = res["rc"] == 0 and not res["aborted"] and bool(verdict and verdict.get("pass"))
            ledger.append(variant=v, kind="inference", phase="verify",
                          state="passed" if passed else "failed(verify)",
                          returncode=res["rc"], stderr_tail=res["stderr_tail"], verdict=verdict,
                          evidence={"command": verify, "degraded": bool(verdict and verdict.get("degraded")),
                                    "log": str(runtime / ".sweep" / "verify" / f"{env}.{v}.log"),
                                    "out_log": res["out_log"], "err_log": res["err_log"],
                                    "min_free_gib": res["min_free_gib"]}, **base)
            phase_result(res, f"verify {v}")

        # --- 11. fine-tune sweep (monitored as one child group)
        if hooks.ft_dataset is None:
            for v in variants:
                ledger.append(variant=v, kind="finetune", phase="ft_sweep", state="failed(ft_dataset_missing)",
                              returncode=None, stderr_tail="no --ft-dataset given; fine-tune rows cannot pass", **base)
        elif ft_block_reason is not None:
            # the first run would download artifacts of UNKNOWN size into the
            # root: not budgeted, so not attempted (inference rows above stand)
            for v in variants:
                ledger.append(variant=v, kind="finetune", phase="ft_sweep", state="resource-blocked",
                              returncode=None, stderr_tail=ft_block_reason,
                              evidence={"first_run_ft_downloads": ftdl}, **base)
        else:
            floor_check("ft_sweep")
            ft_ledger = runtime / ".ft" / f"{env}.jsonl"
            ft = (hooks.ft_cmd or [sys.executable, str(runtime / "scripts" / "ft_sweep.py")]) + [
                "--env", env, "--dataset", str(hooks.ft_dataset), "--ledger", str(ft_ledger),
                "--campaign-id", ledger.campaign_id, "--manifest-sha256", sha,
                "--out-root", str(runtime / ".ft" / env), "--min-free-gib", str(hooks.min_free_gib),
            ] + (["--audit", str(hooks.ft_audit)] if hooks.ft_audit else []) + list(hooks.ft_extra_args)
            res = run_phase_monitored(ft, runtime, cycle_env, log_stem=logs / "ft_sweep", hooks=hooks,
                                      df_path=df_path, launched=launched_groups)
            ft_rows = []
            if ft_ledger.exists():
                ft_rows = [json.loads(ln) for ln in ft_ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
            terminal = {r["variant"]: r for r in ft_rows if r.get("state") and r.get("variant")}
            for v in variants:
                r = terminal.get(v)
                if r is None:
                    ledger.append(variant=v, kind="finetune", phase="ft_sweep",
                                  state="resource-blocked" if res["aborted"] == "disk_floor" else "failed(ft_sweep)",
                                  returncode=res["rc"],
                                  stderr_tail=res["stderr_tail"] or f"ft_sweep produced no terminal row (aborted={res['aborted']})",
                                  evidence={"command": ft, "ft_ledger": str(ft_ledger), "out_log": res["out_log"],
                                            "err_log": res["err_log"], "aborted": res["aborted"]}, **base)
                    continue
                ev = {**(r.get("evidence") or {}), "ft_ledger": str(ft_ledger)}
                # the checkpoint, the rerun script and ft_run's provenance
                # record (ft_run.json: rematerialize argv, seed control,
                # designated checkpoint) are REQUIRED preserved artifacts
                for key in ("ckpt", "sh", "ft_run_json"):
                    if ev.get(key):
                        final_artifacts.append(ev[key])
                if r["state"] == "passed" and not ev.get("ft_run_json"):
                    missing_final.append(f"{v}: passed fine-tune row carries no ft_run.json provenance record")
                # a checkpoint that is one file of a bundle (GRACE SavedModel:
                # saved_model.pb + variables/ + ...) is preserved WHOLE -- the
                # witness file alone does not reload
                for f in (ev.get("ckpt_bundle") or {}).get("files") or []:
                    if f.get("path") and f["path"] not in final_artifacts:
                        final_artifacts.append(f["path"])
                # the bundle's directories (an empty assets/ included) are
                # recreated under the preserved copy: file copies alone drop them
                for d in [(ev.get("ckpt_bundle") or {}).get("dir")] + ((ev.get("ckpt_bundle") or {}).get("dirs") or []):
                    if d and d not in bundle_dirs:
                        bundle_dirs.append(d)
                cfg = Path(ev["out"]) / "finetune_config.json" if ev.get("out") else None
                if cfg and cfg.is_file():
                    final_artifacts.append(str(cfg))
                if r["state"] == "passed" and not ev.get("ckpt"):
                    # a pass without a checkpoint path has nothing to preserve:
                    # recorded so preserve fails closed and the root is retained
                    missing_final.append(f"{v}: passed fine-tune row carries no checkpoint path")
                ledger.append(variant=v, kind="finetune", phase=f"ft:{r.get('phase')}", state=r["state"],
                              returncode=r.get("returncode"), stderr_tail=r.get("stderr_tail"), evidence=ev, **base)
            phase_result(res, "ft_sweep")

        # --- 12. source-hash verify
        src = fresh_root.verify_sources(runtime, manifest)
        ledger.append(variant="*", phase="sources", state=src["state"], returncode=0 if src["ok"] else 1,
                      evidence={"mutated": src["mutated"], "missing": src["missing"]}, **base)
    except CycleAbort as exc:
        aborted = exc.state
        terminated = exc.state == "terminated"
        ledger.append(variant="*", phase="abort", state=exc.state, returncode=1, stderr_tail=exc.detail, **base)
        rows_for_all("aborted", exc.state, f"cycle stopped before this variant completed: {exc.detail}"[:STDERR_TAIL_CHARS])
    except Exception as exc:  # noqa: BLE001 -- recorded, never swallowed silently
        aborted = f"failed(exception:{type(exc).__name__})"
        ledger.append(variant="*", phase="abort", state=aborted, returncode=1, stderr_tail=str(exc)[:STDERR_TAIL_CHARS], **base)
        rows_for_all("aborted", aborted, str(exc)[:STDERR_TAIL_CHARS])
    finally:
        peak = sampler.stop()

    # --- 13. mandatory hashes AFTER (attribution). The row reports what was
    # OBSERVED (before/after hashes, which targets changed); a change is
    # attributed to nobody -- the reader concludes, the campaign pauses.
    hashes_after = mandatory_hash_targets(hub, env)
    changed = {p: {"before": hashes_before[p], "after": hashes_after.get(p)}
               for p in hashes_before if hashes_before[p] != hashes_after.get(p)}
    pause = bool(changed)
    ledger.append(variant="*", phase="attribution", state="failed(attribution)" if changed else "passed",
                  returncode=1 if changed else 0,
                  evidence={"targets_before": hashes_before, "targets_after": hashes_after, "changed": changed,
                            "attribution": "unknown" if changed else "unchanged",
                            "note": "observed hashes only; a changed target is not attributed to any writer"}, **base)

    # --- 14. budget row (GiB; estimate vs measured actual)
    wall = time.monotonic() - started
    free_end = free_gib()
    home_dir = runtime / "home"
    isolated_home_gib = round(fresh_root.du_bytes(home_dir) / GIB, 3) if home_dir.is_dir() else 0.0
    ledger.append(variant="*", phase="budget", state="passed", returncode=0,
                  evidence={**budget_evidence,
                            "budget_actual": {"unit": "GiB", "peak_bytes": peak, "peak_gib": round(peak / GIB, 3),
                                              "isolated_home_gib": isolated_home_gib,
                                              "wall_seconds": round(wall, 1),
                                              "free_gib_start": round(free_start, 3), "free_gib_end": round(free_end, 3),
                                              "aborted": aborted}}, **base)

    # --- 15. preserve (verified copies OUTSIDE the root; cleanup needs this)
    # Every final artifact a fine-tune row named (checkpoint, rerun script,
    # config) is REQUIRED: a missing one fails preserve and, below, keeps the
    # root -- cleanup never proceeds on logs alone.
    preserved_record: Path | None = None
    try:
        if missing_final:
            raise fresh_root.FreshRootError("failed(preserve:required_missing)", missing_final)
        rec = fresh_root.preserve_artifacts(runtime, campaign_dir / run_id, required=final_artifacts)
        preserved_record = Path(rec["record"])
        # checkpoint-bundle directories mirrored under the copy (empty ones too)
        mirrored_dirs = []
        for d in bundle_dirs:
            src = Path(d)
            if src.is_dir() and not src.is_symlink() and fresh_root._inside(src, runtime):
                dst = Path(rec["dest"]) / src.relative_to(runtime)
                dst.mkdir(parents=True, exist_ok=True)
                mirrored_dirs.append(str(dst))
        ledger.append(variant="*", phase="preserve", state="passed", returncode=0,
                      evidence={"record": rec["record"], "dest": rec["dest"], "file_count": rec["file_count"],
                                "bytes": rec["bytes"], "final_artifacts": final_artifacts,
                                "required_artifacts": final_artifacts,
                                "files": [f["dst"] for f in rec["files"]], "bundle_dirs": mirrored_dirs}, **base)
    except fresh_root.FreshRootError as exc:
        ledger.append(variant="*", phase="preserve", state=exc.state, returncode=1,
                      stderr_tail=json.dumps(exc.detail),
                      evidence={"final_artifacts": final_artifacts, "required_artifacts": final_artifacts,
                                "missing": exc.detail}, **base)

    # --- 16. guarded cleanup (never after a termination: preserve, retain, stop)
    if terminated:
        ledger.append(variant="*", phase="cleanup", state="retained", returncode=0,
                      stderr_tail="driver terminated: runtime root retained for inspection", **base)
    elif not hooks.cleanup:
        ledger.append(variant="*", phase="cleanup", state="retained", returncode=0,
                      stderr_tail="--no-cleanup: runtime root kept", **base)
    elif preserved_record is None:
        ledger.append(variant="*", phase="cleanup", state="failed(cleanup:evidence_not_durable)", returncode=1,
                      stderr_tail="preserve did not succeed (logs and/or required final artifacts not durable); "
                                  "the runtime root is retained", **base)
    else:
        try:
            result = fresh_root.cleanup(runtime, ledger.path, preserved=preserved_record,
                                        owned_registry=owned_registry, allowed_parent=runtime_parent,
                                        protected=[hub], proc_root=hooks.proc_root,
                                        owned_groups=launched_groups)
            ledger.append(variant="*", phase="cleanup", state="cleaned", returncode=0, evidence=result, **base)
        except fresh_root.FreshRootError as exc:
            ledger.append(variant="*", phase="cleanup", state=exc.state, returncode=1,
                          stderr_tail=json.dumps(exc.detail), **base)
    # proof of cleanup = owned path absence; df is a diagnostic level only
    ledger.append(variant="*", phase="post_cleanup", state="passed" if (not runtime.exists() or not hooks.cleanup or terminated)
                  else "failed(cleanup:incomplete)", returncode=0,
                  evidence={"runtime_absent": not runtime.exists(), "unit": "GiB",
                            "free_gib_after_cleanup": round(free_gib(), 3)}, **base)
    return pause or terminated


def fresh_sweep(targets: list[str], hub: Path, runtime_parent: Path, campaign_id: str,
                allowlist_path: Path | None, hooks: FreshHooks, ledger_path: Path | None = None) -> Path:
    """Serialized campaign over `targets`; returns the campaign ledger path.
    SIGTERM/SIGINT set hooks.cancel: the running child group is stopped in
    order, evidence is preserved, the root retained, the campaign stops."""
    hub = hub.resolve()
    campaign_dir = hub / ".sweep" / f"campaign_{campaign_id}"
    ledger = CampaignLedger(ledger_path or campaign_dir / "ledger.jsonl", campaign_id)
    try:
        allowlist = fresh_root.load_allowlist(allowlist_path)
    except fresh_root.FreshRootError as exc:
        ledger.append(phase="plan", state=exc.state, returncode=1, stderr_tail=json.dumps(exc.detail))
        return ledger.path
    envs = [(t, resolve_target_env(t, hub)) for t in targets]
    ledger.append(phase="plan", targets=[e or t for t, e in envs], hub=str(hub),
                  runtime_parent=str(runtime_parent), allowlist=str(allowlist_path) if allowlist_path else None,
                  ft_dataset=str(hooks.ft_dataset) if hooks.ft_dataset else None,
                  token_source=token_source(), preflight_loaded=bool(hooks.preflight),
                  unit="GiB", min_free_gib=hooks.min_free_gib, start_reserve_gib=hooks.start_reserve_gib,
                  df_poll_seconds=hooks.df_poll_seconds)

    try:
        foreign = scan_foreign_processes(proc_root=hooks.proc_root)
        if foreign:
            ledger.append(phase="concurrency", state="failed(concurrency)", returncode=1,
                          stderr_tail="other install/finetune processes running", evidence={"processes": foreign})
            return ledger.path
        lock = acquire_lock(hub / ".sweep", campaign_id, hooks.proc_root)
    except fresh_root.FreshRootError as exc:
        ledger.append(phase="concurrency", state=exc.state, returncode=1, stderr_tail=json.dumps(exc.detail))
        return ledger.path

    previous = {}
    if threading.current_thread() is threading.main_thread():
        def _on_signal(signum, _frame):
            hooks.cancel.set()
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, _on_signal)
    try:
        runtime_parent.mkdir(parents=True, exist_ok=True)
        # the CANONICAL parent is what the ownership registry binds and what
        # cleanup is bounded to (a symlinked parent would never clean up)
        runtime_parent = runtime_parent.resolve()
        for target, env in envs:
            if env is None:
                ledger.append(phase="resolve", state="failed(resolve)", returncode=1,
                              stderr_tail=f"unknown target: {target}", env=None, variant=target)
                continue
            if hooks.cancel.is_set():
                ledger.append(phase="stop", state="terminated", returncode=None, env=env,
                              stderr_tail="driver stop signal before this env started")
                break
            stop = fresh_cycle(env, hub, runtime_parent, campaign_dir, ledger, allowlist, hooks)
            if hooks.cancel.is_set():
                ledger.append(phase="stop", state="terminated", returncode=None, env=env,
                              stderr_tail="driver stop signal: campaign stopped after preserving evidence")
                break
            if stop:
                ledger.append(phase="pause", state="paused", returncode=None,
                              stderr_tail="campaign paused: unattributable change to a mandatory-hash target")
                break
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        try:
            lock.unlink()
        except OSError:
            pass
    return ledger.path


def fresh_report(ledger: Path) -> str:
    rows = [json.loads(ln) for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = [f"fresh-root campaign report -- ledger: {ledger}"]
    terminal: dict[tuple, dict] = {}
    for r in rows:
        if r.get("kind") and r.get("variant") and r.get("state"):
            terminal[(r["env"], r["kind"], r["variant"])] = r
    counts: dict[str, int] = {}
    for (env, kind, variant), r in sorted(terminal.items()):
        counts[r["state"]] = counts.get(r["state"], 0) + 1
        out.append(f"  {env:<12} {kind:<9} {variant:<24} {r['state']}")
    guards = [r for r in rows if r.get("variant") == "*" and r.get("state") and not r["state"].startswith("passed")]
    for g in guards:
        out.append(f"  [{g['env']}] {g['phase']}: {g['state']}")
    for r in rows:
        if r.get("phase") == "post_cleanup":
            ev = r.get("evidence") or {}
            out.append(f"  [{r['env']}] runtime root absent: {ev.get('runtime_absent')} "
                       f"(free after cleanup {ev.get('free_gib_after_cleanup')} GiB, diagnostic)")
    out.append("  " + " / ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    return "\n".join(out)


def target_status(records: list[dict]) -> str:
    """Status of one target, strictly from its ledger records."""
    if not records:
        return "not_attempted"
    if any(r["phase"] == "skipped_gated" for r in records):
        return "skipped_gated"
    if any(r["phase"] == "skipped_disk" for r in records):
        return "skipped_disk"
    verifies = [r for r in records if r["phase"] == "verify"]
    if verifies:
        verdict = verifies[-1].get("verdict") or {}
        if verdict.get("pass") and verdict.get("degraded"):
            return "degraded"
        if verdict.get("pass"):
            return "verified"
    return "failed"


def report(ledger: Path) -> str:
    lines = [json.loads(ln) for ln in ledger.read_text().splitlines() if ln.strip()]
    plan = next((ln for ln in lines if ln.get("phase") == "plan"), None)
    targets = plan["targets"] if plan else sorted({ln.get("target") for ln in lines if ln.get("target")})
    out = [f"setup_sweep report -- ledger: {ledger}"]
    counts: dict[str, int] = {}
    for target in targets:
        records = [ln for ln in lines if ln.get("target") == target]
        status = target_status(records)
        counts[status] = counts.get(status, 0) + 1
        detail = ""
        if status == "failed":
            last = records[-1]
            detail = f"  ({last['phase']} rc={last['returncode']})"
        out.append(f"  {target:<22} {status}{detail}")
    out.append("  " + " / ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="sweep", choices=("sweep", "report"))
    ap.add_argument("--targets", default="", help="comma-separated model names (families or versions)")
    ap.add_argument("--ledger", default=None, help="ledger path (default: new for sweep, latest for report)")
    ap.add_argument("--install-cmd", default=None, help="TEST ONLY: replacement install command")
    ap.add_argument("--verify-cmd", default=None, help="TEST ONLY: replacement verify command")
    fr = ap.add_argument_group("fresh-root cycle (plan G2)")
    fr.add_argument("--fresh-root", action="store_true", help="one disposable runtime root per target env")
    fr.add_argument("--runtime-parent", default=None, help="default: <hub parent>/.omm_fresh")
    fr.add_argument("--campaign-id", default=None, help="default: UTC timestamp")
    fr.add_argument("--allowlist", default=None, help="default: .omc/state/omm-e2e-allowlist.json (see fresh_root.py)")
    fr.add_argument("--preflight", default=None, help="G0 preflight JSON for per-env disk bands")
    fr.add_argument("--ft-dataset", default=None, help="dataset for ft_sweep.py (absent => finetune rows cannot pass)")
    fr.add_argument("--ft-audit", default=None, help="current-campaign FT support audit JSON")
    fr.add_argument("--seed-spec", default=None, help="JSON {dst_rel: src_abs} copied into the runtime (never symlinked)")
    fr.add_argument("--no-cleanup", action="store_true", help="keep the runtime root (cleanup row = retained)")
    fr.add_argument("--min-free-gib", type=float, default=10.0,
                    help="MEASURED free-space floor, GiB (2^30): checked before every phase and while children run")
    fr.add_argument("--start-reserve-gib", type=float, default=15.0,
                    help="margin above the G0 conservative peak (GiB) the measured free space must clear to start")
    fr.add_argument("--cold-cache-gib", type=float, default=None,
                    help="operator's CONSERVATIVE estimate (GiB) of the cold HOME caches, used ONLY when their "
                         "host measurement is partial (symlinked/unreadable); without it such a cycle is resource-blocked")
    fr.add_argument("--ft-download-gib", type=float, default=None,
                    help="operator's CONSERVATIVE estimate (GiB) for first-run fine-tune downloads of unknown size "
                         "(GRACE checkpoint, Allegro package); without it that fine-tune phase is resource-blocked")
    fr.add_argument("--peak-gib", type=float, default=None,
                    help="operator's CONSERVATIVE install/fine-tune peak (GiB) for an env the G0 preflight gives no "
                         "estimate for; without one such a cycle is resource-blocked (the floor is not a budget)")
    fr.add_argument("--df-poll-seconds", type=float, default=15.0, help="free-space re-measure interval while a child runs")
    fr.add_argument("--grace-seconds", type=float, default=20.0, help="SIGTERM -> SIGKILL grace for a stopped child group")
    fr.add_argument("--du-sample-seconds", type=float, default=30.0)
    fr.add_argument("--probe-python", default=None, help="interpreter for the isolation probe (default: this one)")
    fr.add_argument("--ft-cmd", default=None, help="TEST ONLY: replacement ft_sweep command")
    fr.add_argument("--ft-extra-args", default="", help="TEST ONLY: extra args appended to the ft_sweep command")
    fr.add_argument("--conda-cmd", default=None, help="TEST ONLY: replacement for `conda` in the routes check")
    fr.add_argument("--inventory-cmd", default=None, help="TEST ONLY: replacement for the conda/pip inventory tools")
    fr.add_argument("--proc-root", default="/proc", help="TEST ONLY: fake /proc for lock/concurrency checks")
    args = ap.parse_args()

    home = resolve_home()
    if args.mode == "report":
        ledger = Path(args.ledger) if args.ledger else latest_ledger_path(home)
        if ledger is None or not ledger.exists():
            print("no sweep ledger found", file=sys.stderr)
            return 1
        first = next((ln for ln in ledger.read_text(encoding="utf-8").splitlines() if ln.strip()), "{}")
        is_campaign = "campaign_id" in json.loads(first)
        print(fresh_report(ledger) if is_campaign else report(ledger))
        return 0

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        print("no targets given (--targets M1,M2,...)", file=sys.stderr)
        return 1

    if args.fresh_root:
        campaign_id = args.campaign_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        hooks = FreshHooks(
            install_cmd=args.install_cmd.split() if args.install_cmd else None,
            verify_cmd=args.verify_cmd.split() if args.verify_cmd else None,
            ft_cmd=args.ft_cmd.split() if args.ft_cmd else None,
            ft_extra_args=args.ft_extra_args.split(),
            conda_cmd=args.conda_cmd.split() if args.conda_cmd else None,
            proc_root=Path(args.proc_root),
            seed_items=json.loads(Path(args.seed_spec).read_text(encoding="utf-8")) if args.seed_spec else {},
            preflight=json.loads(Path(args.preflight).read_text(encoding="utf-8")) if args.preflight else None,
            ft_dataset=Path(args.ft_dataset).resolve() if args.ft_dataset else None,
            ft_audit=Path(args.ft_audit).resolve() if args.ft_audit else None,
            cleanup=not args.no_cleanup,
            min_free_gib=args.min_free_gib,
            start_reserve_gib=args.start_reserve_gib,
            df_poll_seconds=args.df_poll_seconds,
            grace_seconds=args.grace_seconds,
            du_sample_seconds=args.du_sample_seconds,
            probe_python=args.probe_python,
            inventory_cmd=[args.inventory_cmd] if args.inventory_cmd else None,
            cold_cache_gib=args.cold_cache_gib,
            ft_download_gib=args.ft_download_gib,
            peak_gib=args.peak_gib,
        )
        runtime_parent = Path(args.runtime_parent) if args.runtime_parent else home.parent / ".omm_fresh"
        allowlist_path = Path(args.allowlist) if args.allowlist else fresh_root.default_allowlist_path(home)
        ledger = fresh_sweep(targets, home, runtime_parent, campaign_id, allowlist_path, hooks,
                             ledger_path=Path(args.ledger) if args.ledger else None)
        print(fresh_report(ledger))
        return 0

    ledger = Path(args.ledger) if args.ledger else next_ledger_path(home)
    sweep(
        targets, home, ledger,
        args.install_cmd.split() if args.install_cmd else None,
        args.verify_cmd.split() if args.verify_cmd else None,
    )
    print(report(ledger))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
