#!/usr/bin/env python3
"""catbench_version.py — the catbench version rule as a file, not a habit.

Rule (recipes/catbench.md §2): a NEW job discovers the current official
stable release and proposes it for approval; a RERUN uses the version the
approved job recorded and never re-discovers. Three numbers are always
disclosed side by side — official latest, hub pin, what each target env holds
— with the lookup URL and UTC time, so the plan can show them verbatim.

Nothing here installs, upgrades or pins an env. An env holding an older
catbench than the approved version is listed under `requires_env_upgrade`;
that upgrade is a separate step with its own approval.

Usage:
  python3 scripts/catbench_version.py --mode new [--python ENV/bin/python ...] --out catbench_version.json
  python3 scripts/catbench_version.py --mode rerun --record catbench_version.json [--python ...]
  python3 scripts/catbench_version.py --mode new --offline     # no network: pin only, latest = null
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYPI_URL = "https://pypi.org/pypi/catbench/json"
_PIN_RE = re.compile(r'^CATBENCH_PIN="\$\{OMM_CATBENCH_VERSION:-([0-9][0-9A-Za-z.]*)\}"', re.M)
_STABLE_RE = re.compile(r"^\d+(\.\d+)*$")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _default_fetch(url: str, timeout: float = 30.0) -> tuple[int, bytes, str | None]:
    """(http_code, body, error). Never raises: unreachable → (0, b"", reason)."""
    req = urllib.request.Request(url, headers={"User-Agent": "oh-my-mlip/catbench_version"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), None
    except Exception as exc:  # noqa: BLE001 — recorded, not raised
        code = getattr(exc, "code", 0) or 0
        return code, b"", f"{type(exc).__name__}: {exc}"


def _version_key(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split("."))


def discover_latest(fetch=None) -> dict:
    """Official latest STABLE release from PyPI (pre-releases and yanked
    files excluded), with the release time and the lookup provenance."""
    fetch = fetch or _default_fetch
    checked = utc_now()
    code, body, err = fetch(PYPI_URL)
    out = {"source_url": PYPI_URL, "checked_utc": checked, "http_code": code,
           "latest": None, "latest_release_utc": None, "error": err}
    if code != 200 or not body:
        out["error"] = err or f"http {code}"
        return out
    try:
        data = json.loads(body)
        releases = data.get("releases", {})
    except (ValueError, AttributeError) as exc:
        out["error"] = f"unparseable PyPI JSON: {exc}"
        return out
    stable = []
    for ver, files in releases.items():
        if not _STABLE_RE.match(ver):
            continue
        live = [f for f in (files or []) if not f.get("yanked")]
        if not live:
            continue
        stable.append((_version_key(ver), ver, min(f.get("upload_time_iso_8601", "") for f in live)))
    if not stable:
        out["error"] = "no stable, non-yanked release listed"
        return out
    _, ver, when = max(stable)
    out["latest"], out["latest_release_utc"] = ver, when or None
    return out


def hub_pin(install_sh: Path | None = None, env: dict | None = None) -> dict:
    """The hub's install-time pin: `OMM_CATBENCH_VERSION` if exported, else
    the default baked into install.sh (`CATBENCH_PIN="${OMM_CATBENCH_VERSION:-X}"`)."""
    env = os.environ if env is None else env
    install_sh = install_sh or (REPO / "install.sh")
    override = env.get("OMM_CATBENCH_VERSION")
    if override:
        return {"pin": override, "source": "OMM_CATBENCH_VERSION (environment)"}
    m = _PIN_RE.search(install_sh.read_text()) if install_sh.is_file() else None
    if not m:
        return {"pin": None, "source": f"{install_sh}: CATBENCH_PIN line not found"}
    return {"pin": m.group(1), "source": f"{install_sh}: CATBENCH_PIN default"}


def installed_version(python: str, runner=None) -> dict:
    """`catbench.__version__` under a target interpreter (subprocess; injectable)."""
    runner = runner or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True, timeout=120))
    cmd = [python, "-c", "import catbench, sys; sys.stdout.write(catbench.__version__)"]
    if not Path(python).is_file():
        return {"python": python, "installed": None, "error": "interpreter not found"}
    try:
        proc = runner(cmd)
    except Exception as exc:  # noqa: BLE001
        return {"python": python, "installed": None, "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {"python": python, "installed": None,
                "error": (proc.stderr or "").strip().splitlines()[-1:] or "import failed"}
    return {"python": python, "installed": proc.stdout.strip(), "error": None}


def decide(mode: str, *, latest: str | None, pin: str | None, record_version: str | None,
           envs: list[dict], offline: bool = False, discovery_error: str | None = None) -> dict:
    """Apply the rule. `approved_version` is PROPOSED for a new job (status
    "proposed": the plan still needs approval) and RECORDED for a rerun. A
    failed discovery is not silently replaced by the pin — only an explicit
    `--offline` falls back to it, and says so."""
    if mode == "rerun":
        if not record_version:
            return {"approved_version": None, "ok": False, "status": "none",
                    "reason": "rerun requires the version recorded by the approved job (--record)"}
        approved, basis, status = record_version, "recorded by the approved job (pinned; not re-discovered)", "recorded"
    else:
        status = "proposed"
        if latest:
            approved, basis = latest, "official latest stable (proposed; needs approval)"
        elif offline and pin:
            approved, basis = pin, "hub pin (--offline: latest not looked up; disclose and confirm)"
        elif not offline:
            return {"approved_version": None, "ok": False, "status": "none",
                    "reason": f"official latest could not be discovered ({discovery_error}); "
                              "rerun with --offline to propose the hub pin explicitly, or fix the network"}
        else:
            return {"approved_version": None, "ok": False, "status": "none", "reason": "no latest and no hub pin"}
    upgrade = [e["python"] for e in envs
               if e.get("installed") and _safe_key(e["installed"]) < _safe_key(approved)]
    mismatch = [e["python"] for e in envs if e.get("installed") and e["installed"] != approved]
    unknown = [e["python"] for e in envs if not e.get("installed")]
    out = {
        "approved_version": approved, "status": status, "ok": True, "basis": basis,
        "pin_matches_approved": (pin == approved) if pin else None,
        "requires_env_upgrade": upgrade, "env_mismatch": mismatch, "env_unknown": unknown,
        "note": ("env upgrade is a SEPARATE step needing its own approval; nothing here installs"
                 if upgrade else "no env upgrade needed"),
    }
    if mode == "rerun" and (mismatch or unknown):
        out["ok"] = False
        out["reason"] = (f"rerun refused: env(s) no longer hold the recorded {approved}: mismatch={mismatch} "
                         f"unknown={unknown} (reported, never upgraded here)")
    return out


def _safe_key(v: str) -> tuple:
    try:
        return _version_key(v)
    except ValueError:
        return (-1,)


def build_record(mode: str, *, pythons: list[str], record: Path | None, offline: bool,
                 fetch=None, runner=None, install_sh: Path | None = None) -> dict:
    prior = json.loads(record.read_text()) if record and record.is_file() else {}
    record_version = prior.get("approved_version")
    latest = {"latest": None, "source_url": PYPI_URL, "checked_utc": None,
              "note": "offline / rerun: no discovery performed"}
    if mode == "new" and not offline:
        latest = discover_latest(fetch)
    pin = hub_pin(install_sh)
    envs = [installed_version(p, runner) for p in pythons]
    verdict = decide(mode, latest=latest.get("latest"), pin=pin["pin"], record_version=record_version,
                     envs=envs, offline=offline, discovery_error=latest.get("error"))
    return {"mode": mode, "generated_utc": utc_now(), "official": latest, "hub_pin": pin,
            "envs": envs, "record_read": str(record) if record else None, **verdict}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["new", "rerun"], required=True)
    ap.add_argument("--python", action="append", default=[], help="target env interpreter (repeatable)")
    ap.add_argument("--record", default=None, help="catbench_version.json of the approved job (rerun)")
    ap.add_argument("--offline", action="store_true", help="skip the PyPI lookup")
    ap.add_argument("--out", default=None, help="write the record here (JSON)")
    ap.add_argument("--json", action="store_true", help="print the record as JSON")
    args = ap.parse_args(argv)

    rec = build_record(args.mode, pythons=args.python,
                       record=Path(args.record) if args.record else None, offline=args.offline)
    if args.out:
        Path(args.out).write_text(json.dumps(rec, indent=2) + "\n")
    if args.json:
        print(json.dumps(rec, indent=2))
    else:
        print(f"mode            : {rec['mode']}")
        print(f"official latest : {rec['official'].get('latest')}  ({rec['official'].get('source_url')} @ {rec['official'].get('checked_utc')})")
        print(f"hub pin         : {rec['hub_pin']['pin']}  ({rec['hub_pin']['source']})")
        for e in rec["envs"]:
            print(f"env             : {e['python']} -> {e['installed']}" + (f"  [{e['error']}]" if e.get("error") else ""))
        print(f"approved version: {rec.get('approved_version')}  [{rec.get('status')}] ({rec.get('basis') or rec.get('reason')})")
        if rec.get("requires_env_upgrade"):
            print(f"needs upgrade   : {rec['requires_env_upgrade']}  (separate approval)")
        if rec.get("env_mismatch"):
            print(f"env mismatch    : {rec['env_mismatch']}")
        if not rec.get("ok"):
            print(f"[stop] {rec.get('reason')}", file=sys.stderr)
    return 0 if rec.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
