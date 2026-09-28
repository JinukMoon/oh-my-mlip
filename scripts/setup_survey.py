#!/usr/bin/env python3
"""One-shot deterministic survey for the setup skill (GPU-free, stdlib-only).

The setup skill's plan/approval step must never improvise its facts or their
order: this script computes ATOMICALLY, in one read-only invocation, everything
that step needs —

  * per-env install state (same rules as ``install.sh --status``:
    ready / partial / broken / not installed),
  * the disk math, counting ONLY envs that would actually be built
    (ready envs cost zero new disk),
  * leak-safe HF-token availability (source name only; the token value is
    never read into this process and never printed),
  * which envs are gated (any gated version in the family's roster).

The skill renders this output and asks its approval question from it; it must
not recompute, reorder, or partially re-derive any of these numbers. That is
what makes the survey-before-any-disk-judgment ordering deterministic instead
of a prose promise.

Usage:
  python3 scripts/setup_survey.py                # JSON on stdout (agent path)
  python3 scripts/setup_survey.py --table        # human-readable table
  python3 scripts/setup_survey.py MACE sevennet  # restrict to targets
                                                 # (family or env name)

Exit code is always 0 on a successful survey, even when the budget does not
fit — "does not fit" is a plan fact, not an error.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Conservative per-env build budget (GB). Matches the skill contract's
# "~10 GB x missing/broken" plan math; partial envs count full because
# adopt-or-heal may fall back to a rebuild.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oh_my_mlip.hub import DISK_FLOOR_GB, ENV_BUDGET_GB  # noqa: E402

PER_ENV_GB = ENV_BUDGET_GB


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup_common import load_local_env_map, refuse_plugin_copy, resolve_home  # noqa: E402


def env_state(prefix: Path) -> str:
    """Mirror install.sh --status exactly (keep the two in lockstep)."""
    if (prefix / ".omm_ready").exists():
        return "ready"
    if os.access(prefix / "bin" / "python", os.X_OK):
        return "partial"
    if prefix.exists():
        return "broken"
    return "not_installed"


def token_source() -> str:
    """Name the first available token source; never touch the value.

    Order mirrors oh_my_mlip.fetch._resolve_token: HF_TOKEN, HF_TOKEN_PATH,
    OMM_HF_TOKEN_FILE, then the `hf auth login` cache.
    """
    if os.environ.get("HF_TOKEN"):
        return "HF_TOKEN"
    path = os.environ.get("HF_TOKEN_PATH")
    if path and Path(path).is_file():
        return "HF_TOKEN_PATH"
    omm = os.environ.get("OMM_HF_TOKEN_FILE")
    if omm and Path(omm).is_file():
        return "OMM_HF_TOKEN_FILE"
    if (Path.home() / ".cache" / "huggingface" / "token").is_file():
        return "hf_cache"
    return "none"


def survey(home: Path, targets: list[str]) -> dict:
    registry = json.loads((home / "models.json").read_text())
    families = {k: v for k, v in registry.items() if not k.startswith("_")}

    wanted = {t.lower() for t in targets}
    matched: set[str] = set()
    adopted_map = load_local_env_map(home)
    rows: list[dict] = []
    seen_envs: set[str] = set()
    for family, spec in families.items():
        env = spec["env"]
        names = {family.lower(), env.lower()} | {v.lower() for v in (spec.get("versions") or {})}
        if wanted and not (names & wanted):
            continue
        matched |= names & wanted
        gated = any(
            bool(v.get("gated")) for v in (spec.get("versions") or {}).values()
        )
        if env in seen_envs:
            for row in rows:
                if row["env"] == env:
                    row["families"].append(family)
                    row["gated"] = row["gated"] or gated
            continue
        seen_envs.add(env)
        # An adopted env (env_map.local.json, import-verified at adopt time)
        # counts as ready: it costs zero disk and needs no install.
        adopted_prefix = adopted_map.get(env)
        adopted = bool(
            adopted_prefix
            and os.access(Path(adopted_prefix) / "bin" / "python", os.X_OK)
        )
        rows.append(
            {
                "env": env,
                "families": [family],
                "gated": gated,
                "state": "ready" if adopted else env_state(home / "envs" / env),
                "adopted": adopted,
            }
        )

    counts = {s: 0 for s in ("ready", "partial", "broken", "not_installed")}
    for row in rows:
        counts[row["state"]] += 1
    to_build = [r["env"] for r in rows if r["state"] != "ready"]

    envs_dir = home / "envs"
    probe = envs_dir if envs_dir.exists() else home
    free_gb = shutil.disk_usage(probe).free / 1024**3
    budget_gb = PER_ENV_GB * len(to_build)

    source = token_source()
    sys.path.insert(0, str(home))
    from oh_my_mlip.registry import detect_host_arch
    arch = detect_host_arch()
    return {
        "gpu": {"arch": arch, "found": arch is not None},
        "home": str(home),
        "envs": rows,
        "counts": counts,
        "to_build": to_build,
        "disk": {
            "free_gb": round(free_gb, 1),
            "budget_gb": budget_gb,
            "per_env_gb": PER_ENV_GB,
            # the last build must still start above the floor install.sh enforces
            "floor_gb": DISK_FLOOR_GB,
            "fits": not to_build or free_gb >= budget_gb - PER_ENV_GB + DISK_FLOOR_GB,
        },
        "token": {"available": source != "none", "source": source},
        "gated_envs": [r["env"] for r in rows if r["gated"]],
        "unknown_targets": sorted(t for t in targets if t.lower() not in matched),
    }


def print_table(result: dict) -> None:
    print(f"oh-my-mlip setup survey — {result['home']}")
    print(f"{'env':<14} {'state':<14} gated  families")
    for row in result["envs"]:
        state = row["state"] + ("*" if row.get("adopted") else "")
        print(
            f"{row['env']:<14} {state:<14} "
            f"{'yes' if row['gated'] else 'no':<6} {', '.join(row['families'])}"
        )
    if any(r.get("adopted") for r in result["envs"]):
        print("(* adopted external env via env_map.local.json — zero disk, no install)")
    c, d, t = result["counts"], result["disk"], result["token"]
    print(
        f"\nready {c['ready']} · partial {c['partial']} · broken {c['broken']}"
        f" · not installed {c['not_installed']}"
    )
    print(
        f"disk: {d['budget_gb']} GB needed for {len(result['to_build'])} builds"
        f" ({d['per_env_gb']:g} GB each, and {d['floor_gb']:g} GB free to start each build; ready envs cost zero)"
        f" vs {d['free_gb']} GB free -> "
        + ("fits" if d["fits"] else "DOES NOT FIT")
    )
    g = result["gpu"]
    print("gpu: " + (f"found ({g['arch']})" if g["found"] else
                     "NO NVIDIA GPU found (nvidia-smi missing or lists none): the models run on a GPU; "
                     "installing works, running needs the NVIDIA driver or a GPU host"))
    print(
        "hf token: "
        + (f"available (source: {t['source']})" if t["available"] else "none found")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="*", help="family or env names; default all")
    parser.add_argument("--table", action="store_true", help="human-readable output")
    args = parser.parse_args()

    refuse_plugin_copy(resolve_home())
    result = survey(resolve_home(), args.targets)
    if result["unknown_targets"]:
        registry = json.loads((resolve_home() / "models.json").read_text())
        fams = sorted(k for k in registry if not k.startswith("_"))
        print(f"[stop] not a family, env or variant in models.json: {', '.join(result['unknown_targets'])}. "
              f"Families: {', '.join(fams)} (variant names: python3 -c \"import oh_my_mlip; "
              f"print(oh_my_mlip.list_versions('<Family>'))\").", file=sys.stderr)
        return 2
    if args.table:
        print_table(result)
    else:
        json.dump(result, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
