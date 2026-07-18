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
PER_ENV_GB = 10


def resolve_home() -> Path:
    home = os.environ.get("OH_MY_MLIP_HOME") or os.environ.get("OMM_HOME")
    if home and Path(home).is_dir():
        return Path(home).resolve()
    return Path(__file__).resolve().parents[1]


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

    Order mirrors oh_my_mlip.fetch resolution: HF_TOKEN env, then the
    standard huggingface_hub file paths, then the OMM convenience variable.
    """
    if os.environ.get("HF_TOKEN"):
        return "HF_TOKEN"
    path = os.environ.get("HF_TOKEN_PATH")
    if path and Path(path).is_file():
        return "HF_TOKEN_PATH"
    if (Path.home() / ".cache" / "huggingface" / "token").is_file():
        return "hf_cache"
    omm = os.environ.get("OMM_HF_TOKEN_FILE")
    if omm and Path(omm).is_file():
        return "OMM_HF_TOKEN_FILE"
    return "none"


def survey(home: Path, targets: list[str]) -> dict:
    registry = json.loads((home / "models.json").read_text())
    families = {k: v for k, v in registry.items() if not k.startswith("_")}

    wanted = {t.lower() for t in targets}
    rows: list[dict] = []
    seen_envs: set[str] = set()
    for family, spec in families.items():
        env = spec["env"]
        if wanted and family.lower() not in wanted and env.lower() not in wanted:
            continue
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
        rows.append(
            {
                "env": env,
                "families": [family],
                "gated": gated,
                "state": env_state(home / "envs" / env),
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
    return {
        "home": str(home),
        "envs": rows,
        "counts": counts,
        "to_build": to_build,
        "disk": {
            "free_gb": round(free_gb, 1),
            "budget_gb": budget_gb,
            "per_env_gb": PER_ENV_GB,
            "fits": free_gb >= budget_gb,
        },
        "token": {"available": source != "none", "source": source},
        "gated_envs": [r["env"] for r in rows if r["gated"]],
    }


def print_table(result: dict) -> None:
    print(f"oh-my-mlip setup survey — {result['home']}")
    print(f"{'env':<14} {'state':<14} gated  families")
    for row in result["envs"]:
        print(
            f"{row['env']:<14} {row['state']:<14} "
            f"{'yes' if row['gated'] else 'no':<6} {', '.join(row['families'])}"
        )
    c, d, t = result["counts"], result["disk"], result["token"]
    print(
        f"\nready {c['ready']} · partial {c['partial']} · broken {c['broken']}"
        f" · not installed {c['not_installed']}"
    )
    print(
        f"disk: {d['budget_gb']} GB needed for {len(result['to_build'])} builds"
        f" ({d['per_env_gb']} GB each; ready envs cost zero)"
        f" vs {d['free_gb']} GB free -> "
        + ("fits" if d["fits"] else "DOES NOT FIT")
    )
    print(
        "hf token: "
        + (f"available (source: {t['source']})" if t["available"] else "none found")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("targets", nargs="*", help="family or env names; default all")
    parser.add_argument("--table", action="store_true", help="human-readable output")
    args = parser.parse_args()

    result = survey(resolve_home(), args.targets)
    if args.table:
        print_table(result)
    else:
        json.dump(result, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
