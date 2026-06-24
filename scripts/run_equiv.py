#!/usr/bin/env python3
"""run_equiv.py — the per-MLIP equivalence harness (T2a single-point / T2b relax).

Emits a normalized ``equiv_result.json`` carrying a full provenance block plus,
per mode, the numbers ``scripts/compare_equiv.py`` checks against a reference
run. Two modes (see ``docs/equiv_protocol.md`` for the tier model):

  --mode single-point  (T2a) — a CUSTOM per-structure loop over the dataset's
      ASE Atoms. catbench has NO single-point API (AdsorptionCalculation always
      relaxes), so this attaches the calculator to each fixed structure and
      records energy + forces-max WITHOUT any optimizer. Cross-GPU tolerant
      because a single-point is a pure function of the fixed geometry.

  --mode relax  (T2b/T3) — runs catbench AdsorptionCalculation on the dataset
      (n_crit_relax=5, save_files=True), then reads the per-system relaxed
      energies + timing from its result json and harvests terminal coordinates
      from the on-disk ``traj/<key>`` extxyz final frames. Same-GPU only.

This script runs INSIDE the model's own env interpreter (the one resolved by
``oh_my_mlip.resolve``), so it can exec the registry's import + inference lines
in-process to obtain a ``calc``. ALL heavy imports (torch/ase/catbench) are lazy
so ``--help`` and CI shape-checks need no GPU and no model env.

Usage:
    python scripts/run_equiv.py --mode single-point --model MACE \\
        [--version V] [--tag BackSingle2018] [--calc-num 1] [--d3] \\
        [--timestamp 2026-06-24T00:00:00Z] [--out equiv_result.json]

    python scripts/run_equiv.py --mode relax --model MACE \\
        [--n-crit-relax 5] [--calc-num 3] ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
SUFFIX = "_adsorption.json"


# ── provenance helpers (stdlib only) ──────────────────────────────────────────

def _sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _discover_tag(explicit: str | None) -> str:
    raw = Path.cwd() / "raw_data"
    found = sorted(f.name[: -len(SUFFIX)] for f in raw.glob(f"*{SUFFIX}")) if raw.is_dir() else []
    if explicit:
        if explicit not in found:
            print(f"[stop] raw_data/{explicit}{SUFFIX} not found in {raw}.", file=sys.stderr)
            sys.exit(2)
        return explicit
    if not found:
        print(f"[stop] no raw_data/*{SUFFIX} in {raw}. Pass --tag.", file=sys.stderr)
        sys.exit(2)
    if len(found) > 1:
        print(f"[choose] multiple datasets {found} — pass --tag.", file=sys.stderr)
        sys.exit(2)
    return found[0]


def _best_effort_versions() -> dict:
    """Collect runtime versions WITHOUT importing torch eagerly. Each lookup is
    wrapped so a missing package never aborts the run."""
    out: dict = {
        "catbench_version": None,
        "torch_version": None,
        "jax_version": None,
        "cuda_version": None,
        "gpu_name": None,
        "driver": None,
    }
    try:
        import catbench  # type: ignore
        out["catbench_version"] = getattr(catbench, "__version__", None)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        out["torch_version"] = getattr(torch, "__version__", None)
        out["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        if torch.cuda.is_available():  # pragma: no cover - needs GPU
            out["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if out["torch_version"] is None:
        try:
            import jax  # type: ignore
            out["jax_version"] = getattr(jax, "__version__", None)
        except Exception:
            pass
    # Driver: best-effort via nvidia-smi, never fatal.
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if res.returncode == 0 and res.stdout.strip():
            out["driver"] = res.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return out


def _weights_sha256(spec: dict) -> str | None:
    """Best-effort: if the resolved spec exposes a local checkpoint path that
    exists, sha256 it; else None. Never fatal."""
    for key in ("weights_path", "checkpoint", "weights"):
        val = spec.get(key)
        if isinstance(val, str) and val and os.path.sep in val:
            p = Path(os.path.expandvars(val))
            if p.is_file():
                try:
                    return _sha256_bytes(p)
                except OSError:
                    return None
    return None


def _build_provenance(args, tag: str, dataset_path: Path, n_systems: int, spec: dict) -> dict:
    vers = _best_effort_versions()
    backend = "torch" if vers["torch_version"] else ("jax" if vers["jax_version"] else "none")
    return {
        "mode": args.mode,
        "model": args.model,
        "version": spec.get("version", args.version),
        "tag": tag,
        "dataset_sha256": _sha256_bytes(dataset_path),
        "n_systems": n_systems,
        "catbench_version": vers["catbench_version"],
        "python_version": platform.python_version(),
        "backend": backend,
        "torch_version": vers["torch_version"],
        "jax_version": vers["jax_version"],
        "cuda_version": vers["cuda_version"],
        "gpu_name": vers["gpu_name"],
        "driver": vers["driver"],
        "d3": bool(args.d3),
        "weights_sha256": _weights_sha256(spec),
        "command": list(sys.argv),
        "timestamp": args.timestamp,
    }


# ── calculator construction (in-process exec of registry inference) ───────────

def _resolve_spec(model: str, version: str | None):
    sys.path.insert(0, str(_REPO_ROOT))
    from oh_my_mlip import resolve  # noqa: E402  (lazy)
    return resolve(model, version) if version else resolve(model)


def _build_calc(spec: dict, d3: bool):
    """Exec the registry's import + inference lines in-process to bind ``calc``."""
    ns: dict = {}
    for line in spec["imports"]:
        exec(line, ns)  # noqa: S102 - registry code is in-repo + reviewed
    for line in spec["inference"]:
        exec(line, ns)  # noqa: S102
    calc = ns.get("calc")
    if calc is None:
        raise RuntimeError("registry inference did not bind a `calc`")
    if d3:
        from catbench.dispersion import DispersionCorrection  # type: ignore
        calc = DispersionCorrection().apply(calc)
    return calc


# ── mode: single-point (T2a) ──────────────────────────────────────────────────

def _iter_systems(d: dict):
    for k in d:
        if k == "_structures":
            continue
        yield k


def run_single_point(args, tag: str, dataset_path: Path) -> dict:
    from catbench.utils.data_utils import load_catbench_json  # type: ignore
    import numpy as np

    data = load_catbench_json(str(dataset_path))
    spec = _resolve_spec(args.model, args.version)
    calc = _build_calc(spec, args.d3)

    sp: dict = {}
    n_err = 0
    for rkey in _iter_systems(data):
        raw = data[rkey].get("raw", {})
        for skey, sval in raw.items():
            atoms = sval.get("atoms")
            if atoms is None:
                continue
            entry_key = f"{rkey}::{skey}"
            try:
                a = atoms.copy()
                a.calc = calc
                energy = float(a.get_potential_energy())
                forces = a.get_forces()
                fmax = float(np.abs(forces).max())
                sp[entry_key] = {
                    "energy": energy,
                    "fmax": fmax,
                    "natoms": int(len(a)),
                    "formula": a.get_chemical_formula(),
                }
            except Exception as e:  # noqa: BLE001 - resilience: record + continue
                sp[entry_key] = {"error": str(e)}
                n_err += 1

    n_systems = sum(1 for _ in _iter_systems(data))
    out = {
        "provenance": _build_provenance(args, tag, dataset_path, n_systems, spec),
        "single_point": sp,
        "summary": {"n_structures": len(sp), "n_errors": n_err},
    }
    return out


# ── mode: relax (T2b/T3) ──────────────────────────────────────────────────────

def _read_terminal_geom(traj_dir: Path):
    """Read the FINAL frame of the first extxyz under traj/<key>. Returns
    {positions, cell, natoms} (positions rounded to 6 dp) or None."""
    if not traj_dir.is_dir():
        return None
    try:
        from ase.io import read  # type: ignore
    except Exception:
        return None
    candidates = sorted(traj_dir.iterdir())
    for f in candidates:
        if not f.is_file():
            continue
        try:
            frames = read(str(f), index=":")
        except Exception:
            continue
        if not frames:
            continue
        last = frames[-1]
        pos = [[round(float(x), 6) for x in row] for row in last.get_positions()]
        cell = [[round(float(x), 6) for x in row] for row in last.get_cell()]
        return {"positions": pos, "cell": cell, "natoms": int(len(last))}
    return None


def _safe(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def run_relax(args, tag: str, dataset_path: Path) -> dict:
    from catbench.adsorption import AdsorptionCalculation  # type: ignore
    from catbench.utils.data_utils import load_catbench_json  # type: ignore

    spec = _resolve_spec(args.model, args.version)
    mlip_name = (spec.get("version", args.model)) + ("_D3" if args.d3 else "")

    calculators = [_build_calc(spec, args.d3) for _ in range(args.calc_num)]
    AdsorptionCalculation(
        calculators,
        mlip_name=mlip_name,
        benchmark=tag,
        n_crit_relax=args.n_crit_relax,
        save_files=True,
    ).run()

    result_dir = Path.cwd() / "result" / mlip_name
    result_json = result_dir / f"{mlip_name}_result.json"
    traj_root = result_dir / "traj"
    raw_results = {}
    if result_json.is_file():
        try:
            raw_results = json.loads(result_json.read_text(encoding="utf-8"))
        except Exception:
            raw_results = {}

    # Reference adsorption energies (from the dataset) for context.
    data = load_catbench_json(str(dataset_path))

    relax: dict = {}
    times: list[float] = []
    anomaly_counts: dict = {}
    for rkey, rres in raw_results.items():
        if not isinstance(rres, dict):
            continue
        final = rres.get("final", {}) if isinstance(rres.get("final"), dict) else {}
        # relaxed adsorption energy: catbench stores it as ads_eng_median (final)
        relaxed_e = _safe(final, "ads_eng_median")
        mean_t = _safe(final, "time_per_step")
        if isinstance(mean_t, (int, float)) and mean_t:
            times.append(float(mean_t))
        anomaly = rres.get("anomaly")
        if anomaly is not None:
            anomaly_counts[str(anomaly)] = anomaly_counts.get(str(anomaly), 0) + 1
        ref_eng = _safe(data.get(rkey, {}), "reaction", "ads_eng") if rkey in data else None

        geom = _read_terminal_geom(traj_root / rkey)
        relax[rkey] = {
            "relaxed_ads_energy": float(relaxed_e) if isinstance(relaxed_e, (int, float)) else None,
            "ref_ads_eng": float(ref_eng) if isinstance(ref_eng, (int, float)) else None,
            "anomaly": anomaly,
            "slab_max_disp": _safe(final, "slab_max_disp"),
            "adslab_max_disp": _safe(final, "adslab_max_disp"),
            "mean_time_per_step": float(mean_t) if isinstance(mean_t, (int, float)) else None,
            "natoms": (geom or {}).get("natoms"),
            "terminal_geom": geom,
        }

    # Aggregate metrics (best-effort; MAE/RMSE over systems with both energies).
    diffs = [
        v["relaxed_ads_energy"] - v["ref_ads_eng"]
        for v in relax.values()
        if v["relaxed_ads_energy"] is not None and v["ref_ads_eng"] is not None
    ]
    import math
    mae = sum(abs(x) for x in diffs) / len(diffs) if diffs else None
    rmse = math.sqrt(sum(x * x for x in diffs) / len(diffs)) if diffs else None
    mean_time_overall = sum(times) / len(times) if times else None

    n_systems = sum(1 for _ in _iter_systems(data))
    out = {
        "provenance": _build_provenance(args, tag, dataset_path, n_systems, spec),
        "relax": relax,
        "aggregate": {
            "MAE": mae,
            "RMSE": rmse,
            "anomaly_counts": anomaly_counts,
            "mean_time_per_step_overall": mean_time_overall,
            "n_systems": len(relax),
        },
    }
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["single-point", "relax"])
    ap.add_argument("--model", required=True, help="framework key (e.g. MACE)")
    ap.add_argument("--version", default=None, help="pin a specific version")
    ap.add_argument("--tag", default=None, help="benchmark tag (raw_data/<tag>_adsorption.json)")
    ap.add_argument("--n-crit-relax", type=int, default=5, help="catbench relax cap (relax mode)")
    ap.add_argument("--calc-num", type=int, default=None, help="calculator instances (default 3 relax / 1 single-point)")
    ap.add_argument("--d3", action="store_true", help="apply identical D3 dispersion on both sides")
    ap.add_argument("--timestamp", default=None, help="ISO timestamp stamped by the caller (do NOT auto-generate)")
    ap.add_argument("--out", default="equiv_result.json", help="output json path")
    args = ap.parse_args(argv)

    if args.calc_num is None:
        args.calc_num = 1 if args.mode == "single-point" else 3

    tag = _discover_tag(args.tag)
    dataset_path = Path.cwd() / "raw_data" / f"{tag}{SUFFIX}"

    if args.mode == "single-point":
        out = run_single_point(args, tag, dataset_path)
    else:
        out = run_relax(args, tag, dataset_path)

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    prov = out["provenance"]
    print(f"[run_equiv] mode={args.mode} model={args.model} tag={tag}")
    print(f"  dataset_sha256={prov['dataset_sha256'][:12]}… n_systems={prov['n_systems']}")
    print(f"  gpu={prov['gpu_name']} catbench={prov['catbench_version']}")
    print(f"  wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
