#!/usr/bin/env python3
"""compare_equiv.py — the honesty gate for the 4-tier equivalence model.

Diffs two normalized ``equiv_result.json`` files (an OURS and a REF, each emitted
by ``scripts/run_equiv.py``) and prints a per-check PASS/FAIL verdict table. Exit
code 0 means the two runs are equivalent within tolerance; non-zero means a
breach (and the reasons are printed).

The comparison is tier-aware (see ``docs/equiv_protocol.md``):

  * provenance  — HARD-FAIL on a mismatch of any field that makes two runs not
    comparable: catbench_version, dataset_sha256, model, weights_sha256 (only
    when both are non-null), d3.
  * keys        — HARD-FAIL on any missing/extra system/structure key.
  * single-point (T2a) — per-structure energy max-abs-diff PER ATOM > --tol, or
    forces max-abs-diff > --force-tol.
  * relax (T2b) — per-system relaxed-adsorption-energy diff > --tol (per atom if
    natoms known, else absolute), OR terminal-geometry coordinate RMSD >
    --geom-tol. Terminal geometry MUST be present in BOTH files for relax; absent
    on either side is a FAIL. Positions are compared by RMSD, NEVER by hash.
  * time (T3)   — mean time_per_step ratio must fall inside --time-band, but ONLY
    when both provenance gpu_name values are equal AND non-null. Otherwise the
    time check is SKIPPED (printed) and never fails the run.

GPU-free: stdlib + numpy only. Never imports torch/ase/catbench.

Usage:
    python scripts/compare_equiv.py OURS.json REF.json \\
        [--tol 1e-3] [--force-tol 1e-2] [--geom-tol 1e-2] [--time-band 0.8,1.25]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Provenance fields that, when mismatched, make two runs not comparable at all.
# weights_sha256 is compared only when BOTH sides recorded a (non-null) value.
PROVENANCE_HARD_FIELDS = (
    "catbench_version",
    "dataset_sha256",
    "model",
    "d3",
)


class Check:
    """One named PASS/FAIL line in the verdict table."""

    __slots__ = ("name", "ok", "detail")

    def __init__(self, name: str, ok: bool, detail: str = "") -> None:
        self.name = name
        self.ok = ok
        self.detail = detail


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prov(doc: dict) -> dict:
    prov = doc.get("provenance")
    return prov if isinstance(prov, dict) else {}


def check_provenance(ours: dict, ref: dict) -> list[Check]:
    """HARD provenance comparability checks."""
    po, pr = _prov(ours), _prov(ref)
    checks: list[Check] = []
    for field in PROVENANCE_HARD_FIELDS:
        vo, vr = po.get(field), pr.get(field)
        ok = vo == vr
        detail = "" if ok else f"ours={vo!r} ref={vr!r}"
        checks.append(Check(f"provenance.{field}", ok, detail))
    # weights_sha256: only enforced when both sides are non-null.
    wo, wr = po.get("weights_sha256"), pr.get("weights_sha256")
    if wo is not None and wr is not None:
        ok = wo == wr
        detail = "" if ok else f"ours={str(wo)[:12]}… ref={str(wr)[:12]}…"
        checks.append(Check("provenance.weights_sha256", ok, detail))
    else:
        checks.append(
            Check(
                "provenance.weights_sha256",
                True,
                "skipped (one or both null)",
            )
        )
    return checks


def _mode(doc: dict) -> str:
    if "single_point" in doc:
        return "single-point"
    if "relax" in doc:
        return "relax"
    return _prov(doc).get("mode", "unknown")


def check_keys(ours: dict, ref: dict, section: str) -> tuple[Check, set[str]]:
    """Compare the system/structure key sets of a section ('single_point' or
    'relax'). Returns (check, shared_keys)."""
    ko = set(ours.get(section, {}))
    kr = set(ref.get(section, {}))
    missing = kr - ko  # in ref, absent from ours
    extra = ko - kr     # in ours, absent from ref
    ok = not missing and not extra
    parts = []
    if missing:
        parts.append(f"missing {sorted(missing)[:5]}")
    if extra:
        parts.append(f"extra {sorted(extra)[:5]}")
    return Check(f"{section}.keys", ok, "; ".join(parts)), (ko & kr)


def _coerce_natoms(entry: dict) -> int | None:
    n = entry.get("natoms")
    return int(n) if isinstance(n, (int, float)) and n else None


def check_single_point(
    ours: dict, ref: dict, keys: set[str], tol: float, force_tol: float
) -> list[Check]:
    so = ours.get("single_point", {})
    sr = ref.get("single_point", {})
    e_fails: list[str] = []
    f_fails: list[str] = []
    for k in sorted(keys):
        eo, er = so.get(k, {}), sr.get(k, {})
        if "error" in eo or "error" in er:
            e_fails.append(f"{k}: error recorded (ours={eo.get('error')!r} ref={er.get('error')!r})")
            continue
        if "energy" not in eo or "energy" not in er:
            e_fails.append(f"{k}: missing energy")
            continue
        natoms = _coerce_natoms(eo) or _coerce_natoms(er) or 1
        ediff = abs(float(eo["energy"]) - float(er["energy"])) / natoms
        if ediff > tol:
            e_fails.append(f"{k}: |dE|/atom={ediff:.3e} > {tol:.1e}")
        fo, fr = eo.get("fmax"), er.get("fmax")
        if fo is not None and fr is not None:
            fdiff = abs(float(fo) - float(fr))
            if fdiff > force_tol:
                f_fails.append(f"{k}: |dFmax|={fdiff:.3e} > {force_tol:.1e}")
    return [
        Check(
            "single_point.energy",
            not e_fails,
            "; ".join(e_fails[:5]) + (" …" if len(e_fails) > 5 else ""),
        ),
        Check(
            "single_point.forces",
            not f_fails,
            "; ".join(f_fails[:5]) + (" …" if len(f_fails) > 5 else ""),
        ),
    ]


def coord_rmsd(a: Any, b: Any) -> float | None:
    """Coordinate RMSD between two position arrays. None if shapes mismatch or
    either is absent/unparseable. NEVER a hash comparison."""
    if a is None or b is None:
        return None
    try:
        pa = np.asarray(a, dtype=float)
        pb = np.asarray(b, dtype=float)
    except (ValueError, TypeError):
        return None
    if pa.shape != pb.shape or pa.size == 0:
        return None
    diff = pa - pb
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


def check_relax(
    ours: dict, ref: dict, keys: set[str], tol: float, geom_tol: float
) -> list[Check]:
    ro = ours.get("relax", {})
    rr = ref.get("relax", {})
    e_fails: list[str] = []
    g_fails: list[str] = []
    for k in sorted(keys):
        eo, er = ro.get(k, {}), rr.get(k, {})
        # Energy comparison (per atom if natoms known on either side).
        evo, evr = eo.get("relaxed_ads_energy"), er.get("relaxed_ads_energy")
        if evo is None or evr is None:
            e_fails.append(f"{k}: missing relaxed_ads_energy")
        else:
            natoms = _coerce_natoms(eo) or _coerce_natoms(er) or 1
            ediff = abs(float(evo) - float(evr)) / natoms
            if ediff > tol:
                e_fails.append(f"{k}: |dE|={ediff:.3e} > {tol:.1e}")
        # Terminal geometry MUST be present in BOTH for relax.
        go = (eo.get("terminal_geom") or {}).get("positions")
        gr = (er.get("terminal_geom") or {}).get("positions")
        if go is None or gr is None:
            g_fails.append(f"{k}: terminal_geom missing (ours={'y' if go else 'n'} ref={'y' if gr else 'n'})")
            continue
        rmsd = coord_rmsd(go, gr)
        if rmsd is None:
            g_fails.append(f"{k}: geometry shape mismatch / unparseable")
        elif rmsd > geom_tol:
            g_fails.append(f"{k}: RMSD={rmsd:.3e} > {geom_tol:.1e}")
    return [
        Check(
            "relax.energy",
            not e_fails,
            "; ".join(e_fails[:5]) + (" …" if len(e_fails) > 5 else ""),
        ),
        Check(
            "relax.geometry_rmsd",
            not g_fails,
            "; ".join(g_fails[:5]) + (" …" if len(g_fails) > 5 else ""),
        ),
    ]


def _overall_time(doc: dict) -> float | None:
    agg = doc.get("relax", {})
    # run_equiv emits the aggregate under a reserved "_aggregate" key.
    aggregate = doc.get("aggregate") or agg.get("_aggregate") or {}
    t = aggregate.get("mean_time_per_step_overall")
    return float(t) if isinstance(t, (int, float)) and t else None


def check_time(ours: dict, ref: dict, band: tuple[float, float]) -> Check:
    """Time (T3) is compared ONLY when both gpu_name are equal and non-null."""
    go = _prov(ours).get("gpu_name")
    gr = _prov(ref).get("gpu_name")
    if not go or not gr or go != gr:
        return Check("time.ratio", True, "SKIPPED (not same-GPU)")
    to, tr = _overall_time(ours), _overall_time(ref)
    if to is None or tr is None or tr == 0:
        return Check("time.ratio", True, "SKIPPED (no time recorded)")
    ratio = to / tr
    lo, hi = band
    ok = lo <= ratio <= hi
    return Check("time.ratio", ok, f"ratio={ratio:.3f} band=[{lo},{hi}]")


def compare(
    ours: dict,
    ref: dict,
    tol: float = 1e-3,
    force_tol: float = 1e-2,
    geom_tol: float = 1e-2,
    time_band: tuple[float, float] = (0.8, 1.25),
) -> tuple[list[Check], bool]:
    """Run every applicable check. Returns (checks, all_ok)."""
    checks: list[Check] = []
    checks.extend(check_provenance(ours, ref))

    mode_o, mode_r = _mode(ours), _mode(ref)
    if mode_o != mode_r:
        checks.append(Check("mode", False, f"ours={mode_o!r} ref={mode_r!r}"))
        return checks, False
    checks.append(Check("mode", True, mode_o))

    if mode_o == "single-point":
        key_check, shared = check_keys(ours, ref, "single_point")
        checks.append(key_check)
        if key_check.ok:
            checks.extend(check_single_point(ours, ref, shared, tol, force_tol))
    elif mode_o == "relax":
        key_check, shared = check_keys(ours, ref, "relax")
        checks.append(key_check)
        if key_check.ok:
            checks.extend(check_relax(ours, ref, shared, tol, geom_tol))
        checks.append(check_time(ours, ref, time_band))
    else:
        checks.append(Check("mode", False, f"unknown mode {mode_o!r}"))

    all_ok = all(c.ok for c in checks)
    return checks, all_ok


def print_table(checks: list[Check]) -> None:
    print(f"{'check':<28} {'verdict':<7} detail")
    print(f"{'-' * 28} {'-' * 7} {'-' * 40}")
    for c in checks:
        verdict = "PASS" if c.ok else "FAIL"
        print(f"{c.name:<28} {verdict:<7} {c.detail}")


def _parse_band(s: str) -> tuple[float, float]:
    lo, hi = s.split(",")
    return float(lo), float(hi)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ours", help="OURS equiv_result.json")
    ap.add_argument("ref", help="REF (e.g. /TGM) equiv_result.json")
    ap.add_argument("--tol", type=float, default=1e-3, help="energy max-abs-diff per atom (eV/atom)")
    ap.add_argument("--force-tol", type=float, default=1e-2, help="forces max-abs-diff (eV/Angstrom)")
    ap.add_argument("--geom-tol", type=float, default=1e-2, help="terminal-geometry RMSD (Angstrom)")
    ap.add_argument("--time-band", type=str, default="0.8,1.25", help="time_per_step ratio band LO,HI")
    args = ap.parse_args(argv)

    ours = _load(Path(args.ours))
    ref = _load(Path(args.ref))
    band = _parse_band(args.time_band)

    checks, all_ok = compare(
        ours, ref,
        tol=args.tol,
        force_tol=args.force_tol,
        geom_tol=args.geom_tol,
        time_band=band,
    )
    print_table(checks)
    print()
    if all_ok:
        print("EQUIV: PASS")
        return 0
    reasons = ", ".join(c.name for c in checks if not c.ok)
    print(f"EQUIV: FAIL ({reasons})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
