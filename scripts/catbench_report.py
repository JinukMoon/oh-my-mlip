#!/usr/bin/env python3
"""catbench_report.py — turn a catbench_quickstart.py `result/` directory into
a portable report: `report/mae_table.{md,csv}` + `report/mae_comparison.png`.

`catbench.adsorption.AdsorptionAnalysis` is the only thing that computes MAE
(reimplementing the metric would fork it from the library this hub consumes —
see AGENTS.md's "consume, never modify catbench" Non-Goal), so this script
imports `catbench` directly. `catbench` lives INSIDE the model envs, never the
ambient interpreter (N1) — so BEFORE that import happens, this script re-execs
itself under a catbench-bearing interpreter resolved through
`oh_my_mlip.resolve()` (E1: the guard runs above the `from catbench...` import,
so launching under a bare interpreter re-execs instead of failing at module
load). Default target: the env of the first model present in `--result`;
override with `--python`. The resolved interpreter (and the invocation
directory) are recorded in the emitted `report/run_report.sh` — the third
rerun unit in this pipeline, carrying the same `cd` contract as the catbench
jobgen runner (N4) so a rerun from a DIFFERENT cwd reproduces byte-for-byte.

`AdsorptionAnalysis.analysis()` emits its own `.xlsx` + plots as a side
effect (unchanged, "wrap it" per the consensus plan) — this script wraps the
private hook it already calls with the exact per-model MAE rows
(`_create_excel_output`) so the derived table/plot use the SAME numbers
without needing an xlsx-reader engine that not every env ships (some model
envs carry `xlsxwriter` for writing but not `openpyxl` for reading back).

House plotting rules: `savefig` only, never `plt.show()`, never
`plt.grid(True)`.

Usage:
  python3 scripts/catbench_report.py --result ./result --out ./report
  python3 scripts/catbench_report.py --result ./result --out ./report --python /path/to/env/bin/python
  sh ./report/run_report.sh          # emitted rerun unit; reproduces the report from any cwd
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_REEXEC_MARKER = "OMM_CATBENCH_REPORT_REEXEC"

# Column order mirrors AdsorptionAnalysis's own MLIP_Data sheet layout
# (catbench/adsorption/analysis.py: `_create_mlip_data_sheet`), minus the
# per-adsorbate breakdown columns this report does not need.
_COLUMNS = ["MLIP_name", "MAE_total", "MAE_normal", "MAE_single", "ADwT", "AMDwT", "Num_total"]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", default="./result", help="catbench result/ directory (default: ./result)")
    ap.add_argument("--out", default="./report", help="report output directory (default: ./report)")
    ap.add_argument("--python", default=None, help="override the catbench-bearing interpreter (skips resolve())")
    ap.add_argument("--expect-models", default=None,
                    help="comma-separated approved mlip_names: refuse when result/ holds anything else "
                         "(no baseline reruns — recipes/catbench.md §5)")
    ap.add_argument("--catbench-version", default=None,
                    help="approved catbench version: refuse to aggregate under any other")
    return ap.parse_args(argv)


def check_expected_models(result_dir: Path, expected: list[str]) -> dict:
    """`result/` must hold exactly the approved selection, nothing more. Extra
    subdirectories mean a model outside the approval ran (a baseline rerun);
    that is refused before any analysis."""
    present = sorted(p.name for p in result_dir.iterdir() if p.is_dir()) if result_dir.is_dir() else []
    extra = sorted(set(present) - set(expected))
    missing = sorted(set(expected) - set(present))
    return {"ok": not extra, "present": present, "extra": extra, "missing": missing}


def _default_model(result_dir: Path) -> str | None:
    """First model subdirectory under result/, sorted — the default re-exec target."""
    if not result_dir.is_dir():
        return None
    names = sorted(p.name for p in result_dir.iterdir() if p.is_dir())
    return names[0] if names else None


def _resolve_interpreter(result_dir: Path) -> str:
    """Resolve a catbench-bearing interpreter from the first model in result/."""
    sys.path.insert(0, str(REPO))
    from oh_my_mlip import RegistryError, resolve  # noqa: E402

    model = _default_model(result_dir)
    if model is None:
        print(f"[stop] no result/<model>/ subdirectory found under {result_dir}", file=sys.stderr)
        raise SystemExit(2)
    key = model[: -len("_D3")] if model.endswith("_D3") else model
    try:
        spec = resolve(key)
    except RegistryError as exc:
        print(f"[stop] could not resolve a catbench-bearing interpreter for {key!r}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    return spec["python"]


def _capture_excel_wrapper(orig_create_excel, captured: dict):
    """Wrap `AdsorptionAnalysis._create_excel_output` so this script reads the SAME
    per-model rows catbench renders, without reimplementing the metric.

    `*args`/`**kwargs`, never a fixed argument list: catbench 1.1.4 calls the hook with
    the gas-shift-corrected twins (`main_data_shifted`, `anomaly_data_shifted`) as
    keywords, and a wrapper with a fixed tail raises TypeError there — which kills the
    whole report. Everything is forwarded untouched, so the workbook keeps every sheet.
    """
    def _capture_and_delegate(self, main_data, *args, **kwargs):
        captured["main_data"] = main_data
        return orig_create_excel(self, main_data, *args, **kwargs)

    return _capture_and_delegate


def _reexec_before_catbench_import(args: argparse.Namespace, result_dir: Path) -> None:
    """E1: resolve + re-exec BEFORE `import catbench`.

    Must run above every `from catbench...` line in this module — otherwise
    launching under an interpreter without catbench fails at module load
    instead of re-execing, defeating the whole point of N2's contract (and
    making a future `pytest tests/` collection of this file unsafe).
    """
    if os.environ.get(_REEXEC_MARKER) == "1":
        return
    python = args.python or _resolve_interpreter(result_dir)
    if os.path.realpath(python) == os.path.realpath(sys.executable):
        os.environ[_REEXEC_MARKER] = "1"
        return
    env = dict(os.environ)
    env[_REEXEC_MARKER] = "1"
    os.execve(python, [python, str(Path(__file__).resolve())] + sys.argv[1:], env)


def _write_mae_table(out_dir: Path, main_data: list[dict]) -> None:
    csv_path = out_dir / "mae_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in main_data:
            writer.writerow({k: row.get(k) for k in _COLUMNS})

    md_path = out_dir / "mae_table.md"
    header = ["MLIP", "MAE_total (eV)", "MAE_normal (eV)", "MAE_single (eV)", "ADwT (%)", "AMDwT (%)", "N"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in main_data:
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:.2f} | {} |".format(
                row["MLIP_name"], row["MAE_total"], row["MAE_normal"],
                row["MAE_single"], row["ADwT"], row["AMDwT"], row["Num_total"],
            )
        )
    md_path.write_text("\n".join(lines) + "\n")


def _write_mae_plot(out_dir: Path, main_data: list[dict], plt) -> None:
    """House rules: savefig only, never plt.show(), never plt.grid(True)."""
    names = [row["MLIP_name"] for row in main_data]
    mae_total = [row["MAE_total"] for row in main_data]
    mae_normal = [row["MAE_normal"] for row in main_data]

    fig, ax = plt.subplots(figsize=(max(4.0, 1.2 * len(names) + 2), 5))
    x = range(len(names))
    width = 0.35
    ax.bar([i - width / 2 for i in x], mae_total, width, label="MAE_total")
    ax.bar([i + width / 2 for i in x], mae_normal, width, label="MAE_normal")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("MAE (eV)")
    ax.set_title("CatBench adsorption-energy MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mae_comparison.png", dpi=150)
    plt.close(fig)


def _write_run_report_sh(out_dir: Path, python: str, result_dir: Path, invocation_cwd: Path,
                         expected: list[str] | None = None, catbench_version: str | None = None) -> Path:
    """E2/N4: the third emitted `.sh` carries the same `cd` contract as the
    catbench jobgen runner units (verified from a DIFFERENT cwd — E3). The
    approval guards (`--expect-models`, `--catbench-version`) are carried
    into the rerun line so a rerun keeps the same refusals."""
    sh_path = out_dir / "run_report.sh"
    report_py = str(Path(__file__).resolve())
    guards = ""
    if expected:
        guards += f' --expect-models "{",".join(expected)}"'
    if catbench_version:
        guards += f' --catbench-version "{catbench_version}"'
    lines = [
        "#!/bin/sh",
        "# generated by scripts/catbench_report.py -- rerun this file to reproduce the report",
        "set -eu",
        f'cd "{invocation_cwd}"',
        f'exec "{python}" "{report_py}" --result "{result_dir}" --out "{out_dir}" --python "{python}"{guards}',
    ]
    sh_path.write_text("\n".join(lines) + "\n")
    sh_path.chmod(0o755)
    return sh_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    invocation_cwd = Path.cwd().resolve()
    result_dir = Path(args.result).resolve()
    out_dir = Path(args.out).resolve()

    expected = [m.strip() for m in args.expect_models.split(",") if m.strip()] if args.expect_models else None
    if expected is not None:
        guard = check_expected_models(result_dir, expected)
        if not guard["ok"]:
            print(f"[stop] result/ holds models outside the approved selection: {guard['extra']} "
                  f"(present={guard['present']}); no baseline reruns are analysed", file=sys.stderr)
            return 3
        if guard["missing"]:
            print(f"[stop] approved models without a result/ entry: {guard['missing']} — the report must list "
                  f"every selected model and nothing else (recipes/catbench.md §5)", file=sys.stderr)
            return 3

    _reexec_before_catbench_import(args, result_dir)

    # From here on we are guaranteed to run under a catbench-bearing
    # interpreter (E1) — only now is it safe to import catbench.
    import warnings
    warnings.filterwarnings("ignore")
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import catbench
    from catbench.adsorption import AdsorptionAnalysis

    if args.catbench_version and catbench.__version__ != args.catbench_version:
        print(f"[stop] approved catbench {args.catbench_version} but this interpreter has {catbench.__version__}",
              file=sys.stderr)
        return 3

    if not result_dir.is_dir():
        print(f"[stop] no result/ directory at {result_dir}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    # Wrap AdsorptionAnalysis (never reimplement the MAE metric, never modify
    # catbench itself): intercept the private hook it already calls with the
    # exact per-model rows it renders into its own xlsx, so this script's
    # table/plot are derived from those SAME numbers rather than re-parsing
    # the xlsx it wrote (which would need a reader engine not every env has).
    captured: dict = {}
    orig_create_excel = AdsorptionAnalysis._create_excel_output

    AdsorptionAnalysis._create_excel_output = _capture_excel_wrapper(orig_create_excel, captured)
    cwd_before = Path.cwd()
    try:
        os.chdir(out_dir)  # catbench writes its own xlsx/plot outputs relative to cwd
        analysis = AdsorptionAnalysis(
            calculating_path=str(result_dir),
            benchmarking_name="catbench_report",
            plot_enabled=False,
        )
        analysis.analysis()
        # Classification-rate-vs-threshold charts (displacement and bond-length
        # thresholds), written by catbench next to its own analysis outputs.
        analysis.threshold_sensitivity_analysis()
    finally:
        os.chdir(cwd_before)
        AdsorptionAnalysis._create_excel_output = orig_create_excel

    main_data = captured.get("main_data", [])
    if not main_data:
        print(f"[stop] AdsorptionAnalysis produced no rows from {result_dir}", file=sys.stderr)
        return 2

    _write_mae_table(out_dir, main_data)
    _write_mae_plot(out_dir, main_data, plt)
    sh_path = _write_run_report_sh(out_dir, sys.executable, result_dir, invocation_cwd, expected, args.catbench_version)
    # Provenance the comparison step needs: which catbench produced these rows.
    (out_dir / "report_meta.json").write_text(json.dumps({
        "catbench_version": catbench.__version__, "python": sys.executable,
        "result_dir": str(result_dir), "models": [row["MLIP_name"] for row in main_data],
        "expected_models": expected,
    }, indent=2) + "\n")

    print(f"wrote {out_dir / 'mae_table.md'}")
    print(f"wrote {out_dir / 'mae_table.csv'}")
    print(f"wrote {out_dir / 'mae_comparison.png'}")
    print(f"wrote {sh_path} (rerun to reproduce)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
