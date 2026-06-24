#!/usr/bin/env python3
"""catbench_quickstart.py — run a catbench adsorption benchmark across the shipped
roster, locally (NO scheduler), using the public oh_my_mlip interface.

This is the public, single-machine form of the internal roster runner. Because
every model is a different conda env, no single process can host two models — so
each model runs in its own env process (oh_my_mlip handles the dispatch) and all
results land in a shared `result/` directory that catbench aggregates at the end.

This repo BUNDLES NO BENCHMARK DATA. Bring your own:
  put  raw_data/<tag>_adsorption.json  in the current working directory.
See run_examples/README.md for where to get / how to name your data.

Run:
  source env.sh
  cd <your benchmark workdir>            # must contain raw_data/<tag>_adsorption.json
  python <repo>/run_examples/catbench_quickstart.py [TAG] [--only MACE,SevenNet] [--d3]

Skeleton mirrors catb_all: calc_num=3 instances per model,
config={mlip_name, benchmark}, results -> cwd/result, then analysis.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

_HOME = os.environ.get("OH_MY_MLIP_HOME") or str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _HOME)

from oh_my_mlip import RegistryError, list_models, list_versions, resolve  # noqa: E402

SUFFIX = "_adsorption.json"

# Per-model catbench script. It runs INSIDE the model's own env interpreter
# (resolved via oh_my_mlip.resolve), builds calc_num calculator instances from
# the registry's import+inference lines, and runs catbench's AdsorptionCalculation
# with config={mlip_name, benchmark} — the exact skeleton of the internal
# catb_all runner. Results land in the shared cwd/result/ that AdsorptionAnalysis
# aggregates at the end.
_CATBENCH_TEMPLATE = '''\
import warnings
warnings.filterwarnings("ignore")
from catbench.adsorption import AdsorptionCalculation
{d3_import}{import_lines}

calc_num = {calc_num}
calculators = []
print("Calculators Initializing...")
for i in range(calc_num):
    print(f"{{i}}th calculator")
{inference_lines}
{d3_apply}    calculators.append(calc)

config = {{"mlip_name": {mlip_name!r}, "benchmark": {benchmark!r}}}
AdsorptionCalculation(calculators, **config).run()
print("[catbench] {mlip_name} done")
'''


def _discover_tag(explicit):
    raw = Path.cwd() / "raw_data"
    found = sorted(f.name[: -len(SUFFIX)] for f in raw.glob(f"*{SUFFIX}")) if raw.is_dir() else []
    if explicit:
        if explicit not in found:
            print(f"[stop] raw_data/{explicit}{SUFFIX} not found in {raw}.")
            print("       This repo bundles no data — add your own file (see run_examples/README.md).")
            sys.exit(2)
        return explicit
    if not found:
        print(f"[stop] no raw_data/*{SUFFIX} in {raw}. Provide your own dataset; pass its TAG.")
        print("       See run_examples/README.md.")
        sys.exit(2)
    if len(found) > 1:
        print(f"[choose] multiple datasets in raw_data: {found} — pass one as TAG.")
        sys.exit(2)
    print(f"  auto-detected dataset -> {found[0]}")
    return found[0]


def _build_script(spec: dict, mlip_name: str, benchmark: str, calc_num: int, d3: bool) -> str:
    """Render the per-model catbench script from a resolve() spec."""
    indent = "    "
    import_lines = "\n".join(spec["imports"])
    inference_lines = "\n".join(indent + ln for ln in spec["inference"])
    d3_import = "from catbench.dispersion import DispersionCorrection\n" if d3 else ""
    d3_apply = f"{indent}calc = DispersionCorrection().apply(calc)\n" if d3 else ""
    return _CATBENCH_TEMPLATE.format(
        d3_import=d3_import,
        import_lines=import_lines,
        calc_num=calc_num,
        inference_lines=inference_lines,
        d3_apply=d3_apply,
        mlip_name=mlip_name,
        benchmark=benchmark,
    )


def _run_one_model(model: str, spec: dict, benchmark: str, calc_num: int, d3: bool) -> int:
    """Run catbench for one model in its OWN env interpreter (subprocess).

    Uses spec['python'] (the env interpreter) and spec['env_run'] (parsed,
    allow-listed env vars applied to the subprocess environment, never shell).
    The mlip_name gets a _D3 suffix when D3 is on so results stay distinct.
    """
    mlip_name = spec.get("version", model) + ("_D3" if d3 else "")
    script = _build_script(spec, mlip_name, benchmark, calc_num, d3)
    child_env = dict(os.environ)
    child_env.update(spec.get("env_run", {}))
    child_env.setdefault("OH_MY_MLIP_HOME", _HOME)
    print(f"  -> {model} ({mlip_name}) via {spec['python']}")
    proc = subprocess.run(
        [spec["python"], "-c", script],
        env=child_env,
        cwd=str(Path.cwd()),
    )
    if proc.returncode != 0:
        print(f"  [fail] {model}: exit {proc.returncode}")
    return proc.returncode


def _parse_version_pins(pins: list) -> dict:
    """Parse repeated --version MODEL=VER flags into {model: version}."""
    out: dict = {}
    for pin in pins:
        if "=" not in pin:
            print(f"[stop] --version must be MODEL=VER, got {pin!r}", file=sys.stderr)
            sys.exit(2)
        model, ver = pin.split("=", 1)
        out[model.strip()] = ver.strip()
    return out


def _resolve_versions_for(model: str, version_pins: dict) -> list:
    """Yield resolve() specs for a framework.

    - An explicit `--version MODEL=VER` pin wins.
    - Otherwise resolve(model) (which honors the framework's default_version).
    - If the framework is multi-version with NO default_version, resolve(model)
      raises; we warn LOUDLY on stderr and fan out across every declared version
      instead of silently skipping (so --only never dispatches zero subprocesses).
    """
    pinned = version_pins.get(model)
    if pinned is not None:
        try:
            return [resolve(model, pinned)]
        except RegistryError as exc:
            print(f"  [warn] {model}={pinned!r} did not resolve: {exc}", file=sys.stderr)
            return []
    try:
        return [resolve(model)]
    except RegistryError as exc:
        # Genuinely ambiguous (multi-version, no default_version): fan out.
        versions = list_versions(model)
        print(
            f"  [warn] {model} is multi-version with no default_version "
            f"({exc}); running ALL versions: {versions}",
            file=sys.stderr,
        )
        specs = []
        for ver in versions:
            try:
                specs.append(resolve(model, ver))
            except RegistryError as vexc:
                print(f"  [warn] {model}/{ver} did not resolve: {vexc}", file=sys.stderr)
        return specs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tag", nargs="?", default=None, help="benchmark tag (raw_data/<tag>_adsorption.json)")
    ap.add_argument("--only", default=None, help="comma-separated framework filter (e.g. MACE,SevenNet)")
    ap.add_argument("--calc-num", type=int, default=3, help="calculator instances per model (catbench skeleton)")
    ap.add_argument("--d3", action="store_true", help="apply D3 (mlip_name gets a _D3 suffix)")
    ap.add_argument(
        "--version",
        action="append",
        default=[],
        metavar="MODEL=VER",
        help="pin a specific version for a framework (repeatable), e.g. --version MACE=MACE-MH-1-OMAT",
    )
    args = ap.parse_args()

    tag = _discover_tag(args.tag)
    only = [m.strip() for m in args.only.split(",")] if args.only else None
    version_pins = _parse_version_pins(args.version)
    models = [m for m in list_models() if (only is None or m in only)]
    if not models:
        print("[stop] no models selected.")
        return 2

    print(f"  benchmark : {tag}")
    print(f"  models    : {models}")
    print(f"  calc_num  : {args.calc_num}   D3: {args.d3}")
    print(f"  results   : {Path.cwd() / 'result'}")

    # Each model is a different conda env, so we dispatch ONE subprocess per
    # model+version using the registry's own interpreter + import/inference
    # (resolve()), applying the parsed, allow-listed env_run as the subprocess
    # environment. All runs write into the shared cwd/result that catbench
    # aggregates.
    rc = 0
    dispatched = 0
    for model in models:
        for spec in _resolve_versions_for(model, version_pins):
            rc |= _run_one_model(model, spec, tag, args.calc_num, args.d3)
            dispatched += 1

    if dispatched == 0:
        print("[stop] no model+version resolved to a runnable spec.", file=sys.stderr)
        return rc or 2

    print(f"\nAll {dispatched} model runs dispatched. Aggregate with catbench when they finish:")
    print("  from catbench.adsorption import AdsorptionAnalysis")
    print("  a = AdsorptionAnalysis(); a.analysis(); a.threshold_sensitivity_analysis()")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
