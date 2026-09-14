#!/usr/bin/env python3
"""catbench_quickstart.py — run a catbench adsorption benchmark across the shipped
roster, locally (NO scheduler), using the public oh_my_mlip interface.

This is the public, single-machine form of the internal roster runner. Because
every model is a different conda env, no single process can host two models — so
each model runs in its own env process (oh_my_mlip handles the dispatch) and all
results land in a shared `result/` directory that catbench aggregates at the end.

This repo BUNDLES NO BENCHMARK DATA. The dataset is `raw_data/<tag>_adsorption.json`
in the current working directory; a TAG that is not there yet is fetched with
catbench's own loaders under an installed env interpreter (the Zenodo benchmark
through scripts/catbench_datasets.py --fetch, else cathub_preprocessing(TAG)).
Your own VASP tree: scripts/catbench_vasp_stage.py (recipes/catbench.md).

Run:
  source env.sh
  cd <your benchmark workdir>            # results go to ./result
  python <repo>/run_examples/catbench_quickstart.py [TAG] [--only MACE,SevenNet] [--d3]

Every run ALWAYS materializes its artifacts first (scripts/catbench_jobgen.py):
`jobs/catbench_<MLIP>.py` (verbatim resolve() codegen) and
`jobs/run_catbench_<MLIP>.sh` (the AC7 rerun unit — cd + env_run exports +
exec), then executes the `.sh` (never `python -c`). Extra flags:
  --all-versions        every declared version of each framework (the full
                          roster), except versions marked "catbench": false in
                          models.json; default is each framework's default_version
  --arch SMxx            GPU arch for arch-pinned models (NequIP/Allegro .pt2),
                          e.g. sm86 when the jobs run on a different GPU than
                          this host; default: auto-detected on this host
  --no-fetch             never fetch a missing TAG; stop instead
  --emit-only           materialize the job files; do not execute or submit
  --slurm               additionally emit jobs/run_slurm_<MLIP>.sh
  --partition PARTITION SBATCH partition for --slurm (default: gpu)
  --submit               dispatch via scripts/catbench_jobgen.submit() instead
                          of running the local .sh in this process (never
                          fired together with --emit-only)
  --catbench-version V   approved catbench pin (guard inside each job +
                          jobs/catbench_<MLIP>.meta.json)
  --regenerate           a rerun executes the job files already on disk; if
                          their content would change it stops instead of
                          overwriting — this flag replaces them explicitly
After the runs land in cwd/result/, aggregate with
scripts/catbench_report.py --result ./result --out ./report.

Skeleton mirrors catb_all: calc_num=3 instances per model,
config={mlip_name, benchmark}, results -> cwd/result, then analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_HOME = os.environ.get("OH_MY_MLIP_HOME") or str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _HOME)
sys.path.insert(0, str(Path(_HOME) / "scripts"))

from oh_my_mlip import RegistryError, list_models, list_versions, resolve  # noqa: E402

import catbench_jobgen  # noqa: E402

SUFFIX = "_adsorption.json"


def catbench_excluded(version: str | None, registry: dict | None = None) -> bool:
    """True for a version marked `"catbench": false` in models.json — usable for
    single runs, left out of catbench roster runs."""
    reg = registry if registry is not None else json.loads((Path(_HOME) / "models.json").read_text(encoding="utf-8"))
    for family, spec in reg.items():
        if family.startswith("_"):
            continue
        entry = spec.get("versions", {}).get(version)
        if entry is not None:
            return entry.get("catbench") is False
    return False


def _fetch_tag(tag: str, python: str, workdir: Path) -> bool:
    """Fetch raw_data/<tag>_adsorption.json with catbench's own loaders under a
    catbench-bearing env interpreter: a Zenodo benchmark through
    scripts/catbench_datasets.py --fetch (writes the provenance record), any
    other tag through cathub_preprocessing(tag)."""
    is_zenodo = ("import sys\nfrom catbench.adsorption import list_zenodo_benchmarks\n"
                 "sys.exit(0 if sys.argv[1] in list_zenodo_benchmarks() else 1)\n")
    if subprocess.run([python, "-c", is_zenodo, tag], cwd=workdir).returncode == 0:
        datasets = Path(_HOME) / "scripts" / "catbench_datasets.py"
        subprocess.run([python, str(datasets), "--fetch", tag, "--workdir", str(workdir)], cwd=workdir)
    else:
        cathub = "import sys\nfrom catbench.adsorption import cathub_preprocessing\ncathub_preprocessing(sys.argv[1])\n"
        subprocess.run([python, "-c", cathub, tag], cwd=workdir)
    return (workdir / "raw_data" / f"{tag}{SUFFIX}").is_file()


def _fetch_python(models: list, arch: str | None) -> str | None:
    """The first installed env interpreter among the selected models (every env
    ships catbench)."""
    for model in models:
        try:
            spec = resolve(model, arch=arch)
        except RegistryError:
            continue
        if _env_ready(spec):
            return spec["python"]
    return None


def _discover_tag(explicit, fetch_python=None):
    """`fetch_python` is a callable returning a catbench-bearing interpreter (or
    None to never fetch); it is only called when the TAG is missing."""
    raw = Path.cwd() / "raw_data"
    found = sorted(f.name[: -len(SUFFIX)] for f in raw.glob(f"*{SUFFIX}")) if raw.is_dir() else []
    if explicit:
        if explicit not in found:
            python = fetch_python() if fetch_python else None
            if python:
                print(f"  raw_data/{explicit}{SUFFIX} missing -> fetching with catbench (zenodo, else cathub) ...")
                if _fetch_tag(explicit, python, Path.cwd()):
                    return explicit
                print(f"[stop] could not fetch '{explicit}': not a Zenodo benchmark and cathub_preprocessing "
                      f"wrote no raw_data/{explicit}{SUFFIX}. Check the tag spelling.")
                sys.exit(2)
            print(f"[stop] raw_data/{explicit}{SUFFIX} not found in {raw}.")
            print("       Fetching needs an installed env (or pass --no-fetch and add the file yourself;"
                  " see run_examples/README.md).")
            sys.exit(2)
        return explicit
    if not found:
        print(f"[stop] no raw_data/*{SUFFIX} in {raw}. Pass a dataset TAG (fetched automatically).")
        print("       See run_examples/README.md.")
        sys.exit(2)
    if len(found) > 1:
        print(f"[choose] multiple datasets in raw_data: {found} — pass one as TAG.")
        sys.exit(2)
    print(f"  auto-detected dataset -> {found[0]}")
    return found[0]


def _env_ready(spec: dict) -> bool:
    """True if the model's env interpreter is materialized.

    Build-from-recipe writes ``$OH_MY_MLIP_HOME/envs/<env>/bin/python`` plus a
    ``.omm_ready`` sentinel (see install.sh). We accept either the sentinel or
    the interpreter itself so a fresh clone (no envs built) is detected BEFORE we
    dispatch a subprocess — otherwise launching a non-existent interpreter raises
    a raw FileNotFoundError that kills the whole roster on the first unbuilt env.
    """
    python = Path(spec["python"])
    sentinel = python.parent.parent / ".omm_ready"
    return sentinel.exists() or python.exists()


def _run_one_model(
    model: str,
    spec: dict,
    benchmark: str,
    calc_num: int,
    d3: bool,
    *,
    workdir: Path,
    emit_only: bool = False,
    use_slurm: bool = False,
    partition: str = "gpu",
    submit: bool = False,
    submit_hook=None,
    catbench_version: str | None = None,
    regenerate: bool = False,
) -> int:
    """Materialize this model's job artifacts, then (by default) execute them.

    ALWAYS materializes `jobs/catbench_<mlip_name>.py` +
    `jobs/run_catbench_<mlip_name>.sh` (+ `jobs/run_slurm_<mlip_name>.sh` if
    `use_slurm`) via `catbench_jobgen.emit()` — never runs `python -c` (F1's
    fixed AC7 violation). `env_run` now lives INSIDE the emitted `.sh` as
    `export` lines, so this process no longer patches a subprocess
    environment itself. The mlip_name gets a `_D3` suffix when D3 is on so
    results stay distinct.

    - `emit_only`: stop after materializing — no execution, no submit hook.
    - `submit`: dispatch through `catbench_jobgen.submit()` (the SLURM script
      if `use_slurm`, else the local runner) instead of running the local
      `.sh` directly in this process. Never combined with `emit_only`.
    - default (neither flag): `sh jobs/run_catbench_<mlip_name>.sh` in this
      process — the materialize-then-execute contract.
    - rerun: job files already on disk with unchanged bytes are executed as
      they are; files whose bytes WOULD change are never replaced silently —
      the run stops (exit 3) unless `regenerate` (--regenerate) was passed.
    - `catbench_version`: the approved pin, written into the job's version
      guard + `jobs/catbench_<mlip_name>.meta.json`.
    """
    mlip_name = spec.get("version", model) + ("_D3" if d3 else "")
    try:
        artifacts = catbench_jobgen.emit(
            spec, mlip_name, benchmark, calc_num, d3, workdir,
            slurm=use_slurm, partition=partition,
            catbench_version=catbench_version, overwrite=regenerate,
        )
    except FileExistsError as exc:
        print(f"  [stop] {model} ({mlip_name}): {exc}", file=sys.stderr)
        return 3
    print(f"  -> {model} ({mlip_name}) via {spec['python']}")
    for kind, path in artifacts.items():
        if kind == "reused":
            continue
        verb = "reused" if kind in artifacts["reused"] else "emitted"
        print(f"     {verb} {kind}: {path}")

    if emit_only:
        return 0

    if submit:
        target = artifacts.get("slurm", artifacts["sh"])
        rc = catbench_jobgen.submit(target, hook=submit_hook)
    else:
        proc = subprocess.run(["sh", str(artifacts["sh"])])
        rc = proc.returncode

    if rc != 0:
        print(f"  [fail] {model}: exit {rc}")
    return rc


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


def _resolve_versions_for(model: str, version_pins: dict, *, arch: str | None = None,
                          all_versions: bool = False) -> list:
    """Yield resolve() specs for a framework.

    - An explicit `--version MODEL=VER` pin wins (even for a "catbench": false
      version — the user asked for it by name; a warning says so).
    - `all_versions`: every declared version except "catbench": false ones.
    - Otherwise resolve(model) (which honors the framework's default_version).
    - If the framework is multi-version with NO default_version, resolve(model)
      raises; we warn LOUDLY on stderr and fan out across every declared version
      instead of silently skipping (so --only never dispatches zero subprocesses).
    `arch` selects the arch-pinned inference lines (NequIP/Allegro).
    """
    pinned = version_pins.get(model)
    if pinned is not None:
        if catbench_excluded(pinned):
            print(f"  [warn] {model}={pinned!r} is marked \"catbench\": false in models.json; running it as pinned",
                  file=sys.stderr)
        try:
            return [resolve(model, pinned, arch=arch)]
        except RegistryError as exc:
            print(f"  [warn] {model}={pinned!r} did not resolve: {exc}", file=sys.stderr)
            return []

    def fan_out() -> list:
        specs = []
        for ver in list_versions(model):
            if catbench_excluded(ver):
                print(f"  [skip] {model}/{ver}: \"catbench\": false in models.json", file=sys.stderr)
                continue
            try:
                specs.append(resolve(model, ver, arch=arch))
            except RegistryError as vexc:
                print(f"  [warn] {model}/{ver} did not resolve: {vexc}", file=sys.stderr)
        return specs

    if all_versions:
        return fan_out()
    try:
        return [resolve(model, arch=arch)]
    except RegistryError as exc:
        # Genuinely ambiguous (multi-version, no default_version): fan out.
        print(
            f"  [warn] {model} is multi-version with no default_version "
            f"({exc}); running ALL versions: {list_versions(model)}",
            file=sys.stderr,
        )
        return fan_out()


def main(argv: list[str] | None = None, *, submit_hook=None) -> int:
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
    ap.add_argument("--all-versions", action="store_true",
                     help='every declared version (full roster) except "catbench": false ones')
    ap.add_argument("--arch", default=None,
                     help="GPU arch for arch-pinned models, e.g. sm86 (default: auto-detected on this host)")
    ap.add_argument("--no-fetch", action="store_true", help="never fetch a missing TAG; stop instead")
    ap.add_argument("--emit-only", action="store_true",
                     help="materialize jobs/ artifacts; do not execute or submit")
    ap.add_argument("--slurm", action="store_true",
                     help="also emit jobs/run_slurm_<MLIP>.sh (no sbatch is issued)")
    ap.add_argument("--partition", default="gpu", help="SBATCH partition for --slurm (default: gpu)")
    ap.add_argument("--submit", action="store_true",
                     help="dispatch via catbench_jobgen.submit() instead of running the local .sh here")
    ap.add_argument("--catbench-version", default=None,
                     help="approved catbench version; each job guards on it and records it in jobs/*.meta.json")
    ap.add_argument("--regenerate", action="store_true",
                     help="replace existing job files whose content would change (a rerun never does this silently)")
    args = ap.parse_args(argv)

    if args.emit_only and args.submit:
        print("[stop] --emit-only and --submit are mutually exclusive.", file=sys.stderr)
        return 2

    only = [m.strip() for m in args.only.split(",")] if args.only else None
    version_pins = _parse_version_pins(args.version)
    models = [m for m in list_models() if (only is None or m in only)]
    if not models:
        print("[stop] no models selected.")
        return 2
    fetcher = None if args.no_fetch else (lambda: _fetch_python(models, args.arch))
    tag = _discover_tag(args.tag, fetcher)

    print(f"  benchmark : {tag}")
    print(f"  models    : {models}")
    print(f"  calc_num  : {args.calc_num}   D3: {args.d3}   arch: {args.arch or 'auto'}"
          f"   versions: {'all' if args.all_versions else 'default'}")
    print(f"  results   : {Path.cwd() / 'result'}")

    # Each model is a different conda env, so we materialize + dispatch ONE
    # job (jobs/catbench_<MLIP>.py + jobs/run_catbench_<MLIP>.sh) per
    # model+version using the registry's own interpreter + import/inference
    # (resolve()); env_run now lives inside the emitted .sh as export lines.
    # All runs write into the shared cwd/result that catbench aggregates.
    workdir = Path.cwd()
    dispatched = 0
    skipped = 0
    failed = 0
    for model in models:
        for spec in _resolve_versions_for(model, version_pins, arch=args.arch, all_versions=args.all_versions):
            # Check the env interpreter exists BEFORE dispatching: a fresh clone
            # has no envs built, and launching a missing interpreter would raise a
            # raw FileNotFoundError that crashes the whole roster on the first
            # unbuilt env. Skip loudly + actionably and keep going instead.
            if not _env_ready(spec):
                env = spec.get("env", "?")
                print(
                    f"  [skip] env {env!r} for {model} not installed "
                    f"— run: install.sh {model}",
                    file=sys.stderr,
                )
                skipped += 1
                continue
            model_rc = _run_one_model(
                model, spec, tag, args.calc_num, args.d3,
                workdir=workdir,
                emit_only=args.emit_only,
                use_slurm=args.slurm,
                partition=args.partition,
                submit=args.submit,
                submit_hook=submit_hook,
                catbench_version=args.catbench_version,
                regenerate=args.regenerate,
            )
            if model_rc != 0:
                failed += 1
            dispatched += 1

    if dispatched == 0:
        if skipped:
            print(
                f"[stop] no env materialized for the {skipped} selected "
                f"model run(s); build them first (see the install.sh hints above).",
                file=sys.stderr,
            )
        else:
            print("[stop] no model+version resolved to a runnable spec.", file=sys.stderr)
        return 2

    if args.emit_only:
        print(f"\nAll {dispatched} model job(s) emitted under {workdir / 'jobs'}. Nothing executed.")
        return 1 if failed else 0

    if failed:
        print(f"\n{dispatched - failed}/{dispatched} model run(s) succeeded, {failed} failed.")
    else:
        print(f"\nAll {dispatched} model run(s) finished. Aggregate with:")
        print(f"  python3 {Path(__file__).resolve().parent.parent / 'scripts' / 'catbench_report.py'} "
              f"--result {workdir / 'result'} --out {workdir / 'report'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
