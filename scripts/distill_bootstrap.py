#!/usr/bin/env python3
"""distill_bootstrap.py <--teacher V> <--structure FILE> <--work DIR>
[--repo PATH] [--target-ps F] [--lmp-bin PATH] -- render everything a run of
onthefly-distill (`D`, GPL-2.0, orchestrated -- never absorbed) needs to
distill one oh-my-mlip teacher into a CPU LAMMPS NN-MTP student (P4.2).

Deterministic, write-then-execute (Principle 2): every artifact below is
written to the ABSOLUTE work dir before anything computes. Each step is
printed to stdout and appended to `<work>/bootstrap.log`.

  1. locate D                       -- `--repo`, else `$ONTHEFLY_REPO`, else
                                        `~/01_2026/onthefly-distill`. No clone,
                                        no `pip install`, no env mutation
                                        (D2b) -- `D` is invoked in place.
  2. `omm_teacher.py`                -- a zero-arg `make_calc()` whose body is
                                        `resolve(<teacher>)`'s import +
                                        inference lines pasted VERBATIM. The
                                        frozen registry codegen IS what
                                        labels; editing a character here would
                                        make the teacher an unvalidated model.
  3. `config.yaml`                   -- from `D/config.example.yaml`'s own
                                        structure (deep-merge overlay, so
                                        every key `D/ontheflydistill/config.py`
                                        reads stays present): teacher ->
                                        ase_calculator/omm_teacher:make_calc,
                                        python_bin -> the TEACHER env's own
                                        interpreter (so every AL-loop
                                        subprocess -- train/MD/label -- runs
                                        inside an env that already has the
                                        teacher framework), lmp_bin, work_dir
                                        as an ABSOLUTE path (F21/N3), and
                                        system.* derived from `--structure`.
  4. `run_distill.sh`                -- PATH pin (F17) + PYTHONPATH covering
                                        both `D` and the work dir (F16) +
                                        `ONTHEFLY_CONFIG` (F21/N3), then seeds
                                        the initial AL-pool dataset exactly as
                                        `examples/ptwater_acid/README.md`
                                        documents (`teacher_md.py` ->
                                        `merge_xyz`) before `exec`ing
                                        `D/scripts/al_loop_local.sh` VERBATIM
                                        -- never a copy, never reimplemented.

`pair_nnmtp` v1 (the style the loop's LAMMPS build uses -- v2 is unused,
F18) hard-codes `species_Z[4]`, so a structure with more than 4 distinct
elements is refused here with an actionable error before anything is
written.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oh_my_mlip import resolve  # noqa: E402

try:
    import yaml
    from ase.data import atomic_masses
    from ase.io import read as ase_read
except ImportError as exc:  # pragma: no cover - environment hint
    raise ImportError(
        "distill_bootstrap.py needs ase + pyyaml on the interpreter that runs "
        "it -- e.g. `/home/jumoon/miniconda3/envs/toolkit/bin/python "
        "scripts/distill_bootstrap.py ...` -- not the ambient system python."
    ) from exc

MAX_SPECIES = 4  # pair_nnmtp v1: `int species_Z[4]` (F18) -- v2 is unbounded
                 # but is documented as "not used by the loop"; bound to v1.

DEFAULT_REPO = os.environ.get(
    "ONTHEFLY_REPO", str(Path.home() / "01_2026" / "onthefly-distill")
)
DEFAULT_LMP_BIN = str(
    Path.home() / ".cache" / "oh-my-mlip" / "lammps" / "build" / "lmp"
)

# Kept tiny so the AC2 demo finishes in minutes, not hours -- a real
# production run should raise these by editing the emitted run_distill.sh
# directly (it is a plain, rerunnable shell script, not a hidden default).
TEACHER_MD_STEPS = 200
TEACHER_MD_SAVE_EVERY = 10


def _log(work: Path | None, msg: str) -> None:
    line = f"[distill_bootstrap] {msg}"
    print(line, flush=True)
    if work is not None:
        with open(work / "bootstrap.log", "a") as fh:
            fh.write(line + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--teacher", required=True, help="oh-my-mlip model or version, e.g. MACE-MPA-0")
    ap.add_argument("--structure", required=True, type=Path, help="anything ase.io.read handles, <=4 species")
    ap.add_argument("--work", required=True, type=Path, help="work dir (created; rendered ABSOLUTE)")
    ap.add_argument("--repo", type=Path, default=Path(DEFAULT_REPO), help="onthefly-distill checkout (D)")
    ap.add_argument("--target-ps", type=float, default=1.0, help="al_loop.target_ps (small default for a quick demo)")
    ap.add_argument("--lmp-bin", type=Path, default=Path(DEFAULT_LMP_BIN), help="LAMMPS binary with pair_style nnmtp")
    return ap.parse_args(argv)


def check_repo(repo: Path, work: Path) -> tuple[Path, Path]:
    """Step 1: locate D. No clone, no install -- just verify it is what it claims."""
    _log(work, f"step 1/4: locate onthefly-distill at {repo}")
    loop = repo / "scripts" / "al_loop_local.sh"
    example_cfg = repo / "config.example.yaml"
    teacher_md = repo / "scripts" / "teacher_md.py"
    missing = [p for p in (loop, example_cfg, teacher_md) if not p.is_file()]
    if missing:
        raise SystemExit(
            f"{repo} does not look like an onthefly-distill checkout -- missing "
            f"{[str(p) for p in missing]}. Pass --repo or set $ONTHEFLY_REPO."
        )
    _log(work, f"  found {loop}")
    return loop, example_cfg


def read_structure(path: Path, work: Path):
    """Step 2 (part 1): derive system.{species,specorder,masses} from --structure.

    First-seen element order is used for `specorder` (and therefore
    `species`/`masses`) so the mapping is deterministic for a given file.
    """
    _log(work, f"step 2/4: read structure {path}")
    if not path.is_file():
        raise SystemExit(f"--structure {path} does not exist")
    atoms = ase_read(str(path))
    symbols = atoms.get_chemical_symbols()
    numbers = atoms.get_atomic_numbers()
    z_by_symbol: dict[str, int] = {}
    for sym, z in zip(symbols, numbers):
        z_by_symbol.setdefault(sym, int(z))
    specorder = list(z_by_symbol.keys())  # first-seen order
    if len(specorder) > MAX_SPECIES:
        raise SystemExit(
            f"{path} has {len(specorder)} distinct species {specorder}, but "
            f"onthefly-distill's LAMMPS AL loop uses pair_nnmtp v1, which "
            f"hard-codes `species_Z[4]` (pair_nnmtp v2 is unbounded but is "
            f"documented as not used by the loop) -- reduce to <= {MAX_SPECIES} "
            f"species or pick a different structure."
        )
    species = [z_by_symbol[s] for s in specorder]
    masses = [round(float(atomic_masses[z]), 6) for z in species]
    _log(work, f"  specorder={specorder} species(Z)={species} masses={masses}")
    return specorder, species, masses


def resolve_teacher(teacher: str, work: Path) -> dict:
    """Step 2 (part 2): resolve the teacher through the registry -- never by hand."""
    _log(work, f"step 2/4: resolve teacher {teacher!r} via oh_my_mlip.resolve()")
    spec = resolve(teacher)
    _log(work, f"  -> family={spec['model']} version={spec['version']} env={spec['env']} python={spec['python']}")
    return spec


def render_omm_teacher(spec: dict) -> str:
    """omm_teacher.py: zero-arg make_calc(), body = resolve()'s import+inference
    lines pasted VERBATIM (never edited -- these exact lines passed this hub's
    equivalence validation). Resolved via `importlib.import_module("omm_teacher")`
    by D's ase_calculator teacher (F16), so it must sit ON the work dir which
    `run_distill.sh` puts on PYTHONPATH.

    Both the import and the inference line run INSIDE make_calc(), under a
    stdout/stderr suppression context -- not a hand edit of either line, only
    where they execute. This is load-bearing, not cosmetic: several
    frameworks (MACE's `mace_mp()` among them) `print()` banner/progress text
    at import and construction time; `D/scripts/can_relabel.py` builds this
    exact teacher and captures its process's stdout via bash command
    substitution (`al_loop_local.sh:44 CAN_RELABEL="$($PY can_relabel.py ...)"`
    then compares it to the literal string `"1"`). Unsuppressed banner text
    contaminates that capture, so the AL-capable ase_calculator teacher gets
    misread as unable to relabel and the loop silently falls back to
    one-shot distillation (`model_scratch1`/`run_scratch1`) instead of the
    AL loop (`model_scratch0`/`run_scratch0`) -- confirmed against this
    hub's MACE-MPA-0 teacher during the AC2 demo."""
    import_lines = "\n".join(spec["imports"])
    inference_lines = "\n".join(spec["inference"])
    body = textwrap.indent(import_lines + "\n" + inference_lines, "        ")
    return (
        '"""omm_teacher.py -- generated by scripts/distill_bootstrap.py.\n\n'
        "Zero-arg make_calc() for onthefly-distill's teacher.type: "
        "ase_calculator path. The import + inference lines below are pasted "
        f"VERBATIM from oh_my_mlip.resolve({spec['version']!r}) -- do not edit "
        "a character; a modified line is an unvalidated model. They run under "
        "a stdout/stderr suppression context because some frameworks print "
        "banner/progress text at import/construction time, which would "
        "otherwise contaminate D/scripts/can_relabel.py's captured-stdout "
        'check (see module docstring).\n"""\n'
        "import contextlib\n"
        "import os\n\n\n"
        "def make_calc():\n"
        "    with open(os.devnull, \"w\") as _devnull, \\\n"
        "            contextlib.redirect_stdout(_devnull), \\\n"
        "            contextlib.redirect_stderr(_devnull):\n"
        f"{body}\n"
        "    return calc\n"
    )


def _deep_merge(base: dict, override: dict) -> dict:
    """Same algorithm as D/ontheflydistill/config.py's own `_deep_merge` --
    override wins, recursion only into dict values, nothing is ever dropped
    from `base`. Kept local rather than imported so this script never needs D
    on sys.path (D is invoked, never imported)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def render_config(
    example_cfg: Path,
    *,
    lmp_bin: Path,
    teacher_python: str,
    work_dir_abs: Path,
    structure_abs: Path,
    specorder: list[str],
    species: list[int],
    masses: list[float],
    target_ps: float,
) -> str:
    """config.yaml, overlaid onto D/config.example.yaml's own structure so
    every key D/ontheflydistill/config.py reads stays present (student/al_loop/
    remote sections carry through untouched; only what this run needs is
    overridden)."""
    with open(example_cfg) as fh:
        base = yaml.safe_load(fh)
    overrides = {
        "lmp_bin": str(lmp_bin),
        "python_bin": teacher_python,
        "work_dir": str(work_dir_abs / "run"),  # F21/N3: ABSOLUTE, not "./run"
        "system": {
            "init_structure": str(structure_abs),
            "species": species,
            "specorder": specorder,
            "masses": masses,
            "fixed_bottom_n": 0,   # no slab assumption for a generic bulk demo
            "force_pbc": True,
        },
        "teacher": {
            "type": "ase_calculator",
            "calculator": "omm_teacher:make_calc",
        },
        "al_loop": {"target_ps": target_ps},
    }
    merged = _deep_merge(copy.deepcopy(base), overrides)
    header = (
        "# generated by scripts/distill_bootstrap.py -- rerun distill_bootstrap.py "
        "to regenerate; edit directly for a production run.\n"
    )
    return header + yaml.safe_dump(merged, sort_keys=False, default_flow_style=False)


def render_run_distill_sh(*, teacher_python: str, repo: Path, work_dir_abs: Path) -> str:
    """run_distill.sh: the AC7 rerun unit. All three environment facts are
    load-bearing (F16/F17/F21-N3); see the module docstring for which silent
    failure each one prevents. Seeds `<work_dir>/dataset.extxyz` the same way
    `examples/ptwater_acid/README.md` documents (teacher_md.py -> merge_xyz)
    before handing off to D's own loop -- al_loop_local.sh itself REQUIRES
    that file to already exist and never builds it."""
    teacher_bin = str(Path(teacher_python).parent)
    repo_abs = str(repo.resolve())
    workdir_abs = str(work_dir_abs)
    al_work = f"{workdir_abs}/run"
    config_path = f"{workdir_abs}/config.yaml"
    return f"""#!/bin/sh
# generated by scripts/distill_bootstrap.py -- rerun this file to reproduce the run
set -eu

# F17: al_loop_local.sh:31-37 bootstraps its OWN config reads with bare
# `python` (seven times) before it ever resolves $PY -- that bare python must
# be able to `import yaml` and `from ontheflydistill import config`.
PATH="{teacher_bin}:$PATH"
export PATH

# F16: every AL-loop stage runs as `python -m ontheflydistill.<mod>` (needs
# {repo_abs} on the path); the teacher.calculator "omm_teacher:make_calc" is
# imported via importlib from a script whose own sys.path[0] is D/scripts, so
# the work dir must ALSO be on PYTHONPATH or omm_teacher is unimportable.
PYTHONPATH="{repo_abs}:{workdir_abs}${{PYTHONPATH:+:$PYTHONPATH}}"
export PYTHONPATH

# F21/N3: al_loop_local.sh:27-29 cd's into D BEFORE reading anything; without
# this, it silently reads (missing) $D/config.yaml, falls back to the built-in
# Pt-water defaults, and every artifact above is ignored.
ONTHEFLY_CONFIG="{config_path}"
export ONTHEFLY_CONFIG

WORK="{al_work}"
mkdir -p "$WORK"

# Seed the initial AL-pool dataset -- exactly the two commands
# examples/ptwater_acid/README.md documents, automated and made idempotent
# (skipped on rerun once dataset.extxyz exists). al_loop_local.sh itself
# never builds this file; it exits 1 if it is missing.
if [ ! -s "$WORK/dataset.extxyz" ]; then
    python "{repo_abs}/scripts/teacher_md.py" "$WORK/teacher_md.extxyz" {TEACHER_MD_STEPS} {TEACHER_MD_SAVE_EVERY}
    python -m ontheflydistill.merge_xyz "$WORK/dataset.extxyz" "$WORK/teacher_md.extxyz"
fi

exec "{repo_abs}/scripts/al_loop_local.sh"
"""


def main(argv=None) -> int:
    args = parse_args(argv)
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)

    loop, example_cfg = check_repo(args.repo, work)
    specorder, species, masses = read_structure(args.structure.resolve(), work)
    spec = resolve_teacher(args.teacher, work)

    _log(work, "step 3/4: write omm_teacher.py")
    (work / "omm_teacher.py").write_text(render_omm_teacher(spec))

    _log(work, "step 3/4: write config.yaml")
    config_text = render_config(
        example_cfg,
        lmp_bin=args.lmp_bin,
        teacher_python=spec["python"],
        work_dir_abs=work,
        structure_abs=args.structure.resolve(),
        specorder=specorder,
        species=species,
        masses=masses,
        target_ps=args.target_ps,
    )
    (work / "config.yaml").write_text(config_text)

    _log(work, "step 4/4: write run_distill.sh")
    run_sh = render_run_distill_sh(teacher_python=spec["python"], repo=args.repo, work_dir_abs=work)
    run_path = work / "run_distill.sh"
    run_path.write_text(run_sh)
    run_path.chmod(0o755)

    _log(work, f"done -- cd {work} && bash run_distill.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
