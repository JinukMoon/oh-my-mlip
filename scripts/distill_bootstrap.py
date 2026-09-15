#!/usr/bin/env python3
"""distill_bootstrap.py <--teacher V> <--structure FILE> <--work DIR>
[--repo PATH] [--target-ps F] [--lmp-bin PATH] -- render everything a run of
onthefly-distill (`D`, GPL-2.0, orchestrated -- never absorbed) needs to
distill one oh-my-mlip teacher into a CPU LAMMPS NN-MTP student (P4.2).

Deterministic, write-then-execute (Principle 2): every artifact below is
written to the ABSOLUTE work dir before anything computes. Each step is
printed to stdout and appended to `<work>/bootstrap.log`.

  1. locate D                       -- `--repo`, else `$ONTHEFLY_REPO` (one of
                                        them is required). No clone,
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

`--acceptance` (recipes/distill.md §2) turns the render into an approvable
proposal. Every target is explicit -- there is no default threshold, target
or budget to fall back on -- and two more files are written:

  5. `acceptance.json`               -- mode (fixture|production), held-out
                                        energy/force MAE thresholds, the
                                        stability target, the bounded budget
                                        (`max_iter`, `no_progress_limit`,
                                        `seed`), the split (AL pool vs
                                        held-out), provenance (hub + engine
                                        git heads, teacher spec, structure
                                        sha256, generated-file sha256s) and
                                        the exact verify command. This is the
                                        contract `scripts/distill_verify.py`
                                        judges against.
  6. `PLAN.md`                       -- the same proposal for a human.

In acceptance mode `config.yaml` also carries the approved `al_loop.*`
values and `run_distill.sh` produces the held-out set BEFORE the loop: a
separate teacher MD trajectory (`teacher_md.py` with a seed that differs
from the pool's) written to `<work>/heldout/heldout.extxyz` -- outside the
engine's `work_dir`, so `train_student.py`'s `al_iter*_labeled.extxyz`
glob can never pick it up -- or, with `--heldout-file`, the user's own
frames relabelled by the same teacher (`label.py`). The trainer's own
random validation split of the AL pool is NEVER the held-out set. Because
`teacher_md.py` saves the init structure as frame 0 before its first MD step
for ANY seed, and the pool's trajectory starts with that same frame,
`run_distill.sh` drops frame 0 from the generated held-out trajectory (a
user-supplied set containing the init structure is refused here).

`--wallclock-max-h H` (acceptance only) makes `run_distill.sh` re-run itself
under GNU coreutils `timeout`, so ONE deadline bounds the whole generated
run -- pool seeding, held-out generation and the AL loop -- in one owned
process group. A start record (`<work>/.distill/attempt.json`) and an
outcome marker (`<work>/.distill/wallclock.json`) carry the same attempt id
and the sha256 of the script that ran; `distill_verify.py` treats
`exhausted` as `unmet(budget:wallclock)` and a missing, unreadable or
unbound marker as `failed(wallclock_evidence)`. Refused on a host without
GNU timeout rather than recorded unenforced. The marker attributes its
return code (`attribution`): rc 124 is expiry; rc 137 at or after the limit
is `kill_at_or_after_limit` -- timeout's KILL after the grace OR an external
SIGKILL that landed after the deadline, which rc + elapsed cannot tell
apart, so it is recorded as ambiguous and still counts as exhaustion; rc
137 before the limit is a SIGKILL of the engine from an unknown source (OOM
killer, `kill -9`, ...), recorded as `engine_exit`, never as exhaustion.
Any `engine_exit` is `failed(supervisor_exit)` at verify time, whatever
`.al_status` says.

The structure is staged: whatever ase-readable file `--structure` names is
re-written once as `<work>/structure_init.vasp` (the source is never
touched) because D's `teacher_md.py` reads its init structure with
`format="vasp"` hard-coded; `config.yaml` points at the staged copy and both
files' sha256 are recorded. The VASP format carries Z, positions, cell, a
fully periodic cell and FixAtoms (selective dynamics) -- nothing else -- so
a structure whose identity it would silently change is refused before
anything is written: partial periodicity (the engine is fully periodic:
teacher MD and labels run with `system.force_pbc`, and the student MD is
LAMMPS `boundary p p f` by the engine's own design), nonzero initial
magnetic moments or charges, or a constraint other than FixAtoms. The
staged copy is read back and compared to the source (numbers, positions,
cell, fixed indices) before it is used. Every path rendered into
`run_distill.sh` is shell-quoted (`shlex.quote`), so a work, engine,
structure or interpreter path with spaces, quotes or `$(...)` is used
literally; host-derived text inside a `#` comment goes through `_c`
(single-line JSON), so a newline in a filename cannot end the comment.

Output ownership: `--work` must be new or an empty directory, checked
before the first write; anything already inside (a previous bootstrap, a
`run/` tree, the source structure parked as `structure_init.vasp`, a
symlink or hardlink at an output name) is refused with nothing written.
Every output is created with O_CREAT|O_EXCL|O_NOFOLLOW (`_write_new`), so
no existing file is overwritten and no symlink is followed, and this script
deletes nothing: a staged copy that fails its read-back is retained and
reported rather than unlinked. Source and held-out files are only ever read.
`bootstrap.log` is created the same way, once, right after the emptiness
check, and the stream is held open for the whole bootstrap (`BootstrapLog`):
later lines go to that retained descriptor, never to a reopen of the
pathname, so an object planted at the name between two log lines is never
written to or opened (a hardlink is not appended to, a FIFO is not opened).
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oh_my_mlip import resolve  # noqa: E402

try:
    import numpy as np
    import yaml
    from ase.constraints import FixAtoms
    from ase.data import atomic_masses
    from ase.io import read as ase_read, write as ase_write
except ImportError as exc:  # pragma: no cover - environment hint
    raise ImportError(
        "distill_bootstrap.py needs ase + pyyaml on the interpreter that runs "
        "it -- use any conda env carrying both (e.g. a model env or your "
        "analysis env), not the bare system python."
    ) from exc

MAX_SPECIES = 4  # pair_nnmtp v1: `int species_Z[4]` (F18) -- v2 is unbounded
                 # but is documented as "not used by the loop"; bound to v1.

DEFAULT_REPO = os.environ.get("ONTHEFLY_REPO")  # no built-in default: pass --repo or set it
DEFAULT_LMP_BIN = str(
    Path.home() / ".cache" / "oh-my-mlip" / "lammps" / "build" / "lmp"
)

# Kept tiny so the AC2 demo finishes in minutes, not hours -- a real
# production run should raise these by editing the emitted run_distill.sh
# directly (it is a plain, rerunnable shell script, not a hidden default).
TEACHER_MD_STEPS = 200
TEACHER_MD_SAVE_EVERY = 10

# D/scripts/teacher_md.py seeds its Langevin MD from $TEACHER_SEED (default
# 42). The AL pool is seeded with that default; the held-out trajectory MUST
# use a different seed or it is the same trajectory under another name.
POOL_TEACHER_SEED = 42
DEFAULT_HELDOUT_SEED = 4242
DEFAULT_HELDOUT_STEPS = 200
DEFAULT_HELDOUT_SAVE_EVERY = 10

# The fixture verdict is about the loop, never the student: bound its rounds
# so a fixture cannot quietly become an unbounded production run.
FIXTURE_MAX_ITER = 3

ACCEPTANCE_MODES = ("fixture", "production")
# --wallclock-max-h enforcement: run_distill.sh (hub-owned) re-runs ITSELF as
# a child of GNU coreutils `timeout`, so ONE deadline covers the whole
# generated run -- pool seeding, held-out generation and D's AL loop -- and on
# expiry timeout signals that whole process group (the sh, teacher_md.py,
# al_loop_local.sh, its python trainer, its lmp) with TERM and, WALLCLOCK_GRACE_S
# later, KILL. The outcome lands in a hub-owned marker bound to the attempt
# (attempt id + start record written by the same supervisor, sha256 of the
# script that ran); the engine's own run/.al_status is never written by the hub.
WALLCLOCK_GRACE_S = 60
WALLCLOCK_MARKER = ".distill/wallclock.json"
WALLCLOCK_ATTEMPT = ".distill/attempt.json"
WALLCLOCK_SCHEMA = "oh-my-mlip.distill.wallclock/3"  # /3: carries `attribution`
ATTEMPT_SCHEMA = "oh-my-mlip.distill.attempt/1"
WALLCLOCK_EXHAUSTED_RCS = (124, 137)  # GNU timeout: expired (TERM) / had to KILL
# rc 137 is also what a SIGKILLed engine returns (OOM killer, `kill -9`, ...)
# at any time. rc + elapsed can only place a 137 before or at/after the
# limit; they cannot say WHO sent the KILL after the deadline (timeout after
# its grace, or something external). The marker therefore records an
# attribution token, never a proven source. Both the supervisor and
# distill_verify.py apply this table; distill_verify.py recomputes the token
# from rc + elapsed and rejects a marker whose token disagrees.
WALLCLOCK_ATTRIBUTION = {
    "timeout_term": "rc 124: GNU timeout's expiry (TERM sufficed) -- budget exhausted",
    "kill_at_or_after_limit": "rc 137 at/after the limit: timeout's KILL after the grace OR an external SIGKILL "
                              "that landed after the deadline -- rc + elapsed cannot tell which (ambiguous, "
                              "recorded as such); the budget had expired either way, so exhausted",
    "sigkill_before_limit": "rc 137 before the limit: SIGKILL of the engine, source unknown (OOM killer, kill -9, "
                            "...) -- not expiry",
    "clean": "rc 0: clean exit within the bound",
    "exit_nonzero": "any other rc: the engine/loop exited non-zero or by another signal (128+N) -- not expiry",
}
WALLCLOCK_RC_RULE = ("rc 124 => exhausted (timeout_term); rc 137 => exhausted when elapsed_s >= limit_s "
                     "(kill_at_or_after_limit: timeout's KILL or an external SIGKILL, unknown which), else "
                     "engine_exit (sigkill_before_limit: source unknown); rc 0 => within (clean); other => "
                     "engine_exit (exit_nonzero). engine_exit is failed(supervisor_exit) at verify time, "
                     "whatever .al_status says")
WALLCLOCK_COVERS = "run_distill.sh end to end: pool seeding, held-out generation, AL loop"
STAGED_STRUCTURE = "structure_init.vasp"  # D/scripts/teacher_md.py reads with format="vasp" hard-coded
# Everything this script creates under --work. It creates them only in a work
# directory that is new or empty (checked before the first write), each with
# O_CREAT|O_EXCL|O_NOFOLLOW, and it never removes anything: a source structure
# or held-out file that resolves to one of these names, a symlink or hardlink
# parked there, or a previous bootstrap's output is refused, never overwritten,
# never regenerated.
BOOTSTRAP_OUTPUTS = ("bootstrap.log", STAGED_STRUCTURE, "omm_teacher.py", "config.yaml", "run_distill.sh",
                    "acceptance.json", "PLAN.md")
WORK_DIR_POLICY = ("--work must be new or empty at bootstrap; every output is created exclusively (O_EXCL, no "
                   "symlink following) and nothing preexisting is overwritten, regenerated or deleted -- a "
                   "non-empty work directory is refused before any write; bootstrap.log is created once, "
                   "exclusively, and its stream is held open for the whole bootstrap (never reopened by name)")
STAGING_ATOL = 1e-8  # read-back tolerance for the staged copy (the writer prints 16 decimals)

# The engine's physical contract, read from its source (never imported):
#   - teacher MD: scripts/teacher_md.py reads the init structure with
#     format="vasp" (always fully periodic) and keeps a FixAtoms constraint;
#   - labels: ontheflydistill/teachers/ase_calculator.py forces pbc=[T,T,T]
#     before every teacher call when system.force_pbc is true (the config
#     this script renders keeps it true, as D's own default does);
#   - student MD: ontheflydistill/student_md_lammps.py run() writes
#     `boundary p p f` with reflective z walls and freezes the FixAtoms ids.
# So the accuracy gate (held-out, periodic labels) and the stability gate
# (student MD, p p f) use different z boundaries BY THE ENGINE'S DESIGN; the
# lmp witness follows the student MD (the deployment boundary) and says so.
ENGINE_LABELING_PBC = "[T, T, T] -- system.force_pbc (ontheflydistill/teachers/ase_calculator.py label())"
ENGINE_STUDENT_MD_BOUNDARY = "p p f -- ontheflydistill/student_md_lammps.py run() (reflective z walls, FixAtoms frozen)"
ENGINE_BOUNDARY_NOTE = ("the held-out accuracy is measured on periodic teacher labels while the student MD and the lmp "
                        "witness run `boundary p p f`; that split is the engine's own design and is disclosed, "
                        "not reconciled. Input structures must be fully periodic (the VASP staging cannot carry "
                        "anything else); a slab needs its vacuum along z inside the periodic cell")

# Engine files whose sha256 the acceptance proposal records, so that
# distill_verify.py can tell an ordinary post-approval edit of the invoked
# engine (a changed trainer, MD driver or loop) from the approved one. Only
# the ones present in the checkout are recorded; none is ever copied.
ENGINE_PROVENANCE_FILES = (
    "scripts/al_loop_local.sh", "scripts/teacher_md.py", "scripts/label.py", "scripts/can_relabel.py",
    "ontheflydistill/__init__.py", "ontheflydistill/config.py", "ontheflydistill/common.py",
    "ontheflydistill/train_student.py", "ontheflydistill/student_md_lammps.py", "ontheflydistill/merge_xyz.py",
)
ENGINE_PROVENANCE_GLOBS = ("ontheflydistill/teachers/*.py",)

# Per-frame identity shared with distill_verify.frame_fingerprints: Z + positions
# rounded to 1e-4 A. Kept in sync by hand (D is invoked, the two hub scripts
# stay import-free of each other).
FINGERPRINT_DECIMALS = 4


def _find_gnu_timeout() -> str | None:
    """Path of a GNU coreutils `timeout` (the only one whose -k and
    process-group semantics the supervision relies on), else None."""
    found = shutil.which("timeout")
    if not found:
        return None
    try:
        out = subprocess.run([found, "--version"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return found if "GNU coreutils" in out else None


def _q(path) -> str:
    """Shell-quote one path/argument for run_distill.sh. Every operator- or
    host-derived string that lands in the script goes through here, so a
    work, engine, structure or interpreter path containing spaces, quotes
    or `$(...)` is used literally -- never expanded, never a redirect."""
    return shlex.quote(str(path))


def _c(text) -> str:
    """Render host/operator-derived text for a `#` comment or a markdown line:
    one line, JSON-quoted, every control character escaped. Shell comments
    end at a newline whatever the quoting, so a filename carrying one would
    otherwise turn the rest of itself into the next command."""
    return json.dumps(str(text))


def wallclock_plan(hours: float, timeout_bin: str, work: Path) -> dict:
    return {"limit_h": hours, "limit_s": max(1, int(round(hours * 3600))), "grace_s": WALLCLOCK_GRACE_S,
            "timeout_bin": timeout_bin, "marker": str(work / WALLCLOCK_MARKER),
            "attempt": str(work / WALLCLOCK_ATTEMPT), "covers": WALLCLOCK_COVERS,
            "marker_schema": WALLCLOCK_SCHEMA, "attempt_schema": ATTEMPT_SCHEMA, "rc_rule": WALLCLOCK_RC_RULE,
            "attribution": dict(WALLCLOCK_ATTRIBUTION),
            "by": "run_distill.sh re-runs itself under GNU `timeout -k <grace_s> <limit_s>` so one deadline covers "
                  "pool seeding, held-out generation and the AL loop (TERM to the whole process group, then KILL); "
                  "artifacts under run/ are left in place; the start record and the outcome marker carry the same "
                  "attempt id and the sha256 of the script that ran; distill_verify.py reports "
                  "unmet(budget:wallclock) on exhaustion and failed(wallclock_evidence) when the marker is missing, "
                  "unreadable or not bound to this attempt -- never success",
            "process_group_note": "behavioural change vs the unbounded launch: GNU timeout puts the supervised run "
                                  "in its own process group (pgid = timeout's pid), so an interactive Ctrl-C at the "
                                  "terminal does not reach it. Stop early with `kill -TERM -- -<pgid>` (the whole "
                                  "run goes with it); the marker then records engine_exit, never exhausted."}


def frame_fingerprint(atoms) -> str:
    pos = np.round(atoms.get_positions(), FINGERPRINT_DECIMALS).astype(np.float64)
    return hashlib.sha256(atoms.get_atomic_numbers().astype(np.int64).tobytes() + pos.tobytes()).hexdigest()


def engine_file_hashes(repo: Path) -> dict[str, str]:
    files = [repo / rel for rel in ENGINE_PROVENANCE_FILES]
    for pattern in ENGINE_PROVENANCE_GLOBS:
        files.extend(sorted(repo.glob(pattern)))
    return {str(p.relative_to(repo)): _sha256(p) for p in files if p.is_file()}


def _create_exclusive(path: Path, mode: int, *, append: bool = False) -> int:
    """os.open(O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW): the one way this script
    brings a file into being. O_EXCL means any existing object at the name --
    a regular file, a hardlink or symlink to something external, a dangling
    symlink, a FIFO -- fails the open with EEXIST before anything is written
    or, for a FIFO, opened; O_NOFOLLOW means a symlink is never followed.
    Refuses, nothing written, if the name exists. No fallback to a
    non-exclusive open: that would reopen exactly the hole this closes."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | (os.O_APPEND if append else 0)
    try:
        return os.open(str(path), flags, mode)
    except FileExistsError as exc:
        raise SystemExit(f"{path} already exists; bootstrap creates its outputs exclusively and never overwrites "
                         "or regenerates one -- pick a new --work or remove it yourself") from exc


class BootstrapLog:
    """`<work>/bootstrap.log`, created exclusively exactly once (right after
    refuse_unowned_work passed, so the directory is empty by construction) and
    then written through the retained descriptor for the rest of the
    bootstrap. The pathname is never reopened: whether the first call creates
    or a later call appends is decided by this object's existence, not by
    probing the filesystem, and an object planted at the name after creation
    is neither written to nor opened. The stream is closed by `close()` /
    the context manager on every exit, including a refusal; the file itself
    is retained (nothing under --work is ever deleted)."""

    def __init__(self, work: Path):
        self.path = work / "bootstrap.log"
        self._stream = os.fdopen(_create_exclusive(self.path, 0o644, append=True), "a", encoding="utf-8")

    def write(self, line: str) -> None:
        self._stream.write(line + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "BootstrapLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _log(log: BootstrapLog | None, msg: str) -> None:
    line = f"[distill_bootstrap] {msg}"
    print(line, flush=True)
    if log is not None:
        log.write(line)


def _write_new(path: Path, text: str, mode: int = 0o644) -> None:
    """Create `path` exclusively (see _create_exclusive) and write `text`:
    a previous bootstrap's output, the user's own file, a hardlink or a symlink
    parked at that name is never overwritten, followed or replaced."""
    with os.fdopen(_create_exclusive(path, mode), "w", encoding="utf-8") as fh:
        fh.write(text)


def refuse_unowned_work(work: Path, *, structure_abs: Path, heldout_abs: Path | None) -> None:
    """Fail closed before the first write: --work must not exist, or exist as an
    empty directory. Anything already inside -- a previous bootstrap's outputs,
    a run/ tree, the user's source structure staged under this very name, a
    symlink or hardlink at an output name -- would be overwritten or regenerated
    by the writes below; the hub refuses instead (no rollback, no implicit
    regeneration). Nothing under --work is ever deleted by this script."""
    if not work.exists():
        return
    if not work.is_dir():
        raise SystemExit(f"--work {work} exists and is not a directory; nothing was written")
    entries = sorted(os.listdir(work))
    if not entries:
        return
    inside = []
    for label, p in (("--structure", structure_abs), ("--heldout-file", heldout_abs)):
        if p is not None and (p == work or work in p.parents):
            inside.append(f"{label} {p} lives inside it")
    shown = ", ".join(entries[:8]) + (f", ... ({len(entries)} entries)" if len(entries) > 8 else "")
    raise SystemExit(
        f"--work {work} is not empty ({shown}); bootstrap writes {', '.join(BOOTSTRAP_OUTPUTS)} only into a new or "
        "empty work directory and never overwrites, regenerates or deletes what is there"
        + ("; " + "; ".join(inside) if inside else "")
        + ". Nothing was written. Pick a new --work, or empty this one yourself."
    )


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--teacher", required=True, help="oh-my-mlip model or version, e.g. MACE-MPA-0")
    ap.add_argument("--structure", required=True, type=Path, help="anything ase.io.read handles, <=4 species")
    ap.add_argument("--work", required=True, type=Path, help="work dir (created; rendered ABSOLUTE)")
    ap.add_argument("--repo", type=Path, default=Path(DEFAULT_REPO) if DEFAULT_REPO else None,
                    help="onthefly-distill checkout (D); required unless $ONTHEFLY_REPO is set")
    ap.add_argument("--target-ps", type=float, default=None,
                    help="al_loop.target_ps; defaults to 1.0 for a quick demo, REQUIRED with --acceptance")
    ap.add_argument("--lmp-bin", type=Path, default=Path(DEFAULT_LMP_BIN), help="LAMMPS binary with pair_style nnmtp")

    acc = ap.add_argument_group("acceptance proposal (recipes/distill.md §2; every target explicit)")
    acc.add_argument("--acceptance", action="store_true",
                     help="also render acceptance.json + PLAN.md and the held-out set; requires --mode, "
                          "--energy-mae-max, --force-mae-max, --target-ps, --max-iter, --no-progress-limit")
    acc.add_argument("--mode", choices=ACCEPTANCE_MODES, default=None,
                     help="fixture (loop exercise, max_iter <= %d) or production (targets gate the verdict)" % FIXTURE_MAX_ITER)
    acc.add_argument("--energy-mae-max", type=float, default=None, help="held-out energy MAE threshold, meV/atom")
    acc.add_argument("--force-mae-max", type=float, default=None, help="held-out force MAE threshold, meV/A")
    acc.add_argument("--max-iter", type=int, default=None, help="al_loop.max_iter (hard backstop on AL rounds)")
    acc.add_argument("--no-progress-limit", type=int, default=None, help="al_loop.no_progress_limit")
    acc.add_argument("--seed", type=int, default=None, help="al_loop.seed (student MD seed; default 42)")
    acc.add_argument("--wallclock-max-h", type=float, default=None,
                     help="wall-clock bound on the WHOLE run_distill.sh (pool seeding, held-out generation, AL "
                          "loop), ENFORCED by run_distill.sh re-running itself under GNU coreutils `timeout` "
                          "(TERM to the whole process group, KILL %d s later); exhaustion is "
                          "unmet(budget:wallclock). Refused on a host without GNU timeout." % WALLCLOCK_GRACE_S)
    acc.add_argument("--pool-steps", type=int, default=None,
                     help="teacher MD steps seeding the AL pool (default %d)" % TEACHER_MD_STEPS)
    acc.add_argument("--pool-save-every", type=int, default=None,
                     help="save interval of the pool MD (default %d)" % TEACHER_MD_SAVE_EVERY)
    acc.add_argument("--heldout-file", type=Path, default=None,
                     help="user frames (ase-readable) to relabel with the teacher as the held-out set; "
                          "default: a separate teacher MD trajectory with --heldout-seed")
    acc.add_argument("--heldout-seed", type=int, default=None,
                     help="TEACHER_SEED of the generated held-out trajectory (default %d; must differ from the "
                          "pool's %d)" % (DEFAULT_HELDOUT_SEED, POOL_TEACHER_SEED))
    acc.add_argument("--heldout-steps", type=int, default=None,
                     help="steps of the generated held-out MD (default %d)" % DEFAULT_HELDOUT_STEPS)
    acc.add_argument("--heldout-save-every", type=int, default=None,
                     help="save interval of the held-out MD (default %d)" % DEFAULT_HELDOUT_SAVE_EVERY)
    acc.add_argument("--manifest-sha256", default=os.environ.get("OMM_MANIFEST_SHA256"),
                     help="snapshot manifest aggregate to record in the provenance (or $OMM_MANIFEST_SHA256)")
    args = ap.parse_args(argv)

    # Every flag of the acceptance group defaults to None so that "typed by the
    # operator" is distinguishable from "absent": outside --acceptance none of
    # them has any effect, and a typed-but-ignored flag would be a silent lie.
    acceptance_defaults = {"seed": 42, "pool_steps": TEACHER_MD_STEPS, "pool_save_every": TEACHER_MD_SAVE_EVERY,
                           "heldout_seed": DEFAULT_HELDOUT_SEED, "heldout_steps": DEFAULT_HELDOUT_STEPS,
                           "heldout_save_every": DEFAULT_HELDOUT_SAVE_EVERY}
    if not args.acceptance:
        typed = [f"--{attr.replace('_', '-')}" for attr in (
            "mode", "energy_mae_max", "force_mae_max", "max_iter", "no_progress_limit", "wallclock_max_h",
            "heldout_file", *acceptance_defaults) if getattr(args, attr) is not None]
        if typed:
            ap.error(", ".join(typed) + " belong(s) to the acceptance proposal and would be ignored without "
                     "--acceptance; add --acceptance (with every target) or drop the flag(s)")
    for attr, default in acceptance_defaults.items():
        if getattr(args, attr) is None:
            setattr(args, attr, default)

    if args.acceptance:
        missing = [flag for flag, val in (
            ("--mode", args.mode), ("--energy-mae-max", args.energy_mae_max),
            ("--force-mae-max", args.force_mae_max), ("--target-ps", args.target_ps),
            ("--max-iter", args.max_iter), ("--no-progress-limit", args.no_progress_limit),
        ) if val is None]
        if missing:
            ap.error("--acceptance needs every target stated explicitly; missing " + ", ".join(missing))
        for flag, val in (("--energy-mae-max", args.energy_mae_max), ("--force-mae-max", args.force_mae_max),
                          ("--target-ps", args.target_ps)):
            if val <= 0:
                ap.error(f"{flag} must be > 0, got {val}")
        if args.max_iter < 1 or args.no_progress_limit < 1:
            ap.error("--max-iter and --no-progress-limit must be >= 1")
        if args.mode == "fixture" and args.max_iter > FIXTURE_MAX_ITER:
            ap.error(f"--mode fixture bounds --max-iter to <= {FIXTURE_MAX_ITER} (got {args.max_iter}); "
                     "a fixture exercises the loop, it is not a production run")
        if args.heldout_file is None and args.heldout_seed == POOL_TEACHER_SEED:
            ap.error(f"--heldout-seed must differ from the AL pool's teacher MD seed ({POOL_TEACHER_SEED}); "
                     "the same seed reproduces the pool trajectory, which is not a held-out set")
        if args.heldout_file is not None and not args.heldout_file.is_file():
            ap.error(f"--heldout-file {args.heldout_file} does not exist")
        for flag, val in (("--pool-steps", args.pool_steps), ("--pool-save-every", args.pool_save_every),
                          ("--heldout-steps", args.heldout_steps), ("--heldout-save-every", args.heldout_save_every)):
            if val < 1:
                ap.error(f"{flag} must be >= 1, got {val}")
        if args.heldout_file is None and args.heldout_steps < args.heldout_save_every:
            # teacher_md.py saves the init structure as frame 0 before the
            # first MD step; run_distill.sh drops that frame (the pool starts
            # with the same one). Fewer steps than the save interval leaves
            # nothing after the drop.
            ap.error(f"--heldout-steps ({args.heldout_steps}) must be >= --heldout-save-every "
                     f"({args.heldout_save_every}): the generated held-out MD saves only the pre-MD init frame "
                     "otherwise, and that frame is dropped because the AL pool starts with it too")
        if args.wallclock_max_h is not None:
            if args.wallclock_max_h <= 0:
                ap.error(f"--wallclock-max-h must be > 0, got {args.wallclock_max_h}")
            if _find_gnu_timeout() is None:
                # An accepted-but-unenforced bound would be a silent lie in the
                # plan; refuse the option instead of recording it.
                ap.error("--wallclock-max-h cannot be enforced on this host: no GNU coreutils `timeout` on PATH "
                         "(its -k / process-group semantics are what run_distill.sh relies on). Refusing to "
                         "record an unenforced limit -- drop the flag (operator-supervised, unbounded) or "
                         "install coreutils.")
    elif args.target_ps is None:
        args.target_ps = 1.0
    if args.repo is None:
        ap.error("--repo is required (or set $ONTHEFLY_REPO): an onthefly-distill checkout, "
                 "https://github.com/JinukMoon/onthefly-distill")
    return args


def check_repo(repo: Path, log: BootstrapLog | None) -> tuple[Path, Path]:
    """Step 2: locate D. No clone, no install -- just verify it is what it claims."""
    _log(log, f"step 2/4: locate onthefly-distill at {repo}")
    loop = repo / "scripts" / "al_loop_local.sh"
    example_cfg = repo / "config.example.yaml"
    teacher_md = repo / "scripts" / "teacher_md.py"
    missing = [p for p in (loop, example_cfg, teacher_md) if not p.is_file()]
    if missing:
        raise SystemExit(
            f"{repo} does not look like an onthefly-distill checkout -- missing "
            f"{[str(p) for p in missing]}. Pass --repo or set $ONTHEFLY_REPO."
        )
    _log(log, f"  found {loop}")
    return loop, example_cfg


def read_structure(path: Path):
    """Step 1: validate + derive system.{species,specorder,masses} from
    --structure.

    Runs BEFORE the work dir (and bootstrap.log) is created: a structure
    with too many distinct elements must be refused with nothing written to
    disk, not after a work dir and log file already exist (AGENTS.md §3D).

    First-seen element order is used for `specorder` (and therefore
    `species`/`masses`) so the mapping is deterministic for a given file.
    """
    if not path.is_file():
        raise SystemExit(f"--structure {path} does not exist")
    try:
        atoms = ase_read(str(path))
    except Exception as exc:
        raise SystemExit(f"--structure {path}: ase.io.read cannot read it ({type(exc).__name__}: {exc})") from exc
    if atoms.cell.volume < 1e-6:
        # teacher_md.py runs periodic MD (config force_pbc) and LAMMPS needs a
        # box; a cell-less xyz/molecule would fail far downstream.
        raise SystemExit(f"--structure {path} has no periodic cell (volume {atoms.cell.volume:g} A^3); "
                         "the teacher MD and the LAMMPS student need a box -- supply a cell.")
    unsupported = unsupported_identity(atoms)
    if unsupported:
        # The VASP staging would silently turn each of these into different
        # physics (pbc forced periodic, magmoms/charges dropped, constraint
        # dropped or degraded); refuse instead of converting.
        raise SystemExit(f"--structure {path} carries scientific identity the engine cannot take as-is: "
                         + "; ".join(unsupported) + ". Nothing was written. Supply a structure without it, or "
                         "stop -- this hub does not convert it silently.")
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
    return atoms, specorder, species, masses


def _fixed_indices(atoms) -> list[int]:
    return sorted({int(i) for c in atoms.constraints if isinstance(c, FixAtoms) for i in c.get_indices()})


def unsupported_identity(atoms) -> list[str]:
    """What the VASP staging (and the engine behind it) would silently change,
    each as one actionable sentence; empty when the structure round-trips.
    ase's POSCAR writer carries Z, positions, cell and FixAtoms/FixScaled
    flags and nothing else; the engine honours FixAtoms only
    (teacher_md.py keeps it, student_md_lammps.py freezes its ids)."""
    out = []
    pbc = [bool(b) for b in atoms.pbc]
    if not all(pbc):
        out.append(f"pbc={pbc} is not fully periodic -- the engine is: teacher_md.py reads VASP (always [T,T,T]), "
                   "labels run with system.force_pbc=[T,T,T], and the student MD is LAMMPS `boundary p p f` by "
                   "the engine's design; staging would silently make the structure periodic. Put the vacuum "
                   "inside a fully periodic cell and set pbc=[T,T,T] yourself")
    if np.any(atoms.get_initial_magnetic_moments() != 0):
        out.append("nonzero initial magnetic moments -- the VASP staging drops them, so the teacher would label a "
                   "non-magnetic structure; unsupported")
    if np.any(atoms.get_initial_charges() != 0):
        out.append("nonzero initial charges -- dropped by the VASP staging; unsupported")
    foreign = sorted({type(c).__name__ for c in atoms.constraints if not isinstance(c, FixAtoms)})
    if foreign:
        out.append(f"constraint(s) {foreign} -- only FixAtoms survives the VASP staging as selective dynamics AND "
                   "is what the engine honours (teacher_md.py keeps it, student_md_lammps.py freezes those ids); "
                   "anything else is dropped or degraded. Remove it or express it as FixAtoms")
    return out


def structure_identity(atoms) -> dict:
    """The identity the staged copy carries, recorded in the proposal."""
    fixed = _fixed_indices(atoms)
    return {
        "pbc": [bool(b) for b in atoms.pbc],
        "n_atoms": len(atoms),
        "fixed_atoms": fixed,
        "fixed_atoms_note": ("FixAtoms carried as selective dynamics; teacher_md.py keeps it and the student MD "
                             "freezes those ids" if fixed else
                             "no constraint; config system.fixed_bottom_n=0, so the engine freezes nothing"),
        "initial_momenta_present": bool(np.any(atoms.get_momenta() != 0)),
        "initial_momenta_note": "ignored either way: teacher_md.py draws fresh Maxwell-Boltzmann velocities "
                                "(TEACHER_SEED) before its MD",
        "initial_magmoms": "none (nonzero refused at bootstrap)",
        "initial_charges": "none (nonzero refused at bootstrap)",
        "carried_by_staging": "Z, positions (Cartesian, 16 decimals), cell, pbc [T,T,T], FixAtoms",
    }


def stage_structure(atoms, work: Path) -> Path:
    """`<work>/structure_init.vasp`: the init structure D actually reads.
    teacher_md.py opens `system.init_structure` with `format="vasp"`
    hard-coded, so an .xyz/.cif accepted by ase here would fail only at run
    time. The source file is never modified; this owned copy (atom order and
    cell as read, Cartesian coordinates) is what config.yaml points at, and
    both files' sha256 go into the provenance. The copy is read back the way
    teacher_md.py reads it and compared to the source -- numbers, positions,
    cell, periodicity, fixed indices -- so the staging demonstrably preserves
    the structure rather than being assumed to. The copy is created
    exclusively (O_EXCL|O_NOFOLLOW -- see _write_new): whatever sits at that
    name already, including the source itself, is never overwritten; a copy
    that fails the read-back is retained and reported, never unlinked."""
    staged = work / STAGED_STRUCTURE
    buf = io.StringIO()
    ase_write(buf, atoms, format="vasp", direct=False, sort=False)
    _write_new(staged, buf.getvalue())
    back = ase_read(str(staged), format="vasp")
    mismatch = []
    if list(back.get_atomic_numbers()) != list(atoms.get_atomic_numbers()):
        mismatch.append("atomic numbers/order")
    if len(back) == len(atoms) and not np.allclose(back.get_positions(), atoms.get_positions(), atol=STAGING_ATOL, rtol=0):
        mismatch.append("positions")
    if not np.allclose(np.array(back.get_cell()), np.array(atoms.get_cell()), atol=STAGING_ATOL, rtol=0):
        mismatch.append("cell")
    if not all(back.pbc):
        mismatch.append("pbc")
    if _fixed_indices(back) != _fixed_indices(atoms):
        mismatch.append("FixAtoms indices")
    if mismatch:
        # Retain the rejected copy (created by this invocation) rather than
        # unlink anything: no deletion under --work, and the next bootstrap
        # refuses the now non-empty directory until the operator clears it.
        raise SystemExit(f"staging the structure as {staged} did not reproduce it ({', '.join(mismatch)} differ "
                         "on read-back); refusing to hand the engine a different structure. The rejected copy "
                         "and bootstrap.log are retained (nothing is deleted, nothing else was written); "
                         "remove the work directory yourself before retrying")
    return staged


def resolve_teacher(teacher: str, log: BootstrapLog | None) -> dict:
    """Step 3 (part 1): resolve the teacher through the registry -- never by hand."""
    _log(log, f"step 3/4: resolve teacher {teacher!r} via oh_my_mlip.resolve()")
    spec = resolve(teacher)
    _log(log, f"  -> family={spec['model']} version={spec['version']} env={spec['env']} python={spec['python']}")
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
    al_loop: dict | None = None,
) -> str:
    """config.yaml, overlaid onto D/config.example.yaml's own structure so
    every key D/ontheflydistill/config.py reads stays present (student/al_loop/
    remote sections carry through untouched; only what this run needs is
    overridden). `al_loop` carries the approved budget in acceptance mode
    (max_iter / no_progress_limit / seed) so no manual post-edit is needed."""
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
        "al_loop": {"target_ps": target_ps, **(al_loop or {})},
    }
    merged = _deep_merge(copy.deepcopy(base), overrides)
    header = (
        "# generated by scripts/distill_bootstrap.py -- rerun distill_bootstrap.py "
        "to regenerate. Under --acceptance its sha256 is part of the approved contract: "
        "an edit after approval is reported by distill_verify.py as failed(provenance_drift).\n"
    )
    return header + yaml.safe_dump(merged, sort_keys=False, default_flow_style=False)


def heldout_path(work_dir_abs: Path) -> Path:
    """The held-out set lives OUTSIDE the engine's work_dir (`<work>/run`):
    train_student.py globs `al_iter*_labeled.extxyz` next to dataset.extxyz,
    and merge_xyz/label writes land there too, so a file under run/ can be
    absorbed into the pool. `<work>/heldout/` never is."""
    return work_dir_abs / "heldout" / "heldout.extxyz"


def render_heldout_block(*, repo_abs: str, work_dir_abs: Path, heldout_file: Path | None,
                         heldout_seed: int, heldout_steps: int, heldout_save_every: int) -> str:
    """The held-out step of run_distill.sh (acceptance mode only). Either a
    second teacher MD under a DIFFERENT seed (`teacher_md.py` reads
    $TEACHER_SEED), or the user's own frames relabelled by the same teacher
    through D's `label.py` so the labels sit on the teacher's PES. Idempotent:
    skipped once the file exists, like the pool seeding."""
    out = heldout_path(work_dir_abs)
    if heldout_file is not None:
        make = (f'    python {_q(f"{repo_abs}/scripts/label.py")} {_q(heldout_file.resolve())} "$HELDOUT"')
        # the path lands in a `#` comment: _c keeps it on one line (a newline
        # in the filename would otherwise end the comment and run the rest)
        how = f"relabel user frames {_c(heldout_file.resolve())} with the configured teacher (label.py)"
    else:
        raw = heldout_raw_path(work_dir_abs)
        make = (f'    TEACHER_SEED={heldout_seed} python {_q(f"{repo_abs}/scripts/teacher_md.py")} '
                f'{_q(raw)} {heldout_steps} {heldout_save_every}\n'
                f"{render_drop_frame0(raw=str(raw))}")
        how = (f"separate teacher MD, TEACHER_SEED={heldout_seed} (pool uses {POOL_TEACHER_SEED}), "
               f"{heldout_steps} steps, every {heldout_save_every}, pre-MD frame 0 dropped")
    return f"""
# Held-out set (recipes/distill.md §2 item 4): {how}.
# Written OUTSIDE $WORK so the trainer's al_iter*_labeled.extxyz glob and the
# pool merge can never absorb it; distill_verify.py asserts that and that no
# held-out frame coincides with a pool frame. Never the trainer's own split.
HELDOUT={_q(out)}
mkdir -p "$(dirname "$HELDOUT")"
if [ ! -s "$HELDOUT" ]; then
{make}
fi
"""


def heldout_raw_path(work_dir_abs: Path) -> Path:
    return work_dir_abs / "heldout" / "heldout_md_raw.extxyz"


def render_drop_frame0(*, raw: str) -> str:
    """The hub-owned step between D's teacher_md.py and the held-out file.
    teacher_md.py calls save_frame() once BEFORE dyn.run() for every
    TEACHER_SEED, so its frame 0 is the init structure byte-for-byte -- the
    same frame the AL pool's own teacher MD starts with. A held-out set that
    keeps it shares a frame with the pool; this drops it (nothing else) and
    refuses an empty remainder. Pure python, no ase: extxyz is
    `natoms / comment / natoms lines` per frame."""
    return textwrap.dedent(f'''\
        # teacher_md.py saves the init structure as frame 0 BEFORE the first MD
        # step (for any seed) and the pool's trajectory starts with that very
        # frame: drop it here, hub-owned, engine untouched. Empty remainder => exit 1.
        python - {_q(raw)} "$HELDOUT" <<'PY'
    import sys
    raw, out = sys.argv[1], sys.argv[2]
    lines = open(raw, encoding="utf-8").read().splitlines(keepends=True)
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        frames.append(lines[i:i + n + 2])
        i += n + 2
    if len(frames) < 2:
        sys.exit(f"[run_distill] held-out MD wrote {{len(frames)}} frame(s) to {{raw}}; dropping the pre-MD "
                 "frame 0 (shared with the pool) leaves nothing -- raise --heldout-steps (>= --heldout-save-every)")
    with open(out, "w", encoding="utf-8") as fh:
        fh.writelines(line for frame in frames[1:] for line in frame)
    print(f"[run_distill] held-out: dropped frame 0 (init structure, shared with the pool); "
          f"kept {{len(frames) - 1}} of {{len(frames)}} -> {{out}}", flush=True)
    PY''')


def render_run_distill_sh(*, teacher_python: str, repo: Path, work_dir_abs: Path,
                          pool_steps: int = TEACHER_MD_STEPS, pool_save_every: int = TEACHER_MD_SAVE_EVERY,
                          heldout_block: str = "", wallclock: dict | None = None) -> str:
    """run_distill.sh: the AC7 rerun unit. All three environment facts are
    load-bearing (F16/F17/F21-N3); see the module docstring for which silent
    failure each one prevents. Seeds `<work_dir>/dataset.extxyz` the same way
    `examples/ptwater_acid/README.md` documents (teacher_md.py -> merge_xyz)
    before handing off to D's own loop -- al_loop_local.sh itself REQUIRES
    that file to already exist and never builds it. `heldout_block` (acceptance
    mode) is spliced in between the pool seeding and the hand-off; `wallclock`
    (from wallclock_plan) wraps the WHOLE script in the supervised re-exec."""
    teacher_bin = str(Path(teacher_python).parent)
    repo_abs = str(repo.resolve())
    workdir_abs = str(work_dir_abs)
    al_work = f"{workdir_abs}/run"
    config_path = f"{workdir_abs}/config.yaml"
    supervisor = render_supervisor(work_dir_abs=work_dir_abs, wallclock=wallclock)
    launch = f'exec {_q(f"{repo_abs}/scripts/al_loop_local.sh")}\n'
    # Every host/operator-derived path below is shell-quoted (_q); no comment
    # line carries raw operator text (a `#` comment ends at a newline whatever
    # the quoting -- anything host-derived in a comment goes through _c).
    return f"""#!/bin/sh
# generated by scripts/distill_bootstrap.py -- rerun this file to reproduce the run
set -eu
{supervisor}
# F17: al_loop_local.sh:31-37 bootstraps its OWN config reads with bare
# `python` (seven times) before it ever resolves $PY -- that bare python must
# be able to `import yaml` and `from ontheflydistill import config`.
PATH={_q(teacher_bin)}:$PATH
export PATH

# F16: every AL-loop stage runs as `python -m ontheflydistill.<mod>` (needs
# the engine checkout on the path); the teacher.calculator "omm_teacher:make_calc"
# is imported via importlib from a script whose own sys.path[0] is D/scripts, so
# the work dir must ALSO be on PYTHONPATH or omm_teacher is unimportable.
PYTHONPATH={_q(repo_abs)}:{_q(workdir_abs)}${{PYTHONPATH:+:$PYTHONPATH}}
export PYTHONPATH

# F21/N3: al_loop_local.sh:27-29 cd's into D BEFORE reading anything; without
# this, it silently reads (missing) $D/config.yaml, falls back to the built-in
# Pt-water defaults, and every artifact above is ignored.
ONTHEFLY_CONFIG={_q(config_path)}
export ONTHEFLY_CONFIG

WORK={_q(al_work)}
mkdir -p "$WORK"

# Seed the initial AL-pool dataset -- exactly the two commands
# examples/ptwater_acid/README.md documents, automated and made idempotent
# (skipped on rerun once dataset.extxyz exists). al_loop_local.sh itself
# never builds this file; it exits 1 if it is missing.
if [ ! -s "$WORK/dataset.extxyz" ]; then
    python {_q(f"{repo_abs}/scripts/teacher_md.py")} "$WORK/teacher_md.extxyz" {pool_steps} {pool_save_every}
    python -m ontheflydistill.merge_xyz "$WORK/dataset.extxyz" "$WORK/teacher_md.extxyz"
fi
{heldout_block}
{launch}"""


def render_supervisor(*, work_dir_abs: Path, wallclock: dict | None) -> str:
    """With an approved wall-clock bound, the block right after `set -eu`:
    the first invocation (no $OMM_DISTILL_ATTEMPT) becomes the supervisor --
    it writes a start record, re-runs this same file as a child of GNU
    `timeout`, and writes the outcome marker; the child (attempt id in its
    environment) falls through to the body, so pool seeding, held-out
    generation and the exec'd loop all live under ONE deadline and one
    process group. Without a bound the block is empty. The engine's files
    are never touched; its .al_status is never written here."""
    if wallclock is None:
        return ""
    w = wallclock
    self_path = str(work_dir_abs / "run_distill.sh")
    # The script path goes into attempt.json as a JSON string: encoded here
    # (json.dumps escapes quotes/backslashes) and passed to printf as a %s
    # ARGUMENT, so neither printf nor the shell reinterprets it.
    self_json = _q(json.dumps(self_path))
    return f"""
# Wall-clock bound (approved budget), ONE deadline over this whole script:
# the supervisor below re-runs this file under GNU `timeout`, which after
# {w['limit_s']} s sends TERM to the whole process group (this sh, teacher_md.py,
# al_loop_local.sh, its python trainer, its lmp) and KILL {w['grace_s']} s later.
# Everything under the work dir stays on disk. Start record and outcome
# marker share an attempt id and the sha256 of the script that ran, so
# distill_verify.py can bind the outcome to THIS attempt; a marker left by
# an earlier attempt is removed before the new one starts.
# Outcome rule (the marker's `attribution` names the case; verify recomputes it):
#   rc 124                         -> exhausted, timeout_term
#   rc 137, elapsed >= {w['limit_s']} s -> exhausted, kill_at_or_after_limit: timeout's
#           KILL after the grace OR an external SIGKILL after the deadline --
#           rc + elapsed cannot tell which; recorded as ambiguous, not proven
#   rc 137, earlier                -> engine_exit, sigkill_before_limit (source
#           unknown: OOM killer, kill -9, ...) -- never exhaustion
#   rc 0                           -> within, clean
#   other                          -> engine_exit, exit_nonzero
# engine_exit is failed(supervisor_exit) at verify time whatever .al_status says.
# Note (behavioural change vs the unbounded launch): timeout puts the run in
# its own process group, so a terminal Ctrl-C does not reach it; stop early
# with `kill -TERM -- -<pgid>` (pgid = timeout's pid) -> engine_exit.
if [ -z "${{OMM_DISTILL_ATTEMPT:-}}" ]; then
    SELF={_q(self_path)}
    MARKER={_q(w['marker'])}
    ATTEMPT_FILE={_q(w['attempt'])}
    mkdir -p "$(dirname "$MARKER")"
    ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)-$$"
    START_S=$(date +%s)
    START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    SELF_SHA=$(sha256sum "$SELF" | cut -d ' ' -f 1)
    rm -f "$MARKER"
    printf '{{"schema": "{ATTEMPT_SCHEMA}", "attempt": "%s", "started_utc": "%s", "started_epoch_s": %d, "limit_s": {w['limit_s']}, "grace_s": {w['grace_s']}, "run_script": %s, "run_script_sha256": "%s", "supervisor_pid": %d, "covers": "{WALLCLOCK_COVERS}"}}\\n' \\
        "$ATTEMPT" "$START_UTC" "$START_S" {self_json} "$SELF_SHA" "$$" > "$ATTEMPT_FILE"
    echo "[run_distill] attempt $ATTEMPT: wall-clock bound {w['limit_s']} s over the whole run -> $ATTEMPT_FILE"
    set +e
    OMM_DISTILL_ATTEMPT="$ATTEMPT" {_q(w['timeout_bin'])} -k {w['grace_s']} {w['limit_s']} /bin/sh "$SELF"
    RC=$?
    set -e
    END_S=$(date +%s)
    ELAPSED_S=$(( END_S - START_S ))
    if [ "$RC" -eq 124 ]; then
        STATE=exhausted; WHY=timeout_term
    elif [ "$RC" -eq 137 ] && [ "$ELAPSED_S" -ge {w['limit_s']} ]; then
        STATE=exhausted; WHY=kill_at_or_after_limit
    elif [ "$RC" -eq 137 ]; then
        STATE=engine_exit; WHY=sigkill_before_limit
    elif [ "$RC" -eq 0 ]; then
        STATE=within; WHY=clean
    else
        STATE=engine_exit; WHY=exit_nonzero
    fi
    printf '{{"schema": "{WALLCLOCK_SCHEMA}", "attempt": "%s", "state": "%s", "attribution": "%s", "returncode": %d, "limit_h": {w['limit_h']}, "limit_s": {w['limit_s']}, "grace_s": {w['grace_s']}, "started_utc": "%s", "started_epoch_s": %d, "ended_utc": "%s", "ended_epoch_s": %d, "elapsed_s": %d, "run_script_sha256": "%s", "covers": "{WALLCLOCK_COVERS}"}}\\n' \\
        "$ATTEMPT" "$STATE" "$WHY" "$RC" "$START_UTC" "$START_S" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$END_S" "$ELAPSED_S" "$SELF_SHA" > "$MARKER"
    echo "[run_distill] wall-clock $STATE ($WHY): rc=$RC after $ELAPSED_S s of {w['limit_s']} s -> $MARKER"
    exit "$RC"
fi
# (child of the supervisor above: attempt $OMM_DISTILL_ATTEMPT, bounded)
"""


# ── acceptance proposal (recipes/distill.md §2) ─────────────────────────────
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(path: Path) -> dict:
    """{head, dirty} for a checkout, or {head: None} when it is not one. Never
    raises: provenance records what can be observed."""
    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip() != ""
        return {"head": head, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}


def verify_command(work: Path) -> str:
    return f"python3 scripts/distill_verify.py --work {_q(work)} --json"


def run_command(work: Path) -> str:
    """The launch line the plan prints. No `| tee`: a pipeline reports tee's
    status, not the run's, so the operator would read a masked exit code.
    Output goes to distill.log (follow with `tail -f`); the exit status is
    run_distill.sh's own (the supervisor's marker records it too)."""
    return f"cd {_q(work)} && sh run_distill.sh > distill.log 2>&1"


def render_acceptance(args: argparse.Namespace, *, work: Path, structure_abs: Path, staged: Path, spec: dict,
                      specorder: list[str], generated: dict[str, Path], wallclock: dict | None = None,
                      identity: dict | None = None) -> dict:
    """acceptance.json -- the approved contract distill_verify.py judges
    against. Nothing in here is a default: every threshold and budget came
    from a flag the operator typed (parse_args refuses otherwise)."""
    repo_abs = args.repo.resolve()
    heldout = heldout_path(work)
    if args.heldout_file is not None:
        split_heldout = {
            "source": "user_frames_relabelled_by_teacher",
            "input_file": str(args.heldout_file.resolve()),
            "input_sha256": _sha256(args.heldout_file.resolve()),
            "labeller": f"{repo_abs}/scripts/label.py",
            "seed": None,
        }
    else:
        split_heldout = {
            "source": "separate_teacher_md",
            "generator": f"{repo_abs}/scripts/teacher_md.py",
            "seed": args.heldout_seed,
            "steps": args.heldout_steps,
            "save_every": args.heldout_save_every,
            "raw_path": str(heldout_raw_path(work)),
            "drops_frame0": "teacher_md.py saves the init structure before its first MD step for any seed; "
                            "run_distill.sh drops that frame (the pool starts with the same one)",
        }
    split_heldout.update({
        "path": str(heldout),
        "outside_engine_work_dir": True,
        "independence_rule": (
            "never merged into run/dataset.extxyz nor written under run/; "
            "distill_verify.py asserts the path, the seed, that no held-out frame is the "
            "init structure, and an empty per-frame fingerprint intersection with the AL "
            "pool. The trainer's random validation split of the pool is NOT a held-out set."
        ),
    })
    return {
        "schema": "oh-my-mlip.distill.acceptance/1",
        "mode": args.mode,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "accuracy": {
            "energy_mae_max_mev_per_atom": args.energy_mae_max,
            "force_mae_max_mev_per_a": args.force_mae_max,
            "evaluated_on": "held-out set below, final student .pt through the engine's own validate()",
        },
        "stability": {"target_ps": args.target_ps,
                      "oracle": "run/.al_status == 'SUCCESS round<N> stable' (engine failure.json status stable_in_dump)"},
        "budget": {"max_iter": args.max_iter, "no_progress_limit": args.no_progress_limit, "seed": args.seed,
                   "wallclock_max_h": args.wallclock_max_h,
                   "wallclock_enforcement": wallclock,
                   "fixture_max_iter_bound": FIXTURE_MAX_ITER if args.mode == "fixture" else None},
        "split": {
            "pool": {"teacher_md_seed": POOL_TEACHER_SEED, "steps": args.pool_steps,
                     "save_every": args.pool_save_every, "path": str(work / "run" / "dataset.extxyz"),
                     "grows_by": "run/al_iter<K>_labeled.extxyz (teacher-relabelled pre-crash windows)"},
            "heldout": split_heldout,
        },
        "verdict_rule": {
            "production": "passed iff .al_status is SUCCESS (round 0 included) AND both held-out MAEs are "
                          "within thresholds AND the final .bin runs under lmp; stable-but-inaccurate => unmet; "
                          "STALLED / backstop / label_fail => unmet; FAILED / oneshot fallback => failed",
            "fixture": "passed iff at least one real relabel round (al_iter*_labeled non-empty AND a retrain "
                       "after it) occurred, the loop reached a terminal state, held-out metrics were computed "
                       "and the final .bin runs under lmp; thresholds are reported, not gating",
        },
        "provenance": {
            "hub": {"root": str(REPO_ROOT), **_git_head(REPO_ROOT), "manifest_sha256": args.manifest_sha256},
            "engine": {"repo": str(repo_abs), "licence": "GPL-2.0 (invoked, never absorbed)", **_git_head(repo_abs),
                       "files": engine_file_hashes(repo_abs)},
            "teacher": {"requested": args.teacher, "model": spec["model"], "version": spec["version"],
                        "env": spec["env"], "python": spec["python"]},
            "structure": {"path": str(structure_abs), "sha256": _sha256(structure_abs), "specorder": specorder,
                          "staged_path": str(staged), "staged_sha256": _sha256(staged),
                          "staged_note": f"{STAGED_STRUCTURE} is the copy D's teacher_md.py reads (format=vasp "
                                         "hard-coded); config.yaml system.init_structure points at it; the "
                                         "source file is never modified; the copy was read back and compared "
                                         "to the source (numbers, positions, cell, pbc, FixAtoms) at bootstrap",
                          "identity": identity or {},
                          "refused_at_bootstrap": "partial pbc, nonzero initial magmoms/charges, constraints "
                                                  "other than FixAtoms (the VASP staging would silently change "
                                                  "them; nothing is converted)"},
            "lmp_bin": str(args.lmp_bin),
            # the binary's identity (scripts/build_lammps_nnmtp.sh records no --ref
            # into the work dir; the hash pins WHICH build the witness will run)
            "lmp_bin_sha256": _sha256(args.lmp_bin) if args.lmp_bin.is_file() else None,
            "generated_files": {name: {"path": str(p), "sha256": _sha256(p)} for name, p in generated.items()},
            "work_dir_policy": WORK_DIR_POLICY,
            "identity_rule": "distill_verify.py recomputes every sha256 above (generated files, structure, "
                             "engine files); any mismatch is failed(provenance_drift). This detects ordinary "
                             "post-approval edits, not an owner rewriting acceptance.json itself.",
            "engine_facts": {
                "student_md_temperature_K": 300.0,
                "student_md_temperature_note": "fixed inside ontheflydistill/student_md_lammps.py run() "
                                               "(not a config key); a 'hotter' fixture is not available "
                                               "without an engine change",
                "labeling_pbc": ENGINE_LABELING_PBC,
                "student_md_boundary": ENGINE_STUDENT_MD_BOUNDARY,
                "boundary_note": ENGINE_BOUNDARY_NOTE,
                "input_requirement": "fully periodic cell, pbc [T,T,T]; FixAtoms is the only constraint the "
                                     "engine honours",
            },
        },
        "commands": {
            "run": run_command(work),
            "follow": f"tail -f {_q(work / 'distill.log')}",
            "verify": verify_command(work),
        },
        "witness_scope": {
            "accuracy_artifact": "run/model_scratch<N>.pt (torch checkpoint) -- held-out E/F MAE",
            "witness_artifact": "run/model_scratch<N>.bin (nnmtp export) -- lmp run 0: exit 0, finite energy, no ERROR",
            "note": "the witness establishes finiteness/stability of the exported binary, NOT accuracy "
                    "equivalence with the .pt; a mis-exported .bin can pass both gates",
            "witness_boundary": "p p f -- the same boundary as the engine's student MD, whose stability the "
                                "verdict cites; the held-out accuracy uses periodic labels (engine design, see "
                                "provenance.engine_facts.boundary_note)",
        },
    }


def render_plan_md(acc: dict, *, work: Path) -> str:
    a, s, b = acc["accuracy"], acc["stability"], acc["budget"]
    h = acc["split"]["heldout"]
    p = acc["provenance"]
    heldout_line = (f"separate teacher MD, seed {h['seed']} (pool seed {acc['split']['pool']['teacher_md_seed']}), "
                    f"{h['steps']} steps every {h['save_every']}" if h["source"] == "separate_teacher_md"
                    else f"user frames {_c(h['input_file'])} relabelled by the teacher")
    ident = p["structure"].get("identity") or {}
    ef = p["engine_facts"]
    w = b.get("wallclock_enforcement")
    wall = (f"{b['wallclock_max_h']} h = {w['limit_s']} s, enforced over the whole run ({w['covers']}) by "
            f"run_distill.sh re-running itself under GNU `timeout -k {w['grace_s']} {w['limit_s']}` (one process "
            f"group); exhaustion => unmet(budget:wallclock); start record `{w['attempt']}`, marker `{w['marker']}` "
            "bound to the attempt -- missing/unreadable/stale marker => failed(wallclock_evidence). "
            f"Outcome rule: {w['rc_rule']}. {w['process_group_note']}"
            if w else "none stated (unbounded; operator-supervised)")
    ws = acc["witness_scope"]
    return textwrap.dedent(f"""\
        # Distillation acceptance proposal — {acc['mode']}

        Generated by `scripts/distill_bootstrap.py --acceptance` at {acc['generated_utc']}.
        Machine copy: `{work / 'acceptance.json'}`. Approve this file before running.

        ## Inputs

        - Teacher: {p['teacher']['version']} (family {p['teacher']['model']}, env `{p['teacher']['env']}`)
        - Structure: {_c(p['structure']['path'])} (sha256 {p['structure']['sha256'][:12]}…, species {p['structure']['specorder']}); staged as {_c(p['structure']['staged_path'])} (sha256 {p['structure']['staged_sha256'][:12]}…) -- {p['structure']['staged_note']}
        - Structure identity carried: pbc {ident.get('pbc')}, {ident.get('n_atoms')} atoms, fixed atoms {ident.get('fixed_atoms')} ({ident.get('fixed_atoms_note')}); initial momenta present: {ident.get('initial_momenta_present')} ({ident.get('initial_momenta_note')}). Refused at bootstrap: {p['structure'].get('refused_at_bootstrap')}
        - Work directory ownership: {p['work_dir_policy']}
        - Physical contract (engine design, disclosed not reconciled): labels {ef['labeling_pbc']}; student MD `{ef['student_md_boundary']}`; input requirement: {ef['input_requirement']}. {ef['boundary_note']}
        - Engine: `{p['engine']['repo']}` @ {p['engine']['head'] or 'not a git checkout'} ({p['engine']['licence']})
        - Hub: {p['hub']['head'] or 'not a git checkout'}{' (dirty working tree)' if p['hub']['dirty'] else ''}
        - LAMMPS: `{p['lmp_bin']}`

        ## 1. Accuracy targets (held-out)

        - Energy MAE <= {a['energy_mae_max_mev_per_atom']} meV/atom
        - Force MAE  <= {a['force_mae_max_mev_per_a']} meV/A

        ## 2. Stability target

        - `al_loop.target_ps` = {s['target_ps']} ps of stable student MD (engine oracle: {s['oracle']})
        - Student MD temperature: {p['engine_facts']['student_md_temperature_K']} K ({p['engine_facts']['student_md_temperature_note']})

        ## 3. Budget

        - `al_loop.max_iter` = {b['max_iter']}, `al_loop.no_progress_limit` = {b['no_progress_limit']}, `al_loop.seed` = {b['seed']}
        - Wall-clock: {wall}

        ## 4. Split

        - AL pool: teacher MD seed {acc['split']['pool']['teacher_md_seed']}, {acc['split']['pool']['steps']} steps every {acc['split']['pool']['save_every']} -> `{acc['split']['pool']['path']}`, grown by {acc['split']['pool']['grows_by']}
        - Held-out: {heldout_line} -> `{h['path']}` (outside the engine work dir; {h['independence_rule']})

        ## Identity

        - {p['identity_rule']}

        ## 5. Commands (these are the commands that run)

        ```bash
        {acc['commands']['run']}
        {acc['commands']['verify']}
        ```

        Follow the run with `{acc['commands']['follow']}`. The launch line deliberately has no `| tee`:
        a pipeline would report tee's exit status, not the run's.

        ## Verdict rule

        - production: {acc['verdict_rule']['production']}
        - fixture: {acc['verdict_rule']['fixture']}
        - witness scope: accuracy is measured on {ws['accuracy_artifact']}; the lmp witness covers {ws['witness_artifact']}. {ws['note']}. Witness boundary: {ws['witness_boundary']}.
        - `--no-lmp-witness` at verify time records the witness as skipped and the verdict as `incomplete(lmp_witness_skipped)` -- never passed, never failed.
        - a supervised run whose marker says `engine_exit` (any nonzero rc that is not expiry: rc 1, an early rc 137, rc 143 ...) is `failed(supervisor_exit)` even when `.al_status` reads SUCCESS -- no metrics and no witness are computed for it.
        """)


def main(argv=None) -> int:
    args = parse_args(argv)
    work = args.work.resolve()
    structure_abs = args.structure.resolve()

    # Validate the structure BEFORE creating the work dir or its log: a
    # too-many-species structure must be refused with nothing written to
    # disk yet, not after a work dir and bootstrap.log already exist.
    atoms, specorder, species, masses = read_structure(structure_abs)
    identity = structure_identity(atoms)
    if args.acceptance and args.heldout_file is not None:
        # Read it here, before any output: an unreadable/empty held-out set is
        # refused with an actionable message, not a traceback and not a work
        # dir that would fail at label time.
        try:
            user_frames = ase_read(str(args.heldout_file), index=":")
        except Exception as exc:
            raise SystemExit(
                f"--heldout-file {args.heldout_file}: ase.io.read cannot parse it ({type(exc).__name__}: {exc}). "
                "Supply an ase-readable frame set (extxyz, traj, ...) or drop the flag to let run_distill.sh "
                "generate the held-out set from a separate teacher MD. Nothing was written."
            ) from exc
        if not user_frames:
            raise SystemExit(f"--heldout-file {args.heldout_file} carries no frames; nothing to hold out. "
                             "Nothing was written.")
        # The student only knows the structure's species (config system.species);
        # a held-out frame with an element outside that set cannot be evaluated.
        extra = sorted({s for fr in user_frames for s in fr.get_chemical_symbols()} - set(specorder))
        if extra:
            raise SystemExit(
                f"--heldout-file {args.heldout_file} contains species {extra} absent from "
                f"--structure ({specorder}); the student cannot be evaluated on them."
            )
        # The AL pool's teacher MD starts with the init structure itself
        # (teacher_md.py frame 0): a user held-out set containing that frame
        # would share it with the pool. Refuse now rather than fail at verify.
        init_fp = frame_fingerprint(atoms)
        shared = [i for i, fr in enumerate(user_frames) if frame_fingerprint(fr) == init_fp]
        if shared:
            raise SystemExit(
                f"--heldout-file {args.heldout_file}: frame(s) {shared} are the init structure "
                f"{structure_abs}, which the AL pool's teacher MD saves as its frame 0 -- a held-out set "
                "may not share a frame with the pool; remove them."
            )

    # Output ownership, before the first write: a non-empty --work (a previous
    # bootstrap, a run/ tree, the source parked at structure_init.vasp, a
    # symlink/hardlink at an output name) is refused; nothing is overwritten,
    # regenerated or deleted. Every write below is O_EXCL|O_NOFOLLOW.
    heldout_abs = args.heldout_file.resolve() if (args.acceptance and args.heldout_file is not None) else None
    refuse_unowned_work(work, structure_abs=structure_abs, heldout_abs=heldout_abs)
    work.mkdir(parents=True, exist_ok=True)
    # The log is created exclusively once, here, and every later line goes to
    # this retained stream -- the pathname is never reopened, so nothing
    # planted at bootstrap.log afterwards is written to or opened. Closed on
    # every exit (a refusal included); the file is retained, never unlinked.
    with BootstrapLog(work) as log:
        _log(log, f"step 1/4: read structure {structure_abs}")
        _log(log, f"  specorder={specorder} species(Z)={species} masses={masses}")
        staged = stage_structure(atoms, work)
        _log(log, f"  staged as {staged} (D's teacher_md.py reads format=vasp only; source untouched; read back and "
                   f"compared: pbc={identity['pbc']}, fixed atoms={identity['fixed_atoms']})")

        loop, example_cfg = check_repo(args.repo, log)
        spec = resolve_teacher(args.teacher, log)

        _log(log, "step 3/4: write omm_teacher.py")
        _write_new(work / "omm_teacher.py", render_omm_teacher(spec))

        al_loop = None
        if args.acceptance:
            al_loop = {"max_iter": args.max_iter, "no_progress_limit": args.no_progress_limit, "seed": args.seed}
        _log(log, "step 3/4: write config.yaml" + (f" (al_loop={al_loop})" if al_loop else ""))
        config_text = render_config(
            example_cfg,
            lmp_bin=args.lmp_bin,
            teacher_python=spec["python"],
            work_dir_abs=work,
            structure_abs=staged,
            specorder=specorder,
            species=species,
            masses=masses,
            target_ps=args.target_ps,
            al_loop=al_loop,
        )
        _write_new(work / "config.yaml", config_text)

        heldout_block = ""
        wallclock = None
        if args.acceptance and args.wallclock_max_h is not None:
            wallclock = wallclock_plan(args.wallclock_max_h, _find_gnu_timeout(), work)
        if args.acceptance:
            heldout_block = render_heldout_block(
                repo_abs=str(args.repo.resolve()), work_dir_abs=work, heldout_file=args.heldout_file,
                heldout_seed=args.heldout_seed, heldout_steps=args.heldout_steps,
                heldout_save_every=args.heldout_save_every,
            )
        _log(log, "step 4/4: write run_distill.sh")
        run_sh = render_run_distill_sh(
            teacher_python=spec["python"], repo=args.repo, work_dir_abs=work,
            pool_steps=args.pool_steps, pool_save_every=args.pool_save_every,
            heldout_block=heldout_block,
            wallclock=wallclock,
        )
        run_path = work / "run_distill.sh"
        _write_new(run_path, run_sh, mode=0o755)

        if args.acceptance:
            _log(log, f"acceptance ({args.mode}): write acceptance.json + PLAN.md")
            generated = {"config.yaml": work / "config.yaml", "run_distill.sh": run_path,
                         "omm_teacher.py": work / "omm_teacher.py", STAGED_STRUCTURE: staged}
            acc = render_acceptance(args, work=work, structure_abs=structure_abs, staged=staged, spec=spec,
                                    specorder=specorder, generated=generated, wallclock=wallclock, identity=identity)
            if wallclock:
                _log(log, f"  wall-clock: {wallclock['limit_s']} s enforced by {wallclock['timeout_bin']} "
                           f"(-k {wallclock['grace_s']}); marker {wallclock['marker']}")
            _write_new(work / "acceptance.json", json.dumps(acc, indent=2) + "\n")
            _write_new(work / "PLAN.md", render_plan_md(acc, work=work))
            _log(log, f"  thresholds: E <= {args.energy_mae_max} meV/atom, F <= {args.force_mae_max} meV/A; "
                       f"target_ps={args.target_ps}; max_iter={args.max_iter}; no_progress_limit={args.no_progress_limit}; "
                       f"seed={args.seed}")
            _log(log, f"  held-out: {acc['split']['heldout']['source']} -> {acc['split']['heldout']['path']}")
            _log(log, f"awaiting approval of {work / 'PLAN.md'} -- then: {acc['commands']['run']}")
            _log(log, f"verify with: {acc['commands']['verify']}")
            return 0

        _log(log, f"done -- {run_command(work)}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
