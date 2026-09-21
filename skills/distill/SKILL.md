---
name: distill
description: Distill any oh-my-mlip teacher MLIP into a CPU-deployable LAMMPS NN-MTP student via the onthefly-distill (GPL-2.0, orchestrated sibling repo) active-learning loop. Triggers on requests to distill, compress, or convert a foundation MLIP into a fast LAMMPS potential — "distill MACE into a LAMMPS student", "onthefly-distill", "NN-MTP student", "AL loop for a student potential", "shrink this teacher into something CPU-deployable" — even when "oh-my-mlip" or "distill" is not named.
argument-hint: "--teacher <variant> --structure <file> --work <dir> [--target-ps F]"
---

Defer entirely to `AGENTS.md §3D` (distill a teacher into a CPU LAMMPS student).

Read `AGENTS.md §3D` now. Do not reproduce its content here; follow it verbatim.
Key entry points: `scripts/distill_bootstrap.py` (args per its `--help`:
teacher variant, structure path, work dir) renders `omm_teacher.py` +
`config.yaml` + `run_distill.sh` into the work dir — nothing executes yet.
Then `cd <work> && bash run_distill.sh` seeds the initial teacher-MD dataset
and hands off to `onthefly-distill`'s own active-learning loop.

**Default to the acceptance path** for any student whose result will be used:
bootstrap with `--acceptance`, stating every target the way `recipes/distill.md`
§2 lists them, then judge the finished run with `scripts/distill_verify.py`.
That path renders an independent held-out set and gates the student's accuracy
on it. Without `--acceptance` there is no held-out set and no accuracy gate:
the engine's `SUCCESS` only means the student's MD survived the target length,
which says nothing about whether it reproduces the teacher. Use the bare form
only as a quick demo, and tell the user that is what it is.

Interview first if missing: **which teacher** (any oh-my-mlip model/version),
**which structure** (<=4 distinct elements — `pair_nnmtp` v1's bound), and
**how long** (`--target-ps`; keep it small for a first try).

If the teacher's env is not yet materialized, run `/oh-my-mlip:setup <teacher>`
first. If `$HOME/.cache/oh-my-mlip/lammps/build/lmp` does not exist yet, run
`scripts/build_lammps_nnmtp.sh` first (one-time, unattended, long-running).

Whole-job requests (interview, scoped plan, approval, execution, evidence)
follow the shared recipe `recipes/distill.md` — read it after the section
above; it says which helpers exist today and which are still planned.
