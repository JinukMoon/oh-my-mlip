---
name: distill
description: Distill any oh-my-mlip teacher MLIP into a CPU-deployable LAMMPS NN-MTP student via the onthefly-distill (GPL-2.0, orchestrated sibling repo) active-learning loop. Triggers on requests to distill, compress, or convert a foundation MLIP into a fast LAMMPS potential — "distill MACE into a LAMMPS student", "onthefly-distill", "NN-MTP student", "AL loop for a student potential", "shrink this teacher into something CPU-deployable" — even when "oh-my-mlip" or "distill" is not named.
argument-hint: "--teacher <variant> --structure <file> --work <dir> [--target-ps F]"
---

Defer entirely to `AGENTS.md §3D` (distill a teacher into a CPU LAMMPS student).

Read `AGENTS.md §3D` now. Do not reproduce its content here; follow it verbatim.
Key entry points: `scripts/distill_bootstrap.py --teacher <variant> --structure
<file> --work <dir>` (renders `omm_teacher.py` + `config.yaml` +
`run_distill.sh` into the work dir — nothing executes yet), then `cd <work>
&& bash run_distill.sh` (seeds the initial teacher-MD dataset and hands off
to `onthefly-distill`'s own active-learning loop).

Interview first if missing: **which teacher** (any oh-my-mlip model/version),
**which structure** (<=4 distinct elements — `pair_nnmtp` v1's bound), and
**how long** (`--target-ps`; keep it small for a first try).

If the teacher's env is not yet materialized, run `/oh-my-mlip:setup <teacher>`
first. If `$HOME/.cache/oh-my-mlip/lammps/build/lmp` does not exist yet, run
`scripts/build_lammps_nnmtp.sh` first (one-time, unattended, long-running).
