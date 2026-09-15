# Recipe: distill a teacher into a LAMMPS NN-MTP student (active learning)

Mechanics live in `AGENTS.md §3D` (the three bootstrap files,
`run_distill.sh`, stop conditions, the species bound). The engine is
`onthefly-distill`, a separate GPL-2.0 repository that is invoked, never
absorbed; its README sections 8 (configuration reference) and 9
(limitations) are the authority on its keys. This recipe adds the job
protocol from `recipes/README.md`, the executable chain, and the
acceptance rule: a distillation passes only when accuracy AND stability
targets are met within the approved budget.

## Executable chain

The user's inputs are the teacher, the structure, the intended use of the
student (hence the stability target), the budget and the split. The loop
itself, the student model, the teacher calculator and the LAMMPS build are
owned by files:

| Step | Owner file | What it fixes |
|---|---|---|
| engine | the `onthefly-distill` checkout at `--repo` / `$ONTHEFLY_REPO`; its own `al_loop_local.sh` loop script runs in place, `exec`'d verbatim | the AL loop: train → student MD → crash detection → relabel |
| engine version | `git -C <repo> rev-parse HEAD`, recorded in `PLAN.md` | which loop code ran |
| LAMMPS student binary | `scripts/build_lammps_nnmtp.sh --repo <engine> [--prefix <root>] [--ref <lammps ref>] [-j N]` → `<prefix>/lammps/build/lmp`; oracle `"$LMP_BIN" -h | grep -c nnmtp` non-zero | a LAMMPS with `pair_style nnmtp`, built as the engine's `lammps/BUILD.md` says |
| teacher calculator | `<work>/omm_teacher.py`, rendered by `scripts/distill_bootstrap.py` with the `resolve(<teacher>)` lines verbatim | the registry teacher, unmodified |
| run configuration | `<work>/config.yaml`, the engine's `config.example.yaml` deep-merged with this run's `lmp_bin`, `python_bin` (the teacher env interpreter), absolute `work_dir`, `system.*` from the structure, `teacher.*`, and the approved `al_loop.*` (`target_ps`, `max_iter`, `no_progress_limit`, `seed`); its header says that editing it after approval is `failed(provenance_drift)` | every key the engine reads |
| run unit | `<work>/run_distill.sh` — `PATH` and `PYTHONPATH` pins, `ONTHEFLY_CONFIG`, the held-out set produced BEFORE the loop, initial-pool seeding as the engine's example documents, then `exec` of the loop; with a wall-clock bound the script re-runs itself under one GNU `timeout` so that all three stages share one deadline | the rerunnable command |
| bootstrap + acceptance proposal | `python3 scripts/distill_bootstrap.py --teacher <Variant> --structure <file> --work <absdir> --acceptance --mode fixture|production --energy-mae-max <E> --force-mae-max <F> --target-ps <P> --max-iter <I> --no-progress-limit <L> [--seed <S>] [--wallclock-max-h <H>] [--heldout-file <frames> | --heldout-seed <S2> --heldout-steps <N> --heldout-save-every <K>] [--pool-steps <N> --pool-save-every <K>] [--repo <engine>] [--lmp-bin <lmp>] [--manifest-sha256 <sha>]` — with `--acceptance` the parser refuses a missing mode, threshold, target or iteration bound (none has a default); without `--acceptance` every flag of that group (`--mode`, the thresholds, `--max-iter`, `--no-progress-limit`, `--seed`, `--wallclock-max-h`, the pool and held-out flags) is refused, exit 2 naming the flag — nothing is silently ignored; defaults under `--acceptance` when omitted: seed 42, pool of 200 steps saved every 10, held-out seed 4242 with the same 200 steps saved every 10; `--heldout-seed 42` (equal to the pool default), more than 4 species, a structure with no periodic cell ("no periodic cell") or that cannot be read ("cannot read it") are refused before anything is written, and so is a `--structure` whose pbc is not `[T, T, T]`, that carries nonzero initial magmoms or initial charges, or any constraint other than `FixAtoms` ("carries scientific identity the engine cannot take as-is: … Nothing was written."); the structure named by `--structure` is staged untouched as `<work>/structure_init.vasp`, read back and compared (numbers, positions to 1e-8, cell, pbc, `FixAtoms`) with a mismatch refusing (`config.yaml` `system.init_structure` points at the copy; `provenance.structure` holds path, sha256, `staged_path`, `staged_sha256`, `staged_note`, `identity` {pbc, n_atoms, fixed_atoms, fixed_atoms_note, initial_momenta_present, initial_momenta_note, initial_magmoms, initial_charges, carried_by_staging} and `refused_at_bootstrap`; `provenance.engine_facts` carries `labeling_pbc`, `student_md_boundary`, `boundary_note`, `input_requirement`; `witness_scope` carries `witness_boundary`; the staged file is in `provenance.generated_files`); `PLAN.md` states "Structure identity carried", "Physical contract" and "Witness boundary" with every path as a JSON-quoted one-liner; every path in `run_distill.sh` is shell-quoted and the held-out `#` comment renders the user path JSON-escaped on one line — no comment carries raw operator text; `--wallclock-max-h <H>` (needs `--acceptance`, must be > 0) is enforced over the whole of `run_distill.sh` — pool seeding, held-out generation and the AL loop under one deadline: the script re-runs itself as `timeout -k 60 <round(H*3600)> /bin/sh <work>/run_distill.sh` (GNU coreutils; TERM at expiry, KILL 60 s later; the child carries `$OMM_DISTILL_ATTEMPT`), writes the start record `<work>/.distill/attempt.json` at launch (schema `oh-my-mlip.distill.attempt/1`: attempt, `started_utc`, `started_epoch_s`, `limit_s`, `grace_s`, `run_script`, `run_script_sha256`, `supervisor_pid`, `covers`; a previous attempt's marker is removed then), leaves `run/` in place, never writes the engine's `.al_status`, writes the outcome marker `<work>/.distill/wallclock.json` (schema `oh-my-mlip.distill.wallclock/3`: `state` `exhausted|within|engine_exit`, `attribution`, return code, limit, grace, timestamps, elapsed, attempt, `started_epoch_s`, `ended_epoch_s`, `run_script_sha256`, `covers`) and exits with `timeout`'s rc; the marker's `state` and `attribution` follow `budget.wallclock_enforcement.rc_rule` exactly, no slack, in supervisor and verifier alike: rc 124 ⇒ `exhausted` / `timeout_term`; rc 137 with `elapsed_s` ≥ `limit_s` ⇒ `exhausted` / `kill_at_or_after_limit` (`timeout`'s KILL after the grace or an external SIGKILL after the deadline — ambiguous, never a proven source); rc 137 earlier ⇒ `engine_exit` / `sigkill_before_limit` (source unknown: OOM killer, `kill -9`, …); rc 0 ⇒ `within` / `clean`; anything else ⇒ `engine_exit` / `exit_nonzero` — the supervisor prints `[run_distill] wall-clock <state> (<attribution>): rc=N after E s of L s -> <marker>`, `acceptance.json` `budget.wallclock_enforcement.attribution` carries the token table, and the verifier rejects a marker contradicting this rule as `failed(wallclock_evidence)`. The supervised run sits in its own process group (pgid = `timeout`'s pid; `process_group_note`, `PLAN.md` §3): a terminal Ctrl-C does not reach it; stop it early with `kill -TERM -- -<pgid>`, which yields `engine_exit`, never `exhausted`. `acceptance.json` records `budget.wallclock_enforcement` (with attempt, covers, `marker_schema`, `attempt_schema`, `rc_rule`) and `PLAN.md` the enforced line. The parser refuses (exit 2, nothing written) a bound ≤ 0, a bound without `--acceptance`, a host with no GNU `timeout` on `PATH`, any of `--pool-steps` / `--pool-save-every` / `--heldout-steps` / `--heldout-save-every` below 1, and `--heldout-steps` < `--heldout-save-every` (the generated set would keep only the dropped init frame); a `--heldout-file` containing a frame equal to the init structure exits 1 naming the frame indices, nothing written; a `--heldout-file` that `ase` cannot parse ("cannot parse it … drop the flag …") or that carries zero frames ("carries no frames") is refused before any write. Without the flag: plain `exec`, `wallclock_enforcement` null, plan says "none stated (unbounded; operator-supervised)". `distill_verify.py` is a separate command and is not inside the bound; each bootstrap step is logged to `<work>/bootstrap.log`. Work-directory ownership: `--work` must be new or an existing empty directory — before the first write the bootstrap refuses (exit 1, nothing written) a `--work` that is not a directory ("exists and is not a directory; nothing was written") or that holds any entry (dotfiles, a `run/` tree, `.distill/`, a previous bootstrap's outputs, a symlink or hardlink at an output name), naming the entries ("--work X is not empty (a, b, …); bootstrap writes bootstrap.log, structure_init.vasp, omm_teacher.py, config.yaml, run_distill.sh, acceptance.json, PLAN.md only into a new or empty work directory and never overwrites, regenerates or deletes what is there … Pick a new --work, or empty this one yourself.") and adding "--structure <p> lives inside it" / "--heldout-file <p> lives inside it" when either resolves under `--work`; there is no re-bootstrap into an existing proposal. The six generated proposal outputs (`structure_init.vasp`, `omm_teacher.py`, `config.yaml`, `run_distill.sh`, `acceptance.json`, `PLAN.md`) are created exclusively (`O_CREAT|O_EXCL|O_NOFOLLOW`; `run_distill.sh` created with mode 0755, subject to the operator's umask, e.g. 0700 under umask 077), and `bootstrap.log` is created once with `O_CREAT|O_EXCL|O_NOFOLLOW|O_APPEND` right after the emptiness check and written through that retained stream for the rest of the bootstrap — never reopened by name, so an object planted there later (hardlink, symlink, FIFO) is neither written to nor opened; closed on every exit, refusal included, and never unlinked: an output name that already exists, a dangling symlink included, is refused ("already exists; bootstrap creates its outputs exclusively and never overwrites or regenerates one") and a symlink is never followed. The bootstrap deletes nothing under `--work`: a staged copy failing the read-back comparison is retained with `bootstrap.log` ("The rejected copy and bootstrap.log are retained (nothing is deleted, nothing else was written); remove the work directory yourself before retrying") and the next bootstrap into that directory is refused as non-empty. `--structure` and `--heldout-file` are only read (resolved path, `ase` read, sha256), never modified. Disclosed as `acceptance.json` `provenance.work_dir_policy` and the PLAN.md line "Work directory ownership: …" | the files above plus `acceptance.json` and `PLAN.md` (with an `## Identity` section), written before anything computes |
| held-out set | rendered into `run_distill.sh` by `--acceptance`: a separate teacher MD with a seed that must differ from the pool's, or the user's `--heldout-file` frames relabelled by the same teacher; the raw teacher MD goes to `<work>/heldout/heldout_md_raw.extxyz` and `<work>/heldout/heldout.extxyz` is that set with frame 0 dropped (the engine's `teacher_md.py` saves the pre-MD init frame first for every seed, so frame 0 is shared with the pool), both outside the engine's `work_dir`; `acceptance.json` records `split.heldout.raw_path` and `drops_frame0` | an accuracy witness that is not the trainer's random validation split |
| status | `<work>/run/.al_status` — `SUCCESS round<N> stable`, `STALLED`, `STOPPED backstop`, `STOPPED label_fail`, `FAILED ...` | the loop's own verdict on stability |
| verdict | `python3 scripts/distill_verify.py --work <work> --json [--python <teacher interp>] [--lmp-bin <lmp>] [--repo <engine>] [--ledger <jsonl>] [--manifest-sha256 <sha>] [--campaign-id <id>]` — in order: `acceptance.json` present and `config.yaml`'s `al_loop.*` unchanged from it, `.al_status` classified as written, real relabel rounds counted, held-out independence (path outside `work_dir`, distinct seed, empty per-frame fingerprint intersection with the pool), held-out MAEs computed under the teacher env through the engine's own modules, the loop's final `lmp` run cited plus a fresh `run 0` witness of the final `.bin`; writes `<work>/verify/distill_verify.json` and `verify.log`, appends one ledger row, exits 0 iff `passed` (`--no-lmp-witness` on a run that would otherwise pass is `incomplete(lmp_witness_skipped)`, never a pass; a crashing or non-finite `lmp` is `failed(lmp_witness)`); on `config_drift` / `provenance_drift` it stops before deriving the run dir, reading nothing under `run/` (`stability.kind` `not_inspected`) | the accuracy half of the verdict and the LAMMPS requirement, never judged by reading logs |

The loop is never reimplemented, patched or driven step by step by the
agent; a change the engine cannot express through `config.yaml` is out of
scope for a job.

## 1. Ask

- **Teacher** variant; **structure** file with at most four distinct
  elements; **work dir** (absolute).
- **Intended use of the student** — temperature, ensemble, duration of the
  production MD. The stability target is set from this, not from a default.
- **Fixture or production run** — a fixture exercises the loop cheaply; a
  production run must meet the acceptance targets.
- **LAMMPS binary** — present at the `--lmp-bin` default, or built once
  with `scripts/build_lammps_nnmtp.sh`; a production run passes `--ref`
  with a specific LAMMPS tag or commit and records it (the default `stable`
  is a moving upstream branch).
- **Engine checkout** — `--repo` or `$ONTHEFLY_REPO` (required; there is no
  built-in default); clone it from https://github.com/JinukMoon/onthefly-distill.

## 2. Plan — the acceptance proposal, written before anything runs

The proposal is rendered, not typed: the bootstrap's `--acceptance` flags
carry every target, and `<work>/acceptance.json` (machine copy) plus
`<work>/PLAN.md` (human copy) are what the user approves. The agent's
part is choosing the values:

1. **Accuracy targets** — `--energy-mae-max` (meV per atom) and
   `--force-mae-max` (meV per Å) on the held-out set, chosen for this task;
   the parser has no default for either.
2. **Stability target** — `--target-ps` of stable student MD at the
   stated conditions. The engine fixes the student MD temperature inside
   its own code; it is not a config key, and the plan says so.
3. **Budget** — `--max-iter`, `--no-progress-limit`, `--seed`, and
   `--wallclock-max-h` (enforced by `run_distill.sh` through one GNU
   `timeout` over the whole run — pool seeding, held-out generation and
   the AL loop share the deadline; exhaustion is `unmet(budget:wallclock)`,
   and a bound whose start record or marker cannot be tied to the attempt
   is `failed(wallclock_evidence)`, never a pass). `distill_verify.py` is
   a separate command outside the bound, and the plan says so. Omitting
   the flag means an unbounded, operator-supervised run, and the plan
   says so.
4. **Split** — the AL pool (`--pool-steps`, `--pool-save-every`) and the
   held-out set: a separate teacher MD (`--heldout-seed`, which must
   differ from the pool's seed, `--heldout-steps`, `--heldout-save-every`)
   or the user's own frames (`--heldout-file`) relabelled by the same
   teacher. A generated held-out set drops its frame 0 (the shared
   pre-MD init frame), so `--heldout-steps` must be at least
   `--heldout-save-every`; a `--heldout-file` frame equal to the init
   structure is refused at bootstrap. The trainer's random validation
   split of the pool is never the held-out set; `acceptance.json` states
   the independence rule.
5. **Student settings** — the engine's example values under `student:`
   unless the user changes them in `config.yaml` after rendering; any
   change is listed key by key with the edited file's sha256.
6. **Commands** — the bootstrap line from the chain table with every
   value filled in, then `cd <work> && sh run_distill.sh > distill.log 2>&1`
   (recorded under `commands.run` in `acceptance.json` and `PLAN.md` §5;
   no `| tee`, so the exit status is the script's own; follow with
   `tail -f <work>/distill.log`), then the `distill_verify.py` line from
   the chain table (`commands.verify`).

Fixture definition (`--mode fixture`): a small initial pool and a long
enough student MD that at least one crash → relabel → retrain round
actually happens; the parser bounds `--max-iter` to the fixture ceiling.
Its verdict is about the loop, never about the student's quality —
thresholds are reported, not gating.

A bootstrap without `--acceptance` is the quick demo path (a default
`target_ps`, the engine's example budget); it is not a job under this
recipe.

## 3. Approve

Present `<work>/PLAN.md` as rendered — targets, budget, split, student
settings, commands, provenance (hub and engine commits, teacher spec,
structure sha256, generated-file sha256s, `lmp_bin` with its
`lmp_bin_sha256` — the binary is pinned by hash at bootstrap and re-hashed
at verify; the hash-to-build-ref mapping is a missing part below), expected duration
class (held-out MD and teacher MD first, then repeated train / student MD
/ relabel rounds) — and wait for approval of that file.

## 4. Execute

Launch in the background with `tee` and poll `<work>/run/.al_status` and
the round logs. The engine's scripts and modules are never edited; a
needed change in `config.yaml` beyond the approved values is an out-of-
scope change. A failure inside the teacher env (import, missing package)
is `failed(<class>)` and goes back to `recipes/setup.md`; the env is not
repaired from here.

## 5. Verify — the verdict rule

The rule is written into `acceptance.json` under `verdict_rule` and
applied by `scripts/distill_verify.py`, whose verdict is rendered as
printed. Per mode:

| `.al_status` | held-out MAEs vs thresholds | Verdict (production) |
|---|---|---|
| `SUCCESS round<N> stable` (round 0 included) | within both thresholds, and the final `.bin` passes the `lmp` witness | `passed` |
| `SUCCESS round<N> stable` | above a threshold | `unmet(accuracy)` — stable but not accurate; the verdict carries a reconfigured-rerun proposal (more pool data, more rounds) inside the approved budget class, needing approval |
| `STALLED`, `STOPPED backstop`, `STOPPED label_fail` | any | `unmet(stability)` / `unmet(budget)` — loop or budget exhausted; artifacts kept |
| `FAILED ...`, `ERROR`, or the engine's one-shot fallback (`DONE oneshot`: the teacher could not relabel and the loop silently degraded) | any | `failed(<class>)` |
| any of the above, but a valid `<work>/.distill/wallclock.json` says `exhausted` | any | `unmet(budget:wallclock)` — outranks every engine state (a `SUCCESS` line written while the group was being signalled included); the proposal's first option is `--wallclock-max-h <2H>`, needing approval. A valid marker with `within` changes nothing; `engine_exit` is the row below |
| a bound was approved but its evidence does not hold | any | `failed(wallclock_evidence)` — start record missing; marker missing while `.al_status` is terminal; marker unreadable or of an old schema; marker not bound to the attempt (attempt id or start differ, limit or grace not the approved ones, script sha256 not the approved `run_distill.sh`, state, `attribution` and rc/elapsed inconsistent (the token is recomputed from rc and elapsed and must match, and must be one of the five), elapsed ≠ ended − started, `.al_status` written outside the attempt window); or a bound stated with no enforcement plan |
| a valid marker says `engine_exit` (any non-expiry nonzero rc: 1, an early 137, 143, …), whatever `.al_status` says — `SUCCESS` included | not computed | `failed(supervisor_exit)` — no held-out metrics and no `lmp` witness are computed |
| start record present, no marker yet, loop running | — | `incomplete(loop_not_terminal)` |

Precedence, top wins, as `distill_verify.py::decide` implements it:
`failed(config_drift)` (an `al_loop.*` value in `config.yaml` differs
from the approved `acceptance.json`) > `failed(provenance_drift)` (any
recorded sha256 — `config.yaml` whole file, `run_distill.sh`,
`omm_teacher.py`, `structure_init.vasp`, the source structure, the engine
files it invoked — mismatched or missing; both drifts return before
anything under `run/` is read) > `failed(wallclock_evidence)` (marker
invalid or unbound) > `unmet(budget:wallclock)` >
`failed(wallclock_evidence)` (no marker while `.al_status` is terminal) >
`failed(supervisor_exit)` (a valid marker with state `engine_exit`,
whatever `.al_status` says) >
`incomplete(loop_not_terminal)` > `failed(teacher_cannot_relabel)` >
`failed(engine_failed | engine_error)` > `failed(malformed_evidence)` (a
file under `run/` matching the trainer's `al_iter*_labeled.extxyz` glob
that is not the engine's `al_iter<K>_labeled.extxyz` with decimal `K`;
listed in `relabel.malformed_files`, never a crash) >
`failed(heldout_leak)` (a held-out frame also in the pool —
`dataset.extxyz` or any `al_iter*_labeled.extxyz` — or a held-out frame
that is the init structure, even with no pool file on disk; the detail
names the cause) > `failed(heldout_missing)` (zero frames included) >
`failed(heldout_eval)` > the fixture rule (`unmet(fixture:no_relabel)`)
or the production rules above (`unmet(stability:stalled)`,
`unmet(budget:backstop|label_fail)`, `unmet(accuracy)`) >
`incomplete(lmp_witness_skipped)` | `failed(lmp_witness)` > `passed`.
Only `config_drift` and `provenance_drift` outrank the wall-clock
verdicts; `heldout_leak` sits below them. A `SUCCESS` line under an
`engine_exit` marker never reaches the status rules. The identity checks detect
ordinary post-approval drift and are not a defence against an owner
rewriting `acceptance.json` itself. The report's and the ledger row's
`artifacts` block names `accuracy_artifact` (the `.pt`),
`accuracy_scope`, `witness_artifact` (the `.bin`), `witness_scope`
("finiteness/stability only, NOT accuracy equivalence with the .pt";
`acceptance.json` `witness_scope` carries the same statement),
`witness_state` `ok|failed|skipped|not_run`, `witness_boundary` ("p p f",
with `witness_boundary_source` `ontheflydistill/student_md_lammps.py
run()`), `accuracy_pbc` ("[T, T, T] — force_pbc") and `boundary_note`
("disclosed, not reconciled") — the witness itself is unchanged.
The verify report carries `contract.provenance_drift` (`{name: {path,
recorded, actual}}`) and a `wallclock` section (marker state,
`attribution`, `attribution_note`, `attempt_record`, `attempt`, `valid`,
`pending`, `problems`); the ledger row's evidence gains `wallclock`,
`wallclock_marker`, `wallclock_attempt`, `wallclock_valid`,
`wallclock_attribution`, `wallclock_returncode`, `provenance_drift` (a
list) and `malformed_files`. In the five-state accounting of
`recipes/README.md` every `unmet(<class>)` and `incomplete` is not
passed and is reported as `failed(unmet:<class>)` / not attempted, with
the artifacts kept.

A fixture (`--mode fixture`) passes when at least one real relabel round
occurred (a non-empty `al_iter<K>_labeled.extxyz` and a retrain after
it), the loop reached a terminal state, the held-out metrics were
computed, and the `lmp` witness ran; its thresholds are reported, not
gating. `SUCCESS` at round 0 in a fixture is `unmet(fixture:no_relabel)`
— the loop was never exercised.

The held-out MAEs come from `<work>/verify/metrics.json`, computed by the
oracle under the teacher env from the final `.pt` through the engine's
own modules; the trainer's `E=` / `F=` / `best F_MAE` lines in
`train_*.log` are pool-validation context, never the held-out result.
The exported `.bin` is witnessed by `lmp` for stability and finiteness
only.

- Evidence to cite: `<work>/verify/distill_verify.json` (the verdict,
  the `.al_status` line it classified, the relabel-round count, the
  held-out sha256 and fingerprint check, `metrics.json`, the witness
  directory), the last round's `train_*.log`, the student MD log and
  `failure.json` per round, the final `.bin` path,
  `heldout/heldout.extxyz`, and the ledger row the oracle appended.
- The record also carries `acceptance.json` (provenance: hub and engine
  commits, teacher spec, structure sha256, generated-file sha256s,
  `provenance.engine.files` — the sha256 of each engine file the run
  invokes — and `provenance.identity_rule`, `manifest_sha256` when
  given), `.distill/attempt.json` and `.distill/wallclock.json` when a
  bound was set, the LAMMPS binary path and the ref it
  was built from, the final `config.yaml` (with its sha256),
  `omm_teacher.py`, `run_distill.sh`, `bootstrap.log`, `distill.log`. A
  rerun from the same files reproduces the procedure; MD and training are
  stochastic and the student weights are not expected to match bitwise.
- Every stopped run keeps its model, trajectories and logs; the reason for
  stopping is part of the record.

## 6. Out of scope

Raising the budget, changing the teacher or structure, moving the run to a
remote machine, or editing engine code — renewed approval each; the last
one is a change to the sibling repository, not to this one.

## Planned helpers (not implemented)

None remaining for this recipe: the acceptance proposal (`--acceptance`)
and the verify oracle (`scripts/distill_verify.py`) have landed and are
named above.

Landed without a unit test: none for this recipe (`tests/test_distill_verify.py`
exists in the tree; that is a tree fact, not a statement that the test has
been independently reviewed or is adequate). The oracle's refusals
(`config_drift`, `provenance_drift`, `wallclock_evidence`,
`malformed_evidence`, `heldout_leak`, a skipped witness as
`incomplete(lmp_witness_skipped)`) stand as printed.

Missing executable parts (described by function; no owner file yet):

- the LAMMPS build script does not record the ref it built from into the
  work dir; `acceptance.json` records the binary path, the plan records
  the ref;
- `distill_verify.py` runs as a separate command after `run_distill.sh`
  and is outside the wall-clock bound; its own duration is
  operator-supervised.
