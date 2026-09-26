# Recipe: catbench adsorption benchmark

Mechanics live in `AGENTS.md §3B` (materialized jobs through
`scripts/catbench_jobgen.py`, one process per model, aggregation through
`scripts/catbench_report.py`), `run_examples/README.md` (data layout) and
`docs/catbench_data_format.md` (JSON schema). This recipe adds the job
protocol from `recipes/README.md`, the executable chain, the version rule,
the copy-only VASP staging, the comparison rule, and the non-registry
calculator path.

## Executable chain

The user's inputs are the dataset, the reaction coefficients, the models
and versions, D3, `calc_num`, and the catbench version pin. Every step
that turns them into results has an owner:

| Step | Owner file | What it fixes |
|---|---|---|
| public dataset download | catbench's own `get_benchmark` in `catbench.adsorption.data.zenodo`, run under a catbench-bearing env interpreter | download routes (Zenodo, CDN, CatHub); nothing re-implemented |
| VASP tree → JSON | catbench's own `vasp_preprocessing(dataset_name, coeff_setting)` — destructive on its input tree, hence the copy-only staging below | reaction energies from `OSZICAR` + `CONTCAR` pairs |
| copy-only staging | `scripts/catbench_vasp_stage.py scan|stage|verify` — scan lists pairs, half-pairs and symlinks, writes `mapping_proposal.json` and the owner marker, and refuses an `--out` inside, equal to, or containing the source (exit 3, `out_refused`); stage refuses, before any write, a source or dest whose path passes through a symlink at any level (`destination_refused`), symlinks inside the source or an existing staged copy (`symlinks_in_source`, `symlinks_in_staged_tree`), a mountpoint at or under source or dest (`mounted_tree_unsupported`; `mountinfo_unavailable` fails closed), pre-existing content it did not produce at a path it writes (`unrelated_destination_content`), a dataset name outside `[A-Za-z0-9][A-Za-z0-9_.-]*` (`dataset_name_invalid`), a non-empty dest whose marker names another source, and missing DFT results upstream would need (`missing_dft_results`); it hashes the originals (`originals.sha256`) plus a size/mtime inventory, copies only what upstream reads, re-hashes the copy, and emits `stage_preprocess.py` (with an exit-3 guard when `--catbench-version <ver>` is given) + a shlex-quoted `run_stage.sh`; verify re-hashes the originals afterwards and fails on symlinks that appeared since staging (`symlinks_now`) | the originals are never touched; the conversion stays upstream |
| official values | `scripts/catbench_leaderboard.py snapshot|compare` — snapshot records URL, HTTP code and UTC time; compare renders the hub's `report/mae_table.csv` next to the fetched values in the user's model order, lists every condition difference as a discrepancy, asserts `result/` holds only the selected models, and writes "no comparable official value" rows with the attempted URL and time when nothing comparable exists | fetched, never recomputed; never a ranking of unlike results |
| existing JSON check | `scripts/catbench_datasets.py --check` (row below), which loads the file through catbench's own reader under the same interpreter | schema validity, with each problem named |
| calculator lines | `oh_my_mlip.resolve()` — verbatim from `models.json` | no hand-written calculator |
| job files | `run_examples/catbench_quickstart.py <tag> --only ... --catbench-version <ver> [--version FAM=VER] [--all-versions] [--arch smNN] [--d3] [--calc-num N] [--max-reactions N] [--regenerate] [--no-fetch] --emit-only` (`--max-reactions N` benchmarks the first N reaction ids of `<tag>` in sorted order, written as `raw_data/<tag>_first<N>_adsorption.json` with only the `_structures` those reactions use; every run records `result/omm_dataset.json` {benchmark, source_tag, source_sha256, reactions, of, subset, and for a subset reaction_ids}; the report prints "Subset: N of M" under the MAE table and leaves an empty class's MAE as "0 reactions in this class" (an empty CSV cell); a result folder already holding another dataset or subset, or a subset over results with no record, stops the run (exit 2); a `<tag>` missing from `raw_data/` is fetched first under an installed env interpreter — a Zenodo benchmark through `catbench_datasets.py --fetch`, any other tag through upstream `cathub_preprocessing`; `--no-fetch` stops instead; `--all-versions` covers every declared version except those marked `"catbench": false` in `models.json`, which stay usable for single runs; `--arch` selects the arch-pinned NequIP/Allegro `.pt2` for a GPU other than this host's) → `scripts/catbench_jobgen.py` → `jobs/catbench_<MLIP>.py`, `jobs/catbench_<MLIP>.meta.json` and `jobs/run_catbench_<MLIP>.sh` (`--slurm` adds `jobs/run_slurm_<MLIP>.sh` with the identical body); one registry model at a time, with the approved version pin: `python3 scripts/catbench_jobgen.py --tag <tag> --workdir <work> --model <Family> [--version V] [--calc-num N] [--d3] --catbench-version <ver>` | byte-identical files for identical inputs; `--catbench-version` is guarded and recorded in the job's meta |
| non-registry calculator | `python3 scripts/catbench_jobgen.py --tag <tag> --workdir <work> --calc-file <calc.py> --python <interp> [--name N] [--structure <file>] [--calc-num N] [--d3] --catbench-version <ver>` — the file must define `make_calculator()`; the script builds the spec from it, emits `jobs/witness_<name>.py` + `jobs/run_witness_<name>.sh` (energy and forces on a small structure) and the job pair | no hand-written spec or witness; `models.json` untouched |
| execution | each emitted `.sh`: `set -eu`, `cd` to the work dir, one `export` per `env_run` key, `exec` of the env interpreter on the `.py` | one process per model, no inline python |
| report | `scripts/catbench_report.py --result ./result --out ./report [--python <interp>] --expect-models <A,B,...> --catbench-version <ver>` → `report/mae_table.md`, `report/mae_table.csv`, `report/report_meta.json` (catbench version, interpreter, models), plus catbench's own `AdsorptionAnalysis.analysis()` and `threshold_sensitivity_analysis()` outputs (classification rate vs displacement / bond-length threshold) in `report/`; `--expect-models` stops (exit 3) on an extra or a missing result directory, `--catbench-version` stops after import when the interpreter's catbench differs; both are carried into `report/run_report.sh` | aggregation under a catbench-bearing interpreter, over exactly the approved model set |
| catbench version in hub-built envs | `CATBENCH_PIN` in `install.sh`; `OMM_CATBENCH_VERSION=<ver>` for one build | the version a job actually imports |
| version rule | `scripts/catbench_version.py --mode new|rerun` → `catbench_version.json` (official latest, hub pin, per-env held version, lookup URL and UTC time, `requires_env_upgrade`, `status: proposed|recorded`, `env_mismatch`, `env_unknown`); `--mode new` without `--offline` stops (exit 2, "rerun with --offline") when the PyPI lookup fails instead of proposing a pin; `--mode rerun` stops (exit 2) when any `--python` env does not hold the recorded version — reported, never upgraded | latest for a new job, the recorded pin for a rerun; never a silent upgrade |
| dataset recommendation | `scripts/catbench_datasets.py --list|--target "<text>" [--confirm] [--json]` — upstream-cited catalogue rows kept apart from the user-supplied hints; `--confirm` checks the Zenodo file size against the live record (`zenodo_size_confirmed`) and never verifies the leaderboard alias (`leaderboard_alias_verified` always false) | a recommendation with its citation, or the question to ask |
| dataset check / fetch | `<env>/bin/python scripts/catbench_datasets.py --check <json> [--record <file>] [--catbench-version <ver>] [--json]` validates the adsorption format upstream reads and prints counts, gas keys, sha256 and catbench version; `--fetch <name> --workdir <work> [--catbench-version <ver>] [--json]` downloads a Zenodo benchmark itself: first upstream's catbench.org gzip copy, accepted only when the unpacked file has the size and md5 of upstream's Zenodo listing, else Zenodo resumably (a dropped connection keeps `<name>_adsorption.json.part` and the same command resumes it; the file appears only after size and md5 match), and runs upstream `get_benchmark` for any other name, into `<work>/raw_data/`; a failed download is a `[stop]` line naming the kept bytes, never a traceback; it never replaces an existing file, and writes `<name>_adsorption.provenance.json`; exit 0 ok, 3 format problem (each named) or version mismatch, 2 without `catbench` in the interpreter; `--check`/`--fetch` exclusive | the dataset's identity (sha256) and provenance, produced by an owned script rather than a pasted body |

## 1. Ask

Required inputs (skip any the user already gave):

1. **Scientific target** — surfaces, adsorbates, chemistry the user wants
   to test. Ask only when no dataset is named.
2. **Dataset** — one of: an existing `raw_data/<tag>_adsorption.json`; a
   public catbench dataset name; the user's own VASP result tree.
3. **Models** — registry names or versions, or a non-registry calculator
   (the user's ASE calculator file plus the interpreter that runs it).
4. **D3** on or off; `calc_num` if not the default.
5. **New job or rerun** of an approved job — this decides the version rule.

Dataset recommendation, only when the target had to be asked:
`python3 scripts/catbench_datasets.py --target "<the user's words>" --confirm --json`.
The script keeps two layers apart — the catalogue rows quoted from the
catbench README with their citation, and the science-to-dataset rules of
thumb that users supplied, labelled as hints — and returns either a
recommendation, `ask` with the question to put to the user, or
`needs_target`. Each row carries two separate flags:
`zenodo_size_confirmed` (Zenodo lists the file at a README-consistent
size) and `leaderboard_alias_verified`, which is always false —
`leaderboard_alias_note` says the alias is README-count inference, and
`confirmation.leaderboard.note` that "listed" only means the id string
exists on `meta.json`, never that it is this Zenodo dataset. The text
row reads "[zenodo size confirmed; leaderboard alias '<id>' NOT
verified; <citation>]" or, offline, "[NOT live-confirmed; leaderboard
alias … NOT verified; …]"; nothing about a dataset's content or size is
asserted from memory. Never propose a fixed "representative suite"
unasked; the script does not either.

## 2. Plan

### Version rule — latest for a new job, pinned for a rerun

- New job:
  `python3 scripts/catbench_version.py --mode new --python <env1>/bin/python [--python <env2>/bin/python ...] --out <work>/catbench_version.json`
  (`--offline` when there is no network: latest is recorded as null, the
  pin still is). The record holds the official latest release, the hub
  pin, what each target env holds, the lookup URL and UTC time, and the
  envs listed under `requires_env_upgrade`. `PLAN.md` quotes the record
  and names the single version the job runs with. The version that runs
  is the one the env holds; an existing env is never upgraded silently.
  For an env under `requires_env_upgrade` the only executable path today
  is a fresh env build with `OMM_CATBENCH_VERSION=<ver> bash install.sh <env>`
  inside a fresh root (`recipes/setup.md`), listed as its own approved
  step — or the plan keeps the held version and says so.
- Rerun:
  `python3 scripts/catbench_version.py --mode rerun --record <work>/catbench_version.json --python <env>/bin/python ...`
  — the approved record is reused, nothing is re-discovered, and an env
  that no longer holds the recorded version is reported rather than
  upgraded. The rerun executes the `jobs/*.sh` files already on disk;
  re-emitting them with the same inputs produces byte-identical files
  (`diff` is the check).

### Data path

**A. Existing JSON** — owned by `scripts/catbench_datasets.py`, run under
a catbench-bearing env interpreter:

```bash
<env>/bin/python scripts/catbench_datasets.py --check raw_data/<tag>_adsorption.json --record <work>/dataset_check.json --catbench-version <ver> --json
```

It validates the adsorption format upstream reads (the exact `star` slab
key, `<X>star` adslab keys, `<G>gas` keys; a non-gas entry needs `atoms`,
a finite `energy_ref` and a finite `stoi`, a gas entry `atoms` and
`stoi`) and prints the reaction and structure counts, the gas keys, the
file's sha256 and the catbench version. Exit 0 valid; 3 with every
format problem named, or when the `catbench` installed in the running
interpreter (`catbench.__version__`) is not `--catbench-version` — the
check is against the env, never against anything recorded in the file
or its provenance (`[stop] approved catbench V but this env has X`);
2 when the interpreter has no `catbench`. `--record` keeps that report as
the plan's dataset evidence.

**B. Public dataset** — the same script, from `<work>`:

```bash
<env>/bin/python scripts/catbench_datasets.py --fetch <dataset name> --workdir <work> --catbench-version <ver> --json
```

Upstream `get_benchmark(<name>)` writes `<work>/raw_data/<name>_adsorption.json`;
the script never replaces an existing file (it reports `pre_existing`)
and writes `<work>/raw_data/<name>_adsorption.provenance.json` (name,
catbench version, sha256, size, UTC time, `fetched` or `pre_existing`).
An existing provenance record is kept only when its recorded sha256
equals the current file (`provenance_note: "existing provenance record
kept (sha256 matches the file)"`, exit 0); a mismatch is
`provenance_drift {recorded_sha256, current_sha256}` plus `error` in the
JSON and `[stop] provenance drift: <prov> records sha256 <recorded>,
but <file> now hashes to <current>; the file changed after it was
recorded. Not rewritten — move the stale record and the file aside (or
use a fresh --workdir) and fetch again`, exit 3; an unreadable record is
`[stop] existing provenance record <prov> is unreadable (<exc>); not
rewritten`, exit 3; a name that is not a plain file-name stem is
`[stop] dataset name '<x>' is not a plain file-name stem`, exit 2.
Otherwise the same exit codes as `--check`; `--check` and `--fetch` are
mutually exclusive.

The provenance record (or the `--record` file) is the dataset's evidence
in the plan; a JSON whose sha256 differs from it is a new dataset, not
the same one.

**C. User VASP results — copy-only staging**

`vasp_preprocessing` runs `cleanup_vasp_files(keep_files=["OSZICAR",
"CONTCAR"])` on the tree it is given and deletes every other file, so it
only ever sees a copy. `scripts/catbench_vasp_stage.py` owns that
boundary, in this order:

1. `python3 scripts/catbench_vasp_stage.py scan --source <originals> --out <stage>`
   — writes `<stage>/mapping_proposal.json`: every `CONTCAR` + `OSZICAR`
   pair, every half-pair (a missing DFT result is reported, never
   invented), every symlink (refused). `slab` and `adslab` coefficients
   follow the upstream convention; gas terms are candidates. The user
   confirms the mapping — which slab pairs with which adslab, each gas
   coefficient, every missing reference — and the confirmed result is
   saved as `<stage>/coeff_setting.json`. Nothing is guessed, computed or
   filled in.
2. `python3 scripts/catbench_vasp_stage.py stage --source <originals> --dest <stage> --dataset-name <name> --coeff <stage>/coeff_setting.json --python <env>/bin/python --catbench-version <ver>`
   — refuses before any write and before the owner marker: symlinks in the
   source or anywhere under an existing staged copy (`symlinks_in_source`,
   `symlinks_in_staged_tree`, naming each link); a mountpoint at or under the
   chosen source or dest (`mounted_tree_unsupported`; st_dev pinned plus
   `/proc/self/mountinfo`, so same-device bind mounts are caught while ordinary
   ancestor mounts such as `/` are not — `mountinfo_unavailable` fails closed);
   a dest inside/equal to the source or not owned for it; and pre-existing
   content at any path it writes (`destination_refused` for a symlink or
   escape, `unrelated_destination_content` for a file it did not produce).
   It then hashes the originals, copies only what upstream reads into
   `<dest>/.<name>.staging.tmp`, validates the exact planned set, renames it
   into place, and emits `stage_preprocess.py` + `run_stage.sh`; nothing runs
   yet. An identical re-stage is reused (`reused_identical_staging`, previous
   outputs kept); `staged_tree_differs`, `staging_tmp_exists` and
   `stale_outputs_without_copy_tree` are refused with nothing removed. A copy
   that fails part way RETAINS its tmp tree and names it in `retained_tmp`
   (`copy_failed`, `staged_tree_mismatch`): never auto-deleted, originals
   untouched, a rerun refuses the leftover. There is no replace flag.
   Still in the frozen script: `--all-files` copies everything rather than
   only what upstream reads, and `copy_hash_mismatch` (the manifest re-hash
   of the copy differs) is a further exit-3 failure after the copy that
   leaves the tree in place.
3. `bash <stage>/run_stage.sh` — calls upstream `vasp_preprocessing` under
   the named interpreter; the JSON lands in `raw_data/`.
4. `python3 scripts/catbench_vasp_stage.py verify --source <originals> --dest <stage>`
   — an unchanged originals tree is a verification item, not an assumption.

### Jobs

`python3 run_examples/catbench_quickstart.py <tag> --only M1,M2 --catbench-version <ver> [--version FAM=VER] [--d3] [--calc-num N] --emit-only`
writes `jobs/catbench_<MLIP>.py`, `jobs/run_catbench_<MLIP>.sh` and
`jobs/catbench_<MLIP>.meta.json`; the plan lists those files.
`--catbench-version` is passed into each job's guard and its meta file —
the same value the version rule above fixed. A non-registry calculator
is emitted by the `--calc-file` line of the chain table; its witness
runner is listed in the plan before its job.

Rerun rule: a rerun executes the job files already on disk. The
quickstart and `scripts/catbench_jobgen.py` keep every file whose
rendered bytes are unchanged (reported `reused`) and refuse to replace a
file whose rendering changed — nothing is written, every stale path is
named, exit 3 (`[stop] existing job file(s) differ ...`). Only
`--regenerate` (both entry points) replaces them, and a plan that uses
it says why the rendering changed (a new pin, a new calculator, a new
dataset).

### Comparison rule

Official leaderboard values are fetched at request time, never recomputed.
No published uMLIP is rerun as a baseline: `result/` must contain only the
models the user selected, and the plan says so.

## 3. Approve

Present `PLAN.md`: the version triple and the version that runs, the
dataset and its provenance (name or path, sha256), the mapping table (VASP
case), the models and versions, the job files, the compute class (one GPU
process per model, serialized), and the comparison source. Wait for
approval.

## 4. Execute

- Materialize, then run the files: `bash jobs/run_catbench_<MLIP>.sh > jobs/<MLIP>.log 2>&1`
  per model, in the approved order, one at a time. This is also the rerun
  command: the `.sh` on disk (edited by the user or not) is what runs.
  `catbench_quickstart.py` without `--emit-only` is a shortcut for a
  first run into an empty `jobs/` only — it re-emits every file before
  executing and would overwrite an edited one; `--submit` dispatches
  through the injectable hook instead of running locally, and with
  `--calc-file` it refuses (exit 3) unless `jobs/witness_<name>.json`
  exists with `ok: true` and a `calc_file_sha256` equal to the current
  file's hash.
- Non-registry calculator: `bash jobs/run_witness_<name>.sh` first — it
  must print a finite energy and a forces shape of `(N, 3)`, otherwise
  the job is `failed(witness)` and is not run — then the job's `.sh` as
  above. No model is installed from an arbitrary repository and
  `models.json` is not edited.
- A job that fails on import or on a missing package is `failed(<class>)`
  with its log; the env is not repaired from here (`recipes/setup.md`
  owns the env recipe).
- Report: `python3 scripts/catbench_report.py --result ./result --out ./report --expect-models <approved list> --catbench-version <ver>` — the model list and version are the plan's, so an extra, missing or version-mismatched result stops the report instead of being averaged in.
- Official values, at request time:

  ```bash
  python3 scripts/catbench_leaderboard.py snapshot --dataset <leaderboard id or Zenodo stem> --out ./leaderboard
  python3 scripts/catbench_leaderboard.py compare --snapshot ./leaderboard --report ./report --result ./result \
      --models <M1,M2,...> --dataset <tag> --official-id <leaderboard id> --official-conditions ./official_conditions.json \
      --catbench-version <ver> --d3 <0|1> --calc-num <N>
  ```

  The snapshot records URL, HTTP code and UTC time, plus
  `leaderboard_id_source` (`exact` | `alias` | null, null with the
  reason `not_on_leaderboard`; the key is absent altogether when
  `meta.json` is unreachable, `meta_unreachable`) and
  `leaderboard_id_note`;
  the endpoints and their schema are the ones the script inspected on
  its stated date, and a changed schema, an unreachable endpoint or a
  dataset absent from the leaderboard is recorded as such — every row
  then reads "no comparable official value". `compare` separates what
  the machine read from what the operator declared, and verifies
  neither:
  - Every row of `comparison.json` carries `comparison: {status,
    basis[], verified: false, label}`; the markdown column is
    "comparable (status + basis; never verified)" and prints `label`
    verbatim (stdout is `comparison.md` followed by one newline). `status` is
    `comparable_by_name_match` (label begins "yes (basis: name-matched
    official id, equal reaction count, published D3 […]; dataset content
    identity not measured)"), `conditional_declared` (label
    "conditional (operator-declared, unverified: …)") or
    `not_comparable` (label "no"); the `comparable` / `conditional`
    booleans derive from it.
  - "yes" needs every machine-read signal: `--dataset <tag>` equal to
    the official id string (a name match — the hub never measures
    content identity), equal reaction count, and per-entry D3 published
    on the official JSON (a boolean field or a `_D3` name marker). The
    header then reads "Dataset identity: NAME_MATCH — … (content
    identity NOT measured by the hub)".
  - Any operator declaration makes the row "conditional", never "yes":
    `--official-id <id>` (identity status `declared`, header "Dataset
    identity: DECLARED — operator declared --official-id '<id>' …:
    recorded, NOT verified by the hub") and/or `--official-conditions
    <json>` (the file `{"<official entry name>": {"d3": <bool>,
    "source": "<citation/URL>"}}`; D3 `kind: "declared"`, source text
    "operator-declared via --official-conditions, citing: <source> (not
    verified by the hub)"). A cited evidence file is a declaration, not
    verification: with it the rows read conditional even when
    `--dataset` is the exact id. Nothing is ever "verified comparable"
    (`verified_comparable_rows: 0`, `comparison_basis_note`,
    `official_conditions_note` at the top level). An uncited,
    non-boolean or empty entry is `[stop] --official-conditions
    rejected: …`, exit 2.
  - Alias-only identity (`FG_dataset` -> `FG` through the README-count
    table) stays "no" with the discrepancy "dataset identity not
    evidenced: … README-count alias only; pass --official-id FG to
    record your declaration (rows then read 'conditional', never
    'yes')"; page-level `has_d3` alone stays "no" ("official D3
    condition not evidenced for '<name>' (page-level has_d3=… only;
    supply --official-conditions with a cited source — the row then
    reads 'conditional', never 'yes')").
  - Preconditions: pass `--official-id` only after establishing outside
    the hub that the local file is that page's dataset (for instance
    fetched with `--fetch <name>` from the Zenodo record the page
    cites); pass `--official-conditions` only with a citable source
    stating per-model D3. Both are recorded as declarations, so the
    write-up never calls such rows verified and never ranks them.

  `comparison.json` also records `dataset_identity`,
  `leaderboard_id_source`, `official_conditions_evidence` and per-row
  `official_d3 {value, source, kind}` (`kind` is `published` |
  `declared` | null). Exit codes: 0; 3 when the `result/` guard fails;
  2 when the evidence file is rejected.

## 5. Verify

- `report/mae_table.md` and `report/mae_table.csv` list every selected
  model and nothing else; `result/` holds only those models.
- VASP case: `catbench_vasp_stage.py verify` reported the originals
  unchanged.
- The record carries: `PLAN.md`; `catbench_version.json`; the hub
  commit; the dataset provenance (and the `catbench_datasets.py` output
  when a recommendation was made; `mapping_proposal.json`,
  `coeff_setting.json` and `originals.sha256` in the VASP case); the
  `jobs/` files and logs; `result/` and `report/`; the leaderboard
  snapshot and `comparison.{json,md}`.
- Comparison table: `comparison.md` as `catbench_leaderboard.py compare`
  wrote it — per model, the hub result next to the official value, every
  discrepancy string on the row (`<reason>; attempted [...]`, `model not
  listed on the official <id> page; attempted [...]`, `official fields
  not finite numbers: [...]`, `dataset identity not evidenced: ...`,
  `reaction count differs: hub N vs official M (subset)`, `hub
  conditions say D3 on, but the model name carries no _D3 suffix`,
  `official D3 condition not evidenced for '<name>' (...)`, `D3 differs:
  hub on/off vs official on/off (<source>)`), "no comparable official
  value" with the attempted URL and UTC time where nothing comparable
  exists. An unpublished official catbench version, calc_num or
  relaxation setting is not a discrepancy: it is a per-row
  "caveat: ..." in the same column and does not block "yes"; caveats
  always carry `dataset identity: <how>` and, when declared, `official
  D3: <source>`. The header's "Dataset identity" line and the
  `official_d3.source` of each row are cited in the record, since an
  asserted identity is the user's claim, not the hub's.
  Rows keep the user's model order; unlike results are never presented as
  a ranking, and the agent does not re-sort or re-total the table.

## 6. Out of scope

- Adding models, moving to another catbench version, re-staging with a
  changed mapping, rerunning any published model as a baseline, or deleting
  `result/` — renewed approval each.
- Running the missing DFT calculations — never part of this recipe.

## Planned helpers (not implemented)

None remaining for this recipe: the version rule, the dataset
recommendation and the owned `--check`/`--fetch`, the VASP staging, the
leaderboard helper, the `catbench_jobgen.py` CLI (`--catbench-version`,
`--calc-file`, `--python`, `--regenerate`) and the quickstart passthrough
(`--catbench-version`, `--regenerate`) have all landed and are named
above.

Landed without a unit test: none for this recipe. Every helper in the
chain table has a dedicated test file in the tree as of this revision —
a tree fact, not a statement of review or adequacy; the readiness test
re-derives this list from `tests/` on every run, and a helper whose test
file is later removed reappears here (its output then falls under the
`failed(untested:<helper stem>)` rule of `recipes/README.md`).

The command lines named above are exercised by their test files as of
this revision (`catbench_jobgen.py`: `--calc-file`, `--catbench-version`,
witness emission, rerun refusal; `catbench_quickstart.py`: the
`--catbench-version` passthrough, rerun reuse/refusal/`--regenerate`, a
stale job never executed; `catbench_datasets.py`: `--check`/`--fetch`
under a stub `catbench`, exit codes). That is a tree fact about test
presence, not a review verdict, and none of it has run against a real
`catbench` install on this host.

Missing executable parts (described by function; no owner file yet):

- a real-`catbench` run of `--check`/`--fetch` on this host (the tests
  use a stub package; the upstream download and its format have not been
  exercised here);
- changing the catbench version of an existing env in place has no
  executable path (only a fresh build with `OMM_CATBENCH_VERSION`).
