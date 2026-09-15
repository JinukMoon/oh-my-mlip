# Recipe: fine-tune a registered model on your own data

Mechanics live in `AGENTS.md §3C` (convert → `ft_run` → `ft_verify`, exit
codes, licence notice) and `docs/finetune.md` (generated per-framework
audit: upstream command, dataset format, per-variant status matrix). This
recipe adds the job protocol from `recipes/README.md`, the executable
chain, the per-variant support audit, and the coverage rule for "all
supported models".

## Executable chain

The user's inputs are the variants, the dataset, and the values of the
settings each framework exposes (listed by `scripts/ft_run.py <Variant>
--show-settings` from `finetune/settings/<Framework>.json`), plus device,
split and seed. The framework-
specific conversion, command, checkpoint location and reload are owned by
files:

| Step | Owner file | What it fixes |
|---|---|---|
| upstream knowledge | `scripts/upstream_finetune.py` — per family: entrypoint, selector kind and path, dataset format, variant args, status, doc URL — ingested into the `finetune` block of `models.json`; `docs/finetune.md` rendered by `scripts/gen_finetune.py` (`--check` in CI) | what upstream's own training path is, with its source |
| dataset conversion | `scripts/ft_dataset.py --input <files> --to <format> --out <out>/data --split <f> --seed <n> [--energy-key K] [--force-key K]`, invoked by `ft_run.py`; an empty validation split with `--split` below 1.0 is refused (`--split 1.0` opts out explicitly) | the framework-native training files and the train/valid split |
| materialization | `scripts/ft_run.py <Variant> --dataset <data> --out <out>/<variant> --epochs <N> --batch-size <B> --device cuda --emit-only` (default `--out` is `./ft_<version>`) → `<out>/data/`, the patched `input.yaml`/`input.json` for config-key families, `finetune_<Variant>.sh` (`--slurm` adds `slurm_finetune_<Variant>.sh`, identical body, emission only — it says so on stderr and never submits), and `<out>/ft_run.json` (schema `ft_run.json/1`: family, version, env, python, absolute dataset, out, epochs, batch_size, device, split, seed, `seed_requested`, `seed_control {scope, basis}`, `designated_checkpoint` (one glob, `{version}` substituted), `artifacts {sh, slurm|null, config, extra_files[], conversion}`, n_train, n_valid, elements, `command` (the exec'd argv), `pre_steps` (in-env prestage argvs), `rematerialize` (the exact `ft_run.py` argv plus `--emit-only` that re-creates the run after a cache cleanup; hub-owned absolute inputs come back from `install.sh`/prestage, not from the `.sh`)). SevenNet emit writes `sevennet_patch.json` + `sevennet_prestage.py` and materializes `input.yaml` only inside the env, so `--emit-only` no longer needs the SevenNet env | the exact command, written before anything runs |
| seed control | `--seed <n>` on `ft_run.py`; `SEED_CONTROL` in `scripts/ft_run.py` records per family how far the seed reaches in that family's own trainer (`scope`, with the installed-source `basis` — source inspection, not a runtime determinism proof). Native: MACE (`mace_run_train --seed`), SevenNet (`train.random_seed` patched over the installed preset, which ships 1 or 777), DeePMD and DPA4 (`training.seed`, data sampling; model-init seeds come from the checkpoint's own model section), GRACE (top-level `seed`, also names the `seed/<seed>/` output dir), PET (top-level `seed`), MatterSim (`--seed`), TACE (`misc.global_seed` + `dataset.split_seed`), CHGNet (the generated driver seeds random/numpy/torch and passes `Trainer(torch_seed=, data_seed=)`). Data-split-only: NequIP and Allegro (`data.seed` only; the training seed is hard-coded `seed_everything(123)` in `nequip/utils/global_state.py:79` of the pinned nequip 0.15.0). No family has scope `none`. Without `--seed` seed 0 is used and recorded as `seed_requested: false`; an explicit `--seed` on a data-split-only family is refused with exit 5 before any file is written, stderr naming the scope, `global_state.py:79` and `--allow-partial-seed`, which lets the run proceed with the note still printed and `scope: "data-split-only"` in `ft_run.json`. Native families never refuse. The sweep passes no `--seed`, so it never reaches exit 5 | what a seed does and does not fix per family |
| execution | `bash <out>/<variant>/finetune_<Variant>.sh` — `set -eu`, `cd` to `<out>`, one `export` per `env_run` key, `exec` of the variant's env interpreter | the run the user can repeat |
| dependencies | the variant's env, built from `envs/<env>.yml` (plus `CATBENCH_PIN`) by `install.sh`; nothing else is available to a fine-tune | the package set |
| verification | `scripts/ft_verify.py <ckpt> --model <Variant> [--version <V>] --device cuda --json` — reload the checkpoint with the framework's own loader; the verdict is pass iff `reason` is empty, checked in order: finite energy (`non_finite_energy`), the probe's `[4,3]` forces shape (`bad_forces_shape`), `forces_finite` (`non_finite_forces`), a backend execution record with a non-negative compute-op count (`witness_record_missing`), `gpu_used` agreeing with that count (`witness_inconsistent`), and on `--device cuda` at least one GPU compute op (`gpu_not_used`). An unknown family or version is rc 2 (`RegistryError`). A bare family name in `--model` resolves the family default; the sweep passes the variant name | pass / fail per checkpoint |
| native builders | the `BUILDERS` table of `scripts/ft_run.py`, alphabetically Allegro · CHGNet · DeePMD · DPA4 · GRACE · MACE · MatterSim · NequIP · PET · SevenNet · TACE — one command per family taken from that package's installed sources, each cited in `scripts/upstream_finetune.py` (`note`) and shown in `docs/finetune.md`; nothing renders a command for a family absent from the table. Such a family, if it passes the registry gate, is refused with exit 4 before any dataset conversion or file write (`EXIT_NO_BUILDER`, "no real command builder … implementation gap, not an upstream limitation"); nothing is emitted, `--emit-only` included. Today the registry status already stops each absent family earlier (exit 2 or exit 3), which makes exit 4 a guard, not an expected path | which families have a runnable command at all |
| dataset formats | `scripts/ft_dataset.py --to <format>`: `chgnet` is an alias of extxyz (the CHGNet driver converts to pymatgen Structures in memory); `orb`, `aselmdb`/`nequix` and `alphanet` are documented but not implemented and exit 2 from `ft_dataset.py` | which training-file formats can be produced at all |
| prestage line | some emitted `finetune_<Variant>.sh` (SevenNet, NequIP/Allegro, TACE) run one prestage command before `exec`, under the same `set -eu`; still one rerunnable file, and a failing prestage aborts the run | what runs before the training process starts |
| exit codes of `ft_run.py` | 1 usage or unknown model; 2 refused by registry status; 3 not runnable as installed (remaining blockers named); 4 no real builder in this hub; 5 explicit `--seed` the family cannot honour in full (`EXIT_SEED_UNHONOURED`, before any write; see the seed-control row); any other value is the training process's own return code. `ft_sweep.py` has no exit-5 class because it passes no `--seed` | how a refusal is told apart from a training failure |
| loader roster | `scripts/ft_verify.py --list-loaders --json` — one JSON line `{"loaders": [...], "source": ...}` naming the families `ft_verify.py` can reload; `ft_sweep.py` reads it before classifying | which checkpoints can be verified at all |
| per-env sweep | `scripts/ft_sweep.py --env <env> --dataset <frames> --ledger <ledger.jsonl> [--audit <audit.json>] [--epochs N] [--device cuda|cpu] [--out-root DIR] [--campaign-id ID --manifest-sha256 SHA]` — classifies every variant of the env from its `finetune.status`, runs the real chain for the attemptable ones, one JSONL row per phase, one terminal state per variant | "every supported model" for one env, serialized, ledger-first |
| re-audit artifact | the `--audit` JSON `{variant: {supported: bool, citation: str}}` for the code-excavation and not-supported candidates; a candidate marked supported is attempted and joins the required set, a code-excavation variant with no entry is `failed(audit_missing)`; `scripts/evidence_report.py --audit` reads the same file | the per-campaign support audit, with citations |

## 1. Ask

- **Variants** — by version name. "Every supported model" expands to the
  variant list built in the audit below; families are always shown as their
  named variants, never as a count.
- **Dataset** — a path `ase.io.read` handles, and whether energies/forces
  sit on a calculator or under `info`/`arrays` keys (`--energy-key` /
  `--force-key` of `scripts/ft_dataset.py`).
- **Purpose** — (a) a real fine-tune; (b) a pipeline proof on small data
  with a short run — the default when the question is "does fine-tuning
  work for X".
- **Settings** — from the variant's `--show-settings` table: training length
  (epochs, or steps for DeePMD), batch size, learning rate, energy / force /
  stress weights and stress on or off. Present these defaults with their origin
  (official fine-tuning value or upstream default), let the user keep or replace
  them, and run with what the user chose. A setting with neither value must be
  asked.
- **Output directory.**

## 2. Plan

### Support audit per variant — before any command

For each requested variant read its `finetune` block in `models.json` (the
ingested audit from `scripts/upstream_finetune.py`) and the matching
section of `docs/finetune.md`, then classify:

| `finetune` block says | Classification |
|---|---|
| status `documented` (any flavour), `runnable_as_installed: true` | required to train |
| status `documented`, `runnable_as_installed: false` | required after the named blocker is fixed; `ft_run.py` exits 3 and names it. The fix is a reviewed change to `envs/<env>.yml` (or the sidecar) followed by a rebuild through `install.sh` — its own plan step with its own approval; `ft_run.py` re-checks import-type blockers live, so no ingestion rerun is needed afterwards |
| status `code-excavation-needed` | decided by this campaign's audit file (`--audit`, format in the chain table): an entry `supported: false` with a citation → `unsupported`; `supported: true` → attempted and part of the required set; no entry → `failed(audit_missing)` — the variant is not attempted and the state is recorded as such, in a sweep and in a single run alike |
| status `not-supported` | `unsupported` with the citation from `models.json` or the audit file; without any citation the state is `failed(uncited_unsupported)` |
| status `documented`, `runnable_as_installed: true`, family not in `BUILDERS` | `failed(no_builder)` — `ft_run.py` exits 4 before writing anything; listed in the plan as such, not attempted, and never `unsupported` (the gap is this hub's, not upstream's) |
| `gated: true` and no token reachable | `access-blocked` until the licence is accepted and a token is available |
| `finetune.licence` set (the licence-gated checkpoints of `AGENTS.md §3C` item 4) | trains anyway; the licence name and URL that `ft_run.py` prints are relayed to the user, and a derivative checkpoint inherits those terms |

`PLAN.md` records the required set as an explicit variant list, and the
excluded variants each with their citation. Only the current `models.json`
status counts; a status remembered from another campaign or an older
document is not carried forward. A variant whose upstream training path
has since been found is re-audited by updating
`scripts/upstream_finetune.py` with the new source cited — a separate
reviewed change, not a run-time decision.

Builder honesty: only the eleven `BUILDERS` families have a command built
from the installed package's sources. A variant whose registry status
says "go" but whose family has no builder is refused by `ft_run.py` with
exit 4 before any file is written — there is no generic rendering, no
stub to inspect, and nothing to emit with `--emit-only`. The sweep
records it as `failed(no_builder)` (an implementation gap of this hub,
not an upstream limitation), a single run reports the exit code the same
way, and the plan lists such a variant under `failed(no_builder)` in
advance rather than as a run to attempt. As the registry stands, each
family without a builder is turned away one step earlier: AlphaNet, EqV3,
Eqnorm and MatRIS by exit 2 (`not-supported` / `code-excavation-needed`),
UMA, ORB, Nequix, EquFlash and fairchemv1 by exit 3 (documented, not
runnable as installed). Exit 4 therefore protects against a future
registry edit rather than describing a current path. Of all these, only
Eqnorm and MatRIS carry a cited upstream absence in `models.json`; every
other stop is a blocker of this hub's installation — including
`UMA-s-1p2-OC22`, whose `oc22` task missing from the installed
`fairchem-core` 2.19.1 `UMATask` enum is a version-compatibility blocker
(exit 3 like the rest of UMA), not an upstream exclusion.

Eleven builders is the implementation coverage of this hub today, not
the fulfilment of the fine-tuning requirement. The exit-3 families are
ones upstream documents as fine-tunable (repo-only configs, gated
checkpoints, an unshipped `finetune.py`, a missing dataset writer) — they
are pending implementation here, and the plan records them as
`failed(not_runnable_as_installed)` gaps of this hub, never as
`unsupported`, which is reserved for a cited upstream absence. The same
holds for the dataset writers (`orb`, `aselmdb`/`nequix`, `alphanet`).
A family leaves any of these states only through a reviewed change to
`scripts/ft_run.py` (for exit 3, together with `envs/<env>.yml` or the
writer in `scripts/ft_dataset.py`) compared against block A of
`docs/finetune.md` when it is written, not at run time.

### Commands — copied into `PLAN.md` exactly

Named variants, one at a time:

```bash
python3 scripts/ft_run.py <Variant> --dataset <data> --out <out>/<variant> --epochs <N> --batch-size <B> --device cuda --split <f> --seed <n> --emit-only
bash <out>/<variant>/finetune_<Variant>.sh        # or omit --emit-only and let ft_run.py run it
python3 scripts/ft_verify.py <ckpt> --model <Variant> --device cuda --json
```

"Every supported model" of an env — the sweep owns the loop and the
ledger; the audit file is written first (one entry per code-excavation or
not-supported candidate, each with its citation) and quoted in the plan:

```bash
python3 scripts/ft_sweep.py --env <env> --dataset <data> --ledger <work>/ft_<env>.jsonl --audit <work>/ft_audit.json --epochs <N> --device cuda
```

Resource bound: GPU memory and time depend on the variant; state a bound
per variant. When a large model needs a reduced batch size or shorter
schedule to fit, that setting is disclosed in the plan, not discovered
after the fact.

## 3. Approve

Present the variant table (required / unsupported with citation / blocked),
the dataset summary (frames, species, forces present or not), the exact
commands, and the licence notices. Wait for approval. Several variants run
one after another, never concurrently.

## 4. Execute

Run the variants in the approved order, capturing stdout and stderr under
`<out>/<variant>/`. `ft_run.py` writes every artifact first (converted
dataset, patched config, the `.sh`); `ft_verify.py` takes the checkpoint
path the run printed. The emitted config is not edited to "make it work":
a failure is recorded as `failed` with its error class and either fixed at
the source (`scripts/upstream_finetune.py` / `models.json` for the
command, `envs/<env>.yml` for a dependency — separate reviewed changes)
or left as failed. A deliberate hyperparameter edit of the emitted config
for a production run is allowed; the edited file's path and sha256 go into
the record and the rerun starts from it.

## 5. Verify

One row per variant: variant, state, evidence (`ft_verify.py --json`
output — pass iff energy and forces from the reloaded checkpoint are
finite, with the `device` it ran on and, for a cuda witness, the measured
`gpu_used`; the `.sh` path; the config path; the log path; the checkpoint
path; `<out>/ft_run.json` with its `seed`, `seed_requested` and
`seed_control.scope`, so a data-split-only seed is never reported as a
fully seeded run). For a sweep the rows are the ledger's terminal rows, one state
each from `passed` / `unsupported` (cited) / `access-blocked` /
`resource-blocked` / `failed(<class>)`, rendered as written — an uncited
`unsupported` is already `failed(uncited_unsupported)` in the ledger, a
family without a builder is `failed(no_builder)`, and a cuda run whose
witness could not measure the GPU is `failed(ft_verify:no_gpu_witness)`.
`scripts/evidence_report.py --strict` accepts a fine-tune `passed` row
only when its device is cuda and `gpu_witness` is true, and rejects a
ledger whose rows carry mixed `manifest_sha256` values. A checkpoint on
disk without a passing `ft_verify.py` is not a pass. The record also carries the hub commit, the dataset path with its
sha256 and the split/seed used, the env interpreter path, and the upstream
doc URL from the variant's `finetune` block. A rerun from the same `.sh`
reproduces the procedure; the trained weights are not expected to match
bitwise. Small-data runs prove the pipeline (train → save → reload →
infer), not an accuracy gain — say so in the report.
`python3 scripts/gen_finetune.py --check` confirms `docs/finetune.md`
still matches `models.json`; regenerating with `--write` is its own
reviewed change.

## 6. Out of scope

Longer schedules, more variants, hyperparameter search, package changes
inside an env, or writing a command for a family that `ft_run.py` refuses
with exit 4 — renewed approval or a separate reviewed change each.

## Planned helpers (not implemented)

Design targets from the consensus plan; none exist yet (the per-variant
sweep has landed and is named above):

- `scripts/ft_run.py --ledger` — one ledger row per single run outside a
  sweep.
- an executed column in `scripts/gen_finetune.py` rendered from a campaign
  ledger (today `demonstrated` is metadata maintained in `models.json`).

Landed without a unit test: `scripts/upstream_finetune.py` — the
research table; `tests/test_gen_finetune.py` checks the rendering from
`models.json`, not the table's ingestion.

Missing executable parts (described by function; no owner file yet):

- builders for the families outside `BUILDERS` (UMA, ORB, Nequix,
  EquFlash, fairchemv1, AlphaNet, EqV3, Eqnorm, MatRIS); until one lands
  (a reviewed change to `scripts/ft_run.py`, plus the env or registry
  change that lifts the earlier exit 2 or exit 3 stop), that family is refused before any
  file is written and recorded with the refusing exit code, never
  `passed`, and nothing is emitted for it;
- dataset writers for `orb`, `aselmdb`/`nequix` and `alphanet`
  (`ft_dataset.py` exits 2 for them today);
- an executable comparison of a builder's emitted command against block A
  of `docs/finetune.md`, for whoever promotes a family into `BUILDERS`;
- capturing the env's installed package inventory next to the checkpoint.
