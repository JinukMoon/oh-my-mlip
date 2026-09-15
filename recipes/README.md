# recipes/ — end-to-end workflow recipes shared by every host

A recipe turns a natural-language request ("benchmark my adsorption data",
"fine-tune SevenNet on these frames", "distill MACE into a LAMMPS student")
into a scoped plan, an explicit approval, an execution made of rerunnable
user files, and a verification record. One recipe per workflow; the same
text serves Claude Code (reached through `skills/<name>/SKILL.md`), Codex
(reached through `AGENTS.md §10`), and a human reading it directly.

These are workflow protocols. `docs/recipes.md` is a different, GENERATED
file (the per-env environment-recipe listing) — do not confuse the two.

| Request kind | Recipe | Mechanics it layers on |
|---|---|---|
| install / set up one, several or all model envs | `recipes/setup.md` | `AGENTS.md §9` (with §5, §6, §8) |
| single-point energy/forces or relaxation | `recipes/run.md` | `AGENTS.md §3A` (with §4) |
| catbench adsorption benchmark | `recipes/catbench.md` | `AGENTS.md §3B` |
| fine-tune from a foundation checkpoint | `recipes/finetune.md` | `AGENTS.md §3C` |
| distill a teacher into a LAMMPS student | `recipes/distill.md` | `AGENTS.md §3D` |

## Reading rule — on demand, one recipe at a time

Load only the recipe for the request at hand, after the `AGENTS.md` section
it names. Do not preload all five. No efficiency figure is claimed for this
split; none has been measured.

## Executable source of truth — the rule every recipe is built on

The language model's job in a recipe is to pick the recipe and its
scientific inputs: which models, which dataset, which coefficients, which
targets, which budget. Everything that turns those inputs into an outcome
is owned by a versioned file in this repo (or in a sibling repo the recipe
names) and runs exactly as written:

- **Installation** is `install.sh` reading `envs/<env>.yml` (one conda
  solve with the primary packages pinned) or `envs/<env>.build.sh` (an
  ordered multi-pass build), then its own fixed post-steps. The recipe
  never lists packages of its own. What those pins cover is stated
  exactly: python, torch with its CUDA-runtime tag, the framework release
  or git SHA, and the packages each recipe names are fixed; transitive
  dependencies the recipe does not name resolve at build time (several
  recipes say so in their own comments), and no lock file of the solved
  closure exists yet. That is an open gap against the requirement of
  persisted, tested dependencies — not the accepted end state: until a
  lock or an installed-inventory capture exists, the evidence records the
  recipe file's sha256 and the pin values, and every report names the
  floating remainder as a gap rather than presenting the pins as the
  whole dependency set.
- **Execution** is an emitted file — a `.sh` runner, a `.py` job, a config —
  produced by a repo script from the approved inputs, kept on disk, and
  rerunnable without the agent.
- **Evidence** is what those files and scripts print or write: JSON
  verdicts, JSONL ledgers, logs, checkpoints, the hub's git commit, the
  pinned versions the plan named, the seeds and inputs. A rerun reuses the
  same recipe, the same pins and the same files; it reproduces the
  procedure, not the bitwise numerical result of a stochastic training or
  MD run.

Consequences the recipes enforce:

1. **No improvised package operations.** If a build or run fails for a
   reason the owner files do not already handle, the item ends as
   `failed(<class>)` with its log. The fix — a new pin, another pass, a
   different flag — is a reviewed change to the owner file, followed by a
   rebuild through the same chain. Nothing is installed, upgraded or edited
   inside an env by hand to make a verdict pass.
2. **Host and architecture conditions are branches inside the owner
   files** (driver below the recipe's CUDA runtime, `nvcc` absent, an
   arch-pinned compiled artifact, a gated download). The recipe names the
   branch and the file that takes it; the agent does not invent a fourth
   option at run time.
3. **Upstream research is authoring time, not run time.** Reading an
   upstream README to decide how a framework is installed or trained is
   what `scripts/upstream_recipes.py` and `scripts/upstream_finetune.py`
   already did; their result is in `models.json` and the generated docs.
   A routine run consults those, and a discovered gap becomes an update to
   them, with the source cited.
4. **Generated run files stay user-editable.** A rendered `config.yaml`,
   `finetune_<V>.sh`, `jobs/run_catbench_<M>.sh` is the user's file after
   emission; a deliberate edit for a production run is recorded (path and
   hash) in the plan, and a rerun starts from that edited file.

Each recipe opens with an **Executable chain** table naming, per step, the
owner file and what it fixes. Prose after that table explains and invokes;
it never substitutes.

## The six stages every recipe follows

1. **Ask** — collect only the inputs the recipe lists as required. An input
   the user already supplied is not asked for again.
2. **Plan** — write the plan as a file in the work directory (`PLAN.md`
   unless the user names another): inputs, the exact commands that will run,
   versions and pins, resource bounds, expected artifacts, and what is out of
   scope. The commands in the plan are the commands that run — never a
   hidden alternative path.
3. **Approve** — present the plan and wait for an explicit approval; use the
   host's interactive question UI when it has one. Approval covers the
   disclosed scope only.
4. **Execute** — run the approved commands. Every executed action exists
   first as a file the user can rerun without the agent (`.sh`, `.py`,
   config), never as an inline `python -c`.
5. **Verify** — cite evidence: JSON verdicts, ledger rows, artifact paths.
   Render tool verdicts as printed; never re-judge them or run a second,
   different check in their place.
6. **Out of scope** — anything beyond the approved plan (more models, a
   version bump, a rerun with changed settings, deleting anything) needs a
   renewed approval. A permission the host denied is never obtained by
   another route (a different shell, another session, a peer agent).

## Result states

Every required item ends in exactly one state:

| State | Meaning |
|---|---|
| `passed` | the recipe's oracle returned pass, evidence path recorded |
| `unsupported` | upstream provides no such capability — valid ONLY with a citation (upstream URL or `file:line`) |
| `access-blocked` | token, licence or gated-download condition unmet |
| `resource-blocked` | disk, GPU memory, driver level or time budget insufficient |
| `failed` | an error occurred, or an acceptance target was not met; the class is named (`failed(<class>)`; an oracle's `unmet(<class>)` is reported here as `failed(unmet:<class>)`, artifacts kept) |

A blocked or unsupported item is never reported as passed; an emitted
script or config is never a pass on its own; an item absent from the
record is reported as not attempted, never dropped.

## Implemented, landed-without-a-test, planned — read from the working tree

Each recipe ends with a section headed **Planned helpers (not implemented)**.
Names listed there are design targets taken from the consensus plan; they do
not exist in this repo and must not be invoked or promised as working.
Everything above that heading exists today, and every flag the recipe
advertises on a script is one that script's argument parser really adds.
`tests/test_recipe_contract.py` enforces both directions from the working
tree: an advertised path that disappears fails the test; a planned helper
that appears on disk, or a planned flag that the parser starts adding,
fails it too, forcing the recipe to promote it.

Existing is not verified. The same section names, under **Landed without
a unit test**, every helper the recipe invokes that has no dedicated test
under `tests/` yet; the list is checked against the tree in both
directions (a helper gaining a test must leave it). Such a helper is
invoked as its `--help` describes — there is no better-owned alternative —
but its output is never filed as a scientific `passed`. A job whose
verify step rests on an untested helper is recorded
`failed(untested:<helper stem>)` with every artifact kept; it becomes
`passed` only when the helper's test has landed, an independent review
has read it, and the verify step is rerun from the kept artifacts.

The same section also lists **missing executable parts**: steps that no
owner file covers yet, described by function without inventing a script
name, so that a later owner can pick them up. A step with no owner is not
improvised at run time. Where the recipe carries a verbatim file body for
it, the agent writes that body byte-for-byte (its sha256 goes in the
record) and runs it; where it carries none, the step is not attempted and
the item ends as `failed(no_owner)` with the function named. Nothing
between those two options — a freshly composed script, an inline
`python -c` — is a job action.

## Candidate-local discovery — testing a checkout that is not the installed plugin

A candidate hub (a working-tree snapshot materialized by
`scripts/fresh_root.py`, or any checkout under review) is exercised by
pointing each host at that directory, not by trusting whatever plugin the
host already has installed:

- Claude Code: `claude --plugin-dir <candidate root>` loads that root's
  `.claude-plugin/` and `skills/` for the session (session-local; nothing
  is installed or replaced), with `OH_MY_MLIP_HOME=<candidate root>` set in
  the environment so the skills' `python3 scripts/...` lines and the
  registry resolve inside the candidate.
- Codex, and any agent that reads `AGENTS.md`: start it with the candidate
  root as the working directory; it discovers `AGENTS.md` from its cwd.

The record of such a session names the loading method and the root used.
A session started without either is running the host's default-installed
plugin, which may be older than the candidate; its results are not
evidence about the candidate and are not reported as such.

## Public-framing rule

Recipes state technical requirements — a CUDA-capable NVIDIA GPU, a host
driver at or above the CUDA runtime of the env's torch build, Linux or WSL —
and never hardware product names, measured numbers, host counts, or
claims that tie validation to one named machine. Validation evidence lives in the
generated status docs and in job ledgers, not in recipe text.
