# Recipe: single-point energy/forces or relaxation

Mechanics live in `AGENTS.md §3A` (`resolve` / `run` / `Worker`, arch-pinned
models, the `env_run` prefix) and `§4` (D3). This recipe only adds the job
protocol from `recipes/README.md` and names the files that own the run.

## Executable chain

| Step | Owner file | What it fixes |
|---|---|---|
| calculator code | `oh_my_mlip.resolve(<model>)` — the `import` and `inference` lines and the interpreter path come verbatim from `models.json` | no hand-written calculator construction, ever |
| one-shot single point | `<hub root>/run_examples/single_point.py <Model> --structure <file>`, launched by absolute path with `OH_MY_MLIP_HOME=<hub root>` exported, by any interpreter that imports `oh_my_mlip` (the launcher is ase-free; `run()` spawns the model env) | structure loading (any `ase.io.read` format), device, D3, `--arch`, the JSON verdict shape |
| one-shot relaxation | `<hub root>/run_examples/relax.py <Model> --structure <file>`, launched by absolute path with the interpreter `resolve(<Model>)` printed and `OH_MY_MLIP_HOME=<hub root>` exported — the launcher runs the ASE optimizer itself (it must import `ase`; the model env's interpreter does) and starts the model in a `Worker` subprocess of that same env. `--structure` takes any `ase.io.read` format; a missing or unreadable path exits 2 before any model starts and nothing is substituted. The script prints the input's resolved path, sha256, atom count and formula before the first force call (`tests/test_relax_input.py`) | optimizer, `--fmax`, `--steps`, D3; writes `relaxed.extxyz` in the current directory |
| legacy demo mode of `relax.py` | without `--structure`: cwd `POSCAR` if present (reported as `cwd POSCAR` with its sha256), else a built-in rattled Cu cell printed as `structure   : [demo] …` — a smoke test, never a job result | kept for compatibility only; a job always passes `--structure` |
| embedded use | the user's own script carrying the `resolve()` lines unmodified, executed with the interpreter `resolve()` returned (and its `env_run` prefix) | the user's pipeline stays theirs |
| verified-here record | `models.local.json` (`local_verified` entries written by `scripts/setup_verify.py`) | whether this model has a witnessed pass on this hub |

The model choice, the structure, the task and D3 are the user's inputs;
nothing else is decided at run time.

## 1. Ask

- **Model** — roster via `oh_my_mlip.list_models()` when none is given;
  MACE is the quickstart default. Version if the family has several.
- **Structure** file (anything `ase.io.read` handles).
- **Task** — single point, or relaxation with `fmax` and a step cap.
- **D3** on or off.
- **Where the calculator goes** — a one-shot number, or embedded in the
  user's own script (MD loop, custom pipeline).

## 2. Plan

State whether the env is materialized (`python3 scripts/setup_survey.py
--table <model>`); if not, `recipes/setup.md` comes first. Then the exact
file that will run:

- single point:
  `cd <work> && OH_MY_MLIP_HOME=<hub root> <python3> <hub root>/run_examples/single_point.py <Model> --structure <file> [--version V] [--d3] [--device cuda|cpu] [--arch <tag>] [--json]`
  — any `ase.io.read` format, an absolute path named on the command line
  and read inside the model's worker; `<python3>` is any interpreter
  that can import `oh_my_mlip` (the launcher is ase-free and starts the
  model's own env through `run()`), and the script is named by its
  absolute hub path for the same reason as the relaxation below;
- relaxation:
  `cd <work> && OH_MY_MLIP_HOME=<hub root> <interpreter> <hub root>/run_examples/relax.py <Model> --structure <file> --fmax <F> --steps <N> [--version V] [--d3] [--arch <tag>]`,
  where `<file>` is the user's structure in any `ase.io.read` format (an
  absolute path; the script resolves and hashes it), `<interpreter>` is
  the path `resolve(<Model>)` printed, and `<hub root>` is the absolute
  checkout being exercised (the candidate root under `--plugin-dir`, or
  the installed one) — never a bare `python3` and never a relative
  `run_examples/` path, since `<work>` is not the hub. `--structure` is
  always present in a job command: a wrong path exits 2 and relaxes
  nothing, and the no-argument demo mode is not a job. The plan quotes the
  interpreter path and the file's sha256 (the same value `relax.py` prints
  at start-up);
- embedded: the user's script, into which the `resolve()` import and
  inference lines are pasted unmodified, executed with the interpreter
  `resolve()` returns (and the `env_run` prefix if any). The plan quotes
  the lines and the interpreter path as `resolve()` printed them.

## 3. Approve

A single calculation on an already-verified model is approved by the
request itself. Anything that installs, downloads weights, compiles an
arch-pinned artifact, or runs long (a relaxation with a high step cap on a
large cell) is spelled out in the plan and confirmed first.

## 4. Execute

Run the file, from `<work>` for a relaxation. For the embedded case the
user's script is the artifact — keep it; a `python -c` invocation is never
the deliverable. A missing package or a failing import inside the env is
`failed(<class>)` and goes back to `recipes/setup.md`; it is not patched
from here. A `relax.py` run without `--structure` is the legacy demo
mode (cwd `POSCAR` or the built-in cell, labelled in its own output) and
is never filed as a result for the user's structure, whatever it
printed; an exit 2 from `--structure` is `failed(input)` with the
message it printed, and the command is not retried with a guessed path.

## 5. Verify

- Energy finite; forces shape `(N_atoms, 3)`; `device` as reported by the
  worker. Keep the `--json` output when it was requested; the record also
  names the model version, the structure path, D3 on/off, and the
  interpreter path used.
- If `resolve()` shows no `local_verified` entry for this host, say the
  model is unverified here rather than implying it was.
- A relaxation reports the final `fmax` reached and whether the step cap
  hit first. Its record cites the `structure   :` / `sha256      :` /
  `atoms       :` lines `relax.py` printed (the sha256 must equal the
  plan's) and `<work>/relaxed.extxyz`; the output must hold the atom
  count the `atoms` line reported, otherwise the run is `failed(input)`.
  A record whose stdout carries `structure   : [demo]` is the demo mode
  and is not a result.

## 6. Out of scope

Switching versions, adding D3 afterwards, relaxing further structures, or
changing the convergence target — restate and confirm before doing any of
them.

## Planned helpers (not implemented)

None remaining for this recipe. `run_examples/relax.py --structure`
landed with `tests/test_relax_input.py` (explicit input loaded and
reported, missing/unreadable input exits 2 with no substitute, demo mode
labelled); the test replaces `Worker` with a fake and runs no model.

Landed without a unit test: none for this recipe.

Missing executable parts (described by function; no owner file yet):

- the atom-count agreement between the reported input and
  `relaxed.extxyz` is checked by reading the output file, not by a
  script;
- `relax.py` has no `--json` output; the identity lines are read from
  captured stdout.
