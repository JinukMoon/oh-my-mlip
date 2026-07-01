# Teacher-query on-ramp for downstream distillation (Phase 2)

> **This is a Phase-2 roadmap note, not a v1 feature.** `oh-my-mlip` v1 does
> **not** depend on any distillation tool. This document only states the *stable
> teacher-provider contract* — a way to query a teacher MLIP for energies/forces
> on many structures cheaply — so the contract can be frozen now and a future,
> *separate* distillation tool can bind to it later. If you just want to run
> MLIPs, you never need this file.

A common downstream use of a fast, accurate teacher MLIP is **distillation**:
label a set of structures with the teacher, then train a small
**CPU-deployable NN-MTP** (neural-network moment-tensor potential) **student**
that runs in **LAMMPS** without Python/LibTorch at MD time. The expensive part of
generating that teacher-labeled training set is process + model startup, so a
labeling workflow should bind to a **long-lived worker**, not pay
subprocess-per-call startup. `oh-my-mlip` exposes exactly that through two stable
surfaces from `oh_my_mlip`.

> The downstream distillation tool (teacher-labeled set → NN-MTP student →
> CPU/LAMMPS deployment) ships **separately** on its own timeline; `oh-my-mlip`
> only provides the teacher-query surface below.

## The two surfaces a downstream tool binds to

### 1. `get_calculator(...)` — INTRA-ENV (canonical signature: AGENTS.md §2)

Returns a ready-to-use ASE `Calculator`.

**Precondition (hard):** it is called from *inside* the model's own conda-env
interpreter (`<env>/bin/python`), because it imports that framework
(`sevenn` / `mace` / `nequip` / `fairchem` / ...). A teacher *worker* process —
one per teacher env — imports this directly:

```python
# running under e.g. $OH_MY_MLIP_HOME/envs/mace/bin/python
from oh_my_mlip import get_calculator
calc = get_calculator("MACE", "MACE-MPA-0", device="cuda", apply_d3=False)
energy = atoms_with(calc).get_potential_energy()
```

This is the thinnest possible binding: the caller owns the calculator object and
calls it in-process at full speed.

### 2. Persistent `Worker` / `WorkerPool` (JSONL wire contract) — CROSS-ENV

When the caller runs in a *different* interpreter than the teacher (the normal
case — the caller is one env, each teacher is another), it talks to a persistent
worker over the **frozen JSONL wire contract**. One worker == one env == one
model. This is what makes **bulk teacher labeling** cheap: interpreter + model
load is paid once, then many structures stream through.

```python
from oh_my_mlip import WorkerPool

pool = WorkerPool(device="cuda")          # lazily spawns one worker per model
for atoms in structures_to_label:         # bulk teacher labeling
    resp = pool.request("MACE", atoms, properties=("energy", "forces"),
                        version="MACE-MPA-0")
    if resp["ok"]:
        e = resp["results"]["energy"]; f = resp["results"]["forces"]
pool.shutdown()
```

`WorkerPool` amortizes interpreter + model load across the whole run (no
subprocess-per-call latency) and routes responses by `id`, so it is safe under
concurrent ids.

## Frozen JSONL wire contract

One process per env, launched as
`<env>/bin/python -m oh_my_mlip._worker --model <M> [--version <V>] [--device cuda] [--apply-d3]`,
speaking line-delimited JSON on stdin/stdout. The parsed, allow-listed `env_run`
from `models.json` is applied as the subprocess **environment** (never
shell-interpolated).

| Phase | Direction | Message |
| --- | --- | --- |
| ready-handshake | worker → supervisor | `{"ready": true, "model": "...", "version": "..."}` (or `{"ready": false, "error": "..."}` then non-zero exit) |
| request | supervisor → worker | `{"id": <any>, "atoms": <ase Atoms.todict() json>, "properties": ["energy","forces","stress"]}` |
| response | worker → supervisor | `{"id": <same id>, "ok": true, "results": {"energy": float, "forces": [[...]], "stress": [...]}}` |
| per-request failure | worker → supervisor | `{"id": <same id>, "ok": false, "error": "<repr>"}` (worker stays alive) |
| shutdown | supervisor → worker | EOF on stdin **or** `{"shutdown": true}` → clean exit 0 |

**Guarantees**

- Responses **carry the request `id`**; ordering is **not** assumed FIFO. The
  supervisor routes by `id`.
- A per-request exception surfaces as `ok: false` and does **not** kill the
  worker.
- A worker crash mid-request surfaces as
  `{"id": <inflight>, "ok": false, "error": "worker crashed"}`; `WorkerPool`
  respawns and retries once.
- The atoms payload is pure JSON via `Atoms.todict()` / `Atoms.fromdict()` — no
  pickle, no `eval`.

## Status

The contract/format is **frozen here** (v1, proven on MACE + SevenNet). The
binding acceptance test — **100 single-point calls against one long-lived MACE
worker returning 100 results with no respawn** — is **deferred to the compute
checkpoint** (it needs a real GPU model env). Until that test passes, the README
describes this as a *planned* on-ramp ("teacher-query on-ramp planned (Phase 2)");
once it passes the wording becomes "ready". The routing/protocol logic itself is
unit-tested with a mocked worker today (`tests/test_registry.py`).
