# Python API

`oh_my_mlip` is imported from the repository path; it is not a pip package.

```python
import sys; sys.path.insert(0, "<repo>")   # or $OH_MY_MLIP_HOME
import oh_my_mlip
```

Importing it never loads a framework, torch or a GPU.

## Registry

`list_models() -> list[str]`
: Registered frameworks.

`list_versions(model) -> list[str]`
: Versions of one framework.

`resolve(model, version=None, *, arch=None) -> dict`
: The lines that use a model, without loading it. `model` is a framework
  (`"MACE"`, resolving its default version) or a version (`"MACE-MH-1-OMAT"`).
  Main keys:

| Key | Meaning |
|---|---|
| `python` | interpreter of the model's env |
| `imports` | import lines |
| `inference` | line(s) that create `calc` |
| `env_run` | environment variables to set when running |
| `version` | the resolved version |
| `gated`, `license_url` | whether the weights need a license and token |
| `arch_pinned`, `arch` | NequIP/Allegro: compiled model for a GPU architecture |

## Inside a model's env

`get_calculator(model, version=None, device="cuda", apply_d3=False, *, arch=None)`
: Build the ASE calculator. Must run under that model's interpreter.

## Across envs

`run(model, atoms, properties=("energy", "forces"), device="cuda", apply_d3=False, *, version=None, arch=None) -> dict`
: Start the model's env process, compute once, stop it. Returns the results,
  e.g. `{"energy": ..., "forces": ...}`.

`Worker(model, version=None, device="cuda", apply_d3=False, arch=None)`
: One persistent env process. `start()`, `request(atoms, properties)`,
  `shutdown()`; usable as a context manager. `request` returns
  `{"id": ..., "ok": True, "results": {...}}` or `{"ok": False, "error": ...}`.

`WorkerPool(device="cuda", apply_d3=False)`
: One worker per model, started on first use and restarted once if it crashes.
  `request(model, atoms, properties, *, version=None)`, `shutdown()`; usable as
  a context manager.

Atoms are passed to worker processes as JSON; no pickle is used.
