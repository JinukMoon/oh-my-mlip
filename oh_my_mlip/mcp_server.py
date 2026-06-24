"""oh_my_mlip.mcp_server — a thin MCP adapter over the public oh_my_mlip API.

This module exposes the existing tiered teacher-provider interface
(``list_models`` / ``resolve`` / ``run`` / ``Worker`` / ``fetch_env``) as
Model-Context-Protocol tools so a tool-calling agent can drive the hub the same
way a human follows ``AGENTS.md``. It REIMPLEMENTS NOTHING — every tool is a
small wrapper that validates its arguments and forwards to the real function.

Two classes of tool:

  * GPU-free (work right now, pure registry reads, no env/torch/GPU needed):
      ``list_models``, ``describe_model``, ``model_status``.
  * compute-dependent (need a materialized conda env + GPU at runtime; they are
      validated end-to-end at the compute checkpoint):
      ``run_singlepoint``, ``run_relax``, ``run_catbench``, ``install_model``.

The ``mcp`` SDK is imported LAZILY (see ``_require_mcp``) so that importing
``oh_my_mlip`` never requires ``mcp`` to be installed — mirroring how torch /
ase / huggingface_hub are guarded elsewhere in this package. Building the server
(``build_server``) or running it (``main``) DOES require ``mcp``; install it with
``pip install -r requirements-mcp.txt``.

Launch the server over stdio::

    python -m oh_my_mlip.mcp_server

or point your MCP client at that command.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oh_my_mlip import registry

__all__ = ["build_server", "main", "TOOL_NAMES", "GPU_FREE_TOOLS"]

# The exact set of tools the server registers (kept here so tests can assert the
# surface without importing mcp). Order matches AGENTS.md's section -> tool map.
TOOL_NAMES: tuple[str, ...] = (
    "list_models",
    "describe_model",
    "model_status",
    "run_singlepoint",
    "run_relax",
    "install_model",
    "run_catbench",
)

# Tools that need no GPU / conda env / heavy deps — they answer from the registry.
GPU_FREE_TOOLS: frozenset[str] = frozenset(
    {"list_models", "describe_model", "model_status"}
)


# ── lazy mcp guard ────────────────────────────────────────────────────────────
def _require_mcp():
    """Import and return ``FastMCP`` lazily.

    Importing ``oh_my_mlip`` must never require ``mcp``; only constructing /
    running the server does. Raise a clear, actionable error if it is absent.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised via test skip
        raise ImportError(
            "the MCP server requires the 'mcp' package, which is an optional "
            "extra. Install it with:  pip install -r requirements-mcp.txt"
        ) from exc
    return FastMCP


# ── structure -> ase.Atoms conversion (run_singlepoint / run_relax input) ─────
def structure_to_atoms(structure: Any):
    """Convert a tool ``structure`` argument into an ``ase.Atoms``.

    Accepts any of:
      * a path (str) to an ASE-readable structure file (POSCAR, .cif, .xyz, ...);
      * a full ``Atoms.todict()`` dict (round-trips losslessly);
      * a simple dict ``{"symbols", "positions", [cell], [pbc]}`` describing a
        cell of atoms.

    ``ase`` is imported lazily so this module imports without it; the conversion
    itself needs ase (it is present in every model env and in the supervisor's
    runtime). Raises ``ValueError`` with an actionable message on bad input.
    """
    if isinstance(structure, str):
        from ase.io import read

        path = Path(structure).expanduser()
        if not path.exists():
            raise ValueError(f"structure path does not exist: {structure!r}")
        return read(str(path))

    if not isinstance(structure, dict):
        raise ValueError(
            "structure must be a file path (str) or a dict "
            "(Atoms.todict() or {symbols, positions, cell?, pbc?}); "
            f"got {type(structure).__name__}"
        )

    import numpy as np
    from ase import Atoms

    # Full Atoms.todict() round-trip: it carries 'numbers' (+ positions/cell/pbc).
    if "numbers" in structure and "symbols" not in structure:
        from oh_my_mlip._worker import decode_atoms

        return decode_atoms(structure)

    # Simple {symbols, positions, cell?, pbc?} form.
    if "symbols" not in structure or "positions" not in structure:
        raise ValueError(
            "structure dict must contain 'symbols' and 'positions' "
            "(or be a full Atoms.todict() with 'numbers')"
        )
    kwargs: dict[str, Any] = {
        "symbols": structure["symbols"],
        "positions": np.asarray(structure["positions"], dtype=float),
    }
    if structure.get("cell") is not None:
        kwargs["cell"] = np.asarray(structure["cell"], dtype=float)
    if structure.get("pbc") is not None:
        kwargs["pbc"] = structure["pbc"]
    return Atoms(**kwargs)


# ── graceful "env not installed" guard for compute tools ─────────────────────
def _env_hint(model: str, version: str | None) -> str:
    """Return an actionable hint pointing at install_model / install.sh."""
    try:
        spec = registry.resolve(model, version=version)
        env = spec["env"]
    except Exception:  # noqa: BLE001 - resolution errors handled by caller
        env = "<env>"
    return (
        f"the conda env {env!r} for {model} is not materialized yet. Install it "
        f"first via the install_model tool (model={model!r}), or build it "
        f'locally with:  bash "$OH_MY_MLIP_HOME/install.sh" {env}'
    )


def _env_python_exists(model: str, version: str | None) -> bool:
    """True if the model's env interpreter exists on disk (env is materialized)."""
    spec = registry.resolve(model, version=version)
    return Path(spec["python"]).exists()


# ── status data (mirrors scripts/gen_status_table.py, GPU-free) ──────────────
_VALIDATION_LABEL = {
    "validated_sm86": "validated (sm86)",
    "validated_sm89": "validated (sm89)",
    "gpu_pending": "gpu pending",
    "cpu_only": "cpu only",
}


def _status_rows() -> list[dict[str, Any]]:
    """Build the per-(model, version) status rows from models.json.

    Renders the SAME data as ``scripts/gen_status_table.py``: mlip name,
    framework, weights, validation (human label), gated, and whether a v1 tarball
    is authored (``shipped_v1`` in ``_meta``). Pure registry read — no GPU.
    """
    models = registry.load_models()
    shipped = set(models.get("_meta", {}).get("shipped_v1", []))
    rows: list[dict[str, Any]] = []
    for framework, info in models.items():
        if framework.startswith("_"):
            continue
        for version, vinfo in info.get("versions", {}).items():
            code = vinfo.get("validation", "unknown")
            rows.append(
                {
                    "model": vinfo.get("mlip_name", version),
                    "framework": framework,
                    "version": version,
                    "weights": vinfo.get("weights", "bundled"),
                    "validation": _VALIDATION_LABEL.get(code, code),
                    "validation_code": code,
                    "gated": bool(vinfo.get("gated", False)),
                    "v1_tarball": "upload-pending"
                    if framework in shipped
                    else "Phase 2",
                }
            )
    return rows


# ── server construction ───────────────────────────────────────────────────────
def build_server():
    """Construct and return the ``FastMCP`` server with all tools registered.

    Requires ``mcp`` (raises a clear ImportError otherwise). The tool docstrings
    become the descriptions the calling agent sees, so they state which tools are
    GPU-free and which execute at GPU runtime.
    """
    FastMCP = _require_mcp()
    mcp = FastMCP("oh-my-mlip")

    # ── GPU-free registry tools ──────────────────────────────────────────────
    @mcp.tool()
    def list_models() -> dict:
        """List every registered MLIP framework and its versions. GPU-FREE.

        Pure registry read (no GPU, no conda env, no model load). Returns the
        framework keys in registry order plus each framework's version list and
        default version. Start here (maps to AGENTS.md sections 0-1).
        """
        out: dict[str, Any] = {"models": []}
        models = registry.load_models()
        for name in registry.list_models(models):
            info = models[name]
            out["models"].append(
                {
                    "framework": name,
                    "versions": registry.list_versions(name, models),
                    "default_version": info.get("default_version"),
                }
            )
        return out

    @mcp.tool()
    def describe_model(model: str, version: str | None = None) -> dict:
        """Resolve a model into its codegen dict. GPU-FREE.

        Wraps ``oh_my_mlip.resolve(model, version)``: returns env name, env
        interpreter path, import + inference code lines, parsed ``env_run``,
        and the ``arch_pinned`` / ``gated`` / ``weights`` / ``validation`` flags.
        No model is loaded. ``version=None`` selects the framework's default.
        Use this before run_singlepoint to see a model's run conditions.
        """
        spec = registry.resolve(model, version=version)
        # Return a JSON-safe copy (env_run is already a plain dict of strings).
        return dict(spec)

    @mcp.tool()
    def model_status() -> dict:
        """Per-model ship status: validation / gated / weights / v1 tarball. GPU-FREE.

        Renders the same data as ``scripts/gen_status_table.py`` (the README
        ``## Models & status`` table) straight from ``models.json`` — the single
        source of truth, so it can never overclaim. Pure registry read.
        """
        return {"rows": _status_rows()}

    # ── compute-dependent tools (need a materialized env + GPU at runtime) ────
    @mcp.tool()
    def run_singlepoint(
        model: str,
        structure: Any,
        version: str | None = None,
        apply_d3: bool = False,
    ) -> dict:
        """Single-point energy + forces for one structure. NEEDS A GPU/ENV AT RUNTIME.

        Wraps ``oh_my_mlip.run(model, atoms, properties=("energy","forces"))``,
        which spawns the model's own env interpreter, computes, and tears it
        down. ``structure`` is an ASE-readable spec — one of:
          * a path to a structure file (POSCAR / .cif / .xyz / ...);
          * a full ``Atoms.todict()`` dict;
          * a simple dict ``{"symbols", "positions", "cell"?, "pbc"?}``.
        Set ``apply_d3=True`` for the D3 dispersion correction. Returns
        ``{"energy", "forces"}``. If the model's env is not installed yet, returns
        an actionable error pointing at install_model / install.sh instead of a
        traceback.
        """
        from oh_my_mlip import run

        try:
            atoms = structure_to_atoms(structure)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if not _env_python_exists(model, version):
            return {"ok": False, "error": _env_hint(model, version)}
        results = run(
            model,
            atoms,
            properties=("energy", "forces"),
            version=version,
            apply_d3=apply_d3,
        )
        return {"ok": True, "results": results}

    @mcp.tool()
    def run_relax(
        model: str,
        structure: Any,
        fmax: float = 0.05,
        steps: int = 200,
        version: str | None = None,
        apply_d3: bool = False,
    ) -> dict:
        """Relax a structure with BFGS against one MLIP. NEEDS A GPU/ENV AT RUNTIME.

        Drives an ASE ``BFGS`` optimizer against a persistent ``Worker`` (one
        long-lived env process for the whole relaxation), mirroring
        ``run_examples/relax.py``. ``structure`` accepts the same forms as
        run_singlepoint. Returns the final energy, max force, step count, and the
        relaxed structure as ``Atoms.todict()``. If the env is not installed,
        returns an actionable error pointing at install_model / install.sh.
        """
        try:
            atoms = structure_to_atoms(structure)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if not _env_python_exists(model, version):
            return {"ok": False, "error": _env_hint(model, version)}

        import numpy as np
        from ase.calculators.calculator import Calculator, all_changes
        from ase.optimize import BFGS

        from oh_my_mlip import Worker

        class _WorkerCalculator(Calculator):
            implemented_properties = ["energy", "forces"]

            def __init__(self, worker: "Worker", **kwargs):
                super().__init__(**kwargs)
                self._worker = worker

            def calculate(
                self,
                atoms=None,
                properties=("energy", "forces"),
                system_changes=all_changes,
            ):
                super().calculate(atoms, properties, system_changes)
                resp = self._worker.request(
                    self.atoms, properties=("energy", "forces")
                )
                if not resp.get("ok"):
                    raise RuntimeError(
                        f"worker request failed: {resp.get('error')}"
                    )
                res = resp["results"]
                self.results["energy"] = float(res["energy"])
                self.results["forces"] = np.asarray(res["forces"], dtype=float)

        with Worker(model, version=version, apply_d3=apply_d3) as worker:
            atoms.calc = _WorkerCalculator(worker)
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=fmax, steps=steps)
            nsteps = opt.get_number_of_steps()

        forces = atoms.get_forces()
        fmax_final = float(np.linalg.norm(forces, axis=1).max()) if len(atoms) else 0.0
        return {
            "ok": True,
            "energy": float(atoms.get_potential_energy()),
            "fmax": fmax_final,
            "steps": int(nsteps),
            "converged": fmax_final <= fmax,
            "atoms": json.loads(json.dumps(atoms.todict(), default=_todict_default)),
        }

    @mcp.tool()
    def install_model(model: str, version: str | None = None) -> dict:
        """Materialize a model's conda env (download + relocate). NEEDS HF/COMPUTE.

        Wraps ``oh_my_mlip.fetch.fetch_env``: resolves the env's conda-pack
        tarball from ``dist_manifest.json``, downloads it (revision-pinned),
        verifies its sha256, runs ``conda-unpack`` once, and returns the relocated
        interpreter path. For GATED models you must first accept the upstream
        license and export ``HF_TOKEN``; without it this returns the license URL
        and stops by design (it never redistributes gated weights). If no tarball
        is published yet, it points you at the ``install.sh`` local-build fallback.
        """
        from oh_my_mlip import fetch

        try:
            python_path = fetch.fetch_env(model, version=version)
        except fetch.GatedError as exc:
            spec = registry.resolve(model, version=version)
            return {
                "ok": False,
                "gated": True,
                "license_url": spec.get("license_url"),
                "error": str(exc),
            }
        except fetch.FetchError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "python": python_path}

    @mcp.tool()
    def run_catbench(
        tag: str,
        data_dir: str,
        models: list[str] | None = None,
        calc_num: int = 3,
        apply_d3: bool = False,
    ) -> dict:
        """Run a catbench adsorption benchmark across models. NEEDS A GPU/ENV AT RUNTIME.

        The single-machine catbench roster runner (mirrors
        ``run_examples/catbench_quickstart.py``). You bring your own data: a file
        ``<data_dir>/raw_data/<tag>_adsorption.json`` must exist (this repo bundles
        no benchmark data). Each selected model runs in its OWN env interpreter
        (one subprocess per model) writing into ``<data_dir>/result/`` for catbench
        to aggregate. ``models`` filters the roster (default: all). Returns per-model
        exit codes and the result directory. Models whose env is not installed are
        skipped with an actionable note rather than crashing the whole run.
        """
        import os
        import subprocess

        from oh_my_mlip import list_models as _list_models
        from oh_my_mlip import resolve as _resolve

        work = Path(data_dir).expanduser()
        data_file = work / "raw_data" / f"{tag}_adsorption.json"
        if not data_file.exists():
            return {
                "ok": False,
                "error": (
                    f"benchmark data not found: {data_file}. This repo bundles "
                    "no data — provide your own raw_data/<tag>_adsorption.json "
                    "(see run_examples/README.md)."
                ),
            }

        roster = _list_models()
        selected = [m for m in roster if (models is None or m in models)]
        if not selected:
            return {"ok": False, "error": "no models selected from the roster."}

        template = (
            'import warnings\n'
            'warnings.filterwarnings("ignore")\n'
            "from catbench.adsorption import AdsorptionCalculation\n"
            "{d3_import}{import_lines}\n\n"
            "calc_num = {calc_num}\n"
            "calculators = []\n"
            "for i in range(calc_num):\n"
            "{inference_lines}\n"
            "{d3_apply}    calculators.append(calc)\n\n"
            'config = {{"mlip_name": {mlip_name!r}, "benchmark": {benchmark!r}}}\n'
            "AdsorptionCalculation(calculators, **config).run()\n"
        )

        per_model: list[dict[str, Any]] = []
        for model in selected:
            try:
                spec = _resolve(model)
            except registry.RegistryError as exc:
                per_model.append(
                    {"model": model, "status": "skipped", "reason": str(exc)}
                )
                continue
            if not Path(spec["python"]).exists():
                per_model.append(
                    {
                        "model": model,
                        "status": "skipped",
                        "reason": _env_hint(model, None),
                    }
                )
                continue

            mlip_name = spec.get("version", model) + ("_D3" if apply_d3 else "")
            indent = "    "
            script = template.format(
                d3_import=(
                    "from catbench.dispersion import DispersionCorrection\n"
                    if apply_d3
                    else ""
                ),
                import_lines="\n".join(spec["imports"]),
                calc_num=calc_num,
                inference_lines="\n".join(
                    indent + ln for ln in spec["inference"]
                ),
                d3_apply=(
                    f"{indent}calc = DispersionCorrection().apply(calc)\n"
                    if apply_d3
                    else ""
                ),
                mlip_name=mlip_name,
                benchmark=tag,
            )
            child_env = dict(os.environ)
            child_env.update(spec.get("env_run", {}))
            child_env.setdefault("OH_MY_MLIP_HOME", registry.home())
            proc = subprocess.run(
                [spec["python"], "-c", script],
                env=child_env,
                cwd=str(work),
            )
            per_model.append(
                {
                    "model": model,
                    "mlip_name": mlip_name,
                    "status": "ok" if proc.returncode == 0 else "failed",
                    "returncode": proc.returncode,
                }
            )

        ran = [m for m in per_model if m["status"] in ("ok", "failed")]
        return {
            "ok": bool(ran),
            "result_dir": str(work / "result"),
            "models": per_model,
            "aggregate_hint": (
                "from catbench.adsorption import AdsorptionAnalysis; "
                "a = AdsorptionAnalysis(); a.analysis(); "
                "a.threshold_sensitivity_analysis()"
            ),
        }

    return mcp


def _todict_default(obj):
    """JSON default for Atoms.todict() (numpy arrays/scalars -> native)."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def main() -> None:
    """Build the server and run it over stdio. Requires ``mcp``.

    Launch with:  ``python -m oh_my_mlip.mcp_server``
    """
    server = build_server()
    server.run()


if __name__ == "__main__":
    main()
