"""oh_my_mlip._worker — persistent single-env MLIP worker (Layer 4 wire side).

FROZEN JSONL WIRE CONTRACT (proven on MACE + SevenNet in v1)
===========================================================
One worker process == one conda env == one model. The worker is launched as::

    <env>/bin/python -m oh_my_mlip._worker --model MACE [--version ...]
                                           [--device cuda] [--apply-d3]

It then speaks line-delimited JSON (JSONL) over stdin/stdout. Every message is
exactly one JSON object on one line. The lifecycle is:

1. spawn        : supervisor starts the process with the env's interpreter and
                  the parsed `env_run` applied as the subprocess ENVIRONMENT
                  (never shell-interpolated).
2. ready-handshake : the worker builds the calculator (intra-env
                  `get_calculator`) and emits exactly one line::
                      {"ready": true,  "model": "<m>", "version": "<v>"}
                  on success, or on construction failure::
                      {"ready": false, "error": "<repr>"}
                  then exits non-zero. The supervisor MUST read this line first.
3. serve        : for each request line on stdin::
                      {"id": <any>, "atoms": <ase .todict() json>,
                       "properties": ["energy","forces","stress"]}
                  the worker computes and replies with exactly one line::
                      {"id": <same id>, "ok": true,
                       "results": {"energy": float,
                                   "forces": [[...]],
                                   "stress": [...]}}
                  on failure (per-request, calculator raised)::
                      {"id": <same id>, "ok": false, "error": "<repr>"}
                  Responses CARRY the request id; ordering is NOT assumed to be
                  FIFO by the supervisor (it routes by id).
4. shutdown     : an empty stdin (EOF) OR a line `{"shutdown": true}` ends the
                  serve loop cleanly (exit 0).

CRASH SEMANTICS: if the process dies mid-request, the supervisor surfaces
`{"id": <inflight>, "ok": false, "error": "worker crashed"}` to the caller and
respawns. A per-request exception does NOT kill the worker (it stays alive for
the next id).

NOTE: the 100-call live-GPU loop (one long-lived worker, 100 ids, no respawn)
is DEFERRED to the compute checkpoint. This module's protocol/routing is
unit-tested with a mocked worker; it is not run against a real GPU model here.

The atoms payload uses ASE's dict round-trip (`Atoms.todict()` /
`Atoms.fromdict()`), encoded with a small numpy-aware JSON helper so the wire
stays pure JSON (no pickle, no eval).
"""
from __future__ import annotations

import argparse
import json
import sys


# ── numpy-aware JSON (lazy: numpy only needed when actually (de)serializing) ──
def _json_default(obj):
    # numpy scalars/arrays -> native python; imported lazily so this module
    # still imports on a host without numpy.
    try:
        import numpy as np  # noqa: WPS433  (lazy import is deliberate)
    except Exception:  # pragma: no cover - numpy absent
        np = None
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def encode_atoms(atoms) -> dict:
    """Serialize an ase.Atoms into a JSON-safe dict via Atoms.todict()."""
    raw = atoms.todict()
    # Round-trip through the numpy-aware encoder to flatten ndarrays.
    return json.loads(json.dumps(raw, default=_json_default))


def decode_atoms(payload: dict):
    """Reconstruct an ase.Atoms from an encoded dict (lazy ase import)."""
    import numpy as np
    from ase import Atoms

    d = {}
    for key, value in payload.items():
        # Atoms.fromdict expects numpy arrays for array-like fields.
        if isinstance(value, list):
            d[key] = np.array(value)
        else:
            d[key] = value
    return Atoms.fromdict(d)


def compute(calc, atoms, properties) -> dict:
    """Run the requested properties through a built ASE calculator."""
    atoms = atoms.copy()
    atoms.calc = calc
    results: dict = {}
    if "energy" in properties:
        results["energy"] = float(atoms.get_potential_energy())
    if "forces" in properties:
        results["forces"] = atoms.get_forces().tolist()
    if "stress" in properties:
        results["stress"] = atoms.get_stress().tolist()
    return results


def serve(calc, infile=sys.stdin, outfile=sys.stdout) -> None:
    """Run the serve loop against an already-built calculator.

    Reads one JSON request per line; writes one JSON response per line. Returns
    when stdin reaches EOF or a `{"shutdown": true}` line is received. A
    per-request exception is caught and surfaced as `ok: false` (worker stays
    alive).
    """
    for line in infile:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit(outfile, {"id": None, "ok": False, "error": f"bad json: {exc}"})
            continue
        if req.get("shutdown"):
            break
        rid = req.get("id")
        try:
            atoms = decode_atoms(req["atoms"])
            props = tuple(req.get("properties", ("energy", "forces")))
            results = compute(calc, atoms, props)
            _emit(outfile, {"id": rid, "ok": True, "results": results})
        except Exception as exc:  # noqa: BLE001 - surface as ok:false, stay alive
            _emit(outfile, {"id": rid, "ok": False, "error": repr(exc)})


def _emit(outfile, obj: dict) -> None:
    outfile.write(json.dumps(obj, default=_json_default) + "\n")
    outfile.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="oh_my_mlip._worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--version", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--apply-d3", action="store_true")
    args = parser.parse_args(argv)

    # Intra-env construction. Imported here so the module imports without ase.
    from oh_my_mlip.provider import get_calculator

    try:
        calc = get_calculator(
            args.model,
            version=args.version,
            device=args.device,
            apply_d3=args.apply_d3,
        )
    except Exception as exc:  # noqa: BLE001 - handshake failure
        _emit(sys.stdout, {"ready": False, "error": repr(exc)})
        return 1

    _emit(sys.stdout, {"ready": True, "model": args.model, "version": args.version})
    serve(calc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
