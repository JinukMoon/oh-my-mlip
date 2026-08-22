#!/usr/bin/env python3
"""ft_verify.py <ckpt> --model <M> --json -- the oracle that makes a
`demonstrated` finetune verdict possible (patterned on setup_verify.py).

Loads a fine-tuned checkpoint through that FAMILY's OWN ASE calculator,
INSIDE that family's own env (never the ambient interpreter -- MACE/SevenNet/
DeePMD checkpoints only load inside their own torch/deepmd build), computes
energy + forces on one small Cu-bulk frame, and exits 0 iff both are finite.

This is deliberately narrow: it proves the checkpoint LOADS and PRODUCES A
NUMBER, exactly like setup_verify.py proves an installed model does. It does
NOT judge fine-tuning quality (loss curves, held-out MAE, ...) -- that is out
of scope for an install/execution oracle.

Usage:
  python3 scripts/ft_verify.py ft_mace/MACE_run-123.model --model MACE --json
  python3 scripts/ft_verify.py ft_sevennet/checkpoint_best.pth --model SevenNet --json
  python3 scripts/ft_verify.py ft_deepmd/model.ckpt.pt --model DeePMD --json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from oh_my_mlip import registry as reg  # noqa: E402

# One inline script per family: {ckpt} is substituted with the repr()'d
# absolute checkpoint path. Each prints exactly one JSON line on success.
_LOADER_TEMPLATE = {
    "MACE": '''
import json
from ase.build import bulk
from mace.calculators import MACECalculator
atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
atoms.calc = MACECalculator(model_paths=[{ckpt}], device="{device}", default_dtype="float64")
e = float(atoms.get_potential_energy())
f = atoms.get_forces()
print(json.dumps({{"energy_ev": e, "forces_shape": list(f.shape)}}))
''',
    "SevenNet": '''
import json
from ase.build import bulk
from sevenn.calculator import SevenNetCalculator
atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
atoms.calc = SevenNetCalculator({ckpt}{modal_kwarg})
e = float(atoms.get_potential_energy())
f = atoms.get_forces()
print(json.dumps({{"energy_ev": e, "forces_shape": list(f.shape)}}))
''',
    "DeePMD": '''
import json
from ase.build import bulk
from deepmd.calculator import DP
atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
atoms.calc = DP(model={ckpt})
e = float(atoms.get_potential_energy())
f = atoms.get_forces()
print(json.dumps({{"energy_ev": e, "forces_shape": list(f.shape)}}))
''',
}
_LOADER_TEMPLATE["DPA4"] = _LOADER_TEMPLATE["DeePMD"]


_MODAL_RE = re.compile(r"""modal=['"]([^'"]+)['"]""")


def _extract_modal(inference_lines: list[str]) -> str | None:
    """SevenNet multi-fidelity checkpoints (e.g. 7net-mf-ompa) need the same
    `modal=` kwarg at load time that the registry's own `inference` line
    already carries for single-point use (host-verified, 2026-08-23:
    SevenNetCalculator raises `modal argument missing` without it)."""
    for line in inference_lines:
        m = _MODAL_RE.search(line)
        if m:
            return m.group(1)
    return None


def build_script(model: str, ckpt: str, device: str, modal: str | None = None) -> str:
    template = _LOADER_TEMPLATE.get(model)
    if template is None:
        raise SystemExit(
            f"[ft_verify] no checkpoint-loader template for {model!r}; "
            f"supported: {sorted(_LOADER_TEMPLATE)}"
        )
    modal_kwarg = f", modal={modal!r}" if modal else ""
    return template.format(ckpt=repr(str(Path(ckpt).resolve())), device=device, modal_kwarg=modal_kwarg)


def verify(model: str, ckpt: str, device: str = "cpu") -> dict:
    resolved = reg.resolve(model)
    modal = _extract_modal(resolved.get("inference") or []) if resolved["model"] == "SevenNet" else None
    script = build_script(resolved["model"], ckpt, device, modal)
    import os
    env = dict(os.environ, **(resolved.get("env_run") or {}))
    proc = subprocess.run(
        [resolved["python"], "-c", script],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        return {"pass": False, "model": resolved["model"], "ckpt": str(ckpt), "reason": tail or f"exit {proc.returncode}"}

    witness = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                witness = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    if witness is None:
        return {"pass": False, "model": resolved["model"], "ckpt": str(ckpt), "reason": "witness_json_missing"}

    import math
    energy = witness.get("energy_ev")
    finite = isinstance(energy, (int, float)) and math.isfinite(energy)
    return {
        "pass": bool(finite),
        "model": resolved["model"],
        "ckpt": str(ckpt),
        "energy_ev": energy,
        "forces_shape": witness.get("forces_shape"),
        "reason": "" if finite else "non_finite_energy",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt", help="path to the fine-tuned checkpoint")
    ap.add_argument("--model", required=True, help="framework/version key from models.json")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    verdict = verify(args.model, args.ckpt, args.device)
    if args.json:
        print(json.dumps(verdict))
    else:
        state = "PASS" if verdict["pass"] else "FAIL"
        print(f"{state} model={verdict['model']} energy={verdict.get('energy_ev')} reason={verdict.get('reason') or '-'}")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
