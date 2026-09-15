#!/usr/bin/env python3
"""ft_verify.py <ckpt> --model <M> [--device cuda|cpu] --json -- the oracle
that makes a `demonstrated` finetune verdict possible (patterned on
setup_verify.py).

Loads a fine-tuned checkpoint through that FAMILY's OWN ASE calculator,
INSIDE that family's own env (never the ambient interpreter -- MACE/SevenNet/
DeePMD checkpoints only load inside their own torch/deepmd build), computes
energy + forces on one small Cu-bulk frame, and exits 0 iff both are finite
(and, on --device cuda, iff the forward demonstrably ran on the GPU).

This is deliberately narrow: it proves the checkpoint LOADS and PRODUCES A
NUMBER, exactly like setup_verify.py proves an installed model does. It does
NOT judge fine-tuning quality (loss curves, held-out MAE, ...) -- that is out
of scope for an install/execution oracle.

Witness contract (consumed by scripts/ft_sweep.py): the verdict JSON carries
  pass            bool
  version         the EXACT registry variant the reload was built for --
                  `--model <family>` alone resolves the family default, so
                  pass `--version <variant>` (or the variant name as --model)
                  for a checkpoint fine-tuned from a non-default variant (a
                  SevenNet checkpoint reloaded under the wrong modal= would
                  otherwise be judged against the wrong construct)
  energy_ev       finite float
  forces_shape    exactly [4, 3] (the Cu-bulk probe frame has 4 atoms)
  forces_finite   True (every force component finite)
  device          the device requested ("cuda" | "cpu")
  gpu_used        MEASURED inside the family env DURING the E/F forward, from
                  the backend's own execution record -- never a constant, and
                  never a memory-allocation proxy:
                    torch families  a TorchDispatchMode wraps exactly the
                                    get_potential_energy()/get_forces() calls
                                    and counts ATen compute ops that received
                                    a CUDA tensor INPUT (allocation, fill, copy
                                    and view ops are excluded), so a CPU
                                    forward next to an unrelated CUDA
                                    allocation records zero and fails;
                    GRACE (TF)      tf.debugging.set_log_device_placement is
                                    switched on for the forward only and the
                                    placer/executor log lines are parsed for
                                    compute ops placed on device:GPU:N; the TF
                                    profiler also traces the forward and the
                                    events on its /device:GPU:N planes (CUPTI
                                    kernel activity) count as GPU compute, which
                                    covers a saved_model forward run as one XLA
                                    cluster that logs no per-op placement.
  witness         the raw counts behind gpu_used (backend, op totals, names)
A `--device cpu` run pins CUDA_VISIBLE_DEVICES="" (and deepmd's DEVICE=cpu)
in the child so families whose calculators auto-select a GPU really stay on
the CPU; `--device cuda` leaves the env alone and lets the calculator choose.

Loader capability discovery (ft_sweep.py's first probe):
  python3 scripts/ft_verify.py --list-loaders --json
prints ONE JSON line {"loaders": [family, ...], "source": "..."} and exits 0.

Usage:
  python3 scripts/ft_verify.py ft_mace/MACE_run-123.model --model MACE --json
  python3 scripts/ft_verify.py ft_sevennet/checkpoint_best.pth --model SevenNet --json
  python3 scripts/ft_verify.py ft_deepmd/model.ckpt.pt --model DeePMD --json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from oh_my_mlip import registry as reg  # noqa: E402

# Witness tails, appended UNFORMATTED (no str.format placeholders -- the
# template head binds DEVICE, atoms and atoms.calc; the tail owns the forward).
#
# torch: a TorchDispatchMode sees every ATen op the forward dispatches,
# including ops issued from inside TorchScript modules (metatomic/PET exports
# are scripted) -- checked in the MACE (torch 2.6.0+cu124) and
# pet (torch 2.9.1+cu128) envs: a CUDA forward of a python or scripted module
# records addmm/tanh/sum/... with CUDA inputs, while a CPU forward run next to
# torch.empty/zeros/ones(device="cuda") or an H2D copy records zero
# CUDA-input compute ops. torch.profiler was rejected for this job because
# kineto captured no CUDA activity at all on some hosts (CUPTI), i.e. it
# would fail a genuine GPU forward too.
_TORCH_WITNESS_TAIL = '''
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

# ops that touch a CUDA tensor without computing anything on it: allocation,
# initialisation, host<->device movement, metadata views, scalar readback.
_NON_COMPUTE = {
    "empty", "empty_like", "empty_strided", "zeros", "zeros_like", "ones", "ones_like",
    "full", "full_like", "fill_", "zero_", "copy_", "_to_copy", "to", "clone", "contiguous",
    "detach", "alias", "view", "view_as", "reshape", "unsqueeze", "squeeze", "t", "transpose",
    "permute", "expand", "narrow", "select", "slice", "as_strided", "resize_", "_local_scalar_dense",
    "is_nonzero", "_unsafe_view", "lift_fresh",
}


class _ForwardWitness(TorchDispatchMode):
    """Records, for the ops dispatched while the mode is active, how many
    compute ops received at least one CUDA tensor as input."""

    def __init__(self):
        super().__init__()
        self.ops_total = 0
        self.cuda_ops = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.ops_total += 1
        packet = getattr(func, "overloadpacket", None)
        name = getattr(packet, "__name__", None) or str(func)
        if name not in _NON_COMPUTE:
            for t in tree_flatten((args, kwargs))[0]:
                if isinstance(t, torch.Tensor) and t.device.type == "cuda":
                    self.cuda_ops[name] = self.cuda_ops.get(name, 0) + 1
                    break
        return func(*args, **kwargs)


with _ForwardWitness() as _w:
    e = float(atoms.get_potential_energy())
    f = atoms.get_forces()
gpu_used = bool(_w.cuda_ops)
import numpy as _np
print(json.dumps({
    "energy_ev": e, "forces_shape": list(f.shape), "forces_finite": bool(_np.isfinite(_np.asarray(f, dtype=float)).all()),
    "gpu_used": gpu_used, "device": DEVICE,
    "witness": {"backend": "torch", "method": "TorchDispatchMode cuda-input compute ops during E/F forward",
                "ops_total": _w.ops_total, "cuda_compute_ops": sum(_w.cuda_ops.values()),
                "cuda_op_names": sorted(_w.cuda_ops)[:16]},
}))
'''

# TensorFlow (GRACE): TF logs the device every placed op / executed eager op
# lands on when set_log_device_placement(True) is on (placer.cc "name:
# (OpType): /job:.../device:GPU:0" and execute.cc "Executing op X in device
# ..."). The C++ log goes to fd 2, so fd 2 is redirected to a temp file for
# the forward only; non-placement lines are re-emitted to stderr afterwards.
# Checked in the grace env (TF 2.16.2): eager and jit_compile'd
# tf.function ops both log placement lines in this exact format -- but a
# restored gracemaker saved_model runs its forward as one XLA cluster and logs
# no per-op placement. The TF profiler traces the same forward: checked on an
# RTX 4060 Ti, that forward put 10444 events on the /device:GPU:0 plane, and
# with CUDA_VISIBLE_DEVICES="" the trace has no GPU plane at all. Those GPU
# plane events are CUPTI kernel activity, the backend's execution record, and
# are added to the GPU compute count; a profiler that cannot start records zero
# and the GPU branch fails closed. On a host
# where the grace env registers no GPU device at all ("Cannot dlopen some GPU
# libraries"), the GPU branch of this witness cannot pass -- it fails closed,
# which is the correct verdict.
_TF_WITNESS_TAIL = '''
import os
import re
import sys
import tempfile
import tensorflow as tf

_gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
_PLACE = re.compile(r"^\\s*(?:[0-9.: -]*[IWE] [^\\]]*\\] )?([^\\s:]+): \\(([A-Za-z0-9_]+)\\): /job:\\S*/device:(GPU|CPU):\\d+\\s*$")
_EXEC = re.compile(r"Executing op ([A-Za-z0-9_]+) in device /job:\\S*/device:(GPU|CPU):\\d+")
_SKIP = {"_Arg", "_Retval", "_EagerConst", "_DeviceArg", "_DeviceRetval", "Const", "Identity",
         "IdentityN", "NoOp", "Placeholder", "VarHandleOp", "ReadVariableOp"}
_log = tempfile.TemporaryFile("w+")
_prof_dir = tempfile.mkdtemp(prefix="ft_verify_tfprof_")
_prof_error = None
try:
    tf.profiler.experimental.start(_prof_dir)
except Exception as _exc:  # no CUPTI / profiler already running: no GPU plane evidence
    _prof_error = repr(_exc)
tf.debugging.set_log_device_placement(True)
_saved = os.dup(2)
os.dup2(_log.fileno(), 2)
try:
    e = float(atoms.get_potential_energy())
    f = atoms.get_forces()
finally:
    os.dup2(_saved, 2)
    os.close(_saved)
    tf.debugging.set_log_device_placement(False)
_log.seek(0)
gpu_ops, cpu_ops, other = {}, 0, []
for line in _log:
    m = _PLACE.search(line)
    if m:
        op, dev = m.group(2), m.group(3)
    else:
        m = _EXEC.search(line)
        if not m:
            other.append(line)
            continue
        op, dev = m.group(1), m.group(2)
    if op in _SKIP:
        continue
    if dev == "GPU":
        gpu_ops[op] = gpu_ops.get(op, 0) + 1
    else:
        cpu_ops += 1
sys.stderr.write("".join(other))
gpu_plane_events = 0
if _prof_error is None:
    try:
        tf.profiler.experimental.stop()
        import glob
        try:
            from tsl.profiler.protobuf import xplane_pb2
        except ImportError:
            from tensorflow.tsl.profiler.protobuf import xplane_pb2
        for _p in glob.glob(os.path.join(_prof_dir, "**", "*.xplane.pb"), recursive=True):
            _space = xplane_pb2.XSpace()
            with open(_p, "rb") as _fh:
                _space.ParseFromString(_fh.read())
            for _plane in _space.planes:
                if _plane.name.startswith("/device:GPU:"):
                    gpu_plane_events += sum(len(_line.events) for _line in _plane.lines)
    except Exception as _exc:
        _prof_error = repr(_exc)
        gpu_plane_events = 0
import shutil
shutil.rmtree(_prof_dir, ignore_errors=True)
gpu_count = sum(gpu_ops.values()) + gpu_plane_events
gpu_used = bool(_gpus) and gpu_count > 0
import numpy as _np
print(json.dumps({
    "energy_ev": e, "forces_shape": list(f.shape), "forces_finite": bool(_np.isfinite(_np.asarray(f, dtype=float)).all()),
    "gpu_used": gpu_used, "device": DEVICE,
    "witness": {"backend": "tensorflow",
                "method": "log_device_placement compute ops on device:GPU plus TF profiler /device:GPU plane events, "
                          "during E/F forward",
                "tf_gpu_devices": _gpus, "gpu_compute_ops": gpu_count, "cpu_compute_ops": cpu_ops,
                "gpu_op_names": sorted(gpu_ops)[:16], "gpu_plane_events": gpu_plane_events,
                "profiler_error": _prof_error},
}))
'''

# Which witness tail each family's backend needs.
_WITNESS_FOR = {"GRACE": _TF_WITNESS_TAIL}

# One inline script per family: {ckpt} is substituted with the repr()'d
# absolute checkpoint path, {device} with "cuda"/"cpu". Each template binds
# `atoms` (a small Cu-bulk frame) with its calculator attached; the witness
# tail then runs the E/F forward under the backend's execution record and
# prints exactly one JSON line on success.
_LOADER_TEMPLATE = {
    "MACE": '''
import json
from ase.build import bulk
from mace.calculators import MACECalculator
atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
atoms.calc = MACECalculator(model_paths=[{ckpt}], device="{device}", default_dtype="float64")
''',
    "SevenNet": '''
import json
from ase.build import bulk
from sevenn.calculator import SevenNetCalculator
atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
atoms.calc = SevenNetCalculator({ckpt}, device="{device}"{modal_kwarg})
''',
    # deepmd's DP calculator takes no device argument: deepmd.pt.utils.env
    # picks cuda iff torch sees a GPU and the DEVICE env var is not "cpu"
    # (the --device cpu child env sets both CUDA_VISIBLE_DEVICES="" and
    # DEVICE=cpu, see _child_env).
    "DeePMD": '''
import json
from ase.build import bulk
from deepmd.calculator import DP
atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
atoms.calc = DP(model={ckpt})
''',
    # metatrain's `mtt train -o model-ft.pt` exports a metatomic model;
    # MetatomicCalculator(model, device=...) (metatomic-torch 0.1.7
    # ase_calculator.py:63) is the same entry the registry's PET inference
    # line uses. The export is TorchScript -- the dispatch-mode witness still
    # sees its ops (host-probed, see _TORCH_WITNESS_TAIL).
    "PET": '''
import json
from ase.build import bulk
from metatomic.torch.ase_calculator import MetatomicCalculator
atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
atoms.calc = MetatomicCalculator({ckpt}, device="{device}")
''',
    # nequip 0.17.1 (NequIP env) and 0.15.0 (Allegro env) both expose
    # NequIPCalculator._from_saved_model(model_path, device=...) which loads
    # a Lightning checkpoint, a .nequip.zip package or a nequip.net ID
    # (installed docstrings). It is the only uncompiled ASE path; the
    # public from_compiled_model needs a nequip-compile step first. The
    # 0.17.1 class lives in nequip.integrations.ase (nequip.ase is a
    # deprecation shim), 0.15.0 only has nequip.ase.
    "NequIP": '''
import json
from ase.build import bulk
try:
    from nequip.integrations.ase import NequIPCalculator
except ImportError:
    from nequip.ase import NequIPCalculator
atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
atoms.calc = NequIPCalculator._from_saved_model({ckpt}, device="{device}")
''',
    # mattersim 1.2.1: Potential.from_checkpoint(load_path, device=...,
    # load_training_state=False) reads the best_model.pth the fine-tune
    # driver saves; MatterSimCalculator(potential=..., device=...)
    # (forcefield/potential.py:849, :1240).
    "MatterSim": '''
import json
from ase.build import bulk
from mattersim.forcefield.potential import MatterSimCalculator, Potential
atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
potential = Potential.from_checkpoint(load_path={ckpt}, device="{device}", load_training_state=False)
atoms.calc = MatterSimCalculator(potential=potential, device="{device}")
''',
    # tace 0.2.0: TACEAseCalc(model=<.ckpt|.pt|.pth>, device, fidelity_idx,
    # target_property) -> load_tace() (interface/ase/calculator.py:57,
    # lightning/lit_model.py:688); same kwargs as the registry inference
    # line, minus the stress target the demo data does not carry.
    "TACE": '''
import json
from ase.build import bulk
from tace.interface.ase import TACEAseCalc
atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
atoms.calc = TACEAseCalc(model={ckpt}, dtype="float32", device="{device}", fidelity_idx=0, target_property=["energy", "forces"])
''',
    # chgnet 0.4.0: Trainer.save() stores {"model": CHGNet.as_dict()}, which
    # CHGNet.from_file(path) rebuilds (model/model.py:681); CHGNetCalculator
    # (model/dynamics.py:56) takes the model + use_device.
    "CHGNet": '''
import json
from ase.build import bulk
from chgnet.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
model = CHGNet.from_file({ckpt})
atoms.calc = CHGNetCalculator(model=model, use_device="{device}")
''',
    # GRACE is TensorFlow, not torch: gracemaker's final_model is a
    # tf.saved_model DIRECTORY (saved_model.pb + fingerprint.pb +
    # variables/{{variables.index, variables.data-*}} + assets/ from
    # tf.saved_model.save, plus gracemaker's own metadata.yaml next to them;
    # tensorpotential 0.5.3 tpmodel.py:436-453). TPCalculator loads it by
    # directory path (calculator/asecalculator.py:302,319). The witness file
    # handed in is <final_model>/saved_model.pb; its parent is the model.
    "GRACE": '''
import json
import os
from ase.build import bulk
from tensorpotential.calculator import TPCalculator
model_dir = os.path.dirname({ckpt}) if {ckpt}.endswith("saved_model.pb") else {ckpt}
atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
atoms.calc = TPCalculator(model_dir)
''',
    # fairchem's fine-tune writes checkpoints/final/inference_ckpt.pt; it loads with
    # load_predict_unit and must be used with the task it was trained on (the registry's
    # inference task_name), fairchem-core docs/core/common_tasks/fine_tuning.md.
    "UMA": '''
import json
from ase.build import bulk
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
atoms.calc = FAIRChemCalculator(load_predict_unit({ckpt}, device="{device}"), task_name={task!r})
''',
}
_LOADER_TEMPLATE["DPA4"] = _LOADER_TEMPLATE["DeePMD"]
_LOADER_TEMPLATE["Allegro"] = _LOADER_TEMPLATE["NequIP"]


def loader_families() -> list[str]:
    """The families whose fine-tuned checkpoint this script can really
    reload. Public so ft_sweep.py can query capability instead of guessing;
    a family outside this set is an implementation gap of THIS script."""
    return sorted(_LOADER_TEMPLATE)


_MODAL_RE = re.compile(r"""modal=['"]([^'"]+)['"]""")
_TASK_RE = re.compile(r"""task_name=['"]([^'"]+)['"]""")


def _extract_task(inference_lines: list[str]) -> str | None:
    """A fine-tuned UMA checkpoint is used with the task it was trained on -- the
    same task_name the registry's inference line carries for that variant."""
    for line in inference_lines:
        m = _TASK_RE.search(line)
        if m:
            return m.group(1)
    return None


def _extract_modal(inference_lines: list[str]) -> str | None:
    """SevenNet multi-fidelity checkpoints (e.g. 7net-mf-ompa) need the same
    `modal=` kwarg at load time that the registry's own `inference` line
    already carries for single-point use (without it,
    SevenNetCalculator raises `modal argument missing` without it)."""
    for line in inference_lines:
        m = _MODAL_RE.search(line)
        if m:
            return m.group(1)
    return None


def witness_tail(model: str) -> str:
    """The unformatted witness code appended after `model`'s loader head.
    Public so tests can run it against a fake `atoms` in a real family env
    (adversarial CPU-forward-plus-CUDA-allocation must come out gpu_used
    false; a CUDA forward true)."""
    return _WITNESS_FOR.get(model, _TORCH_WITNESS_TAIL)


def build_script(model: str, ckpt: str, device: str, modal: str | None = None,
                 task: str | None = None) -> str:
    template = _LOADER_TEMPLATE.get(model)
    if template is None:
        raise SystemExit(
            f"[ft_verify] no checkpoint-loader template for {model!r}; "
            f"supported: {loader_families()}"
        )
    if model == "UMA" and not task:
        raise SystemExit("[ft_verify] a UMA checkpoint needs the task it was fine-tuned for (task_name)")
    modal_kwarg = f", modal={modal!r}" if modal else ""
    head = template.format(ckpt=repr(str(Path(ckpt).resolve())), device=device, modal_kwarg=modal_kwarg,
                           task=task)
    return f"DEVICE = {device!r}\n" + head + witness_tail(model)


def _child_env(resolved: dict, device: str) -> dict:
    env = dict(os.environ)
    # The env's own lib dir first on the loader path, as oh_my_mlip/provider.py does for
    # inference: GRACE's TensorFlow finds its cuDNN 8 only there and otherwise registers no
    # GPU. env_run is applied after this, so its overrides (e.g. LD_LIBRARY_PATH="") still win.
    python = os.path.expandvars(resolved.get("python") or "")
    env_lib = os.path.join(os.path.dirname(os.path.dirname(python)), "lib") if python else ""
    if env_lib and os.path.isdir(env_lib):
        # the env's pip CUPTI wheel too, so the TF profiler witness can trace GPU kernels
        cupti = sorted(glob.glob(os.path.join(env_lib, "python3*", "site-packages", "nvidia", "cuda_cupti", "lib")))
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join([env_lib, *cupti]) + (os.pathsep + existing if existing else "")
    env.update(resolved.get("env_run") or {})
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["DEVICE"] = "cpu"  # deepmd.pt.utils.env honours this literal
    return env


def _last_json_line(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


# Every loader head builds the same frame: ase.build.bulk("Cu", "fcc", a=3.61,
# cubic=True) -- 4 atoms -- so the forces the witness reports must be [4, 3].
N_ATOMS = 4

# The count the witness tail must have recorded on the GPU for gpu_used to be
# true -- the verdict re-derives gpu_used from it instead of trusting the flag.
_GPU_COUNT_KEY = {"torch": "cuda_compute_ops", "tensorflow": "gpu_compute_ops"}


def judge(witness: dict, device: str) -> str:
    """Reason a witness JSON line fails, or "" when it passes: finite energy,
    finite [N_ATOMS, 3] forces, and -- on cuda -- a backend execution record
    whose GPU compute-op count is > 0 and agrees with the gpu_used flag."""
    energy = witness.get("energy_ev")
    if not (isinstance(energy, (int, float)) and not isinstance(energy, bool) and math.isfinite(energy)):
        return "non_finite_energy"
    if witness.get("forces_shape") != [N_ATOMS, 3]:
        return "bad_forces_shape"
    if witness.get("forces_finite") is not True:
        return "non_finite_forces"
    record = witness.get("witness")
    if not isinstance(record, dict) or record.get("backend") not in _GPU_COUNT_KEY:
        return "witness_record_missing"
    count = record.get(_GPU_COUNT_KEY[record["backend"]])
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        return "witness_record_missing"
    gpu_used = witness.get("gpu_used")
    if gpu_used is not (count > 0):
        # the flag and the raw count disagree: a hand-edited or stubbed line
        return "witness_inconsistent"
    if device == "cuda" and not gpu_used:
        # no compute op of the E/F forward was recorded on the GPU -- a CPU
        # forward with GPU memory merely allocated lands here on purpose.
        return "gpu_not_used"
    return ""


def verify(model: str, ckpt: str, device: str = "cpu", version: str | None = None) -> dict:
    """`model` may be a family or a version name; `version` pins the exact
    variant (M3: a family name alone resolves the family DEFAULT, whose
    inference line -- e.g. SevenNet's modal='mpa' -- may not be the one the
    checkpoint was fine-tuned from)."""
    resolved = reg.resolve(model, version)
    family = resolved["model"]
    modal = _extract_modal(resolved.get("inference") or []) if family == "SevenNet" else None
    task = _extract_task(resolved.get("inference") or []) if family == "UMA" else None
    script = build_script(family, ckpt, device, modal, task)
    proc = subprocess.run(
        [resolved["python"], "-c", script],
        capture_output=True, text=True, env=_child_env(resolved, device),
    )
    base = {"pass": False, "model": family, "version": resolved.get("version"),
            "modal": modal, "ckpt": str(ckpt), "device": device}
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        return {**base, "reason": tail or f"exit {proc.returncode}"}

    witness = _last_json_line(proc.stdout)
    if witness is None:
        return {**base, "reason": "witness_json_missing"}

    reason = judge(witness, device)
    return {
        **base,
        "pass": reason == "",
        "energy_ev": witness.get("energy_ev"),
        "forces_shape": witness.get("forces_shape"),
        "forces_finite": witness.get("forces_finite"),
        "gpu_used": witness.get("gpu_used"),
        "witness": witness.get("witness"),
        "reason": reason,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt", nargs="?", default=None, help="path to the fine-tuned checkpoint")
    ap.add_argument("--model", default=None, help="framework/version key from models.json")
    ap.add_argument("--version", default=None,
                    help="exact variant the checkpoint was fine-tuned from (a bare family "
                         "name in --model resolves that family's default version)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list-loaders", action="store_true",
                    help="print the families this script can reload and exit")
    args = ap.parse_args()

    if args.list_loaders:
        payload = {"loaders": loader_families(), "source": "ft_verify._LOADER_TEMPLATE"}
        if args.json:
            print(json.dumps(payload))
        else:
            print(" ".join(payload["loaders"]))
        return 0

    if not args.ckpt or not args.model:
        ap.error("ckpt and --model are required unless --list-loaders is given")

    try:
        verdict = verify(args.model, args.ckpt, args.device, args.version)
    except reg.RegistryError as exc:
        print(f"[ft_verify] {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(verdict))
    else:
        state = "PASS" if verdict["pass"] else "FAIL"
        print(f"{state} model={verdict['model']}/{verdict['version']} device={verdict['device']} "
              f"gpu_used={verdict.get('gpu_used')} energy={verdict.get('energy_ev')} "
              f"reason={verdict.get('reason') or '-'}")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
