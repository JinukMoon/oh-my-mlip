#!/usr/bin/env python3
"""ft_run.py <model> --dataset <path> [--out DIR] [--epochs N] [--seed S]
[--emit-only] [--slurm [--partition P]] -- resolve one variant's `finetune`
block (models.json, ingested from scripts/upstream_finetune.py in P2a) into
a concrete, materialized fine-tuning run.

Write-then-execute, same discipline as scripts/catbench_jobgen.py (P1): ALL
artifacts land on disk before anything runs --

  <out>/data/...                converted dataset (via scripts/ft_dataset.py)
  <out>/input.yaml|input.json   the patched training config, when the
                                 family's `selector_kind` is `config_key`
                                 (C6 -- a YAML/JSON-PATCHING path, not a flag
                                 builder: MACE and the deepmd family are the
                                 only `cli_flag` families). SevenNet writes
                                 sevennet_patch.json + a prestage step that
                                 materializes input.yaml from the installed
                                 `sevenn_preset` inside the env (emit-only
                                 never needs the env)
  <out>/*_prestage.py           in-env steps run under the .sh's `set -eu`
                                 before the trainer (NequIP/Allegro package
                                 fetch + type-name extraction, TACE foundation
                                 path, SevenNet preset materialization)
  <out>/finetune_<version>.sh   the AC7 rerun unit: `set -eu`, `cd` to the
                                 absolute `<out>` dir (N4), one `export` per
                                 `env_run` key, then `exec` the real command
  <out>/slurm_finetune_<v>.sh   (--slurm) the IDENTICAL body under an SBATCH
                                 header (C17) -- emission only, never sbatch'd
  <out>/ft_run.json             provenance: the exact arguments (and the
                                 argv that re-materializes this run after a
                                 cache cleanup), how far --seed reaches in
                                 this family (SEED_CONTROL), the ONE designated
                                 checkpoint glob (FAMILY_CHECKPOINT_GLOBS),
                                 and every artifact path written

Default `--out` is ./ft_<version> (per VARIANT, so two variants of one
family never overwrite each other's run).

Refusal gate, checked BEFORE any file is written:
  status in {not-supported, code-excavation-needed}  -> exit 2 (reason + any
                                                          evidence URLs)
  runnable_as_installed: false (after a live blocker  -> exit 3 (remaining
    recheck -- see `live_recheck_blockers`)              blockers, each
                                                          already phrased as
                                                          its own fix)
  family has no real builder in `BUILDERS`           -> exit 4 (implementation
                                                          gap in THIS hub, not
                                                          an upstream fact)
  explicit --seed for a family whose trainer only    -> exit 5 (unless
    seeds the data split (NequIP/Allegro, see           --allow-partial-seed;
    SEED_CONTROL)                                        the scope is then
                                                          recorded, never
                                                          silently narrowed)

`live_recheck_blockers` re-verifies two blocker SHAPES against the actual
installed state rather than trusting the models.json snapshot forever: a
"pip install X" blocker is dropped once `import X` succeeds in the target
env's interpreter (so `pip install dpdata` between two ft_run.py invocations
is enough to unblock DeePMD/DPA4 without an ingestion re-run), and a
"symlink ... to a .pt name" blocker for the deepmd family is dropped
unconditionally because `build_deepmd` performs that exact symlink itself
before executing. Every other blocker shape is left standing -- ORB's
"finetune.py is not shipped in the wheel" blocker fetches a file and is
deliberately never auto-applied.

Only families with a real, upstream-source-verified command builder in
`BUILDERS` are ever rendered or executed. There is deliberately NO generic
fallback renderer: a family that passes the registry gate but has no
builder here exits 4 before any dataset conversion or file write, so a
"documented + runnable_as_installed" classification can never be mistaken
for a working command (fail closed; the sweep records it as an
implementation gap, never as `unsupported`).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPTS_DIR))

from oh_my_mlip import registry as reg  # noqa: E402
import ft_settings  # noqa: E402

FT_DATASET = _SCRIPTS_DIR / "ft_dataset.py"

# Which ft_dataset.py --to alias each registry family's training entrypoint
# consumes. EXHAUSTIVE over models.json (tests pin the key set) and with NO
# default: ft_dataset.canonical_target() is the single place that decides
# whether a target is implemented, and a family whose target it refuses can
# never be handed the wrong (extxyz) layout by accident. The not-yet-written
# targets (orb / chgnet / aselmdb / alphanet) are still named here so the
# refusal is explicit and the gap is visible in one table.
FAMILY_DATASET_TARGET = {
    "MACE": "mace", "SevenNet": "sevennet", "NequIP": "nequip", "Allegro": "allegro",
    "GRACE": "grace", "MatterSim": "mattersim", "PET": "pet", "TACE": "tace",
    "DeePMD": "deepmd", "DPA4": "deepmd",
    "CHGNet": "chgnet",
    "ORB": "orb",
    "UMA": "aselmdb", "EquiformerV3": "aselmdb", "Nequix": "aselmdb",
    "fairchemv1": "aselmdb", "EquFlash": "aselmdb",
    "AlphaNet": "alphanet",
    # not-supported upstream (registry status not-supported, refused at exit 2
    # before any dataset is built); listed so the table stays exhaustive.
    "Eqnorm": "extxyz", "MatRIS": "extxyz",
}

# ft_run's own exit code for "registry says go, but this hub has no real
# builder for the family" -- distinct from the registry-driven 2/3 so the
# sweep can class it as an implementation gap (failed(no_builder)).
EXIT_NO_BUILDER = 4
# An explicit --seed that the family's upstream trainer cannot fully honour
# is refused (never silently narrowed) unless --allow-partial-seed says the
# caller accepts the narrower scope recorded in SEED_CONTROL / ft_run.json.
EXIT_SEED_UNHONOURED = 5

# How far `--seed` reaches in each family's OWN trainer -- "native" means the
# value is handed to the upstream seed knob that seeds the training run;
# "data-split-only" means only the data pipeline is seeded and the training
# seed is fixed upstream. Each basis names the installed upstream source
# inspected; none of these is a runtime-proven determinism claim.
SEED_CONTROL = {
    "MACE": ("native", "mace_run_train --seed (mace 0.3.15 tools/arg_parser.py --seed, default 123; "
                       "run_train.py calls tools.set_seeds(args.seed))"),
    "SevenNet": ("native", "train.random_seed patched into the sevenn_preset output by the "
                           "sevennet_prestage.py step (sevenn 0.12.2.dev0 presets carry random_seed: 1 "
                           "/ 777; sevenn/_keys.py RANDOM_SEED)"),
    "DeePMD": ("native", "training.seed (deepmd-kit 3.1.2 pt/entrypoints/main.py:167 data_seed -> "
                         "DpLoaderSet); descriptor/fitting_net init seeds come from the checkpoint's "
                         "own model section under --use-pretrain-script"),
    "DPA4": ("native", "training.seed (deepmd-kit 3.2.0b0, same pt/entrypoints/main.py plumbing as DeePMD)"),
    "GRACE": ("native", "top-level `seed` (tensorpotential 0.5.3 gracemaker.py: args_yaml['seed'] seeds "
                        "tf/np and names the seed/<seed>/ output dir)"),
    "PET": ("native", "top-level `seed` (metatrain 2026.1 options.yaml: seeds torch / numpy / random / "
                      "PYTHONHASHSEED)"),
    "NequIP": ("data-split-only", "data.seed (ASEDataModule split/shuffle) ONLY -- the training seed is "
                                  "hard-coded seed_everything(123) in pinned nequip 0.15.0 utils/global_state.py:79 "
                                  "and no config key reaches it"),
    "Allegro": ("data-split-only", "same as NequIP: data.seed only; pinned nequip 0.15.0 utils/global_state.py:79 "
                                   "pins the training seed to 123"),
    "MatterSim": ("native", "--seed (mattersim 1.2.1 training/finetune_mattersim.py seeds random / numpy / torch)"),
    "TACE": ("native", "misc.global_seed + dataset.split_seed (tace 0.2.0 scripts/train.py, dataset/read.py)"),
    "CHGNet": ("native", "generated driver seeds random / numpy / torch itself and passes "
                         "Trainer(torch_seed=, data_seed=) (chgnet 0.4.0 trainer/trainer.py:130-133 -- "
                         "its `if data_seed:` skips seed 0, hence the explicit random.seed in the driver)"),
}


def dataset_target(family: str) -> str:
    """Fail closed: an unknown family is a registry/ft_run drift, not 'extxyz'."""
    try:
        return FAMILY_DATASET_TARGET[family]
    except KeyError:
        raise SystemExit(
            f"[ft_run] family {family!r} has no entry in FAMILY_DATASET_TARGET -- "
            f"refusing to guess a dataset layout"
        ) from None

_GEN_COMMENT = "# generated by scripts/ft_run.py -- rerun this file to reproduce the fine-tune"

# Policy: advertise licence-gated fine-tunes, do not
# refuse them -- print the licence (+ a URL where the registry doesn't carry
# one itself) and proceed. oh-my-mlip is MIT and redistributes no weights
# either way; this is disclosure, not a capability gate.
_LICENCE_URLS = {
    "ASL": "https://github.com/ACEsuit/mace-foundations",
    "CC-BY-NC-SA-4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
}
_PIP_INSTALL_RE = re.compile(r"pip install ([A-Za-z0-9_\-.]+)")
_QUOTED_RE = re.compile(r"""['"]([^'"]+)['"]""")

# Family-specific overrides where the version's regular single-point
# `inference` line points at a DIFFERENT weight file than the one
# upstream_finetune.py's blockers name as the fine-tunable checkpoint (e.g.
# DeePMD's inference uses frozen-omat24.pth for single-point, but the
# FT-designated weight is dpa-3.1-3m-ft.pth, matching the version key
# DPA-3.1-3M-FT). Extracting from `inference` alone would silently fine-tune
# the wrong file.
_FOUNDATION_OVERRIDE = {
    "DPA-3.1-3M-FT": "${OH_MY_MLIP_HOME}/models/deepmd/dpa-3.1-3m-ft.pth",
    # PET single-point uses the exported metatomic .pt; metatrain's
    # `training.finetune.read_from` needs the training CHECKPOINT (.ckpt)
    # that install.sh keeps next to it (metatrain 2026.1 pet/trainer.py:105).
    "PET-OAM-XL": "${OH_MY_MLIP_HOME}/models/pet/pet-oam-xl-v1.0.0.ckpt",
}


@dataclass
class Context:
    family: str
    version: str
    finetune: dict
    resolved: dict
    out: Path
    epochs: int | None
    batch_size: int | None
    device: str
    dataset_paths: dict
    elements: list
    seed: int = 0
    settings: dict = field(default_factory=dict)  # resolved framework settings
    settings_origins: dict = field(default_factory=dict)  # origins of each setting

    def native(self, name: str, default: Any = None) -> Any:
        """Resolve a native setting by name or knob from the framework's settings.

        Retrieves the value of a native setting by either its native name
        (e.g. 'learning_rate', 'targets.<energy>.stress') or its common knob name
        (e.g. 'lr', 'include_stress'). The setting must have been resolved by
        ft_settings.resolve_settings().

        Args:
            name: Native setting name or knob name as defined in the framework's JSON config
            default: Default value to return if setting not found (None means not found)

        Returns:
            The resolved value for this setting, or default if not found

        Raises:
            ValueError if default is None and the setting has no resolved value
        """
        # Try direct lookup by native name first
        if name in self.settings:
            return self.settings[name]

        # If no default provided, raise error
        if default is None:
            origin = self.settings_origins.get(name)
            raise ValueError(
                f"[Context.native] {self.family}/{self.version}: setting '{name}' has no resolved value "
                f"(origin: {origin or 'unknown'}). Check that the setting exists and has a default, "
                f"ft_value, or user-supplied value."
            )

        return default


@dataclass
class CommandSpec:
    """What a builder hands back. `argv` is the single `exec`'d training
    command; `pre_steps` are argv lists run before it in the same `.sh`
    (under `set -eu`, so a failing prestage aborts the run); `extra_files`
    are additional artifacts (driver/prestage scripts) written next to the
    config; `extra_env` is exported in the `.sh` after the registry's
    `env_run` keys."""
    argv: list
    config_path: Path | None = None
    config_text: str | None = None
    pre_steps: list = field(default_factory=list)
    extra_files: dict = field(default_factory=dict)
    extra_env: dict = field(default_factory=dict)


# ── model/version/finetune resolution ────────────────────────────────────────
def load_finetune(model: str, version: str | None) -> tuple[str, str, dict, dict]:
    """Resolve `model` (family or version name) via the SAME name resolution
    as `oh_my_mlip.resolve()`, then read that version's `finetune` block
    straight out of models.json -- no separate/duplicated lookup logic."""
    resolved = reg.resolve(model, version)
    family, version = resolved["model"], resolved["version"]
    raw = reg.load_models()
    finetune = (raw[family]["versions"][version] or {}).get("finetune")
    if finetune is None:
        raise SystemExit(f"[ft_run] {family}/{version} has no 'finetune' block in models.json")
    return family, version, finetune, resolved


def entrypoint_bin(resolved: dict, name: str) -> str:
    """`<the resolved env's bin dir>/<name>` -- honors env adoption the same
    way `resolved['python']` already does (registry.resolve() rewrites
    'python' for an adopted env; every other console script in that env's
    bin/ moves with it)."""
    return str(Path(resolved["python"]).parent / name)


def _importable(python_bin: str, module: str) -> bool:
    if not Path(python_bin).exists():
        # Fresh clone: the env interpreter is not built yet, so no blocker fix
        # can be observed -- treat as not importable and let the caller's
        # refusal name the real remedy (install.sh) instead of tracing back.
        return False
    check = subprocess.run([python_bin, "-c", f"import {module}"], capture_output=True)
    return check.returncode == 0


def live_recheck_blockers(family: str, blockers: list[str], python_bin: str) -> list[str]:
    """Blockers whose fix has demonstrably already been applied are dropped;
    everything else is returned unchanged. See module docstring."""
    remaining = []
    for b in blockers:
        m = _PIP_INSTALL_RE.search(b)
        if m and _importable(python_bin, m.group(1)):
            continue
        if family in ("DeePMD", "DPA4") and "symlink" in b.lower():
            continue  # build_deepmd() performs this symlink itself
        remaining.append(b)
    return remaining


def resolve_foundation_checkpoint(version: str, resolved: dict) -> str:
    """The checkpoint/keyword this hub already trusts for `version` -- reused
    from `resolved['inference']` (the same weight validated for single-point)
    unless `_FOUNDATION_OVERRIDE` names a different FT-specific file."""
    override = _FOUNDATION_OVERRIDE.get(version)
    if override:
        return override.replace("${OH_MY_MLIP_HOME}", reg.home())
    for line in resolved["inference"]:
        m = _QUOTED_RE.search(line)
        if m:
            return m.group(1)
    raise SystemExit(f"[ft_run] could not extract a foundation checkpoint from {version}'s inference line")


def set_dotted(d: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


# ── dataset conversion ────────────────────────────────────────────────────────
def run_ft_dataset(dataset: Path, target: str, out: Path, split: float, seed: int,
                   python_bin: str | None = None) -> dict:
    # The model env is the one interpreter guaranteed to carry ase/numpy; the
    # ambient interpreter that launched ft_run.py carries no such guarantee.
    cmd = [
        python_bin or sys.executable, str(FT_DATASET),
        "--input", str(dataset),
        "--split", str(split), "--seed", str(seed),
        "--to", target, "--out", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"[ft_run] ft_dataset.py failed (exit {proc.returncode})")
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ── per-family command builders ─────────────────────────────────────────────────
def build_mace(ctx: Context) -> CommandSpec:
    foundation = resolve_foundation_checkpoint(ctx.version, ctx.resolved)
    variant_args = ctx.finetune.get("variant_args") or {}
    train_path, valid_path = ctx.dataset_paths["train"], ctx.dataset_paths.get("valid")

    # Read all resolved settings by native name from ctx.settings
    multiheads = ctx.settings.get("--multiheads_finetuning", False)
    lr = ctx.settings.get("--lr", 0.01)
    batch_size = ctx.settings.get("--batch_size", 10)
    max_num_epochs = ctx.settings.get("--max_num_epochs", 2048)
    energy_weight = ctx.settings.get("--energy_weight", 1.0)
    forces_weight = ctx.settings.get("--forces_weight", 100.0)
    stress_weight = ctx.settings.get("--stress_weight", 1.0)
    ema = ctx.settings.get("--ema", False)
    ema_decay = ctx.settings.get("--ema_decay", 0.99)
    default_dtype = ctx.settings.get("--default_dtype", "float64")
    scheduler_patience = ctx.settings.get("--scheduler_patience", 50)
    patience = ctx.settings.get("--patience", 2048)

    argv = [
        entrypoint_bin(ctx.resolved, "mace_run_train"),
        f"--name={ctx.version}",
        f"--foundation_model={foundation}",
        f"--multiheads_finetuning={str(multiheads).lower()}",
        f"--train_file={train_path}",
        *( [f"--valid_file={valid_path}"] if valid_path else ["--valid_fraction=0.1"] ),
        "--energy_key=REF_energy",
        "--forces_key=REF_forces",
        "--E0s=average",
        f"--lr={lr}",
        f"--batch_size={int(batch_size)}",
        f"--max_num_epochs={int(max_num_epochs)}",
        f"--energy_weight={float(energy_weight)}",
        f"--forces_weight={float(forces_weight)}",
        f"--stress_weight={float(stress_weight)}",
        f"--seed={int(ctx.seed)}",
        f"--ema={str(ema).lower()}",
        f"--ema_decay={float(ema_decay)}",
        f"--default_dtype={default_dtype}",
        f"--scheduler_patience={int(scheduler_patience)}",
        f"--patience={int(patience)}",
        f"--device={ctx.device}",
        f"--model_dir={ctx.out}",
        f"--checkpoints_dir={ctx.out}/checkpoints",
        f"--log_dir={ctx.out}/logs",
        f"--results_dir={ctx.out}/results",
    ]
    for k, v in variant_args.items():
        argv.append(f"{k}={v}")
    return CommandSpec(argv=argv)


_SEVENNET_PRESET = {
    # SevenNet-Omni ships no dedicated preset file (only mf_ompa_fine_tune.yaml
    # and the generic fine_tune.yaml exist in installed sevenn 0.12.2.dev0) --
    # it falls back to the generic preset, a documented best-effort.
    "SevenNet-MF-OMPA": "mf_ompa_fine_tune",
}
# SevenNet-MF-ompa is a MULTI-MODAL checkpoint: its preset's data section
# expects `load_trainset_path`/`load_<modality>_validset_path` as
# `[{data_modality: ..., file_list: [{file: ...}]}]`, NOT a plain path list —
# a plain list raises `TypeError: ... is not dict or str` deep in
# sevenn.train.modal_dataset. Versions absent
# here use the plain non-modal preset and need no modality name.
_SEVENNET_MODALITY = {"SevenNet-MF-OMPA": "mpa"}


def _sevenn_modal_path(modality: str, path: str) -> list:
    return [{"data_modality": modality, "file_list": [{"file": path}]}]


# The preset itself is NOT read at emit time (M9): `sevenn_preset` lives in
# the SevenNet env, and --emit-only must render without that env. ft_run
# writes the dotted-key PATCH it wants applied (sevennet_patch.json) and a
# prestage step that, inside the env and under the .sh's `set -eu`, prints
# the pinned upstream preset through the installed `sevenn_preset` binary,
# applies the patch, and writes input.yaml -- so the preset content is
# always the installed package's own (MIT), never a copy carried here.
_SEVENNET_PRESTAGE = '''\
"""Materialize input.yaml for a SevenNet fine-tune: the installed package's
own `sevenn_preset <name>` output with scripts/ft_run.py's patch applied.
Generated by scripts/ft_run.py; runs inside the SevenNet env before
`sevenn train`."""
import json
import pathlib
import subprocess

import yaml

PRESET_BIN = {preset_bin!r}
PRESET = {preset!r}
PATCH = pathlib.Path({patch!r})
CONFIG = pathlib.Path({config!r})

proc = subprocess.run([PRESET_BIN, PRESET], capture_output=True, text=True)
if proc.returncode != 0:
    raise SystemExit(f"[sevennet_prestage] {{PRESET_BIN}} {{PRESET}} failed (exit {{proc.returncode}}): {{proc.stderr}}")
cfg = yaml.safe_load(proc.stdout)
if not isinstance(cfg, dict):
    raise SystemExit(f"[sevennet_prestage] preset {{PRESET}} did not print a YAML mapping")

for dotted, value in json.loads(PATCH.read_text()).items():
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {{}})
    node[parts[-1]] = value

CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"[sevennet_prestage] preset={{PRESET}} patched_keys={{len(json.loads(PATCH.read_text()))}} -> {{CONFIG}}")
'''


def sevennet_patch(ctx: Context) -> dict:
    """The dotted-key patch applied on top of the upstream preset; insertion
    order is the application order (variant_args last, so they win)."""
    foundation = resolve_foundation_checkpoint(ctx.version, ctx.resolved)
    # Read resolved settings by native name from ctx.settings
    train_epoch = ctx.settings.get("train.epoch", 100)
    batch_size = ctx.settings.get("data.batch_size", 4)
    train_lr = ctx.settings.get("train.lr", 0.004)
    train_energy_weight = ctx.settings.get("train.energy_weight")
    train_force_weight = ctx.settings.get("train.force_weight", 1.0)
    train_stress_weight = ctx.settings.get("train.stress_weight")

    patch: dict = {
        "train.continue.checkpoint": foundation,
        "train.continue.reset_optimizer": True,
        "train.continue.reset_scheduler": True,
        "train.continue.reset_epoch": True,
        "train.epoch": int(train_epoch),
        # mf_ompa_fine_tune.yaml sets no `per_epoch` at all (unlike the generic
        # fine_tune.yaml's per_epoch: 10) -- on a short demo run (epochs=2)
        # that leaves only the epoch-0 pre-training snapshot on disk. Force a
        # checkpoint every epoch so a `--epochs N` run always produces one.
        "train.per_epoch": 1,
        # every preset ships its own random_seed (1 or 777); --seed replaces it
        "train.random_seed": int(ctx.seed),
        "train.lr": float(train_lr),
    }
    # Add energy and force weights if they are resolved
    if train_energy_weight is not None:
        patch["train.energy_weight"] = float(train_energy_weight)
    if train_force_weight is not None:
        patch["train.force_weight"] = float(train_force_weight)
    if train_stress_weight is not None:
        patch["train.stress_weight"] = float(train_stress_weight)

    modality = _SEVENNET_MODALITY.get(ctx.version)
    valid = ctx.dataset_paths.get("valid")
    if modality:
        patch["data.load_trainset_path"] = _sevenn_modal_path(modality, str(ctx.dataset_paths["train"]))
        if valid:
            patch[f"data.load_{modality}_validset_path"] = _sevenn_modal_path(modality, str(valid))
    else:
        patch["data.load_trainset_path"] = [str(ctx.dataset_paths["train"])]
        if valid:
            patch["data.load_validset_path"] = [str(valid)]
    patch["data.batch_size"] = int(batch_size)
    for k, v in (ctx.finetune.get("variant_args") or {}).items():
        patch[k] = v
    return patch


def build_sevennet(ctx: Context) -> CommandSpec:
    preset = _SEVENNET_PRESET.get(ctx.version, "fine_tune")
    patch_path = ctx.out / "sevennet_patch.json"
    config_path = ctx.out / "input.yaml"
    prestage_path = ctx.out / "sevennet_prestage.py"
    prestage_text = _SEVENNET_PRESTAGE.format(
        preset_bin=entrypoint_bin(ctx.resolved, "sevenn_preset"), preset=preset,
        patch=str(patch_path), config=str(config_path),
    )
    argv = [entrypoint_bin(ctx.resolved, "sevenn"), "train", str(config_path)]
    return CommandSpec(
        argv=argv,
        config_path=patch_path,
        config_text=json.dumps(sevennet_patch(ctx), indent=2) + "\n",
        pre_steps=[[ctx.resolved["python"], str(prestage_path)]],
        extra_files={prestage_path: prestage_text},
    )


def build_deepmd(ctx: Context) -> CommandSpec:
    foundation = resolve_foundation_checkpoint(ctx.version, ctx.resolved)
    foundation_path = Path(os.path.expandvars(foundation))
    ckpt_for_cli = foundation_path
    if foundation_path.suffix == ".pth":
        # `dp --pt` dispatches on suffix -- symlink to a .pt name (the
        # blocker `live_recheck_blockers` drops unconditionally for this
        # exact reason).
        link = ctx.out / (foundation_path.stem + ".pt")
        if not (link.is_symlink() or link.exists()):
            link.symlink_to(foundation_path)
        ckpt_for_cli = link

    # Resolved settings (user > official fine-tuning value > upstream default), by the
    # native names of each family's settings file: DeePMD nests them (training.*,
    # learning_rate.*, loss.*); DPA4's file lists the same argcheck keys unnested.
    dpa4 = ctx.family == "DPA4"

    def setting(deepmd_name: str, dpa4_name: str):
        return ctx.settings.get(dpa4_name if dpa4 else deepmd_name)

    numb_steps = setting("training.numb_steps", "numb_steps")
    numb_epoch = ctx.settings.get("numb_epoch") if dpa4 else None
    if numb_steps is None and numb_epoch is None:
        raise ValueError(f"{ctx.family}: training length is not set; pass --max-steps"
                         + (" or --epochs" if dpa4 else "") + " (deepmd-kit requires numb_steps)")
    start_lr = setting("learning_rate.start_lr", "start_lr")
    if start_lr is None:
        raise ValueError(f"{ctx.family}: learning rate has no default at the pinned deepmd-kit; pass --lr")
    train_batch_size = ctx.settings.get("training.training_data.batch_size", "auto") if not dpa4 else "auto"
    valid_batch_size = ctx.settings.get("training.validation_data.batch_size", "auto") if not dpa4 else "auto"
    loss_dict = {"type": "ener"}
    for key in ("start_pref_e", "limit_pref_e", "start_pref_f", "limit_pref_f", "start_pref_v", "limit_pref_v"):
        val = setting(f"loss.{key}", key)
        if val is not None:
            loss_dict[key] = float(val)

    train_systems = ctx.dataset_paths.get("train_systems") or []
    valid_systems = ctx.dataset_paths.get("valid_systems") or []
    training: dict = {
        "training_data": {"systems": train_systems, "batch_size": train_batch_size},
        "seed": int(ctx.seed),
        "disp_freq": 10,   # logging cadence, not a model setting
        "save_freq": 50,   # checkpoint cadence, not a model setting
    }
    if numb_steps is not None:
        training["numb_steps"] = int(numb_steps)
    if numb_epoch is not None:
        training["numb_epoch"] = int(numb_epoch)
    if valid_systems:
        training["validation_data"] = {"systems": valid_systems, "batch_size": valid_batch_size}

    input_json = {
        # descriptor/fitting_net left empty: --use-pretrain-script (below)
        # pulls the real model section from the checkpoint (same reason
        # DPA4's upstream cmd uses it -- the local architecture is not
        # otherwise known to a hand-written input.json).
        "model": {"type_map": ctx.elements, "descriptor": {}, "fitting_net": {}},
        "learning_rate": {k: v for k, v in (("type", setting("learning_rate.type", "type") or "exp"),
                                              ("decay_steps", setting("learning_rate.decay_steps", "decay_steps")),
                                              ("start_lr", float(start_lr)),
                                              ("stop_lr", setting("learning_rate.stop_lr", "stop_lr"))) if v is not None},
        "loss": loss_dict,
        "training": training,
    }
    config_path = ctx.out / "input.json"
    config_text = json.dumps(input_json, indent=2) + "\n"

    argv = [
        entrypoint_bin(ctx.resolved, "dp"), "--pt", "train", str(config_path),
        "--finetune", str(ckpt_for_cli), "--use-pretrain-script",
    ]
    return CommandSpec(argv=argv, config_path=config_path, config_text=config_text)


# ── GRACE (tensorpotential 0.5.3, `gracemaker input.yaml`) ────────────────────
# Source of every key below: the installed tensorpotential/cli/gracemaker.py
# (finetune_foundation_model -> get_or_download_checkpoint(), reduce_elements
# -> convert_model_reduce_elements(), output dir seed/<seed>/, unconditional
# `final_model` saved_model export after training, gracemaker.py:578),
# cli/data.py (data.filename / test_filename / reference_energy, extxyz read
# through ase with calculator energy+forces), cli/prepare.py (loss `type`
# square|huber) and resources/input_template.yaml (fit section keys).
def build_grace(ctx: Context) -> CommandSpec:
    train_path, valid_path = ctx.dataset_paths["train"], ctx.dataset_paths.get("valid")
    # Read resolved settings by native name from ctx.settings
    maxiter = ctx.settings.get("maxiter", 500)
    batch_size = ctx.settings.get("batch_size", 8)
    test_batch_size = ctx.settings.get("test_batch_size", 1)
    opt_params_lr = ctx.settings.get("opt_params.learning_rate", 0.01)
    opt_params_use_ema = ctx.settings.get("opt_params.use_ema", False)
    opt_params_ema_momentum = ctx.settings.get("opt_params.ema_momentum", 0.99)
    energy_weight = ctx.settings.get("loss.energy.weight", 1.0)
    forces_weight = ctx.settings.get("loss.forces.weight", 100.0)
    loss_stress = ctx.settings.get("loss.stress")
    loss_stress_weight = ctx.settings.get("loss.stress.weight")
    float_dtype = ctx.settings.get("float_dtype", "float64")
    stop_at_min = ctx.settings.get("stop_at_min", False)

    cfg = {
        "seed": int(ctx.seed),
        # gracemaker overwrites `cutoff` from the foundation model.yaml (rcut)
        # once the checkpoint is loaded (add_loaded_model_parameter); the key
        # is still required by the input schema.
        "cutoff": 6.0,
        "data": {
            "filename": str(train_path),
            **({"test_filename": str(valid_path)} if valid_path else {}),
            "reference_energy": 0,
        },
        "potential": {
            # The registry version key IS the upstream foundation-model name
            # (tensorpotential.calculator.foundation_models.MODELS_METADATA).
            "finetune_foundation_model": ctx.version,
            # Restrict the chemical embedding to the elements in the data.
            "reduce_elements": True,
        },
        "float_dtype": float_dtype,
        "fit": {
            "loss": {
                "energy": {"weight": float(energy_weight), "type": "huber", "delta": 0.01},
                "forces": {"weight": float(forces_weight), "type": "huber", "delta": 0.01},
            },
            "maxiter": int(maxiter),
            "optimizer": "Adam",
            "opt_params": {
                "learning_rate": float(opt_params_lr), "amsgrad": True, "use_ema": bool(opt_params_use_ema),
                "ema_momentum": float(opt_params_ema_momentum), "weight_decay": None, "clipvalue": 1.0,
            },
            # Bug fix: gracemaker expects scheduler_params for Adam optimizer even when
            # scheduler is not specified; values from tensorpotential 0.5.3 cli/train_callbacks.py
            # DEFAULT_VALUES (unconditionally read by train_adam, tensorpotential/cli/train.py:336)
            "scheduler_params": {
                "minimum_learning_rate": 1e-4,
                "warmup_epochs": 0,
                "cold_learning_rate": 1e-7,
                "reduction_factor": 0.8,
                "patience": 10,
                "monitor": "test_loss",
                "cooldown": 0,
            },
            "batch_size": int(batch_size),
            "test_batch_size": int(test_batch_size),
            "jit_compile": True,
            "eval_init_stats": True,
            "checkpoint_freq": 1,
            "progressbar": False,
            "train_shuffle": True,
            "stop_at_min": stop_at_min,
        },
    }
    # Add stress loss if specified
    if loss_stress is not None:
        cfg["fit"]["loss"]["stress"] = loss_stress
    if loss_stress_weight is not None:
        cfg["fit"]["loss"]["stress"] = {"weight": float(loss_stress_weight)}

    config_path = ctx.out / "input.yaml"
    config_text = yaml.safe_dump(cfg, sort_keys=False)
    argv = [entrypoint_bin(ctx.resolved, "gracemaker"), str(config_path)]
    # GRACE_CACHE roots BOTH the saved-model cache (<cache>/<model>, which
    # is exactly the hub's models/grace/<model> layout) and the fine-tune
    # checkpoint cache (<cache>/checkpoints/<model>) -- pin it to the hub so
    # the checkpoint gracemaker fetches on first use lands in a
    # deterministic, hub-owned location. Prestage without a training run:
    #   GRACE_CACHE=$OH_MY_MLIP_HOME/models/grace grace_models checkpoint <model>
    extra_env = {"GRACE_CACHE": str(Path(reg.home()) / "models" / "grace")}
    return CommandSpec(argv=argv, config_path=config_path, config_text=config_text, extra_env=extra_env)


# ── PET (metatrain 2026.1, `mtt train options.yaml -o model-ft.pt`) ───────────
# Source: installed metatrain/cli/train.py (-o/-e flags, final
# `<output>.ckpt` + exported `<output>` written unconditionally after
# training, train.py:568-590), share/base_hypers.py (training_set /
# validation_set / test_set dataset spec, systems/targets keys),
# pet/documentation.py (training.finetune {read_from, method, config,
# inherit_heads}; batch_size / num_epochs / checkpoint_interval names),
# pet/modules/finetuning.py (method in {full, heads, lora}), and
# utils/data/readers/ase.py (energy read from atoms.info[key], forces from
# atoms.arrays[key]; a SinglePointCalculator's results are copied into
# info["energy"] / arrays["forces"] first).
def _pet_dataset_spec(path: str) -> dict:
    return {
        "systems": {"read_from": path, "reader": "ase", "length_unit": "angstrom"},
        "targets": {
            "energy": {
                "quantity": "energy", "read_from": path, "reader": "ase",
                "key": "energy", "unit": "eV",
                "forces": {"read_from": path, "reader": "ase", "key": "forces"},
            }
        },
    }


def build_pet(ctx: Context) -> CommandSpec:
    foundation = resolve_foundation_checkpoint(ctx.version, ctx.resolved)
    train_path, valid_path = ctx.dataset_paths["train"], ctx.dataset_paths.get("valid")

    # Read resolved settings by native name from ctx.settings
    batch_size = ctx.settings.get("batch_size", 32)
    num_epochs = ctx.settings.get("num_epochs", 1000)
    learning_rate = ctx.settings.get("learning_rate", 1e-3)
    base_precision = ctx.settings.get("base_precision", 32)
    include_stress = ctx.settings.get("targets.<energy>.stress", False)
    loss_energy_weight = ctx.settings.get("loss.<target>.weight")
    loss_force_weight = ctx.settings.get("loss.<target>.gradients.positions.weight")
    loss_stress_weight = ctx.settings.get("loss.<target>.gradients.strain.weight")

    cfg = {
        "seed": int(ctx.seed),
        "device": "gpu" if ctx.device == "cuda" else "cpu",
        "base_precision": int(base_precision),
        "architecture": {
            "name": "pet",
            "training": {
                "batch_size": int(batch_size),
                "num_epochs": int(num_epochs),
                "checkpoint_interval": 1,
                "log_interval": 1,
                "learning_rate": float(learning_rate),
                "finetune": {"read_from": foundation, "method": "full"},
            },
        },
        "training_set": _pet_dataset_spec(str(train_path)),
        "validation_set": _pet_dataset_spec(str(valid_path)) if valid_path else 0.1,
        "test_set": 0.0,
    }
    # Add stress to the energy target if enabled
    if include_stress:
        cfg["training_set"]["targets"]["energy"]["stress"] = {
            "read_from": str(train_path), "reader": "ase", "key": "stress"
        }
    # Add loss weights if specified
    if loss_energy_weight is not None:
        cfg["training_set"]["targets"]["energy"]["loss_weight"] = float(loss_energy_weight)
    if loss_force_weight is not None:
        cfg["training_set"]["targets"]["energy"]["forces"]["loss_weight"] = float(loss_force_weight)
    if loss_stress_weight is not None and include_stress:
        cfg["training_set"]["targets"]["energy"]["stress"]["loss_weight"] = float(loss_stress_weight)

    config_path = ctx.out / "options.yaml"
    config_text = yaml.safe_dump(cfg, sort_keys=False)
    argv = [
        entrypoint_bin(ctx.resolved, "mtt"), "train", str(config_path),
        "-o", str(ctx.out / "model-ft.pt"),
        "-e", str(ctx.out / "extensions"),
    ]
    return CommandSpec(argv=argv, config_path=config_path, config_text=config_text)


# ── NequIP / Allegro (nequip 0.17.1 / nequip 0.15.0 + nequip-allegro 0.7.1) ──
# Source: installed nequip/scripts/train.py (hydra config in cwd, required
# top-level keys run / data / trainer / training_module, `model` nested
# under training_module in BOTH versions), nequip/train/lightning.py
# (EMALightningModule args model/loss/val_metrics/optimizer; logged metric
# names `val{idx}_epoch/<name>` with `weighted_sum`), train/metrics_manager.py
# (EnergyForceLoss coeffs keyed total_energy/forces, EnergyForceMetrics coeffs
# keyed total_energy_rmse/forces_rmse), data/datamodule/_ase_datamodule.py
# (ASEDataModule seed/train_file_path/val_file_path/transforms +
# train_dataloader/val_dataloader dicts), data/transforms
# (ChemicalSpeciesToAtomTypeMapper(chemical_symbols=...) accepted by both
# versions; NeighborListTransform(r_max=...)), model/saved_models/package.py
# (ModelFromPackage(package_path) with the `.nequip.zip` extension enforced),
# model/saved_models/load_utils.py (`_get_model_file_path` resolves a
# `nequip.net:<group>/<model>:<version>` ID; 0.17.1 caches under
# NEQUIP_CACHE_DIR, 0.15.0 downloads to a temp file every call), and
# nn/graph_model.py (GraphModel.type_names / .metadata["r_max"], both
# versions). The `${type_names_from_package:}` / `${cutoff_radius_from_package:}`
# resolvers exist only in 0.17.1 (utils/resolvers.py), so the type names and
# r_max are extracted ONCE by a prestage step that also pins the package to a
# hub-owned path -- the same config then works under 0.15.0 (Allegro).
_NEQUIP_PRESTAGE = '''\
"""Prestage the foundation package for a NequIP-framework fine-tune and fill
the config with the package's own type names and cutoff. Generated by
scripts/ft_run.py; runs inside the family env before nequip-train."""
import json
import pathlib
import shutil

import yaml

MODEL_ID = {model_id!r}
PACKAGE = pathlib.Path({package!r})
TEMPLATE = pathlib.Path({template!r})
CONFIG = pathlib.Path({config!r})

if not PACKAGE.exists():
    from nequip.model.saved_models.load_utils import _get_model_file_path
    PACKAGE.parent.mkdir(parents=True, exist_ok=True)
    with _get_model_file_path(MODEL_ID) as fetched:
        shutil.copyfile(fetched, PACKAGE)
    print(f"[nequip_prestage] fetched {{MODEL_ID}} -> {{PACKAGE}}")
else:
    print(f"[nequip_prestage] using existing package {{PACKAGE}}")

from nequip.model import ModelFromPackage
model = ModelFromPackage(str(PACKAGE))
type_names = list(model.type_names)
r_max = float(model.metadata["r_max"])
print(f"[nequip_prestage] type_names={{len(type_names)}} r_max={{r_max}}")

cfg = yaml.safe_load(TEMPLATE.read_text())
cfg["model_type_names"] = type_names
cfg["cutoff_radius"] = r_max
cfg["training_module"]["model"]["package_path"] = str(PACKAGE)
CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
(CONFIG.parent / "foundation.json").write_text(json.dumps(
    {{"model_id": MODEL_ID, "package_path": str(PACKAGE), "type_names": type_names, "r_max": r_max}}, indent=2))
'''


def nequip_net_model_id(resolved: dict) -> str:
    """`weights_source` for these variants is the nequip.net model page
    (https://www.nequip.net/models/<group>/<model>:<version>); the package
    loader wants the ID form `nequip.net:<group>/<model>:<version>`."""
    src = str(resolved.get("weights_source") or "")
    marker = "nequip.net/models/"
    if marker not in src:
        raise SystemExit(f"[ft_run] {resolved.get('version')}: weights_source {src!r} is not a nequip.net model page")
    return "nequip.net:" + src.split(marker, 1)[1].strip("/")


def build_nequip_framework(ctx: Context) -> CommandSpec:
    model_id = nequip_net_model_id(ctx.resolved)
    package = Path(reg.home()) / "models" / ctx.resolved["env"] / f"{ctx.version}.nequip.zip"
    train_path, valid_path = ctx.dataset_paths["train"], ctx.dataset_paths.get("valid")
    if not valid_path:
        # the ModelCheckpoint monitor below (val0_epoch/weighted_sum) is
        # unconditional in nequip's trainer: without a validation file the
        # run dies inside Lightning instead of here.
        raise SystemExit(f"[ft_run] {ctx.family}/{ctx.version}: the NequIP-framework fine-tune needs a "
                         f"validation split (none was produced -- see <out>/data/conversion.json)")

    # Read resolved settings by native name from ctx.settings
    batch_size = ctx.settings.get("data.train_dataloader.batch_size", 8)
    max_epochs = ctx.settings.get("trainer.max_epochs")
    max_steps = ctx.settings.get("trainer.max_steps")
    lr = ctx.settings.get("training_module.optimizer.lr", 0.001)
    energy_weight = ctx.settings.get("training_module.loss.coeffs.total_energy", 1.0)
    force_weight = ctx.settings.get("training_module.loss.coeffs.forces", 1.0)
    stress_weight = ctx.settings.get("training_module.loss.coeffs.stress")
    include_stress_setting_name = "include stress (training_module.loss._target_: nequip.train.EnergyForceStressLoss)"
    include_stress = ctx.settings.get(include_stress_setting_name, False)
    ema_decay = ctx.settings.get("training_module.ema_decay", 0.99)

    # Choose loss class based on whether stress is included
    if include_stress:
        loss_target = "nequip.train.EnergyForceStressLoss"
        loss_coeffs = {"total_energy": float(energy_weight), "forces": float(force_weight),
                       "stress": float(stress_weight) if stress_weight is not None else 0.001}
        val_metrics_target = "nequip.train.EnergyForceStressMetrics"
        val_metrics_coeffs = {"total_energy_rmse": 1.0, "forces_rmse": 1.0, "stress_rmse": 1.0}
    else:
        loss_target = "nequip.train.EnergyForceLoss"
        loss_coeffs = {"total_energy": float(energy_weight), "forces": float(force_weight)}
        val_metrics_target = "nequip.train.EnergyForceMetrics"
        val_metrics_coeffs = {"total_energy_rmse": 1.0, "forces_rmse": 1.0}

    # Resolve max_epochs: use resolved max_epochs if available, otherwise fallback to ctx.epochs
    if max_epochs is None:
        max_epochs = ctx.epochs

    template = {
        "run": ["train"],
        # filled by the prestage step from the package itself
        "cutoff_radius": None,
        "model_type_names": None,
        "data": {
            "_target_": "nequip.data.datamodule.ASEDataModule",
            "seed": int(ctx.seed),
            "train_file_path": str(train_path),
            **({"val_file_path": str(valid_path)} if valid_path else {}),
            "transforms": [
                {"_target_": "nequip.data.transforms.ChemicalSpeciesToAtomTypeMapper",
                 "chemical_symbols": "${model_type_names}"},
                {"_target_": "nequip.data.transforms.NeighborListTransform",
                 "r_max": "${cutoff_radius}"},
            ],
            "train_dataloader": {"_target_": "torch.utils.data.DataLoader",
                                 "batch_size": int(batch_size), "shuffle": True},
            "val_dataloader": {"_target_": "torch.utils.data.DataLoader",
                               "batch_size": int(batch_size)},
        },
        "trainer": {
            "_target_": "lightning.Trainer",
            "accelerator": "gpu" if ctx.device == "cuda" else "cpu",
            "devices": 1,
            "max_epochs": int(max_epochs),
            "callbacks": [
                {"_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                 "dirpath": str(ctx.out / "checkpoints"),
                 "filename": "best",
                 "monitor": "val0_epoch/weighted_sum",
                 "save_last": True},
            ],
        },
        "training_module": {
            "_target_": "nequip.train.EMALightningModule",
            "model": {"_target_": "nequip.model.ModelFromPackage", "package_path": None},
            "loss": {"_target_": loss_target, "per_atom_energy": True, "coeffs": loss_coeffs},
            "val_metrics": {"_target_": val_metrics_target, "coeffs": val_metrics_coeffs},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": float(lr)},
            "ema_decay": float(ema_decay),
        },
        "hydra": {"run": {"dir": str(ctx.out / "hydra")}},
    }
    # Add max_steps if resolved
    if max_steps is not None:
        template["trainer"]["max_steps"] = int(max_steps)

    template_path = ctx.out / "config_template.yaml"
    config_path = ctx.out / "config.yaml"
    prestage_path = ctx.out / "nequip_prestage.py"
    prestage_text = _NEQUIP_PRESTAGE.format(
        model_id=model_id, package=str(package), template=str(template_path), config=str(config_path),
    )
    argv = [entrypoint_bin(ctx.resolved, "nequip-train"), "-cp", str(ctx.out), "-cn", "config.yaml"]
    return CommandSpec(
        argv=argv,
        config_path=template_path,
        config_text=yaml.safe_dump(template, sort_keys=False),
        pre_steps=[[ctx.resolved["python"], str(prestage_path)]],
        extra_files={prestage_path: prestage_text},
        extra_env={"NEQUIP_CACHE_DIR": str(Path(reg.home()) / "models" / ctx.resolved["env"] / "model_cache")},
    )


# ── MatterSim (mattersim 1.2.1, torchrun finetune_mattersim.py) ───────────────
# Source: installed mattersim/training/finetune_mattersim.py (reads
# LOCAL_RANK from the environment at import -> must run under torchrun;
# argparse flags --load_model_path/--train_data_path/--valid_data_path/
# --save_path/--save_checkpoint (BooleanOptionalAction, default False)/
# --epochs/--batch_size/--lr/--include_forces/--device/--seed/--run_name;
# extxyz read via ase with get_potential_energy()/get_forces(); model built
# with Potential.from_checkpoint(load_model_path, load_training_state=False))
# and forcefield/potential.py (save_model writes <save_path>/best_model.pth
# and last_model.pth only when save_checkpoint is True; from_checkpoint
# resolves the 'mattersim-v1.0.0-5m' name case-insensitively to
# ~/.local/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth).
def build_mattersim(ctx: Context) -> CommandSpec:
    foundation = resolve_foundation_checkpoint(ctx.version, ctx.resolved)
    train_path, valid_path = ctx.dataset_paths["train"], ctx.dataset_paths.get("valid")
    # Native names from finetune/settings/MatterSim.json. There is no energy
    # loss weight upstream (finetune_mattersim.py:216-217 expose only
    # --force_loss_ratio / --stress_loss_ratio; energy is implicitly 1).
    epochs = ctx.native("--epochs", default=ctx.epochs)
    batch_size = ctx.native("--batch_size", default=ctx.batch_size)
    lr = ctx.native("--lr", default=2e-4)
    force_loss_ratio = ctx.native("--force_loss_ratio", default=1.0)
    stress_loss_ratio = ctx.native("--stress_loss_ratio", default=0.1)
    include_stresses = bool(ctx.native("--include_stresses", default=False))
    argv = [
        entrypoint_bin(ctx.resolved, "torchrun"), "--nproc_per_node=1", "--standalone",
        "-m", "mattersim.training.finetune_mattersim",
        "--load_model_path", foundation,
        "--train_data_path", str(train_path),
        *(["--valid_data_path", str(valid_path)] if valid_path else []),
        "--save_path", str(ctx.out / "results"),
        "--save_checkpoint",
        f"--epochs={int(epochs)}",
        f"--batch_size={int(batch_size)}",
        f"--lr={float(lr)}",
        "--include_forces",
        f"--force_loss_ratio={float(force_loss_ratio)}",
        # BooleanOptionalAction (finetune_mattersim.py:210-215): emit the
        # explicit form either way so the choice is visible in the .sh
        "--include_stresses" if include_stresses else "--no-include_stresses",
        f"--stress_loss_ratio={float(stress_loss_ratio)}",
        "--device", ctx.device,
        "--seed", str(int(ctx.seed)),
        "--run_name", ctx.version,
    ]
    return CommandSpec(argv=argv)


# ── TACE (tace 0.2.0, `tace-train -cn tace`) ──────────────────────────────────
# Source: installed tace/scripts/train.py (@hydra.main(config_path=cwd,
# config_name="tace"); `finetune_from_model` -> finetune(cfg) loads the
# checkpoint, statistics and model config come from it, `finetune:` empty +
# no finetune_config.yaml in cwd => full fine-tune with a logged warning),
# tace/lightning/lit_model.py (load_tace accepts .ckpt/.pt/.pth; synth_metric
# optional), tace/lightning/trainer.py (at least one ModelCheckpoint callback
# required), tace/dataset/read.py (calculator energy/forces copied into
# info/arrays under keys.energy_key/forces_key), tace/dataset/quantity.py
# (target property = loss.loss_property), hard-indexed keys enumerated from
# the package (misc.global_seed, misc.LossSkipController, trainer.precision,
# dataset.train_dataloader/valid_dataloader, optimizer, scheduler, loss,
# model.config.fidelity), and the upstream example/train/tace.yaml (key
# layout for callbacks / dataset / scheduler.extra / synth_metric). The
# foundation file path comes from tace.foundations.tace_foundations[<name>]
# (HF download into its cache on first use), resolved by a prestage step.
_TACE_PRESTAGE = '''\
"""Resolve the TACE foundation checkpoint path and pin it into tace.yaml.
Generated by scripts/ft_run.py; runs inside the tace env before tace-train."""
import pathlib

import yaml

from tace.foundations import tace_foundations

NAME = {name!r}
CONFIG = pathlib.Path({config!r})

path = pathlib.Path(tace_foundations[NAME]).resolve()
cfg = yaml.safe_load(CONFIG.read_text())
cfg["finetune_from_model"] = str(path)
CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"[tace_prestage] finetune_from_model = {{path}}")
'''


def build_tace(ctx: Context) -> CommandSpec:
    foundation_name = resolve_foundation_checkpoint(ctx.version, ctx.resolved)
    train_path, valid_path = ctx.dataset_paths["train"], ctx.dataset_paths.get("valid")

    # Bug fix: TACE LoRA fine-tuning is broken at the pinned commit
    # Only allow full fine-tuning
    if ctx.settings.get("finetune_method") == "lora":
        raise SystemExit(f"[ft_run] {ctx.family}/{ctx.version}: LoRA fine-tuning is not supported "
                         f"(broken at pinned commit tace 0.2.0) -- use full fine-tuning only")

    # Native names from finetune/settings/TACE.json
    epochs = ctx.native("trainer.max_epochs", default=ctx.epochs)
    precision = ctx.native("trainer.precision", default=32)
    lr = ctx.native("optimizer.lr", default=1e-4)
    energy_weight = ctx.native("loss.loss_property_weights[energy]", default=1.0)
    forces_weight = ctx.native("loss.loss_property_weights[forces]", default=5.0)
    stress_weight = ctx.native("loss.loss_property_weights[stress]", default=1.0)
    # the batch_size knob lands on dataset.train_dataloader as a plain int; a
    # --set of the whole dataloader dict carries its own batch_size
    train_loader_setting = ctx.native("dataset.train_dataloader", default=ctx.batch_size)
    batch_size = (train_loader_setting.get("batch_size", ctx.batch_size)
                  if isinstance(train_loader_setting, dict) else train_loader_setting)
    # include_stress knob -> loss.loss_property (True/False from the flag, or
    # an explicit property list via --set)
    loss_property_setting = ctx.native("loss.loss_property", default=False)
    include_stress = ("stress" in loss_property_setting
                      if isinstance(loss_property_setting, (list, tuple)) else bool(loss_property_setting))

    loss_property = ["energy", "forces"]
    loss_function_name = ["mse_energy_per_atom", "mse_forces"]
    loss_property_weights = [float(energy_weight), float(forces_weight)]
    keys = {"energy_key": "energy", "forces_key": "forces"}
    if include_stress:
        # tace/utils/loss/mse_fn.py:57 mse_stress; dataset/quantity.py maps
        # `<property>_key` -> info key, so stress_key reads ASE's stress
        loss_property.append("stress")
        loss_function_name.append("mse_stress")
        loss_property_weights.append(float(stress_weight))
        keys["stress_key"] = "stress"

    monitor = "val/synth_metric"
    loader = {"_target_": "torch_geometric.loader.DataLoader", "drop_last": False, "num_workers": 0}
    cfg = {
        "defaults": ["_self_"],
        "resume_from_model": None,
        "finetune_from_model": None,  # filled by tace_prestage.py
        "finetune": None,             # + no finetune_config.yaml => full fine-tune (not LoRA)
        "misc": {
            "project_name": ctx.version,
            "global_seed": int(ctx.seed),
            "device": ctx.device,
            "allow_tf32": False,
            "ignore_warning": True,
            "log_level": "info",
            "env": {"WANDB_MODE": "offline"},
            "LossSkipController": {"enable": False, "manual_threshold": 1e6, "start_step": 1000,
                                   "ema_window": 1000, "multiplier": 1e6, "skip_nan": True, "skip_large": True},
        },
        "trainer": {
            "_target_": "lightning.Trainer",
            "num_nodes": 1,
            "accelerator": "gpu" if ctx.device == "cuda" else "cpu",
            "devices": 1,
            "max_epochs": int(epochs),
            "min_epochs": 1,
            "precision": precision,
            "strategy": "auto",
            "gradient_clip_val": 10.0,
            "enable_progress_bar": False,
            "log_every_n_steps": 1,
            "enable_model_summary": False,
            "enable_checkpointing": True,
            "check_val_every_n_epoch": 1,
            "inference_mode": False,
            "deterministic": False,
        },
        "callbacks": {
            "ema": {"_target_": "tace.utils.callbacks.EMACallback", "decay": 0.999, "use_num_updates": True},
            "checkpoint_epoch": {
                "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                "dirpath": "checkpoints_epoch",
                "filename": "TACE-{epoch}-{step}",
                "save_top_k": -1,
                "save_last": True,
                "every_n_epochs": 1,
                "save_weights_only": False,
                "auto_insert_metric_name": False,
                "verbose": False,
            },
        },
        "dataset": {
            "type": "ase",
            "augmentation": [],
            "split_seed": int(ctx.seed),
            "train_file": str(train_path),
            "valid_file": str(valid_path) if valid_path else None,
            "valid_ratio": 0.1,
            "valid_from_index": False,
            "no_valid_set": False,
            "neighborlist_backend": "matscipy",
            "storage_mode": "memory",
            "keys": keys,
            "train_dataloader": {**loader, "batch_size": int(batch_size), "shuffle": True},
            "statistics_dataloader": {"batch_size": int(batch_size)},
            "valid_dataloader": {**loader, "batch_size": int(batch_size), "shuffle": False},
            "test_dataloader": "${dataset.valid_dataloader}",
        },
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": float(lr), "weight_decay": 1e-8},
        "scheduler": {
            "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "mode": "min", "factor": 0.5, "patience": 25,
            "extra": {"monitor": monitor, "interval": "epoch", "frequency": 1},
        },
        "synth_metric": {"monitor_metric_name": monitor,
                         "val/energy_per_atom_mae": 1.0, "val/forces_mae": 1.0},
        "loss": {
            "_target_": "tace.utils.loss.NormalLoss",
            "loss_property": loss_property,
            "loss_function_name": loss_function_name,
            "loss_property_weights": loss_property_weights,
            "loss_function_kwargs": [{} for _ in loss_property],
        },
        # replaced from the checkpoint at fine-tune time; `fidelity` is read
        # before that replacement so it must be present.
        "model": {"config": {"cutoff": 6.0, "max_neighbors": None,
                             "fidelity": [{"name": "PBE", "atomic_energy": None, "magnetic_scale": None}],
                             "universal_embedding": {}}},
    }
    config_path = ctx.out / "tace.yaml"
    prestage_path = ctx.out / "tace_prestage.py"
    prestage_text = _TACE_PRESTAGE.format(name=foundation_name, config=str(config_path))
    argv = [entrypoint_bin(ctx.resolved, "tace-train"), "-cn", "tace"]
    return CommandSpec(
        argv=argv,
        config_path=config_path,
        config_text=yaml.safe_dump(cfg, sort_keys=False),
        pre_steps=[[ctx.resolved["python"], str(prestage_path)]],
        extra_files={prestage_path: prestage_text},
    )


# ── CHGNet (chgnet 0.4.0, python API driver) ──────────────────────────────────
# Source: installed chgnet/trainer/trainer.py (Trainer(model, targets,
# optimizer, scheduler, criterion, epochs, learning_rate, use_device);
# train(train_loader, val_loader, save_dir=...) writes epoch<N>_*.pth.tar
# and copies the best-val-energy epoch to bestE_epoch<N>_*.pth.tar;
# Trainer.save() stores {"model": CHGNet.as_dict(), ...}), chgnet/data/
# dataset.py (StructureData(structures, energies, forces) over pymatgen
# Structures; get_loader(dataset, batch_size=...)), chgnet/model/model.py
# (CHGNet.load(model_name="0.3.0", use_device=...); CHGNet.from_file(path)
# reads state["model"]). CHGNet's energy target is eV/atom (dataset.py
# energy_key defaults "energy_per_atom"), so the driver divides the canonical
# extxyz total energy by the atom count.
_CHGNET_DRIVER = '''\
"""CHGNet fine-tune driver generated by scripts/ft_run.py (python API only;
chgnet ships no training CLI). Runs inside the chgnet env."""
import pathlib
import random

import numpy as np
import torch
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from chgnet.data.dataset import StructureData, get_loader
from chgnet.model import CHGNet
from chgnet.trainer import Trainer

TRAIN = {train!r}
VALID = {valid!r}
SAVE_DIR = {save_dir!r}
MODEL_NAME = {model_name!r}
EPOCHS = {epochs}
BATCH_SIZE = {batch_size}
LEARNING_RATE = {learning_rate!r}
TARGETS = {targets!r}
ENERGY_LOSS_RATIO = {energy_loss_ratio!r}
FORCE_LOSS_RATIO = {force_loss_ratio!r}
STRESS_LOSS_RATIO = {stress_loss_ratio!r}
DEVICE = {device!r}
SEED = {seed}

# get_loader() builds a shuffle=True DataLoader on torch's global RNG and
# Trainer's own `if data_seed:` skips seed 0 -- seed everything here first,
# then hand the same value to Trainer so its record matches.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def dataset(path):
    structures, energies, forces, stresses = [], [], [], []
    for atoms in read(path, index=":"):
        structures.append(AseAtomsAdaptor.get_structure(atoms))
        energies.append(float(atoms.get_potential_energy()) / len(atoms))  # eV/atom
        forces.append(atoms.get_forces().tolist())
        if "s" in TARGETS:
            # CHGNet labels are GPa with ASE's sign (model/dynamics.py:156
            # returns prediction["s"] / 160.21766208 as the ASE stress)
            stresses.append((atoms.get_stress(voigt=False) * 160.21766208).tolist())
    return StructureData(structures=structures, energies=energies, forces=forces,
                         stresses=stresses or None, shuffle=False)


train_loader = get_loader(dataset(TRAIN), batch_size=BATCH_SIZE)
val_loader = get_loader(dataset(VALID), batch_size=BATCH_SIZE) if VALID else train_loader
model = CHGNet.load(model_name=MODEL_NAME, use_device=DEVICE)
trainer = Trainer(model=model, targets=TARGETS, energy_loss_ratio=ENERGY_LOSS_RATIO,
                  force_loss_ratio=FORCE_LOSS_RATIO, stress_loss_ratio=STRESS_LOSS_RATIO,
                  optimizer="Adam", scheduler="CosLR", criterion="MSE",
                  epochs=EPOCHS, learning_rate=LEARNING_RATE, use_device=DEVICE, torch_seed=SEED, data_seed=SEED)
pathlib.Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
trainer.train(train_loader, val_loader, save_dir=SAVE_DIR)
print(f"[chgnet_finetune] checkpoints in {{SAVE_DIR}}: " + ", ".join(sorted(p.name for p in pathlib.Path(SAVE_DIR).glob("*.pth.tar"))))
'''


def build_chgnet(ctx: Context) -> CommandSpec:
    model_name = resolve_foundation_checkpoint(ctx.version, ctx.resolved)  # '0.3.0'
    train_path, valid_path = ctx.dataset_paths["train"], ctx.dataset_paths.get("valid")
    # Native names from finetune/settings/CHGNet.json (Trainer kwargs)
    epochs = ctx.native("epochs", default=ctx.epochs)
    batch_size = ctx.native("batch_size", default=ctx.batch_size)
    learning_rate = ctx.native("learning_rate", default=1e-3)
    energy_loss_ratio = ctx.native("energy_loss_ratio", default=1.0)
    force_loss_ratio = ctx.native("force_loss_ratio", default=1.0)
    stress_loss_ratio = ctx.native("stress_loss_ratio", default=0.1)
    # include_stress knob -> targets. The official "efsm" needs magmom labels
    # the canonical extxyz does not carry, and stress labels it may not carry,
    # so only a user choice turns on "s": True/False from the flag, or an
    # explicit string via --set (its "m" is still dropped).
    targets_setting = ctx.native("targets", default="ef")
    if ctx.settings_origins.get("targets") != "user":
        targets = "ef"
    elif isinstance(targets_setting, bool):
        targets = "efs" if targets_setting else "ef"
    else:
        targets = "efs" if "s" in str(targets_setting) else "ef"
    driver_path = ctx.out / "finetune_chgnet.py"
    driver_text = _CHGNET_DRIVER.format(
        train=str(train_path), valid=str(valid_path) if valid_path else None,
        save_dir=str(ctx.out / "chgnet_ft"), model_name=model_name,
        epochs=int(epochs), batch_size=int(batch_size), device=ctx.device,
        learning_rate=float(learning_rate), targets=targets,
        energy_loss_ratio=float(energy_loss_ratio), force_loss_ratio=float(force_loss_ratio),
        stress_loss_ratio=float(stress_loss_ratio),
        seed=int(ctx.seed),
    )
    argv = [ctx.resolved["python"], str(driver_path)]
    return CommandSpec(argv=argv, config_path=driver_path, config_text=driver_text)


# Real builders only. A family absent here is refused at EXIT_NO_BUILDER by
# main() before any artifact is written -- there is no generic renderer.
BUILDERS = {
    "MACE": build_mace,
    "SevenNet": build_sevennet,
    "DeePMD": build_deepmd,
    "DPA4": build_deepmd,
    "GRACE": build_grace,
    "PET": build_pet,
    "NequIP": build_nequip_framework,
    "Allegro": build_nequip_framework,
    "MatterSim": build_mattersim,
    "TACE": build_tace,
    "CHGNet": build_chgnet,
}

# The ONE designated checkpoint each builder leaves for ft_verify.py to
# reload, as a glob relative to <out> ({version} is substituted by the
# consumer). Exported for scripts/ft_sweep.py's CKPT_GLOBS (harness-owned),
# which reads this table by AST and takes the newest non-symlink match of
# the first pattern that matches anything. Every pattern below designates a
# single file (or a single per-run family of files whose newest member IS
# the designated one), so the pick cannot flip between two candidates on
# mtime. Basis per family (installed upstream sources):
#   MACE      model_dir/<name>.model with --name=<version> (mace 0.3.15
#             cli/run_train.py:1075; the *_compiled.model twin is skipped by
#             the consumer's "_compiled" filter)
#   SevenNet  checkpoint_best.pth (best validation loss; validation split is
#             mandatory, ft_dataset refuses an empty one)
#   DeePMD/   model.ckpt-<step>.pt real files, newest = final step
#   DPA4      (pt/train/training.py:1150); model.ckpt.pt is only a symlink
#             (:1154 symlink_prefix_files) and would be rejected anyway
#   GRACE     seed/<seed>/final_model/saved_model.pb -- gracemaker's
#             unconditional final export (gracemaker.py:578), one per seed
#   PET       model-ft.pt, the `-o` export (metatrain cli/train.py:568-590)
#   NequIP/   checkpoints/last.ckpt -- ModelCheckpoint(save_last=True) writes
#   Allegro   a real file (lightning 2.6.x _save_last_checkpoint: only
#             save_last="link" symlinks); best.ckpt is deliberately NOT a
#             fallback
#   MatterSim results/best_model.pth (potential.py:391/398; written on the
#             first epoch since the initial best is +inf)
#   TACE      checkpoints_epoch/last.ckpt (save_last=True, same Lightning
#             callback as NequIP)
#   CHGNet    chgnet_ft/bestE_epoch<N>_*.pth.tar -- trainer.py:645-651 removes
#             the previous bestE_* before copying, so exactly one exists
FAMILY_CHECKPOINT_GLOBS = {
    "MACE": ["{version}.model"],
    "SevenNet": ["checkpoint_best.pth"],
    "DeePMD": ["model.ckpt-*.pt"],
    "DPA4": ["model.ckpt-*.pt"],
    "GRACE": ["seed/*/final_model/saved_model.pb"],
    "PET": ["model-ft.pt"],
    "NequIP": ["checkpoints/last.ckpt"],
    "Allegro": ["checkpoints/last.ckpt"],
    "MatterSim": ["results/best_model.pth"],
    "TACE": ["checkpoints_epoch/last.ckpt"],
    "CHGNet": ["chgnet_ft/bestE_*.pth.tar"],
}


# ── .sh rendering (same body-then-header pattern as catbench_jobgen.py) ─────
def _render_body(resolved: dict, argv: list[str], out_dir: Path, *,
                 pre_steps: list | None = None, extra_env: dict | None = None) -> str:
    lines = [_GEN_COMMENT, "set -eu", f'cd "{Path(out_dir).resolve()}"']
    for key, value in (resolved.get("env_run") or {}).items():
        lines.append(f'export {key}="{value}"')
    for key, value in (extra_env or {}).items():
        lines.append(f'export {key}="{value}"')
    for step in pre_steps or []:
        lines.append(" ".join(shlex.quote(str(a)) for a in step))
    lines.append("exec " + " ".join(shlex.quote(str(a)) for a in argv))
    return "\n".join(lines) + "\n"


def render_sh(resolved: dict, argv: list[str], out_dir: Path, *,
              pre_steps: list | None = None, extra_env: dict | None = None) -> str:
    return "#!/bin/sh\n" + _render_body(resolved, argv, out_dir, pre_steps=pre_steps, extra_env=extra_env)


def render_slurm(resolved: dict, argv: list[str], out_dir: Path, *, job_name: str, partition: str = "gpu",
                 pre_steps: list | None = None, extra_env: dict | None = None) -> str:
    out_dir = Path(out_dir).resolve()
    header = "\n".join([
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --output={out_dir}/{job_name}.slurm.out",
        f"#SBATCH --error={out_dir}/{job_name}.slurm.err",
        "#SBATCH --ntasks=1",
        "#SBATCH --gres=gpu:1",
    ]) + "\n"
    return "#!/bin/sh\n" + header + _render_body(resolved, argv, out_dir, pre_steps=pre_steps, extra_env=extra_env)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", nargs="?", help="framework or version name from models.json")
    ap.add_argument("--version", default=None, help="specific version (default: family default_version)")
    ap.add_argument("--dataset", type=Path, help="anything ase.io.read handles")
    ap.add_argument("--out", default=None, help="output directory (default: ./ft_<version>)")
    ap.add_argument("--show-settings", action="store_true",
                    help="print the model's available settings and exit")
    # Common knob flags
    ap.add_argument("--epochs", type=int, default=None, help="override official FT value")
    ap.add_argument("--max-steps", type=int, default=None, help="override official FT value")
    ap.add_argument("--batch-size", type=int, default=None, help="override official FT value")
    ap.add_argument("--lr", type=float, default=None, help="learning rate")
    ap.add_argument("--scheduler", default=None, help="learning rate scheduler type")
    ap.add_argument("--energy-weight", type=float, default=None, help="weight of energy loss")
    ap.add_argument("--force-weight", type=float, default=None, help="weight of force loss")
    ap.add_argument("--stress-weight", type=float, default=None, help="weight of stress loss")
    ap.add_argument("--include-stress", action="store_true", default=None,
                    help="include stress in training (explicit on)")
    ap.add_argument("--no-stress", action="store_true", default=None,
                    help="exclude stress from training (explicit off)")
    ap.add_argument("--patience", type=int, default=None, help="early stopping patience")
    ap.add_argument("--ema", action="store_true", default=None, help="enable exponential moving average")
    ap.add_argument("--precision", default=None, help="floating-point precision (float32/float64)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--split", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed handed to the family's own trainer (see SEED_CONTROL; default 0). "
                         "Refused with exit 5 for a family whose trainer cannot honour it fully "
                         "unless --allow-partial-seed is given")
    ap.add_argument("--allow-partial-seed", action="store_true",
                    help="accept a data-split-only seed for NequIP/Allegro (recorded in ft_run.json)")
    ap.add_argument("--set", action="append", dest="set_args", default=[],
                    help="pass native setting: --set NAME=VALUE (repeatable)")
    ap.add_argument("--emit-only", action="store_true", help="write artifacts, never execute")
    ap.add_argument("--slurm", action="store_true",
                    help="also emit an SBATCH-wrapped script; implies --emit-only (nothing is run or sbatch'd)")
    ap.add_argument("--partition", default="gpu")
    args = ap.parse_args()

    # Handle --show-settings early (no model/dataset required)
    if args.show_settings:
        if not args.model:
            print("Error: model required for --show-settings", file=sys.stderr)
            return 1
        try:
            resolved = reg.resolve(args.model)
            framework = resolved["model"]
            print(ft_settings.show_settings(framework))
            return 0
        except (ft_settings.SettingsError, reg.RegistryError) as e:
            print(f"[ft_run] {e}", file=sys.stderr)
            return 1

    # Normal mode: model and dataset are required
    if not args.model or not args.dataset:
        ap.print_help(file=sys.stderr)
        return 1
    seed_requested = args.seed is not None
    seed = int(args.seed) if seed_requested else 0

    try:
        family, version, finetune, resolved = load_finetune(args.model, args.version)
    except reg.RegistryError as exc:
        print(f"[ft_run] {exc}", file=sys.stderr)
        return 1

    # Parse --set NAME=VALUE arguments
    user_values = {}
    for arg in args.set_args:
        if "=" not in arg:
            print(f"[ft_run] --set format: NAME=VALUE (got {arg!r})", file=sys.stderr)
            return 1
        name, val_str = arg.split("=", 1)
        user_values[name] = ft_settings._parse_yaml_scalar(val_str)

    # Build knob overrides from CLI flags
    user_knobs = {}
    if args.epochs is not None:
        user_knobs["epochs"] = args.epochs
    if args.max_steps is not None:
        user_knobs["max_steps"] = args.max_steps
    if args.batch_size is not None:
        user_knobs["batch_size"] = args.batch_size
    if args.lr is not None:
        user_knobs["lr"] = args.lr
    if args.scheduler is not None:
        user_knobs["scheduler"] = args.scheduler
    if args.energy_weight is not None:
        user_knobs["energy_weight"] = args.energy_weight
    if args.force_weight is not None:
        user_knobs["force_weight"] = args.force_weight
    if args.stress_weight is not None:
        user_knobs["stress_weight"] = args.stress_weight
    if args.include_stress:
        user_knobs["include_stress"] = True
    if args.no_stress:
        user_knobs["include_stress"] = False
    if args.patience is not None:
        user_knobs["patience"] = args.patience
    if args.ema:
        user_knobs["ema"] = True
    if args.precision is not None:
        user_knobs["precision"] = args.precision

    # Handle --include-stress/--no-stress for frameworks that don't expose include_stress knob
    # but do expose stress_weight knob (MACE, SevenNet, DeePMD). Map to stress_weight instead.
    if "include_stress" in user_knobs:
        settings_file = ft_settings.load_settings_file(family)
        available_knobs = {s.get("knob") for s in settings_file.get("settings", []) if s.get("knob")}
        if "include_stress" not in available_knobs and "stress_weight" in available_knobs:
            # Framework doesn't expose include_stress knob, but does expose stress_weight
            # Map --include-stress to stress_weight=positive (e.g., 1.0) or --no-stress to stress_weight=0
            if user_knobs["include_stress"]:
                # --include-stress: set stress_weight to a positive value if not already set
                if "stress_weight" not in user_knobs:
                    user_knobs["stress_weight"] = 1.0
            else:
                # --no-stress: set stress_weight to 0
                user_knobs["stress_weight"] = 0.0
            del user_knobs["include_stress"]  # remove the knob we just handled

    # Validate that all requested knobs are supported by this framework (defect 3)
    try:
        ft_settings.validate_knobs_exist(family, set(user_knobs.keys()))
    except ft_settings.SettingsError as e:
        print(f"[ft_run] {e}", file=sys.stderr)
        return 1

    # Resolve settings
    try:
        resolved_settings, settings_origins = ft_settings.resolve_settings(
            family, user_values=user_values, user_knobs=user_knobs
        )
    except ft_settings.SettingsError as e:
        print(f"[ft_run] {e}", file=sys.stderr)
        return 1

    status = finetune.get("status")
    if status in ("not-supported", "code-excavation-needed"):
        print(f"[ft_run] {family}/{version}: finetuning is {status} -- "
              f"{finetune.get('reason') or 'no reason recorded'}", file=sys.stderr)
        for url in finetune.get("evidence") or []:
            print(f"  evidence: {url}", file=sys.stderr)
        return 2

    blockers = list(finetune.get("blockers") or [])
    remaining = live_recheck_blockers(family, blockers, resolved["python"])
    if not finetune.get("runnable_as_installed"):
        if remaining:
            print(f"[ft_run] {family}/{version}: not runnable as installed:", file=sys.stderr)
            for b in remaining:
                print(f"  blocker: {b}", file=sys.stderr)
            if not Path(resolved["python"]).exists():
                print(f"  (the env interpreter {resolved['python']} does not exist -- "
                      f"build it first: ./install.sh {resolved['env']})", file=sys.stderr)
            return 3
        if not blockers:
            # runnable_as_installed:false with an empty blocker list: nothing
            # to live-recheck, so the registry flag stands (schema forbids
            # this shape, but a hand-edited registry must not slip through).
            print(f"[ft_run] {family}/{version}: registry marks this variant not "
                  f"runnable as installed and lists no verifiable blockers -- refusing.",
                  file=sys.stderr)
            return 3
        # every declared blocker demonstrably cleared -> proceed.

    builder = BUILDERS.get(family)
    if builder is None:
        print(f"[ft_run] {family}/{version}: registry classifies fine-tuning as "
              f"{status!r} and runnable, but this hub has no real command builder "
              f"for the {family} family yet (implementation gap, not an upstream "
              f"limitation) -- refusing rather than rendering a guessed command. "
              f"Families with builders: {sorted(BUILDERS)}", file=sys.stderr)
        return EXIT_NO_BUILDER

    seed_scope, seed_basis = SEED_CONTROL[family]
    if seed_scope != "native":
        msg = (f"[ft_run] {family}/{version}: --seed reaches the {seed_scope} scope only -- {seed_basis}")
        if seed_requested and not args.allow_partial_seed:
            print(msg + "\n  refusing an explicit --seed this trainer cannot honour; pass "
                  "--allow-partial-seed to accept the narrower scope (it is recorded in ft_run.json)",
                  file=sys.stderr)
            return EXIT_SEED_UNHONOURED
        print(msg, file=sys.stderr)

    licence = finetune.get("licence")
    if licence:
        url = resolved.get("license_url") or _LICENCE_URLS.get(licence, "")
        note = f"[ft_run] NOTE: {version}'s checkpoint carries a non-permissive licence: {licence}"
        if url:
            note += f" ({url})"
        print(note + " -- oh-my-mlip is MIT and redistributes no weights; "
              "a fine-tuned derivative inherits this licence's terms.", file=sys.stderr)

    # per-VERSION default so two variants of one family never share a dir
    out = Path(args.out).resolve() if args.out else (Path.cwd() / f"ft_{version}").resolve()
    out.mkdir(parents=True, exist_ok=True)

    dataset = args.dataset.resolve()
    target = dataset_target(family)
    conv = run_ft_dataset(dataset, target, out / "data", args.split, seed,
                          python_bin=resolved["python"])

    # Epochs and batch size as the user's knob values, when the framework exposes those
    # knobs. Builders read every emitted value from `resolved_settings` by native name and
    # refuse a setting that has no value; these two only feed builders that still take
    # the common value directly.
    knob_names = {s_["name"]: s_.get("knob") for s_ in ft_settings.load_settings_file(family)["settings"]}

    def knob_value(knob):
        for native, value in resolved_settings.items():
            if knob_names.get(native) == knob and not isinstance(value, (dict, list)):
                return value
        return None

    epochs_value = knob_value("epochs")
    batch_value = knob_value("batch_size")
    ctx = Context(
        family=family, version=version, finetune=finetune, resolved=resolved, out=out,
        epochs=int(epochs_value) if isinstance(epochs_value, (int, float)) else None,
        batch_size=int(batch_value) if isinstance(batch_value, (int, float)) else None,
        device=args.device, dataset_paths=conv["outputs"], elements=conv["elements"], seed=seed,
        settings=resolved_settings, settings_origins=settings_origins,
    )
    try:
        spec = builder(ctx)
    except ValueError as exc:
        print(f"[ft_run] {version}: {exc}", file=sys.stderr)
        return 1

    if spec.config_path is not None:
        spec.config_path.write_text(spec.config_text)
    for path, text in spec.extra_files.items():
        Path(path).write_text(text)

    sh_path = out / f"finetune_{version}.sh"
    sh_path.write_text(render_sh(resolved, spec.argv, out, pre_steps=spec.pre_steps, extra_env=spec.extra_env))
    sh_path.chmod(0o755)
    print(f"[ft_run] wrote {sh_path}")

    slurm_path = None
    if args.slurm:
        slurm_path = out / f"slurm_finetune_{version}.sh"
        slurm_path.write_text(render_slurm(resolved, spec.argv, out, job_name=f"finetune_{version}", partition=args.partition,
                                           pre_steps=spec.pre_steps, extra_env=spec.extra_env))
        slurm_path.chmod(0o755)
        print(f"[ft_run] wrote {slurm_path}")

    record = run_record(args, version=version, family=family, resolved=resolved, out=out, dataset=dataset,
                        seed=seed, seed_requested=seed_requested, seed_scope=seed_scope, seed_basis=seed_basis,
                        spec=spec, sh_path=sh_path, slurm_path=slurm_path, conversion=conv,
                        resolved_settings=resolved_settings, settings_origins=settings_origins,
                        final_epochs=ctx.epochs, final_batch_size=ctx.batch_size)
    (out / "ft_run.json").write_text(json.dumps(record, indent=2) + "\n")

    if args.slurm:
        print("[ft_run] --slurm is emission-only: NOT executing locally and NOT sbatch'ing "
              f"{slurm_path.name} -- submit it yourself", file=sys.stderr)
        return 0
    if args.emit_only:
        return 0

    print(f"[ft_run] executing {sh_path}")
    return subprocess.run(["bash", str(sh_path)]).returncode


def run_record(args, *, version: str, family: str, resolved: dict, out: Path, dataset: Path,
               seed: int, seed_requested: bool, seed_scope: str, seed_basis: str,
               spec: CommandSpec, sh_path: Path, slurm_path: Path | None, conversion: dict,
               resolved_settings: dict | None = None, settings_origins: dict | None = None,
               final_epochs: int | None = None, final_batch_size: int | None = None) -> dict:
    """<out>/ft_run.json -- the provenance the emitted artifacts alone do not
    carry: what was asked (the exact ft_run.py argv to re-materialize this
    run), how far the seed reaches, which file is the designated checkpoint,
    and which artifacts were written. The .sh reruns THIS materialization;
    hub-owned inputs it names by absolute path (models/<env>/... foundation
    weights, prestage caches) are re-created by install.sh and the prestage
    steps, not by the .sh -- after a fresh_root cleanup use `rematerialize`."""
    # Build rematerialize argv from what was actually used
    rematerialize = [
        sys.executable, str(_SCRIPTS_DIR / "ft_run.py"), version,
        "--dataset", str(dataset), "--out", str(out),
        "--device", args.device, "--split", str(args.split), "--seed", str(seed),
    ]
    # Add epochs/batch-size if they came from CLI (not defaults)
    if args.epochs is not None:
        rematerialize.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        rematerialize.extend(["--batch-size", str(args.batch_size)])
    rematerialize.extend([
        *(["--allow-partial-seed"] if seed_scope != "native" else []),
        "--emit-only",
    ])
    result = {
        "schema": "ft_run.json/1",
        "family": family, "version": version,
        "env": resolved.get("env"), "python": resolved.get("python"),
        "dataset": str(dataset),
        "out": str(out),
        "epochs": int(final_epochs) if final_epochs is not None else int(args.epochs or 0),
        "batch_size": int(final_batch_size) if final_batch_size not in (None, "auto") else (int(args.batch_size or 0) if args.batch_size else 0),
        "device": args.device, "split": float(args.split),
        "seed": int(seed), "seed_requested": bool(seed_requested),
        "seed_control": {"scope": seed_scope, "basis": seed_basis},
        "designated_checkpoint": [g.format(version=version) for g in FAMILY_CHECKPOINT_GLOBS[family]],
        "artifacts": {
            "sh": str(sh_path),
            "slurm": str(slurm_path) if slurm_path else None,
            "config": str(spec.config_path) if spec.config_path else None,
            "extra_files": [str(p) for p in spec.extra_files],
            "conversion": str(out / "data" / "conversion.json"),
        },
        "n_train": conversion.get("n_train"), "n_valid": conversion.get("n_valid"),
        "elements": conversion.get("elements"),
        "command": [str(a) for a in spec.argv],
        "pre_steps": [[str(a) for a in step] for step in spec.pre_steps],
        "rematerialize": rematerialize,
    }
    # Record all resolved settings with their origins
    if resolved_settings:
        result["settings"] = {name: {"value": val, "origin": settings_origins.get(name, "unknown")}
                             for name, val in resolved_settings.items()}
    return result


if __name__ == "__main__":
    raise SystemExit(main())
