"""oh_my_mlip — public teacher-provider base for the oh-my-mlip HUB.

A path-importable (NOT pip) package that turns the trusted ``models.json`` +
``dist_manifest.json`` registry into a tiered, machine-readable interface for
running 20 frameworks / 31 model variants, each in its own conda env. The four
layers:

  1. resolve(model, version=None)              -> codegen dict  (registry)
  2. get_calculator(model, ...)                -> ase Calculator (INTRA-ENV)
  3. run(model, atoms, ...)                     -> results dict  (CROSS-ENV)
  4. Worker / WorkerPool                        -> persistent JSONL workers

Use by putting the repo root on sys.path (e.g. ``import sys;
sys.path.insert(0, OH_MY_MLIP_HOME)``) — there is no setup.py.

All framework / torch / ase / huggingface_hub imports are lazy, so importing
this package never requires a GPU or any model env to be present.
"""
from __future__ import annotations

from oh_my_mlip.registry import (
    ENV_RUN_ALLOWLIST,
    RegistryError,
    home,
    list_models,
    list_versions,
    parse_env_run,
    resolve,
)
from oh_my_mlip.provider import Worker, WorkerPool, get_calculator, run

__all__ = [
    "list_models",
    "list_versions",
    "resolve",
    "get_calculator",
    "run",
    "parse_env_run",
    "Worker",
    "WorkerPool",
    "home",
    "RegistryError",
    "ENV_RUN_ALLOWLIST",
]

__version__ = "0.1.0"
