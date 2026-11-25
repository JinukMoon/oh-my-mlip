"""Refuse D3 at once in an env whose D3 kernel build failed at install.

install.sh copies this file into every env it builds as ``_omm_d3_guard.py`` and
adds ``omm_d3_guard.pth`` (one line: ``import _omm_d3_guard``), so it runs at the
start of every interpreter of that env, whatever the entry point is (run(...,
apply_d3=True), get_calculator, a CatBench job, a user script).

A failed build is not fatal for the env (energies and forces do not need D3),
but catbench compiles the kernel on the first D3 call when it is missing. On the
host where the build already failed that repeats the failure on the GPU node,
or waits on a build lock, so the user sees an obscure error or a hang. When the
env's marker (scripts/build_d3.py: <prefix>/.omm_d3, or the per-user copy for a
read-only env) says ``state=failed``, constructing catbench's
DispersionCorrection raises at once with the reason and how to retry.
A marker in any other state, or none, changes nothing. Standard library only.
"""
import os
import sys

_STATE = None


def _marker_path():
    prefix = sys.prefix
    if os.access(prefix, os.W_OK):
        return os.path.join(prefix, ".omm_d3")
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    flat = os.path.realpath(prefix).strip("/").replace("/", "_")
    return os.path.join(root, "oh-my-mlip", "d3", flat, ".omm_d3")


def _read_marker(path):
    fields = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                key, sep, value = line.rstrip("\n").partition("=")
                if sep:
                    fields[key] = value
    except OSError:
        return None
    return fields


class D3UnavailableError(RuntimeError):
    pass


def _message(fields):
    env = os.path.basename(sys.prefix.rstrip("/"))
    log = fields.get("log") or "no log recorded"
    detail = fields.get("detail") or "unknown reason"
    return (f"D3 is unavailable in this env: the D3 kernel build failed at install ({detail}; log: {log}). "
            f"Energies and forces work without D3. To retry the build, re-run install.sh {env}, "
            f"or: python3 $OH_MY_MLIP_HOME/scripts/build_d3.py {sys.prefix}")


def _patch(module):
    base = getattr(module, "DispersionCorrection", None)
    if base is None or getattr(base, "_omm_guarded", False):
        return
    fields = _STATE

    class DispersionCorrection(base):  # noqa: D401 - same name for users and tracebacks
        _omm_guarded = True

        def __init__(self, *args, **kwargs):
            raise D3UnavailableError(_message(fields))

    module.DispersionCorrection = DispersionCorrection


def _install():
    import importlib.abc
    import importlib.util

    class _Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "catbench.dispersion":
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
                return spec
            loader = spec.loader
            real_exec = loader.exec_module

            class _Loader(importlib.abc.Loader):
                def create_module(self, spec_):
                    return loader.create_module(spec_)

                def exec_module(self, module):
                    real_exec(module)
                    _patch(module)

            spec.loader = _Loader()
            return spec

    sys.meta_path.insert(0, _Guard())
    if "catbench.dispersion" in sys.modules:
        _patch(sys.modules["catbench.dispersion"])


_fields = _read_marker(_marker_path())
if _fields and _fields.get("state") == "failed":
    _STATE = _fields
    _install()
