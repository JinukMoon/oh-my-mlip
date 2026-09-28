"""First-contact messages an agent acts on (review finding 12): each names the next command."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def _hide(modules, script, *args, cwd=None):
    """Run `script` with `modules` made unimportable, as on a bare system python."""
    code = ("import sys, runpy\n"
            + "".join(f"sys.modules[{m!r}] = None\n" for m in modules)
            + f"sys.argv = [{str(script)!r}] + {list(args)!r}\n"
            + f"runpy.run_path({str(script)!r}, run_name='__main__')\n")
    env = dict(os.environ, OH_MY_MLIP_HOME=str(REPO))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=cwd)


def test_not_installed_reason_keeps_the_runnable_command():
    import setup_verify
    stderr = ("[oh-my-mlip] the conda env 'mace' for MACE-MPA-0 is not materialized yet (interpreter not found: "
              "/h/envs/mace/bin/python). Install it first:\n    bash \"/h/install.sh\" MACE-MPA-0\n")
    verdict = setup_verify.decide_verdict({"skew": False}, 1, False, None, stderr)
    assert 'bash "/h/install.sh" MACE-MPA-0' in verdict["reason"]


def test_install_sh_names_the_valid_envs_for_an_unknown_target():
    r = subprocess.run(["bash", str(REPO / "install.sh"), "--status", "Foo"], capture_output=True, text=True,
                       env=dict(os.environ, OH_MY_MLIP_HOME=str(REPO)))
    assert r.returncode == 2 and "Foo" in r.stderr and "envs: " in r.stderr and "mace" in r.stderr
    assert "families: " in r.stderr and "MACE" in r.stderr


def test_distill_help_needs_no_scientific_packages():
    for name in ("distill_bootstrap.py", "distill_verify.py"):
        r = _hide(("numpy", "yaml", "ase"), REPO / "scripts" / name, "--help")
        assert r.returncode == 0 and "usage:" in r.stdout, (name, r.stderr[-300:])
    r = _hide(("numpy", "yaml", "ase"), REPO / "scripts" / "distill_verify.py", "--work", "/nonexistent")
    assert r.returncode == 2 and "teacher env's interpreter" in r.stderr


def test_relax_hint_is_an_exact_command_from_any_directory(tmp_path):
    r = _hide(("ase",), REPO / "run_examples" / "relax.py", "MACE", "--structure", "x.vasp", cwd=tmp_path)
    assert r.returncode == 2
    last = r.stderr.strip().splitlines()[-1]
    assert f"OH_MY_MLIP_HOME={REPO}" in last and str(REPO / "run_examples" / "relax.py") in last
    assert "--structure x.vasp" in last


def test_a_gpu_less_host_gets_a_driver_message_not_a_compile_hint(monkeypatch):
    from oh_my_mlip import fetch, registry
    monkeypatch.setattr(registry, "detect_host_arch", lambda: None)
    spec = registry.resolve("NequIP")
    assert spec["arch_source"] == "default"
    monkeypatch.setattr(fetch, "_inference_weight_targets", lambda s: [Path("/nonexistent/model.pt2")])
    import pytest
    with pytest.raises(fetch.FetchError, match="no NVIDIA GPU was found"):
        fetch.ensure_weights(spec["model"], spec["version"], spec=spec)
    assert registry.resolve("NequIP", arch="sm86")["arch_source"] == "explicit"


def test_nvidia_smi_is_found_in_wsls_driver_directory(monkeypatch):
    from oh_my_mlip import registry
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr(registry, "_WSL_NVIDIA_SMI", sys.executable)   # any executable file
    assert registry.nvidia_smi() == sys.executable
