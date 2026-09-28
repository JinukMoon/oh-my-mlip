"""The pypi.nvidia.com recovery (AGENTS.md §8) as a script: pins from PyPI, guarded, no network in tests."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import install_nvidia_wheels as inw  # noqa: E402

REQ = ['nvidia-cuda-runtime-cu12==12.6.77; platform_system == "Linux" and platform_machine == "x86_64"',
       'nvidia-cudnn-cu12==9.5.1.17; platform_system == "Linux" and platform_machine == "x86_64"',
       'nvidia-foo-cu12==1.0; platform_system == "Windows"',
       'filelock', 'sympy>=1.13.3']


@pytest.fixture()
def hub(tmp_path):
    home = tmp_path / "hub"
    (home / "envs").mkdir(parents=True)
    shutil.copy(REPO / "models.json", home / "models.json")
    (home / "envs" / "mace.yml").write_text("  - torch==2.7.1+cu126\n")
    return home


def _partial(home):
    py = home / "envs" / "mace" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    return py


def test_pins_are_torchs_linux_nvidia_requirements_only():
    assert inw.nvidia_pins(REQ) == ["nvidia-cuda-runtime-cu12==12.6.77", "nvidia-cudnn-cu12==9.5.1.17"]
    assert inw.cuda_tag_of(inw.nvidia_pins(REQ)) == "cu126"


def test_plan_for_a_partial_env(hub):
    py = _partial(hub)
    out = inw.plan("MACE-MPA-0", hub, fetch=lambda url: {"info": {"requires_dist": REQ}})
    assert out["ok"] and out["env"] == "mace" and out["torch"] == "2.7.1+cu126"
    assert out["command"][:6] == [str(py), "-m", "pip", "install", "--index-url", "https://pypi.org/simple"]
    assert out["command"][-2:] == out["pins"] and out["next"].endswith("install.sh mace")


def test_refusals(hub):
    fetch = lambda url: {"info": {"requires_dist": REQ}}           # noqa: E731
    assert inw.plan("NotAModel", hub, fetch)["exit"] == 2
    assert "no interpreter yet" in inw.plan("mace", hub, fetch)["reason"]
    _partial(hub)
    (hub / "envs" / "mace" / ".omm_ready").write_text("")
    assert "already installed" in inw.plan("mace", hub, fetch)["reason"]
    (hub / "envs" / "mace" / ".omm_ready").unlink()
    (hub / "envs" / "mace.yml").write_text("  - torch==2.7.1+cu128\n")   # a build PyPI does not describe
    out = inw.plan("mace", hub, fetch)
    assert out["exit"] == 2 and "cu128" in out["reason"] and "cu126" in out["reason"]


def test_a_symlinked_env_is_refused(hub, tmp_path):
    target = tmp_path / "user_env"
    (target / "bin").mkdir(parents=True)
    (target / "bin" / "python").write_text("")
    (hub / "envs" / "mace").symlink_to(target)
    assert "symlink" in inw.plan("mace", hub, lambda url: {})["reason"]


def test_pypi_down_is_exit_3(hub):
    _partial(hub)

    def down(url):
        raise ConnectionError("timed out")
    assert inw.plan("mace", hub, down)["exit"] == 3
