"""install.sh never deletes an env on its own: --rebuild, symlinks, timeouts, one build per env."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def hub(tmp_path):
    """A hub with one recipe (mace), a fake conda that records calls and fails, and no built envs."""
    home = tmp_path / "hub"
    shutil.copytree(REPO, home, ignore=shutil.ignore_patterns(".git", "envs", "models", ".sweep"))
    (home / "envs").mkdir()
    shutil.copy(REPO / "envs" / "mace.yml", home / "envs" / "mace.yml")
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "conda").write_text('#!/bin/sh\necho "conda $*" >> "$CALLS"\nexit 1\n')
    (fakebin / "conda").chmod(0o755)
    env = {"HOME": str(tmp_path), "PATH": f"{fakebin}:/usr/bin:/bin", "OH_MY_MLIP_HOME": str(home),
           "CALLS": str(tmp_path / "calls")}
    return home, env


def _env_python(prefix: Path, body: str) -> None:
    """A fake env interpreter: the models.json read goes to the real python, the import check runs `body`."""
    (prefix / "bin").mkdir(parents=True)
    py = prefix / "bin" / "python"
    py.write_text(f'#!/bin/sh\nif [ "$1" = "-" ]; then exec python3 "$@"; fi\n{body}\n')
    py.chmod(0o755)


def _install(env, *args):
    return subprocess.run(["bash", str(Path(env["OH_MY_MLIP_HOME"]) / "install.sh"), *args], env=env,
                          capture_output=True, text=True)


def test_failing_imports_keep_the_env_without_rebuild(hub):
    home, env = hub
    prefix = home / "envs" / "mace"
    _env_python(prefix, "exit 1")
    r = _install(env, "mace")
    assert r.returncode == 1 and "Nothing was deleted" in r.stderr and "--rebuild mace" in r.stderr
    assert (prefix / "bin" / "python").exists()


def test_rebuild_deletes_and_rebuilds(hub):
    home, env = hub
    prefix = home / "envs" / "mace"
    _env_python(prefix, "exit 1")
    r = _install(env, "--rebuild", "mace")
    assert f"deleting {prefix}" in r.stdout and not prefix.exists()
    assert "env create" in Path(env["CALLS"]).read_text()          # it went on to build


def test_a_timed_out_check_never_deletes_even_with_rebuild(hub):
    home, env = hub
    prefix = home / "envs" / "mace"
    _env_python(prefix, "sleep 5")
    r = _install(dict(env, OMM_IMPORT_CHECK_TIMEOUT="1"), "--rebuild", "mace")
    assert r.returncode == 1 and "timed out" in r.stderr and prefix.exists()


def test_a_symlinked_env_is_never_touched(hub, tmp_path):
    home, env = hub
    target = tmp_path / "user_env"
    _env_python(target, "exit 1")
    (home / "envs" / "mace").symlink_to(target)
    r = _install(env, "--rebuild", "mace")
    assert r.returncode == 1 and "symlink" in r.stderr and (target / "bin" / "python").exists()
    assert not Path(env["CALLS"]).exists() or "remove" not in Path(env["CALLS"]).read_text()


def test_a_second_install_of_the_same_env_waits_for_the_first(hub):
    home, env = hub
    lock = home / "envs" / ".mace.install.lock"
    holder = subprocess.Popen(["sleep", "30"])
    try:
        lock.mkdir()
        (lock / "pid").write_text(str(holder.pid))
        r = _install(env, "mace")
        assert r.returncode == 1 and f"already being built by install.sh (pid {holder.pid})" in r.stderr
        assert lock.is_dir()                                      # the live holder's lock is left alone
    finally:
        holder.kill()
        holder.wait()
    (lock / "pid").write_text(str(holder.pid))                    # now a dead holder: taken over, then released
    r = _install(env, "mace")
    assert "already being built" not in r.stderr and not lock.exists()
