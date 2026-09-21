"""Tests for the adopted-env (bring-your-own-env) feature.

Pins the contract across its three surfaces:
  * oh_my_mlip.registry.resolve dispatches to the adopted interpreter when
    env_map.local.json names one, and fails ACTIONABLY on a stale entry;
  * scripts/adopt_env.py refuses an env whose registry imports fail and
    records only verified adoptions;
  * scripts/setup_survey.py counts an adopted env as ready (zero disk).

GPU-free; fake interpreters are tiny shell scripts.
"""
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oh_my_mlip import registry  # noqa: E402

_SURVEY_SPEC = importlib.util.spec_from_file_location(
    "setup_survey", REPO_ROOT / "scripts" / "setup_survey.py"
)
survey_mod = importlib.util.module_from_spec(_SURVEY_SPEC)
_SURVEY_SPEC.loader.exec_module(survey_mod)


def _fake_env(tmp_path: Path, name: str, import_ok: bool = True) -> Path:
    prefix = tmp_path / name
    (prefix / "bin").mkdir(parents=True)
    py = prefix / "bin" / "python"
    py.write_text(f"#!/bin/sh\nexit {0 if import_ok else 1}\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    return prefix


def _home(tmp_path: Path) -> Path:
    (tmp_path / "models.json").write_text(json.dumps({
        "_meta": {},
        "Alpha": {
            "env": "alpha",
            "python": "${OH_MY_MLIP_HOME}/envs/alpha/bin/python",
            "import": ["import alpha"],
            "default_version": "A-1",
            "versions": {"A-1": {"gated": False, "inference": ["calc = None"]}},
        },
    }))
    (tmp_path / "envs").mkdir()
    return tmp_path


def _models(home: Path) -> dict:
    return json.loads((home / "models.json").read_text())


def test_resolve_uses_adopted_interpreter(tmp_path, monkeypatch):
    home = _home(tmp_path)
    prefix = _fake_env(tmp_path, "external_alpha")
    (home / "env_map.local.json").write_text(json.dumps({"alpha": str(prefix)}))
    monkeypatch.setenv("OH_MY_MLIP_HOME", str(home))
    spec = registry.resolve("Alpha", models=_models(home))
    assert spec["python"] == str(prefix / "bin" / "python")


def test_resolve_without_map_uses_hub_prefix(tmp_path, monkeypatch):
    home = _home(tmp_path)
    monkeypatch.setenv("OH_MY_MLIP_HOME", str(home))
    spec = registry.resolve("Alpha", models=_models(home))
    assert spec["python"] == str(home / "envs" / "alpha" / "bin" / "python")


def test_resolve_stale_adoption_is_actionable(tmp_path, monkeypatch):
    home = _home(tmp_path)
    (home / "env_map.local.json").write_text(json.dumps({"alpha": "/nonexistent/prefix"}))
    monkeypatch.setenv("OH_MY_MLIP_HOME", str(home))
    with pytest.raises(registry.RegistryError, match="adopt_env"):
        registry.resolve("Alpha", models=_models(home))


def test_resolve_invalid_map_raises(tmp_path, monkeypatch):
    home = _home(tmp_path)
    (home / "env_map.local.json").write_text("not json")
    monkeypatch.setenv("OH_MY_MLIP_HOME", str(home))
    with pytest.raises(registry.RegistryError, match="valid JSON"):
        registry.resolve("Alpha", models=_models(home))


def _run_adopt(home: Path, *args: str) -> subprocess.CompletedProcess:
    import os
    env = dict(os.environ, OH_MY_MLIP_HOME=str(home))
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "adopt_env.py"), *args],
        capture_output=True, text=True, env=env,
    )


def test_adopt_records_only_verified_envs(tmp_path):
    home = _home(tmp_path)
    good = _fake_env(tmp_path, "good", import_ok=True)
    bad = _fake_env(tmp_path, "bad", import_ok=False)

    refused = _run_adopt(home, "Alpha", str(bad))
    assert refused.returncode == 1 and "REFUSED" in refused.stderr
    assert not (home / "env_map.local.json").exists()

    adopted = _run_adopt(home, "Alpha", str(good))
    assert adopted.returncode == 0, adopted.stderr
    data = json.loads((home / "env_map.local.json").read_text())
    assert data == {"alpha": str(good)}

    removed = _run_adopt(home, "--remove", "alpha")
    assert removed.returncode == 0
    assert json.loads((home / "env_map.local.json").read_text()) == {}


def test_survey_counts_adopted_as_ready(tmp_path):
    home = _home(tmp_path)
    prefix = _fake_env(tmp_path, "external_alpha")
    (home / "env_map.local.json").write_text(json.dumps({"alpha": str(prefix)}))
    result = survey_mod.survey(home, [])
    row = result["envs"][0]
    assert row["state"] == "ready" and row["adopted"] is True
    assert result["to_build"] == []
    assert result["disk"]["budget_gb"] == 0


def _passthrough_env(tmp_path: Path, name: str) -> Path:
    """A fake env whose bin/python really runs python -- prepare_* scripts are
    executed BY the adopted interpreter, so a stub that ignores its arguments
    would make the test pass without running anything."""
    prefix = tmp_path / name
    (prefix / "bin").mkdir(parents=True)
    py = prefix / "bin" / "python"
    # `-c` is adopt_env's registry-import check (it must pass for a fake env);
    # a script path is a real weight-preparation run and must really execute.
    py.write_text(f'#!/bin/sh\ncase "$1" in -c) exit 0 ;; esac\nexec {sys.executable} "$@"\n')
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    return prefix


def test_adoption_prepares_weights_like_install_sh(tmp_path: Path):
    """install.sh runs scripts/{prestage,prepare}_<env>_weights.py after building
    an env; adoption skipped them, so a path-based model (GRACE, DeePMD) resolved
    to a checkpoint nobody had created and failed at load with a missing file."""
    home = _home(tmp_path)
    scripts = home / "scripts"
    scripts.mkdir(exist_ok=True)
    marker = tmp_path / "prepared.txt"
    (scripts / "prepare_alpha_weights.py").write_text(
        f"import sys\nopen({str(marker)!r}, 'w').write(' '.join(sys.argv[1:]))\n"
    )
    prefix = _passthrough_env(tmp_path, "alpha_env")

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "adopt_env.py"), "Alpha", str(prefix)],
        capture_output=True, text=True, env={"OH_MY_MLIP_HOME": str(home), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "preparing weights: prepare_alpha_weights.py" in proc.stdout
    # the same arguments install.sh passes; the real prepare scripts require them
    assert marker.read_text() == f"--target-root {home / 'models' / 'alpha'}"


def test_a_failing_weight_preparation_is_reported_but_keeps_the_adoption(tmp_path: Path):
    home = _home(tmp_path)
    scripts = home / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "prepare_alpha_weights.py").write_text("raise SystemExit('no network')\n")
    prefix = _passthrough_env(tmp_path, "alpha_env")

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "adopt_env.py"), "Alpha", str(prefix)],
        capture_output=True, text=True, env={"OH_MY_MLIP_HOME": str(home), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0
    assert "FAILED" in proc.stderr and "no network" in proc.stderr
    assert f"prepare_alpha_weights.py --target-root {home / 'models' / 'alpha'}" in proc.stderr
    assert json.loads((home / "env_map.local.json").read_text())["alpha"] == str(prefix)


def _parse_only(script: Path, argv: list[str]):
    """Run `script`'s main() just far enough to parse argv with its own parser."""
    import argparse
    import importlib.util

    class Parsed(Exception):
        pass

    original = argparse.ArgumentParser.parse_args

    def parse_then_stop(self, args=None, namespace=None):
        raise Parsed(original(self, args, namespace))

    spec = importlib.util.spec_from_file_location(script.stem, script)
    mod = importlib.util.module_from_spec(spec)
    old_argv, old_path = sys.argv, list(sys.path)
    sys.path.insert(0, str(script.parent))
    try:
        spec.loader.exec_module(mod)
        sys.argv = [str(script), *argv]
        argparse.ArgumentParser.parse_args = parse_then_stop
        try:
            mod.main()
        except Parsed as done:
            return done.args[0]
        raise AssertionError(f"{script.name}: main() returned without parsing arguments")
    finally:
        argparse.ArgumentParser.parse_args = original
        sys.argv, sys.path[:] = old_argv, old_path


def test_every_real_weight_script_accepts_the_arguments_adoption_passes(tmp_path: Path):
    """The fake scripts above take any argv; the real ones do not. adopt_env once
    called them with none, and prepare_{nequip,allegro,deepmd,pet} all require
    --target-root, so every adoption of those envs left the weights unprepared."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import adopt_env

    envs = sorted({p.stem.split("_", 1)[1].rsplit("_weights", 1)[0]
                   for p in (REPO_ROOT / "scripts").glob("pre*_*_weights.py")})
    assert {"nequip", "allegro", "deepmd", "pet", "grace"} <= set(envs)
    seen = 0
    for env in envs:
        for argv in adopt_env.weight_steps(REPO_ROOT, env, Path(sys.executable)):
            try:
                _parse_only(Path(argv[1]), argv[2:])
            except SystemExit as exc:  # argparse rejects the argv
                raise AssertionError(f"{Path(argv[1]).name} rejects {argv[2:]}") from exc
            seen += 1
    assert seen >= 8


def test_adoption_and_install_sh_call_the_weight_scripts_the_same_way():
    text = (REPO_ROOT / "install.sh").read_text()
    assert 'python3 "$prestage" ||' in text
    assert '"$prefix/bin/python" "$prepare" --target-root "$OH_MY_MLIP_HOME/models/$env_name"' in text
