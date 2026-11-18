"""Recipe per GPU architecture (envs/_arch_recipes.json, scripts/select_recipe.py).

GPU-free: the host arch is pinned with OMM_INSTALL_ARCH, so these run the same on
a CI runner without a GPU and on an sm120 workstation.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ENVS = REPO / "envs"
sys.path.insert(0, str(REPO / "scripts"))

import gen_env_lock  # noqa: E402
import gen_status_table  # noqa: E402
import lint_recipes  # noqa: E402
import select_recipe  # noqa: E402

TABLE = json.loads((ENVS / "_arch_recipes.json").read_text())
SIBLINGS = sorted(p.stem for p in ENVS.glob("*.*.yml"))


def _install(*args: str, arch: str, home: Path = REPO) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "OMM_INSTALL_ARCH": arch, "OH_MY_MLIP_HOME": str(home)}
    return subprocess.run(["bash", str(home / "install.sh"), *args], capture_output=True,
                          text=True, cwd=str(home), env=env)


# ── the selection rule ───────────────────────────────────────────────────────

@pytest.mark.parametrize("arch", ["sm86", "sm89", "sm90", None])
def test_older_gpus_and_no_gpu_keep_the_base_recipe(arch):
    for env in ("mace", "alphanet", "grace", "uma"):
        assert select_recipe.select(env, arch, ENVS)[0] == env


def test_sm120_picks_the_sibling_for_every_env_that_has_one():
    for stem in SIBLINGS:
        base = stem.split(".", 1)[0]
        assert select_recipe.select(base, "sm120", ENVS)[0] == stem


def test_sm120_unsupported_env_stops_with_its_reason():
    for env, entry in TABLE["sm120"].items():
        if entry["recipe"] is None:
            stem, reason = select_recipe.select(env, "sm120", ENVS)
            assert stem is None and entry["reason"] in reason


def test_unlisted_newer_arch_stops(tmp_path):
    stem, reason = select_recipe.select("mace", "sm100", ENVS)
    assert stem is None and "newer than any architecture" in reason


def test_env_missing_from_an_arch_block_stops(tmp_path):
    envs = tmp_path / "envs"
    envs.mkdir()
    (envs / "_arch_recipes.json").write_text(json.dumps({"base_max_arch": "sm90", "sm120": {}}))
    (envs / "mace.yml").write_text("name: mace\n")
    stem, reason = select_recipe.select("mace", "sm120", envs)
    assert stem is None and "has been checked on sm120" in reason


def test_override_parsing(monkeypatch):
    monkeypatch.setenv("OMM_INSTALL_ARCH", "sm_120")
    assert select_recipe.host_arch() == "sm120"
    monkeypatch.setenv("OMM_INSTALL_ARCH", "none")
    assert select_recipe.host_arch() is None


# ── install.sh uses it ───────────────────────────────────────────────────────

def test_dry_run_sm120_builds_siblings_and_stops_unsupported():
    proc = _install("--dry-run", "mace", "alphanet", "uma", "grace", arch="sm120")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "envs/mace.cu128.yml" in out and "would create env 'mace'" in out
    assert "alphanet.cu128.build.sh" in out
    assert "recipe for 'uma' (GPU sm120): envs/uma.yml" in out
    if TABLE["sm120"]["grace"]["recipe"] is None:
        assert "STOP 'grace' on sm120" in out


def test_dry_run_sm89_never_mentions_a_sibling():
    out = _install("--dry-run", "--all", arch="sm89").stdout
    assert ".cu128" not in out and "STOP" not in out


def test_dry_run_unchecked_arch_stops_every_env():
    out = _install("--dry-run", "mace", "uma", arch="sm100").stdout
    assert out.count("STOP '") == 2 and "would create" not in out


def test_siblings_are_not_install_targets():
    proc = _install("--dry-run", "mace.cu128", arch="sm120")
    assert proc.returncode == 2 and "not an env" in proc.stderr


@pytest.fixture()
def mini_home(tmp_path: Path) -> Path:
    """A clone-shaped copy: install.sh + the files it reads, no envs built."""
    home = tmp_path / "omm"
    for rel in ("install.sh", "env.sh", "models.json", "scripts/select_recipe.py"):
        (home / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, home / rel)
    shutil.copytree(REPO / "oh_my_mlip", home / "oh_my_mlip",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (home / "envs").mkdir()
    for f in ENVS.iterdir():
        if f.is_file():
            shutil.copy2(f, home / "envs" / f.name)
    return home


def test_status_flags_an_env_built_from_another_arch_recipe(mini_home):
    prefix = mini_home / "envs" / "mace"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").write_text("#!/bin/sh\nexit 0\n")
    (prefix / "bin" / "python").chmod(0o755)
    out = _install("--status", "mace", arch="sm120", home=mini_home).stdout
    assert "built from envs/mace.yml, but this GPU (sm120) needs envs/mace.cu128.yml" in out
    (prefix / ".omm_recipe").write_text("mace.cu128 sm120\n")
    out = _install("--status", "mace", arch="sm120", home=mini_home).stdout
    assert "recipe here: envs/mace.cu128.yml" in out and "needs" not in out
    out = _install("--status", "mace", arch="sm89", home=mini_home).stdout
    assert "built from envs/mace.cu128.yml, but this GPU (sm89) needs envs/mace.yml" in out


def test_real_install_stops_before_conda_on_an_unchecked_gpu(mini_home, tmp_path):
    """No recipe for this GPU: install.sh exits before any conda call."""
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    log = tmp_path / "conda_calls"
    (fakebin / "conda").write_text(f"#!/bin/sh\necho \"$@\" >> {log}\nexit 1\n")
    (fakebin / "conda").chmod(0o755)
    env = {**os.environ, "OMM_INSTALL_ARCH": "sm100", "OH_MY_MLIP_HOME": str(mini_home),
           "PATH": f"{fakebin}:{os.environ['PATH']}"}
    env.pop("MAMBA_EXE", None)
    proc = subprocess.run(["bash", str(mini_home / "install.sh"), "mace"], capture_output=True,
                          text=True, cwd=str(mini_home), env=env)
    assert proc.returncode == 1
    assert "STOP 'mace' (GPU sm100)" in proc.stderr and "Nothing was downloaded" in proc.stderr
    assert not log.exists(), log.read_text()
    assert not (mini_home / "envs" / "mace").exists()


# ── recipe hygiene ───────────────────────────────────────────────────────────

def test_lint_passes_on_the_repo():
    errors, _ = lint_recipes.lint()
    assert errors == []


def test_every_sibling_is_selected_and_pins_cu128():
    for stem in SIBLINGS:
        text = (ENVS / f"{stem}.yml").read_text()
        assert "# arch: sm120" in text and f"# sibling_of: {stem.split('.')[0]}" in text
        if "torch==" in text:
            assert "+cu128" in text and "whl/cu128" in text
        sidecar = ENVS / f"{stem}.build.sh"
        if sidecar.exists():
            code = "\n".join(ln for ln in sidecar.read_text().splitlines() if not ln.lstrip().startswith("#"))
            assert "+cu12" not in code.replace("+cu128", ""), stem
            assert "pt2" not in code.replace("pt27cu128", "").replace("pt29cu128", ""), stem


def test_lint_catches_a_dead_or_headerless_sibling(tmp_path):
    envs = tmp_path / "envs"
    shutil.copytree(ENVS, envs, ignore=shutil.ignore_patterns("locks"))
    (envs / "orb.cu999.yml").write_text((envs / "orb.cu128.yml").read_text())
    (envs / "chgnet.cu999.yml").write_text("# build_status: clean\nname: x\ndependencies: [python=3.11.13]\n")
    expected = json.loads((envs / "_expected.json").read_text())
    expected["orb.cu999"] = expected["orb.cu128"]
    expected["chgnet.cu999"] = {"python": "3.11.13"}
    (envs / "_expected.json").write_text(json.dumps(expected))
    errors, _ = lint_recipes.lint(envs_dir=envs, expected_json=envs / "_expected.json")
    joined = "\n".join(errors)
    assert "orb.cu999.yml: _arch_recipes.json does not select it" in joined
    assert "chgnet.cu999.yml: sibling recipe without" in joined


def test_lint_requires_a_reason_and_full_coverage(tmp_path):
    envs = tmp_path / "envs"
    shutil.copytree(ENVS, envs, ignore=shutil.ignore_patterns("locks"))
    table = json.loads((envs / "_arch_recipes.json").read_text())
    table["sm120"]["mace"] = {"recipe": None}
    del table["sm120"]["uma"]
    (envs / "_arch_recipes.json").write_text(json.dumps(table))
    errors = "\n".join(lint_recipes.lint(envs_dir=envs)[0])
    assert "sm120/mace: recipe null without a 'reason'" in errors
    assert "sm120 has no entry for env 'uma'" in errors


def test_sibling_lock_header_names_the_real_replay():
    assert gen_env_lock.replay_line("mace", ENVS) == "OMM_USE_LOCK=1 ./install.sh mace"
    line = gen_env_lock.replay_line("mace.cu128", ENVS)
    assert line.startswith("OMM_USE_LOCK=1 ./install.sh mace ") and "sm120" in line


# ── status docs show the arch on its own ─────────────────────────────────────

def test_status_states_per_arch():
    models = {"F": {"env": "grace", "versions": {"a": {}, "b": {"arch_validation": {"sm120": "validated"}}}},
              "G": {"env": "mace", "versions": {"a": {"arch_validation": {"sm120": "validated"}},
                                                  "b": {"arch_validation": {"sm120": "failed"}},
                                                  "c": {}}}}
    table = {"sm120": {"grace": {"recipe": None, "reason": "r"}, "mace": {"recipe": "mace.cu128"}}}
    assert gen_status_table.arch_state(models, "F", "b", "sm120", table) == "unsupported"
    assert gen_status_table.arch_state(models, "G", "a", "sm120", table) == "validated"
    assert gen_status_table.arch_state(models, "G", "b", "sm120", table) == "failed"
    assert gen_status_table.arch_state(models, "G", "c", "sm120", table) == "not checked"
    assert gen_status_table._framework_arch_state(models, "G", "sm120", table) == "partly validated"


def test_candidate_comes_from_the_arch_table_marker_only():
    table = {"sm120": {"grace": {"recipe": "grace.tfnightly", "status": "candidate", "reason": "nightly"},
                       "nequip": {"recipe": "nequip"}}}
    models = {"G": {"env": "grace", "versions": {"a": {"arch_validation": {"sm120": "validated"}}, "b": {}}},
              "N": {"env": "nequip", "versions": {"a": {"arch_validation": {"sm120": "validated"}}}}}
    assert gen_status_table.arch_state(models, "G", "a", "sm120", table) == "candidate"
    assert gen_status_table.arch_state(models, "G", "b", "sm120", table) == "not checked"
    # envs/nequip.yml says `# build_status: candidate` for a host reason (per-arch
    # .pt2 compile); that must not make a validated variant a candidate
    assert gen_status_table.arch_state(models, "N", "a", "sm120", table) == "validated"


def test_committed_docs_mark_only_grace_as_candidate():
    flagged = {env for env, e in TABLE["sm120"].items() if (e or {}).get("status") == "candidate"}
    assert flagged == {"grace"}
    doc = (REPO / "docs" / "model_status.md").read_text()
    rows = [ln for ln in doc.splitlines() if ln.startswith("| ") and "| candidate |" in ln]
    assert [r.split(" | ")[0] for r in rows] == ["| GRACE-2L-OAM"]
    readme = (REPO / "README.md").read_text()
    rows = [ln for ln in readme.splitlines() if ln.startswith("| ") and ln.rstrip().endswith("| candidate |")]
    assert [r.split(" | ")[0] for r in rows] == ["| GRACE"]


def test_mcp_status_agrees_with_the_generated_table():
    sys.path.insert(0, str(REPO))
    from oh_my_mlip import mcp_server
    models = json.loads((REPO / "models.json").read_text())
    for fam, info in models.items():
        if fam.startswith("_"):
            continue
        for ver in info["versions"]:
            for arch in gen_status_table.ARCH_COLUMNS:
                assert mcp_server._arch_state(models, fam, ver, arch) == gen_status_table.arch_state(models, fam, ver, arch), (fam, ver)


def test_every_arch_validation_value_is_known():
    models = json.loads((REPO / "models.json").read_text())
    for fam, info in models.items():
        if fam.startswith("_"):
            continue
        for ver, vinfo in info["versions"].items():
            for arch, state in (vinfo.get("arch_validation") or {}).items():
                assert arch in TABLE, f"{fam}/{ver}: {arch} not in envs/_arch_recipes.json"
                assert state in ("validated", "failed"), f"{fam}/{ver}: {arch}={state}"


# ── no silent fallback when the recipe cannot be chosen ─────────────────────

def _path_without_python(tmp_path: Path) -> str:
    """A PATH with every tool of the host's but no python interpreter."""
    bindir = tmp_path / "nopy_bin"
    bindir.mkdir()
    for d in os.environ["PATH"].split(os.pathsep):
        if not d or not os.path.isdir(d):
            continue
        for entry in os.scandir(d):
            name = entry.name
            if name.startswith("python") or (bindir / name).exists():
                continue
            try:
                (bindir / name).symlink_to(entry.path)
            except OSError:
                pass
    return str(bindir)


def _fake_conda_env(tmp_path: Path, path: str) -> tuple[dict, Path]:
    fakebin = tmp_path / "fakeconda"
    fakebin.mkdir()
    log = tmp_path / "conda_calls"
    (fakebin / "conda").write_text(f"#!/bin/sh\necho \"$@\" >> {log}\nexit 1\n")
    (fakebin / "conda").chmod(0o755)
    env = {**os.environ, "PATH": f"{fakebin}:{path}", "OMM_INSTALL_ARCH": "sm89"}
    env.pop("MAMBA_EXE", None)
    env.pop("CONDA_EXE", None)
    return env, log


def test_no_python_stops_instead_of_using_the_base_recipe(mini_home, tmp_path):
    env, log = _fake_conda_env(tmp_path, _path_without_python(tmp_path))
    env["OH_MY_MLIP_HOME"] = str(mini_home)
    dry = subprocess.run(["bash", str(mini_home / "install.sh"), "--dry-run", "mace"],
                         capture_output=True, text=True, cwd=str(mini_home), env=env)
    assert "STOP 'mace'" in dry.stdout and "no python3" in dry.stdout and "would create" not in dry.stdout
    real = subprocess.run(["bash", str(mini_home / "install.sh"), "mace"],
                          capture_output=True, text=True, cwd=str(mini_home), env=env)
    assert real.returncode == 1 and "no python3" in real.stderr and "Nothing was downloaded" in real.stderr
    assert not log.exists() and not (mini_home / "envs" / "mace").exists()


def test_failing_select_recipe_stops_instead_of_using_the_base_recipe(mini_home, tmp_path):
    (mini_home / "scripts" / "select_recipe.py").write_text("raise SystemExit('select_recipe exploded')\n")
    env, log = _fake_conda_env(tmp_path, os.environ["PATH"])
    env["OH_MY_MLIP_HOME"] = str(mini_home)
    for args in (["--dry-run", "mace"], ["--status", "mace"]):
        out = subprocess.run(["bash", str(mini_home / "install.sh"), *args],
                             capture_output=True, text=True, cwd=str(mini_home), env=env).stdout
        assert "select_recipe.py failed" in out and "select_recipe exploded" in out, out
        assert "would create" not in out
    real = subprocess.run(["bash", str(mini_home / "install.sh"), "mace"],
                          capture_output=True, text=True, cwd=str(mini_home), env=env)
    assert real.returncode == 1 and "select_recipe.py failed" in real.stderr
    assert not log.exists() and not (mini_home / "envs" / "mace").exists()
