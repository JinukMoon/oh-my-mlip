"""Tests for scripts/fresh_root.py -- content-addressed snapshot, disposable
runtime roots, route/isolation/source guards, guarded cleanup (plan G2).

Every test runs against a throwaway git repo built in tmp_path (git is the
only external tool; no conda, no GPU, no model env). The plan's Part 1.1
fail-closed rules and Part 7 pre-mortem 4 are each pinned by name.
"""
import errno
import importlib.util
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_SPEC = importlib.util.spec_from_file_location("fresh_root", REPO_ROOT / "scripts" / "fresh_root.py")
fr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fr)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout


def make_hub_repo(root: Path) -> Path:
    """A miniature hub: tracked recipes + models.json + a script + the REAL
    oh_my_mlip package (the isolation check runs its registry.resolve()),
    committed; plus the local state / materialized env dir that must NEVER
    be snapshotted."""
    repo = root / "hub"
    repo.mkdir()
    shutil.copytree(REPO_ROOT / "oh_my_mlip", repo / "oh_my_mlip", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "models.json").write_text(json.dumps({
        "_meta": {},
        "Alpha": {"env": "alpha", "python": "${OH_MY_MLIP_HOME}/envs/alpha/bin/python",
                  "versions": {"A-1": {"inference": ["calc = Load('${OH_MY_MLIP_HOME}/models/alpha/a1.pt')"]},
                               "A-2": {"inference": ["calc = Load('by-name')"]}},
                  "default_version": "A-1"},
    }))
    (repo / "envs").mkdir()
    (repo / "envs" / "alpha.yml").write_text("name: alpha\ndependencies:\n  - python=3.11\n")
    (repo / "envs" / "alpha.build.sh").write_text("#!/bin/sh\nexit 0\n")
    (repo / "envs" / "_expected.json").write_text("{}\n")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "tool.py").write_text("print('tool')\n")
    (repo / "install.sh").write_text("#!/bin/sh\nexit 0\n")
    (repo / "install.sh").chmod((repo / "install.sh").stat().st_mode | stat.S_IEXEC)
    (repo / ".gitignore").write_text("envs/*/\n!envs/locks/\nmodels/\n.sweep/\n*.local.json\n__pycache__/\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    # local state + materialized env + weights: excluded from every snapshot
    (repo / "env_map.local.json").write_text(json.dumps({"alpha": "/somewhere/else"}))
    (repo / "models.local.json").write_text("{}")
    (repo / "envs" / "alpha").mkdir()
    (repo / "envs" / "alpha" / "bin").mkdir()
    (repo / "envs" / "alpha" / "bin" / "python").write_text("")
    (repo / "models").mkdir()
    (repo / "models" / "big.pt").write_bytes(b"\0" * 10)
    (repo / ".sweep").mkdir()
    (repo / ".sweep" / "old.jsonl").write_text("")
    return repo


@pytest.fixture
def hub(tmp_path):
    return make_hub_repo(tmp_path)


# ── allowlist ────────────────────────────────────────────────────────────────

def test_allowlist_is_generated_from_approved_part31_list(tmp_path):
    out = tmp_path / "allow.json"
    rc = fr.main(["--repo", str(tmp_path), "allowlist", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["patterns"] == fr.APPROVED_NEW_PATHS
    assert "recipes/*.md" in data["patterns"] and "scripts/fresh_root.py" in data["patterns"]
    # the coordinator-approved G1 contract test is an EXACT path, no blanket allow
    assert "tests/test_recipe_contract.py" in data["patterns"]
    assert not any(p in ("tests/*.py", "tests/*", "*") for p in data["patterns"])
    assert "Part 3.1" in data["source"]
    # the file is read back only as a subset of the approved constant
    assert fr.load_allowlist(out) == fr.APPROVED_NEW_PATHS
    widened = tmp_path / "widened.json"
    widened.write_text(json.dumps({"patterns": ["recipes/*.md", "tests/*.py"]}))
    with pytest.raises(fr.FreshRootError) as exc:
        fr.load_allowlist(widened)
    assert exc.value.state == "failed(snapshot:allowlist_unapproved)" and exc.value.detail == ["tests/*.py"]


# ── snapshot: include / exclude / fail-closed ────────────────────────────────

def test_snapshot_includes_tracked_env_recipes_and_excludes_local_state(hub):
    m = fr.build_manifest(hub, [])
    files = m["files"]
    assert "envs/alpha.yml" in files and "envs/alpha.build.sh" in files and "envs/_expected.json" in files
    assert "models.json" in files and "scripts/tool.py" in files and "install.sh" in files
    for bad in ("env_map.local.json", "models.local.json", "envs/alpha/bin/python", "models/big.pt", ".sweep/old.jsonl"):
        assert bad not in files
    assert not any(p.startswith(".git/") for p in files)
    assert m["manifest_sha256"] == fr.aggregate_manifest_sha256(files)
    assert m["git_head"] == _git(hub, "rev-parse", "HEAD").strip()


def test_snapshot_reads_working_tree_not_head(hub):
    """Pre-mortem 4: an edited tracked file enters the manifest with its NEW
    hash (working-tree read), and is reported as dirty."""
    before = fr.build_manifest(hub, [])["files"]["scripts/tool.py"]
    (hub / "scripts" / "tool.py").write_text("print('edited candidate code')\n")
    m = fr.build_manifest(hub, [])
    assert m["files"]["scripts/tool.py"] != before
    assert m["files"]["scripts/tool.py"] == fr.sha256_file(hub / "scripts" / "tool.py")
    assert "scripts/tool.py" in m["dirty_tracked_paths"]


def test_snapshot_includes_allowlisted_new_files(hub):
    (hub / "recipes").mkdir()
    (hub / "recipes" / "setup.md").write_text("# recipe\n")
    (hub / "scripts" / "fresh_root.py").write_text("# candidate\n")
    m = fr.build_manifest(hub, ["recipes/*.md", "scripts/fresh_root.py"])
    assert "recipes/setup.md" in m["files"] and "scripts/fresh_root.py" in m["files"]
    assert sorted(m["included_untracked"]) == ["recipes/setup.md", "scripts/fresh_root.py"]


def test_snapshot_includes_env_locks_but_never_materialized_env_prefixes(hub):
    """envs/locks/*.txt (exact-replay locks) is source; envs/<env>/ is a built prefix."""
    (hub / "envs" / "locks").mkdir()
    (hub / "envs" / "locks" / "alpha.pip.txt").write_text("torch==2.7.1+cu126\n")
    m = fr.build_manifest(hub, ["envs/locks/*.txt"])
    assert "envs/locks/alpha.pip.txt" in m["files"]
    assert m["included_untracked"] == ["envs/locks/alpha.pip.txt"]
    assert "envs/alpha/bin/python" not in m["files"]


def test_unexpected_untracked_code_fails_closed_listing_paths(hub):
    """Pre-mortem 4: an unlisted scripts/x.py must never be silently omitted."""
    (hub / "scripts" / "x.py").write_text("# not approved\n")
    (hub / "notes.txt").write_text("non-code is omitted but listed\n")
    with pytest.raises(fr.FreshRootError) as exc:
        fr.build_manifest(hub, ["recipes/*.md"])
    assert exc.value.state == "failed(snapshot:unexpected_untracked)"
    assert exc.value.detail == ["scripts/x.py"]
    # with it allowlisted the non-code untracked file is visible in the manifest
    m = fr.build_manifest(hub, ["scripts/x.py"])
    assert m["omitted_untracked_noncode"] == ["notes.txt"]


def test_tracked_deletion_fails_closed(hub):
    (hub / "scripts" / "tool.py").unlink()
    with pytest.raises(fr.FreshRootError) as exc:
        fr.build_manifest(hub, [])
    assert exc.value.state == "failed(snapshot:tracked_deleted)"
    assert exc.value.detail == ["scripts/tool.py"]


def test_symlink_escape_fails_closed(hub, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("secret\n")
    (hub / "recipes").mkdir()
    os.symlink(outside, hub / "recipes" / "leak.md")
    with pytest.raises(fr.FreshRootError) as exc:
        fr.build_manifest(hub, ["recipes/*.md"])
    assert exc.value.state == "failed(snapshot:symlink_escape)"
    assert exc.value.detail == ["recipes/leak.md"]


def test_untracked_token_pattern_never_enters_snapshot(hub):
    (hub / "recipes").mkdir()
    (hub / "recipes" / "my_token.md").write_text("hf_xxx\n")
    m = fr.build_manifest(hub, ["recipes/*.md"])
    assert "recipes/my_token.md" not in m["files"]


def test_snapshot_check_cli_writes_nothing(hub, tmp_path):
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"patterns": []}))
    rc = fr.main(["--repo", str(hub), "snapshot", "--check", "--allowlist", str(allow),
                  "--campaign-dir", str(tmp_path / "camp")])
    assert rc == 0
    assert not (tmp_path / "camp").exists()
    assert not (hub / ".sweep" / "snapshot").exists()


def test_snapshot_is_content_addressed_and_read_only(hub, tmp_path):
    m = fr.build_manifest(hub, [])
    camp = tmp_path / "camp"
    w1 = fr.write_snapshot(hub, m, camp)
    w2 = fr.write_snapshot(hub, m, camp)
    assert w1["reused"] is False and w2["reused"] is True
    assert Path(w1["snapshot_dir"]).name == m["manifest_sha256"][:12]
    for p in (w1["tarball"], w1["manifest"]):
        assert not os.access(p, os.W_OK)


# ── materialize ──────────────────────────────────────────────────────────────

def _snapshot(hub, tmp_path, allow=None):
    m = fr.build_manifest(hub, allow or [])
    w = fr.write_snapshot(hub, m, tmp_path / "camp")
    return m, Path(w["snapshot_dir"])


def test_materialize_extracts_creates_writable_dirs_and_ownership(hub, tmp_path):
    m, snap = _snapshot(hub, tmp_path)
    runtime = tmp_path / ".omm_fresh" / "alpha_1"
    res = fr.materialize(snap, runtime, campaign_id="c1", repo=hub, allowlist=[])
    assert res["manifest_sha256"] == m["manifest_sha256"]
    assert (runtime / "models.json").is_file() and (runtime / "envs" / "alpha.yml").is_file()
    assert os.access(runtime / "install.sh", os.X_OK)
    for d in fr.WRITABLE_DIRS:
        assert (runtime / d).is_dir()
    assert not (runtime / "env_map.local.json").exists()
    rec = json.loads((runtime / fr.OWNERSHIP_FILE).read_text())
    assert rec["owner"] == fr.OWNER_TAG and rec["runtime_root"] == str(runtime.resolve())
    assert rec["manifest_sha256"] == m["manifest_sha256"]
    # every mandatory route is exported inside the runtime root
    for var in fr.ROUTE_VARS:
        assert res["exports"][var].startswith(str(runtime.resolve()))
    env_sh = (runtime / fr.ENV_EXPORT_FILE).read_text()
    assert 'export CONDA_PKGS_DIRS="' in env_sh and str(runtime.resolve()) in env_sh
    # the campaign-EXTERNAL ownership binding: canonical root, parent, manifest, campaign
    registry = tmp_path / "camp" / fr.OWNED_ROOTS_FILE
    assert res["owned_registry"] == str(registry) and rec["owned_registry"] == str(registry)
    entry = json.loads(registry.read_text())["roots"][str(runtime.resolve())]
    assert entry["runtime_parent"] == str(runtime.resolve().parent) and entry["campaign_id"] == "c1"
    assert entry["manifest_sha256"] == m["manifest_sha256"] and entry["cleaned_utc"] is None
    # a registry inside the root is refused; re-registering a live root is refused
    with pytest.raises(fr.FreshRootError) as exc:
        fr.materialize(snap, tmp_path / "rt2", campaign_id="c1", repo=hub, allowlist=[],
                       owned_registry=tmp_path / "rt2" / "owned.json")
    assert exc.value.state == "failed(ownership:registry_inside_runtime)"
    with pytest.raises(fr.FreshRootError) as exc:
        fr.register_owned_root(registry, runtime, rec)
    assert exc.value.state == "failed(ownership:already_registered)"


def test_materialize_refuses_stale_snapshot_unless_allowed(hub, tmp_path):
    m, snap = _snapshot(hub, tmp_path)
    (hub / "scripts" / "tool.py").write_text("changed after snapshot\n")
    with pytest.raises(fr.FreshRootError) as exc:
        fr.materialize(snap, tmp_path / "rt", campaign_id="c", repo=hub, allowlist=[])
    assert exc.value.state == "failed(materialize:stale_snapshot)"
    res = fr.materialize(snap, tmp_path / "rt2", campaign_id="c", repo=hub, allowlist=[], allow_stale=True)
    assert res["ownership"]["stale_snapshot"]["snapshot"] == m["manifest_sha256"]


def test_materialize_recheck_blocks_on_unexpected_untracked(hub, tmp_path):
    """`snapshot --check` runs before EVERY materialization and blocks it."""
    m, snap = _snapshot(hub, tmp_path)
    (hub / "scripts" / "rogue.py").write_text("x\n")
    with pytest.raises(fr.FreshRootError) as exc:
        fr.materialize(snap, tmp_path / "rt", campaign_id="c", repo=hub, allowlist=[])
    assert exc.value.state == "failed(snapshot:unexpected_untracked)"
    assert not (tmp_path / "rt").exists()


def test_materialize_refuses_non_empty_runtime(hub, tmp_path):
    _, snap = _snapshot(hub, tmp_path)
    rt = tmp_path / "rt"
    rt.mkdir()
    (rt / "junk").write_text("")
    with pytest.raises(fr.FreshRootError) as exc:
        fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    assert exc.value.state == "failed(materialize:runtime_not_empty)"


# ── seed-cache ───────────────────────────────────────────────────────────────

def test_seed_cache_copies_never_symlinks_and_records_hashes(hub, tmp_path):
    _, snap = _snapshot(hub, tmp_path)
    rt = tmp_path / "rt"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    src_dir = tmp_path / "native_cache" / "hf" / "model-x"
    src_dir.mkdir(parents=True)
    (src_dir / "w.bin").write_bytes(b"weights")
    rec = fr.seed_cache(rt, {"cache/shared/hf/model-x": str(src_dir)})
    dst = rt / "cache" / "shared" / "hf" / "model-x" / "w.bin"
    assert dst.is_file() and not dst.is_symlink() and not (dst.parent.is_symlink())
    assert rec["weights_source"] == "reused-native-cache"
    assert rec["seeded"][0]["sha256"] == fr.sha256_file(src_dir / "w.bin")
    assert (src_dir / "w.bin").read_bytes() == b"weights"  # source untouched
    assert json.loads((rt / ".sweep" / "seed" / "seeded.json").read_text())["seeded"]
    with pytest.raises(fr.FreshRootError) as exc:
        fr.seed_cache(rt, {"cache/shared/hf/model-x": str(src_dir)})
    assert exc.value.state == "failed(seed:dst_exists)"
    with pytest.raises(fr.FreshRootError) as exc:
        fr.seed_cache(rt, {"../escape": str(src_dir)})
    assert exc.value.state == "failed(seed:dst_outside_runtime)"


# ── verify: routes / isolation / sources ─────────────────────────────────────

def _fake_conda(tmp_path, pkgs_dirs_json: str) -> list[str]:
    script = tmp_path / "fake_conda.sh"
    script.write_text(f"#!/bin/sh\necho '{pkgs_dirs_json}'\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return [str(script)]


def test_routes_pass_when_every_route_inside_root_and_conda_agrees(hub, tmp_path):
    _, snap = _snapshot(hub, tmp_path)
    rt = tmp_path / "rt"
    res = fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    exports = res["exports"]
    conda = _fake_conda(tmp_path, json.dumps({"pkgs_dirs": [exports["CONDA_PKGS_DIRS"]]}))
    out = fr.verify_routes(rt, exports, conda_cmd=conda)
    assert out["ok"] and out["state"] == "passed"


def test_routes_abort_on_out_of_root_cache_or_shared_conda_pkgs(hub, tmp_path):
    _, snap = _snapshot(hub, tmp_path)
    rt = tmp_path / "rt"
    exports = fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])["exports"]
    bad = dict(exports, HF_HOME=str(tmp_path / "home" / ".cache" / "huggingface"))
    conda = _fake_conda(tmp_path, json.dumps({"pkgs_dirs": [exports["CONDA_PKGS_DIRS"]]}))
    out = fr.verify_routes(rt, bad, conda_cmd=conda)
    assert not out["ok"] and out["state"] == "failed(routes:out_of_root)"
    assert "HF_HOME" in out["problems"]
    # conda pointing at the shared ~/miniconda3/pkgs is refused even if the var is right
    conda_shared = _fake_conda(tmp_path, json.dumps({"pkgs_dirs": ["/home/user/miniconda3/pkgs"]}))
    out = fr.verify_routes(rt, exports, conda_cmd=conda_shared)
    assert not out["ok"] and "conda" in out["problems"]
    # an unset route is a problem, never silently defaulted
    out = fr.verify_routes(rt, {k: v for k, v in exports.items() if k != "TMPDIR"}, conda_cmd=conda)
    assert out["problems"]["TMPDIR"] == "unset"
    # XDG_CACHE_HOME only covers XDG-aware libraries; literal $HOME writers
    # (env.sh sections 4/5, MatRIS/TACE os.mkdir, conda environments.txt) need
    # HOME itself routed to the isolated campaign home inside the root
    assert exports["XDG_CACHE_HOME"].startswith(str(rt.resolve()))
    assert exports["HOME"] == str(rt.resolve() / "home") and (rt / "home").is_dir()
    out = fr.verify_routes(rt, dict(exports, XDG_CACHE_HOME=str(tmp_path / "home" / ".cache")), conda_cmd=conda)
    assert "XDG_CACHE_HOME" in out["problems"]
    out = fr.verify_routes(rt, dict(exports, HOME=str(tmp_path / "realhome")), conda_cmd=conda)
    assert "HOME" in out["problems"]
    # JIT / compile caches that default under $HOME are routed and asserted too
    for var in ("TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR", "HF_HUB_CACHE", "CONDA_ENVS_DIRS"):
        assert exports[var].startswith(str(rt.resolve()))
        assert var in fr.verify_routes(rt, {k: v for k, v in exports.items() if k != var}, conda_cmd=conda)["problems"]
    # an inherited override huggingface_hub / conda would honour over the routed var is refused
    out = fr.verify_routes(rt, dict(exports, TRANSFORMERS_CACHE=str(tmp_path / "home" / ".cache")), conda_cmd=conda)
    assert "inherited override" in out["problems"]["TRANSFORMERS_CACHE"]
    # a read-only input must point at an existing regular file (a path export, never a copy)
    out = fr.verify_routes(rt, dict(exports, HF_TOKEN_PATH=str(tmp_path / "nope")), conda_cmd=conda)
    assert "HF_TOKEN_PATH" in out["problems"]


def test_cycle_env_is_the_effective_child_environment(hub, tmp_path):
    """build_cycle_env drops inherited cache overrides, routes HOME into the
    root and re-supplies the user's read-only inputs as PATH exports; the
    route check runs on THAT env, not on the export list."""
    _, snap = _snapshot(hub, tmp_path)
    rt = tmp_path / "rt"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    real_home = tmp_path / "realhome"
    (real_home / ".cache" / "huggingface").mkdir(parents=True)
    (real_home / ".cache" / "huggingface" / "token").write_text("hf_not-a-real-token\n")
    (real_home / ".condarc").write_text("channels: [defaults]\n")
    base = {"PATH": "/usr/bin", "HOME": str(real_home),
            "TRANSFORMERS_CACHE": str(real_home / ".cache" / "hf"), "HUGGINGFACE_HUB_CACHE": "/x",
            "CONDA_ENVS_PATH": "/y", "OMM_HOME": str(hub), "HF_HOME": str(real_home / ".cache" / "huggingface")}
    env, report = fr.build_cycle_env(rt, base)
    assert report["rejected_inherited"] == ["CONDA_ENVS_PATH", "HUGGINGFACE_HUB_CACHE", "OMM_HOME", "TRANSFORMERS_CACHE"]
    assert not any(v in env for v in fr.REJECT_INHERITED_VARS)
    assert env["HOME"] == str(rt.resolve() / "home") and env["HF_HOME"].startswith(str(rt.resolve()))
    assert env["PATH"] == "/usr/bin"
    # read-only inputs: the real files are pointed at, nothing copied into the root
    assert env["HF_TOKEN_PATH"] == str(real_home / ".cache" / "huggingface" / "token")
    assert env["CONDARC"] == str(real_home / ".condarc")
    assert report["read_only_inputs"] == {"HF_TOKEN_PATH": env["HF_TOKEN_PATH"], "CONDARC": env["CONDARC"]}
    assert not list((rt / "home").rglob("token"))
    # an explicitly set HF_TOKEN_PATH is respected, never replaced
    env2, _ = fr.build_cycle_env(rt, dict(base, HF_TOKEN_PATH=str(real_home / "alt_token")))
    assert env2["HF_TOKEN_PATH"] == str(real_home / "alt_token")
    conda = _fake_conda(tmp_path, json.dumps({"pkgs_dirs": [env["CONDA_PKGS_DIRS"]]}))
    assert fr.verify_routes(rt, env, conda_cmd=conda)["ok"]
    # the exported env file carries the same routing (HOME included) and unsets the overrides
    text = (rt / fr.ENV_EXPORT_FILE).read_text()
    assert f'export HOME="{rt.resolve() / "home"}"' in text and "unset " in text and "TRANSFORMERS_CACHE" in text


def test_env_sh_home_writes_land_in_the_isolated_home(tmp_path):
    """env.sh sections 4/5 write literal $HOME paths (symlink seed for
    matris/tace/eqnorm, torch_extensions seed). With the campaign env those
    writes land under <root>/home; the real home is untouched. No env.sh
    change is needed for that -- this pins the behaviour."""
    root = tmp_path / "rt"
    (root / "models" / "matris").mkdir(parents=True)
    (root / "models" / "torch_ext" / "k").mkdir(parents=True)
    (root / "models" / "torch_ext" / "k" / "kernel.so").write_text("bin")
    (root / "home").mkdir()
    real_home = tmp_path / "realhome"
    real_home.mkdir()
    exports = fr.route_exports(root)
    env = {"PATH": os.environ.get("PATH", "/usr/bin"), **exports}
    subprocess.run(["bash", "-c", f'source "{REPO_ROOT / "env.sh"}"'], env=env, check=True,
                   cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert (root / "home" / ".cache" / "matris").is_symlink()
    assert (root / "cache" / "torch_ext" / "k" / "kernel.so").is_file()
    assert not list(real_home.iterdir()), "the real home received a campaign write"
    # the same env.sh against the REAL home would have written there (the hazard this closes)
    env_real = dict(env, HOME=str(real_home))
    env_real.pop("TORCH_EXTENSIONS_DIR")
    subprocess.run(["bash", "-c", f'source "{REPO_ROOT / "env.sh"}"'], env=env_real, check=True,
                   cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert (real_home / ".cache" / "matris").is_symlink() and (real_home / ".cache" / "torch_extensions").is_dir()


def test_isolation_fails_on_adoption_entry_or_out_of_root_interpreter(hub, tmp_path):
    """The check is the runtime copy's OWN registry.resolve() run with
    OH_MY_MLIP_HOME = the root -- not a re-implementation of its path logic."""
    _, snap = _snapshot(hub, tmp_path)
    rt = tmp_path / "rt"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    out = fr.verify_isolation(rt)
    assert out["ok"], out["problems"]
    assert out["resolved"] == 2 and "registry.resolve()" in out["probe"]
    (rt / "env_map.local.json").write_text(json.dumps({"alpha": "/home/user/miniconda3/envs/Alpha"}))
    out = fr.verify_isolation(rt)
    assert out["state"] == "failed(isolation)" and "adoption entry" in out["problems"][0]
    # resolve() itself refuses the dead adopted prefix -- reported verbatim
    assert any("resolve() failed" in p for p in out["problems"])
    (rt / "env_map.local.json").unlink()
    reg = json.loads((rt / "models.json").read_text())
    reg["Alpha"]["python"] = "/home/user/miniconda3/envs/Alpha/bin/python"
    (rt / "models.json").write_text(json.dumps(reg))
    out = fr.verify_isolation(rt)
    assert out["state"] == "failed(isolation)" and "interpreter route outside" in out["problems"][0]
    # a weight path resolve() expands to somewhere outside the root is caught the same way
    reg["Alpha"]["python"] = "${OH_MY_MLIP_HOME}/envs/alpha/bin/python"
    reg["Alpha"]["versions"]["A-1"]["inference"] = ["calc = Load('/home/user/models/alpha/a1.pt')"]
    (rt / "models.json").write_text(json.dumps(reg))
    out = fr.verify_isolation(rt)
    assert out["state"] == "failed(isolation)" and "weight route outside" in out["problems"][0]
    # a runtime whose registry cannot resolve at all is an isolation failure, never a pass
    (rt / "models.json").write_text("{not json")
    assert fr.verify_isolation(rt)["state"] == "failed(isolation)"


def test_source_mutation_detected(hub, tmp_path):
    m, snap = _snapshot(hub, tmp_path)
    rt = tmp_path / "rt"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    assert fr.verify_sources(rt)["ok"]
    (rt / "scripts" / "tool.py").write_text("mutated in runtime\n")
    (rt / "envs" / "alpha.yml").unlink()
    out = fr.verify_sources(rt)
    assert out["state"] == "failed(source_mutated)"
    assert out["mutated"] == ["scripts/tool.py"] and out["missing"] == ["envs/alpha.yml"]
    # designated outputs never count as drift
    (rt / "models" / "new.pt").write_bytes(b"x")
    (rt / "envs" / "alpha.yml").write_text((hub / "envs" / "alpha.yml").read_text())
    (rt / "scripts" / "tool.py").write_text((hub / "scripts" / "tool.py").read_text())
    assert fr.verify_sources(rt)["ok"]


# ── cleanup guards ───────────────────────────────────────────────────────────

def _ledger_with(tmp_path, sha) -> Path:
    ledger = tmp_path / "camp" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({"seq": 1, "manifest_sha256": sha}) + "\n")
    return ledger


def test_preserve_copies_logs_and_named_artifacts_verified_by_hash(hub, tmp_path):
    _, snap = _snapshot(hub, tmp_path)
    rt = tmp_path / ".omm_fresh" / "alpha_1"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    (rt / ".sweep" / "verify").mkdir(parents=True)
    (rt / ".sweep" / "verify" / "alpha.A-1.log").write_text("verify log\n")
    (rt / ".ft" / "alpha" / "A-1").mkdir(parents=True)
    (rt / ".ft" / "alpha" / "A-1" / "finetune_A-1.sh").write_text("#!/bin/sh\n")
    ckpt = rt / ".ft" / "alpha" / "A-1" / "A-1.model"
    ckpt.write_bytes(b"\x01\x02" * 500)
    (rt / ".ft" / "alpha" / "A-1" / "data.extxyz").write_text("dataset -- never preserved by suffix\n")
    dest = tmp_path / "camp" / "alpha_1"
    rec = fr.preserve_artifacts(rt, dest, extra_files=[str(ckpt)])
    dsts = {Path(f["dst"]).relative_to(dest).as_posix() for f in rec["files"]}
    assert dsts == {".sweep/verify/alpha.A-1.log", ".ft/alpha/A-1/finetune_A-1.sh", ".ft/alpha/A-1/A-1.model"}
    assert all(f["sha256"] == fr.sha256_file(Path(f["dst"])) and f["bytes"] > 0 for f in rec["files"])
    record = Path(rec["record"])
    assert record == dest / fr.PRESERVED_FILE and json.loads(record.read_text())["manifest_sha256"] == rec["manifest_sha256"]
    assert fr.verify_preserved(record, rt)["file_count"] == 3
    # a named artifact that does not exist refuses (a missing checkpoint is not preservable),
    # whichever list named it: there is no silently-skipped optional artifact
    with pytest.raises(fr.FreshRootError) as exc:
        fr.preserve_artifacts(rt, tmp_path / "camp2", required=[".ft/alpha/A-1/missing.model"])
    assert exc.value.state == "failed(preserve:required_missing)"
    with pytest.raises(fr.FreshRootError) as exc:
        fr.preserve_artifacts(rt, tmp_path / "camp2", extra_files=[str(ckpt), ".ft/alpha/A-1/missing.model"])
    assert exc.value.state == "failed(preserve:required_missing)" and exc.value.detail == [".ft/alpha/A-1/missing.model"]
    assert not (tmp_path / "camp2").exists()  # nothing partial was written
    # a symlink, or a file outside the root, is not a preservable artifact either
    os.symlink(ckpt, rt / ".ft" / "alpha" / "A-1" / "link.model")
    outside = tmp_path / "outside.model"
    outside.write_bytes(b"x")
    with pytest.raises(fr.FreshRootError) as exc:
        fr.preserve_artifacts(rt, tmp_path / "camp2", extra_files=[".ft/alpha/A-1/link.model", str(outside)])
    assert set(exc.value.detail) == {".ft/alpha/A-1/link.model", str(outside)}
    # a destination inside the root is not durable
    with pytest.raises(fr.FreshRootError) as exc:
        fr.preserve_artifacts(rt, rt / "work" / "keep")
    assert exc.value.state == "failed(preserve:dest_inside_runtime)"
    # a copy that no longer re-verifies (tampered/lost after preserve) is caught
    (dest / ".ft" / "alpha" / "A-1" / "A-1.model").write_bytes(b"truncated")
    with pytest.raises(fr.FreshRootError) as exc:
        fr.verify_preserved(record, rt)
    assert exc.value.state == "failed(cleanup:evidence_not_durable)"


def test_cleanup_removes_only_owned_root_after_durable_evidence(hub, tmp_path):
    m, snap = _snapshot(hub, tmp_path)
    parent = tmp_path / ".omm_fresh"
    rt = parent / "alpha_1"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    registry = tmp_path / "camp" / fr.OWNED_ROOTS_FILE
    bound = dict(owned_registry=registry, allowed_parent=parent)
    (rt / "envs" / "alpha").mkdir()
    (rt / "envs" / "alpha" / "big").write_bytes(b"\0" * 100)
    (rt / ".sweep" / "run.log").write_text("log\n")
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    preserved = Path(fr.preserve_artifacts(rt, tmp_path / "camp" / "alpha_1")["record"])
    # no evidence ledger carrying this manifest => refuse
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(rt, _ledger_with(tmp_path, "0" * 64), preserved=preserved, proc_root=proc_root, **bound)
    assert exc.value.state == "failed(cleanup:evidence_not_durable)"
    assert rt.exists()
    # evidence inside the runtime is not durable either
    inner = rt / ".sweep" / "ledger.jsonl"
    inner.write_text(json.dumps({"manifest_sha256": m["manifest_sha256"]}) + "\n")
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(rt, inner, preserved=preserved, proc_root=proc_root, **bound)
    assert exc.value.state == "failed(cleanup:evidence_inside_runtime)"
    ledger = _ledger_with(tmp_path, m["manifest_sha256"])
    # a ledger row alone is NOT enough: the preserved copies must exist and re-verify
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(rt, ledger, proc_root=proc_root, **bound)
    assert exc.value.state == "failed(cleanup:no_preservation_record)" and rt.exists()
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(rt, ledger, preserved=inner, proc_root=proc_root, **bound)
    assert exc.value.state == "failed(cleanup:evidence_inside_runtime)"
    (tmp_path / "camp" / "alpha_1" / ".sweep" / "run.log").unlink()
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(rt, ledger, preserved=preserved, proc_root=proc_root, **bound)
    assert exc.value.state == "failed(cleanup:evidence_not_durable)" and rt.exists()
    preserved = Path(fr.preserve_artifacts(rt, tmp_path / "camp" / "alpha_1b")["record"])
    # a process living in the root blocks removal
    (proc_root / "4242").mkdir()
    os.symlink(rt / "envs" / "alpha", proc_root / "4242" / "cwd")
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(rt, ledger, preserved=preserved, proc_root=proc_root, **bound)
    assert exc.value.state == "failed(cleanup:in_use)" and exc.value.detail == [{"pid": 4242, "via": "cwd"}]
    os.unlink(proc_root / "4242" / "cwd")
    # the entry is still there but nothing of it can be read: "could not
    # look" is not "nobody", the root is retained
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(rt, ledger, preserved=preserved, proc_root=proc_root, **bound)
    assert exc.value.state == "failed(cleanup:proc_uninspectable)" and exc.value.detail["pids"] == [4242]
    assert rt.exists()
    (proc_root / "4242").rmdir()  # the process exited
    res = fr.cleanup(rt, ledger, preserved=preserved, protected=[hub], proc_root=proc_root, **bound)
    # proof of cleanup is the owned path's absence; df is a diagnostic only
    assert res["ok"] and res["absent"] and not rt.exists() and res["bytes_freed"] > 0
    # the quiescence scan is part of the record (what was inspected before removal)
    assert res["process_scan"]["proc_root"] == str(proc_root) and res["process_scan"]["owned_groups"] == []
    assert "envs" in res["removed_children"] and "note" in res["df_diagnostic_gib"]
    assert Path(res["record"]).is_file()
    assert res["allowed_parent"] == str(parent.resolve()) and res["owned_registry"] == str(registry.resolve())
    # the registry keeps the audit trail; the preserved evidence is untouched by the removal
    assert json.loads(registry.read_text())["roots"][str(rt.resolve())]["cleaned_utc"]
    assert (tmp_path / "camp" / "alpha_1b" / ".sweep" / "run.log").read_text() == "log\n"


def test_cleanup_is_bounded_to_the_registered_root_under_the_allowed_parent(hub, tmp_path):
    """The in-root marker is never trusted alone: removal needs the
    campaign-external registry entry whose canonical root / parent /
    manifest / campaign all match, a symlink-free path, and a direct parent
    equal to the canonical allowed parent."""
    m, snap = _snapshot(hub, tmp_path)
    parent = tmp_path / ".omm_fresh"
    rt = parent / "alpha_1"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    registry = tmp_path / "camp" / fr.OWNED_ROOTS_FILE
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    ledger = _ledger_with(tmp_path, m["manifest_sha256"])
    preserved = Path(fr.preserve_artifacts(rt, tmp_path / "camp" / "alpha_1")["record"])

    def attempt(root=rt, **kw):
        args = dict(preserved=preserved, owned_registry=registry, allowed_parent=parent, proc_root=proc_root)
        args.update(kw)
        with pytest.raises(fr.FreshRootError) as exc:
            fr.cleanup(root, ledger, **args)
        assert rt.exists(), "a refused cleanup must leave the root in place"
        return exc.value

    # no bound parent / no registry => refused before anything else is trusted
    assert attempt(allowed_parent=None).state == "failed(cleanup:no_allowed_parent)"
    assert attempt(owned_registry=None).state == "failed(cleanup:no_ownership_registry)"
    # a different (even existing) parent, or a registry that never bound this root
    other = tmp_path / "elsewhere"
    other.mkdir()
    assert attempt(allowed_parent=other).state == "failed(cleanup:outside_allowed_parent)"
    assert attempt(owned_registry=tmp_path / "camp" / "nothing.json").state == "failed(cleanup:not_registered)"
    assert attempt(owned_registry=rt / fr.OWNERSHIP_FILE).state == "failed(cleanup:registry_inside_runtime)"
    # a symlink to the root, or a symlinked parent, is not the canonical owned path
    link = tmp_path / "rt_link"
    os.symlink(rt, link)
    assert attempt(root=link).state == "failed(cleanup:symlink_root)"
    plink = tmp_path / "parent_link"
    os.symlink(parent, plink)
    assert attempt(root=plink / "alpha_1").state == "failed(cleanup:symlink_root)"
    assert attempt(allowed_parent=plink).state == "failed(cleanup:symlink_parent)"
    # a forged in-root marker cannot re-bind the root: the registry entry must match it
    marker = json.loads((rt / fr.OWNERSHIP_FILE).read_text())
    (rt / fr.OWNERSHIP_FILE).write_text(json.dumps(dict(marker, campaign_id="forged")))
    err = attempt()
    assert err.state == "failed(cleanup:registry_mismatch)" and "campaign_id" in err.detail
    (rt / fr.OWNERSHIP_FILE).write_text(json.dumps(marker))
    # a registry entry whose parent binding differs from the canonical parent is refused
    data = json.loads(registry.read_text())
    data["roots"][str(rt.resolve())]["runtime_parent"] = str(other)
    registry.write_text(json.dumps(data))
    assert "runtime_parent" in attempt().detail
    data["roots"][str(rt.resolve())]["runtime_parent"] = str(parent.resolve())
    registry.write_text(json.dumps(data))
    # an unregistered directory carrying a copied marker under the allowed parent: still refused
    clone = parent / "alpha_clone"
    shutil.copytree(rt, clone, symlinks=True)
    forged = dict(marker, runtime_root=str(clone.resolve()))
    (clone / fr.OWNERSHIP_FILE).write_text(json.dumps(forged))
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(clone, ledger, preserved=preserved, owned_registry=registry, allowed_parent=parent,
                   proc_root=proc_root)
    assert exc.value.state == "failed(cleanup:not_registered)" and clone.exists()
    # everything bound and matching => removed; a second attempt sees the audit-trailed entry
    res = fr.cleanup(rt, ledger, preserved=preserved, owned_registry=registry, allowed_parent=parent,
                     proc_root=proc_root)
    assert res["absent"] and not rt.exists() and clone.exists()


def _fake_proc_entry(proc_root: Path, pid: int, *, pgrp: int | None = None, session: int | None = None,
                     cwd: Path | None = None, fds: dict[int, Path] | None = None,
                     maps: list[Path] | None = None, cmdline_only: bool = False,
                     stat_only: bool = False, uid: int | None = None, cap_eff: int = 0) -> Path:
    """`stat_only` mimics a ptrace-opaque pid: world-readable stat/status,
    every use probe (cwd/exe/root/fd/maps) absent -- the code takes the
    same `except OSError` branch for ENOENT as for the real EACCES."""
    entry = proc_root / str(pid)
    entry.mkdir()
    if cmdline_only:
        (entry / "cmdline").write_bytes(b"sleep\0")
        return entry
    (entry / "stat").write_text(f"{pid} (a comm) with) parens) S 1 {pgrp or pid} {session or pid} 0 -1 4194560 0\n")
    if uid is not None:
        (entry / "status").write_text(f"Name:\tsvc\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\nCapEff:\t{cap_eff:016x}\n")
    if stat_only:
        return entry
    if cwd is not None:
        os.symlink(cwd, entry / "cwd")
    (entry / "fd").mkdir()
    for n, target in (fds or {}).items():
        os.symlink(target, entry / "fd" / str(n))
    lines = [f"7f0000000000-7f0000001000 r-xp 00000000 08:01 {i}    {p}" for i, p in enumerate(maps or [], 1)]
    lines.append("7f0000002000-7f0000003000 rw-p 00000000 00:00 0")
    lines.append("7ffd00000000-7ffd00021000 rw-p 00000000 00:00 0                          [stack]")
    (entry / "maps").write_text("\n".join(lines) + "\n")
    return entry


def test_cleanup_quiescence_scan_fails_closed_and_sees_fds_maps_and_owned_groups(hub, tmp_path):
    """Before removal every live process is inspected: an open fd or a mapped
    file inside the root, a cwd/exe/root link into it, or membership of a
    launched session (owned_groups) refuses; an unreadable proc root refuses
    outright (never "nobody"); an entry that cannot be inspected at all is
    reported, not silently treated as idle."""
    m, snap = _snapshot(hub, tmp_path)
    parent = tmp_path / ".omm_fresh"
    rt = parent / "alpha_1"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    (rt / "envs" / "alpha" / "lib").mkdir(parents=True)
    (rt / "envs" / "alpha" / "lib" / "libx.so").write_bytes(b"\0" * 10)
    (rt / "work" / "log.txt").write_text("x\n")
    registry = tmp_path / "camp" / fr.OWNED_ROOTS_FILE
    ledger = _ledger_with(tmp_path, m["manifest_sha256"])
    preserved = Path(fr.preserve_artifacts(rt, tmp_path / "camp" / "alpha_1")["record"])
    bound = dict(preserved=preserved, owned_registry=registry, allowed_parent=parent)

    def refused(proc_root, **kw):
        with pytest.raises(fr.FreshRootError) as exc:
            fr.cleanup(rt, ledger, proc_root=proc_root, **bound, **kw)
        assert rt.exists()
        return exc.value

    # unreadable proc root => fail closed
    err = refused(tmp_path / "no_such_proc")
    assert err.state == "failed(cleanup:proc_unreadable)" and "no_such_proc" in err.detail["proc_root"]
    err = refused(tmp_path / "camp" / "alpha_1" / fr.PRESERVED_FILE)  # a file, not a directory
    assert err.state == "failed(cleanup:proc_unreadable)"
    # an open fd inside the root
    proc = tmp_path / "proc_fd"
    proc.mkdir()
    _fake_proc_entry(proc, 100, cwd=tmp_path, fds={0: Path("/dev/null"), 5: rt / "work" / "log.txt"})
    assert refused(proc).detail == [{"pid": 100, "via": "fd:5"}]
    # a mapped shared object inside the root (cwd and fds elsewhere)
    proc = tmp_path / "proc_maps"
    proc.mkdir()
    _fake_proc_entry(proc, 101, cwd=tmp_path, fds={0: Path("/dev/null")},
                     maps=[Path("/usr/lib/libc.so.6"), rt / "envs" / "alpha" / "lib" / "libx.so"])
    assert refused(proc).detail == [{"pid": 101, "via": "maps"}]
    # a member of a launched session (a phase child's orphaned descendant): nothing of it
    # touches the root any more, but it belongs to an owned group
    proc = tmp_path / "proc_group"
    proc.mkdir()
    _fake_proc_entry(proc, 102, pgrp=7000, session=5000, cwd=tmp_path, fds={})
    assert refused(proc, owned_groups=[5000]).detail == [{"pid": 102, "via": "group:5000"}]
    assert refused(proc, owned_groups=[7000]).detail == [{"pid": 102, "via": "group:7000"}]
    # an entry with nothing inspectable that is STILL PRESENT blocks (B5): "could
    # not look" never vouches for quiescence; one that vanished during the scan
    # (its /proc entry is gone: a dangling link stands in for the race) is an
    # exited process and is ignored, recorded as `vanished`
    proc = tmp_path / "proc_ok"
    proc.mkdir()
    _fake_proc_entry(proc, 103, pgrp=7000, session=5000, cwd=tmp_path, fds={0: Path("/dev/null")},
                     maps=[Path("/usr/lib/libc.so.6")])
    _fake_proc_entry(proc, 104, cmdline_only=True)
    os.symlink(proc / "gone_during_scan", proc / "105")
    (proc / "not_a_pid").mkdir()
    scan = fr._pids_using(rt, proc, owned_groups=[9999])
    assert scan["users"] == [] and scan["inspected"] == 1
    assert scan["uninspectable"] == [104] and scan["vanished"] == [105]
    err = refused(proc, owned_groups=[9999])
    assert err.state == "failed(cleanup:proc_uninspectable)"
    assert err.detail == {"pids": [104], "vanished": [105], "proc_root": str(proc)}
    shutil.rmtree(proc / "104")  # it exited
    res = fr.cleanup(rt, ledger, proc_root=proc, owned_groups=[9999], **bound)
    assert res["absent"] and res["process_scan"] == {"inspected": 1, "uninspectable": [], "vanished": [105],
                                                     "unknown_use": [], "excluded_by_perm": [], "quiescence": "total",
                                                     "root_private": True,
                                                     "owned_groups": [9999], "proc_root": str(proc)}


def test_stat_only_pids_never_count_as_inspected_and_block_unless_excluded_by_permission(hub, tmp_path):
    """A ptrace-opaque pid (stat/status readable, every use probe denied) is
    the class a stock kernel actually presents -- ~60 of them on this host.
    It must NOT be counted as inspected for root use: group identity from
    `stat` is not a use inspection. It is `unknown_use` with a census and
    blocks cleanup (failed(cleanup:proc_unknown_use)), except a process of a
    foreign non-root uid without DAC capabilities against a PRIVATE (0700)
    root, which provably cannot reach it (`excluded_by_perm`). uid 0, our own
    opaque uid, DAC-capable or unidentifiable pids stay unknown; the scan
    reports quiescence 'partial' and never claims total."""
    m, snap = _snapshot(hub, tmp_path)
    parent = tmp_path / ".omm_fresh"
    rt = parent / "alpha_1"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    assert fr.root_privacy(rt) == {"path": str(rt), "mode": "0700", "uid": os.getuid(), "private": True}
    registry = tmp_path / "camp" / fr.OWNED_ROOTS_FILE
    ledger = _ledger_with(tmp_path, m["manifest_sha256"])
    preserved = Path(fr.preserve_artifacts(rt, tmp_path / "camp" / "alpha_1")["record"])
    bound = dict(preserved=preserved, owned_registry=registry, allowed_parent=parent)
    me = os.getuid()
    proc = tmp_path / "proc"
    proc.mkdir()
    _fake_proc_entry(proc, 200, cwd=tmp_path, fds={0: Path("/dev/null")})            # fully inspected, idle
    _fake_proc_entry(proc, 201, stat_only=True, uid=0)                                 # root daemon
    _fake_proc_entry(proc, 202, stat_only=True, uid=me)                                # (sd-pam)-like, same uid
    _fake_proc_entry(proc, 203, stat_only=True, uid=me + 1000, cap_eff=0)              # service, no DAC caps
    _fake_proc_entry(proc, 204, stat_only=True, uid=me + 1000, cap_eff=1 << fr.CAP_DAC_READ_SEARCH)
    _fake_proc_entry(proc, 205, stat_only=True)                                        # no status at all
    _fake_proc_entry(proc, 206, stat_only=True, pgrp=5000, session=5000, uid=me + 1000)  # opaque but OWNED group

    scan = fr._pids_using(rt, proc, owned_groups=[5000], root_private=True)
    assert scan["inspected"] == 1 and scan["quiescence"] == "partial"
    assert scan["users"] == [{"pid": 206, "via": "group:5000"}]
    assert scan["uninspectable"] == [] and scan["vanished"] == []
    assert scan["excluded_by_perm"] == [203]
    assert [(c["pid"], c["reason"]) for c in scan["unknown_use"]] == [
        (201, "uid_0"), (202, "same_uid_ptrace_opaque"), (204, "dac_capability"), (205, "identity_unreadable")]
    assert scan["unknown_use"][0]["uids"] == [0, 0, 0, 0] and scan["unknown_use"][2]["cap_eff"] == "0000000000000004"
    # without a private root even the service pid cannot be excluded
    scan = fr._pids_using(rt, proc, owned_groups=[5000], root_private=False)
    assert scan["excluded_by_perm"] == [] and (203, "root_not_private") in [(c["pid"], c["reason"]) for c in scan["unknown_use"]]

    # cleanup: the owned-group member blocks first; then the unknown class blocks
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(rt, ledger, proc_root=proc, owned_groups=[5000], **bound)
    assert exc.value.state == "failed(cleanup:in_use)"
    shutil.rmtree(proc / "206")
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(rt, ledger, proc_root=proc, owned_groups=[5000], **bound)
    assert exc.value.state == "failed(cleanup:proc_unknown_use)" and rt.exists()
    d = exc.value.detail
    assert d["pids"] == [201, 202, 204, 205] and d["excluded_by_perm"] == [203]
    assert d["quiescence"] == "partial" and d["root"]["private"] is True
    assert "no total quiescence is claimed" in d["note"]
    # only the provably-excluded pid remains -> total quiescence -> cleaned
    for pid in (201, 202, 204, 205):
        shutil.rmtree(proc / str(pid))
    res = fr.cleanup(rt, ledger, proc_root=proc, owned_groups=[5000], **bound)
    assert res["absent"] and res["process_scan"]["quiescence"] == "total"
    assert res["process_scan"]["excluded_by_perm"] == [203] and res["process_scan"]["inspected"] == 1


def test_quiescence_classifier_against_the_real_proc_does_not_raise(hub, tmp_path):
    """Smoke test on the live /proc: the three buckets exist, nothing raises,
    and no pid is both inspected and unknown."""
    _, snap = _snapshot(hub, tmp_path)
    rt = tmp_path / ".omm_fresh" / "alpha_1"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    scan = fr._pids_using(rt, Path("/proc"), root_private=fr.root_privacy(rt)["private"])
    assert scan["quiescence"] in ("total", "partial")
    unknown = {c["pid"] for c in scan["unknown_use"]}
    assert unknown.isdisjoint(scan["excluded_by_perm"]) and unknown.isdisjoint(scan["uninspectable"])
    assert scan["inspected"] >= 1
    assert all(c["reason"] in {"uid_0", "same_uid_ptrace_opaque", "dac_capability", "identity_unreadable",
                               "root_not_private"} for c in scan["unknown_use"])


def test_materialize_creates_a_private_root_and_refuses_a_shared_one(hub, tmp_path):
    """New owned roots are 0700 (verified, recorded); an existing empty
    directory that is not private is refused, never chmod'ed."""
    _, snap = _snapshot(hub, tmp_path)
    rt = tmp_path / ".omm_fresh" / "alpha_1"
    res = fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    assert stat.S_IMODE(rt.stat().st_mode) == 0o700
    assert res["ownership"]["root_mode"] == "0700" and res["ownership"]["root_uid"] == os.getuid()
    shared = tmp_path / ".omm_fresh" / "alpha_2"
    shared.mkdir(mode=0o755)
    os.chmod(shared, 0o755)
    with pytest.raises(fr.FreshRootError) as exc:
        fr.materialize(snap, shared, campaign_id="c", repo=hub, allowlist=[])
    assert exc.value.state == "failed(materialize:root_not_private)" and exc.value.detail["mode"] == "0755"
    assert stat.S_IMODE(shared.stat().st_mode) == 0o755 and not any(shared.iterdir())


def test_cleanup_refuses_a_mount_point_inside_the_root_before_removing_anything(hub, tmp_path):
    """A mounted filesystem below the root (kernel mount table, octal-escaped
    path) or an unreadable mount table refuses BEFORE rmtree -- post-removal
    residue cannot protect mounted external data."""
    m, snap = _snapshot(hub, tmp_path)
    parent = tmp_path / ".omm_fresh"
    rt = parent / "alpha_1"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    (rt / "envs" / "alpha" / "ext data").mkdir(parents=True)
    (rt / "envs" / "alpha" / "ext data" / "precious.bin").write_bytes(b"x")
    registry = tmp_path / "camp" / fr.OWNED_ROOTS_FILE
    ledger = _ledger_with(tmp_path, m["manifest_sha256"])
    preserved = Path(fr.preserve_artifacts(rt, tmp_path / "camp" / "alpha_1")["record"])
    proc = tmp_path / "proc"
    proc.mkdir()
    bound = dict(preserved=preserved, owned_registry=registry, allowed_parent=parent, proc_root=proc)
    escaped = str(rt / "envs" / "alpha" / "ext data").replace(" ", "\\040")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("22 1 8:1 / / rw - ext4 /dev/sda1 rw\n"
                         f"99 22 8:9 / {escaped} rw,relatime - ext4 /dev/sdb1 rw\n")
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(rt, ledger, mountinfo=mountinfo, **bound)
    assert exc.value.state == "failed(cleanup:mount_inside_root)"
    assert exc.value.detail["mounts"] == [str(rt / "envs" / "alpha" / "ext data")]
    assert (rt / "envs" / "alpha" / "ext data" / "precious.bin").exists() and rt.exists()
    # the root itself being a mount point refuses as well
    mountinfo.write_text(f"99 22 8:9 / {rt} rw - ext4 /dev/sdb1 rw\n")
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(rt, ledger, mountinfo=mountinfo, **bound)
    assert exc.value.state == "failed(cleanup:mount_inside_root)" and exc.value.detail["mounts"] == [str(rt)]
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(rt, ledger, mountinfo=tmp_path / "no_such_mountinfo", **bound)
    assert exc.value.state == "failed(cleanup:mounts_unreadable)" and rt.exists()
    mountinfo.write_text("22 1 8:1 / / rw - ext4 /dev/sda1 rw\n")
    # a subtree the device-boundary walk cannot list is not known to be
    # mount-free: refused the same way, nothing removed
    locked = rt / "envs" / "alpha" / "locked"
    locked.mkdir()
    (locked / "inner.bin").write_bytes(b"y")
    os.chmod(locked, 0)
    try:
        with pytest.raises(fr.FreshRootError) as exc:
            fr.cleanup(rt, ledger, mountinfo=mountinfo, **bound)
    finally:
        os.chmod(locked, 0o755)
    assert exc.value.state == "failed(cleanup:mounts_unreadable)" and (locked / "inner.bin").exists()
    assert any("locked" in u for u in exc.value.detail["unreadable"])
    # a listed mount point is recorded and NOT descended into by the walk
    assert fr.mounts_inside(rt, mountinfo)["mounts"] == []
    # a mount table that names nothing below the root lets the removal proceed
    res = fr.cleanup(rt, ledger, mountinfo=mountinfo, **bound)
    assert res["absent"] and res["removal"]["removed"] == res["removal"]["files"] + res["removal"]["dirs"] > 0


def test_removal_is_bounded_a_mount_appearing_after_the_precheck_stops_it_and_retains_the_root(hub, tmp_path):
    """The pre-check is not proof at removal time (check-to-remove window):
    the remover re-reads the mount table right before the first unlink and
    stops on a mount or device boundary with NOTHING removed
    (failed(cleanup:remove_stopped)), root retained. Modelled by a mount
    table that gains an entry below the root after `mounts_inside` returned;
    a natural race is not reproduced here, and removal is not claimed
    race-proof."""
    m, snap = _snapshot(hub, tmp_path)
    parent = tmp_path / ".omm_fresh"
    rt = parent / "alpha_1"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    late = rt / "envs" / "alpha" / "late_mount"
    late.mkdir(parents=True)
    (late / "external.bin").write_bytes(b"not ours")
    (rt / "work" / "log.txt").write_text("x\n")
    registry = tmp_path / "camp" / fr.OWNED_ROOTS_FILE
    ledger = _ledger_with(tmp_path, m["manifest_sha256"])
    preserved = Path(fr.preserve_artifacts(rt, tmp_path / "camp" / "alpha_1")["record"])
    proc = tmp_path / "proc"
    proc.mkdir()
    bound = dict(preserved=preserved, owned_registry=registry, allowed_parent=parent, proc_root=proc)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("22 1 8:1 / / rw - ext4 /dev/sda1 rw\n")
    before = sorted(str(p.relative_to(rt)) for p in rt.rglob("*"))
    real_precheck = fr.mounts_inside

    def precheck_then_mount(root, mi):
        out = real_precheck(root, mi)  # the pre-check sees a clean tree ...
        with mountinfo.open("a") as fh:  # ... then a mount lands below the root
            fh.write(f"99 22 8:9 / {late} rw - ext4 /dev/sdb1 rw\n")
        return out

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(fr, "mounts_inside", precheck_then_mount)
    try:
        with pytest.raises(fr.FreshRootError) as exc:
            fr.cleanup(rt, ledger, mountinfo=mountinfo, **bound)
    finally:
        monkeypatch.undo()
    assert exc.value.state == "failed(cleanup:remove_stopped)"
    assert exc.value.detail["reason"] == "mount_point_appeared" and exc.value.detail["removed"] == 0
    assert exc.value.detail["at"] == [str(late)]
    # nothing was removed: the tree is exactly what it was, the "mounted" data intact
    assert sorted(str(p.relative_to(rt)) for p in rt.rglob("*")) == before
    assert (late / "external.bin").read_bytes() == b"not ours"
    # the same boundary seen during the listing walk (device-boundary model:
    # the re-read table names the directory) stops before any unlink too
    mountinfo.write_text("22 1 8:1 / / rw - ext4 /dev/sda1 rw\n")
    with pytest.raises(fr.FreshRootError) as exc:
        fr._remove_tree_bounded(rt, tmp_path / "no_such_mountinfo")
    assert exc.value.state == "failed(cleanup:mounts_unreadable)"
    assert sorted(str(p.relative_to(rt)) for p in rt.rglob("*")) == before
    # a clean table: the bounded removal completes, symlinks inside are unlinked, never followed
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep\n")
    os.symlink(outside, rt / "work" / "link_out")
    res = fr.cleanup(rt, ledger, mountinfo=mountinfo, **bound)
    assert res["absent"] and (outside / "keep.txt").read_text() == "keep\n"


def test_eacces_stat_pid_is_uninspectable_and_refuses_cleanup_without_deleting(hub, tmp_path):
    """A /proc entry that is still present but not even stat-able (EACCES,
    not ENOENT/ESRCH) is `uninspectable`, never `vanished`: cleanup refuses
    (failed(cleanup:proc_uninspectable)) and nothing is deleted. Modelled by
    a proc root that is listable but not searchable, so every entry stat
    fails with EACCES while the listing still names it."""
    if os.geteuid() == 0:
        pytest.skip("permission model needs a non-root uid")
    m, snap = _snapshot(hub, tmp_path)
    parent = tmp_path / ".omm_fresh"
    rt = parent / "alpha_1"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    (rt / "work" / "log.txt").write_text("x\n")
    registry = tmp_path / "camp" / fr.OWNED_ROOTS_FILE
    ledger = _ledger_with(tmp_path, m["manifest_sha256"])
    preserved = Path(fr.preserve_artifacts(rt, tmp_path / "camp" / "alpha_1")["record"])
    bound = dict(preserved=preserved, owned_registry=registry, allowed_parent=parent)
    proc = tmp_path / "proc_eacces"
    proc.mkdir()
    _fake_proc_entry(proc, 300, cwd=tmp_path, fds={0: Path("/dev/null")})
    os.chmod(proc, 0o400)  # readable (listing works), not searchable (stat of entries: EACCES)
    try:
        with pytest.raises(PermissionError):
            os.stat(proc / "300")
        scan = fr._pids_using(rt, proc)
        assert scan["uninspectable"] == [300] and scan["vanished"] == [] and scan["inspected"] == 0
        assert scan["quiescence"] == "partial"
        with pytest.raises(fr.FreshRootError) as exc:
            fr.cleanup(rt, ledger, proc_root=proc, **bound)
    finally:
        os.chmod(proc, 0o755)
    assert exc.value.state == "failed(cleanup:proc_uninspectable)"
    assert exc.value.detail == {"pids": [300], "vanished": [], "proc_root": str(proc)}
    assert rt.exists() and (rt / "work" / "log.txt").read_text() == "x\n"
    # readable again: fully inspected and idle -> cleaned
    res = fr.cleanup(rt, ledger, proc_root=proc, **bound)
    assert res["absent"] and res["process_scan"]["uninspectable"] == [] and res["process_scan"]["inspected"] == 1


def test_quiescence_scan_against_the_real_proc_sees_a_live_child(hub, tmp_path):
    """Real /proc: a sleeping child launched in its own session with cwd in
    the root is seen by cwd, and by owned-group membership when its session
    id is handed over -- and disappears from the scan once it exits."""
    _, snap = _snapshot(hub, tmp_path)
    rt = tmp_path / ".omm_fresh" / "alpha_1"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    (rt / "work").mkdir(exist_ok=True)
    child = subprocess.Popen(["sleep", "30"], cwd=str(rt / "work"), start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        scan = fr._pids_using(rt, Path("/proc"))
        assert {"pid": child.pid, "via": "cwd"} in scan["users"]
        scan = fr._pids_using(rt, Path("/proc"), owned_groups=[child.pid])
        assert {"pid": child.pid, "via": f"group:{child.pid}"} in scan["users"]
        assert scan["inspected"] >= 1
    finally:
        child.kill()
        child.wait()
    scan = fr._pids_using(rt, Path("/proc"), owned_groups=[child.pid])
    assert all(u["pid"] != child.pid for u in scan["users"])


def test_cleanup_refuses_non_owned_paths_and_git_repos(hub, tmp_path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    ledger = _ledger_with(tmp_path, "x")
    plain = tmp_path / "user_data"
    plain.mkdir()
    (plain / "results.csv").write_text("precious\n")
    preserved = tmp_path / "camp" / fr.PRESERVED_FILE
    bound = dict(owned_registry=tmp_path / "camp" / fr.OWNED_ROOTS_FILE, allowed_parent=tmp_path)
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(plain, ledger, preserved=preserved, proc_root=proc_root, **bound)
    assert exc.value.state == "failed(cleanup:not_owned)" and plain.exists()
    # forged ownership record pointing elsewhere
    (plain / fr.OWNERSHIP_FILE).write_text(json.dumps({"owner": fr.OWNER_TAG, "runtime_root": "/elsewhere",
                                                        "manifest_sha256": "x"}))
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(plain, ledger, preserved=preserved, proc_root=proc_root, **bound)
    assert exc.value.state == "failed(cleanup:ownership_mismatch)" and plain.exists()
    # a self-consistent forged marker with NO registry entry is still refused
    (plain / fr.OWNERSHIP_FILE).write_text(json.dumps({"owner": fr.OWNER_TAG, "runtime_root": str(plain.resolve()),
                                                        "manifest_sha256": "x", "campaign_id": "c"}))
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(plain, ledger, preserved=preserved, proc_root=proc_root, **bound)
    assert exc.value.state == "failed(cleanup:not_registered)" and (plain / "results.csv").exists()
    # the hub itself (a git repo) is never a cleanup target even with a record and a registry entry
    rec = {"owner": fr.OWNER_TAG, "runtime_root": str(hub.resolve()), "manifest_sha256": "x", "campaign_id": "c"}
    (hub / fr.OWNERSHIP_FILE).write_text(json.dumps(rec))
    fr.register_owned_root(bound["owned_registry"], hub, rec)
    with pytest.raises(fr.FreshRootError) as exc:
        fr.cleanup(hub, ledger, preserved=preserved, proc_root=proc_root, **bound)
    assert exc.value.state == "failed(cleanup:git_repo)" and (hub / "models.json").exists()
    (hub / fr.OWNERSHIP_FILE).unlink()


def test_materialize_cli_always_rechecks_the_working_tree(hub, tmp_path):
    """The pre-materialization working-tree check is mandatory: the CLI has
    no way to skip it, and a drifted tree is refused."""
    _, snap = _snapshot(hub, tmp_path)
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"patterns": []}))
    with pytest.raises(SystemExit):
        fr.main(["--repo", str(hub), "materialize", "--snapshot-dir", str(snap), "--runtime", str(tmp_path / "rt"),
                 "--campaign-id", "c", "--no-recheck"])
    (hub / "scripts" / "tool.py").write_text("edited after the snapshot\n")
    rc = fr.main(["--repo", str(hub), "materialize", "--snapshot-dir", str(snap), "--runtime", str(tmp_path / "rt"),
                  "--campaign-id", "c", "--allowlist", str(allow)])
    assert rc == 1 and not (tmp_path / "rt").exists()


def test_snapshot_refuses_a_file_edited_between_hash_and_tar(hub, tmp_path):
    m = fr.build_manifest(hub, [])
    (hub / "scripts" / "tool.py").write_text("changed after hashing\n")
    with pytest.raises(fr.FreshRootError) as exc:
        fr.write_snapshot(hub, m, tmp_path / "camp")
    assert exc.value.state == "failed(snapshot:changed_during_write)" and exc.value.detail == "scripts/tool.py"
    assert not (tmp_path / "camp" / "snapshot" / m["manifest_sha256"][:12] / fr.TARBALL_FILE).exists()


def test_du_bytes_counts_hardlinks_once(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a").write_bytes(b"\0" * 1000)
    os.link(d / "a", d / "b")
    assert fr.du_bytes(d) == 1000


def test_disk_free_bytes_is_a_measurement_of_the_nearest_existing_ancestor(tmp_path):
    free = fr.disk_free_bytes(tmp_path / "not" / "yet" / "created")
    assert isinstance(free, int) and free == fr.disk_free_bytes(tmp_path) or free > 0
    assert fr.GIB == 2 ** 30 and abs(fr.GB_PER_GIB - 1.073741824) < 1e-12


def test_cli_verify_reports_state_and_exit(hub, tmp_path):
    _, snap = _snapshot(hub, tmp_path)
    rt = tmp_path / "rt"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    proc = subprocess.run([str(REPO_ROOT / "scripts" / "fresh_root.py"), "--repo", str(hub), "verify",
                           "--runtime", str(rt), "--sources", "--isolation"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["ok"] and out["checks"]["sources"]["state"] == "passed"


# --- handle-based removal (fixtures only; never a real campaign root) -------

def _tree(root: Path) -> set[str]:
    """Every entry below `root` as a root-relative path; symlinks are entries,
    never followed."""
    seen = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for n in dirnames + filenames:
            seen.add(os.path.relpath(os.path.join(dirpath, n), root))
    return seen


def _owned_root(hub, tmp_path):
    """A materialized, registered, evidence-complete root whose cleanup would
    pass every pre-removal guard: the fixture for removal-time behaviour."""
    m, snap = _snapshot(hub, tmp_path)
    parent = tmp_path / ".omm_fresh"
    rt = parent / "alpha_1"
    fr.materialize(snap, rt, campaign_id="c", repo=hub, allowlist=[])
    alpha = rt / "envs" / "alpha"
    alpha.mkdir(parents=True, exist_ok=True)
    (alpha / "aaa.bin").write_bytes(b"sorted before sub")   # unlinked before `sub` is reached
    sub = alpha / "sub"
    sub.mkdir()
    (sub / "a.bin").write_bytes(b"ours")
    (sub / "b.bin").write_bytes(b"ours")
    (rt / "work" / "zzz.txt").write_text("sorted after envs\n")  # still present after a stop inside envs
    registry = tmp_path / "camp" / fr.OWNED_ROOTS_FILE
    ledger = _ledger_with(tmp_path, m["manifest_sha256"])
    preserved = Path(fr.preserve_artifacts(rt, tmp_path / "camp" / "alpha_1")["record"])
    proc = tmp_path / "proc"
    proc.mkdir(exist_ok=True)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("22 1 8:1 / / rw - ext4 /dev/sda1 rw\n")
    bound = dict(preserved=preserved, owned_registry=registry, allowed_parent=parent, proc_root=proc,
                 mountinfo=mountinfo)
    outside = tmp_path / "outside"
    outside.mkdir()
    for n in ("a.bin", "b.bin", "keep.txt"):
        (outside / n).write_bytes(b"sentinel " + n.encode())
    return rt, ledger, bound, outside


def _sentinels_intact(outside: Path) -> bool:
    return all((outside / n).read_bytes() == b"sentinel " + n.encode() for n in ("a.bin", "b.bin", "keep.txt"))


def test_removal_by_handle_a_directory_swapped_to_a_symlink_after_its_lstat_is_refused_outside_untouched(hub, tmp_path):
    """The post-listing parent-swap case: after the remover lstat-ed
    `envs/alpha/sub` (a directory) and before it opened it, the directory is
    renamed away and a symlink to an outside directory holding same-named
    sentinel files is put in its place. A pathname remover would unlink the
    sentinels through the link; the handle remover opens `sub` from its
    parent handle with RESOLVE_NO_SYMLINKS|O_NOFOLLOW, the kernel refuses,
    and removal stops with exact partial accounting: what was unlinked before
    the stop is gone, nothing outside the root was resolved."""
    rt, ledger, bound, outside = _owned_root(hub, tmp_path)
    alpha = rt / "envs" / "alpha"
    sub = alpha / "sub"
    before = _tree(rt)
    real_stat = fr._stat_at
    hits = []

    def stat_then_swap(dfd, name):
        st = real_stat(dfd, name)
        if name == "sub" and stat.S_ISDIR(st.st_mode):
            hits.append(1)
            if len(hits) == 2:  # 1st lstat: verification walk; 2nd: removal walk -> swap now
                os.rename(sub, alpha / "sub_moved")
                os.symlink(outside, sub)
        return st

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(fr, "_stat_at", stat_then_swap)
    try:
        with pytest.raises(fr.FreshRootError) as exc:
            fr.cleanup(rt, ledger, **bound)
    finally:
        monkeypatch.undo()
    d = exc.value.detail
    assert exc.value.state == "failed(cleanup:remove_stopped)"
    assert d["reason"] == "entry_changed_during_removal" and d["phase"] == "remove"
    assert d["at"] == "./envs/alpha/sub" and d["errno"] in ("ENOTDIR", "ELOOP")
    # outside sentinel bytes untouched; the moved-away directory untouched; the link is still a link
    assert _sentinels_intact(outside)
    assert (alpha / "sub_moved" / "a.bin").read_bytes() == b"ours" and (alpha / "sub_moved" / "b.bin").read_bytes() == b"ours"
    assert sub.is_symlink() and os.readlink(sub) == str(outside)
    # partial removal is real and exactly accounted: aaa.bin (sorted before sub) is gone,
    # work/zzz.txt (sorted after envs) is still there, root retained
    assert not (alpha / "aaa.bin").exists() and (rt / "work" / "zzz.txt").exists() and rt.is_dir()
    after = _tree(rt)
    swapped = {"envs/alpha/sub", "envs/alpha/sub/a.bin", "envs/alpha/sub/b.bin"}
    assert d["removed"] == d["removed_files"] + d["removed_dirs"] == len((before - after) - swapped)
    assert d["removed_files"] >= 1 and "root retained" in d["note"] and "NOT covered" in d["note"]


def test_removal_by_handle_a_directory_replaced_by_another_directory_after_its_lstat_is_refused(hub, tmp_path):
    """Same window, same-device directory swap: `sub` is renamed away and an
    outside directory renamed into its place between lstat and open. The open
    succeeds (it is a real directory) but the handle's (st_dev, st_ino) is not
    the lstat-ed one: identity_changed, nothing in the moved-in directory
    touched. BOUNDARY, shown deterministically below: a directory moved into
    the 0700 root BEFORE the remover inspects it is indistinguishable from
    owned content and is removed -- that is a write inside the owned root by
    a process with access to it, not a traversal outside it."""
    rt, ledger, bound, outside = _owned_root(hub, tmp_path)
    alpha = rt / "envs" / "alpha"
    sub = alpha / "sub"
    real_stat = fr._stat_at
    hits = []

    def stat_then_move_in(dfd, name):
        st = real_stat(dfd, name)
        if name == "sub" and stat.S_ISDIR(st.st_mode):
            hits.append(1)
            if len(hits) == 2:
                os.rename(sub, alpha / "sub_moved")
                os.rename(outside, sub)
        return st

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(fr, "_stat_at", stat_then_move_in)
    try:
        with pytest.raises(fr.FreshRootError) as exc:
            fr.cleanup(rt, ledger, **bound)
    finally:
        monkeypatch.undo()
    d = exc.value.detail
    assert exc.value.state == "failed(cleanup:remove_stopped)"
    assert d["reason"] == "identity_changed" and d["phase"] == "remove" and d["at"] == "./envs/alpha/sub"
    assert _sentinels_intact(sub)  # the moved-in directory (now at sub) is intact
    assert (alpha / "sub_moved" / "a.bin").read_bytes() == b"ours" and rt.is_dir()
    # boundary demonstration: content moved into the root before inspection is removed as owned
    os.rename(sub, outside)          # put the outside directory back ...
    os.rename(alpha / "sub_moved", sub)
    moved_in = alpha / "moved_in_before_inspection"
    os.rename(outside, moved_in)     # ... then move it INTO the root before cleanup starts
    res = fr.cleanup(rt, ledger, **bound)
    assert res["absent"] and not moved_in.exists() and not outside.exists()


def test_removal_by_handle_injected_mount_and_state_changes_stop_with_accounting_or_refuse(hub, tmp_path):
    """Mount changes are modelled by the errno the kernel returns for them
    (no real mounts in a fixture): EXDEV from openat2(RESOLVE_NO_XDEV) on a
    directory that is a mount point, EBUSY from rmdir on a directory a mount
    was placed over after its handle was opened, ENOSYS when openat2 is
    missing. Each case: sentinel bytes untouched, exact accounting, root
    retained, and no pathname fallback."""
    rt, ledger, bound, outside = _owned_root(hub, tmp_path)
    alpha = rt / "envs" / "alpha"
    late = alpha / "late_mount"
    late.mkdir()
    (late / "external.bin").write_bytes(b"not ours")
    before = _tree(rt)
    real_open = fr._open_child_dir
    monkeypatch = pytest.MonkeyPatch()

    # (i) a mount point present when the verification walk opens it: nothing removed
    def exdev_always(dfd, name):
        if name == "late_mount":
            raise OSError(errno.EXDEV, "Invalid cross-device link", name)
        return real_open(dfd, name)

    monkeypatch.setattr(fr, "_open_child_dir", exdev_always)
    try:
        with pytest.raises(fr.FreshRootError) as exc:
            fr.cleanup(rt, ledger, **bound)
    finally:
        monkeypatch.undo()
    d = exc.value.detail
    assert exc.value.state == "failed(cleanup:remove_stopped)"
    assert d["reason"] == "mount_or_device_boundary" and d["phase"] == "verify" and d["errno"] == "EXDEV"
    assert d["removed"] == 0 and _tree(rt) == before and (late / "external.bin").read_bytes() == b"not ours"

    # (ii) openat2 unavailable: refused outright, nothing removed, no pathname fallback
    def enosys(dirfd, name, resolve):
        raise OSError(errno.ENOSYS, "Function not implemented", name)

    monkeypatch.setattr(fr, "_openat2", enosys)
    try:
        with pytest.raises(fr.FreshRootError) as exc:
            fr.cleanup(rt, ledger, **bound)
    finally:
        monkeypatch.undo()
    assert exc.value.state == "failed(cleanup:remove_unsupported)"
    assert exc.value.detail["primitive"] == "openat2" and exc.value.detail["errno"] == "ENOSYS"
    assert exc.value.detail["removed"] == 0 and _tree(rt) == before

    # (iii) a mount appearing after the verification walk: the removal walk's open refuses it,
    # everything unlinked before it is gone and counted, the "mounted" data untouched
    hits = []

    def exdev_second_time(dfd, name):
        if name == "late_mount":
            hits.append(1)
            if len(hits) == 2:
                raise OSError(errno.EXDEV, "Invalid cross-device link", name)
        return real_open(dfd, name)

    monkeypatch.setattr(fr, "_open_child_dir", exdev_second_time)
    try:
        with pytest.raises(fr.FreshRootError) as exc:
            fr.cleanup(rt, ledger, **bound)
    finally:
        monkeypatch.undo()
    d = exc.value.detail
    assert d["reason"] == "mount_or_device_boundary" and d["phase"] == "remove" and d["at"] == "./envs/alpha/late_mount"
    assert (late / "external.bin").read_bytes() == b"not ours" and rt.is_dir()
    assert not (alpha / "aaa.bin").exists()                       # sorted before late_mount: gone
    assert (rt / "work" / "zzz.txt").exists() and (alpha / "sub" / "a.bin").exists()  # sorted after: kept
    after = _tree(rt)
    assert d["removed"] == d["removed_files"] + d["removed_dirs"] == len(before - after) >= 1

    # (iv) a mount placed over a directory AFTER its handle was opened: the handle stays bound to the
    # underlying directory (its own entries are removed through it), the rmdir of the covered name
    # fails EBUSY and stops; modelled by the errno, since the fixture cannot mount
    (alpha / "aaa.bin").write_bytes(b"again")
    before = _tree(rt)
    real_rmdir = fr._rmdir_at

    def ebusy_on_late(dfd, name):
        if name == "late_mount":
            raise OSError(errno.EBUSY, "Device or resource busy", name)
        real_rmdir(dfd, name)

    monkeypatch.setattr(fr, "_rmdir_at", ebusy_on_late)
    try:
        with pytest.raises(fr.FreshRootError) as exc:
            fr.cleanup(rt, ledger, **bound)
    finally:
        monkeypatch.undo()
    d = exc.value.detail
    assert d["reason"].startswith("os_error") and d["errno"] == "EBUSY" and d["at"] == "./envs/alpha/late_mount"
    assert late.is_dir() and rt.is_dir()
    assert d["removed"] == d["removed_files"] + d["removed_dirs"] == len(before - _tree(rt))
    assert _sentinels_intact(outside)


def test_removal_by_handle_ordinary_owned_cleanup_unlinks_links_never_their_targets(hub, tmp_path):
    """The ordinary case: a symlink to an outside directory, a symlink to an
    outside file and a hardlink to an outside file inside the root are
    removed as links; the outside bytes and the hardlinked inode survive.
    The result carries the primitive used and full accounting."""
    rt, ledger, bound, outside = _owned_root(hub, tmp_path)
    work = rt / "work"
    os.symlink(outside, work / "link_dir")
    os.symlink(outside / "keep.txt", work / "link_file")
    os.link(outside / "a.bin", work / "hard_a.bin")
    assert os.stat(outside / "a.bin").st_nlink == 2
    n_files = sum(1 for p in _tree(rt) if not (rt / p).is_dir() or (rt / p).is_symlink())
    n_dirs = sum(1 for p in _tree(rt) if (rt / p).is_dir() and not (rt / p).is_symlink())
    res = fr.cleanup(rt, ledger, **bound)
    assert res["absent"] and res["state"] == "cleaned"
    assert _sentinels_intact(outside) and os.stat(outside / "a.bin").st_nlink == 1
    rem = res["removal"]
    assert rem["files"] == n_files and rem["dirs"] == n_dirs + 1        # + the root itself
    assert rem["removed"] == rem["removed_files"] + rem["removed_dirs"] == rem["files"] + rem["dirs"]
    assert rem["verified"] == {"files": n_files, "dirs": n_dirs}
    assert rem["primitive"].startswith("openat2(RESOLVE_BENEATH|NO_SYMLINKS|NO_XDEV")


def test_openat2_primitive_refuses_symlink_mount_and_escape_on_this_kernel(tmp_path):
    """The kernel primitive itself, on the machine the tests run on: a symlink
    child is refused (ENOTDIR/ELOOP), `..`/absolute escape is refused (EXDEV
    from RESOLVE_BENEATH), a real mount point is refused (EXDEV), a plain
    child directory opens and its handle identity matches lstat. Skipped
    where openat2 is unavailable -- the remover then refuses outright, which
    the previous test covers."""
    d = tmp_path / "d"
    d.mkdir()
    (d / "sub").mkdir()
    os.symlink(tmp_path, d / "lnk")
    try:
        dfd = fr._open_abs_dir(d)
    except OSError as exc:
        if exc.errno in fr._UNSUPPORTED_ERRNOS:
            pytest.skip(f"openat2 unavailable here ({errno.errorcode.get(exc.errno)})")
        raise
    try:
        cfd = fr._open_child_dir(dfd, "sub")
        try:
            st = os.fstat(cfd)
            lst = fr._stat_at(dfd, "sub")
            assert (st.st_dev, st.st_ino) == (lst.st_dev, lst.st_ino)
        finally:
            os.close(cfd)
        for name, errs in (("lnk", (errno.ENOTDIR, errno.ELOOP)), ("..", (errno.EXDEV,)), ("/", (errno.EXDEV,))):
            with pytest.raises(OSError) as exc:
                fr._open_child_dir(dfd, name)
            assert exc.value.errno in errs, name
    finally:
        os.close(dfd)
    # a real mount point on this machine (anything but the root's own mount) is refused with EXDEV
    root_dev = os.lstat("/").st_dev
    mount = next((p for p in ("/proc", "/sys", "/dev", "/run", "/tmp", "/home")
                  if os.path.isdir(p) and os.lstat(p).st_dev != root_dev), None)
    if mount is None:
        pytest.skip("no mount point with a different device found to probe")
    rfd = fr._open_abs_dir(Path("/"))
    try:
        with pytest.raises(OSError) as exc:
            fr._open_child_dir(rfd, mount.lstrip("/"))
        assert exc.value.errno == errno.EXDEV
    finally:
        os.close(rfd)
