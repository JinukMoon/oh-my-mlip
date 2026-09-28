"""clean-cache deletes only interrupted-download leftovers under the hub, and only with --yes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / "scripts" / "setup_guardrail.py"


def _setup(tmp_path):
    hub, home = tmp_path / "hub", tmp_path / "home"
    old = time.time() - 3600
    files = {
        "hub_part": hub / "models" / "mace" / "w.pt.part",
        "hub_tmp": hub / "models" / "grace" / "tmp.tar.gz",
        "hub_fresh": hub / "models" / "orb" / "w.ckpt.download",
        "hub_real": hub / "models" / "mace" / "w.pt",
        "te": home / ".cache" / "torch_extensions" / "py311" / "ext.so",
        "hf": home / ".cache" / "huggingface" / "hub" / "blobs" / "abc.incomplete",
        "hf_lock": home / ".cache" / "huggingface" / "hub" / ".locks" / "x.lock",
    }
    for key, p in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 10)
        if key != "hub_fresh":
            os.utime(p, (old, old))
    env = dict(os.environ, OH_MY_MLIP_HOME=str(hub), HOME=str(home))
    env.pop("HF_HOME", None)
    return files, env


def _clean(env, *args):
    r = subprocess.run([sys.executable, str(GUARD), "clean-cache", *args], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_without_yes_nothing_is_deleted(tmp_path):
    files, env = _setup(tmp_path)
    out = _clean(env)
    assert out["dry_run"] and all(p.exists() for p in files.values())
    assert {Path(i["path"]).name for i in out["hub_partials"]["items"]} == {"w.pt.part", "tmp.tar.gz"}


def test_yes_deletes_only_old_hub_leftovers(tmp_path):
    files, env = _setup(tmp_path)
    out = _clean(env, "--yes")
    assert not files["hub_part"].exists() and not files["hub_tmp"].exists()
    for keep in ("hub_fresh", "hub_real", "te", "hf", "hf_lock"):
        assert files[keep].exists(), keep                       # recent, real, or outside the hub
    outside = out["outside_hub_not_touched"]
    assert outside["torch_extensions"]["bytes"] == 10 and outside["hf_incomplete_files"] == [str(files["hf"])]


def test_symlinks_are_never_followed(tmp_path):
    files, env = _setup(tmp_path)
    ext = tmp_path / "elsewhere"
    ext.mkdir()
    victim = ext / "keep.part"
    victim.write_bytes(b"y")
    os.utime(victim, (time.time() - 3600,) * 2)
    (Path(env["OH_MY_MLIP_HOME"]) / "models" / "link").symlink_to(ext)
    (Path(env["OH_MY_MLIP_HOME"]) / "models" / "linked.part").symlink_to(victim)
    _clean(env, "--yes")
    assert victim.exists()
