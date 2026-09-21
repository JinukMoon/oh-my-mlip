"""scripts/_weight_download.py: a checkpoint only reaches its final path whole.

The failure this guards: a connection that closes early ends a urllib read loop
quietly, and the prepare scripts used to reuse any non-empty file at the final
path, so a cut-off PET checkpoint failed at export on every rerun.
"""
from __future__ import annotations

import hashlib
import http.server
import importlib.util
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import _weight_download as wd  # noqa: E402

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


class _Server:
    """Serves PAYLOAD; `cut_at` makes the next plain GET declare the full length
    but close after that many bytes. Range requests are honoured with 206."""

    def __init__(self):
        self.cut_at = None
        self.ranges = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                rng = self.headers.get("Range")
                outer.ranges.append(rng)
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    body = PAYLOAD[start:]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
                else:
                    body = PAYLOAD
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if outer.cut_at is not None and not rng:
                    self.wfile.write(body[: outer.cut_at])
                    outer.cut_at = None
                    self.close_connection = True
                    return
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/ckpt"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    s = _Server()
    yield s
    s.close()


SHA = hashlib.sha256(PAYLOAD).hexdigest()


def test_whole_download_lands_verified(tmp_path, server):
    dest = tmp_path / "m" / "model.ckpt"
    wd.download(server.url, dest, size=len(PAYLOAD), sha256=SHA)
    assert dest.read_bytes() == PAYLOAD
    assert not (dest.parent / ".model.ckpt.download").exists()


def test_cut_off_download_is_kept_aside_then_resumed(tmp_path, server):
    dest = tmp_path / "model.ckpt"
    server.cut_at = 300_000
    with pytest.raises(Exception):
        wd.download(server.url, dest, size=len(PAYLOAD), sha256=SHA)
    assert not dest.exists(), "a cut-off download must never reach the final path"
    part = tmp_path / ".model.ckpt.download"
    assert 0 < part.stat().st_size < len(PAYLOAD)

    wd.download(server.url, dest, size=len(PAYLOAD), sha256=SHA)
    assert dest.read_bytes() == PAYLOAD
    assert server.ranges[-1] == "bytes=300000-"


def test_cut_off_without_recorded_size_uses_content_length(tmp_path, server):
    dest = tmp_path / "model.ckpt"
    server.cut_at = 1000
    with pytest.raises(Exception):
        wd.download(server.url, dest)
    assert not dest.exists()


def test_hash_mismatch_discards_the_file(tmp_path, server):
    dest = tmp_path / "model.ckpt"
    with pytest.raises(RuntimeError, match="sha256"):
        wd.download(server.url, dest, size=len(PAYLOAD), sha256="0" * 64)
    assert not dest.exists()
    assert not (tmp_path / ".model.ckpt.download").exists()


def test_is_complete_rejects_a_stub(tmp_path):
    stub = tmp_path / "model.ckpt"
    stub.write_bytes(b"x" * 10)
    assert not wd.is_complete(stub, 20)
    assert wd.is_complete(stub, 10)
    assert wd.is_complete(stub, None)
    assert not wd.is_complete(tmp_path / "absent", None)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("script,ckpt_name,out_name", [
    ("prepare_pet_weights", "pet-oam-xl-v1.0.0.ckpt", "pet-oam-xl-v1.0.0.pt"),
    ("prepare_deepmd_weights", "dpa-3.1-3m-ft.pth", "frozen-omat24.pth"),
])
def test_prepare_script_replaces_a_truncated_checkpoint(tmp_path, server, monkeypatch, script, ckpt_name, out_name):
    mod = _load(script)
    monkeypatch.setattr(mod, "CHECKPOINT_URL", server.url)
    monkeypatch.setattr(mod, "CKPT_SIZE", len(PAYLOAD))
    monkeypatch.setattr(mod, "CKPT_SHA256", SHA)
    stub = tmp_path / ckpt_name
    stub.write_bytes(PAYLOAD[:1000])  # what an interrupted older version left behind

    seen = {}

    def fake_run(cmd, cwd=None):
        seen["ckpt"] = Path(cmd[cmd.index("-c") + 1] if "-c" in cmd else cmd[2]).read_bytes()
        (tmp_path / out_name).write_bytes(b"exported")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [script, "--target-root", str(tmp_path)])
    assert mod.main() == 0
    assert seen["ckpt"] == PAYLOAD


def test_pinned_checkpoints_match_hugging_face_lfs_records():
    pet = _load("prepare_pet_weights")
    dp = _load("prepare_deepmd_weights")
    assert (pet.CKPT_SIZE, pet.CKPT_SHA256) == (
        2920687712, "c3a67cd019969dfd4dcabe9574682fe035f861d3f1c10190989b36c983699409")
    assert (dp.CKPT_SIZE, dp.CKPT_SHA256) == (
        47176032, "86dd3a804d78ca5d203ebf98747e8f16dff9713ba8950097ceb760b161e19907")


def test_fetch_url_download_resumes_after_a_cut(tmp_path, server, capsys):
    # the url path in oh_my_mlip/fetch.py (figshare-hosted weights) keeps its
    # partial file in the staging dir and continues it on the next attempt
    sys.path.insert(0, str(REPO_ROOT))
    from oh_my_mlip import fetch

    server.cut_at = 400_000
    with pytest.raises(Exception):
        fetch._download_to_temp(server.url, tmp_path)
    part = tmp_path / ".ckpt.download"
    assert 0 < part.stat().st_size < len(PAYLOAD)

    local = fetch._download_to_temp(server.url, tmp_path)
    assert local.read_bytes() == PAYLOAD
    assert server.ranges[-1] == "bytes=400000-"
    assert capsys.readouterr().out == "", "progress must stay off stdout (MCP channel)"
