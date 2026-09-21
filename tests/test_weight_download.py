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


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    # the retry backoff lives in oh_my_mlip/_download.py; the scripts' helper
    # loads its own copy of that file, so both need the sleep removed
    sys.path.insert(0, str(REPO_ROOT))
    from oh_my_mlip import _download

    monkeypatch.setattr(_download.time, "sleep", lambda s: None)
    monkeypatch.setattr(wd._download.time, "sleep", lambda s: None)


def _cut_everything(handler):
    handler.send_response(200)
    handler.send_header("Content-Length", str(len(PAYLOAD)))
    handler.end_headers()
    handler.wfile.write(PAYLOAD[:1000])
    handler.close_connection = True


def test_whole_download_lands_verified(tmp_path, server):
    dest = tmp_path / "m" / "model.ckpt"
    wd.download(server.url, dest, size=len(PAYLOAD), sha256=SHA)
    assert dest.read_bytes() == PAYLOAD
    assert not (dest.parent / ".model.ckpt.download").exists()


def test_cut_off_download_is_resumed_in_the_same_call(tmp_path, server):
    # long connections get dropped (seen after ~10 minutes on Hugging Face);
    # the transfer continues with a Range request instead of failing the run
    dest = tmp_path / "model.ckpt"
    server.cut_at = 300_000
    wd.download(server.url, dest, size=len(PAYLOAD), sha256=SHA)
    assert dest.read_bytes() == PAYLOAD
    assert server.ranges == [None, "bytes=300000-"]


def test_a_partial_file_from_an_earlier_run_is_resumed(tmp_path, server):
    dest = tmp_path / "model.ckpt"
    (tmp_path / ".model.ckpt.download").write_bytes(PAYLOAD[:300_000])
    wd.download(server.url, dest, size=len(PAYLOAD), sha256=SHA)
    assert dest.read_bytes() == PAYLOAD
    assert server.ranges == ["bytes=300000-"]


def test_cut_off_without_recorded_size_uses_content_length(tmp_path, server):
    dest = tmp_path / "model.ckpt"
    server.cut_at = 1000
    wd.download(server.url, dest)
    assert dest.read_bytes() == PAYLOAD
    assert server.ranges == [None, "bytes=1000-"]


def test_when_every_attempt_fails_the_part_is_kept_and_nothing_is_installed(tmp_path, server, monkeypatch):
    monkeypatch.setattr(server.httpd.RequestHandlerClass, "do_GET", _cut_everything)
    dest = tmp_path / "model.ckpt"
    with pytest.raises(wd.DownloadError, match="next attempt resumes"):
        wd.download(server.url, dest, size=len(PAYLOAD), sha256=SHA)
    assert not dest.exists()
    assert (tmp_path / ".model.ckpt.download").stat().st_size > 0


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


def test_fetch_url_download_resumes_after_a_cut(tmp_path, server, capsys, monkeypatch):
    # the url path in oh_my_mlip/fetch.py (figshare-hosted weights) continues a
    # cut-off download with a Range request instead of starting over
    sys.path.insert(0, str(REPO_ROOT))
    from oh_my_mlip import fetch

    server.cut_at = 400_000
    local = fetch._download_to_temp(server.url, tmp_path)
    assert local.read_bytes() == PAYLOAD
    assert server.ranges == [None, "bytes=400000-"]
    assert capsys.readouterr().out == "", "progress must stay off stdout (MCP channel)"

    # and a partial file left by an earlier process is continued, not replaced
    part = tmp_path / ".ckpt.download"
    part.write_bytes(PAYLOAD[:123_456])
    local = fetch._download_to_temp(server.url, tmp_path)
    assert local.read_bytes() == PAYLOAD
    assert server.ranges[-1] == "bytes=123456-"

def test_md5_is_checked_when_given(tmp_path, server):
    dest = tmp_path / "pkg.zip"
    with pytest.raises(RuntimeError, match="md5"):
        wd.download(server.url, dest, md5="0" * 32)
    assert not dest.exists()
    wd.download(server.url, dest, md5=hashlib.md5(PAYLOAD).hexdigest())
    assert dest.read_bytes() == PAYLOAD


def test_nequip_compiles_from_a_package_it_downloaded_itself(tmp_path, server, monkeypatch):
    # nequip-compile's own download of nequip.net URIs cannot resume, and a slow
    # zenodo connection broke it partway; the package is fetched here instead
    mod = _load("prepare_nequip_weights")
    monkeypatch.delenv("OMM_NEQUIP_ZIP_DIR", raising=False)
    md5 = hashlib.md5(PAYLOAD).hexdigest()
    monkeypatch.setitem(mod.MODELS, "allegro", [
        ("nequip.net:x/Allegro-T:0.1", "Allegro-T", [], "Allegro-T.nequip.zip", md5, len(PAYLOAD), SHA)])
    monkeypatch.setattr(mod, "ZENODO", server.url + "?{name}")
    monkeypatch.setattr(mod, "MIRROR", "http://127.0.0.1:9/unused?{name}")
    monkeypatch.setattr(mod, "host_arch", lambda: "sm00")
    target_root = tmp_path / "models" / "allegro"
    (target_root).mkdir(parents=True)
    (target_root / "Allegro-T.nequip.zip").write_bytes(PAYLOAD[:500])  # an earlier cut-off try

    compiled = []

    def fake_run(cmd, env=None):
        compiled.append(cmd[1])
        Path(cmd[2]).write_bytes(b"pt2")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert mod.main(["--target-root", str(target_root)]) == 0
    assert compiled == [str(target_root / "Allegro-T.nequip.zip")]
    assert (target_root / "Allegro-T.nequip.zip").read_bytes() == PAYLOAD


def test_nequip_package_records_match_zenodo():
    mod = _load("prepare_nequip_weights")
    rows = {row[3]: (row[4], row[5], row[6]) for rows in mod.MODELS.values() for row in rows}
    # md5 and size from Zenodo record 18775904; sha256 from the mirror's LFS
    # metadata, computed over the same bytes
    assert rows == {
        "NequIP-OAM-XL-0.1.nequip.zip": ("3d2369c7238eb83a23141abdcb055a8f", 259627903,
                                         "99c3799b28026f1ecf66c413292038a27a0749d4d4d7cd0b3a642f2e68df9e9c"),
        "NequIP-OAM-L-0.1.nequip.zip": ("67144367c710a70a53a8e21acf331980", 78464590,
                                        "5d01a4fab228abb3cdb6ace0033f93993729956bca6a42234a2a8816825b9a0f"),
        "Allegro-OAM-L-0.1.nequip.zip": ("0db7f9b3c3a62e74d78b3fcf2973c462", 80738705,
                                         "3f0d3ca7bb136d4c2ee76278170fcbe756436f037e16c673a86e2c57d271c64e"),
    }


def test_fetch_download_gives_up_on_a_stalled_connection_and_resumes(tmp_path, server, monkeypatch):
    # a server that sends part of the body and then goes silent used to hang the
    # read forever: no timeout, no error, no chance to resume
    sys.path.insert(0, str(REPO_ROOT))
    from oh_my_mlip import fetch

    release = threading.Event()
    original = server.httpd.RequestHandlerClass.do_GET

    def stall_first(handler):
        if handler.headers.get("Range") is None and not release.is_set():
            release.set()
            handler.send_response(200)
            handler.send_header("Content-Length", str(len(PAYLOAD)))
            handler.end_headers()
            handler.wfile.write(PAYLOAD[:200_000])
            handler.wfile.flush()
            threading.Event().wait(3)  # silent, connection still open
            return
        original(handler)

    monkeypatch.setattr(server.httpd.RequestHandlerClass, "do_GET", stall_first)
    local = fetch._download_to_temp(server.url, tmp_path, timeout=0.5)
    assert local.read_bytes() == PAYLOAD
    assert server.ranges[-1] == "bytes=200000-"


def test_fetch_download_keeps_the_partial_file_when_every_attempt_fails(tmp_path, server, monkeypatch):
    sys.path.insert(0, str(REPO_ROOT))
    from oh_my_mlip import fetch

    monkeypatch.setattr(server.httpd.RequestHandlerClass, "do_GET", _cut_everything)
    with pytest.raises(fetch.FetchError, match="next attempt resumes"):
        fetch._download_to_temp(server.url, tmp_path, attempts=2)
    assert (tmp_path / ".ckpt.download").stat().st_size > 0


@pytest.mark.parametrize("code", [504, 503, 429])
def test_a_busy_server_is_retried(tmp_path, server, monkeypatch, code):
    # seen live: Zenodo answered 504 Gateway Time-out while under load
    original = server.httpd.RequestHandlerClass.do_GET
    calls = []

    def busy_once(handler):
        calls.append(1)
        if len(calls) == 1:
            handler.send_response(code)
            handler.send_header("Retry-After", "1")
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        original(handler)

    monkeypatch.setattr(server.httpd.RequestHandlerClass, "do_GET", busy_once)
    slept = []
    monkeypatch.setattr(wd._download.time, "sleep", slept.append)
    dest = tmp_path / "model.ckpt"
    wd.download(server.url, dest, size=len(PAYLOAD), sha256=SHA)
    assert dest.read_bytes() == PAYLOAD
    assert len(calls) == 2 and slept


def test_a_missing_file_is_not_retried(tmp_path, server, monkeypatch):
    calls = []

    def not_found(handler):
        calls.append(1)
        handler.send_response(404)
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    monkeypatch.setattr(server.httpd.RequestHandlerClass, "do_GET", not_found)
    from urllib.error import HTTPError
    with pytest.raises(HTTPError):
        wd.download(server.url, tmp_path / "model.ckpt")
    assert len(calls) == 1


def test_retry_after_is_capped():
    from urllib.error import HTTPError
    from email.message import Message

    headers = Message()
    headers["Retry-After"] = "86400"
    exc = HTTPError("u", 503, "busy", headers, None)
    assert wd._download._retry_after(exc) == 120



def _broken(code):
    def handler(h):
        h.send_response(code)
        h.send_header("Content-Length", "0")
        h.end_headers()
    return handler


def test_a_failing_official_host_hands_over_to_the_mirror(tmp_path, monkeypatch):
    official, mirror = _Server(), _Server()
    try:
        official.cut_at = 250_000  # delivers a quarter, then is down for good
        first = official.httpd.RequestHandlerClass.do_GET
        calls = []

        def cut_then_504(h):
            calls.append(1)
            (first if len(calls) == 1 else _broken(504))(h)

        monkeypatch.setattr(official.httpd.RequestHandlerClass, "do_GET", cut_then_504)
        dest = tmp_path / "pkg.zip"
        used = wd.download_first_available(
            [("Zenodo", official.url), ("mirror", mirror.url)], dest,
            size=len(PAYLOAD), md5=hashlib.md5(PAYLOAD).hexdigest(), label="T")
        assert used == "mirror"
        assert dest.read_bytes() == PAYLOAD
        assert mirror.ranges == ["bytes=250000-"], "the mirror resumes the official host's partial file"
    finally:
        official.close()
        mirror.close()


def test_a_mirror_serving_other_bytes_is_rejected(tmp_path, monkeypatch):
    official, mirror = _Server(), _Server()
    try:
        monkeypatch.setattr(official.httpd.RequestHandlerClass, "do_GET", _broken(404))
        dest = tmp_path / "pkg.zip"
        with pytest.raises(wd.DownloadError, match="md5"):
            wd.download_first_available(
                [("Zenodo", official.url), ("mirror", mirror.url)], dest,
                size=len(PAYLOAD), md5="0" * 32, label="T")
        assert not dest.exists()
    finally:
        official.close()
        mirror.close()


def test_a_host_below_the_minimum_rate_is_given_up_at_once(tmp_path, server):
    calls_before = len(server.ranges)
    with pytest.raises(wd._download.TooSlow):
        wd._download.download_resumable(server.url, tmp_path / ".x.download", label="T",
                                        min_rate=1e15, rate_window=0)
    assert len(server.ranges) - calls_before == 1, "a slow host is not retried"


def test_nequip_packages_come_from_the_cited_record_then_the_mirror():
    mod = _load("prepare_nequip_weights")
    assert "records/18775904/" in mod.ZENODO
    assert mod.MIRROR.startswith("https://huggingface.co/JinukMoon/oh-my-mlip-mirror-nequip/resolve/main/")
