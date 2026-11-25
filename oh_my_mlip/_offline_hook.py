"""Fail-fast network for a node without internet (OMM_OFFLINE=1).

install.sh copies this file into every env it builds as ``_omm_offline.py`` and
adds ``omm_offline.pth`` (one line: ``import _omm_offline``), so it runs at the
start of every interpreter of that env. It does nothing unless OMM_OFFLINE=1.

Why: HPC compute nodes often drop outbound packets without answering. A loader
that checks a URL before using its cache (ORB's cached_path ETag request) then
waits until the job's time limit instead of using the weights the install
already downloaded. With OMM_OFFLINE=1 every name lookup and connection to a
non-local address fails at once with OSError, which such loaders treat as
"offline" and answer from their cache; a weight that is not cached fails at
once with a clear error instead of hanging.

env.sh and oh_my_mlip (setup_verify, run(), Worker) set OMM_OFFLINE=1 when a
3-second connection test to the internet fails; OMM_OFFLINE=0 turns it off.
Standard library only; it must import on every Python the envs use.
"""
import errno
import ipaddress
import os
import socket

_MSG = ("oh-my-mlip offline mode (OMM_OFFLINE=1): this node has no internet, so network "
        "access fails at once; weights must already be downloaded (install.sh does that)")


def _local(host):
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    host = str(host)
    if host in ("", "localhost") or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host.split("%")[0])
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _install():
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def getaddrinfo(host, *args, **kwargs):
        if not _local(host):
            raise socket.gaierror(socket.EAI_NONAME, _MSG)
        return real_getaddrinfo(host, *args, **kwargs)

    def _target(address):
        return address[0] if isinstance(address, tuple) and address else None

    def connect(self, address):
        if not _local(_target(address)):
            raise OSError(errno.ENETUNREACH, _MSG)
        return real_connect(self, address)

    def connect_ex(self, address):
        if not _local(_target(address)):
            return errno.ENETUNREACH
        return real_connect_ex(self, address)

    socket.getaddrinfo = getaddrinfo
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(key, "1")


if os.environ.get("OMM_OFFLINE") == "1":
    _install()
