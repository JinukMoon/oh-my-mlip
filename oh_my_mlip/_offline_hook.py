"""Fail-fast network for a node without internet (OMM_OFFLINE=1).

install.sh copies this file into every env it builds as ``_omm_offline.py`` and
adds ``omm_offline.pth`` (one line: ``import _omm_offline``), so it runs at the
start of every interpreter of that env. It does nothing unless OMM_OFFLINE=1.

Why: HPC compute nodes often drop outbound packets without answering. A loader
that checks a URL before using its cache (ORB's cached_path ETag request) then
waits until the job's time limit instead of using the weights the install
already downloaded. With OMM_OFFLINE=1 every name lookup and connection to a
public internet address fails at once with OSError, which such loaders treat as
"offline" and answer from their cache; a weight that is not cached fails at
once with a clear error instead of hanging. Loopback, private (RFC 1918, ULA) and
link-local destinations stay reachable, also by name (DDP rendezvous with
MASTER_ADDR=<another node>, licence servers, site mirrors), and so do the proxy
hosts named in http(s)_proxy / all_proxy.

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


def _private_ip(text):
    try:
        ip = ipaddress.ip_address(str(text).split("%")[0])
    except ValueError:
        return None
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _local_name(host):
    """True for a destination that is never the internet without a lookup."""
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    host = str(host)
    if host in ("", "localhost") or host.endswith(".localhost"):
        return True
    return _private_ip(host) is True


def _proxy_hosts(env):
    """Host names of the proxies in the environment: always reachable."""
    from urllib.parse import urlsplit

    hosts = set()
    for key in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        value = env.get(key, "").strip()
        if not value:
            continue
        if "://" not in value:
            value = "http://" + value
        try:
            name = urlsplit(value).hostname
        except ValueError:
            name = None
        if name:
            hosts.add(name.lower())
    return hosts


def _install():
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    proxies = _proxy_hosts(os.environ)
    allowed = set()          # addresses that a proxy name resolved to

    def getaddrinfo(host, *args, **kwargs):
        if _local_name(host):
            return real_getaddrinfo(host, *args, **kwargs)
        name = host.decode("ascii", "ignore") if isinstance(host, bytes) else str(host)
        if _private_ip(name) is False:        # a public address literal
            raise socket.gaierror(socket.EAI_NONAME, _MSG)
        infos = real_getaddrinfo(host, *args, **kwargs)
        if name.lower() in proxies:
            allowed.update(info[4][0] for info in infos)
            return infos
        # cluster-internal names (another compute node as MASTER_ADDR, a licence
        # server, a site mirror) resolve to private addresses: keep only those
        local = [info for info in infos if _private_ip(info[4][0])]
        if not local:
            raise socket.gaierror(socket.EAI_NONAME, _MSG)
        return local

    def _blocked(address):
        if not (isinstance(address, tuple) and address):
            return False                      # AF_UNIX and the like
        host = address[0]
        return not (_local_name(host) or host in allowed)

    def connect(self, address):
        if _blocked(address):
            raise OSError(errno.ENETUNREACH, _MSG)
        return real_connect(self, address)

    def connect_ex(self, address):
        if _blocked(address):
            return errno.ENETUNREACH
        return real_connect_ex(self, address)

    socket.getaddrinfo = getaddrinfo
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(key, "1")


if os.environ.get("OMM_OFFLINE") == "1":
    _install()
