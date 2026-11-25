"""Is there internet here? If not, switch the model processes to offline mode.

A node that silently drops outbound packets (common for HPC compute nodes) makes
a loader that checks a URL before using its cache wait until the job's time
limit. ``offline_env(env)`` tests the network once per process (a TCP connection
to huggingface.co:443 or pypi.org:443, ``PROBE_TIMEOUT_S`` each) and, when both
fail, returns the env with OMM_OFFLINE=1 and the Hugging Face offline switches.
In the model's interpreter, ``_offline_hook`` (installed into each env by
install.sh) then makes every non-local connection fail at once, so loaders fall
back to the weights install.sh already downloaded.

OMM_OFFLINE set by the user (0 or 1) is never overridden.
"""
from __future__ import annotations

import os
import socket

PROBE_HOSTS = (("huggingface.co", 443), ("pypi.org", 443))
PROBE_TIMEOUT_S = 3.0
_probe_cache: bool | None = None


def internet_reachable(connect=socket.create_connection) -> bool:
    global _probe_cache
    if _probe_cache is None:
        _probe_cache = False
        for host, port in PROBE_HOSTS:
            try:
                connect((host, port), timeout=PROBE_TIMEOUT_S).close()
            except OSError:
                continue
            _probe_cache = True
            break
    return _probe_cache


def offline_env(env: dict, connect=socket.create_connection) -> dict:
    """``env`` with offline mode switched on when this node has no internet."""
    if env.get("OMM_OFFLINE") in ("0", "1"):
        return env
    if internet_reachable(connect):
        return env
    out = dict(env)
    out["OMM_OFFLINE"] = "1"
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        out.setdefault(key, "1")
    return out
