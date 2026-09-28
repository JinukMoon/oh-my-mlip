"""Where the hub is: the one resolver every script and the package use.

The hub is the clone that holds `models.json`, `install.sh`, `envs/` and the
machine-local files (`models.local.json`, `env_map.local.json`). Order:

1. `$OH_MY_MLIP_HOME`, when set (used as given, even if it does not exist yet,
   so the error names it rather than silently falling back);
2. otherwise the clone this code lives in.

A copy of the repository inside an agent's plugin cache (the Claude Code
plugin ships the whole repository) is never a hub: the host replaces it on
every plugin update, and envs built there are lost. Scripts that write call
`plugin_copy_refusal()` and stop with its message. The default hub for plugin
users is `~/.oh-my-mlip` (AGENTS.md §9.0).

Stdlib only: `install.sh`, `env.sh` and every script share this order.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HUB = "~/.oh-my-mlip"

# Disk, one source for every script and AGENTS.md. An env takes 5 to 15 GB,
# and a build also fills pip/conda download caches while it runs.
DISK_FLOOR_GB = 30.0   # free space install.sh, the sweep and the guardrail require to start one build
ENV_BUDGET_GB = 15.0   # setup_survey's planning estimate per env to build

# Path segment pairs that mark an agent host's plugin cache.
_PLUGIN_CACHE_MARKERS = ((".claude", "plugins"), (".codex", "plugins"))

_CODE_ROOT = Path(__file__).resolve().parents[1]


def resolve_home() -> Path:
    """The hub root: `$OH_MY_MLIP_HOME` if set, else this code's own clone."""
    env = os.environ.get("OH_MY_MLIP_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return _CODE_ROOT


def in_plugin_cache(path: Path | str) -> bool:
    """True when `path` lies inside an agent host's plugin cache."""
    parts = Path(path).expanduser().resolve().parts
    return any(parts[i:i + 2] == pair for pair in _PLUGIN_CACHE_MARKERS for i in range(len(parts) - 1))


def plugin_copy_refusal(home: Path | str | None = None) -> str | None:
    """The message to stop with when `home` is a plugin-cache copy, else None."""
    home = Path(home) if home is not None else resolve_home()
    if not in_plugin_cache(home):
        return None
    return (f"{home} is the copy of oh-my-mlip inside the agent's plugin cache, not a hub: "
            f"it is replaced on every plugin update, and envs built there are lost. "
            f"Use your own clone: export OH_MY_MLIP_HOME=<clone> (default {DEFAULT_HUB}; "
            f"clone https://github.com/JinukMoon/oh-my-mlip.git there if it does not exist), "
            f"then run the scripts from that clone. See AGENTS.md §9.0.")
