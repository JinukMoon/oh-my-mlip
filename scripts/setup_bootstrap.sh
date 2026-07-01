#!/usr/bin/env bash
# scripts/setup_bootstrap.sh -- oh-my-mlip bootstrap contract (Story G002)
#
# PURPOSE
#   Resolve (or clone) the oh-my-mlip repo root BEFORE env.sh / install.sh are
#   sourced.  Designed to be called by the Claude Code plugin skill
#   (skills/setup/SKILL.md) on a machine that may or may not already have a
#   clone.  Emits a parseable status line so the skill loop can branch without
#   screen-scraping.
#
# USAGE (direct)
#   bash scripts/setup_bootstrap.sh
#   eval "$(bash scripts/setup_bootstrap.sh | grep '^export ')"
#
# ENVIRONMENT VARIABLES
#   OMM_HOME         Override: use this directory as the clone root (highest prio).
#   OH_MY_MLIP_HOME  Fallback override (legacy / from env.sh).
#   OMM_REPO_URL     Git remote to clone from when no clone is found.
#                    Official repo: https://github.com/JinukMoon/oh-my-mlip.git
#                    Required only when no clone can be located; set it explicitly.
#   OMM_REF          Git ref (branch / tag / SHA) to check out.  Default: main
#
# OUTPUT (stdout)
#   export OH_MY_MLIP_HOME=<path>   -- eval-able, consumed by callers
#   BOOTSTRAP_STATUS=<value>        -- parseable status token (see below)
#   OH_MY_MLIP_HOME=<path>         -- repeated without 'export' for easy grep
#   (plus human-readable diagnostic lines prefixed with '#')
#
# STATUS TOKENS
#   ok              -- clone found (or successfully cloned) + conda/mamba present
#   need_repo_url   -- no clone found and OMM_REPO_URL is unset; user must act
#   need_conda      -- clone found but conda/mamba not on PATH; user must act
#
# CONSTRAINTS
#   - Bash only; no Python, no awk, no sed, no host-mutating installs.
#   - Only host mutation permitted: git clone into the resolved target directory.
#   - ASCII output only (no Unicode glyphs).
#   - Does NOT modify install.sh, env.sh, models.json, or provider.py.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0) Helpers
# ---------------------------------------------------------------------------

# Print a diagnostic comment line (goes to stdout so the caller sees it but
# grep '^export ' / grep '^BOOTSTRAP_STATUS' will skip it).
info() { printf '# [bootstrap] %s\n' "$*"; }

# Print an error line and set the status token, then exit non-zero.
die_with_status() {
  local status="$1"; shift
  printf '# [bootstrap] ERROR: %s\n' "$*"
  printf 'BOOTSTRAP_STATUS=%s\n' "$status"
  exit 1
}

# ---------------------------------------------------------------------------
# 1) Define "valid clone" predicate
#    A directory is a valid clone when it contains both models.json AND
#    install.sh (the two files that env.sh and install.sh rely on).
# ---------------------------------------------------------------------------
is_valid_clone() {
  local dir="$1"
  [ -d "$dir" ] && [ -f "$dir/models.json" ] && [ -f "$dir/install.sh" ]
}

# ---------------------------------------------------------------------------
# 2) Resolve clone root (precedence order)
#    Priority 1: $OMM_HOME         -- explicit override
#    Priority 2: $OH_MY_MLIP_HOME  -- legacy / already exported by env.sh
#    Priority 3: ~/.oh-my-mlip     -- conventional default location
#    Priority 4: this script's own repo root (running from inside a clone)
# ---------------------------------------------------------------------------
RESOLVED_HOME=""

info "Resolving oh-my-mlip clone root ..."

# Priority 1: $OMM_HOME
if [ -n "${OMM_HOME:-}" ]; then
  if is_valid_clone "$OMM_HOME"; then
    RESOLVED_HOME="$OMM_HOME"
    info "Found valid clone via OMM_HOME: $RESOLVED_HOME"
  else
    info "OMM_HOME set to '$OMM_HOME' but not a valid clone (missing models.json or install.sh) -- skipping"
  fi
fi

# Priority 2: $OH_MY_MLIP_HOME
if [ -z "$RESOLVED_HOME" ] && [ -n "${OH_MY_MLIP_HOME:-}" ]; then
  if is_valid_clone "$OH_MY_MLIP_HOME"; then
    RESOLVED_HOME="$OH_MY_MLIP_HOME"
    info "Found valid clone via OH_MY_MLIP_HOME: $RESOLVED_HOME"
  else
    info "OH_MY_MLIP_HOME set to '$OH_MY_MLIP_HOME' but not a valid clone -- skipping"
  fi
fi

# Priority 3: ~/.oh-my-mlip (conventional default)
if [ -z "$RESOLVED_HOME" ]; then
  _DEFAULT_DIR="$HOME/.oh-my-mlip"
  if is_valid_clone "$_DEFAULT_DIR"; then
    RESOLVED_HOME="$_DEFAULT_DIR"
    info "Found valid clone at default location: $RESOLVED_HOME"
  fi
  unset _DEFAULT_DIR
fi

# Priority 4: this script's own parent-of-parent (scripts/ -> repo root)
if [ -z "$RESOLVED_HOME" ]; then
  _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  _CANDIDATE="$(cd "$_SCRIPT_DIR/.." && pwd)"
  if is_valid_clone "$_CANDIDATE"; then
    RESOLVED_HOME="$_CANDIDATE"
    info "Found valid clone (script's own repo): $RESOLVED_HOME"
  fi
  unset _SCRIPT_DIR _CANDIDATE
fi

# ---------------------------------------------------------------------------
# 3) Clone if still not found
# ---------------------------------------------------------------------------
if [ -z "$RESOLVED_HOME" ]; then
  info "No valid clone found in any search location."

  # Guard: refuse to guess a URL -- the user must provide one explicitly.
  if [ -z "${OMM_REPO_URL:-}" ]; then
    printf '\n'
    printf '# [bootstrap] --------------------------------------------------------\n'
    printf '# [bootstrap] ACTION REQUIRED: no oh-my-mlip clone found and\n'
    printf '# [bootstrap] OMM_REPO_URL is not set.\n'
    printf '#\n'
    printf '# [bootstrap] Choose ONE of:\n'
    printf '#\n'
    printf '# [bootstrap]   A) Run this script from inside an existing clone:\n'
    printf '#       cd /path/to/oh-my-mlip && bash scripts/setup_bootstrap.sh\n'
    printf '#\n'
    printf '# [bootstrap]   B) Set OMM_REPO_URL and re-run:\n'
    printf '#       export OMM_REPO_URL=https://github.com/JinukMoon/oh-my-mlip.git\n'
    printf '#       bash scripts/setup_bootstrap.sh\n'
    printf '#\n'
    printf '# [bootstrap]   C) Set OMM_HOME to point at an existing clone:\n'
    printf '#       export OMM_HOME=/path/to/oh-my-mlip\n'
    printf '#       bash scripts/setup_bootstrap.sh\n'
    printf '#\n'
    printf '# [bootstrap] Official repo URL (for reference):\n'
    printf '#   https://github.com/JinukMoon/oh-my-mlip.git\n'
    printf '# [bootstrap] --------------------------------------------------------\n'
    printf 'BOOTSTRAP_STATUS=need_repo_url\n'
    printf 'OH_MY_MLIP_HOME=\n'
    exit 1
  fi

  # Clone into ~/.oh-my-mlip (the conventional default).
  _CLONE_TARGET="$HOME/.oh-my-mlip"
  _REF="${OMM_REF:-main}"
  info "Cloning $OMM_REPO_URL (ref: $_REF) -> $_CLONE_TARGET ..."

  if ! git clone --branch "$_REF" --depth 1 "$OMM_REPO_URL" "$_CLONE_TARGET"; then
    printf '# [bootstrap] ERROR: git clone failed.\n'
    printf '# [bootstrap] Check OMM_REPO_URL (%s) and network connectivity.\n' "$OMM_REPO_URL"
    printf 'BOOTSTRAP_STATUS=need_repo_url\n'
    printf 'OH_MY_MLIP_HOME=\n'
    unset _CLONE_TARGET _REF
    exit 1
  fi

  if ! is_valid_clone "$_CLONE_TARGET"; then
    printf '# [bootstrap] ERROR: clone completed but models.json or install.sh\n'
    printf '# [bootstrap] is missing in %s -- the repo may be incomplete.\n' "$_CLONE_TARGET"
    printf 'BOOTSTRAP_STATUS=need_repo_url\n'
    printf 'OH_MY_MLIP_HOME=\n'
    unset _CLONE_TARGET _REF
    exit 1
  fi

  RESOLVED_HOME="$_CLONE_TARGET"
  info "Clone successful: $RESOLVED_HOME"
  unset _CLONE_TARGET _REF
fi

# ---------------------------------------------------------------------------
# 4) Export OH_MY_MLIP_HOME (absolute path, NOT clone-relative derivation)
# ---------------------------------------------------------------------------
# Canonicalize to absolute path (handles any remaining symlinks / relative refs).
RESOLVED_HOME="$(cd "$RESOLVED_HOME" && pwd)"
export OH_MY_MLIP_HOME="$RESOLVED_HOME"

# Emit the eval-able export line so callers can do:
#   eval "$(bash scripts/setup_bootstrap.sh | grep '^export ')"
printf 'export OH_MY_MLIP_HOME=%s\n' "$OH_MY_MLIP_HOME"
info "OH_MY_MLIP_HOME resolved to: $OH_MY_MLIP_HOME"

# ---------------------------------------------------------------------------
# 5) conda / mamba detection
#    Print clear guidance and EXIT non-zero if neither is present.
#    Do NOT auto-install (host-mutating without user consent).
# ---------------------------------------------------------------------------
CONDA_BIN=""
if command -v mamba >/dev/null 2>&1; then
  CONDA_BIN="mamba"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="conda"
fi

if [ -z "$CONDA_BIN" ]; then
  printf '\n'
  printf '# [bootstrap] --------------------------------------------------------\n'
  printf '# [bootstrap] ACTION REQUIRED: neither conda nor mamba found on PATH.\n'
  printf '#\n'
  printf '# [bootstrap] oh-my-mlip uses conda environments for each MLIP.\n'
  printf '# [bootstrap] Install Miniconda (recommended) then re-run:\n'
  printf '#\n'
  printf '#   https://docs.conda.io/en/latest/miniconda.html\n'
  printf '#\n'
  printf '# [bootstrap] Quick install (Linux x86_64):\n'
  printf '#   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh\n'
  # shellcheck disable=SC2016  # literal help text shown to the user, not expanded here
  printf '#   bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3\n'
  # shellcheck disable=SC2016  # literal help text shown to the user, not expanded here
  printf '#   eval "$($HOME/miniconda3/bin/conda shell.bash hook)"\n'
  printf '#\n'
  printf '# [bootstrap] After installing, open a new shell (or eval the hook\n'
  printf '# [bootstrap] above), then re-run this script.\n'
  printf '# [bootstrap] --------------------------------------------------------\n'
  printf 'BOOTSTRAP_STATUS=need_conda\n'
  printf 'OH_MY_MLIP_HOME=%s\n' "$OH_MY_MLIP_HOME"
  exit 1
fi

info "Conda/mamba found: $CONDA_BIN ($(command -v "$CONDA_BIN"))"

# ---------------------------------------------------------------------------
# 6) Read-order reminder (per AGENTS.md section 0)
#    Print the three files agents must read before writing any code, so the
#    skill loop can surface them to the model without re-reading AGENTS.md.
# ---------------------------------------------------------------------------
printf '\n'
printf '# [bootstrap] AGENTS.md section 0 -- read these files FIRST (in order):\n'
printf '#   1. %s/models.json\n'        "$OH_MY_MLIP_HOME"
printf '#   2. %s/dist_manifest.json\n' "$OH_MY_MLIP_HOME"
printf '#   3. %s/AGENTS.md\n'          "$OH_MY_MLIP_HOME"
printf '\n'

# ---------------------------------------------------------------------------
# 7) Final parseable status block
# ---------------------------------------------------------------------------
printf 'BOOTSTRAP_STATUS=ok\n'
printf 'OH_MY_MLIP_HOME=%s\n' "$OH_MY_MLIP_HOME"
