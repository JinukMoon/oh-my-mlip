#!/usr/bin/env bash
# sweep_watchdog.sh — keep the local MLIP env sweep alive across WSL restarts.
#
# Idempotent + safe to run every minute from cron (and @reboot):
#   - flock prevents double-launch races.
#   - relaunches sweep_local.py ONLY if it is not already running AND the sweep
#     is not finished (no .sweep/DONE sentinel).
#   - the sweep is resumable (skips already-passed envs via results.jsonl), so a
#     relaunch after a WSL/Windows reboot just continues where it left off.
#
# Install (cron, matching the user's existing @reboot + per-minute pattern):
#   @reboot   /home/jumoon/01_2026/17_MLIP_HUB/oh-my-mlip/scripts/sweep_watchdog.sh
#   * * * * * /home/jumoon/01_2026/17_MLIP_HUB/oh-my-mlip/scripts/sweep_watchdog.sh
#
# Stop it permanently:  touch .sweep/DONE   (and/or remove the cron lines)

set -euo pipefail

HOME_DIR="/home/jumoon/01_2026/17_MLIP_HUB/oh-my-mlip"
SWEEP_DIR="$HOME_DIR/.sweep"
LOCK="$SWEEP_DIR/watchdog.lock"
WLOG="$SWEEP_DIR/watchdog.log"

mkdir -p "$SWEEP_DIR"
exec 9>"$LOCK"
flock -n 9 || exit 0   # another watchdog tick is already running

ts() { date -Iseconds; }

# Stop conditions: explicit DONE sentinel, OR every config env already has a
# terminal row in results.jsonl (so finished sweeps with some failures do NOT
# get relaunched every minute — only an interrupted, incomplete sweep resumes).
if [ -f "$SWEEP_DIR/DONE" ]; then
  exit 0
fi
if python3 - "$HOME_DIR/scripts/sweep_config.json" "$SWEEP_DIR/results.jsonl" <<'PY'
import json, sys
cfg, res = sys.argv[1], sys.argv[2]
try:
    n_cfg = len(json.load(open(cfg))["order"])
except Exception:
    sys.exit(1)  # can't read config -> not "done", allow relaunch
seen = set()
try:
    for line in open(res):
        line = line.strip()
        if not line:
            continue
        try:
            seen.add(json.loads(line).get("env"))
        except Exception:
            pass
except FileNotFoundError:
    sys.exit(1)
sys.exit(0 if len(seen - {None}) >= n_cfg else 1)
PY
then
  exit 0   # all envs have terminal rows -> sweep complete, do not relaunch
fi

# Already running? (match the script path, exclude this watchdog)
if pgrep -f "sweep_local.py" >/dev/null 2>&1; then
  exit 0
fi

# Not running and not done -> (re)launch, resuming from results.jsonl.
cd "$HOME_DIR"
# The driver is confirmed NOT running, so any lingering install.sh / conda-env-create
# / pip-into-envs processes are ORPHANS from a driver that died mid-build. Reap them
# first so a relaunch never stacks two concurrent builds on the same env prefix.
ORPHANS=$(ps -eo pid,cmd | grep -F -e "$HOME_DIR/install.sh" -e "conda env create --prefix $HOME_DIR/envs" -e "$HOME_DIR/envs/" | grep -v grep | awk '{print $1}' || true)
if [ -n "$ORPHANS" ]; then
  echo "[$(ts)] watchdog: reaping orphaned build procs: $(echo $ORPHANS|tr '\n' ' ')" >>"$WLOG"
  kill -9 $ORPHANS 2>/dev/null || true
fi
# cron does NOT source the user's profile, so conda may be off PATH. install.sh
# needs conda -> prepend the miniconda bin dir so the whole subprocess tree finds it.
for CB in /home/jumoon/miniconda3/bin /home/jumoon/miniconda3/condabin; do
  [ -d "$CB" ] && case ":$PATH:" in *":$CB:"*) ;; *) PATH="$CB:$PATH";; esac
done
export PATH
echo "[$(ts)] watchdog: sweep not running -> relaunching (resume); conda=$(command -v conda || echo MISSING)" >>"$WLOG"
# shellcheck disable=SC1091
source env.sh >>"$WLOG" 2>&1 || true
# Use an explicit python3: `python` is not on PATH here and env.sh does not
# conda-activate, so a bare `python` launch fails with "No such file or directory".
PYBIN="$(command -v python3 || echo /usr/bin/python3)"
nohup "$PYBIN" -u scripts/sweep_local.py >>"$SWEEP_DIR/sweep.log" 2>&1 &
disown || true
echo "[$(ts)] watchdog: relaunched pid=$!" >>"$WLOG"
