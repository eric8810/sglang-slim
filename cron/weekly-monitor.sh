#!/usr/bin/env bash
# sglang-slim weekly upstream monitor & update (cron entry point).
#
# Runs `dim exec` with the dimagent-oauth provider (glm-5.3) to have an agent
# inspect upstream sglang changes and run the update pipeline per
# packaging/UPSTREAM_SYNC.md. Logs go to ~/.local/state/sglang-slim-cron/.
#
# Schedule (installed via crontab): every Monday 09:00 Asia/Shanghai — after
# upstream's Saturday-UTC biweekly release.
set -euo pipefail

REPO=/media/eric8810/fast-deliver/code/sglang-slim
SGLANG_CHECKOUT=/media/eric8810/fast-deliver/code/sglang
LOGDIR="$HOME/.local/state/sglang-slim-cron"
LOG="$LOGDIR/$(date +%Y-%m-%d_%H%M).log"
LOCK=/tmp/sglang-slim-weekly.lock

mkdir -p "$LOGDIR"

# --- guard: skip if a previous run is still going (agent pipelines take ~30min) ---
exec 9>"$LOCK"
flock -n 9 || { echo "$(date -Is) another run is active, skipping" >> "$LOGDIR/last.log"; exit 0; }

# --- cheap pre-check: is gh usable? (the agent needs it) ---
if ! gh api repos/sgl-project/sglang/releases/latest --jq .tag_name >/dev/null 2>&1; then
  echo "$(date -Is) gh not authenticated or network down — skipping agent run" >> "$LOG"
  echo "$(date -Is) skipped: gh unavailable" >> "$LOGDIR/last.log"
  exit 0
fi

{
  echo "=== sglang-slim weekly run $(date -Is) ==="
  echo "provider: dimcode-api-oauth/glm-5.3 via dim exec"
} >> "$LOG"

# --- run the agent in the sglang checkout workspace (its cwd = agent workspace) ---
cd "$SGLANG_CHECKOUT"
if timeout 7200 dim exec \
    --provider dimcode-api-oauth \
    --model glm-5.3 \
    "$(cat "$REPO/cron/prompt-weekly.md")" >> "$LOG" 2>&1; then
  echo "=== agent run OK $(date -Is) ===" >> "$LOG"
  echo "$(date -Is) OK" >> "$LOGDIR/last.log"
else
  rc=$?
  echo "=== agent run FAILED (rc=$rc) $(date -Is) ===" >> "$LOG"
  echo "$(date -Is) FAILED rc=$rc (timeout=124)" >> "$LOGDIR/last.log"
fi

# keep a rolling pointer + prune logs older than 12 weeks
ln -sfn "$LOG" "$LOGDIR/last-run.log"
find "$LOGDIR" -name '*.log' -mtime +84 -delete 2>/dev/null || true
