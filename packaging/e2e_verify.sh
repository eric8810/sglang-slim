#!/usr/bin/env bash
# End-to-end delivery verification for the sglang-lite bundle.
#
# Proves the full delivery pipeline in one shot:
#   build-machine warmup cache -> bundled in the tarball -> unpacked on a
#   different path, different HOME, read-only cache, no CUDA toolkit ->
#   server healthy, zero runtime compile, generation works.
#
# Usage:
#   SGLANG_SMOKE_MODEL=/path/to/model bash packaging/e2e_verify.sh [tarball]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARBALL="${1:-$(ls "$REPO_ROOT"/dist/sglang-lite-*.tar.zst 2>/dev/null | head -1 || true)}"
MODEL="${SGLANG_SMOKE_MODEL:?set SGLANG_SMOKE_MODEL=/path/to/model}"
PORT="${SGLANG_SMOKE_PORT:-3972}"
E2E_ROOT="${SGLANG_E2E_ROOT:-/tmp/sglang-lite-e2e}"
FAKE_HOME="$E2E_ROOT/fakehome"

test -f "$TARBALL" || { echo "tarball not found: $TARBALL (run make_bundle.sh)"; exit 1; }

echo "[e2e] tarball: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
chmod -R u+w "$E2E_ROOT" 2>/dev/null || true   # may be read-only from a previous run
rm -rf "$E2E_ROOT"; mkdir -p "$FAKE_HOME"
tar --zstd -xf "$TARBALL" -C "$E2E_ROOT"
BUNDLE="$E2E_ROOT/sglang-lite"
test -x "$BUNDLE/bin/sglang-serve" || { echo "launcher missing"; exit 1; }

echo "[e2e] 1. read-only cache (simulates a sealed delivery)"
chmod -R a-w "$BUNDLE/cache"

echo "[e2e] 2. no-toolkit env (scrub nvcc, unset CUDA_HOME)"
PATH_CLEAN="$(echo "$PATH" | tr ':' '\n' | grep -v '/cuda' | paste -sd:)"
export SGLANG_CRASH_ON_JIT_COMPILE=1   # any cache miss = hard crash = test failure

LOG="$E2E_ROOT/server.log"
env -i HOME="$FAKE_HOME" PATH="$PATH_CLEAN" \
  SGLANG_CRASH_ON_JIT_COMPILE=1 \
  "$BUNDLE/bin/sglang-serve" \
    --model-path "$MODEL" \
    --mem-fraction-static 0.85 \
    --cuda-graph-max-bs-decode 8 --cuda-graph-max-bs-prefill 4 \
    --max-running-requests 8 \
    ${SGLANG_SMOKE_EXTRA_ARGS:-} \
    --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "[e2e] 3. waiting for /health (log: $LOG)"
ok=0
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "[e2e] server DIED:"; tail -25 "$LOG"; exit 2; }
  sleep 5
done
[[ $ok -eq 1 ]] || { echo "[e2e] health TIMEOUT"; tail -25 "$LOG"; exit 3; }
echo "[e2e] healthy after ~$((i*5))s (path=$BUNDLE, HOME=$FAKE_HOME, cache read-only, no toolkit)"

resp=$(curl -sf "http://127.0.0.1:$PORT/generate" -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 32, "temperature": 0}}')
echo "[e2e] response: ${resp:0:160}"
echo "$resp" | grep -q '"text"' && echo "[e2e] PASS: E2E delivery pipeline verified" || { echo "[e2e] FAIL"; exit 4; }
