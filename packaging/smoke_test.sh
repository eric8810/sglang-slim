#!/usr/bin/env bash
# Smoke test for the slim sglang tree (layer 0/2 pruning) on a single GPU.
#
# Stage 1 (this script, first run): start server on Qwen3-8B with fp8 online
#   quantization, issue one /generate request, verify output. This run also
#   acts as JIT warmup (populates ~/.cache/sglang, DG_JIT, triton caches).
# Stage 2 (second run with SGLANG_CRASH_ON_JIT_COMPILE=1 in env): proves the
#   warm cache is hit and no runtime compiler is needed (Go/No-Go for the
#   no-toolkit delivery).
#
# Usage:
#   bash packaging/smoke_test.sh [--crash-on-jit]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${SGLANG_SLIM_BUILD:-$REPO_ROOT/build-slim}"
VENV="$BUILD_DIR/venv"
SLIM_TREE="${SGLANG_SLIM_TREE:-$REPO_ROOT/dist/sglang-slim}"
MODEL="${SGLANG_SMOKE_MODEL:?set SGLANG_SMOKE_MODEL=/path/to/model (e.g. a local Qwen3 checkpoint)}"
PORT="${SGLANG_SMOKE_PORT:-3971}"

if [[ "${1:-}" == "--crash-on-jit" ]]; then
  export SGLANG_CRASH_ON_JIT_COMPILE=1
  echo "[smoke] SGLANG_CRASH_ON_JIT_COMPILE=1 (cache-hit verification mode)"
fi

if [[ "${1:-}" == "--no-toolkit" ]]; then
  # Simulate a toolkit-free target machine: scrub nvcc/CUDA_HOME from the
  # environment. Combined with --crash-on-jit this is the strict no-toolkit
  # Go/No-Go (uninstall the nvidia-cuda-nvcc wheel first, see README).
  export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '/cuda' | paste -sd:)"
  unset CUDA_HOME CUDA_PATH
  export SGLANG_CRASH_ON_JIT_COMPILE=1
  if command -v nvcc >/dev/null 2>&1; then
    echo "[smoke] WARNING: nvcc still resolvable via $(command -v nvcc)" >&2
  else
    echo "[smoke] no-toolkit mode: nvcc unreachable, CUDA_HOME unset"
  fi
fi

test -d "$SLIM_TREE/sglang" || { echo "slim tree missing: $SLIM_TREE (run build_slim.py first)" >&2; exit 1; }
test -d "$MODEL" || { echo "model missing: $MODEL" >&2; exit 1; }
test -x "$VENV/bin/python" || { echo "venv python missing: $VENV (run install_deps.py first)" >&2; exit 1; }

export PYTHONPATH="$SLIM_TREE"
export PATH="$VENV/bin:$PATH"   # JIT toolchain (ninja etc.) lives in the venv
PY="$VENV/bin/python"

echo "[smoke] python: $($PY --version)  model: $MODEL"

# --- start server in background ---
LOG="$BUILD_DIR/smoke-server.log"
$PY -m sglang.launch_server \
  --model-path "$MODEL" \
  --mem-fraction-static 0.85 \
  --cuda-graph-max-bs-decode 8 \
  --cuda-graph-max-bs-prefill 4 \
  --max-running-requests 8 \
  ${SGLANG_SMOKE_EXTRA_ARGS:-} \
  --host 127.0.0.1 --port "$PORT" \
  --log-level info >"$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "[smoke] server pid=$SERVER_PID, waiting for /health (log: $LOG)"

# --- wait for health (up to 15 min: first run includes JIT compile) ---
ok=0
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[smoke] server DIED, last log lines:" >&2; tail -40 "$LOG" >&2; exit 2
  fi
  sleep 5
done
if [[ $ok -ne 1 ]]; then echo "[smoke] health check TIMEOUT" >&2; tail -40 "$LOG" >&2; exit 3; fi
echo "[smoke] server healthy after ~$((i*5))s"

# --- one generation request ---
resp=$(curl -sf "http://127.0.0.1:$PORT/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 32, "temperature": 0}}')
echo "[smoke] response: $resp"

echo "$resp" | grep -q '"text"' && echo "[smoke] PASS: generation returned" || { echo "[smoke] FAIL: no text in response" >&2; exit 4; }

kill $SERVER_PID 2>/dev/null || true
echo "[smoke] done. JIT caches populated under: ~/.cache/sglang (and friends). Next: re-run with --crash-on-jit"
