#!/usr/bin/env bash
# Keep a venv-env server running for interactive probing (not auto-killed).
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${SGLANG_SLIM_BUILD:-/media/eric8810/fast-deliver/sglang-slim-build}"
MODEL="${1:?usage: keep_server.sh /path/to/model [port]}"
PORT="${2:-3975}"
export PYTHONPATH="$REPO_ROOT/dist/sglang-slim"
export PATH="$BUILD_DIR/venv/bin:$PATH"
exec "$BUILD_DIR/venv/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" --mem-fraction-static 0.85 \
  --cuda-graph-max-bs-decode 8 --cuda-graph-max-bs-prefill 4 \
  --max-running-requests 8 --host 127.0.0.1 --port "$PORT" --log-level info
