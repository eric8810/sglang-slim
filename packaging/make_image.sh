#!/usr/bin/env bash
# Build the OCI image form of the bundle (the second delivery artifact).
#
# Design (per research conclusion): the image carries the SELF-CONTAINED
# bundle directory (not a pip install), so Python/package/JIT-cache versions
# are locked at build time — the image is an immutable release artifact.
# Base is ubuntu:24.04 to match the build machine's glibc (2.39).
#
# Target machine requires: docker + nvidia-container-toolkit (for --gpus).
# Usage:
#   SGLANG_SLIM_BUILD=<B> bash packaging/make_image.sh [--save]
# Then on a GPU host:
#   docker run --gpus all -v /path/to/models:/models -p 30000:30000 \
#     sglang-lite:<tag> --model-path /models/qwen3-4b --host 0.0.0.0
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${SGLANG_SLIM_BUNDLE:-$REPO_ROOT/dist/sglang-lite}"
BUILD_DIR="${SGLANG_SLIM_BUILD:-$REPO_ROOT/build-slim}"
IMAGE_TAG="${SGLANG_IMAGE_TAG:-sglang-lite:$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo dev)}"

test -x "$OUT_ROOT/bin/sglang-serve" || { echo "bundle missing: $OUT_ROOT (run make_bundle.sh)"; exit 1; }

CTX="$REPO_ROOT/dist/image-context"
rm -rf "$CTX"; mkdir -p "$CTX"
cp -a "$OUT_ROOT" "$CTX/sglang-lite"

cat > "$CTX/Dockerfile" <<'EOF'
FROM ubuntu:24.04
# glibc 2.39 matches the bundle's build environment.
# The NVIDIA driver (libcuda) is injected at runtime by
# nvidia-container-toolkit; everything else ships in the bundle.
COPY sglang-lite /opt/sglang-lite
# seed cache + launcher work with any writable HOME; model weights are
# expected on a volume mount (/models by convention).
ENV HOME=/root
EXPOSE 30000
WORKDIR /opt/sglang-lite
ENTRYPOINT ["/opt/sglang-lite/bin/sglang-serve"]
EOF

echo "[image] building $IMAGE_TAG from $OUT_ROOT ..."
docker build -t "$IMAGE_TAG" "$CTX"

echo "[image] built:"
docker image ls "$IMAGE_TAG" --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}'
echo "[image] container smoke (no GPU; checks the launcher tree):"
docker run --rm --entrypoint /bin/bash "$IMAGE_TAG" -c \
  'test -x /opt/sglang-lite/bin/sglang-serve && test -x /opt/sglang-lite/bin/ninja && \
   test -d /opt/sglang-lite/cache/sglang/jit && echo "  bundle tree OK"'

if [[ "${1:-}" == "--save" ]]; then
  OUT_TAR="$REPO_ROOT/dist/${IMAGE_TAG//:/-}.oci.tar"
  echo "[image] docker save -> $OUT_TAR"
  docker save "$IMAGE_TAG" -o "$OUT_TAR"
  du -h "$OUT_TAR"
fi
rm -rf "$CTX"
echo "[image] done. Run on a GPU host with nvidia-container-toolkit:"
echo "  docker run --gpus all -v /path/to/models:/models -p 30000:30000 \\"
echo "    $IMAGE_TAG --model-path /models/qwen3-4b --host 0.0.0.0 --port 30000 --mem-fraction-static 0.78"
