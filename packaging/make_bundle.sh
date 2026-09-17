#!/usr/bin/env bash
# Assemble the self-contained slim bundle (deliverable directory).
#
# Inputs (all produced/verified by earlier pipeline stages):
#   dist/sglang-slim      slim source tree      (build_slim.py)
#   $BUILD_DIR/python     PBS Python            (relocatable)
#   $BUILD_DIR/venv       slimmed dep venv      (install_deps.py + uninstalls)
#   $SGLANG_CACHE_DIR     warmed JIT caches     (smoke_test.sh first run)
#
# Output:
#   dist/sglang-lite/                       the bundle itself
#   dist/sglang-lite-<tag>.tar.zst          release tarball
#
# Design (all facts verified on 2026-09-16, see packaging/README.md):
# - One env (SGLANG_CACHE_DIR) redirects every JIT cache family
#   (jit/triton/deep_gemm/inductor/nv/flashinfer workspace) — upstream
#   environ.py's third_party_cache_defaults().
# - PBS Python is relocatable; site-packages is carried via PYTHONPATH
#   (no venv machinery on the target machine).
# - Apache-2.0: ship LICENSE + NOTICE; freeze.txt is the package-set lock
#   (the JIT cache key fingerprints it — see the mathdx lesson).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${SGLANG_SLIM_BUILD:-$REPO_ROOT/build-slim}"
CACHE_SRC="${SGLANG_CACHE_DIR:-$HOME/.cache/sglang}"
OUT_ROOT="${SGLANG_SLIM_BUNDLE:-$REPO_ROOT/dist/sglang-lite}"
TAG="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo node)"
SGL_VERSION="$(python3 -c "
import re,pathlib
t = pathlib.Path('$REPO_ROOT/python/sglang/version.py').read_text()
m = re.search(r'__version__\s*=\s*[\"\\']([^\"\\']+)', t)
print(m.group(1) if m else 'dev')" 2>/dev/null || echo dev)"

SLIM_TREE="$REPO_ROOT/dist/sglang-slim"
PBS="$BUILD_DIR/python"
VENV_SP="$BUILD_DIR/venv/lib/python3.12/site-packages"

test -d "$SLIM_TREE/sglang" || { echo "missing $SLIM_TREE (run build_slim.py)"; exit 1; }
test -x "$PBS/bin/python3" || { echo "missing PBS python at $PBS"; exit 1; }
test -d "$VENV_SP" || { echo "missing venv site-packages at $VENV_SP"; exit 1; }
test -d "$CACHE_SRC/jit" || { echo "missing warmed cache at $CACHE_SRC (run smoke first)"; exit 1; }

echo "[bundle] assembling $OUT_ROOT (upstream sglang $SGL_VERSION, anchor $TAG)"
rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT"/{bin,engine,cache}

# --- runtime: PBS Python (relocatable by design) ---
cp -a "$PBS" "$OUT_ROOT/runtime"

# --- deps: site-packages without venv machinery ---
mkdir -p "$OUT_ROOT/lib/python"
cp -a "$VENV_SP" "$OUT_ROOT/lib/python/site-packages"

# --- runtime block-list prune (verified 2026-09-16 via /proc/<pid>/maps on a
#     live serve: these files are never loaded; ~310M). Since the 2026-09-17
#     multimodal decision (option B) the cudnn engine sublibs and ALL headers
#     are KEPT: VL models may dlopen cudnn for conv-heavy vision encoders and
#     may trigger new flashinfer JIT builds that need headers on the warmup
#     machine. ---
PRUNE_SP="$OUT_ROOT/lib/python/site-packages"
rm -f "$PRUNE_SP/nvidia/cu13/lib/libnvrtc.alt.so.13" \
      "$PRUNE_SP/nvidia/cu13/lib/libcusolverMg.so.12" \
      "$PRUNE_SP/nvidia/cu13/lib/libnvvm.so.4" \
      "$PRUNE_SP/nvidia/cu13/lib/libnvperf_host.so" \
      "$PRUNE_SP/nvidia/cu13/lib/libnvperf_target.so"

# --- engine: slim sglang source tree ---
cp -a "$SLIM_TREE/sglang" "$OUT_ROOT/engine/sglang"

# --- warmed JIT caches (single directory, read-only on target) ---
cp -a "$CACHE_SRC" "$OUT_ROOT/cache/sglang"

# --- launcher + wheel-installed binaries ---
# ninja lives in venv/bin (wheel data_scripts), not site-packages; flashinfer
# spawns it on every startup for its JIT freshness scan (even fully cached).
mkdir -p "$OUT_ROOT/bin"
cp "$BUILD_DIR/venv/bin/ninja" "$OUT_ROOT/bin/ninja"
cat > "$OUT_ROOT/bin/sglang-serve" <<EOF
#!/usr/bin/env bash
# sglang-slim launcher: self-contained, target machine needs only the NVIDIA driver.
# - Bundle stays read-only: the warmed JIT cache is seeded once into a
#   writable runtime dir (flashinfer writes a log into its workspace at
#   import time, so the cache dir itself must be writable).
# - PATH carries the bundled ninja binary: flashinfer 0.6.18's JIT loader
#   always spawns ninja for a freshness scan (try_load returns None on the
#   JIT path by design), even when every artifact is cached.
DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_CACHE="\${XDG_CACHE_HOME:-\$HOME/.cache}/sglang-slim-runtime"
if [ -d "\$DIR/cache/sglang/jit" ] && [ ! -d "\$RUNTIME_CACHE/jit" ]; then
  mkdir -p "\$RUNTIME_CACHE"
  cp -a "\$DIR/cache/sglang/." "\$RUNTIME_CACHE/" || true
  chmod -R u+w "\$RUNTIME_CACHE" || true   # cp -a preserves the read-only source mode
fi
export SGLANG_CACHE_DIR="\$RUNTIME_CACHE"
# SGLANG_JIT_CACHE_DIR's default does NOT derive from SGLANG_CACHE_DIR
# (environ.py:1155 defaults to None -> cache.py falls back to the hardcoded
# ~/.cache/sglang/jit); set it explicitly.
export SGLANG_JIT_CACHE_DIR="\$RUNTIME_CACHE/jit"
export PATH="\$DIR/bin:\$PATH"
export PYTHONPATH="\$DIR/engine:\$DIR/lib/python/site-packages\${PYTHONPATH:+:\$PYTHONPATH}"
exec "\$DIR/runtime/bin/python3" -s -m sglang.launch_server "\$@"
EOF
chmod +x "$OUT_ROOT/bin/sglang-serve"

# --- metadata & compliance ---
cp "$REPO_ROOT/LICENSE" "$OUT_ROOT/LICENSE"
{
  echo "sglang-slim bundle"
  echo "upstream sglang: $SGL_VERSION (anchor: $TAG)"
  echo "python: $($PBS/bin/python3 --version 2>&1)"
  echo "cuda family: cu13 (carried via nvidia-* wheels)"
  echo "cache families: $(ls "$OUT_ROOT/cache/sglang" | tr '\n' ' ')"
} > "$OUT_ROOT/VERSION"
{
  echo "The bundled sglang source (engine/sglang) is Apache-2.0, Copyright contributors to the SGLang project."
  echo "Third-party packages below retain their own licenses (see lib/python/site-packages/*.dist-info)."
} > "$OUT_ROOT/NOTICE"
"$BUILD_DIR/venv/bin/pip" freeze --all > "$OUT_ROOT/freeze.txt" 2>/dev/null || \
  "$PBS/bin/python3" -m pip freeze --path "$VENV_SP" > "$OUT_ROOT/freeze.txt"

# --- absolute-path residue audit (delivery gap check) ---
echo "[bundle] auditing absolute-path residue in site-packages configs ..."
found=0
while IFS= read -r -d '' f; do
  if grep -q "$BUILD_DIR\|$REPO_ROOT" "$f"; then echo "  RESIDUE: $f"; found=1; fi
done < <(find "$OUT_ROOT/lib/python/site-packages" -maxdepth 2 \( -name '*.pth' -o -name '*.egg-link' -o -name 'direct_url.json' \) -print0 2>/dev/null)
if [[ $found -eq 0 ]]; then echo "[bundle] no path residue in .pth/direct_url"; fi

# --- tarball ---
echo "[bundle] creating tarball ..."
TARBALL="$REPO_ROOT/dist/sglang-lite-$SGL_VERSION-$TAG.tar.zst"
tar -C "$REPO_ROOT/dist" --zstd -cf "$TARBALL" sglang-lite

echo "[bundle] done:"
du -sh "$OUT_ROOT" "$TARBALL" || true
echo "[bundle] e2e verify: unpack elsewhere, HOME=elsewhere, chmod a-w cache, then bin/sglang-serve"
