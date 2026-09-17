#!/usr/bin/env python3
"""Install the slim tree's runtime dependencies into a venv.

Reads dependency requirements from the upstream python/pyproject.toml,
drops the packages proven removable by the 2026-09-16 slimming experiments
(see packaging/README.md), and installs torchvision from the CPU-only index
(decode_jpeg only; the CUDA-ops wheel is ~300x bigger).

CRITICAL constraint learned from the mathdx experiment: the sglang JIT cache
key fingerprints the *installed package set* (torch/flashinfer/deep_gemm/
nvidia-mathdx/tvm-ffi versions). The delivery machine's pip package set must
match the warmup machine's exactly, or the prebuilt JIT cache misses. Always
warm up the cache in the FINAL slimmed environment, after all uninstalls.

Usage:
  python3 packaging/install_deps.py --venv <dir> --python <pbs-python>
Then:
  SGLANG_SMOKE_MODEL=... bash packaging/smoke_test.sh          # warmup
  SGLANG_SMOKE_MODEL=... bash packaging/smoke_test.sh --no-toolkit  # verify
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Packages dropped after smoke-verified slimming (2026-09-16, Qwen3-4B).
# Each entry kept here must have passed the --no-toolkit smoke test.
# NOTE: multimodal deps (torchaudio/torchcodec/av/timm) are KEPT since the
# 2026-09-17 product decision to support multimodal models (option B).
DROPPED_REQUIREMENTS = {
    # model-hub / dataset tooling (weights are delivered out-of-band)
    "modelscope",
    "blobfile",
    "datasets",
    # dev tooling
    "py-spy",
    "watchfiles",
    # optional protocol/format support
    "anthropic",         # anthropic API adapter imports lazily per-request
    "mistral_common",    # mistral tokenizer format
    # structured-output backends other than xgrammar (xgrammar is a hard
    # startup dep via function_call/inkling_detector.py)
    "outlines",
    "llguidance",
    "interegular",
    # JIT toolchains: kernels are prebuilt during warmup; target machines
    # never compile. Keep the cache warm in THIS exact package set.
    # WARNING: cutlass-dsl is dropped, but flashinfer cute-dsl ops and
    # tokenspeed-mla declare a dependency on it — if a multimodal model
    # triggers a cute-dsl kernel at runtime, reinstall it (+450M).
    "tilelang",
    "numba",
    "llvmlite",
    "nvidia-cutlass-dsl",
    "nvidia-mathdx",
}


def req_name(dep: str) -> str:
    """Extract the bare package name from a PEP 508 requirement string."""
    return re.split(r"[<>=!;@\[ ]", dep)[0].strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", type=Path, default=REPO_ROOT / "build-slim" / "venv")
    ap.add_argument(
        "--python",
        type=Path,
        default=None,
        help="Interpreter to build the venv with (e.g. python-build-standalone). "
        "Defaults to sys.executable.",
    )
    args = ap.parse_args()

    venv = args.venv
    base_python = str(args.python) if args.python else sys.executable
    if not (venv / "bin" / "pip").exists():
        subprocess.run([base_python, "-m", "venv", str(venv)], check=True)

    pyproject = tomllib.loads((REPO_ROOT / "python" / "pyproject.toml").read_text())
    deps, dropped = [], []
    for d in pyproject["project"]["dependencies"]:
        (dropped if req_name(d) in DROPPED_REQUIREMENTS else deps).append(d)

    req_file = venv.parent / "slim-requirements.txt"
    req_file.write_text("\n".join(deps) + "\n")
    print(f"[deps] {len(deps)} install, {len(dropped)} dropped -> {req_file}")
    for d in dropped:
        print(f"[deps]   dropped: {d}")

    pip = str(venv / "bin" / "pip")
    subprocess.run([pip, "install", "--upgrade", "pip"], check=True)
    subprocess.run([pip, "install", "-r", str(req_file)], check=True)

    # torchvision: CPU-only wheel (decode_jpeg needs no CUDA ops; the cu13
    # wheel is ~300x larger). --no-deps keeps the installed cu13 torch.
    subprocess.run(
        [pip, "install", "--no-deps", "torchvision",
         "--index-url", "https://download.pytorch.org/whl/cpu"],
        check=True,
    )
    print("[deps] done. Remember: warm the JIT cache in THIS package set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
