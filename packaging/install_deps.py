#!/usr/bin/env python3
"""Install the slim tree's runtime dependencies into a venv.

Reads dependency requirements from the upstream python/pyproject.toml
(deps are unchanged by pruning; layer-2 pruning removes code, not wheels),
skips extras (test/diffusion/ray), and installs the sglang package itself
from the staged slim tree in editable-less mode via PYTHONPATH instead.

Usage:
  python3 packaging/install_deps.py --venv .venv-slim
Then:
  .venv-slim/bin/pip install --no-deps -e . (never; we use PYTHONPATH)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Skip deps that are only needed by pruned/optional paths in the smoke test.
SKIP = {
    # multimodal audio/vision input paths (Qwen3-8B is text-only)
    # NOTE: torchvision etc. are still installed because torch/sglang import
    # them opportunistically in a few modules; pruning deps is a later step.
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", type=Path, default=REPO_ROOT / ".venv-slim")
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
    deps = [d for d in pyproject["project"]["dependencies"] if d not in SKIP]

    req_file = venv.parent / "slim-requirements.txt"
    req_file.write_text("\n".join(deps) + "\n")
    print(f"[deps] {len(deps)} requirements -> {req_file}")

    pip = str(venv / "bin" / "pip")
    subprocess.run([pip, "install", "--upgrade", "pip"], check=True)
    subprocess.run([pip, "install", "-r", str(req_file)], check=True)
    print("[deps] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
