#!/usr/bin/env python3
"""Assemble the slim single-node sglang source tree (layer 0/2 pruning).

Copies python/sglang into a staging tree with prune_manifest exclusions
applied, then verifies the zero-diff safety contract:

  1. py_compile every kept .py file (syntax integrity of the staged tree).
  2. AST scan: no module-level import in the kept tree may reference an
     excluded module (this is exactly the "zero patch" contract; violations
     fail the build with file:line details).

Usage (from repo root or packaging/):
  python3 packaging/build_slim.py [--src python/sglang] [--out dist/sglang-slim]

The staged tree is a valid sglang package (same import paths) and is the
input for the wheel build / self-contained packaging stage.
"""

from __future__ import annotations

import argparse
import ast
import compileall
import fnmatch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import prune_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_exclude_matchers():
    """Return (package_patterns, model_filter) from the manifest."""
    patterns = list(prune_manifest.PACKAGE_EXCLUDES)
    return patterns


def is_model_excluded(rel_posix: str) -> bool:
    """Apply the models whitelist: everything under srt/models/ that is not
    whitelisted by prefix / shared file / shared dir is dropped."""
    if not rel_posix.startswith("srt/models/"):
        return False
    rest = rel_posix[len("srt/models/") :]
    if "/" in rest:
        first = rest.split("/", 1)[0]
        return first not in prune_manifest.MODELS_KEEP_DIRS
    return (
        rest not in prune_manifest.MODELS_KEEP_SHARED
        and not rest.startswith(prune_manifest.MODELS_KEEP_PREFIXES)
    )


def should_exclude(rel_posix: str, patterns: list[str]) -> bool:
    if is_model_excluded(rel_posix):
        return True
    for pat in patterns:
        if fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(rel_posix, pat + "/*") or rel_posix == pat:
            return True
    return False


def excluded_module_names() -> set[str]:
    """Importable module paths that will be missing from the staged tree.

    Rough mapping: each excluded path becomes its dotted module name; a
    package dir also matches its submodules, so we check by prefix.
    """
    names = set()
    for pat in prune_manifest.PACKAGE_EXCLUDES:
        p = pat[:-3] if pat.endswith(".py") else pat
        names.add("sglang." + p.replace("/", "."))
    # model modules excluded by the whitelist are all optional (registry
    # scans with pkgutil), so they are not part of the strict contract;
    # only PACKAGE_EXCLUDES entries are asserted.
    return names


def matches_excluded(module: str, excluded: set[str]) -> str | None:
    """Return the excluded module name that `module` references, if any."""
    for ex in excluded:
        if module == ex or module.startswith(ex + "."):
            return ex
    return None


def module_level_imports(py: Path):
    """Yield (lineno, module_name) for module-level imports only.

    Scope rule: module body and class body imports count (they execute at
    import time, including inside module-level `if`/`try` blocks); anything
    inside FunctionDef/AsyncFunctionDef/Lambda is skipped (lazy import).
    """
    try:
        tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"), filename=str(py))
    except SyntaxError:
        # Non-syntax-valid files (templates etc.) are reported by py_compile.
        return

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue  # skip function scope entirely
            if isinstance(child, ast.Import):
                for a in child.names:
                    yield child.lineno, a.name
            elif isinstance(child, ast.ImportFrom):
                if child.module and child.level == 0:
                    yield child.lineno, child.module
            elif isinstance(child, (ast.If, ast.Try, ast.ClassDef)):
                yield from visit(child)

    yield from visit(tree)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=REPO_ROOT / "python" / "sglang")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "dist" / "sglang-slim" / "sglang")
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()

    src: Path = args.src
    out: Path = args.out
    if out.exists():
        import shutil

        shutil.rmtree(out)
    out.mkdir(parents=True)

    patterns = build_exclude_matchers()
    kept_files, excluded_files, kept_lines = 0, 0, 0
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(src).as_posix()
        if should_exclude(rel, patterns):
            excluded_files += 1
            continue
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.read_bytes())
        kept_files += 1
        if rel.endswith(".py"):
            try:
                kept_lines += len(f.read_text(encoding="utf-8", errors="replace").splitlines())
            except Exception:
                pass

    print(f"[stage] kept {kept_files} files ({kept_lines} py lines), excluded {excluded_files} files -> {out}")

    # ---- Verify 1: syntax integrity ----
    if not args.skip_compile:
        ok = compileall.compile_dir(str(out.parent), quiet=2, force=True, rx=None)
        if not ok:
            print("[verify] py_compile FAILED", file=sys.stderr)
            return 2
        print("[verify] py_compile OK")

    # ---- Verify 2: zero module-level references to excluded modules ----
    excluded = excluded_module_names()
    violations = []
    for py in out.rglob("*.py"):
        for lineno, module in module_level_imports(py):
            hit = matches_excluded(module, excluded)
            if hit:
                violations.append(f"{py.relative_to(out)}:{lineno}: imports {module} (excluded: {hit})")
    if violations:
        print("[verify] ZERO-PATCH CONTRACT VIOLATED: module-level imports of excluded modules:", file=sys.stderr)
        for v in violations:
            print("  " + v, file=sys.stderr)
        return 3
    print(f"[verify] zero module-level imports of {len(excluded)} excluded modules: OK")

    stats = {
        "kept_files": kept_files,
        "excluded_files": excluded_files,
        "kept_py_lines": kept_lines,
    }
    (out.parent / "slim-stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[done] {json.dumps(stats)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
