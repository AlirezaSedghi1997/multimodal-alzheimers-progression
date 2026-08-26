"""AST-walk dependency closure for the BrainIAC preprocessing runner."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_ASSET_ALLOWLIST = frozenset({
    "vendor/BrainIAC/ba60f45bed832ee0e5683c678caeaeefe2072f0d/LICENSE",
})


def _local_module_path(name: str) -> Path | None:
    parts = name.split(".")
    candidate = ROOT.joinpath(*parts).with_suffix(".py")
    if candidate.is_file():
        return candidate
    candidate = ROOT.joinpath(*parts, "__init__.py")
    if candidate.is_file():
        return candidate
    return None


def _walk_imports(path: Path, seen: set[Path]) -> list[Path]:
    if path in seen:
        return []
    seen.add(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    locals_found: list[Path] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            resolved = _local_module_path(name)
            if resolved is not None:
                locals_found.append(resolved)
                locals_found.extend(_walk_imports(resolved, seen))
    return locals_found


def build_dependency_closure(entrypoint: Path) -> dict[str, str]:
    """Return {relative_posix_path: sha256} for all local imports reachable from entrypoint."""

    entrypoint = Path(entrypoint).resolve()
    seen: set[Path] = set()
    all_files = sorted({entrypoint, *_walk_imports(entrypoint, seen)})
    closure: dict[str, str] = {}
    for path in all_files:
        rel = path.relative_to(ROOT).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        closure[rel] = digest
    return closure


def verify_closure_tracking(closure: dict[str, str]) -> list[str]:
    """Return list of untracked (not-in-Git) paths from the closure."""

    untracked = []
    for rel in sorted(closure):
        path = ROOT / rel
        result = Path(rel)
        import subprocess
        out = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel],
            capture_output=True, text=True, cwd=str(ROOT), check=False,
        )
        if out.returncode != 0:
            untracked.append(rel)
    return untracked
