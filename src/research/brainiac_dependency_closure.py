"""AST-walk dependency closure for the BrainIAC preprocessing runner."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Mapping

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


def _import_candidates(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom) or not node.module:
        return []
    candidates = [node.module]
    for alias in node.names:
        if alias.name != "*":
            candidates.append(f"{node.module}.{alias.name}")
    return candidates


def _walk_imports(path: Path, seen: set[Path]) -> list[Path]:
    if path in seen:
        return []
    seen.add(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    locals_found: list[Path] = []
    for node in ast.walk(tree):
        for name in _import_candidates(node):
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


def build_execution_dependency_manifest(
    entrypoint: Path,
    *,
    static_inputs: Mapping[str, Path],
    dynamic_code_inputs: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    """Bind local Python closure and non-Python contract inputs.

    ``from package import module`` imports are represented by the imported
    submodule, not merely the package ``__init__``.  Static scientific inputs
    (policy/config) are explicit because AST walking cannot discover them.
    """

    entrypoint = Path(entrypoint).resolve()
    closure = build_dependency_closure(entrypoint)
    def bind_inputs(inputs: Mapping[str, Path]) -> dict[str, dict[str, object]]:
        bound: dict[str, dict[str, object]] = {}
        for role, candidate in sorted(inputs.items()):
            path = Path(candidate).resolve()
            try:
                relative = path.relative_to(ROOT).as_posix()
            except ValueError as exc:
                raise RuntimeError("BRAINIAC_DEPENDENCY_INPUT_OUTSIDE_PROJECT") from exc
            if not path.is_file():
                raise RuntimeError(f"BRAINIAC_DEPENDENCY_INPUT_MISSING:{role}")
            bound[str(role)] = {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        return bound

    bound_inputs = bind_inputs(static_inputs)
    bound_dynamic_code = bind_inputs(dynamic_code_inputs or {})
    body: dict[str, object] = {
        "schema_version": 1,
        "entrypoint": entrypoint.relative_to(ROOT).as_posix(),
        "python_closure": closure,
        "dynamic_code_inputs": bound_dynamic_code,
        "static_inputs": bound_inputs,
    }
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    body["closure_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


def verify_closure_tracking(closure: dict[str, str]) -> list[str]:
    """Return list of untracked (not-in-Git) paths from the closure."""

    untracked = []
    for rel in sorted(closure):
        import subprocess
        out = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel],
            capture_output=True, text=True, cwd=str(ROOT), check=False,
        )
        if out.returncode != 0:
            untracked.append(rel)
    return untracked


def verify_manifest_tracking(manifest: Mapping[str, object]) -> list[str]:
    """Return every untracked executable/config input in a dependency manifest."""

    paths = set(map(str, dict(manifest.get("python_closure") or {}).keys()))
    for role in ("dynamic_code_inputs", "static_inputs"):
        for binding in dict(manifest.get(role) or {}).values():
            if isinstance(binding, Mapping) and binding.get("path"):
                paths.add(str(binding["path"]))
    return verify_closure_tracking({path: "" for path in sorted(paths)})
