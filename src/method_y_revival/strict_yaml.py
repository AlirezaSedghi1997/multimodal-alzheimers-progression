from __future__ import annotations

"""YAML loading that fails closed on duplicate mapping keys.

PyYAML's default loader silently keeps the last value for a duplicate key.  In
research configuration this can change a metric or split rule without leaving
an obvious trace, so production-facing validation should use this loader.
"""

from pathlib import Path
from typing import Any

import yaml


class DuplicateYamlKeyError(ValueError):
    pass


class _UniqueLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            mark = getattr(key_node, "start_mark", None)
            where = f" line {mark.line + 1}" if mark is not None else ""
            raise DuplicateYamlKeyError(f"duplicate YAML key {key!r}{where}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_unique_yaml(path: str | Path) -> Any:
    return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueLoader)
