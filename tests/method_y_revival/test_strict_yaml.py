from __future__ import annotations

from pathlib import Path

import pytest

from src.method_y_revival.strict_yaml import DuplicateYamlKeyError, load_unique_yaml


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("a: 1\na: 2\n", encoding="utf-8")
    with pytest.raises(DuplicateYamlKeyError):
        load_unique_yaml(path)


def test_post_eval_v41_is_duplicate_free() -> None:
    root = Path(r"H:\New project")
    payload = load_unique_yaml(root / "configs/method_y_post_eval_v4_1.yaml")
    assert payload["risk_score"] == "1_minus_S_tau"
    assert payload["tie_rule"] == 0.5
    assert payload["schema_version"] == "4.1"
