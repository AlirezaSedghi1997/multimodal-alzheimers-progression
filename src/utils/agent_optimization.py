from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AgentTokenTracker:
    def __init__(self, tracking_file: Path | None = None):
        self.tracking_file = tracking_file
        self.calls: list[dict[str, Any]] = []
        self.total_tokens: int = 0
        if tracking_file and tracking_file.exists():
            self._load()

    def _load(self) -> None:
        if self.tracking_file and self.tracking_file.exists():
            try:
                data = json.loads(self.tracking_file.read_text(encoding="utf-8"))
                self.calls = data.get("calls", [])
                self.total_tokens = data.get("total_tokens", 0)
            except Exception:
                self.calls = []
                self.total_tokens = 0

    def _save(self) -> None:
        if self.tracking_file:
            data = {
                "calls": self.calls[-1000:],
                "total_tokens": self.total_tokens,
            }
            self.tracking_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def record_call(
        self,
        command: str,
        tokens_used: int,
        *,
        category: str = "unknown",
        success: bool = True,
    ) -> None:
        entry = {
            "command": command,
            "tokens": tokens_used,
            "category": category,
            "success": success,
        }
        self.calls.append(entry)
        self.total_tokens += tokens_used
        self._save()

    def get_category_stats(self) -> dict[str, dict[str, int]]:
        stats: dict[str, dict[str, int]] = {}
        for call in self.calls:
            cat = call.get("category", "unknown")
            if cat not in stats:
                stats[cat] = {"count": 0, "tokens": 0}
            stats[cat]["count"] += 1
            stats[cat]["tokens"] += call.get("tokens", 0)
        return stats

    def should_spawn_subagent(
        self,
        task_complexity: float,
        estimated_tokens: int,
        *,
        threshold: float = 0.3,
    ) -> bool:
        if estimated_tokens < 100:
            return False
        if task_complexity < threshold:
            return False
        return True

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_calls": len(self.calls),
            "total_tokens": self.total_tokens,
            "category_stats": self.get_category_stats(),
        }


class CompactOutputFormatter:
    @staticmethod
    def compact_pytest(output: str, *, max_lines: int = 50) -> str:
        lines = output.strip().split("\n")
        if len(lines) <= max_lines:
            return output
        failed_lines = [l for l in lines if "FAILED" in l or "ERROR" in l]
        summary_lines = [l for l in lines if "passed" in l or "failed" in l]
        compact = failed_lines + ["..."] + summary_lines
        return "\n".join(compact)

    @staticmethod
    def compact_git_status(output: str, *, max_lines: int = 30) -> str:
        lines = output.strip().split("\n")
        if len(lines) <= max_lines:
            return output
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"

    @staticmethod
    def compact_json(data: dict, *, max_keys: int = 20) -> str:
        if len(data) <= max_keys:
            return json.dumps(data, indent=2)
        compact = {k: data[k] for k in list(data.keys())[:max_keys]}
        compact["_remaining_keys"] = len(data) - max_keys
        return json.dumps(compact, indent=2)

    @staticmethod
    def compact_metrics(metrics: dict) -> dict:
        important_keys = [
            "roc_auc",
            "pr_auc",
            "brier",
            "accuracy",
            "sensitivity",
            "specificity",
            "f1",
            "n",
            "status",
        ]
        return {k: v for k, v in metrics.items() if k in important_keys}
