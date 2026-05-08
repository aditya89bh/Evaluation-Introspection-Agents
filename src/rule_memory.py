"""Simple JSON-backed behavioral memory store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


RULES_PATH = Path("data/rules.json")


class RuleMemory:
    def __init__(self, path: Path = RULES_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            self._write([])

    def _read(self) -> List[dict]:
        with open(self.path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _write(self, rules: List[dict]) -> None:
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump(rules, file, indent=2)

    def reset(self) -> None:
        """Clear all stored rules."""
        self._write([])

    def add_rule(self, rule: str, trigger: str) -> None:
        rules = self._read()

        if any(existing["rule"] == rule for existing in rules):
            return

        rules.append(
            {
                "rule": rule,
                "trigger": trigger,
            }
        )

        self._write(rules)

    def get_rules(self) -> List[dict]:
        return self._read()
