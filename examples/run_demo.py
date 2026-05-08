"""Runnable demo for the evaluation-introspection loop.

Usage:
    python examples/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_loop import run_loop


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_score(label: str, evaluation: dict) -> None:
    print(f"{label} score: {evaluation['score']:.2f}")
    print(f"Passed: {evaluation['passed']}")
    if evaluation.get("failure_reason"):
        print(f"Failure reason: {evaluation['failure_reason']}")


if __name__ == "__main__":
    result = run_loop()

    print_section("Evaluation Introspection Agent Demo")
    print("This demo shows an agent failing once, introspecting, storing a rule, and improving on the next attempt.")

    print_section("Task")
    print(result["task"])

    print_section("Run 1: Initial Attempt")
    print(result["run_1"]["output"])
    print()
    print_score("Run 1", result["run_1"]["evaluation"])

    print_section("Introspection")
    introspection = result["run_1"]["introspection"]
    print(f"Failure summary: {introspection['failure_summary']}")
    print(f"Root cause: {introspection['root_cause']}")
    print(f"Behavior rule: {introspection['behavior_rule']}")
    print(f"Future trigger: {introspection['future_trigger']}")

    print_section("Behavioral Memory")
    for index, rule in enumerate(result["memory"], start=1):
        print(f"Rule {index}: {rule['rule']}")
        print(f"Trigger: {rule['trigger']}")

    print_section("Run 2: Improved Attempt")
    print(result["run_2"]["output"])
    print()
    print_score("Run 2", result["run_2"]["evaluation"])

    print_section("Result")
    print("The second attempt improves because the system converted the first failure into a reusable behavior rule.")
