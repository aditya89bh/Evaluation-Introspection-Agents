"""Coding-agent demo for the evaluation-introspection loop.

Usage:
    python examples/run_coding_demo.py

This demo shows a coding agent making a simple bug, receiving deterministic
feedback, storing a behavior rule, and producing corrected code on the next run.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rule_memory import RuleMemory


TASK = "Write a Python function that returns the average of a list of numbers. Return 0 for an empty list."


BAD_CODE = """def average(numbers):
    return sum(numbers) / len(numbers)
"""


GOOD_CODE = """def average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
"""


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def evaluate_code(code: str) -> dict:
    namespace = {}
    exec(code, namespace)
    average = namespace["average"]

    test_cases = [
        ([2, 4, 6], 4),
        ([10, 20], 15),
        ([], 0),
    ]

    failures = []

    for input_value, expected in test_cases:
        try:
            actual = average(input_value)
        except Exception as error:
            failures.append(f"Input {input_value} raised {type(error).__name__}")
            continue

        if actual != expected:
            failures.append(f"Input {input_value} expected {expected}, got {actual}")

    passed = not failures
    score = 1.0 if passed else 0.67

    return {
        "passed": passed,
        "score": score,
        "failure_reason": None if passed else "; ".join(failures),
    }


def generate_code(rules: list[dict]) -> str:
    if not rules:
        return BAD_CODE
    return GOOD_CODE


if __name__ == "__main__":
    memory = RuleMemory(path=ROOT / "data" / "coding_rules.json")
    memory.reset()

    print_section("Coding Agent Self-Correction Demo")
    print("This demo shows a coding agent failing an edge case, storing a rule, and fixing the next attempt.")

    print_section("Task")
    print(TASK)

    print_section("Run 1: Initial Code")
    run_1_code = generate_code(memory.get_rules())
    print(run_1_code)

    run_1_eval = evaluate_code(run_1_code)
    print(f"Run 1 score: {run_1_eval['score']:.2f}")
    print(f"Passed: {run_1_eval['passed']}")
    print(f"Failure reason: {run_1_eval['failure_reason']}")

    print_section("Introspection")
    failure_summary = "The code fails on the empty-list edge case."
    root_cause = "The agent implemented the common case but did not check boundary conditions."
    behavior_rule = "Check edge cases before finalizing code."
    future_trigger = "coding tasks with lists, division, or empty inputs"

    print(f"Failure summary: {failure_summary}")
    print(f"Root cause: {root_cause}")
    print(f"Behavior rule: {behavior_rule}")
    print(f"Future trigger: {future_trigger}")

    memory.add_rule(rule=behavior_rule, trigger=future_trigger)

    print_section("Behavioral Memory")
    for index, rule in enumerate(memory.get_rules(), start=1):
        print(f"Rule {index}: {rule['rule']}")
        print(f"Trigger: {rule['trigger']}")

    print_section("Run 2: Corrected Code")
    run_2_code = generate_code(memory.get_rules())
    print(run_2_code)

    run_2_eval = evaluate_code(run_2_code)
    print(f"Run 2 score: {run_2_eval['score']:.2f}")
    print(f"Passed: {run_2_eval['passed']}")

    print_section("Result")
    print("The second attempt improves because the agent stored an edge-case rule after the first failure.")
