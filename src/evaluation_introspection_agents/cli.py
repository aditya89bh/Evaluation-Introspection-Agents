"""Command line runner for evaluation-introspection agents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from evaluation_introspection_agents.core.feedback_loop import FeedbackLoop
from evaluation_introspection_agents.core.task import Task
from evaluation_introspection_agents.core.trace import BehaviorTrace


def load_task_file(path: Path) -> tuple[Task, str, BehaviorTrace, tuple[str, ...]]:
    """Load a JSON task file into deterministic pipeline inputs."""
    data = json.loads(path.read_text(encoding="utf-8"))
    task = Task(
        objective=data["objective"],
        expected_terms=tuple(data.get("expected_terms", ())),
    )
    output = data.get("output", "")
    trace = BehaviorTrace()
    for step in data.get("trace", []):
        trace = trace.add(step["name"], step["detail"], step.get("observation", ""))
    constraints = tuple(data.get("constraints", task.expected_terms))
    return task, output, trace, constraints


def format_readable(result: Any) -> str:
    """Format a feedback loop result for humans."""
    lines = [
        f"Task: {result.task.objective}",
        f"Score: {result.evaluation.score:.2f}",
        f"Evaluation: {result.evaluation.rationale}",
        f"Introspection: {result.introspection.summary}",
        "Critique:",
    ]
    lines.extend(f"- {item}" for item in result.critiques)
    lines.extend(
        [
            f"Improvement: {result.improvement}",
            "Improved output:",
            result.improved_output,
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="evaluation-agents")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run the evaluation pipeline for a task file.")
    run_parser.add_argument("task_file", type=Path, help="Path to a JSON task file.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        task, output, trace, constraints = load_task_file(args.task_file)
        result = FeedbackLoop().run(task, output, trace, constraints)
        print(format_readable(result))
        return 0
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
