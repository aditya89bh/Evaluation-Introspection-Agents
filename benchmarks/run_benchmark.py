"""Deterministic benchmark harness for example task files."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from evaluation_introspection_agents.cli import load_task_file
from evaluation_introspection_agents.core.feedback_loop import FeedbackLoop

DEFAULT_TASK_DIR = Path("benchmarks/scenarios")
DEFAULT_REPORT_PATH = Path("benchmarks/benchmark_report.json")


def run_benchmark(task_dir: Path = DEFAULT_TASK_DIR) -> dict[str, Any]:
    """Run the feedback loop over all JSON task files in a directory."""
    loop = FeedbackLoop()
    cases: list[dict[str, Any]] = []
    for task_file in sorted(task_dir.glob("*.json")):
        task, output, trace, constraints = load_task_file(task_file)
        result = loop.run(task, output, trace, constraints)
        improved_result = loop.evaluator.evaluate(task, result.improved_output)
        cases.append(
            {
                "task_file": str(task_file),
                "category": json.loads(task_file.read_text(encoding="utf-8")).get("category", "uncategorized"),
                "score": result.evaluation.score,
                "passed": result.evaluation.score >= 0.8,
                "improved_score": improved_result.score,
                "improved": improved_result.score > result.evaluation.score,
                "failure_count": len(result.critiques) if result.evaluation.score < 0.8 else 0,
            }
        )

    total = len(cases)
    pass_count = sum(1 for case in cases if case["passed"])
    improvement_count = sum(1 for case in cases if case["improved"])
    report = {
        "metrics": {
            "case_count": total,
            "pass_rate": round(pass_count / total, 2) if total else 0.0,
            "average_score": round(mean(case["score"] for case in cases), 2) if cases else 0.0,
            "failure_count": sum(case["failure_count"] for case in cases),
            "improvement_rate": round(improvement_count / total, 2) if total else 0.0,
        },
        "cases": cases,
    }
    return report


def write_report(report: dict[str, Any], path: Path = DEFAULT_REPORT_PATH) -> None:
    """Write a benchmark report as stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    """Run the benchmark and write the default report."""
    report = run_benchmark()
    write_report(report)
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
