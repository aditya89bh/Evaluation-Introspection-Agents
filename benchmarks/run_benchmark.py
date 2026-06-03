"""Deterministic benchmark harness for example task files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict
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
    category_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        category_groups[case["category"]].append(case)
    category_metrics = {
        category: {
            "case_count": len(group),
            "pass_rate": round(sum(1 for case in group if case["passed"]) / len(group), 2),
            "average_score": round(mean(case["score"] for case in group), 2),
            "failure_count": sum(case["failure_count"] for case in group),
        }
        for category, group in sorted(category_groups.items())
    }
    leaderboard = {
        "highest_score": sorted(cases, key=lambda case: (-case["score"], case["task_file"]))[:5],
        "lowest_score": sorted(cases, key=lambda case: (case["score"], case["task_file"]))[:5],
        "most_improved": sorted(
            cases,
            key=lambda case: (-(case["improved_score"] - case["score"]), case["task_file"]),
        )[:5],
    }
    report = {
        "metrics": {
            "case_count": total,
            "pass_rate": round(pass_count / total, 2) if total else 0.0,
            "average_score": round(mean(case["score"] for case in cases), 2) if cases else 0.0,
            "failure_count": sum(case["failure_count"] for case in cases),
            "improvement_rate": round(improvement_count / total, 2) if total else 0.0,
        },
        "category_metrics": category_metrics,
        "leaderboard": leaderboard,
        "cases": cases,
    }
    return report


def write_report(report: dict[str, Any], path: Path = DEFAULT_REPORT_PATH) -> None:
    """Write a benchmark report as stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_report(report: dict[str, Any]) -> str:
    """Render a benchmark report as Markdown."""
    metrics = report["metrics"]
    lines = [
        "# Benchmark Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ("case_count", "pass_rate", "average_score", "failure_count", "improvement_rate"):
        lines.append(f"| {key} | {metrics[key]} |")
    lines.extend(["", "## Category breakdown", "", "| Category | Cases | Pass rate | Avg score | Failures |", "|---|---:|---:|---:|---:|"])
    for category, values in report.get("category_metrics", {}).items():
        lines.append(
            f"| {category} | {values['case_count']} | {values['pass_rate']} | {values['average_score']} | {values['failure_count']} |"
        )
    lines.extend(["", "## Leaderboard", ""])
    for title, key in (("Highest score", "highest_score"), ("Lowest score", "lowest_score"), ("Most improved", "most_improved")):
        lines.append(f"### {title}")
        lines.append("")
        for case in report.get("leaderboard", {}).get(key, []):
            delta = round(case["improved_score"] - case["score"], 2)
            lines.append(f"- `{case['task_file']}` score `{case['score']}` improved `{case['improved_score']}` delta `{delta}`")
        lines.append("")
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(report: dict[str, Any], path: Path = Path("results/benchmark_report.md")) -> None:
    """Write a Markdown benchmark report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_report(report), encoding="utf-8")


def write_csv_report(report: dict[str, Any], path: Path = Path("results/benchmark_report.csv")) -> None:
    """Write benchmark cases as a CSV report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task_file", "category", "score", "passed", "improved_score", "improved", "failure_count"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for case in report["cases"]:
            writer.writerow({key: case.get(key, "") for key in fieldnames})


def write_dashboard(report: dict[str, Any], results_dir: Path = Path("results")) -> None:
    """Write latest benchmark dashboard files in JSON, Markdown, and CSV formats."""
    results_dir.mkdir(parents=True, exist_ok=True)
    write_report(report, results_dir / "latest.json")
    write_markdown_report(report, results_dir / "latest.md")
    write_csv_report(report, results_dir / "latest.csv")


def main() -> int:
    """Run the benchmark and write the default report."""
    report = run_benchmark()
    write_report(report)
    write_markdown_report(report)
    write_csv_report(report)
    write_dashboard(report)
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
