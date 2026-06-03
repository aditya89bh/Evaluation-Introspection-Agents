"""Tests for the benchmark harness."""

import json

from benchmarks.run_benchmark import run_benchmark, write_report


def test_benchmark_computes_required_metrics(tmp_path) -> None:
    """The benchmark should compute pass, score, failure, and improvement metrics."""
    task_file = tmp_path / "task.json"
    task_file.write_text(
        json.dumps(
            {
                "category": "planning",
                "objective": "Mention rollback",
                "expected_terms": ["rollback"],
                "output": "Launch it.",
                "trace": [{"name": "draft", "detail": "Wrote output."}],
            }
        ),
        encoding="utf-8",
    )

    report = run_benchmark(tmp_path)

    assert report["metrics"]["case_count"] == 1
    assert report["metrics"]["pass_rate"] == 0.0
    assert report["metrics"]["average_score"] == 0.0
    assert report["metrics"]["failure_count"] > 0
    assert report["metrics"]["improvement_rate"] == 1.0
    assert report["category_metrics"]["planning"]["case_count"] == 1
    assert report["category_metrics"]["planning"]["pass_rate"] == 0.0


def test_benchmark_writes_report(tmp_path) -> None:
    """Benchmark reports should be written as JSON."""
    report_path = tmp_path / "report.json"
    report = {"metrics": {"case_count": 0}, "category_metrics": {}, "cases": []}

    write_report(report, report_path)

    assert json.loads(report_path.read_text(encoding="utf-8")) == report
