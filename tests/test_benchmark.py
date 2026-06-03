"""Tests for the benchmark harness."""

import json

from benchmarks.run_benchmark import markdown_report, run_benchmark, write_csv_report, write_dashboard, write_markdown_report, write_report


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
    assert report["leaderboard"]["most_improved"][0]["task_file"].endswith("task.json")


def test_benchmark_writes_report(tmp_path) -> None:
    """Benchmark reports should be written as JSON."""
    report_path = tmp_path / "report.json"
    report = {"metrics": {"case_count": 0}, "category_metrics": {}, "cases": []}

    write_report(report, report_path)

    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_benchmark_markdown_report_includes_tables(tmp_path) -> None:
    """Markdown reports should include summary and category tables."""
    report = {
        "metrics": {"case_count": 1, "pass_rate": 0.0, "average_score": 0.0, "failure_count": 1, "improvement_rate": 1.0},
        "category_metrics": {"planning": {"case_count": 1, "pass_rate": 0.0, "average_score": 0.0, "failure_count": 1}},
        "leaderboard": {"highest_score": [{"task_file": "x.json", "score": 0.0, "improved_score": 1.0}], "lowest_score": [], "most_improved": []},
        "cases": [{"task_file": "x.json", "score": 0.0, "improved_score": 1.0}],
    }
    text = markdown_report(report)
    path = tmp_path / "report.md"
    write_markdown_report(report, path)

    assert "# Benchmark Report" in text
    assert "| planning | 1 | 0.0 | 0.0 | 1 |" in text
    assert "## Leaderboard" in text
    assert path.read_text(encoding="utf-8") == text


def test_benchmark_csv_report_writes_case_rows(tmp_path) -> None:
    """CSV reports should export benchmark case rows."""
    report = {
        "cases": [
            {"task_file": "x.json", "category": "planning", "score": 0.5, "passed": False, "improved_score": 1.0, "improved": True, "failure_count": 2}
        ]
    }
    path = tmp_path / "report.csv"

    write_csv_report(report, path)

    text = path.read_text(encoding="utf-8")
    assert "task_file,category,score,passed,improved_score,improved,failure_count" in text
    assert "x.json,planning,0.5,False,1.0,True,2" in text


def test_dashboard_writes_latest_files(tmp_path) -> None:
    """Dashboard generation should write latest JSON, Markdown, and CSV files."""
    report = {
        "metrics": {"case_count": 1, "pass_rate": 0.0, "average_score": 0.0, "failure_count": 1, "improvement_rate": 1.0},
        "category_metrics": {},
        "leaderboard": {"highest_score": [], "lowest_score": [], "most_improved": []},
        "cases": [],
    }

    write_dashboard(report, tmp_path)

    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "latest.md").exists()
    assert (tmp_path / "latest.csv").exists()
