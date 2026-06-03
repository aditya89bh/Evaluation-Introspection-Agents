"""Tests for the command line runner."""

import json

from evaluation_introspection_agents.cli import load_task_file, main


def test_cli_loads_task_file_and_runs_pipeline(tmp_path, capsys) -> None:
    """The CLI should load a task file and print readable pipeline output."""
    task_file = tmp_path / "task.json"
    task_file.write_text(
        json.dumps(
            {
                "objective": "Mention rollback and owner",
                "expected_terms": ["rollback", "owner"],
                "output": "Maybe launch it.",
                "trace": [{"name": "draft", "detail": "Wrote a launch note."}],
            }
        ),
        encoding="utf-8",
    )

    assert main(["run", str(task_file)]) == 0
    captured = capsys.readouterr().out

    assert "Task: Mention rollback and owner" in captured
    assert "Score: 0.00" in captured
    assert "Critique:" in captured
    assert "Improved output:" in captured


def test_load_task_file_defaults_constraints_to_expected_terms(tmp_path) -> None:
    """Task files can omit constraints when expected terms are enough."""
    task_file = tmp_path / "task.json"
    task_file.write_text(
        json.dumps({"objective": "Do x", "expected_terms": ["x"], "output": ""}),
        encoding="utf-8",
    )

    task, output, trace, constraints = load_task_file(task_file)

    assert task.objective == "Do x"
    assert output == ""
    assert trace.steps == ()
    assert constraints == ("x",)
