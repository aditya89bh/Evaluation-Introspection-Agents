"""Basic tests for the evaluation-introspection loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_loop import run_loop
from evaluator import evaluate_output


BAD_OUTPUT = """1. Technical architecture.
2. Behavioral embeddings.
3. Retry strategies.
4. Episodic critique.
5. Constraint adherence.
6. Extra bullet."""


GOOD_OUTPUT = """1. The system reviews mistakes.
2. It checks whether instructions were followed.
3. The agent explains why failures happened.
4. Lessons are stored for future tasks.
5. Future attempts improve using feedback."""


def test_bad_output_fails_constraints():
    result = evaluate_output(BAD_OUTPUT)

    assert result.passed is False
    assert result.failure_type == "constraint_failure"
    assert result.score < 0.8


def test_good_output_passes_constraints():
    result = evaluate_output(GOOD_OUTPUT)

    assert result.passed is True
    assert result.score >= 0.8


def test_agent_loop_improves_second_run(tmp_path):
    memory_path = tmp_path / "rules.json"

    result = run_loop(memory_path=memory_path)

    run_1_score = result["run_1"]["evaluation"]["score"]
    run_2_score = result["run_2"]["evaluation"]["score"]

    assert run_2_score > run_1_score

    with open(memory_path, "r", encoding="utf-8") as file:
        rules = json.load(file)

    assert len(rules) >= 1
