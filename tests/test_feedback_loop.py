"""Tests for the deterministic feedback loop."""

from evaluation_introspection_agents.core.feedback_loop import FeedbackLoop
from evaluation_introspection_agents.core.task import Task
from evaluation_introspection_agents.core.trace import Trace


def test_feedback_loop_identifies_missing_terms() -> None:
    """The loop should score outputs and recommend missing objective terms."""
    task = Task(objective="Mention alpha and beta", expected_terms=("alpha", "beta"))
    trace = Trace().add("draft", "The agent wrote only one expected term.")

    result = FeedbackLoop().run(task=task, output="alpha is present", trace=trace)

    assert result.evaluation.score == 0.5
    assert result.evaluation.matched_terms == ("alpha",)
    assert result.evaluation.missing_terms == ("beta",)
    assert "Missing expected terms: beta." in result.critiques
    assert "beta" in result.improvement


def test_feedback_loop_accepts_complete_output() -> None:
    """The loop should avoid false critiques when all terms are present."""
    task = Task(objective="Mention alpha and beta", expected_terms=("alpha", "beta"))
    trace = Trace().add("draft", "The agent included all expected terms.")

    result = FeedbackLoop().run(task=task, output="alpha and beta are present", trace=trace)

    assert result.evaluation.score == 1.0
    assert result.evaluation.missing_terms == ()
    assert result.critiques == ("No major deterministic weaknesses were detected.",)
    assert result.improvement.startswith("Preserve the current strategy")
