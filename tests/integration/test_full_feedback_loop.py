"""Integration tests for the full feedback loop."""

from evaluation_introspection_agents.core.feedback_loop import FeedbackLoop
from evaluation_introspection_agents.core.task import Task
from evaluation_introspection_agents.core.trace import BehaviorTrace


def test_full_feedback_loop_connects_all_stages() -> None:
    """The loop should evaluate, introspect, critique, and improve deterministically."""
    task = Task("Mention rollback and owner", expected_terms=("rollback", "owner"))
    output = "Maybe launch it."
    trace = BehaviorTrace().add("draft", "Produced a launch note.", "Skipped safeguards.")

    result = FeedbackLoop().run(task, output, trace)

    assert result.evaluation.score == 0.0
    assert result.introspection.details == ("draft: Produced a launch note.",)
    assert any("Missing expected terms" in item for item in result.critiques)
    assert "rollback, owner" in result.improved_output
    assert result.to_dict()["evaluation"]["missing_terms"] == ["rollback", "owner"]
