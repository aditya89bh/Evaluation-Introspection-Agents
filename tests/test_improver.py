"""Tests for deterministic output improvement."""

from evaluation_introspection_agents.agents.evaluator import EvaluationResult
from evaluation_introspection_agents.agents.improver import ImproverAgent


def test_improver_appends_missing_terms_and_explains_changes() -> None:
    """The improver should consume evaluator and critic feedback."""
    evaluation = EvaluationResult(0.33, ("plan",), ("rollback", "owner"), "Missing terms.")
    critiques = (
        "Missing expected terms: rollback, owner.",
        "Vague statements detected: maybe, some.",
        "Potential risks mentioned without mitigation: risk.",
    )

    improver = ImproverAgent()
    improved = improver.improve_output("Maybe launch it.", evaluation, critiques)
    explanations = improver.explain_improvements(evaluation, critiques)

    assert "rollback, owner" in improved
    assert "specific actions" in improved
    assert "mitigation" in improved
    assert "Added missing objective coverage." in explanations
    assert "Replaced vague language with concrete guidance." in explanations
    assert "Added risk mitigation guidance." in explanations


def test_improver_preserves_complete_output() -> None:
    """The improver should leave complete outputs unchanged."""
    evaluation = EvaluationResult(1.0, ("rollback",), (), "Complete.")
    critiques = ("No major deterministic weaknesses were detected.",)

    assert ImproverAgent().improve_output("Use rollback monitoring.", evaluation, critiques) == "Use rollback monitoring."
