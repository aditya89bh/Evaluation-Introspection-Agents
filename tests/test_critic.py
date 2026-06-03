"""Tests for deterministic critique analysis."""

from evaluation_introspection_agents.agents.critic import CriticAgent
from evaluation_introspection_agents.agents.evaluator import EvaluationResult
from evaluation_introspection_agents.agents.introspector import IntrospectionResult


def test_critic_detects_vague_missing_risky_failure_modes() -> None:
    """The critic should identify weaknesses, missing constraints, and risks."""
    evaluation = EvaluationResult(
        score=0.33,
        matched_terms=("plan",),
        missing_terms=("rollback", "owner"),
        rationale="Matched 1 of 3 terms.",
    )
    introspection = IntrospectionResult(
        summary="One draft step.",
        details=("draft: Wrote a vague plan.",),
        reasoning_trace="Behavior trace:\n1. draft: Wrote a vague plan.",
        debug_notes=(),
    )

    critiques = CriticAgent().critique(
        evaluation,
        introspection,
        output="Maybe do some stuff. This has security risk.",
        constraints=("rollback", "monitoring"),
    )

    assert "Missing expected terms: rollback, owner." in critiques
    assert "Missing constraints: rollback, monitoring." in critiques
    assert "Vague statements detected: some, stuff, maybe." in critiques
    assert "Potential risks mentioned without mitigation: risk, security." in critiques
    assert "Failure mode: low objective coverage may cause task failure." in critiques


def test_critic_accepts_specific_complete_output() -> None:
    """The critic should avoid false positives for complete specific output."""
    evaluation = EvaluationResult(1.0, ("rollback",), (), "Matched all terms.")
    introspection = IntrospectionResult("One step.", ("draft: Good.",), "Behavior trace", ())

    critiques = CriticAgent().critique(
        evaluation,
        introspection,
        output="Rollback owner is assigned with monitoring.",
        constraints=("rollback", "monitoring"),
    )

    assert critiques == ("No major deterministic weaknesses were detected.",)


def test_critic_structured_findings_include_scores() -> None:
    """Structured critique results should include confidence, severity, and importance."""
    evaluation = EvaluationResult(0.0, (), ("rollback",), "Missing rollback.")
    introspection = IntrospectionResult("Trace", ("draft: x",), "trace", ())

    result = CriticAgent().analyze(evaluation, introspection, output="Maybe risk.", constraints=("rollback",))

    assert result.findings
    assert all(0.0 <= finding.confidence <= 1.0 for finding in result.findings)
    assert {finding.severity for finding in result.findings} <= {"info", "medium", "high"}
    assert result.to_dict()["findings"][0]["importance"] in {"low", "medium", "high"}
