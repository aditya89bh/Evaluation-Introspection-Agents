"""Tests for structured result models."""

from evaluation_introspection_agents.core.results import (
    CritiqueFinding,
    CritiqueResult,
    EvaluationResult,
    ImprovementResult,
    IntrospectionResult,
)


def test_result_models_serialize_to_stable_dicts() -> None:
    """Structured results should serialize tuple fields as JSON-safe lists."""
    evaluation = EvaluationResult(0.5, ("a",), ("b",), "Half matched.")
    introspection = IntrospectionResult("Summary", ("step: detail",), "trace", ("debug",))
    finding = CritiqueFinding("weak", 0.9, "high", "high")
    critique = CritiqueResult(("weak",), ("maybe",), ("owner",), ("risk",), ("failure",), (finding,))
    improvement = ImprovementResult("Do better", "Better output", ("Added owner",))

    assert evaluation.to_dict()["missing_terms"] == ["b"]
    assert introspection.to_dict()["details"] == ["step: detail"]
    assert critique.to_dict()["risks"] == ["risk"]
    assert critique.to_dict()["findings"] == [{"message": "weak", "confidence": 0.9, "severity": "high", "importance": "high"}]
    assert "Missing constraints: owner." in critique.all_findings()
    assert improvement.to_dict()["explanations"] == ["Added owner"]
