"""Tests for multi-critic consensus evaluation."""

from evaluation_introspection_agents.agents.multi_critic import CriticA, MultiCriticEvaluator
from evaluation_introspection_agents.core.results import EvaluationResult, IntrospectionResult


def test_multi_critic_evaluator_returns_configurable_consensus() -> None:
    """Multi-critic evaluation should expose per-critic findings and consensus."""
    evaluation = EvaluationResult(0.0, (), ("rollback",), "Missing rollback.")
    introspection = IntrospectionResult("No trace", (), "No trace", ())

    result = MultiCriticEvaluator().evaluate(
        evaluation,
        introspection,
        output="Maybe launch with security risk.",
        constraints=("rollback",),
    )

    assert set(result.critic_findings) == {"CriticA", "CriticB", "CriticC"}
    assert "Missing expected terms: rollback." in result.consensus
    assert "Failure mode: low objective coverage may cause task failure." in result.consensus


def test_multi_critic_accepts_configured_critic_list() -> None:
    """Callers should be able to configure the critic list."""
    evaluation = EvaluationResult(0.0, (), ("owner",), "Missing owner.")
    introspection = IntrospectionResult("Trace", ("draft: x",), "trace", ())

    result = MultiCriticEvaluator(critics=(CriticA(),)).evaluate(evaluation, introspection)

    assert tuple(result.critic_findings) == ("CriticA",)
    assert result.consensus == ("Missing expected terms: owner.",)
