"""Critique agent for identifying weaknesses and risks."""

from evaluation_introspection_agents.agents.evaluator import EvaluationResult
from evaluation_introspection_agents.agents.introspector import IntrospectionResult


class CriticAgent:
    """Identifies weaknesses, risks, and likely failure modes."""

    def critique(self, evaluation: EvaluationResult, introspection: IntrospectionResult) -> tuple[str, ...]:
        """Return deterministic critique points from evaluation and introspection."""
        critiques: list[str] = []
        if evaluation.missing_terms:
            missing = ", ".join(evaluation.missing_terms)
            critiques.append(f"Missing expected terms: {missing}.")
        if evaluation.score < 0.5:
            critiques.append("Low objective coverage creates a high risk of task failure.")
        if not introspection.details:
            critiques.append("No trace details were available, making the behavior hard to inspect.")
        if not critiques:
            critiques.append("No major deterministic weaknesses were detected.")
        return tuple(critiques)
