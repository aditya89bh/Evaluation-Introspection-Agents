"""Improvement agent for proposing better next actions."""

from evaluation_introspection_agents.agents.evaluator import EvaluationResult


class ImproverAgent:
    """Proposes the next action that should improve future behavior."""

    def improve(self, evaluation: EvaluationResult, critiques: tuple[str, ...]) -> str:
        """Generate a deterministic improvement recommendation."""
        if evaluation.missing_terms:
            missing = ", ".join(evaluation.missing_terms)
            return f"Include the missing objective terms ({missing}) and remove irrelevant detail."
        if critiques and critiques != ("No major deterministic weaknesses were detected.",):
            return "Address the critique points before attempting the task again."
        return "Preserve the current strategy and collect more traces for future comparison."
