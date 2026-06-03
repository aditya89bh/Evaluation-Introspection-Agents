"""Improvement agent for proposing better next actions and revised outputs."""

from __future__ import annotations

from evaluation_introspection_agents.agents.evaluator import EvaluationResult


class ImproverAgent:
    """Proposes deterministic improvements from evaluator and critic feedback."""

    def improve(self, evaluation: EvaluationResult, critiques: tuple[str, ...]) -> str:
        """Generate a deterministic improvement recommendation."""
        if evaluation.missing_terms:
            missing = ", ".join(evaluation.missing_terms)
            return f"Include the missing objective terms ({missing}) and remove irrelevant detail."
        if critiques and critiques != ("No major deterministic weaknesses were detected.",):
            return "Address the critique points before attempting the task again."
        return "Preserve the current strategy and collect more traces for future comparison."

    def improve_output(self, output: str, evaluation: EvaluationResult, critiques: tuple[str, ...]) -> str:
        """Produce a revised output by appending missing deterministic requirements.

        This is intentionally simple and reproducible: it does not rewrite style with an LLM;
        it adds a compact correction sentence when objective terms are missing.
        """
        improved = output.strip()
        additions: list[str] = []
        if evaluation.missing_terms:
            additions.append("Add coverage for: " + ", ".join(evaluation.missing_terms) + ".")
        if any("Vague statements" in critique for critique in critiques):
            additions.append("Replace vague wording with specific actions, owners, and checks.")
        if any("risk" in critique.lower() for critique in critiques):
            additions.append("State a mitigation for each identified risk.")
        if not additions:
            return improved
        separator = "\n" if improved else ""
        return improved + separator + "Improved next draft: " + " ".join(additions)

    def explain_improvements(self, evaluation: EvaluationResult, critiques: tuple[str, ...]) -> tuple[str, ...]:
        """Explain which feedback items shaped the improved output."""
        explanations: list[str] = []
        if evaluation.missing_terms:
            explanations.append("Added missing objective coverage.")
        if any("Vague statements" in critique for critique in critiques):
            explanations.append("Replaced vague language with concrete guidance.")
        if any("risk" in critique.lower() for critique in critiques):
            explanations.append("Added risk mitigation guidance.")
        if not explanations:
            explanations.append("No change required by deterministic feedback.")
        return tuple(explanations)
