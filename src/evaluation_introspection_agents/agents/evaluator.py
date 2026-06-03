"""Deterministic output evaluation agent."""

from evaluation_introspection_agents.core.results import EvaluationResult
from evaluation_introspection_agents.core.task import Task


class EvaluatorAgent:
    """Scores task outputs using simple expected-term coverage."""

    def evaluate(self, task: Task, output: str) -> EvaluationResult:
        """Evaluate output against the task's expected terms.

        The first version is intentionally deterministic and transparent:
        score = matched expected terms / total expected terms.
        """
        terms = task.normalized_terms()
        text = output.lower()
        if not terms:
            return EvaluationResult(
                score=1.0,
                matched_terms=(),
                missing_terms=(),
                rationale="No expected terms were provided, so the output is accepted by default.",
            )

        matched = tuple(term for term in terms if term in text)
        missing = tuple(term for term in terms if term not in text)
        score = round(len(matched) / len(terms), 2)
        rationale = f"Matched {len(matched)} of {len(terms)} expected objective terms."
        return EvaluationResult(score=score, matched_terms=matched, missing_terms=missing, rationale=rationale)
