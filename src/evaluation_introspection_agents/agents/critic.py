"""Critique agent for identifying weaknesses and risks."""

from __future__ import annotations

import re

from evaluation_introspection_agents.agents.evaluator import EvaluationResult
from evaluation_introspection_agents.agents.introspector import IntrospectionResult


class CriticAgent:
    """Identifies weaknesses, risks, and likely failure modes deterministically."""

    vague_terms = ("some", "various", "things", "stuff", "maybe", "probably", "generally")
    risk_terms = ("unsafe", "risk", "harm", "failure", "danger", "privacy", "security")

    def critique(
        self,
        evaluation: EvaluationResult,
        introspection: IntrospectionResult,
        output: str = "",
        constraints: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Return deterministic critique points from feedback, output, and constraints."""
        critiques: list[str] = []
        lowered_output = output.lower()
        output_tokens = set(re.findall(r"[a-z]+", lowered_output))

        if evaluation.missing_terms:
            missing = ", ".join(evaluation.missing_terms)
            critiques.append(f"Missing expected terms: {missing}.")

        missing_constraints = tuple(
            constraint for constraint in constraints if constraint.lower() not in lowered_output
        )
        if missing_constraints:
            critiques.append(f"Missing constraints: {', '.join(missing_constraints)}.")

        vague_found = tuple(term for term in self.vague_terms if term in output_tokens)
        if vague_found:
            critiques.append(f"Vague statements detected: {', '.join(vague_found)}.")

        risk_found = tuple(term for term in self.risk_terms if term in lowered_output)
        if risk_found:
            critiques.append(f"Potential risks mentioned without mitigation: {', '.join(risk_found)}.")

        if evaluation.score < 0.5:
            critiques.append("Failure mode: low objective coverage may cause task failure.")
        elif evaluation.score < 0.8:
            critiques.append("Failure mode: partial coverage may satisfy only part of the task.")

        if not introspection.details:
            critiques.append("No trace details were available, making the behavior hard to inspect.")

        if not critiques:
            critiques.append("No major deterministic weaknesses were detected.")
        return tuple(critiques)
