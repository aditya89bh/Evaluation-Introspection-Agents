"""Critique agent for identifying weaknesses and risks."""

from __future__ import annotations

import re

from evaluation_introspection_agents.core.results import CritiqueFinding, CritiqueResult, EvaluationResult, IntrospectionResult


class CriticAgent:
    """Identifies weaknesses, risks, and likely failure modes deterministically."""

    vague_terms = ("some", "various", "things", "stuff", "maybe", "probably", "generally")
    risk_terms = ("unsafe", "risk", "harm", "failure", "danger", "privacy", "security")

    def analyze(
        self,
        evaluation: EvaluationResult,
        introspection: IntrospectionResult,
        output: str = "",
        constraints: tuple[str, ...] = (),
    ) -> CritiqueResult:
        """Return a structured critique result from feedback, output, and constraints."""
        lowered_output = output.lower()
        output_tokens = set(re.findall(r"[a-z]+", lowered_output))

        weaknesses: list[str] = []
        if evaluation.missing_terms:
            missing = ", ".join(evaluation.missing_terms)
            weaknesses.append(f"Missing expected terms: {missing}.")
        if not introspection.details:
            weaknesses.append("No trace details were available, making the behavior hard to inspect.")

        missing_constraints = tuple(
            constraint for constraint in constraints if constraint.lower() not in lowered_output
        )
        vague_found = tuple(term for term in self.vague_terms if term in output_tokens)
        risk_found = tuple(term for term in self.risk_terms if term in lowered_output)

        failure_modes: list[str] = []
        if evaluation.score < 0.5:
            failure_modes.append("Failure mode: low objective coverage may cause task failure.")
        elif evaluation.score < 0.8:
            failure_modes.append("Failure mode: partial coverage may satisfy only part of the task.")

        result = CritiqueResult(
            weaknesses=tuple(weaknesses),
            vague_statements=vague_found,
            missing_constraints=missing_constraints,
            risks=risk_found,
            failure_modes=tuple(failure_modes),
        )
        return CritiqueResult(
            weaknesses=result.weaknesses,
            vague_statements=result.vague_statements,
            missing_constraints=result.missing_constraints,
            risks=result.risks,
            failure_modes=result.failure_modes,
            findings=self.score_findings(result.all_findings()),
        )

    def score_findings(self, findings: tuple[str, ...]) -> tuple[CritiqueFinding, ...]:
        """Assign deterministic confidence, severity, and importance to critiques."""
        scored: list[CritiqueFinding] = []
        for finding in findings:
            lowered = finding.lower()
            if "no major" in lowered:
                scored.append(CritiqueFinding(finding, confidence=0.7, severity="info", importance="low"))
            elif "risk" in lowered or "failure mode" in lowered:
                scored.append(CritiqueFinding(finding, confidence=0.9, severity="high", importance="high"))
            elif "missing" in lowered:
                scored.append(CritiqueFinding(finding, confidence=0.95, severity="medium", importance="high"))
            else:
                scored.append(CritiqueFinding(finding, confidence=0.8, severity="medium", importance="medium"))
        return tuple(scored)

    def critique(
        self,
        evaluation: EvaluationResult,
        introspection: IntrospectionResult,
        output: str = "",
        constraints: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Return deterministic critique points from feedback, output, and constraints."""
        return self.analyze(evaluation, introspection, output, constraints).all_findings()
