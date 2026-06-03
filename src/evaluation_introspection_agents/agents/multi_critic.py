"""Deterministic multi-critic evaluation and consensus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from evaluation_introspection_agents.core.results import EvaluationResult, IntrospectionResult


class CriticProtocol(Protocol):
    """Protocol for deterministic critic implementations."""

    name: str

    def critique(
        self,
        evaluation: EvaluationResult,
        introspection: IntrospectionResult,
        output: str = "",
        constraints: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Return critique findings."""


class CriticA:
    """Objective coverage critic."""

    name = "CriticA"

    def critique(self, evaluation: EvaluationResult, introspection: IntrospectionResult, output: str = "", constraints: tuple[str, ...] = ()) -> tuple[str, ...]:
        """Identify missing objective terms."""
        if evaluation.missing_terms:
            return (f"Missing expected terms: {', '.join(evaluation.missing_terms)}.",)
        return ("Objective coverage is complete.",)


class CriticB:
    """Specificity and constraint critic."""

    name = "CriticB"
    vague_terms = ("some", "things", "maybe", "probably", "stuff")

    def critique(self, evaluation: EvaluationResult, introspection: IntrospectionResult, output: str = "", constraints: tuple[str, ...] = ()) -> tuple[str, ...]:
        """Identify vague wording and missing constraints."""
        lowered = output.lower()
        findings: list[str] = []
        missing_constraints = tuple(term for term in constraints if term.lower() not in lowered)
        if missing_constraints:
            findings.append(f"Missing constraints: {', '.join(missing_constraints)}.")
        vague = tuple(term for term in self.vague_terms if term in lowered.split())
        if vague:
            findings.append(f"Vague statements detected: {', '.join(vague)}.")
        return tuple(findings or ("Specificity is acceptable.",))


class CriticC:
    """Risk and trace inspectability critic."""

    name = "CriticC"
    risk_terms = ("risk", "unsafe", "security", "privacy", "danger")

    def critique(self, evaluation: EvaluationResult, introspection: IntrospectionResult, output: str = "", constraints: tuple[str, ...] = ()) -> tuple[str, ...]:
        """Identify risks and trace gaps."""
        lowered = output.lower()
        findings: list[str] = []
        risks = tuple(term for term in self.risk_terms if term in lowered)
        if risks:
            findings.append(f"Potential risks mentioned without mitigation: {', '.join(risks)}.")
        if not introspection.details:
            findings.append("No trace details were available, making the behavior hard to inspect.")
        if evaluation.score < 0.5:
            findings.append("Failure mode: low objective coverage may cause task failure.")
        return tuple(findings or ("Risk posture is acceptable.",))


@dataclass(frozen=True)
class MultiCriticResult:
    """Consensus result across multiple deterministic critics."""

    critic_findings: dict[str, tuple[str, ...]]
    consensus: tuple[str, ...]


class MultiCriticEvaluator:
    """Run a configurable list of critics and compute deterministic consensus."""

    def __init__(self, critics: tuple[CriticProtocol, ...] | None = None) -> None:
        """Create a multi-critic evaluator."""
        self.critics = critics or (CriticA(), CriticB(), CriticC())

    def evaluate(
        self,
        evaluation: EvaluationResult,
        introspection: IntrospectionResult,
        output: str = "",
        constraints: tuple[str, ...] = (),
        min_votes: int = 1,
    ) -> MultiCriticResult:
        """Run critics and return findings with vote-based consensus."""
        critic_findings = {
            critic.name: critic.critique(evaluation, introspection, output, constraints)
            for critic in self.critics
        }
        counts: dict[str, int] = {}
        for findings in critic_findings.values():
            for finding in findings:
                if finding.endswith("acceptable.") or finding.endswith("complete."):
                    continue
                counts[finding] = counts.get(finding, 0) + 1
        consensus = tuple(finding for finding, count in sorted(counts.items()) if count >= min_votes)
        return MultiCriticResult(critic_findings=critic_findings, consensus=consensus)
