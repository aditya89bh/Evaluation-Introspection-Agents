"""Structured result models for deterministic agent outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationResult:
    """Structured score and coverage result from the evaluator."""

    score: float
    matched_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the evaluation result to a JSON-safe dictionary."""
        data = asdict(self)
        data["matched_terms"] = list(self.matched_terms)
        data["missing_terms"] = list(self.missing_terms)
        return data


@dataclass(frozen=True)
class IntrospectionResult:
    """Structured explanation derived from a behavior trace."""

    summary: str
    details: tuple[str, ...]
    reasoning_trace: str
    debug_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the introspection result to a JSON-safe dictionary."""
        data = asdict(self)
        data["details"] = list(self.details)
        data["debug_notes"] = list(self.debug_notes)
        return data

@dataclass(frozen=True)
class CritiqueFinding:
    """One structured critique with confidence, severity, and importance."""

    message: str
    confidence: float
    severity: str
    importance: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the critique finding to a JSON-safe dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class CritiqueResult:
    """Structured critique findings from the critic."""

    weaknesses: tuple[str, ...]
    vague_statements: tuple[str, ...]
    missing_constraints: tuple[str, ...]
    risks: tuple[str, ...]
    failure_modes: tuple[str, ...]
    findings: tuple[CritiqueFinding, ...] = ()

    def all_findings(self) -> tuple[str, ...]:
        """Return all critique findings as readable strings."""
        findings: list[str] = []
        findings.extend(self.weaknesses)
        if self.missing_constraints:
            findings.append(f"Missing constraints: {', '.join(self.missing_constraints)}.")
        if self.vague_statements:
            findings.append(f"Vague statements detected: {', '.join(self.vague_statements)}.")
        if self.risks:
            findings.append(f"Potential risks mentioned without mitigation: {', '.join(self.risks)}.")
        findings.extend(self.failure_modes)
        if not findings:
            findings.append("No major deterministic weaknesses were detected.")
        return tuple(findings)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the critique result to a JSON-safe dictionary."""
        return {
            "weaknesses": list(self.weaknesses),
            "vague_statements": list(self.vague_statements),
            "missing_constraints": list(self.missing_constraints),
            "risks": list(self.risks),
            "failure_modes": list(self.failure_modes),
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class ImprovementResult:
    """Structured improvement recommendation and revised output."""

    recommendation: str
    improved_output: str
    explanations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the improvement result to a JSON-safe dictionary."""
        data = asdict(self)
        data["explanations"] = list(self.explanations)
        return data
