"""Simple deterministic evaluator for agent outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationResult:
    score: float
    passed: bool
    failure_type: Optional[str]
    failure_reason: Optional[str]

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "passed": self.passed,
            "failure_type": self.failure_type,
            "failure_reason": self.failure_reason,
        }


def evaluate_output(output: str, required_bullets: int = 5, avoid_jargon: bool = True) -> EvaluationResult:
    """Evaluate a simple constrained-summary output.

    This evaluator intentionally stays deterministic so the repo has a clean,
    reproducible baseline before adding LLM-based critics.
    """
    bullet_lines = [line for line in output.splitlines() if line.strip().startswith(("-", "*", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."))]

    jargon_terms = {
        "architecture",
        "introspection pipeline",
        "constraint adherence",
        "behavioral embeddings",
        "episodic critique",
        "retry strategies",
    }

    bullet_count_ok = len(bullet_lines) == required_bullets
    jargon_found = [term for term in jargon_terms if term.lower() in output.lower()]
    jargon_ok = not jargon_found if avoid_jargon else True

    score = 1.0
    failure_reasons = []

    if not bullet_count_ok:
        score -= 0.35
        failure_reasons.append(f"Expected {required_bullets} bullets, got {len(bullet_lines)}")

    if not jargon_ok:
        score -= 0.25
        failure_reasons.append("Used technical jargon")

    score = max(score, 0.0)
    passed = score >= 0.8

    if passed:
        return EvaluationResult(score=score, passed=True, failure_type=None, failure_reason=None)

    return EvaluationResult(
        score=score,
        passed=False,
        failure_type="constraint_failure",
        failure_reason="; ".join(failure_reasons),
    )
