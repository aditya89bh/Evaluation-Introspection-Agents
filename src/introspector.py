"""Convert evaluation failures into reusable behavioral rules."""

from __future__ import annotations

from dataclasses import dataclass

from evaluator import EvaluationResult


@dataclass
class IntrospectionResult:
    failure_summary: str
    root_cause: str
    behavior_rule: str
    future_trigger: str

    def to_dict(self) -> dict:
        return {
            "failure_summary": self.failure_summary,
            "root_cause": self.root_cause,
            "behavior_rule": self.behavior_rule,
            "future_trigger": self.future_trigger,
        }


def introspect(evaluation: EvaluationResult) -> IntrospectionResult:
    """Generate a lightweight introspection result from evaluator feedback."""

    if evaluation.passed:
        return IntrospectionResult(
            failure_summary="No failure detected.",
            root_cause="No corrective action required.",
            behavior_rule="Maintain current behavior.",
            future_trigger="successful tasks",
        )

    reason = evaluation.failure_reason or "Unknown failure"

    root_cause = (
        "The agent focused on content generation before verifying task constraints."
    )

    rule = (
        "Check bullet count and readability before finalizing the answer."
    )

    trigger = "tasks with explicit formatting or readability instructions"

    return IntrospectionResult(
        failure_summary=reason,
        root_cause=root_cause,
        behavior_rule=rule,
        future_trigger=trigger,
    )
