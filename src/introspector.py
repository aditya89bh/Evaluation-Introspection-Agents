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

    if "Expected" in reason:
        root_cause = (
            "The agent prioritized content completeness before checking output constraints."
        )
        rule = (
            "Before finalizing an answer, verify bullet count, structure, and formatting constraints."
        )
        trigger = "tasks with explicit formatting requirements"
    elif "Found jargon" in reason:
        root_cause = (
            "The agent optimized for technical precision instead of audience readability."
        )
        rule = (
            "Replace technical jargon with simpler language when readability is requested."
        )
        trigger = "tasks requesting simplified communication"
    else:
        root_cause = "The system could not classify the failure pattern precisely."
        rule = "Perform explicit verification before final output generation."
        trigger = "general task execution"

    return IntrospectionResult(
        failure_summary=reason,
        root_cause=root_cause,
        behavior_rule=rule,
        future_trigger=trigger,
    )
