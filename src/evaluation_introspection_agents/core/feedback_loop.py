"""Feedback loop orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from evaluation_introspection_agents.agents.critic import CriticAgent
from evaluation_introspection_agents.agents.evaluator import EvaluationResult, EvaluatorAgent
from evaluation_introspection_agents.agents.improver import ImproverAgent
from evaluation_introspection_agents.agents.introspector import IntrospectionResult, IntrospectorAgent
from evaluation_introspection_agents.core.task import Task
from evaluation_introspection_agents.core.trace import BehaviorTrace


@dataclass(frozen=True)
class FeedbackLoopResult:
    """Complete result of one evaluation-introspection feedback loop."""

    task: Task
    output: str
    evaluation: EvaluationResult
    introspection: IntrospectionResult
    critiques: tuple[str, ...]
    improvement: str
    improved_output: str
    improvement_explanation: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the loop result to a JSON-safe dictionary."""
        data = asdict(self)
        data["task"] = {
            "objective": self.task.objective,
            "expected_terms": list(self.task.expected_terms),
        }
        data["evaluation"] = self.evaluation.to_dict() if hasattr(self.evaluation, "to_dict") else asdict(self.evaluation)
        data["introspection"] = self.introspection.to_dict()
        data["critiques"] = list(self.critiques)
        data["improvement_explanation"] = list(self.improvement_explanation)
        return data


class FeedbackLoop:
    """Connects evaluation, introspection, critique, and improvement."""

    def __init__(
        self,
        evaluator: EvaluatorAgent | None = None,
        introspector: IntrospectorAgent | None = None,
        critic: CriticAgent | None = None,
        improver: ImproverAgent | None = None,
    ) -> None:
        """Create a feedback loop from optional agent components."""
        self.evaluator = evaluator or EvaluatorAgent()
        self.introspector = introspector or IntrospectorAgent()
        self.critic = critic or CriticAgent()
        self.improver = improver or ImproverAgent()

    def run(
        self,
        task: Task,
        output: str,
        trace: BehaviorTrace,
        constraints: tuple[str, ...] = (),
    ) -> FeedbackLoopResult:
        """Run one full feedback loop over a task, output, trace, and constraints."""
        evaluation = self.evaluator.evaluate(task, output)
        introspection = self.introspector.introspect(trace)
        effective_constraints = constraints or task.expected_terms
        critiques = self.critic.critique(evaluation, introspection, output=output, constraints=effective_constraints)
        improvement = self.improver.improve(evaluation, critiques)
        improved_output = self.improver.improve_output(output, evaluation, critiques)
        improvement_explanation = self.improver.explain_improvements(evaluation, critiques)
        return FeedbackLoopResult(
            task=task,
            output=output,
            evaluation=evaluation,
            introspection=introspection,
            critiques=critiques,
            improvement=improvement,
            improved_output=improved_output,
            improvement_explanation=improvement_explanation,
        )
