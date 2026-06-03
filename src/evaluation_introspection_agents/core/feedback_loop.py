"""Feedback loop orchestration."""

from dataclasses import dataclass

from evaluation_introspection_agents.agents.critic import CriticAgent
from evaluation_introspection_agents.agents.evaluator import EvaluationResult, EvaluatorAgent
from evaluation_introspection_agents.agents.improver import ImproverAgent
from evaluation_introspection_agents.agents.introspector import IntrospectionResult, IntrospectorAgent
from evaluation_introspection_agents.core.task import Task
from evaluation_introspection_agents.core.trace import Trace


@dataclass(frozen=True)
class FeedbackLoopResult:
    """Complete result of one evaluation-introspection feedback loop."""

    task: Task
    output: str
    evaluation: EvaluationResult
    introspection: IntrospectionResult
    critiques: tuple[str, ...]
    improvement: str


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

    def run(self, task: Task, output: str, trace: Trace) -> FeedbackLoopResult:
        """Run one full feedback loop over a task, output, and trace."""
        evaluation = self.evaluator.evaluate(task, output)
        introspection = self.introspector.introspect(trace)
        critiques = self.critic.critique(evaluation, introspection)
        improvement = self.improver.improve(evaluation, critiques)
        return FeedbackLoopResult(
            task=task,
            output=output,
            evaluation=evaluation,
            introspection=introspection,
            critiques=critiques,
            improvement=improvement,
        )
