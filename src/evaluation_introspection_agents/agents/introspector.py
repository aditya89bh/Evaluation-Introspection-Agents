"""Trace introspection agent."""

from evaluation_introspection_agents.core.results import IntrospectionResult
from evaluation_introspection_agents.core.trace import BehaviorTrace


class IntrospectorAgent:
    """Explains what happened during execution using trace steps."""

    def introspect(self, trace: BehaviorTrace) -> IntrospectionResult:
        """Produce a deterministic explanation from a behavior trace."""
        details = tuple(f"{step.name}: {step.detail}" for step in trace.steps)
        debug_notes = tuple(
            f"{step.name}: {step.observation}" for step in trace.steps if step.observation
        )
        return IntrospectionResult(
            summary=trace.summary(),
            details=details,
            reasoning_trace=trace.explain(),
            debug_notes=debug_notes,
        )
