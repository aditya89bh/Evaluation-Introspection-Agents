"""Trace introspection agent."""

from dataclasses import dataclass

from evaluation_introspection_agents.core.trace import Trace


@dataclass(frozen=True)
class IntrospectionResult:
    """Explanation derived from an execution trace."""

    summary: str
    details: tuple[str, ...]


class IntrospectorAgent:
    """Explains what happened during execution using trace steps."""

    def introspect(self, trace: Trace) -> IntrospectionResult:
        """Produce a deterministic explanation from a trace."""
        details = tuple(f"{step.name}: {step.detail}" for step in trace.steps)
        return IntrospectionResult(summary=trace.summary(), details=details)
