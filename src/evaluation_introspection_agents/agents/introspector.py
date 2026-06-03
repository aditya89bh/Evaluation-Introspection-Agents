"""Trace introspection agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from evaluation_introspection_agents.core.trace import BehaviorTrace


@dataclass(frozen=True)
class IntrospectionResult:
    """Explanation derived from an execution trace."""

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
