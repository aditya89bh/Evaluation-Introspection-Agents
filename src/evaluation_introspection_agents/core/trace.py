"""Trace structures for explaining an agent run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TraceStep:
    """One observed step in an agent's execution trace.

    Attributes:
        name: Stable identifier for the step.
        detail: Human-readable detail about what happened.
        observation: Optional debugging observation captured at the step.
    """

    name: str
    detail: str
    observation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace step to a JSON-safe dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class BehaviorTrace:
    """A deterministic sequence of steps that produced an output."""

    steps: tuple[TraceStep, ...] = field(default_factory=tuple)

    def add(self, name: str, detail: str, observation: str = "") -> "BehaviorTrace":
        """Return a new trace with one additional step."""
        return BehaviorTrace(steps=(*self.steps, TraceStep(name=name, detail=detail, observation=observation)))

    def summary(self) -> str:
        """Summarize the trace in a stable, human-readable form."""
        if not self.steps:
            return "No trace steps were recorded."
        names = ", ".join(step.name for step in self.steps)
        return f"The trace shows {len(self.steps)} recorded steps: {names}."

    def explain(self) -> str:
        """Generate a step-by-step reasoning trace for debugging."""
        if not self.steps:
            return "No behavior trace is available for inspection."
        lines = ["Behavior trace:"]
        for index, step in enumerate(self.steps, start=1):
            line = f"{index}. {step.name}: {step.detail}"
            if step.observation:
                line += f" Observation: {step.observation}"
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the trace to a JSON-safe dictionary."""
        return {"steps": [step.to_dict() for step in self.steps]}


Trace = BehaviorTrace
