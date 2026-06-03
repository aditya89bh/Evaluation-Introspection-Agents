"""Trace structures for explaining an agent run."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TraceStep:
    """One observed step in an agent's execution trace."""

    name: str
    detail: str


@dataclass(frozen=True)
class Trace:
    """A sequence of steps that describe how an output was produced."""

    steps: tuple[TraceStep, ...] = field(default_factory=tuple)

    def add(self, name: str, detail: str) -> "Trace":
        """Return a new trace with one additional step."""
        return Trace(steps=(*self.steps, TraceStep(name=name, detail=detail)))

    def summary(self) -> str:
        """Summarize the trace in a stable, human-readable form."""
        if not self.steps:
            return "No trace steps were recorded."
        names = ", ".join(step.name for step in self.steps)
        return f"The trace shows {len(self.steps)} recorded steps: {names}."
