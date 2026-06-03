"""Task model used by the feedback loop."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    """A task objective and deterministic success terms.

    Attributes:
        objective: Human-readable description of the desired result.
        expected_terms: Terms that should appear in a successful output.
    """

    objective: str
    expected_terms: tuple[str, ...] = field(default_factory=tuple)

    def normalized_terms(self) -> tuple[str, ...]:
        """Return lowercase expected terms for deterministic matching."""
        return tuple(term.lower().strip() for term in self.expected_terms if term.strip())
