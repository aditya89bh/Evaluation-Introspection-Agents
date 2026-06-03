"""Simple deterministic reflection demo."""

from evaluation_introspection_agents.core.feedback_loop import FeedbackLoop
from evaluation_introspection_agents.core.task import Task
from evaluation_introspection_agents.core.trace import Trace


def main() -> None:
    """Run a small feedback loop demo and print the report."""
    task = Task(
        objective="Write a concise project summary",
        expected_terms=("concise", "project", "summary"),
    )
    output = "This summary explains why agents need feedback loops."
    trace = (
        Trace()
        .add("parse_objective", "Identified that the user wanted a short project summary.")
        .add("draft_output", "Generated a one-sentence response.")
        .add("finalize", "Returned the draft without checking all expected terms.")
    )

    result = FeedbackLoop().run(task=task, output=output, trace=trace)

    print(f"Task: {result.task.objective}")
    print(f"Output score: {result.evaluation.score:.2f}")
    print(f"Introspection: {result.introspection.summary}")
    print(f"Critique: {' '.join(result.critiques)}")
    print(f"Improvement: {result.improvement}")


if __name__ == "__main__":
    main()
