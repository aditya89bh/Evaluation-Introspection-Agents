"""Demo 5: run the complete feedback loop."""

from evaluation_introspection_agents.core.feedback_loop import FeedbackLoop
from evaluation_introspection_agents.core.task import Task
from evaluation_introspection_agents.core.trace import BehaviorTrace


def main() -> None:
    """Run a deterministic full feedback loop demo."""
    task = Task("Create a safe robotics action plan", expected_terms=("precondition", "rollback", "operator"))
    output = "Move the robot and maybe continue if some things look okay."
    trace = (
        BehaviorTrace()
        .add("parse", "Read the robotics planning objective.", "Detected safety-sensitive task.")
        .add("draft", "Generated a short action plan.", "Skipped preconditions and ownership.")
        .add("review", "Checked the plan against expected terms.", "Found missing safety details.")
    )
    result = FeedbackLoop().run(task, output, trace)

    print(f"Task: {result.task.objective}")
    print(f"Score: {result.evaluation.score:.2f}")
    print(f"Introspection: {result.introspection.summary}")
    print("Critique:")
    for critique in result.critiques:
        print(f"- {critique}")
    print(f"Improvement: {result.improvement}")
    print(f"Improved output: {result.improved_output}")


if __name__ == "__main__":
    main()
