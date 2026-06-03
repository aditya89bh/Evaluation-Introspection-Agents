"""Demo 3: critique a weak output."""

from evaluation_introspection_agents.agents.critic import CriticAgent
from evaluation_introspection_agents.agents.evaluator import EvaluatorAgent
from evaluation_introspection_agents.agents.introspector import IntrospectorAgent
from evaluation_introspection_agents.core.task import Task
from evaluation_introspection_agents.core.trace import BehaviorTrace


def main() -> None:
    """Run the critic demo."""
    task = Task("Create a safe deployment plan", expected_terms=("rollback", "monitoring", "owner"))
    output = "We can do some deployment things later. There may be risk."
    trace = BehaviorTrace().add("draft", "Generated a deployment answer.", "Skipped concrete safeguards.")
    evaluation = EvaluatorAgent().evaluate(task, output)
    introspection = IntrospectorAgent().introspect(trace)
    critiques = CriticAgent().critique(
        evaluation,
        introspection,
        output=output,
        constraints=("rollback", "monitoring", "owner"),
    )

    print("Critique:")
    for critique in critiques:
        print(f"- {critique}")


if __name__ == "__main__":
    main()
