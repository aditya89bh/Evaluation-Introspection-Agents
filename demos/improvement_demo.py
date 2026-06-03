"""Demo 4: improve an output using evaluation and critique feedback."""

from evaluation_introspection_agents.agents.critic import CriticAgent
from evaluation_introspection_agents.agents.evaluator import EvaluatorAgent
from evaluation_introspection_agents.agents.improver import ImproverAgent
from evaluation_introspection_agents.agents.introspector import IntrospectorAgent
from evaluation_introspection_agents.core.task import Task
from evaluation_introspection_agents.core.trace import BehaviorTrace


def main() -> None:
    """Run the improvement demo."""
    task = Task("Create a deployment checklist", expected_terms=("rollback", "monitoring", "owner"))
    output = "Maybe deploy it and watch for some things."
    trace = BehaviorTrace().add("draft", "Created a short checklist.", "Skipped ownership and rollback.")
    evaluation = EvaluatorAgent().evaluate(task, output)
    introspection = IntrospectorAgent().introspect(trace)
    critiques = CriticAgent().critique(evaluation, introspection, output, task.expected_terms)
    improver = ImproverAgent()

    print("Recommendation:")
    print(improver.improve(evaluation, critiques))
    print("Improved output:")
    print(improver.improve_output(output, evaluation, critiques))
    print("Explanation:")
    for item in improver.explain_improvements(evaluation, critiques):
        print(f"- {item}")


if __name__ == "__main__":
    main()
