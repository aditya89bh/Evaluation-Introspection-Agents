"""Demo 2: inspect a behavior trace step by step."""

from evaluation_introspection_agents.agents.introspector import IntrospectorAgent
from evaluation_introspection_agents.core.trace import BehaviorTrace


def main() -> None:
    """Run the introspection trace demo."""
    trace = (
        BehaviorTrace()
        .add("read_task", "Parsed the objective and constraints.", "Found a concise summary request.")
        .add("draft_output", "Generated a first answer.", "Answer included the topic but missed one constraint.")
        .add("final_check", "Checked expected terms before returning.", "Detected incomplete coverage.")
    )
    result = IntrospectorAgent().introspect(trace)
    print(result.summary)
    print(result.reasoning_trace)
    if result.debug_notes:
        print("Debug notes:")
        for note in result.debug_notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
