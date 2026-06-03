"""Tests for trace introspection."""

from evaluation_introspection_agents.agents.introspector import IntrospectorAgent
from evaluation_introspection_agents.core.trace import BehaviorTrace, TraceStep


def test_behavior_trace_records_steps_and_explains_them() -> None:
    """A behavior trace should produce readable step-by-step reasoning."""
    trace = BehaviorTrace().add("plan", "Chose a response shape.", "Need five bullets.")

    assert trace.steps == (TraceStep("plan", "Chose a response shape.", "Need five bullets."),)
    assert "1. plan: Chose a response shape." in trace.explain()
    assert trace.to_dict()["steps"][0]["observation"] == "Need five bullets."


def test_introspector_generates_debuggable_trace_result() -> None:
    """The introspector should expose summary, details, reasoning, and debug notes."""
    trace = BehaviorTrace().add("draft", "Wrote the answer.", "Output was vague.")

    result = IntrospectorAgent().introspect(trace)

    assert result.summary == "The trace shows 1 recorded steps: draft."
    assert result.details == ("draft: Wrote the answer.",)
    assert "Behavior trace:" in result.reasoning_trace
    assert result.debug_notes == ("draft: Output was vague.",)
    assert result.to_dict()["debug_notes"] == ["draft: Output was vague."]
