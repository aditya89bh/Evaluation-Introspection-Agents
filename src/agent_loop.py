"""Minimal evaluation-introspection agent loop.

This module demonstrates the core behavior of the project:
1. Produce a first attempt.
2. Evaluate the attempt.
3. Introspect on failure.
4. Store a corrective behavior rule.
5. Produce an improved second attempt using retrieved rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from evaluator import evaluate_output
from introspector import introspect
from rule_memory import RuleMemory


TASK = "Summarize the report in exactly 5 bullet points. Avoid technical jargon."


BAD_ATTEMPT = """1. The architecture introduces an introspection pipeline.
2. The evaluator computes constraint adherence.
3. The retrieval mechanism uses behavioral embeddings.
4. The rule generation stage modifies future planning.
5. The memory system stores episodic critique.
6. The planner updates retry strategies."""


GOOD_ATTEMPT = """1. The system helps AI agents review their own mistakes.
2. It checks whether the answer followed instructions.
3. The agent explains why the mistake happened.
4. Useful lessons are stored for future tasks.
5. The next attempt improves using past feedback."""


def generate_attempt(rules: List[dict]) -> str:
    """Return a weak or improved attempt depending on retrieved rules."""
    if not rules:
        return BAD_ATTEMPT

    return GOOD_ATTEMPT


def run_loop(memory_path: Path | None = None) -> Dict[str, object]:
    """Run the full evaluation-introspection loop once."""
    memory = RuleMemory(path=memory_path) if memory_path else RuleMemory()

    initial_rules = memory.get_rules()
    run_1_output = generate_attempt(initial_rules)
    run_1_eval = evaluate_output(run_1_output)
    run_1_intro = introspect(run_1_eval)

    if not run_1_eval.passed:
        memory.add_rule(
            rule=run_1_intro.behavior_rule,
            trigger=run_1_intro.future_trigger,
        )

    retrieved_rules = memory.get_rules()
    run_2_output = generate_attempt(retrieved_rules)
    run_2_eval = evaluate_output(run_2_output)

    return {
        "task": TASK,
        "run_1": {
            "output": run_1_output,
            "evaluation": run_1_eval.to_dict(),
            "introspection": run_1_intro.to_dict(),
        },
        "memory": retrieved_rules,
        "run_2": {
            "output": run_2_output,
            "evaluation": run_2_eval.to_dict(),
        },
    }
