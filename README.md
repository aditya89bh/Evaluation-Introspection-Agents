# Evaluation Introspection Agents

**Most agents act. Few agents inspect themselves. This repo explores the missing loop: evaluation, introspection, critique, and behavioral improvement.**

## Why this matters

AI agents are often judged only by their final answer or action. Real improvement needs a loop: inspect what happened, identify where behavior fell short, and turn that feedback into a better next action. This repository is a deterministic, testable starting point for studying that loop without depending on any LLM API.

## Architecture

```text
Task + Output + Trace
        |
        v
+------------------+
| Evaluator Agent  |  -> score + objective alignment
+------------------+
        |
        v
+---------------------+
| Introspector Agent  |  -> trace summary + internal explanation
+---------------------+
        |
        v
+--------------+
| Critic Agent |  -> weaknesses + risks + failure modes
+--------------+
        |
        v
+----------------+
| Improver Agent |  -> better next action
+----------------+
        |
        v
 Feedback Loop Report
```

## Core concepts

1. **Evaluator Agent**: scores an output against a task objective.
2. **Introspector Agent**: explains what happened internally using a trace.
3. **Critic Agent**: identifies weaknesses, risks, and failure modes.
4. **Improver Agent**: proposes a better next action.
5. **Feedback Loop**: connects evaluation → introspection → critique → improvement.

## Quickstart

```bash
git clone https://github.com/<your-username>/Evaluation-Introspection-Agents.git
cd Evaluation-Introspection-Agents
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Demo command

```bash
python -m evaluation_introspection_agents.demos.simple_reflection_demo
```

## Example output

```text
Task: Write a concise project summary
Output score: 0.33
Introspection: The trace shows 3 recorded steps: parse_objective, draft_output, finalize.
Critique: Missing expected terms: concise, project. Low objective coverage creates a high risk of task failure.
Improvement: Include the missing objective terms (concise, project) and remove irrelevant detail.
```

## Roadmap

- Add richer deterministic scoring strategies.
- Add trace comparison across repeated attempts.
- Add configurable critique rules.
- Add optional LLM-backed agents behind clean interfaces.
- Add examples for coding, research, and planning agents.
- Add evaluation reports suitable for portfolio demos.

## Portfolio positioning

This project is designed as a portfolio-grade AI agents repository. It demonstrates practical thinking around evaluation loops, self-reflection, error analysis, and iterative correction — the behaviors agents need before they can be trusted with more autonomy.
