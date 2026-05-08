# Architecture

This repo explores a simple architecture for agents that improve through evaluation and introspection.

## Core loop

```text
Task Input
   ↓
Agent Attempt
   ↓
Evaluator
   ↓
Introspector
   ↓
Behavior Rule Generator
   ↓
Persistent Rule Memory
   ↓
Next Attempt with Retrieved Rules
```

## Components

| Component | Role | Output |
|---|---|---|
| Task input | Defines the objective and constraints | Task specification |
| Agent attempt | Produces the first response or action | Attempt result |
| Evaluator | Scores the attempt against success criteria | Score and failure labels |
| Introspector | Explains why the result succeeded or failed | Failure analysis |
| Rule generator | Converts introspection into reusable behavior rules | Behavior rule |
| Rule memory | Stores rules across runs | Persistent rule set |
| Retry policy | Decides how to use prior rules on the next attempt | Improved attempt |

## Design principle

The system separates observation, judgment, explanation, and adaptation.

This separation matters because each layer can be improved independently.

- The evaluator can become stricter.
- The introspector can become more precise.
- The rule generator can become more reusable.
- The memory system can become more selective.
- The agent can become better at applying learned rules.

## Why not just prompt better?

A better prompt helps one run. A persistent introspection loop helps future runs.

The goal is not to create a perfect one-shot agent. The goal is to create an agent that can notice recurring failure patterns and build behavioral memory over time.

## Relationship to memory agents

Memory agents usually focus on remembering facts, preferences, context, or episodes.

Evaluation-introspection agents focus on remembering behavioral lessons.

Example:

| Memory type | Example |
|---|---|
| Factual memory | The user prefers concise answers |
| Episodic memory | The last run failed because constraints were missed |
| Behavioral memory | Always check constraints before producing the final answer |

This repo focuses mainly on behavioral memory.

## Minimal viable implementation

A minimal implementation needs only four artifacts:

1. `task.json` for the task and constraints
2. `attempt.json` for the agent output
3. `evaluation.json` for score and failure reason
4. `rules.json` for learned behavior rules

The simplest version can be deterministic and local. It does not require a vector database, multi-agent orchestration, or training pipeline.

## Long-term direction

This architecture can later expand into:

- agent reliability systems
- coding-agent critique loops
- robotics failure recovery
- design-review copilots
- workflow agents that improve over repeated tasks
- memory systems that store behavior changes, not just context
