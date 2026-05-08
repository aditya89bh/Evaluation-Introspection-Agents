# Evaluation Introspection Agents

A minimal agent architecture for systems that evaluate their own performance, introspect on failures, convert lessons into behavior rules, and improve across future runs without retraining.

## Core thesis

Most agents fail silently. They attempt a task, produce an output, and move on. This project explores a different loop:

> An agent should be able to evaluate what happened, explain why it failed, store the lesson, and change its next behavior.

This makes introspection a practical system primitive, not just a philosophical idea.

## What this repo demonstrates

This repo demonstrates a simple improvement loop:

1. The agent attempts a task.
2. An evaluator scores the attempt.
3. An introspector identifies the failure pattern.
4. A rule generator converts the lesson into an explicit behavior rule.
5. The rule is stored in memory.
6. The next run retrieves the relevant rule.
7. The agent changes behavior and improves.

## Architecture

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

## Why this matters

Evaluation and introspection are key primitives for more reliable AI systems. They help agents move from one-shot output generation toward adaptive behavior across time.

This is useful for:

- Agent reliability
- Memory agents
- Workflow agents
- AI copilots
- Robotics task agents
- Self-improving systems
- Failure recovery loops

## Example improvement loop

| Run | Score | Failure Pattern | Rule Created | Next Behavior |
|---:|---:|---|---|---|
| 1 | 40% | Agent ignored constraints | Check constraints before generating final answer | Adds constraint-checking step |
| 2 | 75% | Agent answered without explaining assumptions | State assumptions before solution | Produces clearer reasoning |
| 3 | 90% | Task completed successfully | No new rule needed | Stable behavior |

## Key concepts

| Concept | Meaning |
|---|---|
| Evaluation | Measuring whether the agent succeeded or failed |
| Introspection | Explaining why the outcome happened |
| Behavior rule | A reusable instruction created from the failure |
| Rule memory | Persistent storage for learned behavior rules |
| Adaptation | Changing future behavior using prior lessons |

## Repository structure

```text
.
├── README.md
├── docs/
│   ├── architecture.md
│   ├── evaluation_loop.md
│   ├── introspection_schema.md
│   └── roadmap.md
└── examples/
    └── sample_run_log.md
```

## Positioning

This project is part of a broader exploration of memory-enabled agents.

Memory agents remember context. Evaluation-introspection agents learn from outcomes. Together, they create systems that do not just recall information, but also improve behavior over time.

## Current status

This repo is in early foundation stage. The main objective is to make the architecture, demo loop, and research framing clear before expanding into richer implementations.

## Next steps

- Add runnable Python demo
- Add rule memory store
- Add evaluation schema
- Add introspection schema
- Add tests for rule extraction and retrieval
- Add multiple example task domains
- Connect this loop to memory-agent and robotics workflows
