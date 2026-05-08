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

## Quickstart

```bash
git clone https://github.com/aditya89bh/Evaluation-Introspection-Agents.git
cd Evaluation-Introspection-Agents
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python examples/run_demo.py
```

Run tests:

```bash
pytest
```

## Expected demo behavior

The demo shows a two-run improvement loop.

| Run | Behavior | Result |
|---:|---|---|
| 1 | Agent violates bullet-count and readability constraints | Evaluator detects failure |
| Memory | Introspector generates a behavior rule | Rule is stored in JSON memory |
| 2 | Agent retrieves the rule and changes behavior | Score improves |

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
├── requirements.txt
├── data/
│   └── rules.json
├── docs/
│   ├── architecture.md
│   ├── architecture_diagram.txt
│   ├── evaluation_loop.md
│   ├── introspection_schema.md
│   ├── next_steps.md
│   ├── roadmap.md
│   └── run_guide.md
├── examples/
│   ├── run_demo.py
│   └── sample_run_log.md
├── src/
│   ├── agent_loop.py
│   ├── evaluator.py
│   ├── introspector.py
│   └── rule_memory.py
└── tests/
    └── test_evaluation_loop.py
```

## Current status

This repo is a completed MVP research/demo repo.

It includes:

- deterministic evaluator
- introspection module
- JSON-backed behavioral memory
- runnable two-pass demo
- basic tests
- architecture and roadmap documentation

It is not yet a production-grade adaptive-agent framework.

## Positioning

This project is part of a broader exploration of memory-enabled agents.

Memory agents remember context. Evaluation-introspection agents learn from outcomes. Together, they create systems that do not just recall information, but also improve behavior over time.

## Future directions

- rule confidence and decay
- rule deduplication
- richer evaluation dimensions
- LLM-powered introspection
- multi-agent critique loops
- robotics failure-recovery integration
- visual demo or UI layer
