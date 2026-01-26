# Project 1 – Evaluation Harness for Agents  
_Measuring Agent Performance with Explicit Rubrics_

This project implements a **systematic evaluation framework** for AI agents, enabling repeatable, rubric-driven assessment of agent outputs across tasks, runs, and configurations.

It establishes a reliable baseline for comparing agents before introducing introspection or behavior updates.

---

## Why This Project Exists

Most agent systems are evaluated using:
- informal prompts
- subjective judgments
- single-run examples

This makes it impossible to answer basic questions like:
- Is the agent actually improving?
- Which version is better?
- What trade-offs are we making?

This project treats **evaluation as infrastructure**, not an afterthought.

---

## Core Objective

> Provide a repeatable, transparent, and extensible evaluation harness for AI agents.

This project answers one question only:
> **“How good was the agent’s output?”**

---

## What This Project Covers (and What It Doesn’t)

### In Scope
- Task execution under controlled conditions
- Explicit evaluation rubrics
- Multi-criteria scoring
- Baseline vs variant comparison
- Run-level logging and aggregation

### Out of Scope
- Failure diagnosis (Project 2)
- Self-critique or introspection
- Behavior modification (Project 3)

---
