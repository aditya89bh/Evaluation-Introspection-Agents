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

## Evaluation Model

Each agent run is evaluated using a **rubric-based scoring system**.

### Example Criteria
- Accuracy
- Completeness
- Constraint adherence
- Clarity
- Safety
- Cost / latency (optional)

Each criterion has:
- a definition
- a scoring range
- a weight

Final scores are computed deterministically.

---

## System Components

### 1. Task Runner
- Executes tasks using a fixed agent configuration
- Ensures consistent prompts, tools, and memory settings

---

### 2. Rubric Engine
- Defines evaluation criteria and weights
- Supports task-specific rubrics
- Produces structured score outputs

---

### 3. Judge Module
- One or more evaluation prompts
- Optional multi-judge aggregation
- Produces criterion-level scores and comments

---

### 4. Run Logger
- Stores task inputs, outputs, and scores
- Enables cross-run comparison
- Produces machine-readable evaluation records

---

