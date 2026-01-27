# Project 2 – Introspection & Root Cause Analysis Agents  
_Diagnosing Why Agents Succeed or Fail_

This project implements **introspection agents** that analyze evaluation results to explain *why* an AI agent performed well or poorly.

Rather than producing surface-level self-critique, this project focuses on **structured failure diagnosis** that can be consumed programmatically by downstream systems.

---

## Why This Project Exists

Evaluation scores alone are not actionable.

Knowing that an agent scored:
- Accuracy: 2.5
- Constraint adherence: 1.0

does not explain:
- what went wrong
- where the reasoning failed
- what should be fixed

This project fills that gap by converting evaluation data into **explicit root-cause explanations**.

---

## Core Objective

> Transform evaluation outcomes into structured diagnostic insight.

This project answers one question only:
> **“Why did the agent behave this way?”**

---

## Scope and Boundaries

### In Scope
- Analysis of evaluation results from Project 1
- Failure pattern detection
- Root cause classification
- Structured introspection reports
- Machine-readable diagnostic outputs

### Out of Scope
- Scoring or judging outputs (Project 1)
- Behavior modification or learning (Project 3)
- Prompt or policy updates

---

## Inputs

This project consumes:
- rubric scores
- judge notes
- task metadata
- agent configuration
- agent output text

All inputs are assumed to be produced by **Project 1 – Evaluation Harness**.

---

## Error Taxonomy

Failures are classified into explicit categories such as:
- Missing or misunderstood constraints
- Incorrect or unsupported assumptions
- Incomplete reasoning or planning gaps
- Tool misuse or overuse
- Overconfidence or lack of uncertainty signaling
- Memory retrieval failure
- Overplanning or underplanning

This taxonomy allows pattern-level analysis across runs.

---

## Introspection Report Structure

Each agent run produces a structured report containing:

- Task summary
- Evaluation highlights
- Primary failure categories
- Supporting evidence from judge notes
- Confidence and uncertainty signals
- Hypothesized root causes
- Improvement hints (non-binding)

Reports are designed to be **machine-consumable**, not narrative-only.

---
