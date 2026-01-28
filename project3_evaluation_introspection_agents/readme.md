# Project 3 – Evaluation & Introspection Agents  
_From Diagnosis to Measurable Behavior Improvement_

This project implements a **behavior update loop** for AI agents using two upstream layers:

- **Project 1** provides repeatable, rubric-based evaluation.
- **Project 2** provides structured introspection and root-cause diagnosis.

**Project 3 turns those diagnostics into explicit behavior rules (policies)** and applies them to future runs, demonstrating **measurable improvement over time without retraining**.

---

## Why This Project Exists

Most “self-improving agents” stop at:
- evaluation scores
- self-critique narratives

But:
- critique does not change behavior
- failures repeat across runs
- improvements are not measurable

This project treats **introspection as a control signal**, not commentary.

---

## Core Objective

> Convert introspection findings into persistent behavior rules and verify impact using evaluation metrics.

This project answers one question:
> **“What should the agent do differently next time?”**

---

## Project Context: The 3-Layer Improvement Stack

### Project 1 – Evaluation Harness for Agents
- Defines rubrics and scoring
- Produces repeatable run logs (JSONL)

Answers:
> *How good was the agent’s output?*

### Project 2 – Introspection & Root Cause Analysis Agents
- Diagnoses why scores are low
- Produces structured introspection reports (JSONL)

Answers:
> *Why did the agent behave this way?*

### Project 3 – Evaluation & Introspection Agents (This Project)
- Converts diagnosis into explicit rules
- Applies rules to agent behavior
- Measures baseline vs improved deltas

Answers:
> *What should change next time?*

---

## How It Works

Project 3 executes a closed loop:

1. **Load Project 1 run logs** (baseline outputs + evaluation)
2. **Load Project 2 introspection reports** (failure categories + evidence)
3. **Synthesize behavior rules** into a versioned policy store
4. **Apply policies** to generate a policy-guided output
5. **Re-evaluate** using the same rubric/judge style
6. **Compare baseline vs improved scores**

---

## Policy Store

Policies are explicit, versioned rules designed to be deterministic and reusable:

- Constraint-first behavior (acknowledge + compliance check)
- Outline-first planning (structure before writing)
- Output structuring (headings + bullets)
- Uncertainty signaling (avoid unsupported claims)

The policy store captures:
- enabled rules
- provenance (which failure categories triggered them)
- category frequency counts (what keeps going wrong)

---

## Inputs and Outputs

### Inputs
- `runs/project1_eval_harness.jsonl`
- `runs/project2_introspection_reports.jsonl`

### Outputs
- `runs/project3_policy_store.json`  
  Versioned behavior rules synthesized from introspection findings.

- `runs/project3_runs.jsonl`  
  Baseline vs policy-guided runs with evaluation deltas.

---
