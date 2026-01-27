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
