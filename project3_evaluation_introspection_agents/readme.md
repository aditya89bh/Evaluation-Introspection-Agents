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
