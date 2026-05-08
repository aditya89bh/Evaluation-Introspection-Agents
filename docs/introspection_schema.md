# Introspection Schema

The introspection schema defines how an agent converts a failed or weak attempt into a reusable behavioral lesson.

The goal is not just to say "the answer was bad." The goal is to identify the failure pattern clearly enough that future behavior can improve.

## Core idea

```text
Outcome → Failure Pattern → Root Cause → Corrective Rule → Future Trigger
```

## Minimal schema

```json
{
  "task_id": "constraint_summary_001",
  "run_id": 1,
  "score": 0.42,
  "outcome": "failed",
  "failure_type": "constraint_failure",
  "failure_summary": "The agent ignored the requested output format.",
  "root_cause": "The agent prioritized content generation before checking explicit constraints.",
  "behavior_rule": "Before finalizing an answer, check all explicit constraints and confirm the output matches them.",
  "future_trigger": "Tasks with explicit formatting, length, structure, or style constraints",
  "confidence": 0.82
}
```

## Field definitions

| Field | Meaning |
|---|---|
| `task_id` | Identifier for the task being attempted |
| `run_id` | Attempt number or episode number |
| `score` | Numeric quality score from evaluator |
| `outcome` | Success, partial success, or failure |
| `failure_type` | Category of failure |
| `failure_summary` | Short explanation of what went wrong |
| `root_cause` | Deeper behavioral reason behind the failure |
| `behavior_rule` | Reusable rule generated from the introspection |
| `future_trigger` | When this rule should be retrieved again |
| `confidence` | Confidence that this rule is useful |

## Failure types

| Failure type | Description |
|---|---|
| `constraint_failure` | Agent ignored explicit instructions |
| `planning_failure` | Agent used the wrong sequence of steps |
| `reasoning_failure` | Agent made an incorrect inference |
| `retrieval_failure` | Agent failed to use relevant memory or context |
| `communication_failure` | Agent produced unclear or poorly structured output |
| `overgeneration_failure` | Agent gave too much irrelevant information |
| `undergeneration_failure` | Agent gave an incomplete response |
| `safety_failure` | Agent violated a safety, policy, or boundary condition |

## Rule quality checklist

A good behavior rule should be:

1. Specific enough to change behavior
2. General enough to apply beyond one task
3. Written as an actionable instruction
4. Attached to a future trigger
5. Easy to inspect and edit

Bad rule:

```text
Do better next time.
```

Good rule:

```text
Before answering tasks with explicit constraints, list the constraints internally and verify the final response satisfies each one.
```

## Behavioral memory format

Rules can be stored as lightweight JSON objects.

```json
{
  "rule_id": "rule_constraint_check_001",
  "rule": "Before answering tasks with explicit constraints, verify the final output satisfies each constraint.",
  "trigger": "explicit user constraints",
  "source_task": "constraint_summary_001",
  "created_from_run": 1,
  "confidence": 0.82,
  "times_used": 0,
  "last_used": null
}
```

## Why schema matters

Without schema, introspection becomes vague commentary.

With schema, introspection becomes a reusable system primitive.

The schema makes it possible to:

- compare failures across runs
- generate explicit behavior rules
- store lessons in memory
- retrieve rules for future tasks
- measure whether rules improve outcomes
