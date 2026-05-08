# Evaluation Loop

The evaluation loop is the core mechanism that allows the agent to improve behavior over time.

Instead of treating each run as isolated, the system treats every outcome as feedback for future behavior.

## Basic sequence

```text
Attempt → Evaluate → Introspect → Extract Rule → Store → Retry
```

## Step-by-step breakdown

### 1. Attempt

The agent performs a task.

Example:

```text
Task:
Summarize the document while respecting all listed constraints.
```

The agent produces an output.

---

### 2. Evaluation

The evaluator checks whether the output satisfies the task requirements.

Example checks:

| Evaluation dimension | Example |
|---|---|
| Accuracy | Did the answer match the source? |
| Constraint adherence | Did the agent obey instructions? |
| Completeness | Were all sections addressed? |
| Clarity | Was the explanation understandable? |
| Efficiency | Was the output unnecessarily verbose? |

Example result:

```json
{
  "score": 0.42,
  "failure": "Ignored formatting constraints"
}
```

---

### 3. Introspection

The introspector attempts to explain why the failure occurred.

Example:

```text
The agent focused on content completeness but failed to prioritize formatting constraints before generation.
```

The introspector should identify:

- what failed
- why it failed
- which behavioral pattern caused the issue
- how the behavior can be corrected

---

### 4. Rule extraction

The introspection result is converted into a reusable rule.

Example:

```text
Before finalizing the response, verify all formatting constraints.
```

The important shift here is:

```text
Failure analysis → reusable behavioral guidance
```

---

### 5. Rule storage

The rule is written into persistent memory.

Example:

```json
{
  "rule_id": "constraint_check_v1",
  "rule": "Verify formatting constraints before final answer",
  "trigger": "Tasks with explicit formatting instructions"
}
```

---

### 6. Retry or future run

On the next similar task, the system retrieves the relevant rule before generation.

This changes the agent behavior.

---

## Why this matters

Most LLM systems do not truly improve between runs.

This loop introduces a lightweight behavioral adaptation layer without requiring retraining.

The system becomes capable of:

- identifying recurring mistakes
- converting lessons into explicit rules
- reducing repeated failures
- improving consistency over time

## Failure categories

The evaluator and introspector can classify failures into categories.

| Failure type | Example |
|---|---|
| Constraint failure | Ignored instructions |
| Planning failure | Steps in wrong order |
| Reasoning failure | Incorrect logic |
| Retrieval failure | Missed relevant memory |
| Communication failure | Output unclear |
| Safety failure | Violated policy or boundary |

## Future expansion

This evaluation loop can later support:

- weighted rule confidence
- rule decay
- contradiction resolution
- multi-agent critique
- memory prioritization
- reinforcement scoring
- robotics recovery loops
- workflow optimization
