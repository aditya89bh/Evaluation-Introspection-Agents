# Results

This document captures deterministic demo and benchmark outputs for the repository.

## Demo outputs

### Introspection demo

```text
The trace shows 3 recorded steps: read_task, draft_output, final_check.
Behavior trace:
1. read_task: Parsed the objective and constraints. Observation: Found a concise summary request.
2. draft_output: Generated a first answer. Observation: Answer included the topic but missed one constraint.
3. final_check: Checked expected terms before returning. Observation: Detected incomplete coverage.
```

### Critic demo

```text
Critique:
- Missing expected terms: rollback, monitoring, owner.
- Missing constraints: rollback, monitoring, owner.
- Vague statements detected: some, things.
- Potential risks mentioned without mitigation: risk.
- Failure mode: low objective coverage may cause task failure.
```

### Full feedback loop demo

```text
Task: Create a safe robotics action plan
Score: 0.00
Introspection: The trace shows 3 recorded steps: parse, draft, review.
Critique:
- Missing expected terms: precondition, rollback, operator.
- Missing constraints: precondition, rollback, operator.
- Vague statements detected: some, things, maybe.
- Failure mode: low objective coverage may cause task failure.
Improvement: Include the missing objective terms (precondition, rollback, operator) and remove irrelevant detail.
```

## Benchmark output

Current deterministic benchmark report over `examples/*.json`:

```json
{
  "average_score": 0.34,
  "case_count": 4,
  "failure_count": 15,
  "improvement_rate": 1.0,
  "pass_rate": 0.0
}
```

## Sample evaluation

```text
Objective: Mention rollback and owner
Output: Maybe launch it.
Score: 0.00
Rationale: Matched 0 of 2 expected objective terms.
Missing terms: rollback, owner
```

## Improvement example

```text
Original output:
Maybe launch it.

Improved output:
Maybe launch it.
Improved next draft: Add coverage for: rollback, owner. Replace vague wording with specific actions, owners, and checks.
```

## Interpretation

The first benchmark intentionally includes weak starting outputs. The improvement rate is high because the deterministic improver appends missing objective terms and concrete correction guidance. This makes the loop easy to inspect before adding any optional LLM-backed behavior later.
