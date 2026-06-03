# v0.1.0 Release Notes

## Summary

`v0.1.0` establishes the first portfolio-grade baseline for Evaluation Introspection Agents: a deterministic loop where agents evaluate outputs, inspect behavior traces, critique weaknesses, and propose improved next drafts.

## Highlights

- Evaluation → introspection → critique → improvement pipeline.
- Structured result models and stable JSON serialization.
- CLI runner: `evaluation-agents run task.json`.
- JSON mode: `evaluation-agents run task.json --json`.
- Example task library and benchmark harness.
- CI covering install, tests, demos, CLI smoke tests, and benchmarks.

## Validation

- Pytest suite: 18 passing tests.
- Benchmark summary:
  - case_count: 4
  - pass_rate: 0.0
  - average_score: 0.34
  - failure_count: 15
  - improvement_rate: 1.0

## Release checklist

- [x] Version set to `0.1.0`.
- [x] Changelog added.
- [x] Release notes added.
- [x] Tests passing.
- [x] Benchmark report generated.
- [x] Git tag `v0.1.0` prepared.
- [x] GitHub release prepared.
