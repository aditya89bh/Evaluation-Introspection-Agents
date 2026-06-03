# Changelog

All notable changes to this project are documented here.

## v0.1.0 - 2026-06-03

### Added

- Deterministic evaluator agent for objective-term scoring.
- Behavior trace and introspection agent for step-by-step debugging.
- Critic agent for weaknesses, vague statements, missing constraints, risks, and failure modes.
- Improver agent for recommendations, revised outputs, and improvement explanations.
- Full feedback loop orchestration.
- Structured result models with JSON-safe serialization.
- `evaluation-agents` CLI runner with readable and JSON modes.
- Example task library for safety, planning, robotics, and customer support.
- Benchmark harness with pass rate, average score, failure count, and improvement rate.
- Results documentation, architecture diagrams, and CI workflow.

### Notes

- No LLM API dependency is included in this release.
- All core behavior is deterministic and covered by pytest tests.
