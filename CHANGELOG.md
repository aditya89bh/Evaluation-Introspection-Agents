# Changelog

All notable changes to this project are documented here.

## v0.2.0 - 2026-06-03

### Added

- Coverage reporting with `pytest-cov`, `coverage.xml`, and a 90% CI quality gate.
- Coverage artifact upload in GitHub Actions.
- CLI demo assets and terminal-style screenshots.
- Professional Mermaid feedback loop architecture diagram.
- Benchmark visual assets and benchmarking documentation.
- Expanded 20-scenario benchmark corpus across planning, safety, customer support, robotics, and reasoning.
- Category-aware benchmark metrics.
- Markdown benchmark report generation.
- CSV benchmark export.
- Benchmark leaderboard for highest score, lowest score, and most improved scenarios.
- Multi-critic evaluation with configurable critic lists.
- Critique confidence, severity, and importance scoring.
- Critic agreement, partial agreement, and disagreement analysis.
- Results dashboard files: `results/latest.json`, `results/latest.md`, and `results/latest.csv`.
- Repository presentation upgrade for v0.2 portfolio review.

### Validation

- Test suite passes with coverage above 90%.
- Benchmark suite includes 20 deterministic cases.
- Reports are generated in JSON, Markdown, and CSV formats.

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
