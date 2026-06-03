# Evaluation-Introspection-Agents v0.2.0

## Summary

v0.2.0 turns the repository from a working deterministic feedback-loop prototype into a more polished portfolio-grade evaluation system with coverage gates, richer benchmark reporting, visual assets, multi-critic analysis, and generated results dashboards.

## Highlights

- Added `pytest-cov` coverage reporting and a CI quality gate requiring at least 90% coverage.
- Added CLI and benchmark demo assets, including terminal-style SVG screenshots.
- Expanded the benchmark corpus to 20 deterministic scenarios across planning, safety, customer support, robotics, and reasoning.
- Added category-aware benchmark metrics, Markdown reports, CSV export, and leaderboard rankings.
- Added multi-critic evaluation with deterministic consensus.
- Added critique confidence, severity, and importance scoring.
- Added critic agreement/disagreement analysis.
- Added generated results dashboard files: `results/latest.json`, `results/latest.md`, and `results/latest.csv`.
- Improved README, architecture docs, benchmark docs, and presentation notes.

## Validation

- Tests: 25 passing.
- Coverage: 91.23% local final run.
- Coverage gate: `--cov-fail-under=90` passes.
- Benchmark cases: 20.
- Benchmark summary:
  - pass_rate: 0.0
  - average_score: 0.02
  - failure_count: 74
  - improvement_rate: 1.0

## Notes

- No external LLM APIs were introduced.
- Core behavior remains deterministic and dependency-light.
- v0.2.0 is prepared for future optional LLM-backed critics behind stable interfaces.
