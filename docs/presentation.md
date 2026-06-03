# Repository Presentation Notes

## Positioning

Evaluation Introspection Agents is a deterministic agent-evaluation portfolio project. It shows the missing loop between agent action and agent improvement: evaluation, trace introspection, critique, consensus, and revised behavior.

## v0.2 demonstration path

1. Open the README hero and architecture sections.
2. Run `evaluation-agents run examples/planning_task.json`.
3. Run `evaluation-agents run examples/planning_task.json --json`.
4. Run `python benchmarks/run_benchmark.py`.
5. Review `results/latest.md` and `results/benchmark_report.csv`.

## What to emphasize

- No external LLM dependency.
- Deterministic tests and benchmark outputs.
- Multi-critic design prepared for later model-backed critics.
- Portfolio-quality CI, coverage gate, reports, and release notes.
