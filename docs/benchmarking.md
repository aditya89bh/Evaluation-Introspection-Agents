# Benchmarking

Run the deterministic benchmark suite with:

```bash
python benchmarks/run_benchmark.py
```

The current assets include:

- `assets/benchmarks/benchmark_metrics_output.txt`
- `assets/benchmarks/benchmark_report_example.json`
- `assets/benchmarks/benchmark_terminal_screenshot.svg`

![Benchmark terminal screenshot](../assets/benchmarks/benchmark_terminal_screenshot.svg)

## Category metrics

The benchmark report includes per-category metrics:

- category pass rate
- category average score
- category failure count

Categories currently include planning, safety, customer support, robotics, and reasoning.

## Results dashboard

Running `python benchmarks/run_benchmark.py` now refreshes:

- `results/latest.json`
- `results/latest.md`
- `results/latest.csv`

These files mirror the latest benchmark run and are intended for quick portfolio review.
