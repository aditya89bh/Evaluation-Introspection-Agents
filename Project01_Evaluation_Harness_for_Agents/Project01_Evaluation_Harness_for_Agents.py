"""
Project 1 – Evaluation Harness for Agents
Rubric-based, repeatable evaluation for agent outputs.

- Task Runner
- Rubric Engine
- Judge Module (heuristic by default; optional LLM judge hook)
- Multi-judge aggregation
- JSONL logging
- Summary reporting

Colab-friendly. No extra files needed.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple


# =========================
# Utilities
# =========================

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def jsonl_append(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================
# Data Models
# =========================

@dataclass
class Criterion:
    name: str
    description: str
    scale_min: int = 0
    scale_max: int = 5
    weight: float = 1.0

@dataclass
class Rubric:
    name: str
    criteria: List[Criterion]

    def weight_sum(self) -> float:
        return sum(c.weight for c in self.criteria)

@dataclass
class Task:
    task_id: str
    title: str
    prompt: str
    task_type: str = "general"
    # Optional "expected" targets, useful for heuristic judging demos
    expected_keywords: Optional[List[str]] = None
    forbidden_keywords: Optional[List[str]] = None
    must_include_constraints: Optional[List[str]] = None

@dataclass
class AgentConfig:
    agent_name: str = "baseline_agent"
    version: str = "v1"
    temperature: float = 0.2
    # Add other knobs later: memory_mode, toolset, etc.
    notes: Optional[str] = None

@dataclass
class JudgeResult:
    judge_name: str
    criterion_scores: Dict[str, float]
    criterion_notes: Dict[str, str]
    raw: Optional[Dict[str, Any]] = None

@dataclass
class RunResult:
    run_id: str
    timestamp: str
    agent: Dict[str, Any]
    task: Dict[str, Any]
    output: str
    judge_results: List[Dict[str, Any]]
    aggregated: Dict[str, Any]


# =========================
# Rubric Engine
# =========================

def default_rubric() -> Rubric:
    return Rubric(
        name="Default Agent Quality Rubric",
        criteria=[
            Criterion(
                name="accuracy",
                description="Correctness of the answer relative to the task intent.",
                scale_min=0, scale_max=5, weight=0.35
            ),
            Criterion(
                name="completeness",
                description="Covers all important parts of the request; minimal missing items.",
                scale_min=0, scale_max=5, weight=0.25
            ),
            Criterion(
                name="constraint_adherence",
                description="Follows explicit constraints and avoids forbidden content.",
                scale_min=0, scale_max=5, weight=0.20
            ),
            Criterion(
                name="clarity",
                description="Clear structure, easy to follow, actionable formatting.",
                scale_min=0, scale_max=5, weight=0.20
            ),
        ],
    )


def aggregate_judges(
    rubric: Rubric,
    judge_results: List[JudgeResult],
    strategy: str = "mean"
) -> Dict[str, Any]:
    """
    Aggregates multiple judge results into final criterion scores and a weighted score.
    strategy: "mean" (default) or "median" (not implemented; easy add)
    """
    if not judge_results:
        raise ValueError("No judge results provided.")

    # Collect per-criterion lists
    per_crit: Dict[str, List[float]] = {c.name: [] for c in rubric.criteria}
    notes_by_crit: Dict[str, List[str]] = {c.name: [] for c in rubric.criteria}

    for jr in judge_results:
        for c in rubric.criteria:
            score = jr.criterion_scores.get(c.name, None)
            if score is not None:
                per_crit[c.name].append(float(score))
            note = jr.criterion_notes.get(c.name, "")
            if note:
                notes_by_crit[c.name].append(f"[{jr.judge_name}] {note}")

    # Aggregate
    agg_scores: Dict[str, float] = {}
    for c in rubric.criteria:
        scores = per_crit[c.name]
        if not scores:
            agg_scores[c.name] = float(c.scale_min)
        else:
            if strategy == "mean":
                agg_scores[c.name] = sum(scores) / len(scores)
            else:
                raise ValueError(f"Unknown aggregation strategy: {strategy}")

        # Clamp to rubric scale
        agg_scores[c.name] = clamp(agg_scores[c.name], c.scale_min, c.scale_max)

    # Weighted total score on same 0..5 scale
    wsum = rubric.weight_sum()
    weighted = 0.0
    for c in rubric.criteria:
        weighted += agg_scores[c.name] * c.weight
    weighted_score = safe_div(weighted, wsum)

    # Compact notes
    agg_notes = {k: " | ".join(v[:6]) for k, v in notes_by_crit.items()}  # cap notes
    return {
        "criterion_scores": agg_scores,
        "weighted_score": weighted_score,
        "aggregation_strategy": strategy,
        "notes": agg_notes,
    }

ef baseline_agent(task: Task, config: AgentConfig) -> str:
    """
    A simple placeholder agent.
    Swap this for: your planning agent, tool-using agent, memory agent, etc.
    """
    # Minimal "agent" behavior: produce structured response
    lines = [
        f"Task: {task.title}",
        "",
        "Response:",
        "- I will address the request directly.",
        "- I will keep it structured and actionable.",
    ]
    # Add simple keyword handling for demo
    if task.expected_keywords:
        lines.append("")
        lines.append("Key points:")
        for kw in task.expected_keywords[:5]:
            lines.append(f"- {kw}")

    # If constraints exist, reflect them
    if task.must_include_constraints:
        lines.append("")
        lines.append("Constraints acknowledged:")
        for c in task.must_include_constraints[:6]:
            lines.append(f"- {c}")

    return "\n".join(lines)


# =========================
# Judges
# 1) Heuristic Judge (default)
# 2) LLM Judge hook (optional)
# =========================

def heuristic_judge(task: Task, output: str, rubric: Rubric, judge_name: str = "heuristic_judge") -> JudgeResult:
    """
    Deterministic scoring to make the harness runnable without any external API.
    This is NOT a perfect judge; it's a scaffold.
    """
    out_lower = output.lower()

    # Helper signals
    keyword_hits = 0
    keyword_total = 0
    if task.expected_keywords:
        keyword_total = len(task.expected_keywords)
        keyword_hits = sum(1 for k in task.expected_keywords if k.lower() in out_lower)

    forbidden_hits = 0
    if task.forbidden_keywords:
        forbidden_hits = sum(1 for k in task.forbidden_keywords if k.lower() in out_lower)

    constraints_missing = 0
    if task.must_include_constraints:
        constraints_missing = sum(1 for c in task.must_include_constraints if c.lower() not in out_lower)

    # Scoring
    # Accuracy (rough): keyword coverage
    if keyword_total > 0:
        acc = 5.0 * (keyword_hits / keyword_total)
    else:
        acc = 3.0  # neutral for open-ended tasks

    # Completeness: length + structure + keyword coverage
    has_bullets = ("- " in output) or ("•" in output)
    length_score = 1.0 if len(output) > 200 else (0.6 if len(output) > 80 else 0.3)
    comp = 2.0 + 2.0 * length_score + (1.0 if has_bullets else 0.0)
    if keyword_total > 0:
        comp += 1.0 * (keyword_hits / keyword_total)
    comp = clamp(comp, 0, 5)

    # Constraint adherence: penalties for forbidden + missing constraints
    cons = 5.0
    if forbidden_hits > 0:
        cons -= min(3.0, 1.5 * forbidden_hits)
    if constraints_missing > 0:
        cons -= min(3.0, 1.0 * constraints_missing)
    cons = clamp(cons, 0, 5)

    # Clarity: structure signals
    clarity = 3.0
    if has_bullets:
        clarity += 1.0
    if "\n\n" in output:
        clarity += 0.5
    if len(output) < 60:
        clarity -= 1.5
    clarity = clamp(clarity, 0, 5)

    notes = {
        "accuracy": f"Keyword hits: {keyword_hits}/{keyword_total}" if keyword_total else "Open-ended task; neutral accuracy baseline.",
        "completeness": f"Length: {len(output)} chars; bullets: {has_bullets}; keyword hits: {keyword_hits}/{keyword_total}",
        "constraint_adherence": f"Forbidden hits: {forbidden_hits}; constraints missing: {constraints_missing}",
        "clarity": f"Bullets: {has_bullets}; paragraph breaks: {output.count(chr(10)+chr(10))}",
    }

    return JudgeResult(
        judge_name=judge_name,
        criterion_scores={
            "accuracy": acc,
            "completeness": comp,
            "constraint_adherence": cons,
            "clarity": clarity,
        },
        criterion_notes=notes,
        raw={
            "keyword_hits": keyword_hits,
            "keyword_total": keyword_total,
            "forbidden_hits": forbidden_hits,
            "constraints_missing": constraints_missing,
        },
    )


def llm_judge_stub(task: Task, output: str, rubric: Rubric, judge_name: str = "llm_judge") -> JudgeResult:
    """
    Optional hook for an LLM-as-a-judge implementation.
    Replace this function body with your OpenAI/Anthropic/etc call if you want.
    """
    raise NotImplementedError(
        "LLM judge not implemented in this template. "
        "Use heuristic_judge for now, or plug in your own API call here."
    )


# =========================
# Harness Runner
# =========================

def run_evaluation(
    tasks: List[Task],
    rubric: Rubric,
    agent_config: AgentConfig,
    log_path: str = "runs/project1_eval_harness.jsonl",
    n_runs_per_task: int = 1,
    use_llm_judge: bool = False,
) -> List[RunResult]:
    results: List[RunResult] = []

    for task in tasks:
        for i in range(n_runs_per_task):
            run_id = str(uuid.uuid4())
            ts = now_iso()

            # Run agent
            output = baseline_agent(task, agent_config)

            # Judge (multi-judge ready)
            judge_results: List[JudgeResult] = []
            judge_results.append(heuristic_judge(task, output, rubric, judge_name="heuristic_judge_v1"))

            # Optional: add another heuristic judge variant (simulates multi-judge)
            judge_results.append(heuristic_judge(task, output, rubric, judge_name="heuristic_judge_v1_clone"))

            if use_llm_judge:
                # Plug your LLM judge here
                judge_results.append(llm_judge_stub(task, output, rubric, judge_name="llm_judge_v1"))

            aggregated = aggregate_judges(rubric, judge_results, strategy="mean")

            record = RunResult(
                run_id=run_id,
                timestamp=ts,
                agent=asdict(agent_config),
                task=asdict(task),
                output=output,
                judge_results=[asdict(jr) for jr in judge_results],
                aggregated=aggregated,
            )

            jsonl_append(log_path, asdict(record))
            results.append(record)

    return results


def print_summary(results: List[RunResult], rubric: Rubric) -> None:
    if not results:
        print("No results.")
        return

    # Aggregate per task
    by_task: Dict[str, List[float]] = {}
    for r in results:
        tid = r.task["task_id"]
        by_task.setdefault(tid, []).append(float(r.aggregated["weighted_score"]))

    print("\n=== Evaluation Summary ===")
    print(f"Runs: {len(results)}")
    print(f"Rubric: {rubric.name}")
    print("")

    # Header
    print(f"{'Task ID':<16} {'Avg Score (0-5)':<16} {'Runs':<6}")
    print("-" * 42)
    for tid, scores in by_task.items():
        avg = sum(scores) / len(scores)
        print(f"{tid:<16} {avg:<16.2f} {len(scores):<6}")

    # Overall
    overall = sum(float(r.aggregated["weighted_score"]) for r in results) / len(results)
    print("\nOverall Avg Score:", f"{overall:.2f} / 5.00")

    # Criterion averages overall
    crit_sums: Dict[str, float] = {c.name: 0.0 for c in rubric.criteria}
    for r in results:
        for c in rubric.criteria:
            crit_sums[c.name] += float(r.aggregated["criterion_scores"].get(c.name, 0.0))
    print("\nCriterion Averages:")
    for c in rubric.criteria:
        avg = crit_sums[c.name] / len(results)
        print(f"- {c.name}: {avg:.2f} / 5.00 (weight {c.weight})")


# =========================
# Demo Tasks (replace with your own benchmark set)
# =========================

def demo_tasks() -> List[Task]:
    return [
        Task(
            task_id="T001",
            title="Write a project summary",
            prompt="Summarize what this agent does in 5 bullet points.",
            task_type="writing",
            expected_keywords=["summarize", "agent", "bullet", "points"],
            must_include_constraints=["bullet"],
        ),
        Task(
            task_id="T002",
            title="Plan a small project",
            prompt="Create a 3-step plan to build an evaluation harness for agents.",
            task_type="planning",
            expected_keywords=["plan", "rubric", "score", "log"],
            must_include_constraints=["3-step", "evaluation", "harness"],
        ),
        Task(
            task_id="T003",
            title="Constraint following check",
            prompt="Give a concise answer under 80 words and include exactly 3 bullets.",
            task_type="constraints",
            expected_keywords=["bullets"],
            must_include_constraints=["3 bullets"],
            forbidden_keywords=["lorem", "ipsum"],
        ),
    ]


# =========================
# Main
# =========================

if __name__ == "__main__":
    rubric = default_rubric()
    tasks = demo_tasks()

    config = AgentConfig(
        agent_name="baseline_agent",
        version="v1",
        temperature=0.2,
        notes="Project 1 evaluation harness demo run"
    )

    log_path = "runs/project1_eval_harness.jsonl"

    results = run_evaluation(
        tasks=tasks,
        rubric=rubric,
        agent_config=config,
        log_path=log_path,
        n_runs_per_task=2,
        use_llm_judge=False,  # set True after you implement llm_judge_stub
    )

    print_summary(results, rubric)

    print("\nLog written to:", log_path)
    print("Tip: Open the JSONL to inspect per-run rubric scores and judge notes.")

# =========================
# Agent (toy baseline)
# Replace this with your actual reasoning/planning agent later.
# =========================
