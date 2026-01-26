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


# =========================
# Agent (toy baseline)
# Replace this with your actual reasoning/planning agent later.
# =========================
