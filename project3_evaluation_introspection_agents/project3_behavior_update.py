"""
Project 3 – Evaluation & Introspection Agents (Behavior Update Loop)
Turns Project 2 diagnosis into persistent behavior rules and measures improvement using Project 1 evaluation.

Inputs:
- runs/project1_eval_harness.jsonl
- runs/project2_introspection_reports.jsonl

Outputs:
- runs/project3_policy_store.json
- runs/project3_runs.jsonl

What it does:
1) Build Policy Store from introspection findings (rules)
2) Apply policies to a policy-guided agent wrapper
3) Re-run tasks and re-evaluate using the same rubric/judge style as Project 1
4) Compare baseline vs policy-guided scores (measurable improvement)
"""

from __future__ import annotations

import json
import os
import re
import uuid
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter


# =========================
# Utilities
# =========================

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def jsonl_append(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================
# Rubric (same shape as Project 1)
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

def default_rubric() -> Rubric:
    return Rubric(
        name="Default Agent Quality Rubric",
        criteria=[
            Criterion("accuracy", "Correctness relative to task intent.", 0, 5, 0.35),
            Criterion("completeness", "Covers important parts; minimal missing items.", 0, 5, 0.25),
            Criterion("constraint_adherence", "Follows constraints and avoids forbidden content.", 0, 5, 0.20),
            Criterion("clarity", "Clear structure; actionable formatting.", 0, 5, 0.20),
        ],
    )

def aggregate_judges(rubric: Rubric, judge_results: List[Dict[str, Any]], strategy: str = "mean") -> Dict[str, Any]:
    if not judge_results:
        raise ValueError("No judge results provided.")

    per_crit: Dict[str, List[float]] = {c.name: [] for c in rubric.criteria}
    notes_by_crit: Dict[str, List[str]] = {c.name: [] for c in rubric.criteria}

    for jr in judge_results:
        scores = jr.get("criterion_scores", {}) or {}
        notes = jr.get("criterion_notes", {}) or {}
        jname = jr.get("judge_name", "judge")
        for c in rubric.criteria:
            if c.name in scores:
                per_crit[c.name].append(float(scores[c.name]))
            if c.name in notes and notes[c.name]:
                notes_by_crit[c.name].append(f"[{jname}] {notes[c.name]}")

    agg_scores: Dict[str, float] = {}
    for c in rubric.criteria:
        vals = per_crit[c.name]
        if not vals:
            agg_scores[c.name] = float(c.scale_min)
        else:
            if strategy == "mean":
                agg_scores[c.name] = sum(vals) / len(vals)
            else:
                raise ValueError(f"Unknown aggregation strategy: {strategy}")
        agg_scores[c.name] = clamp(agg_scores[c.name], c.scale_min, c.scale_max)

    weighted = 0.0
    for c in rubric.criteria:
        weighted += agg_scores[c.name] * c.weight
    weighted_score = safe_div(weighted, rubric.weight_sum())

    agg_notes = {k: " | ".join(v[:6]) for k, v in notes_by_crit.items()}
    return {
        "criterion_scores": agg_scores,
        "weighted_score": weighted_score,
        "aggregation_strategy": strategy,
        "notes": agg_notes,
    }


# =========================
# Heuristic Judge (same spirit as Project 1)
# =========================

