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

def heuristic_judge(task: Dict[str, Any], output: str, judge_name: str = "heuristic_judge_v1") -> Dict[str, Any]:
    out = output or ""
    out_lower = normalize_text(out)

    expected_keywords = task.get("expected_keywords") or []
    forbidden_keywords = task.get("forbidden_keywords") or []
    must_constraints = task.get("must_include_constraints") or []

    kw_total = len(expected_keywords)
    kw_hits = sum(1 for k in expected_keywords if normalize_text(k) in out_lower) if expected_keywords else 0

    forb_hits = sum(1 for k in forbidden_keywords if normalize_text(k) in out_lower) if forbidden_keywords else 0
    constraints_missing = sum(1 for c in must_constraints if normalize_text(c) not in out_lower) if must_constraints else 0

    has_bullets = ("- " in out) or ("•" in out)
    length_score = 1.0 if len(out) > 200 else (0.6 if len(out) > 80 else 0.3)

    # Accuracy: keyword coverage (demo heuristic)
    if kw_total > 0:
        acc = 5.0 * (kw_hits / kw_total)
    else:
        acc = 3.0

    # Completeness
    comp = 2.0 + 2.0 * length_score + (1.0 if has_bullets else 0.0)
    if kw_total > 0:
        comp += 1.0 * (kw_hits / kw_total)
    comp = clamp(comp, 0, 5)

    # Constraint adherence
    cons = 5.0
    if forb_hits > 0:
        cons -= min(3.0, 1.5 * forb_hits)
    if constraints_missing > 0:
        cons -= min(3.0, 1.0 * constraints_missing)
    cons = clamp(cons, 0, 5)

    # Clarity
    clarity = 3.0
    if has_bullets:
        clarity += 1.0
    if "\n\n" in out:
        clarity += 0.5
    if len(out) < 60:
        clarity -= 1.5
    clarity = clamp(clarity, 0, 5)

    notes = {
        "accuracy": f"Keyword hits: {kw_hits}/{kw_total}" if kw_total else "Open-ended task; neutral baseline.",
        "completeness": f"Length: {len(out)} chars; bullets: {has_bullets}; keyword hits: {kw_hits}/{kw_total}",
        "constraint_adherence": f"Forbidden hits: {forb_hits}; constraints missing: {constraints_missing}",
        "clarity": f"Bullets: {has_bullets}; paragraph breaks: {out.count(chr(10)+chr(10))}",
    }

    return {
        "judge_name": judge_name,
        "criterion_scores": {
            "accuracy": acc,
            "completeness": comp,
            "constraint_adherence": cons,
            "clarity": clarity,
        },
        "criterion_notes": notes,
        "raw": {
            "keyword_hits": kw_hits,
            "keyword_total": kw_total,
            "forbidden_hits": forb_hits,
            "constraints_missing": constraints_missing,
        },
    }

# Rules are simple & explicit so they can be applied deterministically.
# Each rule has:
# - rule_id
# - name
# - when: conditions
# - action: what to do in generation
# - provenance: which failure categories triggered it

def synthesize_rules_from_findings(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a policy store from Project 2 reports.
    Strategy:
    - Count most common failure categories
    - Create/enable rules to address those categories
    """
    cat_counter = Counter()
    for r in reports:
        for f in r.get("findings", []) or []:
            cat = f.get("category")
            if cat:
                cat_counter[cat] += 1

    # Base rule templates
    rules: List[Dict[str, Any]] = []

    def add_rule(rule_id: str, name: str, provenance: List[str], action: Dict[str, Any]) -> None:
        rules.append({
            "rule_id": rule_id,
            "name": name,
            "enabled": True,
            "provenance": provenance,
            "action": action,
        })

    # If constraints issues show up, enforce constraint echo + compliance check
    if cat_counter.get("missing_constraints", 0) > 0:
        add_rule(
            "R001",
            "Constraint First",
            ["missing_constraints"],
            {
                "type": "prepend_constraints_ack",
                "description": "Echo constraints explicitly before answering and self-check compliance.",
            },
        )

    # If underplanning/incompleteness appears, enforce outline-first
    if cat_counter.get("underplanning", 0) > 0:
        add_rule(
            "R002",
            "Outline Before Answer",
            ["underplanning"],
            {
                "type": "add_outline",
                "description": "Add a short outline/checklist before the response to increase completeness.",
            },
        )

    # If clarity issues, enforce bullets + headings
    if cat_counter.get("clarity_structure_issues", 0) > 0:
        add_rule(
            "R003",
            "Structure Output",
            ["clarity_structure_issues"],
            {
                "type": "force_structure",
                "description": "Use headings and bullets; avoid dense paragraphs.",
            },
        )

    # If unsupported claims / weak accuracy, add uncertainty & clarifying question
    if cat_counter.get("unsupported_claims", 0) > 0 or cat_counter.get("overconfidence_low_uncertainty", 0) > 0:
        add_rule(
            "R004",
            "Uncertainty Signaling",
            ["unsupported_claims", "overconfidence_low_uncertainty"],
            {
                "type": "add_uncertainty_line",
                "description": "If uncertain, state assumptions and ask for missing info instead of inventing details.",
            },
        )

    policy_store = {
        "policy_version": f"p3-{uuid.uuid4().hex[:8]}",
        "created_at": now_iso(),
        "category_counts": dict(cat_counter),
        "rules": rules,
    }
    return policy_store


# =========================
# Policy-Guided Agent Wrapper
# =========================

# =========================
# Policy Store + Rule Synthesis
# =========================

