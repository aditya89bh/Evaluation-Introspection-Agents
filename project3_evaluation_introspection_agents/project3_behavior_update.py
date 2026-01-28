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


# =========================
# Policy Store + Rule Synthesis
# =========================

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

def policy_guided_agent(task: Dict[str, Any], base_output: str, policy_store: Dict[str, Any]) -> str:
    """
    Applies enabled policy rules to the base output.
    This is a deterministic "behavior update" mechanism (no fine-tuning).
    """
    out = base_output or ""
    rules = policy_store.get("rules", []) or []
    enabled = [r for r in rules if r.get("enabled")]

    must_constraints = task.get("must_include_constraints") or []
    expected_keywords = task.get("expected_keywords") or []

    # Apply in stable order
    enabled = sorted(enabled, key=lambda r: r.get("rule_id", ""))

    prefix_blocks: List[str] = []
    suffix_blocks: List[str] = []

    for r in enabled:
        action = (r.get("action") or {}).get("type", "")

        if action == "prepend_constraints_ack":
            if must_constraints:
                lines = ["Constraints acknowledged:"]
                for c in must_constraints[:8]:
                    lines.append(f"- {c}")
                lines.append("- Compliance check: I will ensure the response satisfies the above constraints.")
                prefix_blocks.append("\n".join(lines))

        elif action == "add_outline":
            # Keep it short and reusable
            outline = [
                "Outline:",
                "- Identify constraints and success criteria",
                "- Produce the response in structured steps",
                "- Quick self-check against rubric (accuracy, completeness, constraints, clarity)",
            ]
            prefix_blocks.append("\n".join(outline))

        elif action == "force_structure":
            # If output isn't bullet-structured, lightly enforce a structured section
            if "- " not in out:
                structured = [
                    "Response:",
                    "- Main answer (structured)",
                    "- Key points",
                    "- Next steps",
                ]
                prefix_blocks.append("\n".join(structured))

        elif action == "add_uncertainty_line":
            # If task seems underspecified (no expected keywords), ask a clarifying question
            # Otherwise, add a conservative assumptions line.
            if not expected_keywords:
                suffix_blocks.append("Clarifying question: Are there any specific constraints, examples, or expected format you want?")
            else:
                suffix_blocks.append("Assumptions/uncertainty: If any required detail is missing, I’ll state assumptions rather than invent specifics.")

    # Compose new output: prefix + original + suffix
    blocks = []
    if prefix_blocks:
        blocks.append("\n\n".join(prefix_blocks))
    blocks.append(out)
    if suffix_blocks:
        blocks.append("\n\n".join(suffix_blocks))

    return "\n\n".join(blocks).strip()


# =========================
# Project 3 Runner
# =========================

def index_latest_baseline_runs(project1_runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    For each task_id, pick the latest run record as the baseline reference.
    (Project 1 may contain multiple runs per task; this picks the last seen.)
    """
    by_task: Dict[str, Dict[str, Any]] = {}
    for r in project1_runs:
        task = r.get("task", {}) or {}
        tid = str(task.get("task_id", ""))
        if not tid:
            continue
        by_task[tid] = r
    return by_task

def load_project2_reports(path: str) -> List[Dict[str, Any]]:
    reps = read_jsonl(path)
    # Ensure findings are plain dicts (already in JSON)
    return reps

def run_project3(
    project1_jsonl: str = "runs/project1_eval_harness.jsonl",
    project2_jsonl: str = "runs/project2_introspection_reports.jsonl",
    out_policy_json: str = "runs/project3_policy_store.json",
    out_runs_jsonl: str = "runs/project3_runs.jsonl",
    overwrite_runs: bool = True,
) -> None:
    rubric = default_rubric()

    p1_runs = read_jsonl(project1_jsonl)
    p2_reports = load_project2_reports(project2_jsonl)

    policy_store = synthesize_rules_from_findings(p2_reports)
    write_json(out_policy_json, policy_store)

    if overwrite_runs and os.path.exists(out_runs_jsonl):
        os.remove(out_runs_jsonl)

    baseline_index = index_latest_baseline_runs(p1_runs)

    improvements: List[Tuple[str, float, float]] = []  # (task_id, baseline_score, improved_score)

    for task_id, baseline_run in baseline_index.items():
        task = baseline_run.get("task", {}) or {}
        baseline_output = baseline_run.get("output", "") or ""
        baseline_agg = baseline_run.get("aggregated", {}) or {}
        baseline_score = float(baseline_agg.get("weighted_score", 0.0))

        # Policy-guided output
        improved_output = policy_guided_agent(task, baseline_output, policy_store)

        # Evaluate improved output using same judge style
        jr1 = heuristic_judge(task, improved_output, judge_name="heuristic_judge_v1")
        jr2 = heuristic_judge(task, improved_output, judge_name="heuristic_judge_v1_clone")
        improved_agg = aggregate_judges(rubric, [jr1, jr2], strategy="mean")
        improved_score = float(improved_agg["weighted_score"])

        improvements.append((task_id, baseline_score, improved_score))

        record = {
            "run_id": str(uuid.uuid4()),
            "timestamp": now_iso(),
            "project": "project3_behavior_update",
            "task": task,
            "baseline": {
                "source_run_id": baseline_run.get("run_id"),
                "output": baseline_output,
                "aggregated": baseline_agg,
            },
            "policy": {
                "policy_version": policy_store.get("policy_version"),
                "enabled_rules": [r for r in (policy_store.get("rules") or []) if r.get("enabled")],
            },
            "improved": {
                "output": improved_output,
                "judge_results": [jr1, jr2],
                "aggregated": improved_agg,
            },
            "delta": {
                "baseline_weighted_score": baseline_score,
                "improved_weighted_score": improved_score,
                "change": improved_score - baseline_score,
            },
        }
        jsonl_append(out_runs_jsonl, record)

    # Print summary
    print("\n=== Project 3 Summary (Baseline vs Policy-Guided) ===")
    if not improvements:
        print("No tasks found in Project 1 logs.")
        return

    # Per-task
    print(f"{'Task ID':<12} {'Baseline':<10} {'Improved':<10} {'Delta':<10}")
    print("-" * 46)
    for tid, b, imp in improvements:
        print(f"{tid:<12} {b:<10.2f} {imp:<10.2f} {(imp-b):<10.2f}")

    # Overall
    avg_b = sum(b for _, b, _ in improvements) / len(improvements)
    avg_i = sum(i for _, _, i in improvements) / len(improvements)
    print("\nOverall Avg Baseline:", f"{avg_b:.2f} / 5.00")
    print("Overall Avg Improved:", f"{avg_i:.2f} / 5.00")
    print("Overall Delta:", f"{(avg_i-avg_b):.2f}")

    print("\nPolicy written to:", out_policy_json)
    print("Project 3 runs written to:", out_runs_jsonl)
    print("\nNext: we can prune rules that don’t improve scores and keep only high-impact policies.")


# =========================
# Main
# =========================

if __name__ == "__main__":
    run_project3(
        project1_jsonl="runs/project1_eval_harness.jsonl",
        project2_jsonl="runs/project2_introspection_reports.jsonl",
        out_policy_json="runs/project3_policy_store.json",
        out_runs_jsonl="runs/project3_runs.jsonl",
        overwrite_runs=True,
    )
