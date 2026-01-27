"""
Project 2 – Introspection & Root Cause Analysis Agents
Consumes Project 1 evaluation logs and produces structured diagnostic reports.

Inputs:
- runs/project1_eval_harness.jsonl (from Project 1)

Outputs:
- runs/project2_introspection_reports.jsonl

What it does:
- Detects weak dimensions from rubric scores
- Extracts signals from judge notes and raw output
- Classifies failure modes via an error taxonomy
- Produces machine-readable introspection reports per run
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict


# =========================
# Utilities
# =========================

def classify_failures(
    run: Dict[str, Any],
    weak_criteria: List[str],
    output_signals: Dict[str, Any],
    judge_notes: Dict[str, str],
) -> Tuple[List[IntrospectionFinding], float]:
    """
    Produces a list of findings + a confidence score for the diagnosis.
    Confidence is heuristic: increases with strong, consistent signals across sources.
    """
    findings: List[IntrospectionFinding] = []
    evidence_pool: List[str] = []

    # Build evidence snippets
    for crit, note in judge_notes.items():
        if note:
            evidence_pool.append(f"{crit}: {note}")

    # Helper to add a finding
    def add(category: str, severity: float, evidence: List[str]) -> None:
        findings.append(IntrospectionFinding(category=category, severity=clamp(severity, 0.0, 1.0), evidence=evidence[:6]))

    # 1) Constraints issues
    if "constraint_adherence" in weak_criteria or output_signals.get("constraints_missing", 0) > 0 or output_signals.get("forbidden_hits", 0) > 0:
        sev = 0.4
        ev = []
        if output_signals.get("constraints_missing", 0) > 0:
            sev += 0.3
            ev.append(f"Missing constraints count: {output_signals['constraints_missing']}")
        if output_signals.get("forbidden_hits", 0) > 0:
            sev += 0.3
            ev.append(f"Forbidden keyword hits: {output_signals['forbidden_hits']}")
        if "constraint_adherence" in judge_notes:
            ev.append(judge_notes["constraint_adherence"])
        add("missing_constraints", sev, ev or evidence_pool)

    # 2) Accuracy issues -> unsupported claims / incorrect assumptions
    if "accuracy" in weak_criteria:
        sev = 0.5
        ev = []
        if output_signals.get("keyword_total", 0) > 0:
            hit_ratio = output_signals.get("keyword_hits", 0) / max(1, output_signals.get("keyword_total", 1))
            if hit_ratio < 0.4:
                sev += 0.2
                ev.append(f"Low keyword coverage ratio: {hit_ratio:.2f}")
        if "accuracy" in judge_notes:
            ev.append(judge_notes["accuracy"])

        # Split into two likely causes: unsupported_claims vs incorrect_assumptions
        # Without external ground truth we keep it conservative and label as unsupported_claims.
        add("unsupported_claims", sev, ev or evidence_pool)

    # 3) Completeness issues -> underplanning / incomplete_reasoning
    if "completeness" in weak_criteria:
        sev = 0.45
        ev = []
        if output_signals.get("length_chars", 0) < 120:
            sev += 0.25
            ev.append(f"Output too short: {output_signals['length_chars']} chars")
        if not output_signals.get("has_bullets", False):
            sev += 0.15
            ev.append("No structural bullets detected")
        if "completeness" in judge_notes:
            ev.append(judge_notes["completeness"])

        # Map to underplanning / incomplete_reasoning
        add("underplanning", sev, ev)

    # 4) Clarity issues -> clarity_structure_issues
    if "clarity" in weak_criteria:
        sev = 0.4
        ev = []
        if output_signals.get("length_chars", 0) < 80:
            sev += 0.2
            ev.append("Very short response; likely unclear")
        if "clarity" in judge_notes:
            ev.append(judge_notes["clarity"])
        add("clarity_structure_issues", sev, ev)

    # 5) Overconfidence vs uncertainty signaling
    # If accuracy is weak but it never signals uncertainty, flag overconfidence.
    if ("accuracy" in weak_criteria) and (not output_signals.get("mentions_uncertainty", False)):
        ev = ["Weak accuracy + no uncertainty language detected"]
        add("overconfidence_low_uncertainty", 0.55, ev)

    # Deduplicate categories (keep max severity)
    by_cat: Dict[str, IntrospectionFinding] = {}
    for f in findings:
        if f.category not in by_cat or f.severity > by_cat[f.category].severity:
            by_cat[f.category] = f
    findings = list(by_cat.values())

    # Confidence heuristic
    # More consistent signals -> higher confidence.
    # If we found at least 2 categories with evidence, confidence rises.
    base_conf = 0.45
    base_conf += 0.10 * min(3, len(findings))
    # Add boost if judge notes exist
    if any(judge_notes.values()):
        base_conf += 0.10
    # Add boost if task metadata has constraints/keywords (clearer expectations)
    if output_signals.get("keyword_total", 0) > 0 or output_signals.get("constraints_missing", 0) > 0:
        base_conf += 0.10

    return findings, clamp(base_conf, 0.0, 0.95)


# =========================
# Improvement Hint Generator (non-binding)
# =========================

def extract_weak_criteria(criterion_scores: Dict[str, float], threshold: float = 2.5) -> List[str]:
    """
    Marks criteria as weak if score < threshold on a 0..5 scale.
    """
    weak = []
    for k, v in criterion_scores.items():
        try:
            if float(v) < threshold:
                weak.append(k)
        except Exception:
            continue
    return weak

def extract_judge_notes(run: Dict[str, Any]) -> Dict[str, str]:
    """
    Aggregated judge notes from Project 1 are already available in run["aggregated"]["notes"].
    If missing, we fallback to per-judge notes.
    """
    agg = run.get("aggregated", {})
    notes = agg.get("notes")
    if isinstance(notes, dict) and notes:
        return {k: str(v) for k, v in notes.items()}

    # fallback: merge first judge notes
    jr = run.get("judge_results", [])
    if jr and isinstance(jr, list):
        first = jr[0]
        cn = first.get("criterion_notes", {})
        if isinstance(cn, dict):
            return {k: str(v) for k, v in cn.items()}

    return {}

def extract_output_signals(task: Dict[str, Any], output: str) -> Dict[str, Any]:
    """
    Lightweight signals from output text and task metadata.
    Works even without tools or LLM.
    """
    out = output or ""
    out_norm = normalize_text(out)

    has_bullets = ("- " in out) or ("•" in out)
    length_chars = len(out)
    mentions_uncertainty = any(w in out_norm for w in ["not sure", "uncertain", "might be", "i think", "low confidence", "cannot confirm"])

    expected_keywords = task.get("expected_keywords") or []
    forbidden_keywords = task.get("forbidden_keywords") or []
    must_constraints = task.get("must_include_constraints") or []

    kw_hits = sum(1 for k in expected_keywords if normalize_text(k) in out_norm) if expected_keywords else 0
    kw_total = len(expected_keywords)

    forb_hits = sum(1 for k in forbidden_keywords if normalize_text(k) in out_norm) if forbidden_keywords else 0
    constraints_missing = sum(1 for c in must_constraints if normalize_text(c) not in out_norm) if must_constraints else 0

    return {
        "has_bullets": has_bullets,
        "length_chars": length_chars,
        "mentions_uncertainty": mentions_uncertainty,
        "keyword_hits": kw_hits,
        "keyword_total": kw_total,
        "forbidden_hits": forb_hits,
        "constraints_missing": constraints_missing,
    }


# =========================
# Failure Classification
# =========================

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

def jsonl_append(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def contains_any(text: str, phrases: List[str]) -> bool:
    t = normalize_text(text)
    return any(normalize_text(p) in t for p in phrases)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =========================
# Taxonomy
# =========================

ERROR_TAXONOMY = [
    "missing_constraints",
    "misunderstood_constraints",
    "incorrect_assumptions",
    "unsupported_claims",
    "incomplete_reasoning",
    "tool_misuse",
    "overconfidence_low_uncertainty",
    "memory_retrieval_failure",
    "overplanning",
    "underplanning",
    "clarity_structure_issues",
]


# =========================
# Data Models
# =========================

@dataclass
class IntrospectionFinding:
    category: str
    severity: float  # 0..1
    evidence: List[str]

@dataclass
class IntrospectionReport:
    run_id: str
    task_id: str
    task_title: str
    task_type: str
    agent_name: str
    agent_version: str

    # Inputs summary
    weighted_score: float
    criterion_scores: Dict[str, float]
    weak_criteria: List[str]

    # Diagnostics
    findings: List[IntrospectionFinding]
    primary_cause: Optional[str]
    confidence: float  # 0..1 (confidence in diagnosis)

    # Optional hints (non-binding; Project 3 will turn these into rules)
    improvement_hints: List[str]

    # Traceability
    judge_notes_excerpt: Dict[str, str]


# =========================
# Signal Extraction
# =========================
