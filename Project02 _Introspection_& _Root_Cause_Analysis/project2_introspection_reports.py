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
