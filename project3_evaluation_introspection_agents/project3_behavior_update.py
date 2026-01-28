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



# =========================
# Utilities
# =========================
