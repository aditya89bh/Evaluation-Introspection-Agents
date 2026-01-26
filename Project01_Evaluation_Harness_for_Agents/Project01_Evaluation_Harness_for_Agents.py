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
