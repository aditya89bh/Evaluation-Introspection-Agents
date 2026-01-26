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
