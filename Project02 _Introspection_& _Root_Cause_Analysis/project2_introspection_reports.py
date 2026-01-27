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
