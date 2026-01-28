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
