"""Runnable demo for the evaluation-introspection loop.

Usage:
    python examples/run_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from agent_loop import run_loop


if __name__ == "__main__":
    result = run_loop()

    print("\n=== Evaluation Introspection Agent Demo ===\n")
    print(json.dumps(result, indent=2))
