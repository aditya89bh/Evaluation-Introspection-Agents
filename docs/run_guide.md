# Run Guide

This guide explains how to run the minimal evaluation-introspection demo.

## Setup

```bash
git clone https://github.com/aditya89bh/Evaluation-Introspection-Agents.git
cd Evaluation-Introspection-Agents
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the demo

```bash
python examples/run_demo.py
```

## What you should see

The demo prints a JSON object with:

- task description
- first agent attempt
- first evaluation result
- introspection result
- behavior rule stored in memory
- second agent attempt
- second evaluation result

## Expected behavior

Run 1 should fail because the agent violates constraints.

Run 2 should improve because the system retrieves the corrective behavior rule from memory.

## Run tests

```bash
pytest
```

## Core idea

The demo proves the basic loop:

```text
Failed attempt → evaluation → introspection → rule memory → improved retry
```

This is a small deterministic prototype, not a production agent framework yet.
