# Demo 3 Plan — Robotics Planning Self-Correction

## Goal

Demo 3 should connect evaluation-introspection agents to robotics without requiring ROS, RViz, MoveIt, or a real robot.

The goal is to show that the same loop used for text and code can also improve a robot task plan.

```text
Bad robot plan → evaluator detects invalid sequence → introspection explains failure → behavior rule stored → corrected robot plan
```

## Scenario

A robot must move a workpiece from a tray into a CNC machine.

The CNC door must be open before insertion.

## Task

```text
Plan a robot sequence to move a workpiece from the tray into the CNC machine.
The robot must open the CNC door before inserting the workpiece.
```

## Run 1 — Bad plan

```text
1. Move to tray.
2. Pick workpiece.
3. Move to CNC.
4. Insert workpiece.
5. Close gripper.
6. Retreat.
```

## Evaluation failure

The evaluator catches that the robot tries to insert the workpiece before opening the CNC door.

Example failure:

```text
Invalid sequence: insert_workpiece happened before open_cnc_door.
```

## Introspection

```text
Failure summary:
The robot plan skipped a required precondition.

Root cause:
The planner generated action steps without checking task preconditions.

Behavior rule:
Check required preconditions before executing each robot action.

Future trigger:
robotics tasks with doors, fixtures, clamps, tools, or machine states
```

## Behavioral memory

The rule is stored:

```json
{
  "rule": "Check required preconditions before executing each robot action.",
  "trigger": "robotics tasks with doors, fixtures, clamps, tools, or machine states"
}
```

## Run 2 — Corrected plan

```text
1. Move to tray.
2. Pick workpiece.
3. Move to CNC.
4. Open CNC door.
5. Insert workpiece.
6. Release workpiece.
7. Retreat.
```

## Why this demo matters

Demo 1 shows language-task adaptation.

Demo 2 shows coding-task adaptation.

Demo 3 shows robotics-planning adaptation.

Together, the repo demonstrates that evaluation-introspection can apply across multiple domains.

## Implementation plan

Add one file:

```text
examples/run_robotics_demo.py
```

The script will include:

- a bad robot plan
- a corrected robot plan
- a deterministic sequence evaluator
- a simple introspection block
- JSON-backed behavior rule memory
- printed before/after result

## Success criteria

The demo should clearly show:

1. Run 1 fails due to missing precondition.
2. Evaluator catches the invalid sequence.
3. Introspection identifies missing precondition checking.
4. Rule memory stores a corrective behavior rule.
5. Run 2 inserts the missing action.
6. Corrected plan passes.

## Future extension

Later, this can connect to the RoboGPT/ekko9 CNC tending simulation.

The lightweight demo should come first because it is easy to run, explain, and include in workshops.
