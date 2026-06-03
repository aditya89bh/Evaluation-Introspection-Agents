# Architecture

The repository models a minimal deterministic feedback loop for AI-agent behavior.

```text
Task + Output + Trace
        |
        v
Evaluator -> Introspector -> Critic -> Improver
        |
        v
FeedbackLoopResult
```

## Components

- `Task`: stores the objective and expected terms.
- `Trace`: stores observable steps from the agent run.
- `EvaluatorAgent`: scores objective coverage.
- `IntrospectorAgent`: summarizes trace behavior.
- `CriticAgent`: identifies weaknesses and risks.
- `ImproverAgent`: proposes the next better action.
- `FeedbackLoop`: orchestrates the components into one report.

The first version intentionally avoids LLM calls so behavior is repeatable and easy to test.
