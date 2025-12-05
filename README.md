# ReAsk

**LLM Conversation & Agent Evaluation**

Evaluate LLM responses and agent execution traces. Detect bad responses via re-ask detection, analyze agent trajectories, measure tool usage quality, track self-correction, and benchmark agents head-to-head.

## Features

### Conversation Evaluation (Original)
| Method | Signal | Example |
|--------|--------|---------|
| **CCM** (Conversation Continuity Metric) | User re-asks similar question | "How do I sort?" → bad response → "Can you show me how to sort?" |
| **RDM** (Response Dissatisfaction Metric) | User explicitly corrects | "I asked for Python, not JavaScript!" |
| **LLM Judge** | Fallback evaluation | Used when no clear CCM/RDM signal |

### Agent Evaluation (New in v0.2)
| Feature | What it Detects |
|---------|-----------------|
| **Trajectory Analysis (ATA)** | Circular patterns, regressions, stalls |
| **Tool Use Metrics (TUM)** | Wrong tool selection, hallucinated parameters |
| **Self-Correction (SCD)** | Error awareness, recovery success |
| **Intent Drift (IDM)** | Task alignment over time |
| **Benchmarking (CAB)** | A/B test multiple agents |

## Installation

```bash
pip install -e .
```

## Quick Start - Conversation Evaluation

```python
from reask import ReAskDetector, Message

detector = ReAskDetector()

result = detector.evaluate_response(
    user_message=Message.user("How do I reverse a string in Python?"),
    assistant_response=Message.assistant("Use a for loop to iterate backwards."),
    follow_up=Message.user("Can you just show me how to reverse a string?")
)

print(result)
# EvalResult(❌ BAD, ccm, conf=0.92)
```

## Quick Start - Agent Evaluation

```python
from reask import (
    AgentTrace, AgentStep, ToolCall,
    TrajectoryAnalyzer, ToolEvaluator,
    SelfCorrectionDetector, IntentDriftMeter,
)

# Create an agent trace
trace = AgentTrace(task="Find and fix the bug in auth.py")

trace.add_step(AgentStep(
    index=0,
    thought="Reading the auth file",
    tool_call=ToolCall(name="read_file", parameters={"path": "auth.py"}, result="..."),
))
trace.add_step(AgentStep(
    index=1,
    thought="Found the issue, fixing it",
    tool_call=ToolCall(name="write_file", parameters={"path": "auth.py"}, result="success"),
))
trace.success = True

# Analyze trajectory
analyzer = TrajectoryAnalyzer()
result = analyzer.analyze(trace)
print(result)
# TrajectoryEval(✅ optimal, eff=0.95, conf=0.92)

# Check for intent drift
drift_meter = IntentDriftMeter()
drift = drift_meter.analyze(trace)
print(f"Max drift: {drift.drift_score:.2f}")
```

## Agent Benchmarking

Compare multiple agents on the same task:

```python
from reask import AgentBenchmark, BenchmarkTask, SimpleAgent

benchmark = AgentBenchmark()

# Define task
task = BenchmarkTask(
    name="code_review",
    description="Review the auth module for security issues",
)

# Compare agents (you provide the agent implementations)
comparison = benchmark.compare(
    agents=[agent_a, agent_b, agent_c],
    task=task
)

print(f"Winner: {comparison.winner}")
print(f"Rankings: {comparison.rankings}")

# Get leaderboard
for entry in benchmark.get_leaderboard():
    print(f"{entry.agent_name}: {entry.win_rate:.0%} wins, {entry.success_rate:.0%} success")
```

## Trajectory Analysis

Detect problematic execution patterns:

```python
from reask import TrajectoryAnalyzer, TrajectorySignal

analyzer = TrajectoryAnalyzer()
result = analyzer.analyze(trace)

# Signals
if result.signal == TrajectorySignal.CIRCULAR:
    print(f"Agent going in circles! {result.circular_count} patterns detected")
elif result.signal == TrajectorySignal.REGRESSION:
    print(f"Agent undoing progress! {result.regression_count} regressions")
elif result.signal == TrajectorySignal.OPTIMAL:
    print(f"Clean execution! Efficiency: {result.efficiency_score:.2f}")
```

## Tool Use Evaluation

Check if agents use tools correctly:

```python
from reask import ToolEvaluator, ToolSignal

evaluator = ToolEvaluator(
    available_tools=["read_file", "write_file", "run_command"]
)

efficiency, results = evaluator.evaluate_tool_chain(trace)

for r in results:
    if r.signal == ToolSignal.TSE:
        print(f"Wrong tool! Used {r.tool_name}, should use {r.expected_tool}")
    elif r.signal == ToolSignal.PH:
        print(f"Hallucinated parameters in {r.tool_name}")
```

## Self-Correction Detection

Measure how well agents recover from errors:

```python
from reask import SelfCorrectionDetector

detector = SelfCorrectionDetector()
result = detector.analyze(trace)

print(f"Self-awareness: {result.self_awareness_score:.2f}")
print(f"Recovery efficiency: {result.correction_efficiency:.2f}")
print(f"Stuck in loop: {result.loops_before_fix > 3}")
```

## Intent Drift Monitoring

Track if agents stay on task:

```python
from reask import IntentDriftMeter

meter = IntentDriftMeter(
    drift_warning_threshold=0.35,
    drift_critical_threshold=0.60,
)

result = meter.analyze(trace)

# Real-time alerts
for step in new_steps:
    alert = meter.get_drift_alert(trace, step)
    if alert:
        print(alert)  # "🚨 CRITICAL DRIFT: Step 5 has drift=0.72"
```

## Configuration

```python
# Conversation evaluation
detector = ReAskDetector(
    embedding_model="text-embedding-3-small",
    judge_model="gpt-5-mini",
    similarity_threshold=0.66,
    use_llm_confirmation=True,
    use_llm_judge_fallback=True
)

# Agent evaluation
analyzer = TrajectoryAnalyzer(
    model="gpt-5-nano",
    similarity_threshold=0.75,  # For circular detection
)
```

## API Reference

### Conversation Types

```python
class EvalResult:
    is_bad: bool
    detection_type: DetectionType  # CCM, RDM, LLM_JUDGE, NONE
    confidence: float
    reason: str
    details: dict
```

### Agent Types

```python
class AgentTrace:
    task: str
    steps: List[AgentStep]
    success: bool
    total_cost: float
    total_duration_ms: int

class AgentStep:
    index: int
    thought: str
    action: str
    tool_call: ToolCall
    observation: str

class ToolCall:
    name: str
    parameters: dict
    result: Any
    error: str
```

### Result Types

```python
class TrajectoryEvalResult:
    signal: TrajectorySignal  # CIRCULAR, REGRESSION, STALL, OPTIMAL, etc.
    efficiency_score: float
    circular_count: int
    regression_count: int

class SelfCorrectionResult:
    detected_error: bool
    correction_success: bool
    self_awareness_score: float
    correction_efficiency: float

class IntentDriftResult:
    drift_score: float
    drift_history: List[float]
    is_legitimate: bool
```

## Examples

Run the examples:

```bash
# Conversation evaluation
python examples/basic_usage.py

# Agent evaluation
python examples/agent_evaluation.py
```

## License

MIT
