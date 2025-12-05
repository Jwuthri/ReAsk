# Multi-Agent Evaluation Plan

## Overview

ReAsk now supports evaluating **multi-agent systems** where multiple AI agents collaborate to complete tasks. This document explains how evaluation works at each level and how scores are computed.

---

## Evaluation Levels

### 1. Session Level (Overall)

The highest level - evaluates the entire multi-agent session.

| Metric | Description | Range |

|--------|-------------|-------|

| `overall_score` | Weighted average of all metrics | 0-1 |

| `coordination_score` | How well agents worked together | 0-1 |

| `intent_drift_score` | Did agents stay on task? (lower = better) | 0-1 |

| `task_completion` | Was the user's goal achieved? | bool |

### 2. Turn Level (User Interaction)

Evaluates each user ↔ agents exchange.

| Metric | Description | Method |

|--------|-------------|--------|

| `is_bad` | Was the final response problematic? | CCM/RDM/Hallucination |

| `detection_type` | What issue was detected | ccm, rdm, hallucination, llm_judge, none |

| `confidence` | Detection confidence | 0-1 |

### 3. Interaction Level (Per-Agent)

Evaluates each agent's contribution within a turn.

| Metric | Description | Applies To |

|--------|-------------|------------|

| `tool_use_score` | Correct tool selection & parameters | Agents with tools |

| `reasoning_score` | Quality of thought process | Reasoner agents |

| `handoff_score` | Quality of context passed to next agent | All except last |

| `response_quality` | Appropriateness for role | All agents |

---

## Score Computation

### Overall Score Formula

```python
overall_score = weighted_average([
    (conversation_score, 0.30),   # Turn-level quality
    (tool_use_score, 0.25),       # Tool usage across all agents
    (coordination_score, 0.20),   # Agent collaboration
    (reasoning_score, 0.15),      # Reasoning quality
    (1 - intent_drift, 0.10),     # Staying on task
])
```

### Per-Agent Score Formula

```python
agent_score[agent_id] = weighted_average([
    (tool_use_score, 0.35),       # If agent uses tools
    (reasoning_score, 0.25),      # If agent reasons
    (handoff_score, 0.20),        # If agent hands off
    (response_quality, 0.20),     # Always
])
```

---

## Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     AGENT SESSION                                │
│  Task: "Process refund for order #12345"                        │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│     TURN 1      │  │     TURN 2      │  │     TURN 3      │
│ "Process refund"│  │ "Is it valid?"  │  │ "Send email"    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
    ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
    ▼         ▼          ▼    ▼    ▼          ▼         ▼
┌───────┐ ┌───────┐  ┌───────┐ ┌───────┐  ┌───────┐ ┌───────┐
│Planner│ │Executor│ │Planner│ │Executor│ │Planner│ │Executor│
│ 0.95  │ │  0.90  │ │ 0.88  │ │  0.92  │ │ 0.90  │ │  0.85  │
└───────┘ └───────┘  └───────┘ └───────┘  └───────┘ └───────┘
    │         │          │    │    │          │         │
    ▼         ▼          ▼    ▼    ▼          ▼         ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Turn Analysis   │  │ Turn Analysis   │  │ Turn Analysis   │
│ CCM: ✗ RDM: ✗   │  │ CCM: ✗ RDM: ✗   │  │ CCM: ✗ RDM: ✓   │
│ Score: 1.0      │  │ Score: 1.0      │  │ Score: 0.0      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   SESSION ANALYSIS      │
                 │                         │
                 │ Overall Score: 0.82     │
                 │ Coordination: 0.88      │
                 │ Intent Drift: 0.05      │
                 │                         │
                 │ Per-Agent Scores:       │
                 │   Planner: 0.91         │
                 │   Executor: 0.89        │
                 └─────────────────────────┘
```

---

## Implementation Tasks

### Backend Tasks

- [ ] **Task 1**: Update `run_background_analysis` to handle multi-agent sessions
  - Detect if session has multiple agents defined
  - Run per-interaction evaluation for each agent
  - Aggregate to per-agent scores

- [ ] **Task 2**: Implement `evaluate_interaction()` function
  - Evaluate tool use for this agent's steps
  - Evaluate reasoning quality
  - Evaluate handoff to next agent

- [ ] **Task 3**: Implement `evaluate_coordination()` function
  - Check if agents are stepping on each other
  - Check if context is preserved across handoffs
  - Check if agents are using correct roles

- [ ] **Task 4**: Implement `generate_agent_recommendations()` function
  - Based on per-agent scores, suggest improvements
  - "Agent X has low tool_use_score - consider improving tool selection"

### Frontend Tasks

- [ ] **Task 5**: Update examples to use multi-agent format
- [ ] **Task 6**: Display per-agent scores in UI
- [ ] **Task 7**: Show agent recommendations

---

## API Response Format

### Session Analysis Response

```json
{
  "overall_score": 0.82,
  "coordination_score": 0.88,
  "intent_drift_score": 0.05,
  
  "per_agent_scores": {
    "planner": {
      "overall": 0.91,
      "tool_use": 0.95,
      "reasoning": 0.90,
      "handoff": 0.88,
      "interactions_count": 3,
      "issues": [],
      "recommendations": []
    },
    "executor": {
      "overall": 0.89,
      "tool_use": 0.92,
      "reasoning": null,
      "handoff": null,
      "interactions_count": 3,
      "issues": ["Tool parameter hallucination in turn 3"],
      "recommendations": ["Validate database query parameters before execution"]
    }
  },
  
  "turn_results": [
    {
      "turn_index": 0,
      "is_bad": false,
      "detection_type": "none",
      "interactions": [
        {"agent_id": "planner", "score": 0.95},
        {"agent_id": "executor", "score": 0.90}
      ]
    }
  ],
  
  "conversation": {
    "total_responses": 3,
    "good_responses": 2,
    "bad_responses": 1,
    "ccm_detections": 0,
    "rdm_detections": 1,
    "hallucination_detections": 0
  }
}
```

---

## Recommendations Engine

Based on scores, generate actionable recommendations:

| Condition | Recommendation |

|-----------|----------------|

| `tool_use_score < 0.7` | "Consider improving tool selection logic for {agent}" |

| `handoff_score < 0.7` | "Agent {agent} should pass more context when delegating" |

| `reasoning_score < 0.7` | "Agent {agent}'s reasoning could be more structured" |

| `coordination_score < 0.7` | "Agents are not coordinating well - consider clearer role definitions" |

| High RDM detections | "Users are frequently correcting responses - improve accuracy" |

| High CCM detections | "Users are re-asking questions - improve response completeness" |

---

## Smart Tool Validation

Since each agent has `tools_available` defined, we can do **accurate validation**:

### Tool Selection Validation

```python
def validate_tool_selection(agent_def, tool_call):
    available_tools = {t['name'] for t in agent_def.tools_available}
    if tool_call.tool_name not in available_tools:
        return {'error': 'UNAUTHORIZED_TOOL', 'message': f"Used '{tool_call.tool_name}' but only has: {available_tools}"}
    return {'valid': True}
```

### Parameter Hallucination Detection

```python
def validate_parameters(tool_schema, tool_call):
    expected = set(tool_schema.parameters_schema.keys())
    provided = set(tool_call.parameters.keys())
    extra = provided - expected  # HALLUCINATED params!
    missing = expected - provided  # Missing required params
    return {'hallucinated': list(extra), 'missing': list(missing)}
```

### Tool Use Issues

| Issue | Detection | Severity |

|-------|-----------|----------|

| Unauthorized Tool | Tool not in agent's `tools_available` | HIGH |

| Parameter Hallucination | Param not in `parameters_schema` | MEDIUM |

| Missing Required Param | Schema param not provided | MEDIUM |

---

## Latency Tracking

Every interaction includes `latency_ms`:

```json
{
  "agent_interactions": [{
    "agent_id": "planner",
    "latency_ms": 520,
    "agent_steps": [{
      "tool_call": { "tool_name": "db_query", "latency_ms": 85 }
    }]
  }]
}
```

---

## Data Schema Summary

```
AgentSession
├── agents[]                    # Agent definitions with tools_available
├── turns[]                     # Conversation turns
│   └── agent_interactions[]    # What each agent did (with latency_ms)
│       └── agent_steps[]       # Individual steps (tool_call.latency_ms)
├── total_cost                  # Total API cost
├── total_latency_ms            # Total session latency
└── analyses[]                  # Analysis runs
    ├── turn_results[]          # Per-turn scores
    └── interaction_results[]   # Per-agent scores (tool validation)
```