<!-- 31fa4e20-eb87-4f80-897a-9edfb8999d94 6606481e-7f94-45cc-bbb0-bdc4797163ff -->
# Fix Agent Evaluation System

## Problem Summary

1. **LLM Judge has no context**: Each turn is evaluated in isolation, so "I'll look up order details" is marked as bad even though it's a correct first step
2. **Intent drift is broken**: Compares to first message only, but user goals evolve during conversation
3. **Per-agent metrics missing**: Only overall scores shown, not tool_use, self_correction, intent_drift per agent
4. **Storage incomplete**: Need to store per-step context and per-agent breakdowns

---

## 1. Add Rolling Conversation Summary

**Files:** [`reask/judge.py`](reask/judge.py), [`api/routes/agent_eval.py`](api/routes/agent_eval.py)

Add a summary generation step after each turn evaluation. Pass the accumulated context to subsequent evaluations. (use openai nano gpt5)

```python
# New method in LLMJudge
def generate_turn_summary(self, turn_index: int, user_message: str, 
                          agent_response: str, previous_summary: str) -> str:
    """Generate a rolling summary of key facts established so far."""
    
# In evaluate(), include previous context:
eval_prompt = f"""
CONVERSATION CONTEXT (what happened before):
{previous_summary}

CURRENT TURN {turn_index}:
User: {user_message.content}
Assistant: {assistant_response.content}
...
"""
```

This ensures the judge knows that "looking up order" is step 1 of a refund process, not a failure to answer.

---

## 2. Goal Hierarchy for Intent Drift

**Files:** [`reask/intent_drift.py`](reask/intent_drift.py)

Replace single-intent tracking with a goal hierarchy:

```python
class GoalHierarchy(BaseModel):
    """Track main task and sub-tasks/follow-ups"""
    main_goal: str  # Original task
    active_goals: List[str]  # Current sub-goals being addressed
    completed_goals: List[str]  # Sub-goals already handled
    goal_history: List[dict]  # {turn_index, goal_type, goal_text}

class IntentDriftMeter:
    def analyze(self, trace: AgentTrace) -> IntentDriftResult:
        # Build goal hierarchy from conversation
        goals = self._build_goal_hierarchy(trace)
        
        # For each step, compare against RELEVANT goal, not just main_goal
        for step in trace.steps:
            relevant_goal = self._get_relevant_goal(step, goals)
            drift = self._measure_drift(step, relevant_goal)
```

When user asks "Is it within the refund window?", this becomes a sub-goal. The agent's response is evaluated against that sub-goal, not the original "process a refund" task.

---

## 3. Per-Agent Metrics Calculation

**Files:** [`api/routes/agent_eval.py`](api/routes/agent_eval.py)

Expand `compute_per_agent_scores()` to calculate all metrics per agent:

```python
def compute_per_agent_scores(session, results, trace_input) -> dict:
    per_agent = {}
    for agent_def in session.agents:
        agent_id = agent_def.id
        agent_turns = get_agent_turns(session, agent_id)
        
        per_agent[agent_id] = {
            "overall_score": ...,
            # NEW: Add all metrics per agent
            "tool_use": {
                "efficiency": ...,
                "calls": [...],
                "correct_count": ...,
            },
            "self_correction": {
                "detected_error": ...,
                "correction_attempt": ...,
                "self_awareness_score": ...,
            },
            "intent_drift": {
                "drift_score": ...,
                "is_legitimate": ...,
                "drift_history": [...],
            },
            "response_quality": {
                "good_count": ...,
                "bad_count": ...,
                "results": [...],
            },
            "reasoning_quality": {...},
        }
    return per_agent
```

---

## 4. Backend Storage Updates

**Files:** [`api/database.py`](api/database.py)

Add new columns to store per-step context and per-agent metrics:

```python
class AgentAnalysis(Base):
    # Add: rolling summary for each turn
    turn_summaries_json = Column(Text, nullable=True)  # [{turn_index, summary}]
    
    # Add: goal hierarchy
    goal_hierarchy_json = Column(Text, nullable=True)  # {main_goal, sub_goals, history}
    
    # Expand: per_agent_scores_json now includes full metric breakdown
    
class TurnAnalysisResult(Base):
    # Add: context used for this evaluation
    context_summary = Column(Text, nullable=True)  # Summary used when evaluating this turn
    active_goal = Column(Text, nullable=True)  # Goal this turn was evaluated against
    
class InteractionAnalysisResult(Base):
    # Add: full metric breakdown per interaction
    tool_use_details_json = Column(Text, nullable=True)
    self_correction_details_json = Column(Text, nullable=True)
    intent_drift_details_json = Column(Text, nullable=True)
```

---

## 5. Frontend: Display Per-Agent Metrics

**Files:** [`web/components/DetailPanel.tsx`](web/components/DetailPanel.tsx), [`web/lib/api.ts`](web/lib/api.ts)

Update `AgentDetailView` to show all metrics for the selected agent:

```tsx
function AgentDetailView({ agentDef, agentScores }) {
  return (
    <div>
      <h3>{agentDef.name}</h3>
      
      {/* Overall Score */}
      <ScoreGauge score={agentScores.overall_score} />
      
      {/* Metric Cards - show ALL metrics */}
      <MetricCard title="Tool Use" score={agentScores.tool_use?.efficiency} />
      <MetricCard title="Self Correction" score={agentScores.self_correction?.self_awareness_score} />
      <MetricCard title="Intent Alignment" score={1 - agentScores.intent_drift?.drift_score} />
      <MetricCard title="Response Quality" score={agentScores.response_quality?.good_rate} />
      
      {/* Detailed breakdown sections */}
      <ExpandableSection title="Tool Use Details">
        {agentScores.tool_use?.results?.map(r => (
          <ToolCallResult {...r} />
        ))}
      </ExpandableSection>
      
      {/* Similar for other metrics */}
    </div>
  );
}
```

Update API types in `api.ts` to include the new per-agent metric structures.

---

## Implementation Order

1. **Rolling summaries** - Fix the context problem first (most impactful)
2. **Goal hierarchy** - Fix intent drift logic
3. **Per-agent metrics** - Calculate all metrics per agent
4. **Backend storage** - Store the new data
5. **Frontend display** - Show per-agent metrics

### To-dos

- [x] Create AgentTree component with collapsible hierarchy (Global > Agents > Turns)
- [x] Create DetailPanel component with context-aware views (global/agent/turn)
- [x] Refactor AgentTraceViewer to use split-view layout with new components
- [x] Implement expandable step-level score breakdown with full metric details
- [x] Update CSS module with split-view layout, tree styles, and detail panel styles
- [x] **Fix rolling context evaluation** - Now builds FULL response from ALL agent interactions per turn
- [x] **Update LLM Judge prompt** - Better understands multi-step processes and intermediate steps
- [x] **Context-aware analysis for all endpoints** - /agent/analyze, /agent/analyze/stream, and background jobs