"""Agent Evaluation routes for ReAsk API"""

import sys
import os
import json
import asyncio
import time
from typing import List, Optional

# Add parent directory to path for reask import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from rich.console import Console

from ..database import (
    get_db, SessionLocal,
    AgentSession, AgentDefinition, SessionTurn, AgentInteraction,
    AgentStep as AgentStepDB,  # Aliased to avoid collision with reask.AgentStep
    AgentAnalysis, TurnAnalysisResult, InteractionAnalysisResult,
    # Aliases
    AgentTraceDB, AgentTurnDB, AnalysisJobDB,
)

from reask import (
    AgentTrace, AgentStep, ToolCall,
    TrajectoryAnalyzer, ToolEvaluator,
    SelfCorrectionDetector,
    AgentBenchmark,
    ReAskDetector, Message as ReAskMessage,
)

console = Console()

router = APIRouter()


# ============================================
# Multi-Agent Input Models
# ============================================

class ToolDefinition(BaseModel):
    """Definition of a tool available to an agent"""
    name: str
    description: Optional[str] = None
    parameters_schema: Optional[dict] = None


class AgentDef(BaseModel):
    """Definition of an agent in the session"""
    id: str  # e.g., "agent1"
    name: Optional[str] = None  # e.g., "ReasoningAgent"
    role: Optional[str] = None  # e.g., "primary_reasoner", "executor"
    description: Optional[str] = None
    capabilities: Optional[List[str]] = None
    tools_available: Optional[List[ToolDefinition]] = None
    config: Optional[dict] = None


class ToolCallInput(BaseModel):
    """A tool call made by an agent"""
    tool_name: str
    parameters: dict
    result: Optional[str] = None
    error: Optional[str] = None


class StepInput(BaseModel):
    """A single step in agent reasoning"""
    thought: Optional[str] = None
    tool_call: Optional[ToolCallInput] = None
    action: Optional[str] = None
    observation: Optional[str] = None


class ToolExecutionResult(BaseModel):
    """Result of tool execution (for executor agents)"""
    tool_name: str
    parameters: dict
    output: Optional[str] = None
    error: Optional[str] = None


class AgentInteractionInput(BaseModel):
    """What a specific agent did in a turn"""
    agent_id: str
    agent_steps: Optional[List[StepInput]] = None
    agent_response: Optional[str] = None
    tool_execution_result: Optional[ToolExecutionResult] = None
    latency_ms: Optional[int] = None


class TurnInput(BaseModel):
    """A turn in the multi-agent conversation"""
    turn_index: Optional[int] = None
    user_message: Optional[str] = None  # Can be null for agent-to-agent
    agent_interactions: List[AgentInteractionInput]


class SessionMetadata(BaseModel):
    """Metadata for the session"""
    timestamp: Optional[str] = None
    tags: Optional[List[str]] = None
    extra: Optional[dict] = None


class AgentSessionInput(BaseModel):
    """Full multi-agent session input"""
    agents: Optional[List[AgentDef]] = None
    session_metadata: Optional[SessionMetadata] = None
    turns: List[TurnInput]
    initial_task: Optional[str] = None
    success: Optional[bool] = None
    total_cost: Optional[float] = None
    
    @property
    def task(self) -> str:
        """Get the main task from initial_task or first user message"""
        if self.initial_task:
            return self.initial_task
        for turn in self.turns:
            if turn.user_message:
                return turn.user_message
        return "Multi-Agent Session"


# Backwards compatibility aliases
class AgentStepInput(StepInput):
    """Alias for backwards compatibility"""
    pass


class AgentTurn(BaseModel):
    """Simple turn format for backwards compatibility"""
    user_message: str
    agent_steps: Optional[List[AgentStepInput]] = None
    agent_response: str


# Alias for AgentTurn
AgentTurnInput = AgentTurn


class AgentTraceInput(BaseModel):
    """Simple trace format - auto-converts to multi-agent format"""
    initial_task: Optional[str] = None
    turns: List[AgentTurn]
    success: Optional[bool] = None
    total_cost: Optional[float] = None
    
    @property
    def task(self) -> str:
        return self.initial_task or (self.turns[0].user_message if self.turns else "")
    
    def to_session(self) -> AgentSessionInput:
        """Convert simple trace to multi-agent session format"""
        return AgentSessionInput(
            agents=[AgentDef(id="agent", name="Agent", role="primary")],
            turns=[
                TurnInput(
                    turn_index=i,
                    user_message=t.user_message,
                    agent_interactions=[
                        AgentInteractionInput(
                            agent_id="agent",
                            agent_steps=[StepInput(**s.dict()) for s in (t.agent_steps or [])],
                            agent_response=t.agent_response,
                        )
                    ]
                )
                for i, t in enumerate(self.turns)
            ],
            initial_task=self.initial_task,
            success=self.success,
            total_cost=self.total_cost,
        )


class AnalysisRequest(BaseModel):
    """Request for analysis - accepts both simple trace or full session"""
    trace: Optional[AgentTraceInput] = None  # Simple single-agent format
    session: Optional[AgentSessionInput] = None  # Full multi-agent format
    analysis_types: List[str]  # conversation, trajectory, tools, self_correction, coordination, full_agent, full_all
    
    def get_session(self) -> AgentSessionInput:
        """Get session input, converting from trace if needed"""
        if self.session:
            return self.session
        if self.trace:
            return self.trace.to_session()
        raise ValueError("Either trace or session must be provided")
    
    def get_trace(self) -> AgentTraceInput:
        """Get trace input, converting from session if needed"""
        if self.trace:
            return self.trace
        if self.session:
            return _session_to_trace(self.session)
        raise ValueError("Either trace or session must be provided")


def _session_to_trace(session: AgentSessionInput) -> AgentTraceInput:
    """Convert multi-agent session to simple trace format"""
    turns = []
    for turn in session.turns:
        if turn.agent_interactions:
            # Use first interaction as main turn
            interaction = turn.agent_interactions[0]
            steps = [
                AgentStepInput(
                    thought=s.thought,
                    action=s.action,
                    observation=s.observation,
                    tool_call=s.tool_call
                ) for s in (interaction.agent_steps or [])
            ]
            turns.append(AgentTurnInput(
                user_message=turn.user_message,
                agent_steps=steps,
                agent_response=interaction.agent_response or turn.final_response or ""
            ))
        else:
            turns.append(AgentTurnInput(
                user_message=turn.user_message,
                agent_steps=[],
                agent_response=turn.final_response or ""
            ))
    return AgentTraceInput(
        initial_task=session.initial_task,
        turns=turns,
        total_cost=session.total_cost
    )


def compute_per_agent_scores(
    session: AgentSessionInput, 
    results: dict,
    trace_input: AgentTraceInput
) -> dict:
    """
    Compute per-agent scores for multi-agent sessions.
    
    NOW includes ALL metrics per agent:
    - tool_use: Efficiency and correctness of tool usage
    - self_correction: Error detection and recovery
    - response_quality: Quality of agent responses
    - reasoning: Quality of reasoning steps
    - handoff: How well agent hands off to others
    
    Returns:
        {
            'per_agent_scores': { agent_id: { 
                overall, 
                tool_use: { efficiency, calls, errors, results },
                self_correction: { detected_error, correction_attempt, awareness_score },
                response_quality: { good_count, bad_count, results },
                reasoning, handoff, ...
            } },
            'coordination_score': float
        }
    """
    if not session.agents or len(session.agents) <= 1:
        # Single agent - still compute full metrics but no coordination
        pass
    
    # Track metrics per agent
    agent_metrics = {}
    agents = session.agents or [AgentDef(id="agent", name="Agent", role="primary")]
    
    for agent in agents:
        agent_id = agent.id
        agent_metrics[agent_id] = {
            # Tool metrics
            'tool_calls': [],
            'tool_errors': 0,
            'tool_results': [],
            # Reasoning metrics
            'reasoning_steps': 0,
            'thoughts': [],
            # Handoff metrics
            'handoffs': 0,
            'handoff_targets': [],
            # Interaction tracking
            'interactions_count': 0,
            'turn_indices': [],
            # Response tracking (for per-agent conversation analysis)
            'responses': [],
            # Issues and recommendations
            'issues': [],
            'recommendations': [],
            'tools_available': set(t.name for t in (agent.tools_available or []) if t.name) if agent.tools_available else set(),
        }

    # Analyze each turn's interactions
    for turn_idx, turn in enumerate(session.turns):
        # Handle both multi-agent (with agent_interactions) and single-agent (direct response) formats
        interactions = turn.agent_interactions
        if not interactions:
            # Single-agent format: create a synthetic interaction from the turn's direct fields
            if turn.agent_response or turn.agent_steps:
                default_agent_id = agents[0].id if agents else 'agent'
                interactions = [AgentInteractionInput(
                    agent_id=default_agent_id,
                    agent_steps=turn.agent_steps,
                    agent_response=turn.agent_response,
                    latency_ms=getattr(turn, 'latency_ms', None),
                )]
            else:
                continue
        
        for idx, interaction in enumerate(interactions):
            agent_id = interaction.agent_id
            if agent_id not in agent_metrics:
                # Create metrics for unknown agent
                agent_metrics[agent_id] = {
                    'tool_calls': [], 'tool_errors': 0, 'tool_results': [],
                    'reasoning_steps': 0, 'thoughts': [],
                    'handoffs': 0, 'handoff_targets': [],
                    'interactions_count': 0, 'turn_indices': [],
                    'responses': [], 'issues': [], 'recommendations': [],
                    'tools_available': set(),
                }
            
            metrics = agent_metrics[agent_id]
            metrics['interactions_count'] += 1
            metrics['turn_indices'].append(turn_idx)
            
            # Track response
            if interaction.agent_response:
                metrics['responses'].append({
                    'turn_index': turn_idx,
                    'response': interaction.agent_response,
                    'latency_ms': interaction.latency_ms,
                })
            
            # Analyze steps
            for step in (interaction.agent_steps or []):
                if step.tool_call:
                    tool_name = step.tool_call.tool_name
                    tool_call_info = {
                        'turn_index': turn_idx,
                        'tool_name': tool_name,
                        'parameters': step.tool_call.parameters,
                        'result': step.tool_call.result,
                        'error': step.tool_call.error,
                    }
                    metrics['tool_calls'].append(tool_call_info)
                    
                    # Check if tool is authorized
                    if metrics['tools_available'] and tool_name not in metrics['tools_available']:
                        metrics['tool_errors'] += 1
                        metrics['issues'].append(f"Used unauthorized tool: {tool_name}")
                    
                    # Check for errors in tool call
                    if step.tool_call.error:
                        metrics['tool_errors'] += 1
                
                if step.thought:
                    metrics['reasoning_steps'] += 1
                    metrics['thoughts'].append({
                        'turn_index': turn_idx,
                        'thought': step.thought,
                    })
            
            # Check for handoffs (not the last interaction in turn)
            if idx < len(turn.agent_interactions) - 1:
                metrics['handoffs'] += 1
                next_agent = turn.agent_interactions[idx + 1].agent_id
                metrics['handoff_targets'].append({
                    'turn_index': turn_idx,
                    'target_agent': next_agent,
                })
    
    # Extract per-agent data from global results
    global_conversation = results.get('conversation', {})
    global_tools = results.get('tools', {})
    global_self_correction = results.get('self_correction', {})
    
    # Compute scores per agent
    per_agent_scores = {}
    for agent in agents:
        agent_id = agent.id
        metrics = agent_metrics.get(agent_id, {})
        
        # ===========================================
        # TOOL USE (full breakdown)
        # ===========================================
        tool_use_data = None
        if metrics.get('tool_calls'):
            tool_calls = metrics['tool_calls']
            error_count = metrics.get('tool_errors', 0)
            efficiency = max(0, 1.0 - (error_count / len(tool_calls))) if tool_calls else 1.0
            
            # Match with global tool results if available
            tool_results = []
            global_tool_results = global_tools.get('results', [])
            for tc in tool_calls:
                # Find matching global result
                matching = next(
                    (r for r in global_tool_results if r.get('tool_name') == tc['tool_name']),
                    None
                )
                tool_results.append({
                    **tc,
                    'signal': matching.get('signal', 'unknown') if matching else 'unknown',
                    'confidence': matching.get('confidence', 0.5) if matching else 0.5,
                    'reason': matching.get('reason', '') if matching else '',
                })
            
            tool_use_data = {
                'efficiency': round(efficiency, 2),
                'total_calls': len(tool_calls),
                'correct_count': len(tool_calls) - error_count,
                'error_count': error_count,
                'results': tool_results,
            }
            
            if efficiency < 0.7:
                metrics['recommendations'].append(f"Improve tool selection - {error_count} errors in {len(tool_calls)} calls")
        
        # ===========================================
        # SELF CORRECTION (per agent)
        # ===========================================
        self_correction_data = None
        if metrics.get('thoughts'):
            # Analyze thoughts for self-correction patterns
            thoughts = [t['thought'] for t in metrics['thoughts']]
            error_keywords = ['error', 'mistake', 'wrong', 'incorrect', 'fix', 'retry', 'oops']
            recovery_keywords = ['fixed', 'corrected', 'resolved', 'now', 'instead']
            
            detected_error = any(kw in ' '.join(thoughts).lower() for kw in error_keywords)
            correction_attempt = any(kw in ' '.join(thoughts).lower() for kw in recovery_keywords)
            
            # Calculate awareness score based on error detection and correction
            # KEY FIX: If no error detected, score should be 1.0 (perfect - no correction needed)
            if not detected_error:
                # No error detected = no correction needed = perfect score
                awareness_score = 1.0
            elif detected_error and correction_attempt:
                # Error detected AND corrected = great self-correction
                awareness_score = 0.9
            elif detected_error:
                # Error detected but not corrected = poor self-correction
                awareness_score = 0.5
            else:
                # Fallback (shouldn't reach here)
                awareness_score = 1.0
            
            self_correction_data = {
                'detected_error': detected_error,
                'correction_attempt': correction_attempt,
                'correction_success': correction_attempt,  # Assume success if attempted
                'self_awareness_score': round(awareness_score, 2),
                'correction_efficiency': round(awareness_score, 2),
                'reasoning_steps': metrics['reasoning_steps'],
            }
        
        # ===========================================
        # RESPONSE QUALITY (per agent)
        # ===========================================
        response_quality_data = None
        if metrics.get('responses') and global_conversation:
            global_results = global_conversation.get('results', [])
            
            agent_responses = metrics['responses']
            agent_results = []
            good_count = 0
            bad_count = 0
            
            for resp in agent_responses:
                turn_idx = resp['turn_index']
                # Find matching conversation result
                matching = next(
                    (r for r in global_results if r.get('step_index') == turn_idx),
                    None
                )
                if matching:
                    agent_results.append({
                        'turn_index': turn_idx,
                        'is_bad': matching.get('is_bad', False),
                        'detection_type': matching.get('detection_type', 'none'),
                        'confidence': matching.get('confidence', 0.5),
                        'reason': matching.get('reason', ''),
                    })
                    if matching.get('is_bad'):
                        bad_count += 1
                    else:
                        good_count += 1
            
            total = good_count + bad_count
            response_quality_data = {
                'good_count': good_count,
                'bad_count': bad_count,
                'total_responses': total,
                'quality_score': round(good_count / total, 2) if total > 0 else 1.0,
                'results': agent_results,
            }
        
        # ===========================================
        # REASONING SCORE
        # ===========================================
        reasoning = None
        if metrics.get('reasoning_steps', 0) > 0:
            # Score based on reasoning depth
            reasoning = min(1.0, metrics['reasoning_steps'] / max(metrics.get('interactions_count', 1), 1))
        
        # ===========================================
        # HANDOFF SCORE
        # ===========================================
        handoff = None
        handoff_data = None
        if metrics.get('handoffs', 0) > 0:
            handoff = 0.85  # Default good score
            handoff_data = {
                'total_handoffs': metrics['handoffs'],
                'targets': metrics.get('handoff_targets', []),
                'quality_score': round(handoff, 2),
            }
        
        # ===========================================
        # CALCULATE OVERALL SCORE
        # ===========================================
        scores = []
        if tool_use_data:
            scores.append(tool_use_data['efficiency'])
        if self_correction_data:
            scores.append(self_correction_data['self_awareness_score'])
        if response_quality_data:
            scores.append(response_quality_data['quality_score'])
        if reasoning is not None:
            scores.append(reasoning)
        if handoff is not None:
            scores.append(handoff)
        
        if scores:
            overall = sum(scores) / len(scores)
        else:
            overall = 0.8  # Default score when no metrics available
        
        # Build the full per-agent score object
        per_agent_scores[agent_id] = {
            'overall': round(overall, 2),
            # Full metric breakdowns
            'tool_use': tool_use_data,
            'self_correction': self_correction_data,
            'response_quality': response_quality_data,
            # Simple scores
            'reasoning': round(reasoning, 2) if reasoning is not None else None,
            'handoff': handoff_data,
            # Metadata
            'interactions_count': metrics.get('interactions_count', 0),
            'issues': metrics.get('issues', []),
            'recommendations': metrics.get('recommendations', []),
        }
    
    # ===========================================
    # COORDINATION SCORE
    # ===========================================
    coordination_score = 0.85  # Default good score
    
    # Penalize if any agent has issues
    total_issues = sum(len(s.get('issues', [])) for s in per_agent_scores.values())
    if total_issues > 0:
        coordination_score = max(0.5, coordination_score - (total_issues * 0.1))
    
    # Bonus for good handoffs
    total_handoffs = sum(m.get('handoffs', 0) for m in agent_metrics.values())
    if total_handoffs > 0:
        coordination_score = min(1.0, coordination_score + 0.05)
    
    return {
        'per_agent_scores': per_agent_scores,
        'coordination_score': round(coordination_score, 2),
    }


class TrajectoryResult(BaseModel):
    signal: str
    confidence: float
    efficiency_score: float
    circular_count: int
    regression_count: int
    reason: str


class ToolResult(BaseModel):
    signal: str
    tool_name: str
    confidence: float
    reason: str
    expected_tool: Optional[str] = None


class SelfCorrectionResultModel(BaseModel):
    detected_error: bool
    correction_attempt: bool
    correction_success: bool
    loops_before_fix: int
    self_awareness_score: float
    correction_efficiency: float
    reason: str


class FullAnalysisResponse(BaseModel):
    trajectory: Optional[TrajectoryResult] = None
    tools: Optional[dict] = None
    self_correction: Optional[SelfCorrectionResultModel] = None


def convert_to_agent_trace(input_trace: AgentTraceInput) -> AgentTrace:
    """Convert API input to AgentTrace object for trajectory/tool/drift analysis"""
    trace = AgentTrace(
        task=input_trace.task,
        success=input_trace.success,
        total_cost=input_trace.total_cost,
    )
    
    step_index = 0
    for turn in input_trace.turns:
        # Add agent steps from this turn
        if turn.agent_steps:
            for step_input in turn.agent_steps:
                tool_call = None
                if step_input.tool_call:
                    tool_call = ToolCall(
                        name=step_input.tool_call.tool_name,
                        parameters=step_input.tool_call.parameters,
                        result=step_input.tool_call.result,
                        error=step_input.tool_call.error,
                    )
                
                step = AgentStep(
                    index=step_index,
                    thought=step_input.thought,
                    action=step_input.action or turn.agent_response,
                    tool_call=tool_call,
                    observation=None,
                )
                trace.add_step(step)
                step_index += 1
        else:
            # If no explicit steps, create one from the response
            step = AgentStep(
                index=step_index,
                thought=None,
                action=turn.agent_response,
                tool_call=None,
                observation=None,
            )
            trace.add_step(step)
            step_index += 1
    
    return trace


def convert_trace_to_conversation(input_trace: AgentTraceInput) -> List[ReAskMessage]:
    """
    Convert agent trace turns into a conversation format for CCM/RDM analysis.
    
    Each turn becomes:
    - User message: user_message
    - Assistant message: agent_response
    
    This is the natural conversation flow that CCM/RDM expects.
    """
    messages = []
    
    for turn in input_trace.turns:
        # User message
        messages.append(ReAskMessage.user(turn.user_message))
        # Agent response
        messages.append(ReAskMessage.assistant(turn.agent_response))
    
    return messages


def build_full_turn_response_from_session(turn: TurnInput) -> str:
    """
    Build a comprehensive response string from ALL agent interactions in a turn.
    This ensures the context includes executor results, tool outputs, etc.
    """
    parts = []
    for interaction in (turn.agent_interactions or []):
        agent_id = interaction.agent_id
        # Include agent thoughts and tool results
        for step in (interaction.agent_steps or []):
            if step.thought:
                parts.append(f"[{agent_id}] Thought: {step.thought}")
            if step.tool_call:
                tc = step.tool_call
                if tc.result:
                    parts.append(f"[{agent_id}] Tool '{tc.tool_name}': {tc.result}")
                elif tc.error:
                    parts.append(f"[{agent_id}] Tool '{tc.tool_name}' error: {tc.error}")
        # Include agent's response
        if interaction.agent_response:
            parts.append(f"[{agent_id}]: {interaction.agent_response}")
    
    return "\n".join(parts) if parts else "(no response)"


def run_conversation_analysis_with_context(session: AgentSessionInput) -> dict:
    """
    Run CCM/RDM/Hallucination/LLM Judge analysis on multi-agent session WITH CONTEXT.
    
    This version:
    1. Uses ALL agent interactions per turn (not just the first)
    2. Maintains a rolling summary passed to each evaluation
    3. Prevents false negatives like "looking up order is bad"
    """
    turns = session.turns
    if not turns:
        return {
            'total_responses': 0,
            'good_responses': 0,
            'bad_responses': 0,
            'ccm_detections': 0,
            'rdm_detections': 0,
            'llm_judge_detections': 0,
            'hallucination_detections': 0,
            'results': [],
            'turn_summaries': [],
            'reason': 'No turns to analyze',
        }
    
    detector = ReAskDetector(
        ccm_model="gpt-5-nano",
        rdm_model="gpt-5-nano", 
        judge_model="gpt-5-mini",
        similarity_threshold=0.5,
        use_llm_confirmation=True,
        use_llm_judge_fallback=True
    )
    
    results = []
    turn_summaries = []
    ccm = rdm = llm_judge = hallucination = bad = 0
    rolling_summary = ""
    
    for turn_idx, turn in enumerate(turns):
        user_msg_text = turn.user_message or ""
        # Build FULL response including ALL agents' work
        full_response = build_full_turn_response_from_session(turn)
        
        user_msg = ReAskMessage.user(user_msg_text)
        assistant_msg = ReAskMessage.assistant(full_response)
        
        # Get follow-up if exists
        follow_up = None
        if turn_idx + 1 < len(turns):
            next_turn = turns[turn_idx + 1]
            if next_turn.user_message:
                follow_up = ReAskMessage.user(next_turn.user_message)
        
        # Evaluate this turn WITH CONTEXT
        result = detector.evaluate_response_with_context(
            user_msg, assistant_msg, follow_up,
            conversation_context=rolling_summary,
            turn_index=turn_idx
        )
        
        detection_type = result.detection_type.value
        results.append({
            'step_index': turn_idx,
            'is_bad': result.is_bad,
            'detection_type': detection_type,
            'confidence': result.confidence,
            'reason': result.reason,
            'context_used': bool(rolling_summary),
        })
        
        if result.is_bad:
            bad += 1
        if detection_type == 'ccm':
            ccm += 1
        elif detection_type == 'rdm':
            rdm += 1
        elif detection_type == 'llm_judge':
            llm_judge += 1
        elif detection_type == 'hallucination':
            hallucination += 1
        
        # Generate rolling summary using FULL response
        summary = detector.generate_turn_summary(
            turn_idx, user_msg_text, full_response, rolling_summary
        )
        turn_summaries.append({
            'turn_index': turn_idx,
            **summary
        })
        rolling_summary = summary.get('summary', '')
    
    total = len(turns)
    good = total - bad
    
    return {
        'total_responses': total,
        'good_responses': good,
        'bad_responses': bad,
        'ccm_detections': ccm,
        'rdm_detections': rdm,
        'llm_judge_detections': llm_judge,
        'hallucination_detections': hallucination,
        'results': results,
        'turn_summaries': turn_summaries,
        'reason': f'Analyzed {total} turns with rolling context: {bad} issues found' if total > 0 else 'No turns to analyze',
    }


def run_conversation_analysis(input_trace: AgentTraceInput) -> dict:
    """Run CCM/RDM/Hallucination/LLM Judge analysis on agent trace (legacy, no context)"""
    messages = convert_trace_to_conversation(input_trace)
    
    if len(messages) < 2:
        return {
            'total_responses': 0,
            'good_responses': 0,
            'bad_responses': 0,
            'ccm_detections': 0,
            'rdm_detections': 0,
            'llm_judge_detections': 0,
            'hallucination_detections': 0,
            'results': [],
            'reason': 'Not enough messages to analyze',
        }
    
    detector = ReAskDetector(
        ccm_model="gpt-5-nano",
        rdm_model="gpt-5-nano", 
        judge_model="gpt-5-mini",
        similarity_threshold=0.5,
        use_llm_confirmation=True,
        use_llm_judge_fallback=True
    )
    
    eval_results = detector.evaluate_conversation(messages)
    
    results = []
    ccm = rdm = llm_judge = hallucination = bad = 0
    
    for idx, result in eval_results:
        detection_type = result.detection_type.value
        results.append({
            'step_index': idx,
            'is_bad': result.is_bad,
            'detection_type': detection_type,
            'confidence': result.confidence,
            'reason': result.reason,
        })
        
        if result.is_bad:
            bad += 1
        if detection_type == 'ccm':
            ccm += 1
        elif detection_type == 'rdm':
            rdm += 1
        elif detection_type == 'llm_judge':
            llm_judge += 1
        elif detection_type == 'hallucination':
            hallucination += 1
    
    total = len(eval_results)
    good = total - bad
    
    return {
        'total_responses': total,
        'good_responses': good,
        'bad_responses': bad,
        'ccm_detections': ccm,
        'rdm_detections': rdm,
        'llm_judge_detections': llm_judge,
        'hallucination_detections': hallucination,
        'results': results,
        'reason': f'Analyzed {total} responses: {bad} issues found' if total > 0 else 'No responses to analyze',
    }


@router.post("/agent/analyze")
async def analyze_agent_trace(request: AnalysisRequest):
    """
    Run selected analyses on an agent trace.
    
    Analysis types:
    - conversation: CCM/RDM/Hallucination/LLM Judge detection
    - trajectory: Detect circular patterns, regressions, efficiency
    - tools: Evaluate tool selection and parameters
    - self_correction: Track error awareness and recovery
    - full_agent: Run all agent analyses
    - full_all: Run all analyses (conversation + agent)
    """
    start_time = time.time()
    # Get session (for context-aware analysis) and trace (for legacy analyzers)
    session = request.get_session()
    trace_input = request.get_trace()
    task_preview = trace_input.task[:50] if trace_input.task else ""
    console.print(f"[bold cyan]📊 Starting analysis:[/] task='[dim]{task_preview}...[/]' types={request.analysis_types}")
    
    trace = convert_to_agent_trace(trace_input)
    results = {}
    
    analysis_types = list(request.analysis_types)  # Make a copy
    
    # Expand full_all to include everything
    if 'full_all' in analysis_types:
        analysis_types = ['conversation', 'trajectory', 'tools', 'self_correction']
    # Expand full_agent to include all agent analyses
    elif 'full_agent' in analysis_types:
        analysis_types = ['trajectory', 'tools', 'self_correction']
    
    # Run analyses in parallel - they do not depend on each other
    analysis_tasks = {}

    async def run_conversation():
        console.print("  [yellow]→[/] Running conversation analysis (CCM/RDM/Hallucination) [bold]with context[/]")
        # Use context-aware analysis with full agent interactions
        conv_result = await asyncio.to_thread(run_conversation_analysis_with_context, session)
        console.print(f"    [green]✓[/] Conversation: {conv_result['bad_responses']}/{conv_result['total_responses']} issues [dim](context-aware)[/]")
        return conv_result

    async def run_trajectory():
        console.print("  [yellow]→[/] Running trajectory analysis")
        analyzer = TrajectoryAnalyzer()
        result = await asyncio.to_thread(analyzer.analyze, trace)
        traj_result = {
            'signal': result.signal.value,
            'confidence': result.confidence,
            'efficiency_score': result.efficiency_score,
            'circular_count': result.circular_count,
            'regression_count': result.regression_count,
            'reason': result.reason,
        }
        console.print(f"    [green]✓[/] Trajectory: signal=[bold]{result.signal.value}[/] efficiency=[cyan]{result.efficiency_score:.0%}[/]")
        return traj_result

    async def run_tools():
        console.print("  [yellow]→[/] Running tool evaluation")
        evaluator = ToolEvaluator()
        efficiency, tool_results = await asyncio.to_thread(evaluator.evaluate_tool_chain, trace)
        tools_result = {
            'efficiency': efficiency,
            'results': [
                {
                    'signal': r.signal.value,
                    'tool_name': r.tool_name,
                    'confidence': r.confidence,
                    'reason': r.reason,
                    'expected_tool': r.expected_tool,
                }
                for r in tool_results
            ],
            'total_calls': len(trace.tool_calls),
            'correct_count': sum(1 for r in tool_results if r.signal.value == 'correct'),
        }
        console.print(f"    [green]✓[/] Tools: {len(trace.tool_calls)} calls, efficiency=[cyan]{efficiency:.0%}[/]")
        return tools_result

    async def run_self_correction():
        console.print("  [yellow]→[/] Running self-correction detection")
        detector = SelfCorrectionDetector()
        result = await asyncio.to_thread(detector.analyze, trace)
        sc_result = {
            'detected_error': result.detected_error,
            'correction_attempt': result.correction_attempt,
            'correction_success': result.correction_success,
            'loops_before_fix': result.loops_before_fix,
            'self_awareness_score': result.self_awareness_score,
            'correction_efficiency': result.correction_efficiency,
            'reason': result.reason,
        }
        console.print(f"    [green]✓[/] Self-correction: awareness=[cyan]{result.self_awareness_score:.0%}[/]")
        return sc_result

    if 'conversation' in analysis_types:
        analysis_tasks['conversation'] = asyncio.create_task(run_conversation())
    if 'trajectory' in analysis_types:
        analysis_tasks['trajectory'] = asyncio.create_task(run_trajectory())
    if 'tools' in analysis_types:
        analysis_tasks['tools'] = asyncio.create_task(run_tools())
    if 'self_correction' in analysis_types:
        analysis_tasks['self_correction'] = asyncio.create_task(run_self_correction())

    outcomes = await asyncio.gather(*analysis_tasks.values(), return_exceptions=True)

    errors = []
    for name, outcome in zip(analysis_tasks.keys(), outcomes):
        if isinstance(outcome, Exception):
            errors.append((name, outcome))
        else:
            results[name] = outcome

    if errors:
        first_name, first_error = errors[0]
        console.print(f"[red]✖[/] Analysis '{first_name}' failed: {first_error}")
        raise HTTPException(status_code=500, detail=f"Analysis '{first_name}' failed: {first_error}")
    
    # Calculate overall score
    overall_score = 0.0
    score_count = 0
    
    if 'conversation' in results:
        total = results['conversation']['total_responses']
        good = results['conversation']['good_responses']
        conv_score = good / total if total > 0 else 1.0
        overall_score += conv_score
        score_count += 1
    
    if 'trajectory' in results:
        overall_score += results['trajectory']['efficiency_score']
        score_count += 1
    
    if 'tools' in results:
        overall_score += results['tools']['efficiency']
        score_count += 1
    
    if 'self_correction' in results:
        overall_score += results['self_correction']['correction_efficiency']
        score_count += 1
    
    results['overall_score'] = overall_score / score_count if score_count > 0 else 0.0
    results['analysis_types'] = analysis_types
    
    # Compute per-agent scores for multi-agent sessions (session already defined above)
    agent_scores = compute_per_agent_scores(session, results, trace_input)
    if agent_scores:
        results['per_agent_scores'] = agent_scores.get('per_agent_scores', {})
        results['coordination_score'] = agent_scores.get('coordination_score')
        console.print(f"  [cyan]✓[/] Per-agent scores: {len(results['per_agent_scores'])} agents")
    
    duration = (time.time() - start_time) * 1000
    score_color = "green" if results['overall_score'] >= 0.7 else "yellow" if results['overall_score'] >= 0.4 else "red"
    console.print(f"[bold green]✅ Analysis complete:[/] overall_score=[bold {score_color}]{results['overall_score']:.0%}[/] [dim]({duration:.0f}ms)[/]")
    
    return results


@router.post("/agent/analyze/stream")
async def analyze_agent_trace_stream(request: AnalysisRequest):
    """
    Run analyses with streaming progress updates.
    """
    # Get session (handles both trace and session formats)
    session = request.get_session()
    task_preview = (session.initial_task or "")[:50]
    console.print(f"[bold cyan]📊 Starting streaming analysis:[/] task='[dim]{task_preview}...[/]' types={request.analysis_types}")
    start_time = time.time()
    
    async def event_generator():
        # Convert session to simple trace for backwards compat with analyzers
        trace_input = _session_to_trace(session)
        trace = convert_to_agent_trace(trace_input)
        
        analysis_types = list(request.analysis_types)
        if 'full_all' in analysis_types:
            analysis_types = ['conversation', 'trajectory', 'tools', 'self_correction']
        elif 'full_agent' in analysis_types:
            analysis_types = ['trajectory', 'tools', 'self_correction']
        
        total_analyses = len(analysis_types)
        current = 0
        
        console.print(f"  [dim]Running {total_analyses} analyses: {analysis_types}[/]")
        yield f"data: {json.dumps({'type': 'start', 'total': total_analyses})}\n\n"
        
        results = {}
        
        # Conversation analysis (CCM/RDM/Hallucination) - stream turn by turn
        # NOW WITH ROLLING CONTEXT to prevent false negatives
        if 'conversation' in analysis_types:
            current += 1
            # Use SESSION turns directly (not converted trace) to preserve all agent interactions
            session_turns = session.turns
            total_turns = len(session_turns)
            
            console.print(f"  [yellow]→[/] [dim][stream][/] Running conversation analysis ({total_turns} turns) [bold]with full context[/]")
            yield f"data: {json.dumps({'type': 'progress', 'current': current, 'total': total_analyses, 'analysis': 'conversation', 'status': 'running', 'turn_total': total_turns})}\n\n"
            
            # Initialize detector once
            detector = ReAskDetector(
                ccm_model="gpt-5-nano",
                rdm_model="gpt-5-nano", 
                judge_model="gpt-5-mini",
                similarity_threshold=0.5,
                use_llm_confirmation=True,
                use_llm_judge_fallback=True
            )
            
            # Track results
            turn_results = []
            turn_summaries = []
            ccm = rdm = llm_judge = hallucination = bad = 0
            
            # Rolling context for context-aware evaluation
            rolling_summary = ""
            
            # Helper function to build a FULL response from all agent interactions in a turn
            def build_full_turn_response(turn: TurnInput) -> str:
                """
                Build a comprehensive response string from ALL agent interactions.
                This ensures the context includes executor results, tool outputs, etc.
                """
                parts = []
                for interaction in (turn.agent_interactions or []):
                    agent_id = interaction.agent_id
                    # Include agent thoughts and tool results
                    for step in (interaction.agent_steps or []):
                        if step.thought:
                            parts.append(f"[{agent_id}] Thought: {step.thought}")
                        if step.tool_call:
                            tc = step.tool_call
                            if tc.result:
                                parts.append(f"[{agent_id}] Tool '{tc.tool_name}': {tc.result}")
                            elif tc.error:
                                parts.append(f"[{agent_id}] Tool '{tc.tool_name}' error: {tc.error}")
                    # Include agent's response
                    if interaction.agent_response:
                        parts.append(f"[{agent_id}]: {interaction.agent_response}")
                
                return "\n".join(parts) if parts else "(no response)"
            
            # Process turn by turn WITH FULL CONTEXT
            for turn_idx, turn in enumerate(session_turns):
                user_msg_text = turn.user_message or ""
                # Build FULL response including ALL agents' work
                full_response = build_full_turn_response(turn)
                
                user_msg = ReAskMessage.user(user_msg_text)
                assistant_msg = ReAskMessage.assistant(full_response)
                
                # Get follow-up if exists
                follow_up = None
                if turn_idx + 1 < total_turns:
                    next_turn = session_turns[turn_idx + 1]
                    if next_turn.user_message:
                        follow_up = ReAskMessage.user(next_turn.user_message)
                
                # Evaluate this turn WITH CONTEXT
                result = detector.evaluate_response_with_context(
                    user_msg, assistant_msg, follow_up,
                    conversation_context=rolling_summary,
                    turn_index=turn_idx
                )
                
                detection_type = result.detection_type.value
                turn_result = {
                    'step_index': turn_idx,
                    'is_bad': result.is_bad,
                    'detection_type': detection_type,
                    'confidence': result.confidence,
                    'reason': result.reason,
                    'context_used': bool(rolling_summary),
                }
                turn_results.append(turn_result)
                
                if result.is_bad:
                    bad += 1
                if detection_type == 'ccm':
                    ccm += 1
                elif detection_type == 'rdm':
                    rdm += 1
                elif detection_type == 'llm_judge':
                    llm_judge += 1
                elif detection_type == 'hallucination':
                    hallucination += 1
                
                # Generate rolling summary using FULL response
                summary = detector.generate_turn_summary(
                    turn_idx, user_msg_text, full_response, rolling_summary
                )
                turn_summaries.append({
                    'turn_index': turn_idx,
                    **summary
                })
                rolling_summary = summary.get('summary', '')
                
                console.print(f"    [dim]Turn {turn_idx + 1}/{total_turns}:[/] {'❌' if result.is_bad else '✅'} {detection_type}")
                
                # Stream this turn's result
                yield f"data: {json.dumps({'type': 'turn_result', 'turn_index': turn_idx, 'turn_total': total_turns, 'result': turn_result})}\n\n"
                await asyncio.sleep(0.01)
            
            # Calculate final conversation results
            total_responses = total_turns
            good = total_responses - bad
            avg_conf = sum(r['confidence'] for r in turn_results) / len(turn_results) if turn_results else 0
            
            conv_result = {
                'total_responses': total_responses,
                'good_responses': good,
                'bad_responses': bad,
                'ccm_detections': ccm,
                'rdm_detections': rdm,
                'llm_judge_detections': llm_judge,
                'hallucination_detections': hallucination,
                'results': turn_results,
                'turn_summaries': turn_summaries,  # Include summaries in response
                'reason': 'Conversation analysis complete (with rolling context).',
                'avg_confidence': avg_conf,
            }
            results['conversation'] = conv_result
            console.print(f"    [green]✓[/] Conversation: {bad}/{total_responses} issues [dim](context-aware)[/]")
            
            yield f"data: {json.dumps({'type': 'result', 'analysis': 'conversation', 'data': results['conversation']})}\n\n"
            await asyncio.sleep(0.01)
        
        # Trajectory
        if 'trajectory' in analysis_types:
            current += 1
            console.print("  [yellow]→[/] [dim][stream][/] Running trajectory analysis")
            yield f"data: {json.dumps({'type': 'progress', 'current': current, 'total': total_analyses, 'analysis': 'trajectory', 'status': 'running'})}\n\n"
            
            analyzer = TrajectoryAnalyzer()
            result = analyzer.analyze(trace)
            results['trajectory'] = {
                'signal': result.signal.value,
                'confidence': result.confidence,
                'efficiency_score': result.efficiency_score,
                'circular_count': result.circular_count,
                'regression_count': result.regression_count,
                'reason': result.reason,
            }
            console.print(f"    [green]✓[/] Trajectory: signal=[bold]{result.signal.value}[/] efficiency=[cyan]{result.efficiency_score:.0%}[/]")
            
            yield f"data: {json.dumps({'type': 'result', 'analysis': 'trajectory', 'data': results['trajectory']})}\n\n"
            await asyncio.sleep(0.01)
        
        # Tools
        if 'tools' in analysis_types:
            current += 1
            console.print("  [yellow]→[/] [dim][stream][/] Running tool evaluation")
            yield f"data: {json.dumps({'type': 'progress', 'current': current, 'total': total_analyses, 'analysis': 'tools', 'status': 'running'})}\n\n"
            
            evaluator = ToolEvaluator()
            efficiency, tool_results = evaluator.evaluate_tool_chain(trace)
            results['tools'] = {
                'efficiency': efficiency,
                'results': [
                    {
                        'signal': r.signal.value,
                        'tool_name': r.tool_name,
                        'confidence': r.confidence,
                        'reason': r.reason,
                    }
                    for r in tool_results
                ],
                'total_calls': len(trace.tool_calls),
            }
            console.print(f"    [green]✓[/] Tools: {len(trace.tool_calls)} calls, efficiency=[cyan]{efficiency:.0%}[/]")
            
            yield f"data: {json.dumps({'type': 'result', 'analysis': 'tools', 'data': results['tools']})}\n\n"
            await asyncio.sleep(0.01)
        
        # Self-correction
        if 'self_correction' in analysis_types:
            current += 1
            console.print("  [yellow]→[/] [dim][stream][/] Running self-correction detection")
            yield f"data: {json.dumps({'type': 'progress', 'current': current, 'total': total_analyses, 'analysis': 'self_correction', 'status': 'running'})}\n\n"
            
            detector = SelfCorrectionDetector()
            result = detector.analyze(trace)
            results['self_correction'] = {
                'detected_error': result.detected_error,
                'correction_attempt': result.correction_attempt,
                'correction_success': result.correction_success,
                'loops_before_fix': result.loops_before_fix,
                'self_awareness_score': result.self_awareness_score,
                'correction_efficiency': result.correction_efficiency,
                'reason': result.reason,
            }
            console.print(f"    [green]✓[/] Self-correction: awareness=[cyan]{result.self_awareness_score:.0%}[/]")
            
            yield f"data: {json.dumps({'type': 'result', 'analysis': 'self_correction', 'data': results['self_correction']})}\n\n"
            await asyncio.sleep(0.01)
        
        # Calculate overall score
        overall_score = 0.0
        score_count = 0
        
        if 'conversation' in results:
            total = results['conversation']['total_responses']
            good = results['conversation']['good_responses']
            conv_score = good / total if total > 0 else 1.0
            overall_score += conv_score
            score_count += 1
        
        if 'trajectory' in results:
            overall_score += results['trajectory']['efficiency_score']
            score_count += 1
        
        if 'tools' in results:
            overall_score += results['tools']['efficiency']
            score_count += 1
        
        if 'self_correction' in results:
            overall_score += results['self_correction']['correction_efficiency']
            score_count += 1
        
        results['overall_score'] = overall_score / score_count if score_count > 0 else 0.0
        results['analysis_types'] = analysis_types
        
        # Compute per-agent scores for multi-agent sessions
        agent_scores = compute_per_agent_scores(session, results, trace_input)
        if agent_scores:
            results['per_agent_scores'] = agent_scores.get('per_agent_scores', {})
            results['coordination_score'] = agent_scores.get('coordination_score')
            console.print(f"  [cyan]✓[/] Per-agent scores: {len(results['per_agent_scores'])} agents")
        
        duration = (time.time() - start_time) * 1000
        score_color = "green" if results['overall_score'] >= 0.7 else "yellow" if results['overall_score'] >= 0.4 else "red"
        console.print(f"[bold green]✅ Streaming analysis complete:[/] overall_score=[bold {score_color}]{results['overall_score']:.0%}[/] [dim]({duration:.0f}ms)[/]")
        
        # Complete
        yield f"data: {json.dumps({'type': 'complete', 'results': results})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@router.post("/agent/trajectory")
async def analyze_trajectory(request: AgentTraceInput):
    """Quick endpoint for trajectory analysis only"""
    console.print(f"[bold magenta]🔄 Trajectory analysis:[/] {len(request.steps)} steps")
    trace = convert_to_agent_trace(request)
    analyzer = TrajectoryAnalyzer()
    result = analyzer.analyze(trace)
    console.print(f"  [green]✓[/] signal=[bold]{result.signal.value}[/] efficiency=[cyan]{result.efficiency_score:.0%}[/]")
    
    return {
        'signal': result.signal.value,
        'confidence': result.confidence,
        'efficiency_score': result.efficiency_score,
        'circular_count': result.circular_count,
        'regression_count': result.regression_count,
        'reason': result.reason,
    }


@router.post("/agent/tools")
async def analyze_tools(request: AgentTraceInput):
    """Quick endpoint for tool evaluation only"""
    console.print(f"[bold magenta]🔧 Tool evaluation:[/] {len(request.steps)} steps")
    trace = convert_to_agent_trace(request)
    evaluator = ToolEvaluator()
    efficiency, tool_results = evaluator.evaluate_tool_chain(trace)
    console.print(f"  [green]✓[/] {len(trace.tool_calls)} calls, efficiency=[cyan]{efficiency:.0%}[/]")
    
    return {
        'efficiency': efficiency,
        'results': [
            {
                'signal': r.signal.value,
                'tool_name': r.tool_name,
                'confidence': r.confidence,
                'reason': r.reason,
                'expected_tool': r.expected_tool,
            }
            for r in tool_results
        ],
        'total_calls': len(trace.tool_calls),
    }


@router.post("/agent/self-correction")
async def analyze_self_correction(request: AgentTraceInput):
    """Quick endpoint for self-correction analysis only"""
    console.print(f"[bold magenta]🔁 Self-correction analysis:[/] {len(request.steps)} steps")
    trace = convert_to_agent_trace(request)
    detector = SelfCorrectionDetector()
    result = detector.analyze(trace)
    console.print(f"  [green]✓[/] awareness=[cyan]{result.self_awareness_score:.0%}[/] efficiency=[cyan]{result.correction_efficiency:.0%}[/]")
    
    return {
        'detected_error': result.detected_error,
        'correction_attempt': result.correction_attempt,
        'correction_success': result.correction_success,
        'loops_before_fix': result.loops_before_fix,
        'self_awareness_score': result.self_awareness_score,
        'correction_efficiency': result.correction_efficiency,
        'reason': result.reason,
    }


# ============================================
# Database Persistence Endpoints
# ============================================

class SaveAnalysisRequest(BaseModel):
    trace: Optional[AgentTraceInput] = None  # Single-agent format
    session: Optional[AgentSessionInput] = None  # Multi-agent format
    results: dict
    name: Optional[str] = None
    
    def is_multi_agent(self) -> bool:
        """Check if this is a multi-agent session"""
        return self.session is not None
    
    def get_session(self) -> AgentSessionInput:
        """Get session input, converting from trace if needed"""
        if self.session:
            return self.session
        if self.trace:
            return self.trace.to_session()
        raise ValueError("Either trace or session must be provided")
    
    def get_trace(self) -> AgentTraceInput:
        """Get trace input, converting from session if needed"""
        if self.trace:
            return self.trace
        if self.session:
            return _session_to_trace(self.session)
        raise ValueError("Either trace or session must be provided")


class SavedTraceResponse(BaseModel):
    id: int
    name: Optional[str]
    task: str
    success: Optional[bool]
    total_cost: Optional[float]
    created_at: str
    overall_score: Optional[float]
    step_count: int


@router.post("/agent/traces", response_model=SavedTraceResponse)
async def save_agent_trace(request: SaveAnalysisRequest, db: Session = Depends(get_db)):
    """Save an analyzed agent trace to the database - supports both single and multi-agent formats"""
    is_multi_agent = request.is_multi_agent()
    session_input = request.get_session()  # Always work with session format internally
    results = request.results
    
    # Get task from initial_task or first turn
    task_desc = session_input.initial_task or ""
    if not task_desc and session_input.turns:
        task_desc = session_input.turns[0].user_message or "Agent Trace"
    console.print(f"[bold blue]💾 Saving trace:[/] '{task_desc[:50]}...' ({'multi-agent' if is_multi_agent else 'single-agent'})")
    
    # Create the dataset record (was session)
    db_dataset = AgentSession(  # Uses alias -> Dataset
        name=request.name or task_desc[:100],
        task=session_input.initial_task,  # new column name
        success=session_input.success,
        total_cost=session_input.total_cost,
    )
    db.add(db_dataset)
    db.flush()
    
    # Create agent definitions - store all agents for multi-agent, or default for single
    agent_def_map = {}  # agent_id -> db_agent
    agents_to_save = session_input.agents or [AgentDef(id="agent", name="Agent", role="primary")]
    
    for agent in agents_to_save:
        db_agent = AgentDefinition(  # Uses alias -> Agent
            dataset_id=db_dataset.id,  # new column name
            agent_id=agent.id,
            name=agent.name or agent.id,
            role=agent.role,
            description=agent.description,
            capabilities_json=agent.capabilities if agent.capabilities else None,
            tools_available_json=[t.dict() for t in agent.tools_available] if agent.tools_available else None,
        )
        db.add(db_agent)
        db.flush()
        agent_def_map[agent.id] = db_agent
    
    # Add conversations (was turns) and their messages/steps
    total_steps = 0
    for conv_idx, turn in enumerate(session_input.turns):
        # Get final response from the last interaction that has one
        final_response = None
        responding_agent_id = None
        for interaction in reversed(turn.agent_interactions or []):
            if interaction.agent_response:
                final_response = interaction.agent_response
                responding_agent_id = interaction.agent_id
                break
        
        db_conversation = SessionTurn(  # Uses alias -> Conversation
            dataset_id=db_dataset.id,
            conversation_index=turn.turn_index if turn.turn_index is not None else conv_idx,
            user_input=turn.user_message,  # new column name
            final_response=final_response,
            responding_agent_id=responding_agent_id,
        )
        db.add(db_conversation)
        db.flush()
        
        # Create messages (was interactions) for each agent in this turn
        for seq, interaction in enumerate(turn.agent_interactions or []):
            agent_def = agent_def_map.get(interaction.agent_id)
            
            db_message = AgentInteraction(  # Uses alias -> Message
                conversation_id=db_conversation.id,  # new column name
                sequence=seq,
                agent_id=interaction.agent_id,
                content=interaction.agent_response,  # new column name
                latency_ms=interaction.latency_ms,
            )
            db.add(db_message)
            db.flush()
            
            # Add steps for this message
            for step_idx, step in enumerate(interaction.agent_steps or []):
                # Determine step type
                if step.tool_call:
                    step_type = "tool_call"
                elif step.thought:
                    step_type = "thought"
                elif step.observation:
                    step_type = "observation"
                else:
                    step_type = "action"
                
                tool_call = step.tool_call
                db_step = AgentStepDB(  # Uses alias -> Step
                    message_id=db_message.id,  # new column name
                    step_index=step_idx,
                    step_type=step_type,
                    content=step.thought or step.action or step.observation,
                    tool_name=tool_call.tool_name if tool_call else None,
                    tool_parameters_json=tool_call.parameters if tool_call and tool_call.parameters else None,
                    tool_result=tool_call.result if tool_call else None,
                    tool_error=tool_call.error if tool_call else None,
                )
                db.add(db_step)
                total_steps += 1
    
    # Create analysis record with results
    db_analysis = AgentAnalysis(  # Uses alias -> Analysis
        dataset_id=db_dataset.id,  # new column name
        status="completed",
        analysis_types_json=results.get('analysis_types', []),
        overall_score=results.get('overall_score'),
        conversation_result_json=results.get('conversation'),
        trajectory_result_json=results.get('trajectory'),
        tools_result_json=results.get('tools'),
        self_correction_result_json=results.get('self_correction'),
        per_agent_scores_json=results.get('per_agent_scores'),
        coordination_score=results.get('coordination_score'),
    )
    db.add(db_analysis)
    
    db.commit()
    db.refresh(db_dataset)
    db.refresh(db_analysis)
    
    console.print(f"  [green]✓[/] Saved as ID=[bold]{db_dataset.id}[/] with {len(session_input.turns)} turns, {total_steps} steps, {len(agents_to_save)} agents")
    
    return SavedTraceResponse(
        id=db_dataset.id,
        name=db_dataset.name,
        task=task_desc,
        success=db_dataset.success,
        total_cost=db_dataset.total_cost,
        created_at=db_dataset.created_at.isoformat(),
        overall_score=db_analysis.overall_score,
        step_count=total_steps,
    )


@router.get("/agent/traces")
async def list_agent_traces(db: Session = Depends(get_db), limit: int = 50, offset: int = 0):
    """List all saved agent traces with their latest analysis"""
    console.print(f"[dim]📋 Listing traces: limit={limit} offset={offset}[/]")
    traces = db.query(AgentTraceDB).order_by(AgentTraceDB.created_at.desc()).offset(offset).limit(limit).all()
    console.print(f"  [green]✓[/] Found {len(traces)} traces")
    
    result = []
    for t in traces:
        # Get latest completed analysis for this trace
        latest_analysis = db.query(AnalysisJobDB).filter(
            AnalysisJobDB.dataset_id == t.id,  # new column name
            AnalysisJobDB.status == 'completed'
        ).order_by(AnalysisJobDB.completed_at.desc()).first()
        
        result.append({
            'id': t.id,
            'name': t.name,
            'task': t.task or (t.conversations[0].user_input if t.conversations else "Agent Trace"),  # new column names
            'success': t.success,
            'total_cost': t.total_cost,
            'created_at': t.created_at.isoformat(),
            'overall_score': latest_analysis.overall_score if latest_analysis else None,
            'step_count': len(t.conversations),  # new column name
            'analysis_types': latest_analysis.analysis_types_json if latest_analysis and latest_analysis.analysis_types_json else [],  # JSON column
            'has_analysis': latest_analysis is not None,
        })
    
    return result


@router.get("/agent/traces/{trace_id}")
async def get_agent_trace(trace_id: int, db: Session = Depends(get_db)):
    """Get a specific agent trace with all details and latest analysis - preserves multi-agent format"""
    console.print(f"[dim]🔍 Getting trace ID={trace_id}[/]")
    trace = db.query(AgentTraceDB).filter(AgentTraceDB.id == trace_id).first()
    
    if not trace:
        console.print(f"  [red]✗[/] Trace not found")
        raise HTTPException(status_code=404, detail="Trace not found")
    
    # Get agent definitions to determine if multi-agent
    agent_defs = db.query(AgentDefinition).filter(AgentDefinition.dataset_id == trace_id).all()  # new column name
    is_multi_agent = len(agent_defs) > 1 or (len(agent_defs) == 1 and agent_defs[0].agent_id != "agent")
    
    # Build agents array
    agents = []
    for agent_def in agent_defs:
        agent = {
            'id': agent_def.agent_id,
            'name': agent_def.name,
            'role': agent_def.role,
        }
        if agent_def.description:
            agent['description'] = agent_def.description
        if agent_def.capabilities_json:
            agent['capabilities'] = agent_def.capabilities_json  # JSON column
        if agent_def.tools_available_json:
            agent['tools_available'] = agent_def.tools_available_json  # JSON column
        agents.append(agent)
    
    # Rebuild turns - multi-agent format with agent_interactions
    turns = []
    for db_conv in trace.conversations:  # new column name
        # Sort messages by sequence
        sorted_messages = sorted(db_conv.messages, key=lambda x: x.sequence or 0)  # new column name
        
        agent_interactions = []
        for message in sorted_messages:  # renamed from interaction
            # Build steps for this message
            steps = []
            for step in sorted(message.steps, key=lambda x: x.step_index):
                step_data = {}
                if step.step_type == 'thought' and step.content:
                    step_data['thought'] = step.content
                elif step.step_type == 'action' and step.content:
                    step_data['action'] = step.content
                elif step.step_type == 'observation' and step.content:
                    step_data['observation'] = step.content
                
                if step.step_type == 'tool_call' and step.tool_name:
                    step_data['tool_call'] = {
                        'tool_name': step.tool_name,
                        'parameters': step.tool_parameters_json or {},  # JSON column
                        'result': step.tool_result,
                        'error': step.tool_error,
                    }
                
                if step_data:
                    steps.append(step_data)
            
            interaction_data = {
                'agent_id': message.agent_id,
                'agent_steps': steps,
            }
            if message.content:  # new column name
                interaction_data['agent_response'] = message.content
            if message.latency_ms:
                interaction_data['latency_ms'] = message.latency_ms
            
            agent_interactions.append(interaction_data)
        
        turn_data = {
            'turn_index': db_conv.conversation_index,  # new column name
            'user_message': db_conv.user_input,  # new column name
            'agent_interactions': agent_interactions,
        }
        turns.append(turn_data)
    
    # Get latest completed analysis
    latest_analysis = db.query(AnalysisJobDB).filter(
        AnalysisJobDB.dataset_id == trace_id,  # new column name
        AnalysisJobDB.status == 'completed'
    ).order_by(AnalysisJobDB.completed_at.desc()).first()
    
    # Rebuild results from analysis
    results = {
        'overall_score': latest_analysis.overall_score if latest_analysis else None,
        'analysis_types': latest_analysis.analysis_types_json if latest_analysis and latest_analysis.analysis_types_json else [],  # JSON column
    }
    
    if latest_analysis:
        if latest_analysis.conversation_result_json:
            results['conversation'] = latest_analysis.conversation_result_json  # JSON column
        if latest_analysis.trajectory_result_json:
            results['trajectory'] = latest_analysis.trajectory_result_json
        if latest_analysis.tools_result_json:
            results['tools'] = latest_analysis.tools_result_json
        if latest_analysis.self_correction_result_json:
            results['self_correction'] = latest_analysis.self_correction_result_json
        if hasattr(latest_analysis, 'per_agent_scores_json') and latest_analysis.per_agent_scores_json:
            results['per_agent_scores'] = latest_analysis.per_agent_scores_json
        if hasattr(latest_analysis, 'coordination_score') and latest_analysis.coordination_score:
            results['coordination_score'] = latest_analysis.coordination_score
    
    console.print(f"  [green]✓[/] Found trace with {len(turns)} turns, {len(agents)} agents")
    
    return {
        'id': trace.id,
        'name': trace.name,
        'created_at': trace.created_at.isoformat(),
        'trace': {
            'agents': agents,
            'initial_task': trace.task,  # new column name
            'turns': turns,
            'success': trace.success,
            'total_cost': trace.total_cost,
        },
        'results': results,
    }


@router.delete("/agent/traces/{trace_id}")
async def delete_agent_trace(trace_id: int, db: Session = Depends(get_db)):
    """Delete an agent trace"""
    console.print(f"[bold red]🗑️  Deleting trace ID={trace_id}[/]")
    trace = db.query(AgentTraceDB).filter(AgentTraceDB.id == trace_id).first()
    
    if not trace:
        console.print(f"  [red]✗[/] Trace not found")
        raise HTTPException(status_code=404, detail="Trace not found")
    
    db.delete(trace)
    db.commit()
    console.print(f"  [green]✓[/] Deleted successfully")
    
    return {'message': 'Trace deleted successfully'}


# ============================================
# Background Job System - Real-time Saving
# ============================================

# Store running background tasks
_background_tasks: dict = {}


def save_session_to_db(db: Session, session_input: AgentSessionInput) -> tuple:
    """Save multi-agent session to database, return the DB object with ID"""
    from datetime import datetime as dt
    
    task_desc = session_input.task[:100] if session_input.task else "Untitled"
    
    # Create dataset (was AgentSession)
    db_dataset = AgentSession(  # Uses alias -> Dataset
        name=task_desc,
        task=session_input.initial_task,  # new column name
        success=session_input.success,
        total_cost=session_input.total_cost,
        metadata_json=session_input.session_metadata.dict() if session_input.session_metadata else None,
    )
    db.add(db_dataset)
    db.flush()
    
    # Save agent definitions
    agent_db_map = {}
    for agent_def in (session_input.agents or []):
        db_agent = AgentDefinition(  # Uses alias -> Agent
            dataset_id=db_dataset.id,  # new column name
            agent_id=agent_def.id,
            name=agent_def.name,
            role=agent_def.role,
            description=agent_def.description,
            capabilities_json=agent_def.capabilities if agent_def.capabilities else None,
            tools_available_json=[t.dict() for t in agent_def.tools_available] if agent_def.tools_available else None,
            config_json=agent_def.config if agent_def.config else None,
        )
        db.add(db_agent)
        db.flush()
        agent_db_map[agent_def.id] = db_agent
    
    # If no agents defined, create a default one
    if not agent_db_map:
        db_agent = AgentDefinition(
            dataset_id=db_dataset.id,
            agent_id="agent",
            name="Agent",
            role="primary",
        )
        db.add(db_agent)
        db.flush()
        agent_db_map["agent"] = db_agent
    
    # Save conversations (was SessionTurn)
    conversation_db_map = {}
    for conv_idx, turn in enumerate(session_input.turns):
        # Find final response from interactions
        final_response = None
        responding_agent = None
        for interaction in turn.agent_interactions:
            if interaction.agent_response:
                final_response = interaction.agent_response
                responding_agent = interaction.agent_id
        
        db_conversation = SessionTurn(  # Uses alias -> Conversation
            dataset_id=db_dataset.id,
            conversation_index=turn.turn_index if turn.turn_index is not None else conv_idx,
            user_input=turn.user_message,  # new column name
            final_response=final_response,
            responding_agent_id=responding_agent,
        )
        db.add(db_conversation)
        db.flush()
        conversation_db_map[conv_idx] = db_conversation
        
        # Save messages (was AgentInteraction)
        for seq, interaction in enumerate(turn.agent_interactions):
            # Get or create agent definition
            if interaction.agent_id not in agent_db_map:
                db_agent = AgentDefinition(
                    dataset_id=db_dataset.id,
                    agent_id=interaction.agent_id,
                    name=interaction.agent_id,
                )
                db.add(db_agent)
                db.flush()
                agent_db_map[interaction.agent_id] = db_agent
            
            db_message = AgentInteraction(  # Uses alias -> Message
                conversation_id=db_conversation.id,  # new column name
                sequence=seq,
                agent_id=interaction.agent_id,
                content=interaction.agent_response,  # new column name
                tool_execution_json=interaction.tool_execution_result.dict() if interaction.tool_execution_result else None,
            )
            db.add(db_message)
            db.flush()
            
            # Save steps
            for step_idx, step in enumerate(interaction.agent_steps or []):
                # Determine step type
                if step.tool_call:
                    step_type = "tool_call"
                    content = None
                elif step.thought:
                    step_type = "thought"
                    content = step.thought
                elif step.action:
                    step_type = "action"
                    content = step.action
                elif step.observation:
                    step_type = "observation"
                    content = step.observation
                else:
                    continue
                
                db_step = AgentStepDB(  # Uses alias -> Step
                    message_id=db_message.id,  # new column name
                    step_index=step_idx,
                    step_type=step_type,
                    content=content,
                    tool_name=step.tool_call.tool_name if step.tool_call else None,
                    tool_parameters_json=step.tool_call.parameters if step.tool_call else None,
                    tool_result=step.tool_call.result if step.tool_call else None,
                    tool_error=step.tool_call.error if step.tool_call else None,
                )
                db.add(db_step)
    
    db.commit()
    db.refresh(db_dataset)
    return db_dataset, conversation_db_map


def save_trace_to_db(db: Session, trace_input: AgentTraceInput) -> tuple:
    """Save simple trace to database by converting to session format"""
    session_input = trace_input.to_session()
    return save_session_to_db(db, session_input)


def run_background_analysis(analysis_id: int, resume_from: int = 0):
    """Run analysis in background thread with real-time saving"""
    from datetime import datetime
    
    db = SessionLocal()
    try:
        analysis = db.query(AnalysisJobDB).filter(AnalysisJobDB.id == analysis_id).first()
        if not analysis:
            return
        
        trace_id = analysis.dataset_id  # new column name
        db_trace = db.query(AgentTraceDB).filter(AgentTraceDB.id == trace_id).first()
        if not db_trace:
            analysis.status = "failed"
            analysis.error_message = "Trace not found"
            db.commit()
            return
        
        # Get analysis types
        analysis_types = json.loads(analysis.analysis_types_json)
        if 'full_all' in analysis_types:
            analysis_types = ['conversation', 'trajectory', 'tools', 'self_correction']
        elif 'full_agent' in analysis_types:
            analysis_types = ['trajectory', 'tools', 'self_correction']
        
        # Update status
        analysis.status = "running"
        analysis.started_at = datetime.utcnow()
        analysis.total_steps = len(analysis_types)
        analysis.current_step = 0
        db.commit()
        
        console.print(f"[bold cyan]🔄 Analysis {analysis_id}:[/] Starting (trace_id={trace_id}, resume_from={resume_from})")
        
        # Rebuild trace for internal analyzers
        turns = db_trace.turns
        
        # DEBUG: Log what we're rebuilding
        total_db_steps = 0
        tool_call_steps = 0
        for t in turns:
            for interaction in t.interactions:
                for s in interaction.steps:
                    total_db_steps += 1
                    if s.step_type == 'tool_call' and s.tool_name:
                        tool_call_steps += 1
                        console.print(f"    [dim]DB tool call: {s.tool_name}[/]")
        console.print(f"  [dim]DB steps: {total_db_steps}, tool calls: {tool_call_steps}[/]")
        
        trace_input = AgentTraceInput(
            initial_task=db_trace.initial_task,
            turns=[
                AgentTurn(
                    user_message=t.user_message,
                    agent_response=t.final_response,
                    agent_steps=[
                        AgentStepInput(
                            thought=s.content if s.step_type == 'thought' else None,
                            action=s.content if s.step_type == 'action' else None,
                            tool_call=ToolCallInput(
                                tool_name=s.tool_name,
                                parameters=json.loads(s.tool_parameters_json) if s.tool_parameters_json else {},
                                result=s.tool_result,
                                error=s.tool_error,
                            ) if s.step_type == 'tool_call' and s.tool_name else None,
                        )
                        for interaction in t.interactions
                        for s in interaction.steps
                    ]
                )
                for t in turns
            ],
            success=db_trace.success,
            total_cost=db_trace.total_cost,
        )
        trace = convert_to_agent_trace(trace_input)
        console.print(f"  [dim]Rebuilt trace: {trace.step_count} steps, {len(trace.tool_calls)} tool calls[/]")
        
        results = {}
        
        # Run conversation analysis with REAL-TIME turn saving and ROLLING CONTEXT
        if 'conversation' in analysis_types:
            analysis.current_step += 1
            analysis.current_analysis = "conversation"
            db.commit()
            
            detector = ReAskDetector(
                ccm_model="gpt-5-nano",
                rdm_model="gpt-5-nano",
                judge_model="gpt-5-mini",
                similarity_threshold=0.5,
                use_llm_confirmation=True,
                use_llm_judge_fallback=True
            )
            
            turn_results = []
            turn_summaries = []
            ccm = rdm = llm_judge = hallucination = bad = 0
            rolling_summary = ""
            
            # Helper to build full response from DB turn (includes ALL agent interactions)
            def build_full_response_from_db(db_turn) -> str:
                parts = []
                sorted_interactions = sorted(db_turn.interactions, key=lambda x: x.sequence or 0)
                for interaction in sorted_interactions:
                    agent_id = interaction.agent_id
                    # Include steps
                    for step in sorted(interaction.steps, key=lambda x: x.step_index):
                        if step.step_type == 'thought' and step.content:
                            parts.append(f"[{agent_id}] Thought: {step.content}")
                        if step.step_type == 'tool_call' and step.tool_name:
                            if step.tool_result:
                                parts.append(f"[{agent_id}] Tool '{step.tool_name}': {step.tool_result}")
                            elif step.tool_error:
                                parts.append(f"[{agent_id}] Tool '{step.tool_name}' error: {step.tool_error}")
                    # Include response
                    if interaction.agent_response:
                        parts.append(f"[{agent_id}]: {interaction.agent_response}")
                return "\n".join(parts) if parts else (db_turn.final_response or "(no response)")
            
            for turn_idx, db_turn in enumerate(turns):
                # Check if already analyzed (for retry)
                existing = db.query(TurnAnalysisResult).filter(
                    TurnAnalysisResult.analysis_id == analysis_id,
                    TurnAnalysisResult.turn_id == db_turn.id
                ).first()
                
                # Build full response for this turn
                full_response = build_full_response_from_db(db_turn)
                user_msg_text = db_turn.user_message or ""
                
                if existing and turn_idx < resume_from:
                    # Use existing result but still update rolling summary
                    turn_result = {
                        'step_index': turn_idx,
                        'is_bad': existing.is_bad,
                        'detection_type': existing.detection_type,
                        'confidence': existing.confidence,
                        'reason': existing.reason,
                    }
                    turn_results.append(turn_result)
                    if existing.is_bad:
                        bad += 1
                    if existing.detection_type == 'ccm':
                        ccm += 1
                    elif existing.detection_type == 'rdm':
                        rdm += 1
                    elif existing.detection_type == 'llm_judge':
                        llm_judge += 1
                    elif existing.detection_type == 'hallucination':
                        hallucination += 1
                    
                    # Still generate summary for context continuity
                    summary = detector.generate_turn_summary(
                        turn_idx, user_msg_text, full_response, rolling_summary
                    )
                    rolling_summary = summary.get('summary', '')
                    turn_summaries.append({'turn_index': turn_idx, **summary})
                    continue
                
                # Analyze this turn WITH CONTEXT
                user_msg = ReAskMessage.user(user_msg_text)
                assistant_msg = ReAskMessage.assistant(full_response)
                follow_up = ReAskMessage.user(turns[turn_idx + 1].user_message) if turn_idx + 1 < len(turns) else None
                
                result = detector.evaluate_response_with_context(
                    user_msg, assistant_msg, follow_up,
                    conversation_context=rolling_summary,
                    turn_index=turn_idx
                )
                detection_type = result.detection_type.value
                
                # Generate rolling summary for next turn
                summary = detector.generate_turn_summary(
                    turn_idx, user_msg_text, full_response, rolling_summary
                )
                rolling_summary = summary.get('summary', '')
                turn_summaries.append({'turn_index': turn_idx, **summary})
                
                # SAVE IMMEDIATELY to TurnAnalysisResult
                if existing:
                    existing.is_bad = result.is_bad
                    existing.detection_type = detection_type
                    existing.confidence = result.confidence
                    existing.reason = result.reason
                else:
                    turn_result_db = TurnAnalysisResult(
                        analysis_id=analysis_id,
                        turn_id=db_turn.id,
                        turn_index=turn_idx,
                        is_bad=result.is_bad,
                        detection_type=detection_type,
                        confidence=result.confidence,
                        reason=result.reason,
                    )
                    db.add(turn_result_db)
                
                analysis.last_successful_turn = turn_idx
                db.commit()
                
                turn_result = {
                    'step_index': turn_idx,
                    'is_bad': result.is_bad,
                    'detection_type': detection_type,
                    'confidence': result.confidence,
                    'reason': result.reason,
                }
                turn_results.append(turn_result)
                
                if result.is_bad:
                    bad += 1
                if detection_type == 'ccm':
                    ccm += 1
                elif detection_type == 'rdm':
                    rdm += 1
                elif detection_type == 'llm_judge':
                    llm_judge += 1
                elif detection_type == 'hallucination':
                    hallucination += 1
                
                console.print(f"    Turn {turn_idx + 1}/{len(turns)}: {'❌' if result.is_bad else '✅'} {detection_type}")
            
            total_responses = len(turns)
            good = total_responses - bad
            avg_conf = sum(r['confidence'] for r in turn_results) / len(turn_results) if turn_results else 0
            
            conv_result = {
                'total_responses': total_responses,
                'good_responses': good,
                'bad_responses': bad,
                'ccm_detections': ccm,
                'rdm_detections': rdm,
                'llm_judge_detections': llm_judge,
                'hallucination_detections': hallucination,
                'results': turn_results,
                'turn_summaries': turn_summaries,
                'reason': 'Conversation analysis complete (with rolling context).',
                'avg_confidence': avg_conf,
            }
            results['conversation'] = conv_result
            analysis.conversation_result_json = json.dumps(conv_result)
            console.print(f"    [green]✓[/] Conversation: {bad}/{total_responses} issues [dim](context-aware)[/]")
            db.commit()
        
        # Run trajectory analysis
        if 'trajectory' in analysis_types:
            analysis.current_step += 1
            analysis.current_analysis = "trajectory"
            db.commit()
            
            analyzer = TrajectoryAnalyzer()
            result = analyzer.analyze(trace)
            traj_result = {
                'signal': result.signal.value,
                'confidence': result.confidence,
                'efficiency_score': result.efficiency_score,
                'circular_count': result.circular_count,
                'regression_count': result.regression_count,
                'reason': result.reason,
            }
            results['trajectory'] = traj_result
            analysis.trajectory_result_json = json.dumps(traj_result)
            db.commit()
        
        # Run tool evaluation
        if 'tools' in analysis_types:
            analysis.current_step += 1
            analysis.current_analysis = "tools"
            db.commit()
            
            evaluator = ToolEvaluator()
            efficiency, tool_results = evaluator.evaluate_tool_chain(trace)
            tools_result = {
                'efficiency': efficiency,
                'results': [{'signal': r.signal.value, 'tool_name': r.tool_name, 'confidence': r.confidence, 'reason': r.reason} for r in tool_results],
                'total_calls': len(trace.tool_calls),
            }
            results['tools'] = tools_result
            analysis.tools_result_json = json.dumps(tools_result)
            db.commit()
        
        # Run self-correction detection
        if 'self_correction' in analysis_types:
            analysis.current_step += 1
            analysis.current_analysis = "self_correction"
            db.commit()
            
            sc_detector = SelfCorrectionDetector()
            result = sc_detector.analyze(trace)
            sc_result = {
                'detected_error': result.detected_error,
                'correction_attempt': result.correction_attempt,
                'correction_success': result.correction_success,
                'loops_before_fix': result.loops_before_fix,
                'self_awareness_score': result.self_awareness_score,
                'correction_efficiency': result.correction_efficiency,
                'reason': result.reason,
            }
            results['self_correction'] = sc_result
            analysis.self_correction_result_json = json.dumps(sc_result)
            db.commit()
        
        # Calculate overall score
        overall_score = 0.0
        score_count = 0
        
        if 'conversation' in results:
            total = results['conversation']['total_responses']
            good = results['conversation']['good_responses']
            conv_score = good / total if total > 0 else 1.0
            overall_score += conv_score
            score_count += 1
        
        if 'trajectory' in results:
            overall_score += results['trajectory']['efficiency_score']
            score_count += 1
        
        if 'tools' in results:
            overall_score += results['tools']['efficiency']
            score_count += 1
        
        if 'self_correction' in results:
            overall_score += results['self_correction']['correction_efficiency']
            score_count += 1
        
        
        overall_score = overall_score / score_count if score_count > 0 else 0.0
        
        # Mark complete
        analysis.status = "completed"
        analysis.completed_at = datetime.utcnow()
        analysis.overall_score = overall_score
        analysis.current_analysis = None
        db.commit()
        
        console.print(f"[bold green]✅ Analysis {analysis_id}:[/] Complete! Score={overall_score:.0%}")
        
    except Exception as e:
        import traceback
        console.print(f"[bold red]❌ Analysis {analysis_id} failed:[/] {str(e)}")
        traceback.print_exc()
        analysis.status = "failed"
        analysis.error_message = str(e)
        analysis.retry_count = (analysis.retry_count or 0) + 1
        db.commit()
    finally:
        db.close()
        _background_tasks.pop(analysis_id, None)


class StartJobRequest(BaseModel):
    """Request to start a background job - accepts both single-agent and multi-agent formats"""
    trace: Optional[AgentTraceInput] = None  # Single-agent format
    session: Optional[AgentSessionInput] = None  # Multi-agent format
    analysis_types: List[str]
    
    def get_trace(self) -> AgentTraceInput:
        """Get trace input, converting from session if needed"""
        if self.trace:
            return self.trace
        if self.session:
            # Convert multi-agent session to simple trace for backwards compat
            turns = []
            for turn in self.session.turns:
                if turn.agent_interactions:
                    # Use first interaction as main turn
                    interaction = turn.agent_interactions[0]
                    steps = [
                        AgentStepInput(
                            thought=s.thought,
                            action=s.action,
                            observation=s.observation,
                            tool_call=s.tool_call
                        ) for s in (interaction.agent_steps or [])
                    ]
                    turns.append(AgentTurnInput(
                        user_message=turn.user_message,
                        agent_steps=steps,
                        agent_response=interaction.final_response or ""
                    ))
                else:
                    turns.append(AgentTurnInput(
                        user_message=turn.user_message,
                        agent_steps=[],
                        agent_response=turn.final_response or ""
                    ))
            return AgentTraceInput(
                name=self.session.name or self.session.initial_task,
                initial_task=self.session.initial_task,
                turns=turns,
                total_cost=self.session.total_cost
            )
        raise ValueError("Either trace or session must be provided")


class JobStatusResponse(BaseModel):
    id: int
    trace_id: int
    status: str
    current_step: int
    total_steps: int
    current_analysis: Optional[str]
    turn_results: Optional[List[dict]]
    result: Optional[dict]
    overall_score: Optional[float]
    error_message: Optional[str]
    retry_count: int = 0
    last_successful_turn: Optional[int]


@router.post("/agent/traces")
async def create_trace(trace: AgentTraceInput, db: Session = Depends(get_db)):
    """Create a trace in the database (without analysis)"""
    db_trace, _ = save_trace_to_db(db, trace)
    console.print(f"[bold blue]💾 Created trace ID={db_trace.id}[/]")
    
    return {
        'id': db_trace.id,
        'name': db_trace.name,
        'turn_count': len(db_trace.turns),
        'created_at': db_trace.created_at.isoformat(),
    }


@router.post("/agent/jobs")
async def start_analysis_job(request: StartJobRequest, db: Session = Depends(get_db)):
    """Save trace and start background analysis"""
    import threading
    
    console.print(f"[bold magenta]🚀 Starting analysis job[/]")
    
    # STEP 1: Save trace FIRST - convert session to trace if needed
    trace_input = request.get_trace()
    db_trace, turn_db_map = save_trace_to_db(db, trace_input)
    console.print(f"  [green]✓[/] Trace saved: ID={db_trace.id}")
    
    # STEP 2: Create analysis linked to trace
    analysis = AnalysisJobDB(
        dataset_id=db_trace.id,  # new column name
        status="pending",
        analysis_types_json=request.analysis_types,  # JSON column
        total_steps=len(request.analysis_types),
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    console.print(f"  [green]✓[/] Analysis created: ID={analysis.id}")
    
    # STEP 3: Start background thread
    thread = threading.Thread(target=run_background_analysis, args=(analysis.id,), daemon=True)
    thread.start()
    _background_tasks[analysis.id] = thread
    
    return {
        'job_id': analysis.id,
        'trace_id': db_trace.id,
        'status': 'pending',
    }


@router.post("/agent/jobs/{job_id}/retry")
async def retry_analysis_job(job_id: int, db: Session = Depends(get_db)):
    """Retry a failed analysis from where it left off"""
    import threading
    
    analysis = db.query(AnalysisJobDB).filter(AnalysisJobDB.id == job_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if analysis.status not in ['failed', 'completed']:
        raise HTTPException(status_code=400, detail="Can only retry failed or completed jobs")
    
    resume_from = (analysis.last_successful_turn or -1) + 1
    console.print(f"[bold yellow]🔄 Retrying job {job_id}:[/] resume_from={resume_from}")
    
    analysis.status = "pending"
    analysis.error_message = None
    db.commit()
    
    thread = threading.Thread(target=run_background_analysis, args=(analysis.id, resume_from), daemon=True)
    thread.start()
    _background_tasks[analysis.id] = thread
    
    return {'job_id': analysis.id, 'status': 'retrying', 'resume_from': resume_from}


@router.get("/agent/jobs/{job_id}")
async def get_job_status(job_id: int, db: Session = Depends(get_db)):
    """Get the status of an analysis job with real-time turn results"""
    analysis = db.query(AnalysisJobDB).filter(AnalysisJobDB.id == job_id).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get real-time turn results
    turn_results = db.query(TurnAnalysisResult).filter(
        TurnAnalysisResult.analysis_id == job_id
    ).order_by(TurnAnalysisResult.turn_index).all()
    
    turn_results_data = [
        {
            'step_index': tr.turn_index,
            'is_bad': tr.is_bad,
            'detection_type': tr.detection_type,
            'confidence': tr.confidence,
            'reason': tr.reason,
        }
        for tr in turn_results
    ]
    
    # Build result from stored JSON
    result = None
    if analysis.status == 'completed':
        result = {
            'conversation': json.loads(analysis.conversation_result_json) if analysis.conversation_result_json else None,
            'trajectory': json.loads(analysis.trajectory_result_json) if analysis.trajectory_result_json else None,
            'tools': json.loads(analysis.tools_result_json) if analysis.tools_result_json else None,
            'self_correction': json.loads(analysis.self_correction_result_json) if analysis.self_correction_result_json else None,
            'overall_score': analysis.overall_score,
            'analysis_types': json.loads(analysis.analysis_types_json) if analysis.analysis_types_json else [],
        }
    
    return JobStatusResponse(
        id=analysis.id,
        trace_id=analysis.dataset_id,  # new column name
        status=analysis.status,
        current_step=analysis.current_step or 0,
        total_steps=analysis.total_steps or 0,
        current_analysis=analysis.current_analysis,
        turn_results=turn_results_data if turn_results_data else None,
        result=result,
        overall_score=analysis.overall_score,
        error_message=analysis.error_message,
        retry_count=analysis.retry_count or 0,
        last_successful_turn=analysis.last_successful_turn,
    )


@router.get("/agent/jobs")
async def list_jobs(db: Session = Depends(get_db), limit: int = 20, status: Optional[str] = None):
    """List analysis jobs with optional status filter"""
    query = db.query(AnalysisJobDB)
    if status:
        query = query.filter(AnalysisJobDB.status == status)
    jobs = query.order_by(AnalysisJobDB.created_at.desc()).limit(limit).all()
    
    return [
        {
            'id': j.id,
            'trace_id': j.dataset_id,  # new column name
            'status': j.status,
            'current_step': j.current_step,
            'total_steps': j.total_steps,
            'current_analysis': j.current_analysis,
            'overall_score': j.overall_score,
            'retry_count': j.retry_count or 0,
            'created_at': j.created_at.isoformat() if j.created_at else None,
        }
        for j in jobs
    ]


@router.delete("/agent/jobs/{job_id}")
async def delete_job(job_id: int, db: Session = Depends(get_db)):
    """Delete an analysis job (keeps the trace)"""
    job = db.query(AnalysisJobDB).filter(AnalysisJobDB.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    db.delete(job)
    db.commit()
    return {'message': 'Job deleted'}

