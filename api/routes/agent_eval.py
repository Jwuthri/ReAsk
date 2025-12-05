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
    SelfCorrectionDetector, IntentDriftMeter,
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
    analysis_types: List[str]  # conversation, trajectory, tools, self_correction, intent_drift, coordination, full_agent, full_all
    
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
    
    Returns:
        {
            'per_agent_scores': { agent_id: { overall, tool_use, reasoning, handoff, ... } },
            'coordination_score': float
        }
    """
    if not session.agents or len(session.agents) <= 1:
        # Single agent - no per-agent breakdown needed
        return {}
    
    # Track metrics per agent
    agent_metrics = {}
    for agent in session.agents:
        agent_id = agent.id
        agent_metrics[agent_id] = {
            'tool_calls': 0,
            'tool_errors': 0,
            'reasoning_steps': 0,
            'handoffs': 0,
            'interactions_count': 0,
            'issues': [],
            'recommendations': [],
            'tools_available': set(t.name for t in (agent.tools_available or []) if t.name) if agent.tools_available else set(),
        }

    # Analyze each turn's interactions
    for turn in session.turns:
        if not turn.agent_interactions:
            continue
        
        for idx, interaction in enumerate(turn.agent_interactions):
            agent_id = interaction.agent_id
            if agent_id not in agent_metrics:
                continue
            
            metrics = agent_metrics[agent_id]
            metrics['interactions_count'] += 1
            
            # Analyze steps
            for step in (interaction.agent_steps or []):
                if step.tool_call:
                    metrics['tool_calls'] += 1
                    tool_name = step.tool_call.tool_name
                    
                    # Check if tool is authorized
                    if metrics['tools_available'] and tool_name not in metrics['tools_available']:
                        metrics['tool_errors'] += 1
                        metrics['issues'].append(f"Used unauthorized tool: {tool_name}")
                    
                    # Check for errors in tool call
                    if step.tool_call.error:
                        metrics['tool_errors'] += 1
                
                if step.thought:
                    metrics['reasoning_steps'] += 1
            
            # Check for handoffs (not the last interaction in turn)
            if idx < len(turn.agent_interactions) - 1:
                metrics['handoffs'] += 1
    
    # Compute scores per agent
    per_agent_scores = {}
    for agent in session.agents:
        agent_id = agent.id
        metrics = agent_metrics[agent_id]
        
        # Tool use score
        tool_use = None
        if metrics['tool_calls'] > 0:
            tool_use = max(0, 1.0 - (metrics['tool_errors'] / metrics['tool_calls']))
            if tool_use < 0.7:
                metrics['recommendations'].append(f"Improve tool selection - {metrics['tool_errors']} errors in {metrics['tool_calls']} calls")
        
        # Reasoning score (simplified - based on having reasoning steps)
        reasoning = None
        if metrics['reasoning_steps'] > 0:
            reasoning = min(1.0, metrics['reasoning_steps'] / max(metrics['interactions_count'], 1))
        
        # Handoff score (placeholder - would need more analysis)
        handoff = None
        if metrics['handoffs'] > 0:
            handoff = 0.85  # Default good score, would need context analysis
        
        # Calculate overall for this agent
        scores = [s for s in [tool_use, reasoning, handoff] if s is not None]
        if scores:
            overall = sum(scores) / len(scores)
        else:
            overall = 0.8  # Default score when no metrics available
        
        # Add tool-related recommendations
        if tool_use is not None and tool_use < 0.7:
            metrics['recommendations'].append("Consider validating tool parameters before execution")
        
        per_agent_scores[agent_id] = {
            'overall': round(overall, 2),
            'tool_use': round(tool_use, 2) if tool_use is not None else None,
            'reasoning': round(reasoning, 2) if reasoning is not None else None,
            'handoff': round(handoff, 2) if handoff is not None else None,
            'response_quality': 0.85,  # Placeholder
            'interactions_count': metrics['interactions_count'],
            'issues': metrics['issues'],
            'recommendations': metrics['recommendations'],
        }
    
    # Coordination score (how well agents worked together)
    # Higher if no duplicate work, proper handoffs, etc.
    coordination_score = 0.85  # Default good score
    
    # Penalize if any agent has issues
    total_issues = sum(len(s['issues']) for s in per_agent_scores.values())
    if total_issues > 0:
        coordination_score = max(0.5, coordination_score - (total_issues * 0.1))
    
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


class IntentDriftResultModel(BaseModel):
    drift_score: float
    step_index: int
    is_legitimate: bool
    reason: str
    drift_history: List[float]


class FullAnalysisResponse(BaseModel):
    trajectory: Optional[TrajectoryResult] = None
    tools: Optional[dict] = None
    self_correction: Optional[SelfCorrectionResultModel] = None
    intent_drift: Optional[IntentDriftResultModel] = None


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


def run_conversation_analysis(input_trace: AgentTraceInput) -> dict:
    """Run CCM/RDM/Hallucination/LLM Judge analysis on agent trace"""
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
    - intent_drift: Measure task alignment
    - full_agent: Run all agent analyses
    - full_all: Run all analyses (conversation + agent)
    """
    start_time = time.time()
    # Get trace input (handles both trace and session formats)
    trace_input = request.get_trace()
    task_preview = trace_input.task[:50] if trace_input.task else ""
    console.print(f"[bold cyan]📊 Starting analysis:[/] task='[dim]{task_preview}...[/]' types={request.analysis_types}")
    
    trace = convert_to_agent_trace(trace_input)
    results = {}
    
    analysis_types = list(request.analysis_types)  # Make a copy
    
    # Expand full_all to include everything
    if 'full_all' in analysis_types:
        analysis_types = ['conversation', 'trajectory', 'tools', 'self_correction', 'intent_drift']
    # Expand full_agent to include all agent analyses
    elif 'full_agent' in analysis_types:
        analysis_types = ['trajectory', 'tools', 'self_correction', 'intent_drift']
    
    # Run conversation analysis (CCM/RDM/Hallucination)
    if 'conversation' in analysis_types:
        console.print("  [yellow]→[/] Running conversation analysis (CCM/RDM/Hallucination)")
        conv_result = run_conversation_analysis(trace_input)
        results['conversation'] = conv_result
        console.print(f"    [green]✓[/] Conversation: {conv_result['bad_responses']}/{conv_result['total_responses']} issues")
    
    # Run trajectory analysis
    if 'trajectory' in analysis_types:
        console.print("  [yellow]→[/] Running trajectory analysis")
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
    
    # Run tool evaluation
    if 'tools' in analysis_types:
        console.print("  [yellow]→[/] Running tool evaluation")
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
                    'expected_tool': r.expected_tool,
                }
                for r in tool_results
            ],
            'total_calls': len(trace.tool_calls),
            'correct_count': sum(1 for r in tool_results if r.signal.value == 'correct'),
        }
        console.print(f"    [green]✓[/] Tools: {len(trace.tool_calls)} calls, efficiency=[cyan]{efficiency:.0%}[/]")
    
    # Run self-correction detection
    if 'self_correction' in analysis_types:
        console.print("  [yellow]→[/] Running self-correction detection")
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
    
    # Run intent drift analysis
    if 'intent_drift' in analysis_types:
        console.print("  [yellow]→[/] Running intent drift analysis")
        meter = IntentDriftMeter()
        result = meter.analyze(trace)
        results['intent_drift'] = {
            'drift_score': result.drift_score,
            'step_index': result.step_index,
            'is_legitimate': result.is_legitimate,
            'reason': result.reason,
            'drift_history': result.drift_history,
        }
        drift_color = "green" if result.drift_score < 0.35 else "yellow" if result.drift_score < 0.6 else "red"
        console.print(f"    [green]✓[/] Intent drift: score=[{drift_color}]{result.drift_score:.0%}[/] legitimate={result.is_legitimate}")
    
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
    
    if 'intent_drift' in results:
        overall_score += (1.0 - results['intent_drift']['drift_score'])
        score_count += 1
    
    results['overall_score'] = overall_score / score_count if score_count > 0 else 0.0
    results['analysis_types'] = analysis_types
    
    # Compute per-agent scores for multi-agent sessions
    session = request.get_session()
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
            analysis_types = ['conversation', 'trajectory', 'tools', 'self_correction', 'intent_drift']
        elif 'full_agent' in analysis_types:
            analysis_types = ['trajectory', 'tools', 'self_correction', 'intent_drift']
        
        total_analyses = len(analysis_types)
        current = 0
        
        console.print(f"  [dim]Running {total_analyses} analyses: {analysis_types}[/]")
        yield f"data: {json.dumps({'type': 'start', 'total': total_analyses})}\n\n"
        
        results = {}
        
        # Conversation analysis (CCM/RDM/Hallucination) - stream turn by turn
        if 'conversation' in analysis_types:
            current += 1
            turns = trace_input.turns
            total_turns = len(turns)
            
            console.print(f"  [yellow]→[/] [dim][stream][/] Running conversation analysis ({total_turns} turns)")
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
            ccm = rdm = llm_judge = hallucination = bad = 0
            
            # Process turn by turn
            messages_so_far = []
            for turn_idx, turn in enumerate(turns):
                # Build context
                user_msg = ReAskMessage.user(turn.user_message)
                assistant_msg = ReAskMessage.assistant(turn.agent_response)
                
                # Get follow-up if exists
                follow_up = None
                if turn_idx + 1 < total_turns:
                    follow_up = ReAskMessage.user(turns[turn_idx + 1].user_message)
                
                # Evaluate this turn
                result = detector.evaluate_response(user_msg, assistant_msg, follow_up)
                
                detection_type = result.detection_type.value
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
                'reason': 'Conversation analysis complete.',
                'avg_confidence': avg_conf,
            }
            results['conversation'] = conv_result
            console.print(f"    [green]✓[/] Conversation: {bad}/{total_responses} issues")
            
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
        
        # Intent drift
        if 'intent_drift' in analysis_types:
            current += 1
            console.print("  [yellow]→[/] [dim][stream][/] Running intent drift analysis")
            yield f"data: {json.dumps({'type': 'progress', 'current': current, 'total': total_analyses, 'analysis': 'intent_drift', 'status': 'running'})}\n\n"
            
            meter = IntentDriftMeter()
            result = meter.analyze(trace)
            results['intent_drift'] = {
                'drift_score': result.drift_score,
                'step_index': result.step_index,
                'is_legitimate': result.is_legitimate,
                'reason': result.reason,
                'drift_history': result.drift_history,
            }
            drift_color = "green" if result.drift_score < 0.35 else "yellow" if result.drift_score < 0.6 else "red"
            console.print(f"    [green]✓[/] Intent drift: score=[{drift_color}]{result.drift_score:.0%}[/] legitimate={result.is_legitimate}")
            
            yield f"data: {json.dumps({'type': 'result', 'analysis': 'intent_drift', 'data': results['intent_drift']})}\n\n"
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
        
        if 'intent_drift' in results:
            overall_score += (1.0 - results['intent_drift']['drift_score'])
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


@router.post("/agent/intent-drift")
async def analyze_intent_drift(request: AgentTraceInput):
    """Quick endpoint for intent drift analysis only"""
    console.print(f"[bold magenta]🎯 Intent drift analysis:[/] {len(request.steps)} steps")
    trace = convert_to_agent_trace(request)
    meter = IntentDriftMeter()
    result = meter.analyze(trace)
    drift_color = "green" if result.drift_score < 0.35 else "yellow" if result.drift_score < 0.6 else "red"
    console.print(f"  [green]✓[/] drift=[{drift_color}]{result.drift_score:.0%}[/] legitimate={result.is_legitimate}")
    
    return {
        'drift_score': result.drift_score,
        'step_index': result.step_index,
        'is_legitimate': result.is_legitimate,
        'reason': result.reason,
        'drift_history': result.drift_history,
    }


# ============================================
# Database Persistence Endpoints
# ============================================

class SaveAnalysisRequest(BaseModel):
    trace: Optional[AgentTraceInput] = None  # Single-agent format
    session: Optional[AgentSessionInput] = None  # Multi-agent format
    results: dict
    name: Optional[str] = None
    
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
    """Save an analyzed agent trace to the database"""
    trace_input = request.get_trace()
    results = request.results
    
    # Get task from initial_task or first turn
    task_desc = trace_input.initial_task or (trace_input.turns[0].user_message if trace_input.turns else "Agent Trace")
    console.print(f"[bold blue]💾 Saving trace:[/] '{task_desc[:50]}...'")
    
    # Create the session record (no analysis fields - those go in AgentAnalysis)
    db_session = AgentSession(
        name=request.name or task_desc[:100],
        initial_task=trace_input.initial_task,
        success=trace_input.success,
        total_cost=trace_input.total_cost,
    )
    db.add(db_session)
    db.flush()
    
    # Create a default agent definition (single-agent trace)
    db_agent = AgentDefinition(
        session_id=db_session.id,
        agent_id="agent",
        name="Agent",
        role="primary",
    )
    db.add(db_agent)
    db.flush()
    
    # Add turns and their steps
    total_steps = 0
    for turn_idx, turn in enumerate(trace_input.turns):
        db_turn = SessionTurn(
            session_id=db_session.id,
            turn_index=turn_idx,
            user_message=turn.user_message,
            final_response=turn.agent_response,
            responding_agent_id="agent",
        )
        db.add(db_turn)
        db.flush()
        
        # Create interaction for this turn
        db_interaction = AgentInteraction(
            turn_id=db_turn.id,
            agent_def_id=db_agent.id,
            sequence=0,
            agent_id="agent",
            agent_response=turn.agent_response,
        )
        db.add(db_interaction)
        db.flush()
        
        # Add steps for this turn under the interaction
        for step_idx, step in enumerate(turn.agent_steps or []):
            # Determine step type
            if step.tool_call:
                step_type = "tool_call"
            elif step.thought:
                step_type = "thought"
            else:
                step_type = "action"
            
            db_step = AgentStepDB(
                interaction_id=db_interaction.id,
                step_index=step_idx,
                step_type=step_type,
                content=step.thought or step.action,
                tool_name=step.tool_call.tool_name if step.tool_call else None,
                tool_parameters_json=json.dumps(step.tool_call.parameters) if step.tool_call and step.tool_call.parameters else None,
                tool_result=step.tool_call.result if step.tool_call else None,
            )
            db.add(db_step)
            total_steps += 1
    
    # Create analysis record with results
    db_analysis = AgentAnalysis(
        session_id=db_session.id,
        status="completed",
        analysis_types_json=json.dumps(results.get('analysis_types', [])),
        overall_score=results.get('overall_score'),
        conversation_result_json=json.dumps(results.get('conversation')) if results.get('conversation') else None,
        trajectory_result_json=json.dumps(results.get('trajectory')) if results.get('trajectory') else None,
        tools_result_json=json.dumps(results.get('tools')) if results.get('tools') else None,
        self_correction_result_json=json.dumps(results.get('self_correction')) if results.get('self_correction') else None,
        intent_drift_result_json=json.dumps(results.get('intent_drift')) if results.get('intent_drift') else None,
    )
    db.add(db_analysis)
    
    db.commit()
    db.refresh(db_session)
    db.refresh(db_analysis)
    
    console.print(f"  [green]✓[/] Saved as ID=[bold]{db_session.id}[/] with {len(trace_input.turns)} turns, {total_steps} steps")
    
    return SavedTraceResponse(
        id=db_session.id,
        name=db_session.name,
        task=task_desc,
        success=db_session.success,
        total_cost=db_session.total_cost,
        created_at=db_session.created_at.isoformat(),
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
            AnalysisJobDB.session_id == t.id,
            AnalysisJobDB.status == 'completed'
        ).order_by(AnalysisJobDB.completed_at.desc()).first()
        
        result.append({
            'id': t.id,
            'name': t.name,
            'task': t.initial_task or (t.turns[0].user_message if t.turns else "Agent Trace"),
            'success': t.success,
            'total_cost': t.total_cost,
            'created_at': t.created_at.isoformat(),
            'overall_score': latest_analysis.overall_score if latest_analysis else None,
            'step_count': len(t.turns),
            'analysis_types': json.loads(latest_analysis.analysis_types_json) if latest_analysis and latest_analysis.analysis_types_json else [],
            'has_analysis': latest_analysis is not None,
        })
    
    return result


@router.get("/agent/traces/{trace_id}")
async def get_agent_trace(trace_id: int, db: Session = Depends(get_db)):
    """Get a specific agent trace with all details and latest analysis"""
    console.print(f"[dim]🔍 Getting trace ID={trace_id}[/]")
    trace = db.query(AgentTraceDB).filter(AgentTraceDB.id == trace_id).first()
    
    if not trace:
        console.print(f"  [red]✗[/] Trace not found")
        raise HTTPException(status_code=404, detail="Trace not found")
    
    # Rebuild turns with steps
    turns = []
    for db_turn in trace.turns:
        # Rebuild steps for this turn from interactions
        agent_steps = []
        for interaction in db_turn.interactions:
            for step in interaction.steps:
                tool_call = None
                if step.step_type == 'tool_call' and step.tool_name:
                    tool_call = {
                        'name': step.tool_name,
                        'parameters': json.loads(step.tool_parameters_json) if step.tool_parameters_json else {},
                        'result': step.tool_result,
                        'error': step.tool_error,
                    }
                agent_steps.append({
                    'thought': step.content if step.step_type == 'thought' else None,
                    'action': step.content if step.step_type == 'action' else None,
                    'observation': step.content if step.step_type == 'observation' else None,
                    'tool_call': tool_call,
                })
        
        turns.append({
            'user_message': db_turn.user_message,
            'agent_response': db_turn.final_response,
            'agent_steps': agent_steps,
        })
    
    # Get latest completed analysis
    latest_analysis = db.query(AnalysisJobDB).filter(
        AnalysisJobDB.session_id == trace_id,
        AnalysisJobDB.status == 'completed'
    ).order_by(AnalysisJobDB.completed_at.desc()).first()
    
    # Rebuild results from analysis
    results = {
        'overall_score': latest_analysis.overall_score if latest_analysis else None,
        'analysis_types': json.loads(latest_analysis.analysis_types_json) if latest_analysis and latest_analysis.analysis_types_json else [],
    }
    
    if latest_analysis:
        if latest_analysis.conversation_result_json:
            results['conversation'] = json.loads(latest_analysis.conversation_result_json)
        if latest_analysis.trajectory_result_json:
            results['trajectory'] = json.loads(latest_analysis.trajectory_result_json)
        if latest_analysis.tools_result_json:
            results['tools'] = json.loads(latest_analysis.tools_result_json)
        if latest_analysis.self_correction_result_json:
            results['self_correction'] = json.loads(latest_analysis.self_correction_result_json)
        if latest_analysis.intent_drift_result_json:
            results['intent_drift'] = json.loads(latest_analysis.intent_drift_result_json)
    
    console.print(f"  [green]✓[/] Found trace with {len(turns)} turns")
    
    return {
        'id': trace.id,
        'name': trace.name,
        'created_at': trace.created_at.isoformat(),
        'trace': {
            'initial_task': trace.initial_task,
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
    
    task_desc = session_input.task[:100]
    
    # Create session
    db_session = AgentSession(
        name=task_desc,
        initial_task=session_input.initial_task,
        success=session_input.success,
        total_cost=session_input.total_cost,
        metadata_json=json.dumps(session_input.session_metadata.dict()) if session_input.session_metadata else None,
    )
    db.add(db_session)
    db.flush()
    
    # Save agent definitions
    agent_db_map = {}
    for agent_def in (session_input.agents or []):
        db_agent = AgentDefinition(
            session_id=db_session.id,
            agent_id=agent_def.id,
            name=agent_def.name,
            role=agent_def.role,
            description=agent_def.description,
            capabilities_json=json.dumps(agent_def.capabilities) if agent_def.capabilities else None,
            tools_available_json=json.dumps([t.dict() for t in agent_def.tools_available]) if agent_def.tools_available else None,
            config_json=json.dumps(agent_def.config) if agent_def.config else None,
        )
        db.add(db_agent)
        db.flush()
        agent_db_map[agent_def.id] = db_agent
    
    # If no agents defined, create a default one
    if not agent_db_map:
        db_agent = AgentDefinition(
            session_id=db_session.id,
            agent_id="agent",
            name="Agent",
            role="primary",
        )
        db.add(db_agent)
        db.flush()
        agent_db_map["agent"] = db_agent
    
    # Save turns with interactions
    turn_db_map = {}
    for turn_idx, turn in enumerate(session_input.turns):
        # Find final response from interactions
        final_response = None
        responding_agent = None
        for interaction in turn.agent_interactions:
            if interaction.agent_response:
                final_response = interaction.agent_response
                responding_agent = interaction.agent_id
        
        db_turn = SessionTurn(
            session_id=db_session.id,
            turn_index=turn.turn_index if turn.turn_index is not None else turn_idx,
            user_message=turn.user_message,
            final_response=final_response,
            responding_agent_id=responding_agent,
        )
        db.add(db_turn)
        db.flush()
        turn_db_map[turn_idx] = db_turn
        
        # Save agent interactions
        for seq, interaction in enumerate(turn.agent_interactions):
            # Get or create agent definition
            if interaction.agent_id not in agent_db_map:
                db_agent = AgentDefinition(
                    session_id=db_session.id,
                    agent_id=interaction.agent_id,
                    name=interaction.agent_id,
                )
                db.add(db_agent)
                db.flush()
                agent_db_map[interaction.agent_id] = db_agent
            
            db_interaction = AgentInteraction(
                turn_id=db_turn.id,
                agent_def_id=agent_db_map[interaction.agent_id].id,
                sequence=seq,
                agent_id=interaction.agent_id,
                agent_response=interaction.agent_response,
                tool_execution_json=json.dumps(interaction.tool_execution_result.dict()) if interaction.tool_execution_result else None,
            )
            db.add(db_interaction)
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
                
                db_step = AgentStepDB(
                    interaction_id=db_interaction.id,
                    step_index=step_idx,
                    step_type=step_type,
                    content=content,
                    tool_name=step.tool_call.tool_name if step.tool_call else None,
                    tool_parameters_json=json.dumps(step.tool_call.parameters) if step.tool_call else None,
                    tool_result=step.tool_call.result if step.tool_call else None,
                    tool_error=step.tool_call.error if step.tool_call else None,
                )
                db.add(db_step)
    
    db.commit()
    db.refresh(db_session)
    return db_session, turn_db_map


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
        
        trace_id = analysis.session_id
        db_trace = db.query(AgentTraceDB).filter(AgentTraceDB.id == trace_id).first()
        if not db_trace:
            analysis.status = "failed"
            analysis.error_message = "Trace not found"
            db.commit()
            return
        
        # Get analysis types
        analysis_types = json.loads(analysis.analysis_types_json)
        if 'full_all' in analysis_types:
            analysis_types = ['conversation', 'trajectory', 'tools', 'self_correction', 'intent_drift']
        elif 'full_agent' in analysis_types:
            analysis_types = ['trajectory', 'tools', 'self_correction', 'intent_drift']
        
        # Update status
        analysis.status = "running"
        analysis.started_at = datetime.utcnow()
        analysis.total_steps = len(analysis_types)
        analysis.current_step = 0
        db.commit()
        
        console.print(f"[bold cyan]🔄 Analysis {analysis_id}:[/] Starting (trace_id={trace_id}, resume_from={resume_from})")
        
        # Rebuild trace for internal analyzers
        turns = db_trace.turns
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
        
        results = {}
        
        # Run conversation analysis with REAL-TIME turn saving
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
            ccm = rdm = llm_judge = hallucination = bad = 0
            
            for turn_idx, db_turn in enumerate(turns):
                # Check if already analyzed (for retry)
                existing = db.query(TurnAnalysisResult).filter(
                    TurnAnalysisResult.analysis_id == analysis_id,
                    TurnAnalysisResult.turn_id == db_turn.id
                ).first()
                
                if existing and turn_idx < resume_from:
                    # Use existing result
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
                    continue
                
                # Analyze this turn
                user_msg = ReAskMessage.user(db_turn.user_message)
                assistant_msg = ReAskMessage.assistant(db_turn.final_response or "")
                follow_up = ReAskMessage.user(turns[turn_idx + 1].user_message) if turn_idx + 1 < len(turns) else None
                
                result = detector.evaluate_response(user_msg, assistant_msg, follow_up)
                detection_type = result.detection_type.value
                
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
                'reason': 'Conversation analysis complete.',
                'avg_confidence': avg_conf,
            }
            results['conversation'] = conv_result
            analysis.conversation_result_json = json.dumps(conv_result)
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
        
        # Run intent drift analysis
        if 'intent_drift' in analysis_types:
            analysis.current_step += 1
            analysis.current_analysis = "intent_drift"
            db.commit()
            
            meter = IntentDriftMeter()
            result = meter.analyze(trace)
            drift_result = {
                'drift_score': result.drift_score,
                'step_index': result.step_index,
                'is_legitimate': result.is_legitimate,
                'reason': result.reason,
                'drift_history': result.drift_history,
            }
            results['intent_drift'] = drift_result
            analysis.intent_drift_result_json = json.dumps(drift_result)
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
        
        if 'intent_drift' in results:
            overall_score += (1.0 - results['intent_drift']['drift_score'])
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
                        ) for s in interaction.steps
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
        session_id=db_trace.id,
        status="pending",
        analysis_types_json=json.dumps(request.analysis_types),
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
            'intent_drift': json.loads(analysis.intent_drift_result_json) if analysis.intent_drift_result_json else None,
            'overall_score': analysis.overall_score,
            'analysis_types': json.loads(analysis.analysis_types_json) if analysis.analysis_types_json else [],
        }
    
    return JobStatusResponse(
        id=analysis.id,
        trace_id=analysis.session_id,
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
            'trace_id': j.session_id,
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

