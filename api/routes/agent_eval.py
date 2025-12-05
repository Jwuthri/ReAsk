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

from ..database import get_db, AgentTraceDB, AgentTurnDB, AgentStepDB, AnalysisJobDB, SessionLocal

from reask import (
    AgentTrace, AgentStep, ToolCall,
    TrajectoryAnalyzer, ToolEvaluator,
    SelfCorrectionDetector, IntentDriftMeter,
    AgentBenchmark,
    ReAskDetector, Message as ReAskMessage,
)

console = Console()

router = APIRouter()


# Request/Response models
class ToolCallInput(BaseModel):
    name: str
    parameters: dict
    result: Optional[str] = None
    error: Optional[str] = None


class AgentStepInput(BaseModel):
    thought: Optional[str] = None
    action: Optional[str] = None
    tool_call: Optional[ToolCallInput] = None


# A single turn: user message → agent steps → agent response
class AgentTurn(BaseModel):
    user_message: str
    agent_steps: Optional[List[AgentStepInput]] = None
    agent_response: str


class AgentTraceInput(BaseModel):
    initial_task: Optional[str] = None  # Optional context
    turns: List[AgentTurn]              # Conversation turns
    success: Optional[bool] = None
    total_cost: Optional[float] = None
    
    # Computed property for backwards compatibility
    @property
    def task(self) -> str:
        return self.initial_task or (self.turns[0].user_message if self.turns else "")


class AnalysisRequest(BaseModel):
    trace: AgentTraceInput
    analysis_types: List[str]  # conversation, trajectory, tools, self_correction, intent_drift, full_agent, full_all


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
                        name=step_input.tool_call.name,
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
    console.print(f"[bold cyan]📊 Starting analysis:[/] task='[dim]{request.trace.task[:50]}...[/]' types={request.analysis_types}")
    
    trace = convert_to_agent_trace(request.trace)
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
        conv_result = run_conversation_analysis(request.trace)
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
    
    duration = (time.time() - start_time) * 1000
    score_color = "green" if results['overall_score'] >= 0.7 else "yellow" if results['overall_score'] >= 0.4 else "red"
    console.print(f"[bold green]✅ Analysis complete:[/] overall_score=[bold {score_color}]{results['overall_score']:.0%}[/] [dim]({duration:.0f}ms)[/]")
    
    return results


@router.post("/agent/analyze/stream")
async def analyze_agent_trace_stream(request: AnalysisRequest):
    """
    Run analyses with streaming progress updates.
    """
    console.print(f"[bold cyan]📊 Starting streaming analysis:[/] task='[dim]{request.trace.task[:50]}...[/]' types={request.analysis_types}")
    start_time = time.time()
    
    async def event_generator():
        trace = convert_to_agent_trace(request.trace)
        
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
            turns = request.trace.turns
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
    trace: AgentTraceInput
    results: dict
    name: Optional[str] = None


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
    trace_input = request.trace
    results = request.results
    
    # Get task from initial_task or first turn
    task_desc = trace_input.initial_task or (trace_input.turns[0].user_message if trace_input.turns else "Agent Trace")
    console.print(f"[bold blue]💾 Saving trace:[/] '{task_desc[:50]}...'")
    
    # Create the trace record
    db_trace = AgentTraceDB(
        name=request.name or task_desc[:100],
        initial_task=trace_input.initial_task,
        success=trace_input.success,
        total_cost=trace_input.total_cost,
        overall_score=results.get('overall_score'),
        analysis_types=json.dumps(results.get('analysis_types', [])),
        conversation_result=json.dumps(results.get('conversation')) if results.get('conversation') else None,
        trajectory_result=json.dumps(results.get('trajectory')) if results.get('trajectory') else None,
        tools_result=json.dumps(results.get('tools')) if results.get('tools') else None,
        self_correction_result=json.dumps(results.get('self_correction')) if results.get('self_correction') else None,
        intent_drift_result=json.dumps(results.get('intent_drift')) if results.get('intent_drift') else None,
    )
    db.add(db_trace)
    db.flush()  # Get the ID
    
    # Add turns and their steps
    total_steps = 0
    for turn_idx, turn in enumerate(trace_input.turns):
        db_turn = AgentTurnDB(
            trace_id=db_trace.id,
            index=turn_idx,
            user_message=turn.user_message,
            agent_response=turn.agent_response,
        )
        db.add(db_turn)
        db.flush()
        
        # Add steps for this turn
        for step_idx, step in enumerate(turn.agent_steps or []):
            db_step = AgentStepDB(
                turn_id=db_turn.id,
                index=step_idx,
                thought=step.thought,
                action=step.action,
                observation=step.observation,
                tool_call_json=json.dumps(step.tool_call.dict()) if step.tool_call else None,
            )
            db.add(db_step)
            total_steps += 1
    
    db.commit()
    db.refresh(db_trace)
    
    console.print(f"  [green]✓[/] Saved as ID=[bold]{db_trace.id}[/] with {len(trace_input.turns)} turns, {total_steps} steps")
    
    return SavedTraceResponse(
        id=db_trace.id,
        name=db_trace.name,
        task=task_desc,
        success=db_trace.success,
        total_cost=db_trace.total_cost,
        created_at=db_trace.created_at.isoformat(),
        overall_score=db_trace.overall_score,
        step_count=len(trace_input.turns),
    )


@router.get("/agent/traces")
async def list_agent_traces(db: Session = Depends(get_db), limit: int = 50, offset: int = 0):
    """List all saved agent traces"""
    console.print(f"[dim]📋 Listing traces: limit={limit} offset={offset}[/]")
    traces = db.query(AgentTraceDB).order_by(AgentTraceDB.created_at.desc()).offset(offset).limit(limit).all()
    console.print(f"  [green]✓[/] Found {len(traces)} traces")
    
    return [
        {
            'id': t.id,
            'name': t.name,
            'task': t.initial_task or (t.turns[0].user_message if t.turns else "Agent Trace"),
            'success': t.success,
            'total_cost': t.total_cost,
            'created_at': t.created_at.isoformat(),
            'overall_score': t.overall_score,
            'step_count': len(t.turns),
            'analysis_types': json.loads(t.analysis_types) if t.analysis_types else [],
        }
        for t in traces
    ]


@router.get("/agent/traces/{trace_id}")
async def get_agent_trace(trace_id: int, db: Session = Depends(get_db)):
    """Get a specific agent trace with all details"""
    console.print(f"[dim]🔍 Getting trace ID={trace_id}[/]")
    trace = db.query(AgentTraceDB).filter(AgentTraceDB.id == trace_id).first()
    
    if not trace:
        console.print(f"  [red]✗[/] Trace not found")
        raise HTTPException(status_code=404, detail="Trace not found")
    
    # Rebuild turns with steps
    turns = []
    for db_turn in trace.turns:
        # Rebuild steps for this turn
        agent_steps = []
        for step in db_turn.steps:
            tool_call = None
            if step.tool_call_json:
                tc_data = json.loads(step.tool_call_json)
                tool_call = {
                    'name': tc_data.get('name'),
                    'parameters': tc_data.get('parameters', {}),
                    'result': tc_data.get('result'),
                    'error': tc_data.get('error'),
                }
            agent_steps.append({
                'thought': step.thought,
                'action': step.action,
                'observation': step.observation,
                'tool_call': tool_call,
            })
        
        turns.append({
            'user_message': db_turn.user_message,
            'agent_response': db_turn.agent_response,
            'agent_steps': agent_steps,
        })
    
    # Rebuild results
    results = {
        'overall_score': trace.overall_score,
        'analysis_types': json.loads(trace.analysis_types) if trace.analysis_types else [],
    }
    
    if trace.conversation_result:
        results['conversation'] = json.loads(trace.conversation_result)
    if trace.trajectory_result:
        results['trajectory'] = json.loads(trace.trajectory_result)
    if trace.tools_result:
        results['tools'] = json.loads(trace.tools_result)
    if trace.self_correction_result:
        results['self_correction'] = json.loads(trace.self_correction_result)
    if trace.intent_drift_result:
        results['intent_drift'] = json.loads(trace.intent_drift_result)
    
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
# Background Job System
# ============================================

# Store running background tasks
_background_tasks: dict = {}


def run_background_analysis(job_id: int):
    """Run analysis in background thread"""
    from datetime import datetime
    import threading
    
    db = SessionLocal()
    try:
        job = db.query(AnalysisJobDB).filter(AnalysisJobDB.id == job_id).first()
        if not job:
            return
        
        # Parse input
        trace_data = json.loads(job.trace_json)
        analysis_types = json.loads(job.analysis_types_json)
        
        # Update status
        job.status = "running"
        job.started_at = datetime.utcnow()
        db.commit()
        
        console.print(f"[bold cyan]🔄 Background job {job_id}:[/] Starting analysis")
        
        # Rebuild trace input
        trace_input = AgentTraceInput(**trace_data)
        trace = convert_to_agent_trace(trace_input)
        
        # Expand analysis types
        if 'full_all' in analysis_types:
            analysis_types = ['conversation', 'trajectory', 'tools', 'self_correction', 'intent_drift']
        elif 'full_agent' in analysis_types:
            analysis_types = ['trajectory', 'tools', 'self_correction', 'intent_drift']
        
        job.total_steps = len(analysis_types)
        job.current_step = 0
        db.commit()
        
        results = {}
        progress_details = {'turn_results': []}
        
        # Run conversation analysis with turn-by-turn progress
        if 'conversation' in analysis_types:
            job.current_step += 1
            job.current_analysis = "conversation"
            db.commit()
            
            turns = trace_input.turns
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
            
            for turn_idx, turn in enumerate(turns):
                user_msg = ReAskMessage.user(turn.user_message)
                assistant_msg = ReAskMessage.assistant(turn.agent_response)
                follow_up = ReAskMessage.user(turns[turn_idx + 1].user_message) if turn_idx + 1 < len(turns) else None
                
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
                progress_details['turn_results'].append(turn_result)
                
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
                
                # Update progress in DB
                job.progress_details_json = json.dumps(progress_details)
                db.commit()
            
            total_responses = len(turns)
            good = total_responses - bad
            avg_conf = sum(r['confidence'] for r in turn_results) / len(turn_results) if turn_results else 0
            
            results['conversation'] = {
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
        
        # Run other analyses
        if 'trajectory' in analysis_types:
            job.current_step += 1
            job.current_analysis = "trajectory"
            db.commit()
            
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
        
        if 'tools' in analysis_types:
            job.current_step += 1
            job.current_analysis = "tools"
            db.commit()
            
            evaluator = ToolEvaluator()
            efficiency, tool_results = evaluator.evaluate_tool_chain(trace)
            results['tools'] = {
                'efficiency': efficiency,
                'results': [{'signal': r.signal.value, 'tool_name': r.tool_name, 'confidence': r.confidence, 'reason': r.reason} for r in tool_results],
                'total_calls': len(trace.tool_calls),
            }
        
        if 'self_correction' in analysis_types:
            job.current_step += 1
            job.current_analysis = "self_correction"
            db.commit()
            
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
        
        if 'intent_drift' in analysis_types:
            job.current_step += 1
            job.current_analysis = "intent_drift"
            db.commit()
            
            meter = IntentDriftMeter()
            result = meter.analyze(trace)
            results['intent_drift'] = {
                'drift_score': result.drift_score,
                'step_index': result.step_index,
                'is_legitimate': result.is_legitimate,
                'reason': result.reason,
                'drift_history': result.drift_history,
            }
        
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
        
        # Save the trace
        task_desc = trace_input.initial_task or (trace_input.turns[0].user_message if trace_input.turns else "Agent Trace")
        
        db_trace = AgentTraceDB(
            name=task_desc[:100],
            initial_task=trace_input.initial_task,
            success=trace_input.success,
            total_cost=trace_input.total_cost,
            overall_score=results.get('overall_score'),
            analysis_types=json.dumps(results.get('analysis_types', [])),
            conversation_result=json.dumps(results.get('conversation')) if results.get('conversation') else None,
            trajectory_result=json.dumps(results.get('trajectory')) if results.get('trajectory') else None,
            tools_result=json.dumps(results.get('tools')) if results.get('tools') else None,
            self_correction_result=json.dumps(results.get('self_correction')) if results.get('self_correction') else None,
            intent_drift_result=json.dumps(results.get('intent_drift')) if results.get('intent_drift') else None,
        )
        db.add(db_trace)
        db.flush()
        
        # Add turns
        for turn_idx, turn in enumerate(trace_input.turns):
            db_turn = AgentTurnDB(
                trace_id=db_trace.id,
                index=turn_idx,
                user_message=turn.user_message,
                agent_response=turn.agent_response,
            )
            db.add(db_turn)
            db.flush()
            
            for step_idx, step in enumerate(turn.agent_steps or []):
                db_step = AgentStepDB(
                    turn_id=db_turn.id,
                    index=step_idx,
                    thought=step.thought,
                    action=step.action,
                    observation=step.observation,
                    tool_call_json=json.dumps(step.tool_call.dict()) if step.tool_call else None,
                )
                db.add(db_step)
        
        # Mark job complete
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.result_json = json.dumps(results)
        job.saved_trace_id = db_trace.id
        job.current_analysis = None
        db.commit()
        
        console.print(f"[bold green]✅ Background job {job_id}:[/] Complete! Saved as trace ID={db_trace.id}")
        
    except Exception as e:
        console.print(f"[bold red]❌ Background job {job_id} failed:[/] {str(e)}")
        job.status = "failed"
        job.error_message = str(e)
        job.completed_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()
        # Clean up from running tasks
        _background_tasks.pop(job_id, None)


class StartJobRequest(BaseModel):
    trace: AgentTraceInput
    analysis_types: List[str]


class JobStatusResponse(BaseModel):
    id: int
    status: str
    current_step: int
    total_steps: int
    current_analysis: Optional[str]
    progress_details: Optional[dict]
    result: Optional[dict]
    saved_trace_id: Optional[int]
    error_message: Optional[str]


@router.post("/agent/jobs")
async def start_background_job(request: StartJobRequest, db: Session = Depends(get_db)):
    """Start a background analysis job"""
    from datetime import datetime
    import threading
    
    console.print(f"[bold magenta]🚀 Starting background job[/]")
    
    # Create job record
    job = AnalysisJobDB(
        status="pending",
        trace_json=json.dumps(request.trace.dict()),
        analysis_types_json=json.dumps(request.analysis_types),
        total_steps=len(request.analysis_types),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Start background thread
    thread = threading.Thread(target=run_background_analysis, args=(job.id,), daemon=True)
    thread.start()
    _background_tasks[job.id] = thread
    
    console.print(f"  [green]✓[/] Job ID={job.id} started")
    
    return {'job_id': job.id, 'status': 'pending'}


@router.get("/agent/jobs/{job_id}")
async def get_job_status(job_id: int, db: Session = Depends(get_db)):
    """Get the status of a background job"""
    job = db.query(AnalysisJobDB).filter(AnalysisJobDB.id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        id=job.id,
        status=job.status,
        current_step=job.current_step,
        total_steps=job.total_steps,
        current_analysis=job.current_analysis,
        progress_details=json.loads(job.progress_details_json) if job.progress_details_json else None,
        result=json.loads(job.result_json) if job.result_json else None,
        saved_trace_id=job.saved_trace_id,
        error_message=job.error_message,
    )


@router.get("/agent/jobs")
async def list_jobs(db: Session = Depends(get_db), limit: int = 10):
    """List recent jobs"""
    jobs = db.query(AnalysisJobDB).order_by(AnalysisJobDB.created_at.desc()).limit(limit).all()
    
    return [
        {
            'id': j.id,
            'status': j.status,
            'current_step': j.current_step,
            'total_steps': j.total_steps,
            'current_analysis': j.current_analysis,
            'saved_trace_id': j.saved_trace_id,
            'created_at': j.created_at.isoformat() if j.created_at else None,
        }
        for j in jobs
    ]


@router.delete("/agent/jobs/{job_id}")
async def delete_job(job_id: int, db: Session = Depends(get_db)):
    """Delete a job record"""
    job = db.query(AnalysisJobDB).filter(AnalysisJobDB.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    db.delete(job)
    db.commit()
    return {'message': 'Job deleted'}

