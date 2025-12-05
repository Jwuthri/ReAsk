"""Agent Trajectory Analysis (ATA) - Evaluate entire agent execution traces"""

import os
from typing import Optional, List, Tuple
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .agent_models import (
    AgentTrace, AgentStep, TrajectorySignal, TrajectoryEvalResult
)
from .embeddings import EmbeddingService

load_dotenv()


class CircularPatternResult(BaseModel):
    """Structured output for circular pattern detection"""
    is_circular: bool = Field(description="Whether the steps represent circular/repetitive behavior")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reason: str = Field(description="Brief explanation under 30 words")


class ProgressAnalysisResult(BaseModel):
    """Structured output for progress analysis"""
    is_progressing: bool = Field(description="Whether the agent is making progress toward the goal")
    progress_score: float = Field(ge=0.0, le=1.0, description="How much progress (0=none, 1=complete)")
    is_regression: bool = Field(description="Whether the agent is undoing previous progress")
    reason: str = Field(description="Brief explanation under 30 words")


class TrajectoryOverallResult(BaseModel):
    """Structured output for overall trajectory evaluation"""
    signal: str = Field(description="One of: optimal, circular, regression, stall, recovery, drift")
    efficiency_score: float = Field(ge=0.0, le=1.0, description="Path efficiency (1.0 = optimal)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in assessment")
    reason: str = Field(description="Brief explanation under 40 words")


def _make_strict_schema(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model to a strict OpenAI-compatible JSON schema."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.pop("$defs", None)
    schema["additionalProperties"] = False
    return schema


class TrajectoryAnalyzer:
    """
    Analyzes agent execution traces for quality signals.
    
    Detects:
    - Circular patterns (agent repeating similar actions)
    - Regression (agent undoing progress)
    - Stalls (agent not making progress)
    - Recovery patterns (agent bouncing back from errors)
    """
    
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = "gpt-5-nano",
        embedding_model: str = "text-embedding-3-small",
        similarity_threshold: float = 0.75,  # For detecting similar steps
    ):
        self.client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.embeddings = EmbeddingService(self.client, embedding_model)
        self.similarity_threshold = similarity_threshold
    
    def analyze(self, trace: AgentTrace) -> TrajectoryEvalResult:
        """
        Analyze an entire agent trace for trajectory quality.
        
        Args:
            trace: Complete agent execution trace
        
        Returns:
            TrajectoryEvalResult with signal, confidence, and metrics
        """
        if not trace.steps:
            return TrajectoryEvalResult(
                signal=TrajectorySignal.OPTIMAL,
                confidence=1.0,
                efficiency_score=1.0,
                circular_count=0,
                regression_count=0,
                reason="Empty trace - nothing to evaluate"
            )
        
        # Step 1: Detect circular patterns via embeddings
        circular_pairs = self._detect_circular_patterns(trace)
        circular_count = len(circular_pairs)
        
        # Step 2: Analyze progress at each step
        progress_analysis = self._analyze_progress(trace)
        regression_count = sum(1 for p in progress_analysis if p.get("is_regression", False))
        
        # Step 3: Get overall assessment from LLM
        overall = self._evaluate_overall(trace, circular_count, regression_count)
        
        # Map string signal to enum
        signal_map = {
            "optimal": TrajectorySignal.OPTIMAL,
            "circular": TrajectorySignal.CIRCULAR,
            "regression": TrajectorySignal.REGRESSION,
            "stall": TrajectorySignal.STALL,
            "recovery": TrajectorySignal.RECOVERY,
            "drift": TrajectorySignal.DRIFT,
        }
        signal = signal_map.get(overall["signal"], TrajectorySignal.STALL)
        
        return TrajectoryEvalResult(
            signal=signal,
            confidence=overall["confidence"],
            efficiency_score=overall["efficiency_score"],
            circular_count=circular_count,
            regression_count=regression_count,
            reason=overall["reason"],
            details={
                "circular_pairs": circular_pairs,
                "progress_analysis": progress_analysis,
                "step_count": trace.step_count,
                "task": trace.task,
            }
        )
    
    def _detect_circular_patterns(self, trace: AgentTrace) -> List[Tuple[int, int, float]]:
        """
        Detect circular patterns using embedding similarity.
        
        Returns list of (step_i, step_j, similarity) tuples where similarity > threshold
        """
        circular_pairs = []
        step_contents = [s.content for s in trace.steps if s.content]
        
        if len(step_contents) < 2:
            return []
        
        # Compare each step with all previous steps
        for i in range(1, len(step_contents)):
            for j in range(i):
                similarity = self.embeddings.similarity(step_contents[j], step_contents[i])
                if similarity >= self.similarity_threshold:
                    # Confirm with LLM that it's actually circular
                    if self._confirm_circular(step_contents[j], step_contents[i]):
                        circular_pairs.append((j, i, similarity))
        
        return circular_pairs
    
    def _confirm_circular(self, step1: str, step2: str) -> bool:
        """Use LLM to confirm if two steps are actually circular behavior"""
        prompt = f"""Are these two agent steps essentially the same action being repeated?

This indicates circular/stuck behavior if the agent is doing the same thing again without progress.

STEP 1:
{step1}

STEP 2:
{step2}

Consider:
- Are they trying to accomplish the same sub-goal?
- Is this repetition without meaningful variation?
- Would a skilled human consider this "going in circles"?"""

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "circular_result",
                        "strict": True,
                        "schema": _make_strict_schema(CircularPatternResult)
                    }
                }
            )
            result = CircularPatternResult.model_validate_json(response.output_text)
            return result.is_circular and result.confidence >= 0.7
        except Exception:
            return False
    
    def _analyze_progress(self, trace: AgentTrace) -> List[dict]:
        """Analyze progress at each step"""
        if len(trace.steps) < 2:
            return []
        
        progress_analysis = []
        
        # Analyze progress between consecutive step pairs
        for i in range(1, len(trace.steps)):
            prev_step = trace.steps[i - 1]
            curr_step = trace.steps[i]
            
            prompt = f"""Analyze if this agent is making progress toward the task.

TASK: {trace.task}

PREVIOUS STEP:
{prev_step.content}

CURRENT STEP:
{curr_step.content}

Is the agent:
1. Making progress toward completing the task?
2. Regressing (undoing previous work or moving backward)?"""

            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "progress_result",
                            "strict": True,
                            "schema": _make_strict_schema(ProgressAnalysisResult)
                        }
                    }
                )
                result = ProgressAnalysisResult.model_validate_json(response.output_text)
                progress_analysis.append({
                    "step_index": i,
                    "is_progressing": result.is_progressing,
                    "progress_score": result.progress_score,
                    "is_regression": result.is_regression,
                    "reason": result.reason
                })
            except Exception as e:
                progress_analysis.append({
                    "step_index": i,
                    "is_progressing": True,
                    "progress_score": 0.5,
                    "is_regression": False,
                    "reason": f"Error: {str(e)}"
                })
        
        return progress_analysis
    
    def _evaluate_overall(self, trace: AgentTrace, circular_count: int, regression_count: int) -> dict:
        """Get overall trajectory assessment"""
        steps_summary = "\n".join([
            f"Step {s.index}: {s.content[:200]}..." if len(s.content) > 200 else f"Step {s.index}: {s.content}"
            for s in trace.steps[:10]  # Limit to first 10 steps for prompt size
        ])
        
        prompt = f"""Evaluate this agent's overall trajectory quality.

TASK: {trace.task}

STEPS TAKEN:
{steps_summary}

DETECTED ISSUES:
- Circular patterns: {circular_count}
- Regressions: {regression_count}
- Total steps: {trace.step_count}

RESULT: {trace.final_result or 'Not yet complete'}
SUCCESS: {trace.success if trace.success is not None else 'Unknown'}

Rate the trajectory:
- "optimal": Clean, efficient path to goal
- "circular": Agent going in circles, repeating actions
- "regression": Agent undoing progress, moving backward
- "stall": Agent stuck, not making progress
- "recovery": Agent recovering from errors successfully
- "drift": Agent solving wrong problem"""

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "trajectory_result",
                        "strict": True,
                        "schema": _make_strict_schema(TrajectoryOverallResult)
                    }
                }
            )
            result = TrajectoryOverallResult.model_validate_json(response.output_text)
            return {
                "signal": result.signal,
                "efficiency_score": result.efficiency_score,
                "confidence": result.confidence,
                "reason": result.reason
            }
        except Exception as e:
            # Fallback based on counts
            if circular_count >= 2:
                signal = "circular"
                efficiency = 0.3
            elif regression_count >= 2:
                signal = "regression"
                efficiency = 0.4
            elif trace.step_count > 10 and circular_count == 0:
                signal = "stall"
                efficiency = 0.5
            else:
                signal = "optimal"
                efficiency = 0.8
            
            return {
                "signal": signal,
                "efficiency_score": efficiency,
                "confidence": 0.5,
                "reason": f"Fallback evaluation: {str(e)}"
            }
    
    def detect_circular_live(self, trace: AgentTrace, new_step: AgentStep) -> Optional[Tuple[int, float]]:
        """
        Live detection: Check if a new step is circular with any previous step.
        Useful for real-time monitoring.
        
        Returns (circular_with_step_index, similarity) if circular, None otherwise
        """
        if not new_step.content:
            return None
        
        for prev_step in trace.steps:
            if not prev_step.content:
                continue
            
            similarity = self.embeddings.similarity(prev_step.content, new_step.content)
            if similarity >= self.similarity_threshold:
                if self._confirm_circular(prev_step.content, new_step.content):
                    return (prev_step.index, similarity)
        
        return None

