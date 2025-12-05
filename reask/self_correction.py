"""Self-Correction Detection (SCD) - Detect and evaluate agent self-correction"""

import os
from typing import Optional, List, Tuple
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .agent_models import (
    AgentTrace, AgentStep, SelfCorrectionSignal, SelfCorrectionResult
)
from .embeddings import EmbeddingService

load_dotenv()


class ErrorDetectionResult(BaseModel):
    """Structured output for error detection in agent step"""
    detected_error: bool = Field(description="Whether the agent recognized an error in this step")
    error_type: str = Field(description="Type of error: execution_failure, wrong_result, logical_error, or none")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reason: str = Field(description="Brief explanation under 30 words")


class CorrectionAttemptResult(BaseModel):
    """Structured output for correction attempt analysis"""
    is_correction_attempt: bool = Field(description="Whether this step is attempting to correct a previous error")
    correcting_step_index: int = Field(description="Index of the step being corrected, or -1 if not a correction")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reason: str = Field(description="Brief explanation under 30 words")


class CorrectionSuccessResult(BaseModel):
    """Structured output for correction success evaluation"""
    correction_successful: bool = Field(description="Whether the correction successfully fixed the error")
    remaining_issues: str = Field(description="Any remaining issues after correction, or 'none'")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reason: str = Field(description="Brief explanation under 30 words")


class OverallSelfCorrectionResult(BaseModel):
    """Structured output for overall self-correction analysis"""
    self_awareness_score: float = Field(ge=0.0, le=1.0, description="How well agent notices its errors (0=oblivious, 1=perfect)")
    correction_efficiency: float = Field(ge=0.0, le=1.0, description="How quickly it fixes errors (0=never, 1=immediate)")
    is_in_loop: bool = Field(description="Whether agent is stuck in a correction loop")
    graceful_degradation: bool = Field(description="Whether agent knows when to ask for help")
    reason: str = Field(description="Brief overall assessment under 40 words")


def _make_strict_schema(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model to a strict OpenAI-compatible JSON schema."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.pop("$defs", None)
    schema["additionalProperties"] = False
    return schema


class SelfCorrectionDetector:
    """
    Detects and evaluates agent self-correction behavior.
    
    This is the inverse of CCM/RDM - instead of detecting when users
    correct agents, we detect when agents correct themselves.
    
    Measures:
    - Self-Awareness: Does the agent notice when it fails?
    - Correction Efficiency: How quickly does it recover?
    - Spiral Detection: Is it stuck in a correction loop?
    - Graceful Degradation: Does it know when to give up?
    """
    
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = "gpt-5-nano",
        embedding_model: str = "text-embedding-3-small",
        loop_threshold: int = 3,  # Attempts before considered "loop"
    ):
        self.client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.embeddings = EmbeddingService(self.client, embedding_model)
        self.loop_threshold = loop_threshold
    
    def analyze(self, trace: AgentTrace) -> SelfCorrectionResult:
        """
        Analyze an agent trace for self-correction patterns.
        
        Args:
            trace: Complete agent execution trace
        
        Returns:
            SelfCorrectionResult with metrics and signals
        """
        if len(trace.steps) < 2:
            return SelfCorrectionResult(
                detected_error=False,
                correction_attempt=False,
                correction_success=False,
                loops_before_fix=0,
                self_awareness_score=1.0,
                correction_efficiency=1.0,
                reason="Not enough steps to analyze self-correction"
            )
        
        # Step 1: Identify error points in the trace
        error_points = self._detect_error_points(trace)
        
        # Step 2: For each error, check if agent attempted correction
        corrections = self._detect_correction_attempts(trace, error_points)
        
        # Step 3: Evaluate correction success
        successful_corrections = self._evaluate_correction_success(trace, corrections)
        
        # Step 4: Detect correction loops
        loops = self._detect_correction_loops(trace, corrections)
        
        # Step 5: Get overall assessment
        overall = self._evaluate_overall(trace, error_points, corrections, successful_corrections, loops)
        
        # Calculate metrics
        detected_error = len(error_points) > 0
        correction_attempt = len(corrections) > 0
        correction_success = len(successful_corrections) > 0
        loops_before_fix = max([len(l) for l in loops] if loops else [0])
        
        return SelfCorrectionResult(
            detected_error=detected_error,
            correction_attempt=correction_attempt,
            correction_success=correction_success,
            loops_before_fix=loops_before_fix,
            self_awareness_score=overall["self_awareness_score"],
            correction_efficiency=overall["correction_efficiency"],
            reason=overall["reason"],
            details={
                "error_points": error_points,
                "corrections": corrections,
                "successful_corrections": successful_corrections,
                "loops": loops,
                "is_in_loop": overall["is_in_loop"],
                "graceful_degradation": overall["graceful_degradation"],
            }
        )
    
    def _detect_error_points(self, trace: AgentTrace) -> List[dict]:
        """Detect steps where agent encountered/recognized errors"""
        error_points = []
        
        for step in trace.steps:
            # Check tool call errors
            if step.tool_call and step.tool_call.error:
                error_points.append({
                    "step_index": step.index,
                    "error_type": "execution_failure",
                    "error": step.tool_call.error,
                    "confidence": 1.0
                })
                continue
            
            # Use LLM to detect recognized errors in thought/observation
            result = self._check_error_recognition(step)
            if result["detected_error"]:
                error_points.append({
                    "step_index": step.index,
                    "error_type": result["error_type"],
                    "confidence": result["confidence"],
                    "reason": result["reason"]
                })
        
        return error_points
    
    def _check_error_recognition(self, step: AgentStep) -> dict:
        """Check if agent recognized an error in this step"""
        prompt = f"""Analyze if the agent recognized an error in this step.

STEP CONTENT:
Thought: {step.thought or 'N/A'}
Action: {step.action or 'N/A'}
Observation: {step.observation or 'N/A'}

Look for signals like:
- "That didn't work"
- "I made a mistake"
- "Let me try again"
- "That's not right"
- Error messages in observation
- Agent acknowledging incorrect approach"""

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "error_detection",
                        "strict": True,
                        "schema": _make_strict_schema(ErrorDetectionResult)
                    }
                }
            )
            result = ErrorDetectionResult.model_validate_json(response.output_text)
            return {
                "detected_error": result.detected_error,
                "error_type": result.error_type,
                "confidence": result.confidence,
                "reason": result.reason
            }
        except Exception as e:
            return {
                "detected_error": False,
                "error_type": "none",
                "confidence": 0.5,
                "reason": f"Error: {str(e)}"
            }
    
    def _detect_correction_attempts(
        self,
        trace: AgentTrace,
        error_points: List[dict]
    ) -> List[dict]:
        """Detect steps that are attempting to correct previous errors"""
        corrections = []
        
        for i, step in enumerate(trace.steps):
            if i == 0:
                continue
            
            # Get context from previous steps
            prev_steps = trace.steps[max(0, i-3):i]
            context = "\n".join([s.content for s in prev_steps if s.content])
            
            prompt = f"""Is this step attempting to correct a previous error?

PREVIOUS CONTEXT:
{context}

CURRENT STEP:
{step.content}

ERROR POINTS IN TRACE: {[e['step_index'] for e in error_points]}

Is the agent trying to fix something that went wrong earlier?"""

            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "correction_attempt",
                            "strict": True,
                            "schema": _make_strict_schema(CorrectionAttemptResult)
                        }
                    }
                )
                result = CorrectionAttemptResult.model_validate_json(response.output_text)
                
                if result.is_correction_attempt:
                    corrections.append({
                        "step_index": i,
                        "correcting_step": result.correcting_step_index,
                        "confidence": result.confidence,
                        "reason": result.reason
                    })
            except Exception:
                pass
        
        return corrections
    
    def _evaluate_correction_success(
        self,
        trace: AgentTrace,
        corrections: List[dict]
    ) -> List[dict]:
        """Evaluate if corrections were successful"""
        successful = []
        
        for correction in corrections:
            step_idx = correction["step_index"]
            step = trace.steps[step_idx]
            
            # Get steps after the correction
            following_steps = trace.steps[step_idx:step_idx+3]
            following_context = "\n".join([s.content for s in following_steps if s.content])
            
            prompt = f"""Did this correction attempt successfully fix the problem?

CORRECTION ATTEMPT:
{step.content}

FOLLOWING STEPS:
{following_context}

TASK: {trace.task}
FINAL SUCCESS: {trace.success if trace.success is not None else 'Unknown'}

Did the correction work?"""

            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "correction_success",
                            "strict": True,
                            "schema": _make_strict_schema(CorrectionSuccessResult)
                        }
                    }
                )
                result = CorrectionSuccessResult.model_validate_json(response.output_text)
                
                if result.correction_successful:
                    successful.append({
                        "correction_step": step_idx,
                        "confidence": result.confidence,
                        "reason": result.reason
                    })
            except Exception:
                pass
        
        return successful
    
    def _detect_correction_loops(
        self,
        trace: AgentTrace,
        corrections: List[dict]
    ) -> List[List[int]]:
        """Detect if agent is stuck in correction loops"""
        loops = []
        
        if len(corrections) < self.loop_threshold:
            return loops
        
        # Group corrections by what they're correcting
        correction_targets = {}
        for c in corrections:
            target = c.get("correcting_step", -1)
            if target not in correction_targets:
                correction_targets[target] = []
            correction_targets[target].append(c["step_index"])
        
        # Find loops (multiple corrections for same target)
        for target, correction_steps in correction_targets.items():
            if len(correction_steps) >= self.loop_threshold:
                loops.append(correction_steps)
        
        # Also check for similar correction attempts using embeddings
        if len(corrections) >= self.loop_threshold:
            correction_contents = [
                trace.steps[c["step_index"]].content 
                for c in corrections 
                if trace.steps[c["step_index"]].content
            ]
            
            similar_group = []
            for i, content in enumerate(correction_contents):
                for j in range(i + 1, len(correction_contents)):
                    similarity = self.embeddings.similarity(content, correction_contents[j])
                    if similarity > 0.8:
                        if corrections[i]["step_index"] not in similar_group:
                            similar_group.append(corrections[i]["step_index"])
                        if corrections[j]["step_index"] not in similar_group:
                            similar_group.append(corrections[j]["step_index"])
            
            if len(similar_group) >= self.loop_threshold:
                loops.append(similar_group)
        
        return loops
    
    def _evaluate_overall(
        self,
        trace: AgentTrace,
        error_points: List[dict],
        corrections: List[dict],
        successful_corrections: List[dict],
        loops: List[List[int]]
    ) -> dict:
        """Get overall self-correction assessment"""
        # Calculate basic metrics
        num_errors = len(error_points)
        num_corrections = len(corrections)
        num_successful = len(successful_corrections)
        
        # Self-awareness: proportion of errors that led to corrections
        if num_errors > 0:
            self_awareness = min(1.0, num_corrections / num_errors)
        else:
            self_awareness = 1.0  # No errors = perfect awareness
        
        # Correction efficiency: successful / attempted
        if num_corrections > 0:
            correction_efficiency = num_successful / num_corrections
        else:
            correction_efficiency = 1.0 if num_errors == 0 else 0.0
        
        # Is in loop?
        is_in_loop = len(loops) > 0
        
        # Graceful degradation: look for "ask for help" signals
        graceful = False
        for step in trace.steps:
            if step.thought and any(phrase in step.thought.lower() for phrase in [
                "ask for help", "need clarification", "cannot proceed",
                "need more information", "stuck", "give up"
            ]):
                graceful = True
                break
        
        # Generate reason
        if is_in_loop:
            reason = f"Agent stuck in correction loop with {len(loops)} detected loops"
        elif num_errors == 0:
            reason = "No errors detected - clean execution"
        elif num_successful == num_errors:
            reason = f"Agent recovered from all {num_errors} errors successfully"
        elif num_corrections > 0:
            reason = f"Agent attempted {num_corrections} corrections, {num_successful} successful"
        else:
            reason = f"Agent encountered {num_errors} errors but did not attempt correction"
        
        return {
            "self_awareness_score": self_awareness,
            "correction_efficiency": correction_efficiency,
            "is_in_loop": is_in_loop,
            "graceful_degradation": graceful,
            "reason": reason
        }
    
    def detect_correction_live(
        self,
        trace: AgentTrace,
        new_step: AgentStep
    ) -> Optional[SelfCorrectionSignal]:
        """
        Live detection: Check if a new step indicates self-correction.
        Useful for real-time monitoring.
        
        Returns signal if detected, None otherwise
        """
        if len(trace.steps) < 1:
            return None
        
        # Check for error recognition
        error_result = self._check_error_recognition(new_step)
        if error_result["detected_error"]:
            return SelfCorrectionSignal.ERROR_DETECTED
        
        # Check for correction attempt
        prev_context = "\n".join([s.content for s in trace.steps[-3:] if s.content])
        
        prompt = f"""Is this a correction attempt?

CONTEXT:
{prev_context}

NEW STEP:
{new_step.content}"""

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "correction_attempt",
                        "strict": True,
                        "schema": _make_strict_schema(CorrectionAttemptResult)
                    }
                }
            )
            result = CorrectionAttemptResult.model_validate_json(response.output_text)
            
            if result.is_correction_attempt:
                return SelfCorrectionSignal.CORRECTION_ATTEMPTED
        except Exception:
            pass
        
        return None

