"""Intent Drift Meter (IDM) - Detect when agents drift from original intent"""

import os
from typing import Optional, List
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .agent_models import AgentTrace, AgentStep, IntentDriftResult
from .embeddings import EmbeddingService

load_dotenv()


class DriftAnalysisResult(BaseModel):
    """Structured output for drift analysis at a single step"""
    drift_score: float = Field(ge=0.0, le=1.0, description="How far from original intent (0=on track, 1=completely off)")
    is_legitimate: bool = Field(description="Whether the drift was requested or makes sense")
    reason: str = Field(description="Brief explanation under 30 words")


class OverallDriftResult(BaseModel):
    """Structured output for overall drift analysis"""
    max_drift_score: float = Field(ge=0.0, le=1.0, description="Maximum drift observed")
    max_drift_step: int = Field(description="Step index with maximum drift")
    is_legitimate: bool = Field(description="Whether overall drift was justified")
    drift_type: str = Field(description="One of: none, scope_creep, tangent, wrong_problem, recovery")
    reason: str = Field(description="Brief explanation under 40 words")


def _make_strict_schema(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model to a strict OpenAI-compatible JSON schema."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.pop("$defs", None)
    schema["additionalProperties"] = False
    return schema


class IntentDriftMeter:
    """
    Measures alignment between agent actions and original intent.
    
    In long multi-step tasks, agents can "drift" - they start solving
    a different problem than what was asked. This detector continuously
    measures the alignment between current action and original task.
    
    Detects:
    - Scope creep: Agent expanding beyond requested task
    - Tangents: Agent going off on unrelated work
    - Wrong problem: Agent solving something entirely different
    - Recovery: Agent getting back on track after drift
    """
    
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = "gpt-5-nano",
        embedding_model: str = "text-embedding-3-small",
        drift_warning_threshold: float = 0.35,
        drift_critical_threshold: float = 0.60,
    ):
        self.client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.embeddings = EmbeddingService(self.client, embedding_model)
        self.drift_warning_threshold = drift_warning_threshold
        self.drift_critical_threshold = drift_critical_threshold
    
    def analyze(self, trace: AgentTrace) -> IntentDriftResult:
        """
        Analyze intent drift across an entire agent trace.
        
        Args:
            trace: Complete agent execution trace
        
        Returns:
            IntentDriftResult with drift scores and analysis
        """
        if not trace.steps:
            return IntentDriftResult(
                drift_score=0.0,
                step_index=-1,
                is_legitimate=True,
                reason="Empty trace - no drift possible",
                drift_history=[]
            )
        
        # Get embedding of original intent
        intent_embedding = self.embeddings.embed(trace.task)
        
        # Calculate drift at each step
        drift_history = []
        for step in trace.steps:
            if not step.content:
                drift_history.append(0.0)
                continue
            
            # Embedding-based drift (1 - similarity = drift)
            step_embedding = self.embeddings.embed(step.content)
            import numpy as np
            similarity = float(np.dot(intent_embedding, step_embedding) / (
                np.linalg.norm(intent_embedding) * np.linalg.norm(step_embedding)
            ))
            embedding_drift = 1.0 - max(0.0, similarity)
            
            # LLM-based drift analysis for better accuracy
            llm_drift = self._analyze_step_drift(trace.task, step)
            
            # Combine scores (weighted average)
            combined_drift = 0.4 * embedding_drift + 0.6 * llm_drift["drift_score"]
            drift_history.append(combined_drift)
        
        # Find max drift
        max_drift = max(drift_history) if drift_history else 0.0
        max_drift_step = drift_history.index(max_drift) if drift_history else -1
        
        # Get overall assessment
        overall = self._evaluate_overall_drift(trace, drift_history)
        
        return IntentDriftResult(
            drift_score=max_drift,
            step_index=max_drift_step,
            is_legitimate=overall["is_legitimate"],
            reason=overall["reason"],
            drift_history=drift_history,
            details={
                "drift_type": overall["drift_type"],
                "warning_threshold": self.drift_warning_threshold,
                "critical_threshold": self.drift_critical_threshold,
                "steps_above_warning": sum(1 for d in drift_history if d >= self.drift_warning_threshold),
                "steps_above_critical": sum(1 for d in drift_history if d >= self.drift_critical_threshold),
            }
        )
    
    def _analyze_step_drift(self, original_task: str, step: AgentStep) -> dict:
        """Analyze drift at a single step"""
        prompt = f"""Measure how far this agent step has drifted from the original task.

ORIGINAL TASK:
{original_task}

CURRENT STEP:
{step.content}

Consider:
- Is this step working toward the original goal?
- Is this a necessary side-quest or completely off-topic?
- Did the user request this change in scope?

Drift score guide:
- 0.0-0.2: Directly working on task
- 0.2-0.4: Related work, minor tangent
- 0.4-0.6: Significant deviation, questionable relevance
- 0.6-0.8: Mostly off-topic
- 0.8-1.0: Completely different problem"""

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "drift_analysis",
                        "strict": True,
                        "schema": _make_strict_schema(DriftAnalysisResult)
                    }
                }
            )
            result = DriftAnalysisResult.model_validate_json(response.output_text)
            return {
                "drift_score": result.drift_score,
                "is_legitimate": result.is_legitimate,
                "reason": result.reason
            }
        except Exception as e:
            return {
                "drift_score": 0.5,
                "is_legitimate": True,
                "reason": f"Error: {str(e)}"
            }
    
    def _evaluate_overall_drift(self, trace: AgentTrace, drift_history: List[float]) -> dict:
        """Evaluate overall drift pattern"""
        if not drift_history:
            return {
                "is_legitimate": True,
                "drift_type": "none",
                "reason": "No steps to analyze"
            }
        
        # Summarize steps
        steps_summary = "\n".join([
            f"Step {i}: (drift={drift_history[i]:.2f}) {trace.steps[i].content[:100]}..."
            if trace.steps[i].content and len(trace.steps[i].content) > 100
            else f"Step {i}: (drift={drift_history[i]:.2f}) {trace.steps[i].content or 'N/A'}"
            for i in range(min(len(trace.steps), 10))
        ])
        
        prompt = f"""Analyze the overall drift pattern in this agent trace.

ORIGINAL TASK:
{trace.task}

STEPS WITH DRIFT SCORES:
{steps_summary}

DRIFT STATISTICS:
- Max drift: {max(drift_history):.2f}
- Avg drift: {sum(drift_history)/len(drift_history):.2f}
- Steps above warning ({self.drift_warning_threshold}): {sum(1 for d in drift_history if d >= self.drift_warning_threshold)}

Classify the drift:
- "none": Agent stayed on task
- "scope_creep": Agent expanded beyond request
- "tangent": Agent went on unrelated side-quest
- "wrong_problem": Agent solved wrong problem entirely
- "recovery": Agent drifted but got back on track"""

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "overall_drift",
                        "strict": True,
                        "schema": _make_strict_schema(OverallDriftResult)
                    }
                }
            )
            result = OverallDriftResult.model_validate_json(response.output_text)
            return {
                "is_legitimate": result.is_legitimate,
                "drift_type": result.drift_type,
                "reason": result.reason
            }
        except Exception as e:
            # Fallback based on scores
            max_drift = max(drift_history)
            if max_drift < self.drift_warning_threshold:
                return {
                    "is_legitimate": True,
                    "drift_type": "none",
                    "reason": "Low drift scores throughout"
                }
            elif max_drift >= self.drift_critical_threshold:
                return {
                    "is_legitimate": False,
                    "drift_type": "wrong_problem",
                    "reason": f"High drift detected (max={max_drift:.2f})"
                }
            else:
                return {
                    "is_legitimate": True,
                    "drift_type": "scope_creep",
                    "reason": f"Moderate drift, possibly scope expansion"
                }
    
    def measure_drift_live(self, trace: AgentTrace, new_step: AgentStep) -> float:
        """
        Live measurement: Get drift score for a new step.
        Useful for real-time monitoring and alerts.
        
        Returns drift score (0.0-1.0)
        """
        if not new_step.content:
            return 0.0
        
        # Quick embedding-based drift
        intent_embedding = self.embeddings.embed(trace.task)
        step_embedding = self.embeddings.embed(new_step.content)
        
        import numpy as np
        similarity = float(np.dot(intent_embedding, step_embedding) / (
            np.linalg.norm(intent_embedding) * np.linalg.norm(step_embedding)
        ))
        
        return 1.0 - max(0.0, similarity)
    
    def get_drift_alert(self, trace: AgentTrace, new_step: AgentStep) -> Optional[str]:
        """
        Live alert: Get alert message if drift exceeds thresholds.
        
        Returns alert message or None
        """
        drift = self.measure_drift_live(trace, new_step)
        
        if drift >= self.drift_critical_threshold:
            return f"🚨 CRITICAL DRIFT: Step {new_step.index} has drift={drift:.2f} - agent may be solving wrong problem"
        elif drift >= self.drift_warning_threshold:
            return f"⚠️ WARNING: Step {new_step.index} has drift={drift:.2f} - agent drifting from original task"
        
        return None

