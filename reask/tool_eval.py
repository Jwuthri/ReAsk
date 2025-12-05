"""Tool Use Quality Metrics (TUM) - Evaluate agent tool usage"""

import os
from typing import Optional, List, Tuple
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .agent_models import (
    AgentTrace, ToolCall, ToolSignal, ToolEvalResult
)

load_dotenv()


class ToolSelectionResult(BaseModel):
    """Structured output for tool selection evaluation"""
    is_correct_tool: bool = Field(description="Whether the correct tool was selected for the task")
    expected_tool: str = Field(description="What tool should have been used (same if correct)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reason: str = Field(description="Brief explanation under 30 words")


class ParameterValidationResult(BaseModel):
    """Structured output for parameter validation"""
    has_hallucination: bool = Field(description="Whether parameters contain hallucinated/invalid values")
    hallucinated_params: str = Field(description="Comma-separated list of hallucinated parameter names, or 'none'")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reason: str = Field(description="Brief explanation under 30 words")


class ToolChainResult(BaseModel):
    """Structured output for tool chain efficiency"""
    is_efficient: bool = Field(description="Whether the tool sequence is efficient")
    optimal_call_count: int = Field(description="Minimum number of calls needed")
    inefficiency_score: float = Field(ge=0.0, le=1.0, description="How inefficient (0=optimal, 1=very wasteful)")
    reason: str = Field(description="Brief explanation under 30 words")


def _make_strict_schema(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model to a strict OpenAI-compatible JSON schema."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.pop("$defs", None)
    schema["additionalProperties"] = False
    return schema


class ToolEvaluator:
    """
    Evaluates agent tool usage quality.
    
    Detects:
    - Tool Selection Errors (TSE): Wrong tool for the job
    - Parameter Hallucination (PH): Made up/invalid parameters
    - Tool Chain Inefficiency (TCI): Suboptimal sequence of calls
    """
    
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = "gpt-5-nano",
        available_tools: Optional[List[str]] = None,
    ):
        self.client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.available_tools = available_tools or []
    
    def evaluate_tool_call(
        self,
        tool_call: ToolCall,
        context: str,
        available_resources: Optional[List[str]] = None,
    ) -> ToolEvalResult:
        """
        Evaluate a single tool call.
        
        Args:
            tool_call: The tool call to evaluate
            context: What the agent was trying to accomplish
            available_resources: Known valid resources (files, APIs, etc.)
        
        Returns:
            ToolEvalResult with signal and confidence
        """
        # Check tool selection
        selection_result = self._evaluate_tool_selection(tool_call, context)
        
        if not selection_result["is_correct_tool"]:
            return ToolEvalResult(
                signal=ToolSignal.TSE,
                confidence=selection_result["confidence"],
                tool_name=tool_call.name,
                expected_tool=selection_result["expected_tool"],
                reason=selection_result["reason"],
                details={"selection_result": selection_result}
            )
        
        # Check for parameter hallucination
        param_result = self._validate_parameters(tool_call, available_resources)
        
        if param_result["has_hallucination"]:
            return ToolEvalResult(
                signal=ToolSignal.PH,
                confidence=param_result["confidence"],
                tool_name=tool_call.name,
                reason=param_result["reason"],
                details={
                    "hallucinated_params": param_result["hallucinated_params"],
                    "param_result": param_result
                }
            )
        
        return ToolEvalResult(
            signal=ToolSignal.CORRECT,
            confidence=selection_result["confidence"],
            tool_name=tool_call.name,
            reason="Tool used correctly",
            details={
                "selection_result": selection_result,
                "param_result": param_result
            }
        )
    
    def evaluate_tool_chain(
        self,
        trace: AgentTrace,
    ) -> Tuple[float, List[ToolEvalResult]]:
        """
        Evaluate the entire tool chain in a trace.
        
        Returns:
            (efficiency_score, list of individual ToolEvalResults)
        """
        tool_calls = trace.tool_calls
        
        if not tool_calls:
            return 1.0, []
        
        # Evaluate each tool call
        results = []
        for i, tc in enumerate(tool_calls):
            context = f"Task: {trace.task}"
            if i > 0:
                prev_tc = tool_calls[i - 1]
                context += f"\nPrevious tool: {prev_tc.name}({prev_tc.parameters})"
                if prev_tc.result:
                    context += f"\nPrevious result: {str(prev_tc.result)[:200]}"
            
            result = self.evaluate_tool_call(tc, context)
            results.append(result)
        
        # Check overall chain efficiency
        chain_result = self._evaluate_chain_efficiency(trace)
        efficiency_score = 1.0 - chain_result["inefficiency_score"]
        
        # Add TCI result if chain is inefficient
        if chain_result["inefficiency_score"] > 0.3:
            tci_result = ToolEvalResult(
                signal=ToolSignal.TCI,
                confidence=chain_result.get("confidence", 0.8),
                tool_name="chain",
                reason=chain_result["reason"],
                details={
                    "optimal_call_count": chain_result["optimal_call_count"],
                    "actual_call_count": len(tool_calls),
                    "inefficiency_score": chain_result["inefficiency_score"]
                }
            )
            results.append(tci_result)
        
        return efficiency_score, results
    
    def _evaluate_tool_selection(self, tool_call: ToolCall, context: str) -> dict:
        """Evaluate if the correct tool was selected"""
        tools_list = ", ".join(self.available_tools) if self.available_tools else "unknown"
        
        prompt = f"""Evaluate if the agent selected the correct tool for the task.

CONTEXT:
{context}

TOOL CALLED: {tool_call.name}
PARAMETERS: {tool_call.parameters}

AVAILABLE TOOLS: {tools_list}

Was this the right tool for what the agent was trying to accomplish?
If not, which tool should have been used instead?"""

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "tool_selection",
                        "strict": True,
                        "schema": _make_strict_schema(ToolSelectionResult)
                    }
                }
            )
            result = ToolSelectionResult.model_validate_json(response.output_text)
            return {
                "is_correct_tool": result.is_correct_tool,
                "expected_tool": result.expected_tool,
                "confidence": result.confidence,
                "reason": result.reason
            }
        except Exception as e:
            return {
                "is_correct_tool": True,
                "expected_tool": tool_call.name,
                "confidence": 0.5,
                "reason": f"Error: {str(e)}"
            }
    
    def _validate_parameters(
        self,
        tool_call: ToolCall,
        available_resources: Optional[List[str]] = None
    ) -> dict:
        """Check for hallucinated parameters"""
        resources_list = ", ".join(available_resources) if available_resources else "unknown"
        
        prompt = f"""Check if the tool parameters contain hallucinated or invalid values.

TOOL: {tool_call.name}
PARAMETERS: {tool_call.parameters}

KNOWN VALID RESOURCES: {resources_list}

Hallucinated parameters include:
- File paths that don't exist
- API endpoints that don't exist  
- IDs or references that were made up
- Values that contradict known context

Are any parameters likely hallucinated/invalid?"""

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "param_validation",
                        "strict": True,
                        "schema": _make_strict_schema(ParameterValidationResult)
                    }
                }
            )
            result = ParameterValidationResult.model_validate_json(response.output_text)
            return {
                "has_hallucination": result.has_hallucination,
                "hallucinated_params": result.hallucinated_params,
                "confidence": result.confidence,
                "reason": result.reason
            }
        except Exception as e:
            return {
                "has_hallucination": False,
                "hallucinated_params": "none",
                "confidence": 0.5,
                "reason": f"Error: {str(e)}"
            }
    
    def _evaluate_chain_efficiency(self, trace: AgentTrace) -> dict:
        """Evaluate overall tool chain efficiency"""
        tool_calls = trace.tool_calls
        
        if len(tool_calls) <= 1:
            return {
                "is_efficient": True,
                "optimal_call_count": len(tool_calls),
                "inefficiency_score": 0.0,
                "reason": "Single or no tool calls"
            }
        
        # Summarize tool chain
        chain_summary = "\n".join([
            f"{i+1}. {tc.name}({tc.parameters}) -> {str(tc.result)[:100] if tc.result else 'error: ' + str(tc.error)[:50] if tc.error else 'pending'}"
            for i, tc in enumerate(tool_calls[:15])  # Limit for prompt size
        ])
        
        prompt = f"""Evaluate the efficiency of this tool call sequence.

TASK: {trace.task}

TOOL CALLS:
{chain_summary}

Consider:
- Could fewer calls have accomplished the same result?
- Were there redundant or unnecessary calls?
- Was the sequence logical and efficient?

What is the minimum number of calls a skilled agent would need?"""

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "chain_efficiency",
                        "strict": True,
                        "schema": _make_strict_schema(ToolChainResult)
                    }
                }
            )
            result = ToolChainResult.model_validate_json(response.output_text)
            return {
                "is_efficient": result.is_efficient,
                "optimal_call_count": result.optimal_call_count,
                "inefficiency_score": result.inefficiency_score,
                "confidence": 0.8,
                "reason": result.reason
            }
        except Exception as e:
            return {
                "is_efficient": True,
                "optimal_call_count": len(tool_calls),
                "inefficiency_score": 0.0,
                "confidence": 0.5,
                "reason": f"Error: {str(e)}"
            }

