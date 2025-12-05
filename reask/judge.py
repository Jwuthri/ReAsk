"""LLM-as-Judge for response evaluation"""

import json
from typing import Optional
from pydantic import BaseModel, Field
from openai import OpenAI

from .models import Message


class JudgeResult(BaseModel):
    """Structured output for response evaluation"""
    is_bad: bool = Field(description="Whether the response was bad/inadequate")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    reason: str = Field(description="Brief explanation for the evaluation under 20 words")


class SimilarityResult(BaseModel):
    """Structured output for similarity confirmation"""
    is_same: bool = Field(description="Whether the two messages are asking the same thing")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    reason: str = Field(description="Brief explanation under 20 words")


class CorrectionResult(BaseModel):
    """Structured output for correction detection (RDM)"""
    is_correction: bool = Field(description="Whether the user is correcting/complaining about the previous response")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    reason: str = Field(description="Brief explanation under 20 words")


class HallucinationResult(BaseModel):
    """Structured output for hallucination detection"""
    is_hallucination: bool = Field(description="Whether the response contains information that contradicts or is not supported by the provided knowledge")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    reason: str = Field(description="Brief explanation under 20 words")


class FollowUpAnalysisResult(BaseModel):
    """Combined structured output for RDM + CCM detection in one call"""
    is_correction: bool = Field(description="Whether the user is correcting/complaining about the previous response (RDM)")
    correction_confidence: float = Field(ge=0.0, le=1.0, description="Confidence for correction detection")
    correction_reason: str = Field(description="Brief explanation for correction detection under 20 words")
    is_reask: bool = Field(description="Whether the user is re-asking the same question (CCM)")
    reask_confidence: float = Field(ge=0.0, le=1.0, description="Confidence for re-ask detection")
    reask_reason: str = Field(description="Brief explanation for re-ask detection under 20 words")


def _make_strict_schema(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model to a strict OpenAI-compatible JSON schema."""
    schema = model.model_json_schema()
    # Remove Pydantic-specific keys that OpenAI doesn't accept
    schema.pop("title", None)
    schema.pop("$defs", None)
    # Add required OpenAI fields
    schema["additionalProperties"] = False
    return schema


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of AI assistant responses.
Your task is to determine if an assistant's response adequately addressed the user's question/request.

Evaluate based on:
1. Did the response answer what was actually asked?
2. Was the response accurate and helpful?
3. Did it miss any key parts of the request?"""


class LLMJudge:
    """Uses an LLM to judge response quality"""
    
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        judge_model: str = "gpt-5-mini",
        ccm_model: str = "gpt-5-nano",
        rdm_model: str = "gpt-5-nano"
    ):
        self.client = client or OpenAI()
        self.judge_model = judge_model
        self.ccm_model = ccm_model
        self.rdm_model = rdm_model
    
    def evaluate(
        self,
        user_message: Message,
        assistant_response: Message,
        follow_up: Optional[Message] = None,
        knowledge: Optional[str] = None
    ) -> dict:
        """
        Evaluate if assistant response was good.
        
        Args:
            user_message: The original user question/request
            assistant_response: The assistant's response to evaluate
            follow_up: Optional follow-up from user (provides context)
            knowledge: Optional ground truth/context to evaluate against
        
        Returns:
            dict with is_bad, confidence, reason
        """
        eval_prompt = f"""{JUDGE_SYSTEM_PROMPT}

Evaluate this interaction:

USER MESSAGE:
{user_message.content}

ASSISTANT RESPONSE:
{assistant_response.content}"""

        if knowledge:
            eval_prompt += f"""

KNOWLEDGE CONTEXT (ground truth available to the assistant):
{knowledge}

Note: The assistant had access to this knowledge. The response does NOT need to include all information - only evaluate if:
1. The information provided is ACCURATE (matches the knowledge)
2. The response adequately answers the user's question
Do NOT penalize for omitting details that weren't specifically asked for."""

        if follow_up:
            eval_prompt += f"""

USER FOLLOW-UP:
{follow_up.content}

Note: The follow-up may indicate dissatisfaction with the response."""

        try:
            response = self.client.responses.create(
                model=self.judge_model,
                input=eval_prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "judge_result",
                        "strict": True,
                        "schema": _make_strict_schema(JudgeResult)
                    }
                }
            )
            
            result = JudgeResult.model_validate_json(response.output_text)
            return {
                "is_bad": result.is_bad,
                "confidence": result.confidence,
                "reason": result.reason
            }
            
        except Exception as e:
            return {
                "is_bad": False,
                "confidence": 0.0,
                "reason": f"Judge error: {str(e)}"
            }
    
    def evaluate_similarity_confirmation(
        self,
        original_question: str,
        follow_up_question: str,
        similarity_score: float
    ) -> dict:
        """
        Confirm if two questions are semantically asking the same thing.
        Used as second stage after embedding similarity check (CCM).
        
        Returns:
            dict with is_same, confidence, reason
        """
        prompt = f"""Are these two messages asking essentially the same question/making the same request?

Examples of SAME question:
- "How do I sort a list?" → "Can you show me how to sort a list?"
- "Write a Python function for X" → "I need that Python function for X"
- "What's the capital of France?" → "You didn't answer - what is France's capital?"

Examples of DIFFERENT questions:
- "How do I sort a list?" → "Now how do I filter it?"
- "What's the capital of France?" → "What about Germany?"
- "Write a sort function" → "Great, now add error handling"

MESSAGE 1:
{original_question}

MESSAGE 2:
{follow_up_question}

EMBEDDING SIMILARITY: {similarity_score:.2f}"""

        try:
            response = self.client.responses.create(
                model=self.ccm_model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "similarity_result",
                        "strict": True,
                        "schema": _make_strict_schema(SimilarityResult)
                    }
                }
            )
            
            result = SimilarityResult.model_validate_json(response.output_text)
            return {
                "is_same": result.is_same,
                "confidence": result.confidence,
                "reason": result.reason
            }
            
        except Exception as e:
            return {
                "is_same": False,
                "confidence": 0.0,
                "reason": f"Error: {str(e)}"
            }
    
    def detect_correction(self, follow_up: str) -> dict:
        """
        Detect if a follow-up message is explicitly correcting or complaining about the previous response (RDM).
        
        Returns:
            dict with is_correction, confidence, reason
        """
        prompt = f"""Is this message explicitly correcting, complaining about, or expressing dissatisfaction with a previous AI response?

Look for signals like:
- "That's not what I asked"
- "I said X not Y"
- "You missed the point"
- "Try again"
- "That's wrong"
- Any explicit frustration or correction

MESSAGE:
{follow_up}"""

        try:
            response = self.client.responses.create(
                model=self.rdm_model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "correction_result",
                        "strict": True,
                        "schema": _make_strict_schema(CorrectionResult)
                    }
                }
            )
            
            result = CorrectionResult.model_validate_json(response.output_text)
            return {
                "is_correction": result.is_correction,
                "confidence": result.confidence,
                "reason": result.reason
            }
            
        except Exception as e:
            return {
                "is_correction": False,
                "confidence": 0.0,
                "reason": f"Error: {str(e)}"
            }
    
    def analyze_follow_up(
        self,
        original_question: str,
        follow_up: str,
        similarity_score: float
    ) -> dict:
        """
        Combined RDM + CCM detection in a single LLM call.
        
        Analyzes a follow-up message to detect:
        1. RDM: Is the user correcting/complaining about the previous response?
        2. CCM: Is the user re-asking the same question?
        
        Args:
            original_question: The original user message
            follow_up: The follow-up user message
            similarity_score: Embedding similarity between the two messages
        
        Returns:
            dict with correction and reask detection results
        """
        prompt = f"""Analyze this follow-up message to detect two things:

1. CORRECTION/COMPLAINT (RDM): Is the user explicitly correcting, complaining about, or expressing dissatisfaction with the previous AI response?
   Look for signals like:
   - "That's not what I asked"
   - "I said X not Y"
   - "You missed the point"
   - "Try again"
   - "That's wrong"
   - Any explicit frustration or correction

2. RE-ASKING (CCM): Is the user re-asking essentially the same question as before?
   Examples of RE-ASKING:
   - "How do I sort a list?" → "Can you show me how to sort a list?"
   - "What's the capital of France?" → "You didn't answer - what is France's capital?"
   
   Examples of DIFFERENT (not re-asking):
   - "How do I sort a list?" → "Now how do I filter it?"
   - "What's the capital of France?" → "What about Germany?"
   - "Write a sort function" → "Great, now add error handling"

ORIGINAL MESSAGE:
{original_question}

FOLLOW-UP MESSAGE:
{follow_up}

EMBEDDING SIMILARITY: {similarity_score:.2f}

Analyze both aspects independently - a message could be both a correction AND a re-ask, or neither."""

        try:
            response = self.client.responses.create(
                model=self.ccm_model,  # Use the faster model for this combined check
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "follow_up_analysis",
                        "strict": True,
                        "schema": _make_strict_schema(FollowUpAnalysisResult)
                    }
                }
            )
            
            result = FollowUpAnalysisResult.model_validate_json(response.output_text)
            return {
                "is_correction": result.is_correction,
                "correction_confidence": result.correction_confidence,
                "correction_reason": result.correction_reason,
                "is_reask": result.is_reask,
                "reask_confidence": result.reask_confidence,
                "reask_reason": result.reask_reason
            }
            
        except Exception as e:
            return {
                "is_correction": False,
                "correction_confidence": 0.0,
                "correction_reason": f"Error: {str(e)}",
                "is_reask": False,
                "reask_confidence": 0.0,
                "reask_reason": f"Error: {str(e)}"
            }
    
    def evaluate_hallucination(
        self,
        assistant_response: Message,
        knowledge: str
    ) -> dict:
        """
        Evaluate if an assistant response contradicts or hallucinates beyond the provided knowledge.
        
        Args:
            assistant_response: The assistant's response to evaluate
            knowledge: Ground truth/context to check against
        
        Returns:
            dict with is_hallucination, confidence, reason
        """
        prompt = f"""You are a fact-checker. Evaluate if the assistant's response contains hallucinations.

A hallucination is when the response:
- States facts that contradict the provided knowledge
- Makes up information not supported by the knowledge
- Provides incorrect details that differ from the source

IMPORTANT: The response can include additional helpful context or explanations - that's fine.
Only flag as hallucination if the response contains INCORRECT or CONTRADICTORY information.

KNOWLEDGE (Ground Truth):
{knowledge}

ASSISTANT RESPONSE:
{assistant_response.content}

Does the response contain hallucinations (incorrect/contradictory information)?"""

        try:
            response = self.client.responses.create(
                model=self.judge_model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "hallucination_result",
                        "strict": True,
                        "schema": _make_strict_schema(HallucinationResult)
                    }
                }
            )
            
            result = HallucinationResult.model_validate_json(response.output_text)
            return {
                "is_hallucination": result.is_hallucination,
                "confidence": result.confidence,
                "reason": result.reason
            }
            
        except Exception as e:
            return {
                "is_hallucination": False,
                "confidence": 0.0,
                "reason": f"Error: {str(e)}"
            }
