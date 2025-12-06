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


class TurnSummaryResult(BaseModel):
    """Structured output for generating a rolling conversation summary"""
    summary: str = Field(description="A concise summary (2-5 sentences) of key facts established in the conversation so far")
    key_facts: list[str] = Field(description="List of key facts/actions completed (max 5 items)")
    current_status: str = Field(description="What is the current state of the task? (1 sentence)")


def _make_strict_schema(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model to a strict OpenAI-compatible JSON schema."""
    schema = model.model_json_schema()
    # Remove Pydantic-specific keys that OpenAI doesn't accept
    schema.pop("title", None)
    schema.pop("$defs", None)
    # Add required OpenAI fields
    schema["additionalProperties"] = False
    return schema


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of AI assistant responses in multi-agent workflows.

Your task is to determine if an assistant's response MADE PROGRESS toward the user's goal.

CRITICAL RULES FOR MULTI-STEP PROCESSES:
- A response that takes a correct INTERMEDIATE STEP is GOOD (e.g., "Looking up order details" before processing a refund)
- A response does NOT need to fully complete the task in one turn
- If agents are working together (planner, executor, reviewer), evaluate the COMBINED output
- Tool executions that successfully retrieve data are GOOD responses
- Only mark as BAD if the response is:
  1. Factually incorrect
  2. Completely ignores the user's request
  3. Makes no progress toward the goal
  4. Provides wrong information

DO NOT mark as bad:
- Intermediate steps in a multi-step process
- Agents delegating to other agents
- Tool calls that successfully retrieve needed data
- Partial answers that make progress toward the goal"""


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
    
    def generate_turn_summary(
        self,
        turn_index: int,
        user_message: str,
        agent_response: str,
        previous_summary: str = ""
    ) -> dict:
        """
        Generate a rolling summary of key facts established so far.
        
        This summary is passed to subsequent turn evaluations so the judge
        has context about what happened before.
        
        Args:
            turn_index: Current turn number (0-indexed)
            user_message: The user's message in this turn
            agent_response: The agent's response in this turn
            previous_summary: Summary from previous turns
        
        Returns:
            dict with summary, key_facts, current_status
        """
        prompt = f"""Generate a concise summary of the conversation so far.

PREVIOUS CONTEXT:
{previous_summary if previous_summary else "(This is the first turn)"}

TURN {turn_index + 1}:
User: {user_message}
Agent: {agent_response}

Create a rolling summary that captures:
1. The main task/goal being worked on
2. Key facts discovered or actions taken
3. Current status of the task

Keep it concise - this will be used as context for evaluating future turns."""

        try:
            response = self.client.responses.create(
                model=self.ccm_model,  # Use faster model for summary
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "turn_summary",
                        "strict": True,
                        "schema": _make_strict_schema(TurnSummaryResult)
                    }
                }
            )
            
            result = TurnSummaryResult.model_validate_json(response.output_text)
            return {
                "summary": result.summary,
                "key_facts": result.key_facts,
                "current_status": result.current_status
            }
            
        except Exception as e:
            # Fallback: create a simple summary
            return {
                "summary": f"Turn {turn_index + 1}: User asked '{user_message[:50]}...', agent responded.",
                "key_facts": [],
                "current_status": "In progress"
            }
    
    def evaluate_with_context(
        self,
        user_message: Message,
        assistant_response: Message,
        follow_up: Optional[Message] = None,
        knowledge: Optional[str] = None,
        conversation_context: str = "",
        turn_index: int = 0
    ) -> dict:
        """
        Evaluate if assistant response was good, WITH conversation context.
        
        This is the context-aware version that knows what happened before,
        preventing false negatives like "looking up order is a bad response"
        when it's actually a correct first step.
        
        Args:
            user_message: The original user question/request
            assistant_response: The assistant's response to evaluate
            follow_up: Optional follow-up from user (provides context)
            knowledge: Optional ground truth/context to evaluate against
            conversation_context: Summary of what happened in previous turns
            turn_index: Current turn number (0-indexed)
        
        Returns:
            dict with is_bad, confidence, reason
        """
        context_section = ""
        if conversation_context:
            context_section = f"""
CONVERSATION CONTEXT (what already happened):
{conversation_context}

CRITICAL: The response below should be evaluated in context of what happened above.
- If data was already retrieved (e.g., order lookup), the agent KNOWS this information
- If a previous step established facts, the current step can BUILD on them
- An answer that references previously-established facts is GOOD, not "assuming"

---

"""

        eval_prompt = f"""{JUDGE_SYSTEM_PROMPT}

{context_section}CURRENT TURN {turn_index + 1}:

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

Note: The follow-up may indicate dissatisfaction OR may be a natural next question in the conversation.
A follow-up question like "What about X?" does NOT mean the previous response was bad."""

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
