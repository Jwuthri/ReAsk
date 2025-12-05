"""Main ReAsk detector"""
import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console

from .models import Message, EvalResult, DetectionType, Role
from .embeddings import EmbeddingService
from .judge import LLMJudge

console = Console()
load_dotenv()

BANNER = """[cyan]
 /$$$$$$$             /$$$$$$            /$$      
| $$__  $$           /$$__  $$          | $$      
| $$  \\ $$  /$$$$$$ | $$  \\ $$  /$$$$$$$| $$   /$$
| $$$$$$$/ /$$__  $$| $$$$$$$$ /$$_____/| $$  /$$/
| $$__  $$| $$$$$$$$| $$__  $$|  $$$$$$ | $$$$$$/ 
| $$  \\ $$| $$_____/| $$  | $$ \\____  $$| $$_  $$ 
| $$  | $$|  $$$$$$$| $$  | $$ /$$$$$$$/| $$ \\  $$
|__/  |__/ \\_______/|__/  |__/|_______/ |__/  \\__/
[/cyan]
[dim]Detect bad LLM responses via re-ask detection[/dim]
"""


class ReAskDetector:
    """
    Detects bad LLM responses by analyzing follow-up messages.
    
    Detection methods:
    - CCM (Conversation Continuity Metric): User re-asks similar question
    - RDM (Response Dissatisfaction Metric): User explicitly corrects
    - LLM Judge: Fallback evaluation using another LLM
    """
    
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        embedding_model: str = "text-embedding-3-small",
        judge_model: str = "gpt-5-mini",
        ccm_model: str = "gpt-5-nano",
        rdm_model: str = "gpt-5-nano",
        similarity_threshold: float = 0.5,
        use_llm_confirmation: bool = True,
        use_llm_judge_fallback: bool = True,
        use_combined_rdm_ccm: bool = True,
    ):
        console.print(BANNER)
        console.print()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embeddings = EmbeddingService(self.client, embedding_model)
        self.judge = LLMJudge(
            self.client,
            judge_model=judge_model,
            ccm_model=ccm_model,
            rdm_model=rdm_model
        )
        self.similarity_threshold = similarity_threshold
        self.use_llm_confirmation = use_llm_confirmation
        self.use_llm_judge_fallback = use_llm_judge_fallback
        self.use_combined_rdm_ccm = use_combined_rdm_ccm
    
    def evaluate_response(
        self,
        user_message: Message,
        assistant_response: Message,
        follow_up: Optional[Message] = None
    ) -> EvalResult:
        """
        Evaluate if an assistant response was good or bad.
        
        Args:
            user_message: The original user question/request
            assistant_response: The assistant's response
            follow_up: Optional next user message (key for detection)
        
        Returns:
            EvalResult with detection type and confidence
        """
        # Step 1: RDM + CCM detection
        if follow_up is not None:
            if self.use_combined_rdm_ccm:
                # Combined approach: single LLM call for both
                combined_result = self._check_rdm_ccm_combined(user_message, follow_up)
                if combined_result.is_bad:
                    return combined_result
            else:
                # Separate approach: RDM first, then CCM (2 LLM calls)
                rdm_result = self._check_rdm(follow_up)
                if rdm_result.is_bad:
                    return rdm_result
                
                ccm_result = self._check_ccm(user_message, follow_up)
                if ccm_result.is_bad:
                    return ccm_result
        
        # Step 2: Check for hallucination if knowledge is provided
        if user_message.knowledge:
            hallucination_result = self._check_hallucination(assistant_response, user_message.knowledge)
            if hallucination_result.is_bad:
                return hallucination_result
        
        # Step 3: Fallback to LLM judge
        if self.use_llm_judge_fallback:
            return self._evaluate_with_judge(user_message, assistant_response, follow_up, user_message.knowledge)
        
        if follow_up is None:
            return EvalResult(
                is_bad=False,
                detection_type=DetectionType.NONE,
                confidence=0.0,
                reason="No follow-up message to analyze"
            )
        
        return EvalResult(
            is_bad=False,
            detection_type=DetectionType.NONE,
            confidence=0.8,
            reason="No re-ask or correction detected"
        )
    
    def _check_rdm_ccm_combined(self, user_message: Message, follow_up: Message) -> EvalResult:
        """
        Combined RDM + CCM check in a single LLM call.
        
        Priority: RDM (corrections) > CCM (re-asks)
        """
        # Get embedding similarity for context
        similarity = self.embeddings.similarity(
            user_message.content,
            follow_up.content
        )
        # console.print(f"[dim]Embedding similarity: {similarity:.2f}[/dim]")
        
        # Single LLM call for both RDM and CCM detection
        result = self.judge.analyze_follow_up(
            user_message.content,
            follow_up.content,
            similarity
        )
        
        # RDM takes priority - explicit corrections are clearest signal
        if result["is_correction"]:
            return EvalResult(
                is_bad=True,
                detection_type=DetectionType.RDM,
                confidence=result["correction_confidence"],
                reason=result["correction_reason"],
                details={
                    "similarity": similarity,
                    "analysis": result
                }
            )
        
        # CCM - user re-asking same question (only if similarity is high enough)
        if result["is_reask"] and similarity >= self.similarity_threshold:
            return EvalResult(
                is_bad=True,
                detection_type=DetectionType.CCM,
                confidence=result["reask_confidence"],
                reason=result["reask_reason"],
                details={
                    "similarity": similarity,
                    "analysis": result
                }
            )
        
        # Neither correction nor re-ask detected
        return EvalResult(
            is_bad=False,
            detection_type=DetectionType.NONE,
            confidence=max(1.0 - result["correction_confidence"], 1.0 - result["reask_confidence"]),
            reason="No correction or re-ask detected",
            details={
                "similarity": similarity,
                "analysis": result
            }
        )
    
    def _check_rdm(self, follow_up: Message) -> EvalResult:
        """Check for Response Dissatisfaction Metric (explicit corrections) using LLM"""
        result = self.judge.detect_correction(follow_up.content)
        
        if result["is_correction"]:
            return EvalResult(
                is_bad=True,
                detection_type=DetectionType.RDM,
                confidence=result["confidence"],
                reason=result["reason"],
                details={"llm_result": result}
            )
        
        return EvalResult(
            is_bad=False,
            detection_type=DetectionType.RDM,
            confidence=0.0
        )
    
    def _check_ccm(self, user_message: Message, follow_up: Message) -> EvalResult:
        """Check for Conversation Continuity Metric (re-asked questions)"""
        # Embedding similarity check
        similarity = self.embeddings.similarity(
            user_message.content,
            follow_up.content
        )
        
        if similarity < self.similarity_threshold:
            return EvalResult(
                is_bad=False,
                detection_type=DetectionType.CCM,
                confidence=1.0 - similarity,
                reason=f"Low similarity ({similarity:.2f}), different question",
                details={"similarity": similarity}
            )
        
        # High similarity - confirm with LLM if enabled
        if self.use_llm_confirmation:
            confirmation = self.judge.evaluate_similarity_confirmation(
                user_message.content,
                follow_up.content,
                similarity
            )
            
            if confirmation["is_same"]:
                return EvalResult(
                    is_bad=True,
                    detection_type=DetectionType.CCM,
                    confidence=confirmation["confidence"],
                    reason=f"User re-asked similar question: {confirmation['reason']}",
                    details={
                        "similarity": similarity,
                        "llm_confirmation": confirmation
                    }
                )
            else:
                return EvalResult(
                    is_bad=False,
                    detection_type=DetectionType.CCM,
                    confidence=confirmation["confidence"],
                    reason=confirmation["reason"],
                    details={"similarity": similarity}
                )
        
        # No LLM confirmation, use embedding alone
        return EvalResult(
            is_bad=True,
            detection_type=DetectionType.CCM,
            confidence=similarity,
            reason=f"High similarity ({similarity:.2f}) suggests re-ask",
            details={"similarity": similarity}
        )
    
    def _check_hallucination(self, assistant_response: Message, knowledge: str) -> EvalResult:
        """Check if the response contradicts provided knowledge (hallucination detection)"""
        result = self.judge.evaluate_hallucination(assistant_response, knowledge)
        
        if result["is_hallucination"]:
            return EvalResult(
                is_bad=True,
                detection_type=DetectionType.HALLUCINATION,
                confidence=result["confidence"],
                reason=result["reason"],
                details={"hallucination_result": result}
            )
        
        return EvalResult(
            is_bad=False,
            detection_type=DetectionType.HALLUCINATION,
            confidence=0.0
        )
    
    def _evaluate_with_judge(
        self,
        user_message: Message,
        assistant_response: Message,
        follow_up: Optional[Message],
        knowledge: Optional[str] = None
    ) -> EvalResult:
        """Fallback to LLM judge evaluation"""
        result = self.judge.evaluate(user_message, assistant_response, follow_up, knowledge)
        
        return EvalResult(
            is_bad=result["is_bad"],
            detection_type=DetectionType.LLM_JUDGE,
            confidence=result["confidence"],
            reason=result["reason"],
            details={"judge_result": result}
        )
    
    def evaluate_conversation(
        self,
        messages: list[Message]
    ) -> list[tuple[int, EvalResult]]:
        """
        Evaluate all assistant responses in a conversation.
        
        Args:
            messages: Full conversation as list of Messages
        
        Returns:
            List of (index, EvalResult) for each assistant response
        """
        results = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role == Role.USER and i + 1 < len(messages):
                next_msg = messages[i + 1]
                if next_msg.role == Role.ASSISTANT:
                    follow_up = None
                    if i + 2 < len(messages) and messages[i + 2].role == Role.USER:
                        follow_up = messages[i + 2]
                    
                    result = self.evaluate_response(msg, next_msg, follow_up)
                    results.append((i + 1, result))
                    i += 2
                    continue
            i += 1
        
        return results
