"""Data models for ReAsk"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class DetectionType(str, Enum):
    """How the bad response was detected"""
    CCM = "ccm"  # Conversation Continuity Metric - user re-asked similar question
    RDM = "rdm"  # Response Dissatisfaction Metric - explicit correction detected
    LLM_JUDGE = "llm_judge"  # LLM evaluated the response as bad
    HALLUCINATION = "hallucination"  # Response contradicts provided knowledge
    NONE = "none"  # No issues detected


@dataclass
class Message:
    """A single message in a conversation"""
    role: Role
    content: str
    knowledge: Optional[str] = None  # Ground truth/context for hallucination detection
    
    @classmethod
    def user(cls, content: str, knowledge: Optional[str] = None) -> "Message":
        return cls(Role.USER, content, knowledge)
    
    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(Role.ASSISTANT, content)


@dataclass
class EvalResult:
    """Result of evaluating a response"""
    is_bad: bool
    detection_type: DetectionType
    confidence: float  # 0.0 to 1.0
    reason: Optional[str] = None
    details: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "❌ BAD" if self.is_bad else "✅ OK"
        return f"EvalResult({status}, {self.detection_type.value}, conf={self.confidence:.2f})"

