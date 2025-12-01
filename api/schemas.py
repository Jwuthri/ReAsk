"""Pydantic schemas for ReAsk API"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# Message schemas
class MessageBase(BaseModel):
    role: str
    content: str


class MessageCreate(MessageBase):
    index: int


class MessageResponse(MessageBase):
    id: int
    index: int
    
    class Config:
        from_attributes = True


# Eval result schemas
class EvalResultResponse(BaseModel):
    id: int
    message_id: int
    is_bad: bool
    detection_type: str
    confidence: float
    reason: Optional[str] = None
    
    class Config:
        from_attributes = True


class MessageWithEval(MessageResponse):
    eval_result: Optional[EvalResultResponse] = None


# Conversation schemas
class ConversationBase(BaseModel):
    conversation_id: str


class ConversationResponse(ConversationBase):
    id: int
    messages: List[MessageResponse] = []
    
    class Config:
        from_attributes = True


class ConversationWithEvals(ConversationBase):
    id: int
    messages: List[MessageWithEval] = []
    
    class Config:
        from_attributes = True


# Dataset schemas
class DatasetBase(BaseModel):
    name: str


class DatasetCreate(DatasetBase):
    file_type: str


class DatasetResponse(DatasetBase):
    id: int
    file_type: str
    uploaded_at: datetime
    evaluated: bool
    conversation_count: int = 0
    message_count: int = 0
    
    class Config:
        from_attributes = True


class DatasetDetail(DatasetResponse):
    conversations: List[ConversationWithEvals] = []


# Stats schemas
class EvalStats(BaseModel):
    total_responses: int
    good_responses: int
    bad_responses: int
    ccm_detections: int
    rdm_detections: int
    llm_judge_detections: int
    avg_confidence: float


class DatasetWithStats(DatasetDetail):
    stats: Optional[EvalStats] = None


# Upload response
class UploadResponse(BaseModel):
    id: int
    name: str
    file_type: str
    conversations_imported: int
    messages_imported: int

