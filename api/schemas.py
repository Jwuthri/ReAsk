"""Pydantic schemas for ReAsk API"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


# ============================================
# Analysis Stats Schema
# ============================================

class EvalStats(BaseModel):
    """Statistics from an analysis run"""
    total_responses: int
    good_responses: int
    bad_responses: int
    ccm_detections: int
    rdm_detections: int
    llm_judge_detections: int
    hallucination_detections: int
    avg_confidence: float


# ============================================
# Dataset Schemas (for API responses)
# ============================================

class DatasetSummary(BaseModel):
    """Summary of a dataset"""
    id: int
    name: Optional[str]
    task: Optional[str]
    created_at: datetime
    conversation_count: int = 0
    success: Optional[bool] = None
    total_cost: Optional[float] = None
    
    class Config:
        from_attributes = True


class DatasetDetail(DatasetSummary):
    """Detailed dataset info with conversations"""
    agents: List[dict] = []
    conversations: List[dict] = []
    latest_analysis: Optional[dict] = None


# ============================================
# Analysis Schemas
# ============================================

class AnalysisSummary(BaseModel):
    """Summary of an analysis"""
    id: int
    dataset_id: int
    status: str
    created_at: datetime
    overall_score: Optional[float] = None
    
    class Config:
        from_attributes = True


class AnalysisDetail(AnalysisSummary):
    """Detailed analysis with all results"""
    analysis_types: List[str] = []
    conversation_results: List[dict] = []
    message_results: List[dict] = []
    step_results: List[dict] = []
