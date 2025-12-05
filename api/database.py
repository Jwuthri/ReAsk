"""Database setup and models for ReAsk API"""

from datetime import datetime
from enum import Enum
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

DATABASE_URL = "sqlite:///./reask_data.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)  # csv or json
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    evaluated = Column(Boolean, default=False)
    
    conversations = relationship("Conversation", back_populates="dataset", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    conversation_id = Column(String(255), nullable=False)  # Original ID from file
    
    dataset = relationship("Dataset", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.index")


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    index = Column(Integer, nullable=False)  # Order in conversation
    role = Column(String(20), nullable=False)  # user or assistant
    content = Column(Text, nullable=False)
    knowledge = Column(Text, nullable=True)  # Optional knowledge/context for RAG evaluation
    
    conversation = relationship("Conversation", back_populates="messages")
    eval_result = relationship("EvalResult", back_populates="message", uselist=False, cascade="all, delete-orphan")


class EvalResult(Base):
    __tablename__ = "eval_results"
    
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False, unique=True)
    is_bad = Column(Boolean, nullable=False)
    detection_type = Column(String(20), nullable=False)  # ccm, rdm, llm_judge, none
    confidence = Column(Float, nullable=False)
    reason = Column(Text, nullable=True)
    
    message = relationship("Message", back_populates="eval_result")


# ============================================
# Agent Trace Models (Turn-based)
# ============================================

class AgentTraceDB(Base):
    __tablename__ = "agent_traces"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=True)  # Optional name for the trace
    initial_task = Column(Text, nullable=True)  # Optional initial task description
    success = Column(Boolean, nullable=True)
    total_cost = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Analysis results (stored as JSON strings)
    overall_score = Column(Float, nullable=True)
    analysis_types = Column(Text, nullable=True)  # JSON array of types run
    
    # Individual results as JSON
    conversation_result = Column(Text, nullable=True)
    trajectory_result = Column(Text, nullable=True)
    tools_result = Column(Text, nullable=True)
    self_correction_result = Column(Text, nullable=True)
    intent_drift_result = Column(Text, nullable=True)
    
    turns = relationship("AgentTurnDB", back_populates="trace", cascade="all, delete-orphan", order_by="AgentTurnDB.index")


class AgentTurnDB(Base):
    """A turn in the conversation: user message -> agent steps -> agent response"""
    __tablename__ = "agent_turns"
    
    id = Column(Integer, primary_key=True, index=True)
    trace_id = Column(Integer, ForeignKey("agent_traces.id", ondelete="CASCADE"), nullable=False)
    index = Column(Integer, nullable=False)
    user_message = Column(Text, nullable=False)
    agent_response = Column(Text, nullable=False)
    
    trace = relationship("AgentTraceDB", back_populates="turns")
    steps = relationship("AgentStepDB", back_populates="turn", cascade="all, delete-orphan", order_by="AgentStepDB.index")


class AgentStepDB(Base):
    __tablename__ = "agent_steps"
    
    id = Column(Integer, primary_key=True, index=True)
    turn_id = Column(Integer, ForeignKey("agent_turns.id", ondelete="CASCADE"), nullable=False)
    index = Column(Integer, nullable=False)
    thought = Column(Text, nullable=True)
    action = Column(Text, nullable=True)
    observation = Column(Text, nullable=True)
    
    # Tool call stored as JSON
    tool_call_json = Column(Text, nullable=True)
    
    turn = relationship("AgentTurnDB", back_populates="steps")


class AnalysisJobDB(Base):
    """Background analysis job tracking"""
    __tablename__ = "analysis_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    status = Column(String(20), nullable=False, default="pending")  # pending, running, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Input data (stored as JSON)
    trace_json = Column(Text, nullable=False)
    analysis_types_json = Column(Text, nullable=False)
    
    # Progress tracking
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    current_analysis = Column(String(50), nullable=True)
    progress_details_json = Column(Text, nullable=True)  # Turn-by-turn results as they come in
    
    # Result (stored as JSON when complete)
    result_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Link to saved trace (set when job completes and saves)
    saved_trace_id = Column(Integer, ForeignKey("agent_traces.id", ondelete="SET NULL"), nullable=True)


def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

