"""Database setup and models for ReAsk API"""

from datetime import datetime
from enum import Enum
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Index
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

DATABASE_URL = "sqlite:///./reask_data.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ============================================
# Dataset Models (for CSV/JSON uploads)
# ============================================

class Dataset(Base):
    """A dataset uploaded by the user (CSV or JSON file)"""
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)  # csv or json
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="dataset", cascade="all, delete-orphan")
    analyses = relationship("DatasetAnalysis", back_populates="dataset", cascade="all, delete-orphan")


class Conversation(Base):
    """A conversation within a dataset"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    conversation_id = Column(String(255), nullable=False)  # Original ID from file
    
    dataset = relationship("Dataset", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.index")


class Message(Base):
    """A message in a conversation"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    index = Column(Integer, nullable=False)
    role = Column(String(20), nullable=False)  # user or assistant
    content = Column(Text, nullable=False)
    knowledge = Column(Text, nullable=True)
    
    conversation = relationship("Conversation", back_populates="messages")
    eval_result = relationship("MessageAnalysisResult", back_populates="message", uselist=False, cascade="all, delete-orphan")


class DatasetAnalysis(Base):
    """An analysis run on a dataset"""
    __tablename__ = "dataset_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Progress
    total_messages = Column(Integer, default=0)
    analyzed_messages = Column(Integer, default=0)
    
    # Results summary
    total_responses = Column(Integer, nullable=True)
    bad_responses = Column(Integer, nullable=True)
    ccm_detections = Column(Integer, nullable=True)
    rdm_detections = Column(Integer, nullable=True)
    llm_judge_detections = Column(Integer, nullable=True)
    hallucination_detections = Column(Integer, nullable=True)
    
    error_message = Column(Text, nullable=True)
    
    dataset = relationship("Dataset", back_populates="analyses")
    message_results = relationship("MessageAnalysisResult", back_populates="analysis", cascade="all, delete-orphan")


class MessageAnalysisResult(Base):
    """Analysis result for a single message (saved in real-time)"""
    __tablename__ = "message_analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("dataset_analyses.id", ondelete="CASCADE"), nullable=False)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    
    is_bad = Column(Boolean, nullable=False)
    detection_type = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    analysis = relationship("DatasetAnalysis", back_populates="message_results")
    message = relationship("Message", back_populates="eval_result")
    
    __table_args__ = (
        Index('ix_message_analysis_unique', 'analysis_id', 'message_id', unique=True),
    )


# ============================================
# Multi-Agent Session Models
# ============================================

class AgentSession(Base):
    """A multi-agent session - the main container"""
    __tablename__ = "agent_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=True)
    initial_task = Column(Text, nullable=True)
    success = Column(Boolean, nullable=True)
    
    # Session metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    total_cost = Column(Float, nullable=True)
    total_latency_ms = Column(Integer, nullable=True)  # Total session latency in ms
    metadata_json = Column(Text, nullable=True)  # Extra metadata as JSON
    
    # Analysis results (stored as JSON strings)
    overall_score = Column(Float, nullable=True)
    analysis_types = Column(Text, nullable=True)  # JSON array of analysis types run
    conversation_result = Column(Text, nullable=True)  # JSON
    trajectory_result = Column(Text, nullable=True)  # JSON
    tools_result = Column(Text, nullable=True)  # JSON
    self_correction_result = Column(Text, nullable=True)  # JSON
    intent_drift_result = Column(Text, nullable=True)  # JSON
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    agents = relationship("AgentDefinition", back_populates="session", cascade="all, delete-orphan")
    turns = relationship("SessionTurn", back_populates="session", cascade="all, delete-orphan", order_by="SessionTurn.turn_index")
    analyses = relationship("AgentAnalysis", back_populates="session", cascade="all, delete-orphan")


class AgentDefinition(Base):
    """Definition of an agent in a session"""
    __tablename__ = "agent_definitions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("agent_sessions.id", ondelete="CASCADE"), nullable=False)
    
    agent_id = Column(String(100), nullable=False)  # e.g., "agent1"
    name = Column(String(255), nullable=True)  # e.g., "ReasoningAgent"
    role = Column(String(100), nullable=True)  # e.g., "primary_reasoner"
    description = Column(Text, nullable=True)
    capabilities_json = Column(Text, nullable=True)  # JSON array
    tools_available_json = Column(Text, nullable=True)  # JSON array of tool definitions
    config_json = Column(Text, nullable=True)  # Extra config
    
    session = relationship("AgentSession", back_populates="agents")
    interactions = relationship("AgentInteraction", back_populates="agent_def", cascade="all, delete-orphan")


class SessionTurn(Base):
    """A turn in the multi-agent conversation"""
    __tablename__ = "session_turns"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("agent_sessions.id", ondelete="CASCADE"), nullable=False)
    turn_index = Column(Integer, nullable=False)
    
    # User message that triggered this turn
    user_message = Column(Text, nullable=True)  # Can be null for agent-to-agent turns
    
    # Final response to user (if any)
    final_response = Column(Text, nullable=True)
    responding_agent_id = Column(String(100), nullable=True)
    
    session = relationship("AgentSession", back_populates="turns")
    interactions = relationship("AgentInteraction", back_populates="turn", cascade="all, delete-orphan", order_by="AgentInteraction.sequence")
    analysis_results = relationship("TurnAnalysisResult", back_populates="turn", cascade="all, delete-orphan")


class AgentInteraction(Base):
    """What a specific agent did in a turn"""
    __tablename__ = "agent_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    turn_id = Column(Integer, ForeignKey("session_turns.id", ondelete="CASCADE"), nullable=False)
    agent_def_id = Column(Integer, ForeignKey("agent_definitions.id", ondelete="CASCADE"), nullable=False)
    sequence = Column(Integer, nullable=False)  # Order in the turn
    
    agent_id = Column(String(100), nullable=False)  # Denormalized for easy access
    
    # The agent's response (if any)
    agent_response = Column(Text, nullable=True)
    
    # Tool execution result (for executor agents)
    tool_execution_json = Column(Text, nullable=True)  # {tool_name, parameters, output, error}
    
    # Performance metrics
    latency_ms = Column(Integer, nullable=True)  # Time taken for this interaction
    token_count = Column(Integer, nullable=True)  # Tokens used
    
    turn = relationship("SessionTurn", back_populates="interactions")
    agent_def = relationship("AgentDefinition", back_populates="interactions")
    steps = relationship("AgentStep", back_populates="interaction", cascade="all, delete-orphan", order_by="AgentStep.step_index")


class AgentStep(Base):
    """A step within an agent interaction (thought, tool call, action)"""
    __tablename__ = "agent_steps"
    
    id = Column(Integer, primary_key=True, index=True)
    interaction_id = Column(Integer, ForeignKey("agent_interactions.id", ondelete="CASCADE"), nullable=False)
    step_index = Column(Integer, nullable=False)
    
    # Step types
    step_type = Column(String(50), nullable=False)  # thought, tool_call, action, observation
    content = Column(Text, nullable=True)  # For thought/action/observation
    
    # Tool call details (if step_type == 'tool_call')
    tool_name = Column(String(100), nullable=True)
    tool_parameters_json = Column(Text, nullable=True)
    tool_result = Column(Text, nullable=True)
    tool_error = Column(Text, nullable=True)
    
    # Performance metrics
    latency_ms = Column(Integer, nullable=True)  # Time taken for tool call
    
    interaction = relationship("AgentInteraction", back_populates="steps")


class AgentAnalysis(Base):
    """
    Analysis run on an agent session.
    
    Evaluation happens at 3 levels:
    
    1. SESSION LEVEL (stored here):
       - overall_score: Aggregated score across all metrics
       - intent_drift: Did the agents drift from original task?
       - coordination: How well did agents work together?
       
    2. TURN LEVEL (TurnAnalysisResult):
       - conversation: CCM/RDM/Hallucination on user ↔ final_response
       - Was the final response good for the user?
       
    3. INTERACTION LEVEL (InteractionAnalysisResult):
       - tool_use: Did this agent use tools correctly?
       - reasoning: Was this agent's reasoning sound?
       - handoff: Did this agent hand off to next agent properly?
       - Each agent in each turn is evaluated separately
    """
    __tablename__ = "agent_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("agent_sessions.id", ondelete="CASCADE"), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # What analyses to run
    analysis_types_json = Column(Text, nullable=False)  # JSON array
    
    # Progress tracking
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    current_analysis = Column(String(50), nullable=True)
    
    # ========== SESSION-LEVEL RESULTS ==========
    overall_score = Column(Float, nullable=True)
    
    # Intent drift across the entire session
    intent_drift_result_json = Column(Text, nullable=True)
    
    # Multi-agent coordination score
    coordination_score = Column(Float, nullable=True)
    coordination_result_json = Column(Text, nullable=True)  # Detailed coordination analysis
    
    # Agent-to-agent handoff quality summary
    handoff_summary_json = Column(Text, nullable=True)
    
    # Per-agent summary scores (aggregated from interactions)
    agent_scores_json = Column(Text, nullable=True)  # {"agent1": 0.85, "agent2": 0.72}
    
    # ========== AGGREGATED RESULTS (for backwards compat) ==========
    conversation_result_json = Column(Text, nullable=True)  # Aggregated turn results
    trajectory_result_json = Column(Text, nullable=True)
    tools_result_json = Column(Text, nullable=True)  # Aggregated tool use
    self_correction_result_json = Column(Text, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    last_successful_turn = Column(Integer, nullable=True)
    
    session = relationship("AgentSession", back_populates="analyses")
    turn_results = relationship("TurnAnalysisResult", back_populates="analysis", cascade="all, delete-orphan")
    interaction_results = relationship("InteractionAnalysisResult", back_populates="analysis", cascade="all, delete-orphan")


class TurnAnalysisResult(Base):
    """Analysis result for a single turn (saved in real-time)"""
    __tablename__ = "turn_analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("agent_analyses.id", ondelete="CASCADE"), nullable=False)
    turn_id = Column(Integer, ForeignKey("session_turns.id", ondelete="CASCADE"), nullable=False)
    turn_index = Column(Integer, nullable=False)
    
    # Conversation analysis results (user ↔ agents)
    is_bad = Column(Boolean, nullable=True)
    detection_type = Column(String(20), nullable=True)  # ccm, rdm, llm_judge, hallucination
    confidence = Column(Float, nullable=True)
    reason = Column(Text, nullable=True)
    
    # Turn-level metrics
    drift_score = Column(Float, nullable=True)
    agent_coordination_score = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    analysis = relationship("AgentAnalysis", back_populates="turn_results")
    turn = relationship("SessionTurn", back_populates="analysis_results")
    
    __table_args__ = (
        Index('ix_turn_analysis_unique', 'analysis_id', 'turn_id', unique=True),
    )


class InteractionAnalysisResult(Base):
    """
    Per-agent evaluation within a turn.
    
    For each agent interaction in a turn, we evaluate:
    - Did this agent use the right tools?
    - Was the reasoning sound?
    - Did it hand off properly to the next agent?
    - Did it follow its role correctly?
    
    Example: Turn 1 has 3 interactions (Planner → Executor → Reviewer)
    → 3 InteractionAnalysisResult records
    """
    __tablename__ = "interaction_analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("agent_analyses.id", ondelete="CASCADE"), nullable=False)
    interaction_id = Column(Integer, ForeignKey("agent_interactions.id", ondelete="CASCADE"), nullable=False)
    
    # Which agent and which turn
    agent_id = Column(String(100), nullable=False)
    turn_index = Column(Integer, nullable=False)
    sequence_in_turn = Column(Integer, nullable=False)  # Order in the turn
    
    # ========== TOOL USE EVALUATION ==========
    tool_use_score = Column(Float, nullable=True)  # 0-1 score
    tool_selection_correct = Column(Boolean, nullable=True)  # Did it pick right tool?
    tool_params_valid = Column(Boolean, nullable=True)  # Were params correct?
    tool_use_issues_json = Column(Text, nullable=True)  # List of tool issues
    
    # ========== REASONING EVALUATION ==========
    reasoning_score = Column(Float, nullable=True)  # 0-1 score
    reasoning_clear = Column(Boolean, nullable=True)
    reasoning_follows_role = Column(Boolean, nullable=True)  # Does reasoning match agent's role?
    reasoning_issues_json = Column(Text, nullable=True)
    
    # ========== HANDOFF EVALUATION ==========
    handoff_score = Column(Float, nullable=True)  # 0-1 score (null if last in turn)
    handoff_to_agent = Column(String(100), nullable=True)  # Who it handed off to
    handoff_context_preserved = Column(Boolean, nullable=True)  # Did it pass enough context?
    handoff_issues_json = Column(Text, nullable=True)
    
    # ========== RESPONSE QUALITY ==========
    response_quality_score = Column(Float, nullable=True)
    response_appropriate = Column(Boolean, nullable=True)  # Was response appropriate for role?
    
    # ========== OVERALL ==========
    overall_score = Column(Float, nullable=True)  # Weighted average
    issues_json = Column(Text, nullable=True)  # All issues combined
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    analysis = relationship("AgentAnalysis", back_populates="interaction_results")
    interaction = relationship("AgentInteraction")
    
    __table_args__ = (
        Index('ix_interaction_analysis_unique', 'analysis_id', 'interaction_id', unique=True),
    )


# ============================================
# Utility functions
# ============================================

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


# Aliases for backwards compatibility / shorter names
AgentTraceDB = AgentSession
AgentTurnDB = SessionTurn
AnalysisJobDB = AgentAnalysis
