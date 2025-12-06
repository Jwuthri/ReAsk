"""
Clean Database Schema for ReAsk API

Structure:
- Dataset: Container for evaluation data (from API payload)
  - Has 1 Agent list (shared across all conversations)
  - Has N Conversations
  - Has N Analyses

- Conversation: A conversation thread within a dataset (maps to "turn" in old schema)
  - Has N Messages

- Message: A single message (user or assistant interaction in a conversation)
  - Has N Steps (for messages with reasoning/tool calls)

- Analysis: An evaluation run on a Dataset
  - Produces ConversationResult, MessageResult, StepResult
"""

from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, 
    DateTime, ForeignKey, Text, Index, JSON
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

DATABASE_URL = "sqlite:///./reask_data.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ============================================
# Core Data Models
# ============================================

class Dataset(Base):
    """
    A dataset containing conversations to evaluate.
    This is the top-level container for all data.
    """
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=True)
    task = Column(Text, nullable=True)  # Initial task/goal
    
    # Metrics
    total_cost = Column(Float, nullable=True)
    total_latency_ms = Column(Integer, nullable=True)
    success = Column(Boolean, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON, nullable=True)
    
    # Relationships
    agents = relationship("Agent", back_populates="dataset", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="dataset", cascade="all, delete-orphan", order_by="Conversation.conversation_index")
    analyses = relationship("Analysis", back_populates="dataset", cascade="all, delete-orphan")


class Agent(Base):
    """
    Definition of an agent in a dataset.
    All conversations in the dataset share these agent definitions.
    """
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    
    agent_id = Column(String(100), nullable=False)  # e.g., "planner", "executor"
    name = Column(String(255), nullable=True)
    role = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    capabilities_json = Column(JSON, nullable=True)
    tools_available_json = Column(JSON, nullable=True)
    config_json = Column(JSON, nullable=True)
    
    dataset = relationship("Dataset", back_populates="agents")


class Conversation(Base):
    """
    A conversation within a dataset.
    Each conversation represents one user interaction thread.
    """
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    
    conversation_index = Column(Integer, nullable=False)  # 0, 1, 2...
    user_input = Column(Text, nullable=True)  # The user's input
    final_response = Column(Text, nullable=True)  # Final response to user
    responding_agent_id = Column(String(100), nullable=True)
    
    dataset = relationship("Dataset", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.sequence")
    results = relationship("ConversationResult", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """
    A message in a conversation.
    Each message represents one agent's contribution.
    """
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    agent_id = Column(String(100), nullable=False)  # Which agent produced this
    
    sequence = Column(Integer, nullable=False)  # Order in conversation
    content = Column(Text, nullable=True)  # The message content/response
    
    # For tool execution results
    tool_execution_json = Column(JSON, nullable=True)
    
    # Performance
    latency_ms = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    
    conversation = relationship("Conversation", back_populates="messages")
    steps = relationship("Step", back_populates="message", cascade="all, delete-orphan", order_by="Step.step_index")
    results = relationship("MessageResult", back_populates="message", cascade="all, delete-orphan")


class Step(Base):
    """
    A step within a message.
    Captures reasoning, tool calls, observations.
    """
    __tablename__ = "steps"
    
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    
    step_index = Column(Integer, nullable=False)
    step_type = Column(String(50), nullable=False)  # thought, tool_call, action, observation
    content = Column(Text, nullable=True)
    
    # Tool call details
    tool_name = Column(String(100), nullable=True)
    tool_parameters_json = Column(JSON, nullable=True)
    tool_result = Column(Text, nullable=True)
    tool_error = Column(Text, nullable=True)
    
    # Performance
    latency_ms = Column(Integer, nullable=True)
    
    message = relationship("Message", back_populates="steps")
    results = relationship("StepResult", back_populates="step", cascade="all, delete-orphan")


# ============================================
# Analysis Models
# ============================================

class Analysis(Base):
    """
    An analysis run on a dataset.
    """
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    
    status = Column(String(20), nullable=False, default="pending")
    analysis_types_json = Column(JSON, nullable=False)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Progress
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    current_analysis = Column(String(50), nullable=True)
    
    # Overall Results
    overall_score = Column(Float, nullable=True)
    coordination_score = Column(Float, nullable=True)
    
    # Per-agent scores
    agent_scores_json = Column(JSON, nullable=True)
    per_agent_scores_json = Column(JSON, nullable=True)
    
    # Detailed results by type
    conversation_result_json = Column(JSON, nullable=True)
    trajectory_result_json = Column(JSON, nullable=True)
    tools_result_json = Column(JSON, nullable=True)
    self_correction_result_json = Column(JSON, nullable=True)
    
    # Context-aware evaluation
    turn_summaries_json = Column(JSON, nullable=True)
    goal_hierarchy_json = Column(JSON, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    last_successful_turn = Column(Integer, nullable=True)
    
    dataset = relationship("Dataset", back_populates="analyses")
    conversation_results = relationship("ConversationResult", back_populates="analysis", cascade="all, delete-orphan")
    message_results = relationship("MessageResult", back_populates="analysis", cascade="all, delete-orphan")
    step_results = relationship("StepResult", back_populates="analysis", cascade="all, delete-orphan")


class ConversationResult(Base):
    """
    Analysis result for a conversation.
    """
    __tablename__ = "conversation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    conversation_index = Column(Integer, nullable=False)
    
    # Conversation-level detection
    is_bad = Column(Boolean, nullable=True)
    detection_type = Column(String(20), nullable=True)
    confidence = Column(Float, nullable=True)
    reason = Column(Text, nullable=True)
    
    # Context-aware fields
    context_summary = Column(Text, nullable=True)
    active_goal = Column(Text, nullable=True)
    context_used = Column(Boolean, nullable=True)
    
    # Scores
    drift_score = Column(Float, nullable=True)
    coordination_score = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    analysis = relationship("Analysis", back_populates="conversation_results")
    conversation = relationship("Conversation", back_populates="results")
    
    __table_args__ = (
        Index('ix_conversation_result_unique', 'analysis_id', 'conversation_id', unique=True),
    )


class MessageResult(Base):
    """
    Analysis result for a message.
    """
    __tablename__ = "message_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    
    # Which agent and sequence
    agent_id = Column(String(100), nullable=False)
    conversation_index = Column(Integer, nullable=False)
    sequence_in_conversation = Column(Integer, nullable=False)
    
    # Tool use evaluation
    tool_use_score = Column(Float, nullable=True)
    tool_selection_correct = Column(Boolean, nullable=True)
    tool_params_valid = Column(Boolean, nullable=True)
    tool_use_issues_json = Column(JSON, nullable=True)
    
    # Reasoning evaluation
    reasoning_score = Column(Float, nullable=True)
    reasoning_clear = Column(Boolean, nullable=True)
    reasoning_follows_role = Column(Boolean, nullable=True)
    reasoning_issues_json = Column(JSON, nullable=True)
    
    # Handoff evaluation
    handoff_score = Column(Float, nullable=True)
    handoff_to_agent = Column(String(100), nullable=True)
    handoff_context_preserved = Column(Boolean, nullable=True)
    handoff_issues_json = Column(JSON, nullable=True)
    
    # Response quality
    response_quality_score = Column(Float, nullable=True)
    response_appropriate = Column(Boolean, nullable=True)
    
    # Overall
    overall_score = Column(Float, nullable=True)
    issues_json = Column(JSON, nullable=True)
    
    # Detailed breakdowns
    tool_use_details_json = Column(JSON, nullable=True)
    self_correction_details_json = Column(JSON, nullable=True)
    response_quality_details_json = Column(JSON, nullable=True)
    intent_drift_details_json = Column(JSON, nullable=True)  # Per-agent intent drift
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    analysis = relationship("Analysis", back_populates="message_results")
    message = relationship("Message", back_populates="results")
    
    __table_args__ = (
        Index('ix_message_result_unique', 'analysis_id', 'message_id', unique=True),
    )


class StepResult(Base):
    """
    Analysis result for a step.
    """
    __tablename__ = "step_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    step_id = Column(Integer, ForeignKey("steps.id", ondelete="CASCADE"), nullable=False)
    
    # Tool evaluation
    tool_selection_correct = Column(Boolean, nullable=True)
    tool_params_valid = Column(Boolean, nullable=True)
    tool_use_score = Column(Float, nullable=True)
    
    # Reasoning
    reasoning_score = Column(Float, nullable=True)
    
    # Self-correction
    detected_error = Column(Boolean, nullable=True)
    correction_attempted = Column(Boolean, nullable=True)
    correction_success = Column(Boolean, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    analysis = relationship("Analysis", back_populates="step_results")
    step = relationship("Step", back_populates="results")
    
    __table_args__ = (
        Index('ix_step_result_unique', 'analysis_id', 'step_id', unique=True),
    )


# ============================================
# Utility functions
# ============================================

def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)


def drop_all_tables():
    """Drop all tables (for fresh start)"""
    Base.metadata.drop_all(bind=engine)


def reset_db():
    """Drop and recreate all tables"""
    drop_all_tables()
    init_db()


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================
# Backwards compatibility aliases
# ============================================
# These allow existing code to work while we migrate

AgentSession = Dataset
AgentDefinition = Agent
SessionTurn = Conversation
AgentInteraction = Message
AgentStep = Step  # Direct alias
AgentStepDB = Step  # Also aliased for imports that use this name
AgentAnalysis = Analysis
TurnAnalysisResult = ConversationResult
InteractionAnalysisResult = MessageResult

# Additional aliases from old code
AgentTraceDB = Dataset
AgentTurnDB = Conversation
AnalysisJobDB = Analysis
