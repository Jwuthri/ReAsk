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

