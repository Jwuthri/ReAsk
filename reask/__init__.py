"""ReAsk - LLM Conversation & Agent Evaluation"""

# Original exports
from .detector import ReAskDetector
from .models import Message, EvalResult, DetectionType

# Agent evaluation exports
from .agent_models import (
    # Signals
    TrajectorySignal,
    ToolSignal,
    SelfCorrectionSignal,
    # Data models
    ToolCall,
    AgentStep,
    AgentTrace,
    # Result models
    TrajectoryEvalResult,
    ToolEvalResult,
    SelfCorrectionResult,
    AgentBenchmarkResult,
)

# Analyzers
from .trajectory import TrajectoryAnalyzer
from .tool_eval import ToolEvaluator
from .self_correction import SelfCorrectionDetector

# Benchmarking
from .benchmark import (
    AgentBenchmark,
    BenchmarkTask,
    BenchmarkRun,
    ComparisonResult,
    LeaderboardEntry,
    SimpleAgent,
    create_mock_trace,
)

__version__ = "0.2.0"

__all__ = [
    # Original
    "ReAskDetector",
    "Message",
    "EvalResult", 
    "DetectionType",
    # Signals
    "TrajectorySignal",
    "ToolSignal",
    "SelfCorrectionSignal",
    # Data models
    "ToolCall",
    "AgentStep",
    "AgentTrace",
    # Results
    "TrajectoryEvalResult",
    "ToolEvalResult",
    "SelfCorrectionResult",
    "AgentBenchmarkResult",
    # Analyzers
    "TrajectoryAnalyzer",
    "ToolEvaluator",
    "SelfCorrectionDetector",
    # Benchmarking
    "AgentBenchmark",
    "BenchmarkTask",
    "BenchmarkRun",
    "ComparisonResult",
    "LeaderboardEntry",
    "SimpleAgent",
    "create_mock_trace",
]
