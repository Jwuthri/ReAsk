"""Data models for Agent Evaluation"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, List
from datetime import datetime


class TrajectorySignal(str, Enum):
    """Signals detected in agent trajectories"""
    CIRCULAR = "circular"       # Agent repeating similar actions
    REGRESSION = "regression"   # Agent undoing previous progress
    STALL = "stall"            # Agent stuck, not progressing
    OPTIMAL = "optimal"        # Clean path to goal
    RECOVERY = "recovery"      # Agent recovering from error
    DRIFT = "drift"            # Agent drifting from original intent


class ToolSignal(str, Enum):
    """Tool use quality signals"""
    TSE = "tool_selection_error"    # Used wrong tool
    PH = "parameter_hallucination"  # Made up parameters
    TCI = "tool_chain_inefficiency" # Suboptimal sequence
    CORRECT = "correct"             # Correct tool use


class SelfCorrectionSignal(str, Enum):
    """Self-correction detection signals"""
    ERROR_DETECTED = "error_detected"       # Agent recognized its mistake
    CORRECTION_ATTEMPTED = "correction_attempted"  # Agent tried to fix
    CORRECTION_SUCCESS = "correction_success"      # Fix worked
    CORRECTION_LOOP = "correction_loop"     # Stuck in correction loop
    GRACEFUL_STOP = "graceful_stop"         # Agent asked for help


@dataclass
class ToolCall:
    """A single tool call by an agent"""
    name: str
    parameters: dict
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None
    duration_ms: Optional[int] = None
    
    def succeeded(self) -> bool:
        return self.error is None


@dataclass
class AgentStep:
    """A single step in an agent's execution trace"""
    index: int
    thought: Optional[str] = None       # Agent's reasoning
    action: Optional[str] = None        # Action description
    tool_call: Optional[ToolCall] = None
    observation: Optional[str] = None   # Result/feedback
    timestamp: Optional[datetime] = None
    
    @property
    def content(self) -> str:
        """Get combined content for embedding"""
        parts = []
        if self.thought:
            parts.append(f"Thought: {self.thought}")
        if self.action:
            parts.append(f"Action: {self.action}")
        if self.tool_call:
            parts.append(f"Tool: {self.tool_call.name}({self.tool_call.parameters})")
        if self.observation:
            parts.append(f"Observation: {self.observation}")
        return "\n".join(parts) if parts else ""


@dataclass
class AgentTrace:
    """Complete execution trace of an agent"""
    task: str                           # Original task/intent
    steps: List[AgentStep] = field(default_factory=list)
    final_result: Optional[str] = None
    success: Optional[bool] = None
    total_cost: Optional[float] = None
    total_duration_ms: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    
    def add_step(self, step: AgentStep) -> None:
        step.index = len(self.steps)
        self.steps.append(step)
    
    @property
    def tool_calls(self) -> List[ToolCall]:
        """Get all tool calls from trace"""
        return [s.tool_call for s in self.steps if s.tool_call]
    
    @property
    def step_count(self) -> int:
        return len(self.steps)


@dataclass
class TrajectoryEvalResult:
    """Result of trajectory analysis"""
    signal: TrajectorySignal
    confidence: float               # 0.0 to 1.0
    efficiency_score: float         # 0.0 to 1.0 (1.0 = optimal path)
    circular_count: int             # Number of circular patterns detected
    regression_count: int           # Number of regressions detected
    reason: str
    details: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "✅" if self.signal == TrajectorySignal.OPTIMAL else "⚠️"
        return f"TrajectoryEval({status} {self.signal.value}, eff={self.efficiency_score:.2f}, conf={self.confidence:.2f})"


@dataclass
class ToolEvalResult:
    """Result of tool use evaluation"""
    signal: ToolSignal
    confidence: float
    tool_name: str
    reason: str
    expected_tool: Optional[str] = None  # What tool should have been used
    details: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "✅" if self.signal == ToolSignal.CORRECT else "❌"
        return f"ToolEval({status} {self.signal.value}, {self.tool_name}, conf={self.confidence:.2f})"


@dataclass
class SelfCorrectionResult:
    """Result of self-correction analysis"""
    detected_error: bool
    correction_attempt: bool
    correction_success: bool
    loops_before_fix: int
    self_awareness_score: float     # 0.0 to 1.0
    correction_efficiency: float    # 0.0 to 1.0 (1.0 = fixed immediately)
    reason: str
    details: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        if self.correction_success:
            status = "✅ RECOVERED"
        elif self.correction_attempt:
            status = "⚠️ TRYING"
        elif self.detected_error:
            status = "👁️ AWARE"
        else:
            status = "❌ UNAWARE"
        return f"SelfCorrection({status}, awareness={self.self_awareness_score:.2f})"


@dataclass
class IntentDriftResult:
    """Result of intent drift analysis"""
    drift_score: float              # 0.0 to 1.0 (1.0 = completely off track)
    step_index: int                 # Which step has max drift
    is_legitimate: bool             # Was drift requested by user?
    reason: str
    drift_history: List[float] = field(default_factory=list)  # Drift at each step
    details: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "✅" if self.drift_score < 0.3 else ("⚠️" if self.drift_score < 0.6 else "❌")
        return f"IntentDrift({status} drift={self.drift_score:.2f}, step={self.step_index})"


@dataclass 
class AgentBenchmarkResult:
    """Result of benchmarking an agent on a task"""
    agent_name: str
    task: str
    success: bool
    trajectory_score: float
    tool_accuracy: float
    self_correction_score: float
    intent_drift: float
    total_cost: float
    total_latency_ms: int
    step_count: int
    details: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        status = "✅" if self.success else "❌"
        return f"Benchmark({status} {self.agent_name}, score={self.trajectory_score:.2f}, cost=${self.total_cost:.4f})"

