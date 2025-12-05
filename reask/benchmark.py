"""Comparative Agent Benchmarking (CAB) - A/B test and compare agents"""

import os
import time
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any, Protocol
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

from .agent_models import (
    AgentTrace, AgentStep, ToolCall, AgentBenchmarkResult
)
from .trajectory import TrajectoryAnalyzer
from .tool_eval import ToolEvaluator
from .self_correction import SelfCorrectionDetector
from .intent_drift import IntentDriftMeter

load_dotenv()


class AgentProtocol(Protocol):
    """Protocol for agents that can be benchmarked"""
    
    def run(self, task: str) -> AgentTrace:
        """Execute task and return trace"""
        ...
    
    @property
    def name(self) -> str:
        """Agent identifier"""
        ...


@dataclass
class BenchmarkTask:
    """A task for benchmarking"""
    name: str
    description: str
    expected_outcome: Optional[str] = None
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard
    max_steps: int = 20
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkRun:
    """A single benchmark run result"""
    agent_name: str
    task_name: str
    result: AgentBenchmarkResult
    trace: AgentTrace
    timestamp: datetime = field(default_factory=datetime.now)
    run_id: str = ""
    
    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "task_name": self.task_name,
            "success": self.result.success,
            "trajectory_score": self.result.trajectory_score,
            "tool_accuracy": self.result.tool_accuracy,
            "self_correction_score": self.result.self_correction_score,
            "intent_drift": self.result.intent_drift,
            "total_cost": self.result.total_cost,
            "total_latency_ms": self.result.total_latency_ms,
            "step_count": self.result.step_count,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
        }


@dataclass
class ComparisonResult:
    """Result of comparing multiple agents"""
    task: BenchmarkTask
    runs: List[BenchmarkRun]
    winner: Optional[str] = None
    rankings: List[str] = field(default_factory=list)
    analysis: str = ""
    
    def to_dict(self) -> dict:
        return {
            "task": self.task.name,
            "winner": self.winner,
            "rankings": self.rankings,
            "analysis": self.analysis,
            "runs": [r.to_dict() for r in self.runs]
        }


@dataclass
class LeaderboardEntry:
    """Entry in the agent leaderboard"""
    agent_name: str
    total_runs: int
    wins: int
    avg_trajectory_score: float
    avg_tool_accuracy: float
    avg_self_correction: float
    avg_drift: float
    avg_cost: float
    avg_latency_ms: float
    success_rate: float
    
    @property
    def win_rate(self) -> float:
        return self.wins / self.total_runs if self.total_runs > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "total_runs": self.total_runs,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "success_rate": self.success_rate,
            "avg_trajectory_score": self.avg_trajectory_score,
            "avg_tool_accuracy": self.avg_tool_accuracy,
            "avg_self_correction": self.avg_self_correction,
            "avg_drift": self.avg_drift,
            "avg_cost": self.avg_cost,
            "avg_latency_ms": self.avg_latency_ms,
        }


class AgentBenchmark:
    """
    A/B testing infrastructure for comparing agents.
    
    Run the same tasks through multiple agents and compare:
    - Success rates
    - Trajectory quality
    - Tool usage accuracy
    - Self-correction ability
    - Intent alignment
    - Cost and latency
    """
    
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = "gpt-5-nano",
        available_tools: Optional[List[str]] = None,
    ):
        self.client = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        
        # Initialize evaluators
        self.trajectory_analyzer = TrajectoryAnalyzer(self.client, model)
        self.tool_evaluator = ToolEvaluator(self.client, model, available_tools)
        self.self_correction_detector = SelfCorrectionDetector(self.client, model)
        self.intent_drift_meter = IntentDriftMeter(self.client, model)
        
        # Storage
        self.runs: List[BenchmarkRun] = []
        self.comparisons: List[ComparisonResult] = []
    
    def evaluate_trace(self, trace: AgentTrace) -> AgentBenchmarkResult:
        """
        Evaluate a single agent trace using all metrics.
        
        Args:
            trace: Complete agent execution trace
        
        Returns:
            AgentBenchmarkResult with all scores
        """
        # Run all evaluations
        trajectory_result = self.trajectory_analyzer.analyze(trace)
        tool_efficiency, tool_results = self.tool_evaluator.evaluate_tool_chain(trace)
        self_correction_result = self.self_correction_detector.analyze(trace)
        drift_result = self.intent_drift_meter.analyze(trace)
        
        # Calculate tool accuracy
        correct_tools = sum(1 for r in tool_results if r.signal.value == "correct")
        tool_accuracy = correct_tools / len(tool_results) if tool_results else 1.0
        
        return AgentBenchmarkResult(
            agent_name=trace.metadata.get("agent_name", "unknown"),
            task=trace.task,
            success=trace.success if trace.success is not None else False,
            trajectory_score=trajectory_result.efficiency_score,
            tool_accuracy=tool_accuracy,
            self_correction_score=self_correction_result.correction_efficiency,
            intent_drift=drift_result.drift_score,
            total_cost=trace.total_cost or 0.0,
            total_latency_ms=trace.total_duration_ms or 0,
            step_count=trace.step_count,
            details={
                "trajectory": {
                    "signal": trajectory_result.signal.value,
                    "circular_count": trajectory_result.circular_count,
                    "regression_count": trajectory_result.regression_count,
                },
                "tools": {
                    "efficiency": tool_efficiency,
                    "errors": [r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in tool_results if r.signal.value != "correct"]
                },
                "self_correction": {
                    "detected_error": self_correction_result.detected_error,
                    "correction_success": self_correction_result.correction_success,
                    "loops_before_fix": self_correction_result.loops_before_fix,
                },
                "drift": {
                    "max_drift": drift_result.drift_score,
                    "drift_step": drift_result.step_index,
                    "is_legitimate": drift_result.is_legitimate,
                }
            }
        )
    
    def run_benchmark(
        self,
        agent: AgentProtocol,
        task: BenchmarkTask,
        run_id: Optional[str] = None,
    ) -> BenchmarkRun:
        """
        Run a single benchmark for an agent on a task.
        
        Args:
            agent: Agent to benchmark
            task: Task to run
            run_id: Optional identifier for this run
        
        Returns:
            BenchmarkRun with results
        """
        run_id = run_id or f"{agent.name}-{task.name}-{int(time.time())}"
        
        # Execute agent
        start_time = time.time()
        trace = agent.run(task.description)
        trace.metadata["agent_name"] = agent.name
        
        # Evaluate
        result = self.evaluate_trace(trace)
        
        run = BenchmarkRun(
            agent_name=agent.name,
            task_name=task.name,
            result=result,
            trace=trace,
            run_id=run_id,
        )
        
        self.runs.append(run)
        return run
    
    def compare(
        self,
        agents: List[AgentProtocol],
        task: BenchmarkTask,
    ) -> ComparisonResult:
        """
        Compare multiple agents on the same task.
        
        Args:
            agents: List of agents to compare
            task: Task to run
        
        Returns:
            ComparisonResult with rankings and analysis
        """
        runs = []
        for agent in agents:
            run = self.run_benchmark(agent, task)
            runs.append(run)
        
        # Rank agents by composite score
        def score_run(run: BenchmarkRun) -> float:
            r = run.result
            # Weighted composite score
            return (
                0.3 * (1.0 if r.success else 0.0) +
                0.25 * r.trajectory_score +
                0.2 * r.tool_accuracy +
                0.15 * r.self_correction_score +
                0.1 * (1.0 - r.intent_drift)
            )
        
        ranked_runs = sorted(runs, key=score_run, reverse=True)
        rankings = [r.agent_name for r in ranked_runs]
        winner = rankings[0] if rankings else None
        
        # Generate analysis
        analysis = self._generate_comparison_analysis(ranked_runs, task)
        
        comparison = ComparisonResult(
            task=task,
            runs=runs,
            winner=winner,
            rankings=rankings,
            analysis=analysis,
        )
        
        self.comparisons.append(comparison)
        return comparison
    
    def _generate_comparison_analysis(
        self,
        ranked_runs: List[BenchmarkRun],
        task: BenchmarkTask
    ) -> str:
        """Generate human-readable comparison analysis"""
        if not ranked_runs:
            return "No runs to analyze"
        
        lines = [f"## Comparison: {task.name}\n"]
        
        for i, run in enumerate(ranked_runs):
            r = run.result
            medal = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else f"#{i+1}"))
            lines.append(f"{medal} **{run.agent_name}**")
            lines.append(f"   - Success: {'✅' if r.success else '❌'}")
            lines.append(f"   - Trajectory: {r.trajectory_score:.2f}")
            lines.append(f"   - Tools: {r.tool_accuracy:.2f}")
            lines.append(f"   - Self-correction: {r.self_correction_score:.2f}")
            lines.append(f"   - Drift: {r.intent_drift:.2f}")
            lines.append(f"   - Cost: ${r.total_cost:.4f}")
            lines.append(f"   - Latency: {r.total_latency_ms}ms")
            lines.append(f"   - Steps: {r.step_count}")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_leaderboard(
        self,
        category: Optional[str] = None,
    ) -> List[LeaderboardEntry]:
        """
        Generate leaderboard from all benchmark runs.
        
        Args:
            category: Optional category filter
        
        Returns:
            Sorted list of LeaderboardEntry
        """
        # Group runs by agent
        agent_runs: Dict[str, List[BenchmarkRun]] = {}
        for run in self.runs:
            if run.agent_name not in agent_runs:
                agent_runs[run.agent_name] = []
            agent_runs[run.agent_name].append(run)
        
        # Count wins
        agent_wins: Dict[str, int] = {name: 0 for name in agent_runs}
        for comparison in self.comparisons:
            if comparison.winner:
                agent_wins[comparison.winner] = agent_wins.get(comparison.winner, 0) + 1
        
        # Generate entries
        entries = []
        for agent_name, runs in agent_runs.items():
            results = [r.result for r in runs]
            
            entry = LeaderboardEntry(
                agent_name=agent_name,
                total_runs=len(runs),
                wins=agent_wins.get(agent_name, 0),
                avg_trajectory_score=sum(r.trajectory_score for r in results) / len(results),
                avg_tool_accuracy=sum(r.tool_accuracy for r in results) / len(results),
                avg_self_correction=sum(r.self_correction_score for r in results) / len(results),
                avg_drift=sum(r.intent_drift for r in results) / len(results),
                avg_cost=sum(r.total_cost for r in results) / len(results),
                avg_latency_ms=sum(r.total_latency_ms for r in results) // len(results),
                success_rate=sum(1 for r in results if r.success) / len(results),
            )
            entries.append(entry)
        
        # Sort by win rate, then success rate, then trajectory score
        entries.sort(key=lambda e: (e.win_rate, e.success_rate, e.avg_trajectory_score), reverse=True)
        
        return entries
    
    def export_results(self, filepath: str) -> None:
        """Export all benchmark results to JSON"""
        data = {
            "runs": [r.to_dict() for r in self.runs],
            "comparisons": [c.to_dict() for c in self.comparisons],
            "leaderboard": [e.to_dict() for e in self.get_leaderboard()],
            "exported_at": datetime.now().isoformat(),
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def load_results(self, filepath: str) -> None:
        """Load benchmark results from JSON"""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Note: This loads summary data only, not full traces
        # Full implementation would reconstruct BenchmarkRun objects


class SimpleAgent:
    """Simple wrapper to make any callable an AgentProtocol"""
    
    def __init__(
        self,
        name: str,
        run_fn: Callable[[str], AgentTrace],
    ):
        self._name = name
        self._run_fn = run_fn
    
    @property
    def name(self) -> str:
        return self._name
    
    def run(self, task: str) -> AgentTrace:
        return self._run_fn(task)


def create_mock_trace(
    task: str,
    steps: List[Dict[str, Any]],
    success: bool = True,
    agent_name: str = "mock_agent",
    total_cost: float = 0.001,
) -> AgentTrace:
    """
    Helper to create mock traces for testing.
    
    Args:
        task: The task description
        steps: List of step dicts with thought, action, tool_call, observation
        success: Whether the task succeeded
        agent_name: Name of the agent
        total_cost: Total cost in USD
    
    Returns:
        AgentTrace
    """
    trace = AgentTrace(
        task=task,
        success=success,
        total_cost=total_cost,
        metadata={"agent_name": agent_name}
    )
    
    for i, step_data in enumerate(steps):
        tool_call = None
        if "tool_call" in step_data:
            tc = step_data["tool_call"]
            tool_call = ToolCall(
                name=tc.get("name", "unknown"),
                parameters=tc.get("parameters", {}),
                result=tc.get("result"),
                error=tc.get("error"),
            )
        
        step = AgentStep(
            index=i,
            thought=step_data.get("thought"),
            action=step_data.get("action"),
            tool_call=tool_call,
            observation=step_data.get("observation"),
        )
        trace.add_step(step)
    
    return trace

