"""
Agent Evaluation Examples - Demonstrating all new evaluation features

Features demonstrated:
1. Agent Trajectory Analysis (ATA) - Detect circular patterns, regressions
2. Tool Use Quality Metrics (TUM) - Evaluate tool selection and parameters
3. Self-Correction Detection (SCD) - Measure self-awareness and recovery
4. Comparative Agent Benchmarking (CAB) - A/B test multiple agents
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from reask import (
    # Agent models
    AgentTrace, AgentStep, ToolCall,
    TrajectorySignal, ToolSignal,
    # Analyzers
    TrajectoryAnalyzer,
    ToolEvaluator,
    SelfCorrectionDetector,
    # Benchmarking
    AgentBenchmark,
    BenchmarkTask,
    SimpleAgent,
    create_mock_trace,
)

console = Console()


def display_section(title: str):
    """Display a section header"""
    console.print()
    console.rule(f"[bold blue]{title}[/bold blue]")
    console.print()


def example_trajectory_analysis():
    """Example 1: Trajectory Analysis - Detect circular behavior"""
    display_section("1. Agent Trajectory Analysis (ATA)")
    
    # Create a trace with circular behavior
    trace = AgentTrace(task="Find the bug in the authentication code")
    
    trace.add_step(AgentStep(
        index=0,
        thought="I need to look at the auth files",
        action="Read auth.py",
        observation="Found auth.py with login() function"
    ))
    trace.add_step(AgentStep(
        index=1,
        thought="Let me check the login function",
        action="Read login() function",
        observation="Function looks fine, checking dependencies"
    ))
    trace.add_step(AgentStep(
        index=2,
        thought="Maybe the issue is in auth.py",
        action="Read auth.py again",
        observation="Same file, seeing login() function"
    ))
    trace.add_step(AgentStep(
        index=3,
        thought="Let me re-examine the login function",
        action="Read login() function again",
        observation="Still looks the same..."
    ))
    
    console.print(Panel(
        f"Task: {trace.task}\n"
        f"Steps: {trace.step_count}\n"
        f"[dim]Agent is reading the same files repeatedly...[/dim]",
        title="📋 Trace Summary"
    ))
    
    # Analyze trajectory
    analyzer = TrajectoryAnalyzer()
    
    with console.status("[bold green]Analyzing trajectory..."):
        result = analyzer.analyze(trace)
    
    # Display results
    color = "red" if result.signal != TrajectorySignal.OPTIMAL else "green"
    console.print(Panel(
        f"Signal: [{color}]{result.signal.value.upper()}[/{color}]\n"
        f"Efficiency: {result.efficiency_score:.2f}\n"
        f"Circular patterns: {result.circular_count}\n"
        f"Regressions: {result.regression_count}\n\n"
        f"[dim]{result.reason}[/dim]",
        title="🔄 Trajectory Analysis Result",
        border_style=color
    ))


def example_tool_evaluation():
    """Example 2: Tool Use Evaluation - Check tool selection and parameters"""
    display_section("2. Tool Use Quality Metrics (TUM)")
    
    # Create trace with tool calls
    trace = AgentTrace(task="Read the config file and update the database URL")
    
    trace.add_step(AgentStep(
        index=0,
        thought="I need to read the config file",
        tool_call=ToolCall(
            name="web_search",  # Wrong tool!
            parameters={"query": "config.yaml contents"},
            error="Cannot search local files"
        )
    ))
    trace.add_step(AgentStep(
        index=1,
        thought="Let me try reading the file directly",
        tool_call=ToolCall(
            name="read_file",
            parameters={"path": "/etc/nonexistent/config.yaml"},  # Hallucinated path
            error="File not found"
        )
    ))
    trace.add_step(AgentStep(
        index=2,
        thought="Found the right path",
        tool_call=ToolCall(
            name="read_file",
            parameters={"path": "./config.yaml"},
            result="database_url: localhost:5432"
        )
    ))
    
    console.print(Panel(
        f"Task: {trace.task}\n"
        f"Tool calls: {len(trace.tool_calls)}\n"
        f"[dim]First call uses wrong tool, second has bad path[/dim]",
        title="🔧 Tool Calls Summary"
    ))
    
    # Evaluate tools
    evaluator = ToolEvaluator(
        available_tools=["read_file", "write_file", "run_command", "web_search"]
    )
    
    with console.status("[bold green]Evaluating tool usage..."):
        efficiency, results = evaluator.evaluate_tool_chain(trace)
    
    # Display results
    table = Table(title="Tool Evaluation Results", box=box.ROUNDED)
    table.add_column("Tool", style="cyan")
    table.add_column("Signal", style="magenta")
    table.add_column("Confidence")
    table.add_column("Reason", style="dim")
    
    for r in results:
        signal_color = "green" if r.signal == ToolSignal.CORRECT else "red"
        table.add_row(
            r.tool_name,
            f"[{signal_color}]{r.signal.value}[/{signal_color}]",
            f"{r.confidence:.2f}",
            r.reason[:40] + "..." if len(r.reason) > 40 else r.reason
        )
    
    console.print(table)
    console.print(f"\n[bold]Chain Efficiency:[/bold] {efficiency:.2f}")


def example_self_correction():
    """Example 3: Self-Correction Detection - Track error recovery"""
    display_section("3. Self-Correction Detection (SCD)")
    
    # Create trace with self-correction pattern
    trace = AgentTrace(task="Calculate the sum of numbers 1-100")
    
    trace.add_step(AgentStep(
        index=0,
        thought="I'll calculate the sum using a loop",
        action="Writing Python code",
        observation="sum = 0; for i in range(100): sum += i"
    ))
    trace.add_step(AgentStep(
        index=1,
        thought="Wait, that gives 4950, not 5050. I made an off-by-one error!",
        action="Fixing the code",
        observation="The range should be range(1, 101)"
    ))
    trace.add_step(AgentStep(
        index=2,
        thought="Let me verify: sum = 0; for i in range(1, 101): sum += i",
        action="Running corrected code",
        observation="Result: 5050 ✓"
    ))
    trace.success = True
    
    console.print(Panel(
        f"Task: {trace.task}\n"
        f"Steps: {trace.step_count}\n"
        f"[dim]Agent makes error, notices, and corrects it[/dim]",
        title="🔧 Self-Correction Trace"
    ))
    
    # Detect self-correction
    detector = SelfCorrectionDetector()
    
    with console.status("[bold green]Analyzing self-correction..."):
        result = detector.analyze(trace)
    
    # Display results
    color = "green" if result.correction_success else ("yellow" if result.correction_attempt else "red")
    console.print(Panel(
        f"Error Detected: {'✅' if result.detected_error else '❌'}\n"
        f"Correction Attempted: {'✅' if result.correction_attempt else '❌'}\n"
        f"Correction Successful: {'✅' if result.correction_success else '❌'}\n"
        f"Loops Before Fix: {result.loops_before_fix}\n\n"
        f"Self-Awareness Score: {result.self_awareness_score:.2f}\n"
        f"Correction Efficiency: {result.correction_efficiency:.2f}\n\n"
        f"[dim]{result.reason}[/dim]",
        title="🔄 Self-Correction Analysis",
        border_style=color
    ))


def example_benchmarking():
    """Example 4: Agent Benchmarking - Compare multiple agents"""
    display_section("4. Comparative Agent Benchmarking (CAB)")
    
    # Create mock agents with different behaviors
    def good_agent_fn(task: str) -> AgentTrace:
        return create_mock_trace(
            task=task,
            steps=[
                {"thought": "Analyzing task requirements", "action": "Planning approach"},
                {"thought": "Executing plan", "tool_call": {"name": "read_file", "parameters": {"path": "./data.json"}, "result": "data loaded"}},
                {"thought": "Processing data", "action": "Running transformation"},
                {"thought": "Task complete", "observation": "Success!"},
            ],
            success=True,
            agent_name="GoodAgent",
            total_cost=0.002,
        )
    
    def mediocre_agent_fn(task: str) -> AgentTrace:
        return create_mock_trace(
            task=task,
            steps=[
                {"thought": "Let me try something", "action": "Random approach"},
                {"thought": "That didn't work", "observation": "Error occurred"},
                {"thought": "Trying again", "action": "Same approach again"},
                {"thought": "Still not working", "observation": "Another error"},
                {"thought": "Maybe this will work", "action": "Different approach"},
                {"thought": "Finally!", "observation": "Got a result"},
            ],
            success=True,
            agent_name="MediocreAgent",
            total_cost=0.005,
        )
    
    def bad_agent_fn(task: str) -> AgentTrace:
        return create_mock_trace(
            task=task,
            steps=[
                {"thought": "I'll solve a different problem", "action": "Doing unrelated work"},
                {"thought": "This is interesting", "action": "Going off on tangent"},
                {"thought": "What was the task again?", "observation": "Lost track"},
            ],
            success=False,
            agent_name="BadAgent",
            total_cost=0.003,
        )
    
    good_agent = SimpleAgent("GoodAgent", good_agent_fn)
    mediocre_agent = SimpleAgent("MediocreAgent", mediocre_agent_fn)
    bad_agent = SimpleAgent("BadAgent", bad_agent_fn)
    
    # Create benchmark task
    task = BenchmarkTask(
        name="data_processing",
        description="Process the JSON data file and extract user statistics",
        category="data",
        difficulty="medium"
    )
    
    console.print(Panel(
        f"Task: {task.description}\n"
        f"Category: {task.category}\n"
        f"Difficulty: {task.difficulty}",
        title="📋 Benchmark Task"
    ))
    
    # Run comparison
    benchmark = AgentBenchmark()
    
    with console.status("[bold green]Running benchmark comparison..."):
        comparison = benchmark.compare(
            agents=[good_agent, mediocre_agent, bad_agent],
            task=task
        )
    
    # Display results table
    table = Table(title="🏆 Benchmark Results", box=box.ROUNDED)
    table.add_column("Rank", style="cyan", justify="center")
    table.add_column("Agent", style="bold")
    table.add_column("Success", justify="center")
    table.add_column("Trajectory", justify="right")
    table.add_column("Tools", justify="right")
    table.add_column("Self-Correct", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Steps", justify="center")
    
    medals = ["🥇", "🥈", "🥉"]
    for i, run in enumerate(comparison.runs):
        r = run.result
        medal = medals[comparison.rankings.index(run.agent_name)] if run.agent_name in comparison.rankings[:3] else f"#{comparison.rankings.index(run.agent_name) + 1}"
        table.add_row(
            medal,
            run.agent_name,
            "✅" if r.success else "❌",
            f"{r.trajectory_score:.2f}",
            f"{r.tool_accuracy:.2f}",
            f"{r.self_correction_score:.2f}",
            f"${r.total_cost:.4f}",
            str(r.step_count)
        )
    
    console.print(table)
    
    # Winner announcement
    console.print(Panel(
        f"[bold green]🏆 Winner: {comparison.winner}[/bold green]\n\n"
        f"Rankings: {' > '.join(comparison.rankings)}",
        title="Final Results",
        border_style="green"
    ))
    
    # Show leaderboard
    console.print("\n[bold]📊 Leaderboard:[/bold]")
    leaderboard = benchmark.get_leaderboard()
    for entry in leaderboard:
        console.print(f"  {entry.agent_name}: Win Rate={entry.win_rate:.0%}, Success={entry.success_rate:.0%}")


def main():
    console.print(Panel.fit(
        "[bold cyan]ReAsk Agent Evaluation[/bold cyan]\n"
        "[dim]Comprehensive agent trace analysis and benchmarking[/dim]",
        border_style="cyan"
    ))
    
    # Run all examples
    example_trajectory_analysis()
    example_tool_evaluation()
    example_self_correction()
    example_benchmarking()
    
    console.print()
    console.rule("[bold green]All Examples Complete[/bold green]")
    console.print()


if __name__ == "__main__":
    main()

