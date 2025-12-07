"""
Customer Support Ticket Evaluation Example

Demonstrates multi-agent customer support evaluation matching the web dataset format.
Evaluates multiple conversations with multiple agents per turn.

Features:
- Interactive analysis type selection
- Parallel execution with configurable max workers
- Async/concurrent processing for speed

Structure:
- Dataset with multiple conversations (tickets)
- Each conversation has multiple turns
- Each turn has multiple agent_interactions
- Each agent_interaction has agent_steps (thoughts + tool_calls)

Evaluates:
- Conversation quality (CCM/RDM/Hallucination)  
- Agent trajectory (efficiency, loops)
- Tool usage (accuracy, parameter validation)
- Self-correction (error recovery)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, Confirm
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()


# ============================================
# DATA MODELS
# ============================================

@dataclass
class AgentDef:
    """Agent definition"""
    id: str
    name: str
    role: str
    tools_available: List[Dict[str, Any]] = field(default_factory=list)


@dataclass  
class AgentStep:
    """Single step in agent execution"""
    thought: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None


@dataclass
class AgentInteraction:
    """One agent's interaction in a turn"""
    agent_id: str
    agent_steps: List[AgentStep]
    agent_response: str
    latency_ms: int = 0


@dataclass
class Turn:
    """One conversation turn with user message and agent interactions"""
    turn_index: int
    user_message: str
    agent_interactions: List[AgentInteraction]


@dataclass
class Conversation:
    """Complete conversation (ticket)"""
    initial_task: str
    agents: List[AgentDef]
    turns: List[Turn]
    total_cost: float = 0.0
    total_latency_ms: int = 0


@dataclass
class Dataset:
    """Full dataset of conversations"""
    name: str
    task: str
    conversations: List[Conversation]


# ============================================
# SUPPORT AGENTS
# ============================================

SUPPORT_AGENTS = [
    AgentDef(
        id="triage_agent",
        name="TriageAgent",
        role="triage",
        tools_available=[
            {"name": "transfer_to_agent", "parameters_schema": {"agent_id": "string"}},
        ]
    ),
    AgentDef(
        id="refund_specialist",
        name="RefundSpecialist", 
        role="refund",
        tools_available=[
            {"name": "check_eligibility", "parameters_schema": {"order_id": "string"}},
            {"name": "process_refund", "parameters_schema": {"order_id": "string", "amount": "number"}},
            {"name": "generate_return_label", "parameters_schema": {"order_id": "string"}},
            {"name": "process_replacement", "parameters_schema": {"order_id": "string"}},
        ]
    ),
    AgentDef(
        id="tech_support",
        name="TechSupport",
        role="technical",
        tools_available=[
            {"name": "check_system_status", "parameters_schema": {"service": "string", "email": "string"}},
            {"name": "unlock_account", "parameters_schema": {"email": "string"}},
        ]
    ),
    AgentDef(
        id="order_tracker",
        name="OrderBot",
        role="orders",
        tools_available=[
            {"name": "get_order_status", "parameters_schema": {"order_id": "string"}},
            {"name": "transfer_to_agent", "parameters_schema": {"agent_id": "string"}},
        ]
    ),
]


# ============================================
# DATASET CREATION
# ============================================

def create_dataset() -> Dataset:
    """Create customer support ticket dataset"""
    
    conversations = [
        # Ticket 1: Product Return - GOOD
        Conversation(
            initial_task="Process return request - defective product",
            agents=SUPPORT_AGENTS,
            turns=[
                Turn(
                    turn_index=0,
                    user_message="I received a defective laptop yesterday. The screen flickers constantly and it won't charge.",
                    agent_interactions=[
                        AgentInteraction(
                            agent_id="triage_agent",
                            agent_steps=[
                                AgentStep(thought="Defective product issue. Routing to refund specialist for return processing."),
                                AgentStep(tool_call={"tool_name": "transfer_to_agent", "parameters": {"agent_id": "refund_specialist"}}),
                            ],
                            agent_response="I'm sorry to hear about the defective laptop. Let me connect you with our returns specialist who can help you right away.",
                            latency_ms=250
                        )
                    ]
                ),
                Turn(
                    turn_index=1,
                    user_message="Thank you. Order number is ORD-78945.",
                    agent_interactions=[
                        AgentInteraction(
                            agent_id="refund_specialist",
                            agent_steps=[
                                AgentStep(thought="Check return eligibility for defective product."),
                                AgentStep(tool_call={"tool_name": "check_eligibility", "parameters": {"order_id": "ORD-78945"}, "result": "Eligible: Defective product policy - full refund or replacement available"}),
                                AgentStep(tool_call={"tool_name": "process_refund", "parameters": {"order_id": "ORD-78945", "amount": 1299.99}, "result": "Success: Refund initiated - Transaction #REF-887766"}),
                            ],
                            agent_response="I've processed your full refund of $1,299.99 for the defective laptop. You'll see it in your account within 3-5 business days. Would you like a prepaid return label?",
                            latency_ms=1100
                        )
                    ]
                ),
                Turn(
                    turn_index=2,
                    user_message="Yes please, and can I get a replacement instead?",
                    agent_interactions=[
                        AgentInteraction(
                            agent_id="refund_specialist",
                            agent_steps=[
                                AgentStep(tool_call={"tool_name": "generate_return_label", "parameters": {"order_id": "ORD-78945"}, "result": "Label sent to email"}),
                                AgentStep(tool_call={"tool_name": "process_replacement", "parameters": {"order_id": "ORD-78945"}, "result": "Replacement order created: ORD-78946"}),
                            ],
                            agent_response="Perfect! I've emailed you a prepaid return label and created a replacement order (ORD-78946). The replacement will ship as soon as we receive your return.",
                            latency_ms=850
                        )
                    ]
                ),
            ],
            total_cost=0.009,
            total_latency_ms=2200
        ),
        
        # Ticket 2: Login Issues - OK (agent makes error but recovers)
        Conversation(
            initial_task="Troubleshoot account access problem",
            agents=SUPPORT_AGENTS,
            turns=[
                Turn(
                    turn_index=0,
                    user_message="I can't log into my account. It says 'invalid credentials' but I'm using the right password.",
                    agent_interactions=[
                        AgentInteraction(
                            agent_id="triage_agent",
                            agent_steps=[
                                AgentStep(thought="Login issue. Route to tech support."),
                                AgentStep(tool_call={"tool_name": "transfer_to_agent", "parameters": {"agent_id": "tech_support"}}),
                            ],
                            agent_response="I'll connect you with tech support to help resolve your login issue.",
                            latency_ms=180
                        )
                    ]
                ),
                Turn(
                    turn_index=1,
                    user_message="My email is john.doe@email.com",
                    agent_interactions=[
                        AgentInteraction(
                            agent_id="tech_support",
                            agent_steps=[
                                AgentStep(thought="Need to check account status."),
                                AgentStep(tool_call={"tool_name": "check_system_status", "parameters": {"service": "user_account"}, "error": "Missing required parameter: email"}),
                            ],
                            agent_response="Let me check your account status...",
                            latency_ms=200
                        )
                    ]
                ),
                Turn(
                    turn_index=2,
                    user_message="Any update?",
                    agent_interactions=[
                        AgentInteraction(
                            agent_id="tech_support",
                            agent_steps=[
                                AgentStep(thought="Oops, I forgot the email parameter. Let me retry with the correct parameters."),
                                AgentStep(tool_call={"tool_name": "check_system_status", "parameters": {"service": "user_account", "email": "john.doe@email.com"}, "result": "Account status: Locked due to 5 failed login attempts"}),
                                AgentStep(tool_call={"tool_name": "unlock_account", "parameters": {"email": "john.doe@email.com"}, "result": "Account unlocked successfully"}),
                            ],
                            agent_response="I see the issue - your account was temporarily locked after 5 failed login attempts. I've unlocked it for you. You can try logging in again now.",
                            latency_ms=900
                        )
                    ]
                ),
            ],
            total_cost=0.006,
            total_latency_ms=1280
        ),
        
        # Ticket 3: Shipping Delay - BAD (circular pattern + wrong info)
        Conversation(
            initial_task="Address shipping delay and customer frustration",
            agents=SUPPORT_AGENTS,
            turns=[
                Turn(
                    turn_index=0,
                    user_message="My order was supposed to arrive 3 days ago! This is unacceptable. Order #ORD-33221.",
                    agent_interactions=[
                        AgentInteraction(
                            agent_id="triage_agent",
                            agent_steps=[
                                AgentStep(thought="Frustrated customer with shipping delay. Route to order tracking."),
                                AgentStep(tool_call={"tool_name": "transfer_to_agent", "parameters": {"agent_id": "order_tracker"}}),
                            ],
                            agent_response="I sincerely apologize for the delay. Let me check your order status right away.",
                            latency_ms=220
                        )
                    ]
                ),
                Turn(
                    turn_index=1,
                    user_message="I need this for a gift tomorrow!",
                    agent_interactions=[
                        AgentInteraction(
                            agent_id="order_tracker",
                            agent_steps=[
                                AgentStep(tool_call={"tool_name": "get_order_status", "parameters": {"order_id": "ORD-33221"}, "result": "Status: In transit. Delayed due to weather. Current location: Distribution center. ETA: 2 days"}),
                            ],
                            agent_response="Your order will arrive tomorrow!",  # WRONG! ETA is 2 days
                            latency_ms=650
                        )
                    ]
                ),
                Turn(
                    turn_index=2,
                    user_message="You said tomorrow, right? I really need it.",
                    agent_interactions=[
                        AgentInteraction(
                            agent_id="order_tracker",
                            agent_steps=[
                                AgentStep(thought="Let me check again..."),
                                AgentStep(tool_call={"tool_name": "get_order_status", "parameters": {"order_id": "ORD-33221"}, "result": "Status: In transit. Delayed due to weather. Current location: Distribution center. ETA: 2 days"}),
                            ],
                            agent_response="Yes, it should arrive soon. Let me check one more time...",
                            latency_ms=650
                        )
                    ]
                ),
                Turn(
                    turn_index=3,
                    user_message="Can you just refund it? This is taking too long.",
                    agent_interactions=[
                        AgentInteraction(
                            agent_id="order_tracker",
                            agent_steps=[
                                AgentStep(thought="Customer wants refund. Need to transfer."),
                                AgentStep(tool_call={"tool_name": "transfer_to_agent", "parameters": {"agent_id": "refund_specialist"}}),
                            ],
                            agent_response="Let me transfer you to billing for a refund.",
                            latency_ms=200
                        )
                    ]
                ),
            ],
            total_cost=0.008,
            total_latency_ms=1720
        ),
    ]
    
    return Dataset(
        name="Customer Support Ticket Evaluation",
        task="Analyze support agent performance on common ticket types",
        conversations=conversations
    )


# ============================================
# ANALYSIS FUNCTIONS (can run in parallel)
# ============================================

def run_conversation_analysis(conv: Conversation) -> dict:
    """Run conversation analysis (CCM/RDM/Hallucination)"""
    from reask import ReAskDetector, Message, Role
    
    detector = ReAskDetector(
        ccm_model="gpt-4o-nano",
        rdm_model="gpt-4o-nano", 
        judge_model="gpt-4o-nano",
        similarity_threshold=0.66,
        use_llm_confirmation=True,
        use_llm_judge_fallback=True
    )
    
    messages = []
    for turn in conv.turns:
        messages.append(Message(role=Role.USER, content=turn.user_message))
        combined_response = " ".join([ia.agent_response for ia in turn.agent_interactions])
        messages.append(Message(role=Role.ASSISTANT, content=combined_response))
    
    conv_results = detector.evaluate_conversation(messages)
    bad_turns = [(idx, r) for idx, r in conv_results if r.is_bad]
    
    return {
        'results': conv_results,
        'bad_count': len(bad_turns),
        'bad_turns': bad_turns
    }


def build_trace(conv: Conversation, conv_num: int):
    """Build AgentTrace from conversation"""
    from reask import AgentTrace, AgentStep as ReaskStep, ToolCall
    
    trace = AgentTrace(
        task=conv.initial_task,
        metadata={"conversation_id": conv_num}
    )
    
    step_idx = 0
    for turn in conv.turns:
        for interaction in turn.agent_interactions:
            for agent_step in interaction.agent_steps:
                reask_step = ReaskStep(
                    index=step_idx,
                    thought=agent_step.thought,
                    action=None,
                    observation=None
                )
                
                if agent_step.tool_call:
                    tc = agent_step.tool_call
                    reask_step.tool_call = ToolCall(
                        name=tc.get("tool_name", "unknown"),
                        parameters=tc.get("parameters", {}),
                        result=tc.get("result"),
                        error=tc.get("error"),
                        duration_ms=interaction.latency_ms
                    )
                
                trace.steps.append(reask_step)
                step_idx += 1
    
    trace.success = True
    trace.total_cost = conv.total_cost
    trace.total_duration_ms = conv.total_latency_ms
    
    return trace


def run_trajectory_analysis(trace) -> dict:
    """Run trajectory analysis"""
    from reask import TrajectoryAnalyzer
    
    analyzer = TrajectoryAnalyzer()
    result = analyzer.analyze(trace)
    
    return {
        'result': result,
        'signal': result.signal.value,
        'efficiency': result.efficiency_score,
        'circular_count': result.circular_count,
        'regression_count': result.regression_count
    }


def run_tool_analysis(trace) -> dict:
    """Run tool usage analysis"""
    from reask import ToolEvaluator
    
    evaluator = ToolEvaluator(
        available_tools=[
            "transfer_to_agent", "check_eligibility", "process_refund",
            "generate_return_label", "process_replacement", "check_system_status",
            "unlock_account", "get_order_status"
        ]
    )
    efficiency, results = evaluator.evaluate_tool_chain(trace)
    
    return {
        'efficiency': efficiency,
        'results': results
    }


def run_self_correction_analysis(trace) -> dict:
    """Run self-correction detection"""
    from reask import SelfCorrectionDetector
    
    detector = SelfCorrectionDetector()
    result = detector.analyze(trace)
    
    return {
        'result': result,
        'detected_error': result.detected_error,
        'correction_attempt': result.correction_attempt,
        'correction_success': result.correction_success,
        'awareness_score': result.self_awareness_score
    }


# ============================================
# PARALLEL EVALUATION
# ============================================

def evaluate_conversation_parallel(
    conv: Conversation, 
    conv_num: int, 
    analysis_types: Set[str],
    max_workers: int = 4
) -> dict:
    """Evaluate a conversation with parallel analysis"""
    
    results = {
        'conversation_bad_count': 0,
        'conversation': [],
        'trajectory': None,
        'tool_efficiency': 1.0,
        'tool_results': [],
        'self_correction': None
    }
    
    # Build trace once (needed for trajectory, tools, self-correction)
    trace = None
    if any(t in analysis_types for t in ['trajectory', 'tools', 'self_correction']):
        trace = build_trace(conv, conv_num)
    
    # Prepare tasks
    tasks = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        if 'conversation' in analysis_types:
            futures['conversation'] = executor.submit(run_conversation_analysis, conv)
        
        if 'trajectory' in analysis_types and trace:
            futures['trajectory'] = executor.submit(run_trajectory_analysis, trace)
        
        if 'tools' in analysis_types and trace:
            futures['tools'] = executor.submit(run_tool_analysis, trace)
        
        if 'self_correction' in analysis_types and trace:
            futures['self_correction'] = executor.submit(run_self_correction_analysis, trace)
        
        # Collect results as they complete
        for name, future in futures.items():
            try:
                result = future.result()
                tasks[name] = result
            except Exception as e:
                console.print(f"[red]Error in {name} analysis: {e}[/red]")
                tasks[name] = None
    
    # Map results
    if 'conversation' in tasks and tasks['conversation']:
        results['conversation'] = tasks['conversation']['results']
        results['conversation_bad_count'] = tasks['conversation']['bad_count']
    
    if 'trajectory' in tasks and tasks['trajectory']:
        results['trajectory'] = tasks['trajectory']['result']
    
    if 'tools' in tasks and tasks['tools']:
        results['tool_efficiency'] = tasks['tools']['efficiency']
        results['tool_results'] = tasks['tools']['results']
    
    if 'self_correction' in tasks and tasks['self_correction']:
        results['self_correction'] = tasks['self_correction']['result']
    
    return results


def display_results(conv: Conversation, conv_num: int, results: dict, analysis_types: Set[str]):
    """Display results for a conversation"""
    console.print()
    console.rule(f"[bold cyan]Ticket #{conv_num}: {conv.initial_task}[/bold cyan]")
    console.print()
    
    # Summary
    agent_ids = set()
    total_steps = 0
    for turn in conv.turns:
        for interaction in turn.agent_interactions:
            agent_ids.add(interaction.agent_id)
            total_steps += len(interaction.agent_steps)
    
    console.print(Panel(
        f"Turns: {len(conv.turns)}\n"
        f"Agents: {len(agent_ids)} ({', '.join(sorted(agent_ids))})\n"
        f"Total Steps: {total_steps}\n"
        f"Cost: ${conv.total_cost:.4f}\n"
        f"Duration: {conv.total_latency_ms}ms",
        title="📋 Conversation Summary",
        border_style="cyan"
    ))
    
    # Conversation results
    if 'conversation' in analysis_types and results.get('conversation'):
        console.print("\n[bold]💬 Conversation Quality[/bold]")
        bad_count = results['conversation_bad_count']
        total = len(results['conversation'])
        
        if bad_count > 0:
            console.print(f"[red]⚠️ Found {bad_count} bad turns out of {total}[/red]")
        else:
            console.print(f"[green]✅ All {total} conversation turns look good![/green]")
    
    # Trajectory results
    if 'trajectory' in analysis_types and results.get('trajectory'):
        console.print("\n[bold]🔄 Trajectory Analysis[/bold]")
        traj = results['trajectory']
        color = "red" if traj.signal.value in ["stall", "circular"] else "green"
        console.print(f"  Signal: [{color}]{traj.signal.value.upper()}[/{color}]")
        console.print(f"  Efficiency: {traj.efficiency_score:.2%}")
        console.print(f"  Circular: {traj.circular_count} | Regressions: {traj.regression_count}")
    
    # Tool results
    if 'tools' in analysis_types:
        console.print("\n[bold]🔧 Tool Usage[/bold]")
        console.print(f"  Efficiency: {results['tool_efficiency']:.2%}")
        if results.get('tool_results'):
            errors = [r for r in results['tool_results'] if r.signal.value != 'correct']
            if errors:
                console.print(f"  [yellow]Issues: {len(errors)} tool calls with problems[/yellow]")
    
    # Self-correction results
    if 'self_correction' in analysis_types and results.get('self_correction'):
        console.print("\n[bold]🔁 Self-Correction[/bold]")
        sc = results['self_correction']
        console.print(f"  Error Detected: {'✅' if sc.detected_error else '❌'}")
        console.print(f"  Correction Success: {'✅' if sc.correction_success else '❌'}")
        console.print(f"  Awareness: {sc.self_awareness_score:.2%}")


# ============================================
# MAIN
# ============================================

def prompt_analysis_types() -> Set[str]:
    """Ask user which analysis types to run"""
    console.print()
    console.print(Panel(
        "[bold]Select Analysis Types[/bold]\n\n"
        "🚀 [cyan]1[/cyan] - Full Analysis (all types) [RECOMMENDED]\n"
        "💬 [cyan]2[/cyan] - Conversation Detection (CCM/RDM/Hallucination)\n"
        "🔄 [cyan]3[/cyan] - Trajectory Analysis (efficiency, loops)\n"
        "🔧 [cyan]4[/cyan] - Tool Use Quality (accuracy, parameters)\n"
        "🔁 [cyan]5[/cyan] - Self-Correction (error recovery)\n"
        "🎯 [cyan]6[/cyan] - Agent Metrics Only (trajectory + tools + self-correction)\n"
        "📝 [cyan]7[/cyan] - Custom (select multiple)",
        title="Analysis Options",
        border_style="cyan"
    ))
    
    choice = Prompt.ask(
        "Enter your choice",
        choices=["1", "2", "3", "4", "5", "6", "7"],
        default="1"
    )
    
    if choice == "1":
        return {"conversation", "trajectory", "tools", "self_correction"}
    elif choice == "2":
        return {"conversation"}
    elif choice == "3":
        return {"trajectory"}
    elif choice == "4":
        return {"tools"}
    elif choice == "5":
        return {"self_correction"}
    elif choice == "6":
        return {"trajectory", "tools", "self_correction"}
    else:
        # Custom selection
        selected = set()
        console.print("\n[dim]Select each analysis type (y/n):[/dim]")
        
        if Confirm.ask("  💬 Conversation Detection?", default=True):
            selected.add("conversation")
        if Confirm.ask("  🔄 Trajectory Analysis?", default=True):
            selected.add("trajectory")
        if Confirm.ask("  🔧 Tool Use Quality?", default=True):
            selected.add("tools")
        if Confirm.ask("  🔁 Self-Correction?", default=True):
            selected.add("self_correction")
        
        if not selected:
            console.print("[yellow]No analysis selected, defaulting to full analysis[/yellow]")
            return {"conversation", "trajectory", "tools", "self_correction"}
        
        return selected


def prompt_max_workers() -> int:
    """Ask user for max workers"""
    console.print()
    max_workers = IntPrompt.ask(
        "Max parallel workers (1-8)",
        default=4
    )
    return max(1, min(8, max_workers))


def main():
    console.print(Panel.fit(
        "[bold cyan]Customer Support Ticket Evaluation[/bold cyan]\n"
        "[dim]Multi-agent, multi-turn conversation analysis with parallel execution[/dim]",
        border_style="cyan"
    ))
    
    # User prompts
    analysis_types = prompt_analysis_types()
    max_workers = prompt_max_workers()
    
    console.print()
    console.print(f"[bold]Selected analyses:[/bold] {', '.join(sorted(analysis_types))}")
    console.print(f"[bold]Max workers:[/bold] {max_workers}")
    console.print()
    
    # Create dataset
    dataset = create_dataset()
    
    console.print(f"[bold]Dataset:[/bold] {dataset.name}")
    console.print(f"[bold]Task:[/bold] {dataset.task}")
    console.print(f"[bold]Conversations:[/bold] {len(dataset.conversations)}\n")
    
    # Evaluate all conversations with progress
    all_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Evaluating conversations...", total=len(dataset.conversations))
        
        for i, conv in enumerate(dataset.conversations):
            results = evaluate_conversation_parallel(conv, i + 1, analysis_types, max_workers)
            all_results.append({
                'conversation': conv,
                'results': results
            })
            
            # Display results immediately
            display_results(conv, i + 1, results, analysis_types)
            progress.advance(task)
    
    # Summary table
    console.print()
    console.rule("[bold green]Summary Report[/bold green]")
    console.print()
    
    summary_table = Table(title="📊 Overall Performance", box=box.ROUNDED)
    summary_table.add_column("Ticket", style="cyan")
    summary_table.add_column("Task", style="magenta")
    
    if 'conversation' in analysis_types:
        summary_table.add_column("Conv", justify="center")
    if 'trajectory' in analysis_types:
        summary_table.add_column("Trajectory", justify="right")
    if 'tools' in analysis_types:
        summary_table.add_column("Tools", justify="right")
    if 'self_correction' in analysis_types:
        summary_table.add_column("Self-Correct", justify="right")
    
    summary_table.add_column("Cost", justify="right")
    summary_table.add_column("Turns", justify="center")
    
    for i, item in enumerate(all_results):
        conv = item['conversation']
        results = item['results']
        
        row = [f"#{i+1}", conv.initial_task[:30] + "..."]
        
        if 'conversation' in analysis_types:
            conv_bad = results['conversation_bad_count']
            conv_total = len(results['conversation']) if results['conversation'] else 0
            conv_status = "✅" if conv_bad == 0 else f"⚠️ {conv_bad}/{conv_total}"
            row.append(conv_status)
        
        if 'trajectory' in analysis_types:
            traj_score = results['trajectory'].efficiency_score if results.get('trajectory') else 0
            row.append(f"{traj_score:.2%}")
        
        if 'tools' in analysis_types:
            row.append(f"{results['tool_efficiency']:.2%}")
        
        if 'self_correction' in analysis_types:
            sc_score = results['self_correction'].self_awareness_score if results.get('self_correction') else 0
            row.append(f"{sc_score:.2%}")
        
        row.extend([f"${conv.total_cost:.4f}", str(len(conv.turns))])
        
        summary_table.add_row(*row)
    
    console.print(summary_table)
    
    # Averages
    console.print()
    avg_parts = []
    
    if 'conversation' in analysis_types:
        avg_conv = sum(1 for r in all_results if r['results']['conversation_bad_count'] == 0) / len(all_results)
        avg_parts.append(f"[bold]Conversations Clean:[/bold] {avg_conv:.2%}")
    
    if 'trajectory' in analysis_types:
        avg_traj = sum(r['results']['trajectory'].efficiency_score for r in all_results if r['results'].get('trajectory')) / len(all_results)
        avg_parts.append(f"[bold]Avg Trajectory:[/bold] {avg_traj:.2%}")
    
    if 'tools' in analysis_types:
        avg_tools = sum(r['results']['tool_efficiency'] for r in all_results) / len(all_results)
        avg_parts.append(f"[bold]Avg Tool Efficiency:[/bold] {avg_tools:.2%}")
    
    if 'self_correction' in analysis_types:
        avg_sc = sum(r['results']['self_correction'].self_awareness_score for r in all_results if r['results'].get('self_correction')) / len(all_results)
        avg_parts.append(f"[bold]Avg Self-Correction:[/bold] {avg_sc:.2%}")
    
    avg_cost = sum(r['conversation'].total_cost for r in all_results) / len(all_results)
    avg_parts.append(f"[bold]Avg Cost:[/bold] ${avg_cost:.4f}")
    
    console.print(Panel(
        "\n".join(avg_parts),
        title="📈 Team Performance",
        border_style="green"
    ))
    
    console.print()
    console.rule("[bold green]Evaluation Complete[/bold green]")
    console.print()


if __name__ == "__main__":
    main()
