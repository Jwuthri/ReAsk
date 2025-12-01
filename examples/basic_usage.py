"""Basic usage examples for ReAsk with Rich output"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich import box
from rich.columns import Columns
from rich.text import Text

from reask import ReAskDetector, Message, DetectionType

console = Console()


def get_status_emoji(is_bad: bool) -> str:
    return "❌" if is_bad else "✅"


def get_detection_color(detection_type: DetectionType) -> str:
    colors = {
        DetectionType.CCM: "cyan",
        DetectionType.RDM: "magenta", 
        DetectionType.LLM_JUDGE: "yellow",
        DetectionType.NONE: "green"
    }
    return colors.get(detection_type, "white")


def display_result(result, title: str = "Result"):
    """Display evaluation result in a nice panel"""
    status = get_status_emoji(result.is_bad)
    color = "red" if result.is_bad else "green"
    detection_color = get_detection_color(result.detection_type)
    
    content = Text()
    content.append(f"{status} ", style="bold")
    content.append("BAD" if result.is_bad else "OK", style=f"bold {color}")
    content.append(" | Detection: ", style="dim")
    content.append(result.detection_type.value.upper(), style=f"bold {detection_color}")
    content.append(f" | Confidence: ", style="dim")
    content.append(f"{result.confidence:.0%}", style="bold")
    content.append(f"\n\n💬 ", style="dim")
    content.append(result.reason or "No reason provided", style="italic")
    
    console.print(Panel(content, title=f"[bold]{title}[/bold]", border_style=color))


def display_conversation(messages: list[Message]) -> Tree:
    """Display conversation as a tree"""
    tree = Tree("💬 [bold]Conversation[/bold]")
    
    for i, msg in enumerate(messages):
        if msg.role.value == "user":
            branch = tree.add(f"[cyan]👤 User[/cyan]")
        else:
            branch = tree.add(f"[green]🤖 Assistant[/green]")
        branch.add(f"[dim]{msg.content}[/dim]")
    
    return tree


def run_example(detector, title: str, user_msg: str, assistant_msg: str, follow_up_msg: str):
    """Run a single example with nice formatting"""
    console.print()
    console.rule(f"[bold blue]{title}[/bold blue]")
    console.print()
    
    # Show the conversation
    table = Table(show_header=True, header_style="bold", box=box.ROUNDED)
    table.add_column("Role", style="cyan", width=12)
    table.add_column("Message", style="white")
    
    table.add_row("👤 User", user_msg)
    table.add_row("🤖 Assistant", assistant_msg)
    table.add_row("👤 Follow-up", follow_up_msg)
    
    console.print(table)
    console.print()
    
    # Evaluate with spinner
    with console.status("[bold green]Evaluating response...", spinner="dots"):
        result = detector.evaluate_response(
            user_message=Message.user(user_msg),
            assistant_response=Message.assistant(assistant_msg),
            follow_up=Message.user(follow_up_msg)
        )
    
    display_result(result, title)
    return result


def main():
    # Config display
    config_table = Table(title="⚙️  Configuration", box=box.SIMPLE)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("CCM Model", "gpt-5-nano")
    config_table.add_row("RDM Model", "gpt-5-nano")
    config_table.add_row("Judge Model", "gpt-5-mini")
    config_table.add_row("Similarity Threshold", "0.75")
    
    console.print(config_table)
    console.print()
    
    # Initialize detector with progress
    with console.status("[bold green]Initializing ReAsk detector...", spinner="dots"):
        detector = ReAskDetector(
            ccm_model="gpt-5-nano",
            rdm_model="gpt-5-nano",
            judge_model="gpt-5-mini",
            similarity_threshold=0.75,
            use_llm_confirmation=True,
            use_llm_judge_fallback=True
        )

    console.print("[green]✓[/green] Detector initialized\n")
    
    # Run examples
    results = []
    
    results.append(run_example(
        detector,
        "Example 1: Explicit Correction (RDM)",
        "Can you write me a Python function to sort a list?",
        "Here's a JavaScript sort: arr.sort()",
        "I asked you for Python, not JavaScript!"
    ))
    
    results.append(run_example(
        detector,
        "Example 2: Re-asked Question (CCM)",
        "How do I reverse a string in Python?",
        "You can use loops to iterate.",
        "Can you show me how to reverse a string?"
    ))
    
    results.append(run_example(
        detector,
        "Example 3: Good Response",
        "How do I reverse a string in Python?",
        "Use slicing: reversed_str = original[::-1]",
        "Great! Now how do I check if it's a palindrome?"
    ))
    
    # Full conversation evaluation
    console.print()
    console.rule("[bold blue]Example 4: Full Conversation Evaluation[/bold blue]")
    console.print()
    
    conversation = [
        Message.user("What's the capital of France?"),
        Message.assistant("The capital of France is Berlin."),
        Message.user("That's wrong. What is the capital of France?"),
        Message.assistant("I apologize, the capital of France is Paris."),
        Message.user("Thanks! What about Germany?"),
        Message.assistant("The capital of Germany is Berlin."),
        Message.user("Perfect!")
    ]
    
    # Display conversation tree
    tree = display_conversation(conversation)
    console.print(tree)
    console.print()
    
    # Evaluate with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[green]Evaluating conversation...", total=None)
        conv_results = detector.evaluate_conversation(conversation)
    
    # Results table
    results_table = Table(title="📊 Evaluation Results", box=box.ROUNDED)
    results_table.add_column("Index", style="cyan", justify="center")
    results_table.add_column("Status", justify="center")
    results_table.add_column("Detection", style="magenta")
    results_table.add_column("Confidence", justify="right")
    results_table.add_column("Reason", style="dim")
    
    for idx, result in conv_results:
        status = get_status_emoji(result.is_bad)
        color = "red" if result.is_bad else "green"
        results_table.add_row(
            str(idx),
            f"[{color}]{status}[/{color}]",
            result.detection_type.value.upper(),
            f"{result.confidence:.0%}",
            result.reason[:50] + "..." if result.reason and len(result.reason) > 50 else (result.reason or "-")
        )
    
    console.print(results_table)
    
    # Summary
    console.print()
    bad_count = sum(1 for _, r in conv_results if r.is_bad)
    good_count = len(conv_results) - bad_count
    
    summary = Columns([
        Panel(f"[bold green]{good_count}[/bold green]", title="✅ Good", border_style="green"),
        Panel(f"[bold red]{bad_count}[/bold red]", title="❌ Bad", border_style="red"),
        Panel(f"[bold cyan]{len(conv_results)}[/bold cyan]", title="📊 Total", border_style="cyan"),
    ])
    console.print(summary)
    console.print()


if __name__ == "__main__":
    main()
