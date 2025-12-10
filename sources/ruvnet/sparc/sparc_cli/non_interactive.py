"""Non-interactive mode handler for SPARC CLI"""

from rich.console import Console
from rich.panel import Panel

console = Console()

def handle_non_interactive():
    """Handle non-interactive mode by displaying help and status"""
    console.print(Panel(
        "[bold green]SPARC CLI Server[/bold green]\n\n"
        "Running in non-interactive mode.\n"
        "Use SSH to connect for interactive access:\n"
        "  flyctl ssh console\n\n"
        "For logs:\n"
        "  flyctl logs\n\n"
        "Status: Ready",
        title="🚀 SPARC CLI",
        border_style="green"
    ))
    
    # Keep the process running
    try:
        import time
        while True:
            time.sleep(3600)  # Sleep for an hour
    except KeyboardInterrupt:
        console.print("\nShutting down...")
