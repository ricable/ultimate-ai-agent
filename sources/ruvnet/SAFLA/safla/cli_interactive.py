"""
Interactive CLI components for SAFLA Management System.

Provides rich interactive components including setup wizard, dashboard,
and configuration helpers with a terminal user interface.
"""

import asyncio
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, track
from rich.layout import Layout
from rich.live import Live
from rich.tree import Tree
from rich.text import Text
from rich.columns import Columns
from rich.align import Align

console = Console()


def _run_setup_wizard(manager):
    """Interactive setup wizard for SAFLA configuration."""
    console.print(Panel.fit(
        "[bold blue]SAFLA System Setup Wizard[/bold blue]\n"
        "This wizard will help you configure SAFLA for optimal performance\n"
        "in your environment.",
        title="Welcome"
    ))
    
    if not Confirm.ask("Continue with setup?", default=True):
        return
    
    config_updates = {}
    
    # System Configuration
    console.print("\n[bold cyan]System Configuration[/bold cyan]")
    
    # Debug mode
    debug_mode = Confirm.ask("Enable debug mode?", default=False)
    config_updates['SAFLA_DEBUG'] = str(debug_mode).lower()
    
    # Log level
    log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
    default_log = 'DEBUG' if debug_mode else 'INFO'
    log_level = Prompt.ask(
        "Log level",
        choices=log_levels,
        default=default_log
    )
    config_updates['SAFLA_LOG_LEVEL'] = log_level
    
    # Performance Configuration
    console.print("\n[bold cyan]Performance Configuration[/bold cyan]")
    
    # Worker threads
    cpu_count = os.cpu_count() or 4
    worker_threads = IntPrompt.ask(
        "Number of worker threads",
        default=min(cpu_count, 8)
    )
    config_updates['SAFLA_WORKER_THREADS'] = str(worker_threads)
    
    # Memory settings
    memory_pool = IntPrompt.ask(
        "Memory pool size (MB)",
        default=512
    )
    config_updates['SAFLA_MEMORY_POOL_SIZE'] = str(memory_pool)
    
    cache_size = IntPrompt.ask(
        "Cache size (MB)",
        default=256
    )
    config_updates['SAFLA_CACHE_SIZE'] = str(cache_size)
    
    # Enable optimizations
    enable_opt = Confirm.ask("Enable performance optimizations?", default=True)
    config_updates['SAFLA_ENABLE_OPTIMIZATIONS'] = str(enable_opt).lower()
    
    # Memory Configuration
    console.print("\n[bold cyan]Memory System Configuration[/bold cyan]")
    
    # Vector dimensions
    vector_dims = Prompt.ask(
        "Vector dimensions (comma-separated)",
        default="512,768,1024,1536"
    )
    config_updates['SAFLA_VECTOR_DIMENSIONS'] = vector_dims
    
    # Max memories
    max_memories = IntPrompt.ask(
        "Maximum number of memories",
        default=10000
    )
    config_updates['SAFLA_MAX_MEMORIES'] = str(max_memories)
    
    # Similarity threshold
    sim_threshold = FloatPrompt.ask(
        "Similarity threshold (0.0-1.0)",
        default=0.8
    )
    config_updates['SAFLA_SIMILARITY_THRESHOLD'] = str(sim_threshold)
    
    # Safety Configuration
    console.print("\n[bold cyan]Safety Configuration[/bold cyan]")
    
    # Memory limit
    memory_limit_gb = FloatPrompt.ask(
        "Memory limit (GB)",
        default=1.0
    )
    config_updates['SAFLA_MEMORY_LIMIT'] = str(int(memory_limit_gb * 1e9))
    
    # CPU limit
    cpu_limit = FloatPrompt.ask(
        "CPU usage limit (0.0-1.0)",
        default=0.9
    )
    config_updates['SAFLA_CPU_LIMIT'] = str(cpu_limit)
    
    # Enable rollback
    enable_rollback = Confirm.ask("Enable automatic rollback?", default=True)
    config_updates['SAFLA_ROLLBACK_ENABLED'] = str(enable_rollback).lower()
    
    # MCP Configuration
    console.print("\n[bold cyan]MCP Server Configuration[/bold cyan]")
    
    # Server settings
    host = Prompt.ask("Server host", default="localhost")
    config_updates['SAFLA_HOST'] = host
    
    port = IntPrompt.ask("Server port", default=8000)
    config_updates['SAFLA_PORT'] = str(port)
    
    # Connection settings
    mcp_timeout = IntPrompt.ask(
        "MCP operation timeout (seconds)",
        default=30
    )
    config_updates['SAFLA_MCP_TIMEOUT'] = str(mcp_timeout)
    
    connection_pool = IntPrompt.ask(
        "Connection pool size",
        default=10
    )
    config_updates['SAFLA_MCP_CONNECTION_POOL_SIZE'] = str(connection_pool)
    
    # Security Configuration
    console.print("\n[bold cyan]Security Configuration[/bold cyan]")
    
    # Rate limiting
    enable_rate_limit = Confirm.ask("Enable API rate limiting?", default=True)
    config_updates['SAFLA_ENABLE_RATE_LIMITING'] = str(enable_rate_limit).lower()
    
    if enable_rate_limit:
        rate_limit = IntPrompt.ask(
            "API rate limit (requests per minute)",
            default=1000
        )
        config_updates['SAFLA_API_RATE_LIMIT'] = str(rate_limit)
    
    # CORS
    enable_cors = Confirm.ask("Enable CORS?", default=True)
    config_updates['SAFLA_ENABLE_CORS'] = str(enable_cors).lower()
    
    # Monitoring Configuration
    console.print("\n[bold cyan]Monitoring Configuration[/bold cyan]")
    
    # Enable monitoring
    enable_monitoring = Confirm.ask("Enable system monitoring?", default=True)
    config_updates['SAFLA_ENABLE_MONITORING'] = str(enable_monitoring).lower()
    
    if enable_monitoring:
        # Metrics
        enable_metrics = Confirm.ask("Enable metrics collection?", default=True)
        config_updates['SAFLA_ENABLE_METRICS'] = str(enable_metrics).lower()
        
        if enable_metrics:
            metrics_interval = IntPrompt.ask(
                "Metrics export interval (seconds)",
                default=60
            )
            config_updates['SAFLA_METRICS_INTERVAL'] = str(metrics_interval)
        
        # Health checks
        enable_health = Confirm.ask("Enable health checks?", default=True)
        config_updates['SAFLA_ENABLE_HEALTH_CHECKS'] = str(enable_health).lower()
    
    # Review Configuration
    console.print("\n[bold cyan]Configuration Review[/bold cyan]")
    
    _display_config_summary(config_updates)
    
    if Confirm.ask("\nSave this configuration?", default=True):
        _save_wizard_config(config_updates)
        console.print("[green]✓ Configuration saved successfully![/green]")
        
        # Offer to start system
        if Confirm.ask("Start SAFLA system now?", default=True):
            _start_full_system(manager, daemon=False)
    else:
        console.print("[yellow]Configuration not saved[/yellow]")


def _display_config_summary(config_updates: Dict[str, str]):
    """Display configuration summary in a nice format."""
    table = Table(title="Configuration Summary")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Description", style="yellow")
    
    config_descriptions = {
        'SAFLA_DEBUG': 'Enable debug mode',
        'SAFLA_LOG_LEVEL': 'Logging verbosity level',
        'SAFLA_WORKER_THREADS': 'Number of processing threads',
        'SAFLA_MEMORY_POOL_SIZE': 'Memory pool size in MB',
        'SAFLA_CACHE_SIZE': 'Cache size in MB',
        'SAFLA_ENABLE_OPTIMIZATIONS': 'Performance optimizations',
        'SAFLA_VECTOR_DIMENSIONS': 'Supported vector dimensions',
        'SAFLA_MAX_MEMORIES': 'Maximum stored memories',
        'SAFLA_SIMILARITY_THRESHOLD': 'Memory similarity threshold',
        'SAFLA_MEMORY_LIMIT': 'System memory limit',
        'SAFLA_CPU_LIMIT': 'CPU usage limit',
        'SAFLA_ROLLBACK_ENABLED': 'Automatic rollback on errors',
        'SAFLA_HOST': 'Server host address',
        'SAFLA_PORT': 'Server port number',
        'SAFLA_MCP_TIMEOUT': 'MCP operation timeout',
        'SAFLA_MCP_CONNECTION_POOL_SIZE': 'Connection pool size',
        'SAFLA_ENABLE_RATE_LIMITING': 'API rate limiting',
        'SAFLA_API_RATE_LIMIT': 'Rate limit (req/min)',
        'SAFLA_ENABLE_CORS': 'Cross-origin requests',
        'SAFLA_ENABLE_MONITORING': 'System monitoring',
        'SAFLA_ENABLE_METRICS': 'Metrics collection',
        'SAFLA_METRICS_INTERVAL': 'Metrics interval (sec)',
        'SAFLA_ENABLE_HEALTH_CHECKS': 'Health check endpoints'
    }
    
    for key, value in config_updates.items():
        description = config_descriptions.get(key, 'Configuration setting')
        table.add_row(key, value, description)
    
    console.print(table)


def _save_wizard_config(config_updates: Dict[str, str]):
    """Save wizard configuration to .env file."""
    env_file = Path('.env')
    
    # Read existing .env file
    existing_config = {}
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    existing_config[key] = value
    
    # Update with new values
    existing_config.update(config_updates)
    
    # Write back to .env file
    with open(env_file, 'w') as f:
        f.write("# SAFLA Configuration - Generated by Setup Wizard\n")
        f.write(f"# Generated on: {datetime.now().isoformat()}\n\n")
        
        for key, value in existing_config.items():
            f.write(f"{key}={value}\n")


def _launch_interactive_dashboard(manager):
    """Launch rich interactive dashboard with live updates."""
    console.print("[bold blue]Launching SAFLA Interactive Dashboard...[/bold blue]")
    
    try:
        # Try to use Textual for advanced TUI
        from textual.app import App, ComposeResult
        from textual.containers import Container, Horizontal, Vertical
        from textual.widgets import Header, Footer, Static, DataTable, Log, Button
        from textual.reactive import reactive
        from textual import events
        
        class SaflaDashboard(App):
            """Advanced SAFLA Dashboard with Textual."""
            
            CSS = """
            .panel-title {
                text-style: bold;
                color: cyan;
                padding: 1;
            }
            
            .left-panel {
                width: 50%;
                padding: 1;
            }
            
            .right-panel {
                width: 50%; 
                padding: 1;
            }
            
            .status-good {
                color: green;
            }
            
            .status-warning {
                color: yellow;
            }
            
            .status-error {
                color: red;
            }
            """
            
            BINDINGS = [
                ("d", "toggle_dark", "Toggle dark mode"),
                ("q", "quit", "Quit"),
                ("r", "refresh", "Refresh"),
                ("s", "start_component", "Start component"),
                ("p", "stop_component", "Stop component"),
            ]
            
            def compose(self) -> ComposeResult:
                yield Header()
                
                with Horizontal():
                    with Vertical(classes="left-panel"):
                        yield Static("System Status", classes="panel-title")
                        yield DataTable(id="status-table")
                        
                        yield Static("Performance Metrics", classes="panel-title")
                        yield DataTable(id="metrics-table")
                        
                        with Horizontal():
                            yield Button("Start System", id="start-btn", variant="success")
                            yield Button("Stop System", id="stop-btn", variant="error")
                            yield Button("Restart", id="restart-btn", variant="warning")
                    
                    with Vertical(classes="right-panel"):
                        yield Static("System Logs", classes="panel-title")
                        yield Log(id="system-logs", auto_scroll=True)
                        
                        yield Static("Deployed Agents", classes="panel-title")
                        yield DataTable(id="agents-table")
                        
                        with Horizontal():
                            yield Button("Deploy Agent", id="deploy-btn", variant="primary")
                            yield Button("Scale Agent", id="scale-btn", variant="default")
                            yield Button("Remove Agent", id="remove-btn", variant="error")
                
                yield Footer()
            
            def on_mount(self) -> None:
                """Initialize dashboard on startup."""
                self.set_interval(5, self.refresh_data)
                self.refresh_data()
            
            def action_refresh(self) -> None:
                """Manual refresh action."""
                self.refresh_data()
            
            def refresh_data(self) -> None:
                """Refresh all dashboard data."""
                try:
                    # Update status table
                    status_table = self.query_one("#status-table", DataTable)
                    status_data = _get_system_status(manager, detailed=True)
                    self._populate_status_table(status_table, status_data)
                    
                    # Update metrics table
                    metrics_table = self.query_one("#metrics-table", DataTable)
                    metrics_data = _get_system_metrics(manager, detailed=True)
                    self._populate_metrics_table(metrics_table, metrics_data)
                    
                    # Update logs
                    log_widget = self.query_one("#system-logs", Log)
                    recent_logs = _get_recent_logs(manager, lines=10)
                    for log_line in recent_logs.split('\n')[-5:]:  # Only add new lines
                        if log_line.strip():
                            log_widget.write_line(log_line)
                    
                    # Update agents table
                    agents_table = self.query_one("#agents-table", DataTable)
                    agents_data = _get_deployed_agents(manager)
                    self._populate_agents_table(agents_table, agents_data)
                    
                except Exception as e:
                    self.log.error(f"Error refreshing dashboard: {e}")
            
            def _populate_status_table(self, table: DataTable, status_data: Dict[str, Any]):
                """Populate status table with data."""
                if not table.columns:
                    table.add_columns("Component", "Status", "Health", "Uptime", "CPU %", "Memory MB")
                
                table.clear()
                for component, info in status_data['components'].items():
                    table.add_row(
                        component.title(),
                        info.get('status', 'unknown'),
                        info.get('health', 'unknown'),
                        info.get('uptime', 'unknown'),
                        f"{info.get('cpu_usage', 0):.1f}",
                        f"{info.get('memory_usage', 0):.1f}"
                    )
            
            def _populate_metrics_table(self, table: DataTable, metrics_data: Dict[str, Any]):
                """Populate metrics table with data."""
                if not table.columns:
                    table.add_columns("Metric", "Value", "Unit")
                
                table.clear()
                system_metrics = metrics_data.get('system', {})
                for key, value in system_metrics.items():
                    if isinstance(value, (int, float)):
                        unit = "%" if "percent" in key else "GB" if "gb" in key else ""
                        table.add_row(key.replace('_', ' ').title(), f"{value:.1f}", unit)
            
            def _populate_agents_table(self, table: DataTable, agents_data: List[Dict[str, Any]]):
                """Populate agents table with data."""
                if not table.columns:
                    table.add_columns("Name", "Type", "Status", "Replicas", "Uptime")
                
                table.clear()
                for agent in agents_data:
                    table.add_row(
                        agent['name'],
                        agent['type'],
                        agent['status'],
                        str(agent['replicas']),
                        agent['uptime']
                    )
            
            # Button event handlers
            def on_button_pressed(self, event: Button.Pressed) -> None:
                """Handle button presses."""
                if event.button.id == "start-btn":
                    self._start_system()
                elif event.button.id == "stop-btn":
                    self._stop_system()
                elif event.button.id == "restart-btn":
                    self._restart_system()
                elif event.button.id == "deploy-btn":
                    self._deploy_agent()
                elif event.button.id == "scale-btn":
                    self._scale_agent()
                elif event.button.id == "remove-btn":
                    self._remove_agent()
            
            def _start_system(self):
                """Start SAFLA system."""
                self.log.info("Starting SAFLA system...")
                _start_full_system(manager, daemon=True)
            
            def _stop_system(self):
                """Stop SAFLA system."""
                self.log.info("Stopping SAFLA system...")
                _stop_full_system(manager, force=False)
            
            def _restart_system(self):
                """Restart SAFLA system."""
                self.log.info("Restarting SAFLA system...")
                _restart_full_system(manager)
            
            def _deploy_agent(self):
                """Deploy new agent."""
                self.log.info("Deploy agent functionality would open deployment dialog")
            
            def _scale_agent(self):
                """Scale existing agent."""
                self.log.info("Scale agent functionality would open scaling dialog")
            
            def _remove_agent(self):
                """Remove agent."""
                self.log.info("Remove agent functionality would show removal confirmation")
        
        # Run the Textual app
        app = SaflaDashboard()
        app.run()
        
    except ImportError:
        console.print("[yellow]Textual not available, falling back to simple dashboard[/yellow]")
        _simple_interactive_dashboard(manager)


def _simple_interactive_dashboard(manager):
    """Simple fallback interactive dashboard using Rich only."""
    console.print("[bold blue]SAFLA Simple Interactive Dashboard[/bold blue]")
    console.print("Press Ctrl+C to exit\n")
    
    def make_layout():
        """Create layout for simple dashboard."""
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split(
            Layout(name="status"),
            Layout(name="metrics")
        )
        
        layout["right"].split(
            Layout(name="logs"),
            Layout(name="agents")
        )
        
        return layout
    
    layout = make_layout()
    
    try:
        with Live(layout, refresh_per_second=0.2) as live:
            while True:
                # Update header
                layout["header"].update(Panel(
                    f"[bold blue]SAFLA Dashboard[/bold blue] - {datetime.now().strftime('%H:%M:%S')} | "
                    f"Press Ctrl+C to exit",
                    style="blue"
                ))
                
                # Update status
                try:
                    status_data = _get_system_status(manager, detailed=True)
                    status_content = _create_status_display(status_data)
                    layout["status"].update(Panel(status_content, title="System Status"))
                except Exception as e:
                    layout["status"].update(Panel(f"Error: {e}", title="System Status"))
                
                # Update metrics
                try:
                    metrics_data = _get_system_metrics(manager, detailed=False)
                    metrics_content = _create_metrics_display(metrics_data)
                    layout["metrics"].update(Panel(metrics_content, title="Metrics"))
                except Exception as e:
                    layout["metrics"].update(Panel(f"Error: {e}", title="Metrics"))
                
                # Update logs
                try:
                    logs_content = _get_recent_logs(manager, lines=8)
                    layout["logs"].update(Panel(logs_content, title="Recent Logs"))
                except Exception as e:
                    layout["logs"].update(Panel(f"Error: {e}", title="Recent Logs"))
                
                # Update agents
                try:
                    agents_data = _get_deployed_agents(manager)
                    agents_content = _create_agents_display(agents_data)
                    layout["agents"].update(Panel(agents_content, title="Agents"))
                except Exception as e:
                    layout["agents"].update(Panel(f"Error: {e}", title="Agents"))
                
                # Update footer
                layout["footer"].update(Panel(
                    "[bold]Commands:[/bold] Use CLI commands in another terminal: "
                    "safla system start, safla monitor metrics, safla agents list",
                    style="dim"
                ))
                
                import time
                time.sleep(5)  # Update every 5 seconds
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/yellow]")


def _create_status_display(status_data: Dict[str, Any]) -> str:
    """Create status display content."""
    lines = []
    
    # Overall health
    health = status_data.get('health', 'unknown')
    health_color = {
        'healthy': 'green',
        'warning': 'yellow',
        'error': 'red',
        'unknown': 'blue'
    }.get(health, 'blue')
    
    lines.append(f"[{health_color}]Overall Health: {health.title()}[/{health_color}]")
    lines.append(f"Uptime: {status_data.get('uptime', 'unknown')}")
    lines.append("")
    
    # Component status
    for component, info in status_data.get('components', {}).items():
        status = info.get('status', 'unknown')
        health = info.get('health', 'unknown')
        
        status_icon = {
            'running': '🟢',
            'stopped': '🔴',
            'error': '❌',
            'unknown': '⚪'
        }.get(status, '⚪')
        
        lines.append(f"{status_icon} {component.title()}: {status} ({health})")
    
    return '\n'.join(lines)


def _create_metrics_display(metrics_data: Dict[str, Any]) -> str:
    """Create metrics display content."""
    lines = []
    
    system_metrics = metrics_data.get('system', {})
    
    # Key system metrics
    if 'cpu_percent' in system_metrics:
        cpu = system_metrics['cpu_percent']
        cpu_color = 'green' if cpu < 70 else 'yellow' if cpu < 90 else 'red'
        lines.append(f"[{cpu_color}]CPU: {cpu:.1f}%[/{cpu_color}]")
    
    if 'memory_percent' in system_metrics:
        mem = system_metrics['memory_percent']
        mem_color = 'green' if mem < 70 else 'yellow' if mem < 90 else 'red'
        lines.append(f"[{mem_color}]Memory: {mem:.1f}%[/{mem_color}]")
    
    if 'disk_percent' in system_metrics:
        disk = system_metrics['disk_percent']
        disk_color = 'green' if disk < 80 else 'yellow' if disk < 95 else 'red'
        lines.append(f"[{disk_color}]Disk: {disk:.1f}%[/{disk_color}]")
    
    lines.append("")
    lines.append("[dim]Component Performance:[/dim]")
    
    # Component metrics summary
    components = metrics_data.get('components', {})
    for component in ['memory', 'cognition', 'mcp']:
        comp_data = components.get(component, {})
        if comp_data:
            cpu = comp_data.get('cpu_usage_percent', 0)
            lines.append(f"  {component.title()}: {cpu:.1f}% CPU")
    
    return '\n'.join(lines)


def _create_agents_display(agents_data: List[Dict[str, Any]]) -> str:
    """Create agents display content."""
    if not agents_data:
        return "[dim]No agents deployed[/dim]"
    
    lines = []
    for agent in agents_data:
        status_icon = {
            'running': '🟢',
            'scaling': '🟡',
            'error': '🔴',
            'stopped': '⚪'
        }.get(agent['status'], '⚪')
        
        lines.append(
            f"{status_icon} {agent['name']} ({agent['type']}) - "
            f"{agent['replicas']} replicas"
        )
    
    return '\n'.join(lines)


# Additional helper functions for interactive features
def _interactive_config_editor(manager):
    """Interactive configuration editor."""
    console.print("[bold cyan]Interactive Configuration Editor[/bold cyan]")
    
    while True:
        console.print("\n[bold]Available actions:[/bold]")
        console.print("1. View current configuration")
        console.print("2. Modify setting")
        console.print("3. Reset to defaults")
        console.print("4. Save and exit")
        console.print("5. Exit without saving")
        
        choice = Prompt.ask("Choose action", choices=['1', '2', '3', '4', '5'])
        
        if choice == '1':
            _display_config(manager.config, 'yaml')
        elif choice == '2':
            _interactive_modify_setting(manager)
        elif choice == '3':
            if Confirm.ask("Reset all settings to defaults?"):
                _reset_config_to_defaults(manager)
        elif choice == '4':
            _save_config(manager.config)
            console.print("[green]Configuration saved[/green]")
            break
        elif choice == '5':
            if Confirm.ask("Exit without saving changes?"):
                break


def _interactive_modify_setting(manager):
    """Interactive setting modification."""
    key = Prompt.ask("Setting name (use dot notation for nested keys)")
    
    try:
        current_value = _get_config_value(manager.config, key)
        console.print(f"Current value: {current_value}")
    except:
        console.print("Setting not found, creating new setting")
        current_value = None
    
    new_value = Prompt.ask("New value", default=str(current_value) if current_value else "")
    
    try:
        _set_config_value(manager.config, key, new_value)
        console.print(f"[green]Updated {key} = {new_value}[/green]")
    except Exception as e:
        console.print(f"[red]Error setting value: {e}[/red]")


def _show_help_menu():
    """Show interactive help menu."""
    help_content = """
[bold cyan]SAFLA CLI Help[/bold cyan]

[bold]System Management:[/bold]
  safla system status     - Show system status
  safla system start      - Start all components
  safla system stop       - Stop all components
  safla system validate   - Validate installation

[bold]Configuration:[/bold]
  safla config show       - Show configuration
  safla config set        - Set configuration value
  safla config edit       - Edit in text editor
  safla config backup     - Backup configuration

[bold]Monitoring:[/bold]
  safla monitor live      - Live monitoring dashboard
  safla monitor logs      - View system logs
  safla monitor metrics   - Show system metrics

[bold]Optimization:[/bold]
  safla optimize analyze  - Analyze optimization opportunities
  safla optimize apply    - Apply optimizations
  safla optimize memory   - Optimize memory usage

[bold]Benchmarking:[/bold]
  safla benchmark run     - Run benchmark suite
  safla benchmark stress  - Run stress tests

[bold]Agent Management:[/bold]
  safla agents list       - List deployed agents
  safla agents deploy     - Deploy new agent
  safla agents scale      - Scale agent replicas

[bold]Interactive:[/bold]
  safla dashboard         - Launch interactive dashboard
  safla setup             - Run setup wizard
"""
    
    console.print(Panel(help_content, title="SAFLA CLI Help", border_style="blue"))


if __name__ == "__main__":
    # Demo of interactive components
    from safla.cli_manager import SaflaCliManager
    
    manager = SaflaCliManager()
    manager.load_config()
    
    console.print("[bold blue]SAFLA Interactive Components Demo[/bold blue]")
    
    choice = Prompt.ask(
        "Choose demo",
        choices=['setup', 'dashboard', 'help'],
        default='setup'
    )
    
    if choice == 'setup':
        _run_setup_wizard(manager)
    elif choice == 'dashboard':
        _launch_interactive_dashboard(manager)
    elif choice == 'help':
        _show_help_menu()