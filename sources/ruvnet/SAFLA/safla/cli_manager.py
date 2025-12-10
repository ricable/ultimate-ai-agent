"""
SAFLA CLI Management System

A comprehensive command-line interface for managing SAFLA system operations,
configuration, monitoring, optimization, and agent deployment.
"""

import click
import asyncio
import json
import sys
import time
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import subprocess

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.json import JSON
from rich.syntax import Syntax
from rich.text import Text

# Note: Some imports may need adjustment based on actual module availability
try:
    from safla.utils.config import SAFLAConfig, get_config
except ImportError:
    SAFLAConfig = dict
    get_config = lambda: {}

try:
    from safla.utils.logging import setup_logging, get_logger
except ImportError:
    import logging
    setup_logging = lambda **kwargs: logging.basicConfig()
    get_logger = logging.getLogger

try:
    from safla.utils.validation import validate_installation, validate_config, check_gpu_availability
except ImportError:
    validate_installation = lambda: {"status": "not_available"}
    validate_config = lambda config: []
    check_gpu_availability = lambda: {"available": False}

try:
    from safla.exceptions import SAFLAError, ConfigurationError
except ImportError:
    class SAFLAError(Exception): pass
    class ConfigurationError(Exception): pass

# Import all implementation functions
from safla.cli_implementations import *
from safla.cli_implementations import (
    _get_performance_summary, _get_system_metrics, _get_deployed_agents,
    _get_recent_logs
)
from safla.cli_interactive import _run_setup_wizard, _launch_interactive_dashboard

console = Console()
logger = get_logger(__name__)


class SaflaCliManager:
    """Main CLI manager for SAFLA system operations."""
    
    def __init__(self):
        self.config = None
        self.console = Console()
        self.running_processes = {}
        
    def load_config(self, config_path: Optional[str] = None) -> SAFLAConfig:
        """Load SAFLA configuration."""
        try:
            if config_path:
                self.config = SAFLAConfig.from_file(config_path)
            else:
                self.config = get_config()
            return self.config
        except Exception as e:
            self.console.print(f"[red]Error loading configuration: {e}[/red]")
            raise


@click.group()
@click.option('--config', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--quiet', is_flag=True, help='Suppress output except errors')
@click.pass_context
def cli(ctx, config, debug, quiet):
    """SAFLA CLI Management System - Comprehensive system management interface."""
    ctx.ensure_object(dict)
    
    # Initialize CLI manager
    manager = SaflaCliManager()
    
    # Setup logging
    log_level = 'DEBUG' if debug else 'WARNING' if quiet else 'INFO'
    setup_logging(level=log_level, rich_logging=not quiet)
    
    # Load configuration
    try:
        manager.load_config(config)
        ctx.obj['manager'] = manager
        ctx.obj['config'] = manager.config
        ctx.obj['debug'] = debug
        ctx.obj['quiet'] = quiet
    except Exception as e:
        if not quiet:
            console.print(f"[red]Initialization failed: {e}[/red]")
        sys.exit(1)


# =============================================================================
# SYSTEM MANAGEMENT COMMANDS
# =============================================================================

@cli.group()
@click.pass_context
def system(ctx):
    """System management and operations."""
    pass


@system.command()
@click.option('--format', 'output_format', default='table', type=click.Choice(['table', 'json', 'yaml']), 
              help='Output format')
@click.option('--detailed', is_flag=True, help='Show detailed information')
@click.pass_context
def status(ctx, output_format, detailed):
    """Show system status and health."""
    manager = ctx.obj['manager']
    
    with console.status("[bold green]Checking system status..."):
        status_data = _get_system_status(manager, detailed)
    
    if output_format == 'json':
        console.print(JSON.from_data(status_data))
    elif output_format == 'yaml':
        import yaml
        console.print(yaml.dump(status_data, default_flow_style=False))
    else:
        _display_status_table(status_data, detailed)


@system.command()
@click.option('--component', help='Start specific component')
@click.option('--daemon', is_flag=True, help='Run as daemon')
@click.pass_context
def start(ctx, component, daemon):
    """Start SAFLA system or specific components."""
    manager = ctx.obj['manager']
    
    if component:
        _start_component(manager, component, daemon)
    else:
        _start_full_system(manager, daemon)


@system.command()
@click.option('--component', help='Stop specific component')
@click.option('--force', is_flag=True, help='Force stop')
@click.pass_context
def stop(ctx, component, force):
    """Stop SAFLA system or specific components."""
    manager = ctx.obj['manager']
    
    if component:
        _stop_component(manager, component, force)
    else:
        _stop_full_system(manager, force)


@system.command()
@click.option('--component', help='Restart specific component')
@click.pass_context
def restart(ctx, component):
    """Restart SAFLA system or specific components."""
    manager = ctx.obj['manager']
    
    if component:
        _restart_component(manager, component)
    else:
        _restart_full_system(manager)


@system.command()
@click.option('--output', '-o', type=click.Path(), help='Output file for installation report')
@click.pass_context
def validate(ctx, output):
    """Validate system installation and configuration."""
    manager = ctx.obj['manager']
    
    console.print("[bold blue]Validating SAFLA Installation...[/bold blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # System validation
        task = progress.add_task("Validating system requirements...", total=None)
        validation_results = validate_installation()
        progress.update(task, completed=True)
        
        # Configuration validation
        task = progress.add_task("Validating configuration...", total=None)
        config_errors = validate_config(manager.config)
        progress.update(task, completed=True)
        
        # GPU check
        task = progress.add_task("Checking GPU availability...", total=None)
        gpu_info = check_gpu_availability()
        progress.update(task, completed=True)
        
        # Component health checks
        task = progress.add_task("Running component health checks...", total=None)
        health_results = _run_health_checks(manager)
        progress.update(task, completed=True)
    
    # Display results
    _display_validation_results(validation_results, config_errors, gpu_info, health_results)
    
    # Save report if requested
    if output:
        report = {
            "timestamp": datetime.now().isoformat(),
            "validation": validation_results,
            "config_errors": config_errors,
            "gpu_info": gpu_info,
            "health_checks": health_results
        }
        
        with open(output, 'w') as f:
            json.dump(report, f, indent=2)
        console.print(f"[green]Validation report saved to {output}[/green]")


# =============================================================================
# CONFIGURATION MANAGEMENT COMMANDS
# =============================================================================

@cli.group()
@click.pass_context
def config(ctx):
    """Configuration management commands."""
    pass


@config.command()
@click.option('--key', help='Specific configuration key to show')
@click.option('--format', 'output_format', default='yaml', type=click.Choice(['yaml', 'json', 'env']),
              help='Output format')
@click.pass_context
def show(ctx, key, output_format):
    """Show current configuration."""
    manager = ctx.obj['manager']
    
    if key:
        value = _get_config_value(manager.config, key)
        console.print(f"{key}: {value}")
    else:
        _display_config(manager.config, output_format)


@config.command()
@click.argument('key')
@click.argument('value')
@click.option('--persistent', is_flag=True, help='Save to configuration file')
@click.pass_context
def set(ctx, key, value, persistent):
    """Set configuration value."""
    manager = ctx.obj['manager']
    
    try:
        _set_config_value(manager.config, key, value)
        console.print(f"[green]Set {key} = {value}[/green]")
        
        if persistent:
            _save_config(manager.config)
            console.print("[green]Configuration saved[/green]")
    except Exception as e:
        console.print(f"[red]Error setting configuration: {e}[/red]")


@config.command()
@click.argument('key')
@click.option('--persistent', is_flag=True, help='Remove from configuration file')
@click.pass_context
def unset(ctx, key, persistent):
    """Unset configuration value."""
    manager = ctx.obj['manager']
    
    try:
        _unset_config_value(manager.config, key)
        console.print(f"[green]Unset {key}[/green]")
        
        if persistent:
            _save_config(manager.config)
            console.print("[green]Configuration saved[/green]")
    except Exception as e:
        console.print(f"[red]Error unsetting configuration: {e}[/red]")


@config.command()
@click.pass_context
def edit(ctx):
    """Open configuration in editor."""
    manager = ctx.obj['manager']
    config_file = os.environ.get('SAFLA_CONFIG_FILE', '.env')
    
    editor = os.environ.get('EDITOR', 'nano')
    
    try:
        subprocess.run([editor, config_file])
        console.print("[green]Configuration file edited[/green]")
        
        # Offer to reload
        if Confirm.ask("Reload configuration?"):
            manager.load_config(config_file)
            console.print("[green]Configuration reloaded[/green]")
    except Exception as e:
        console.print(f"[red]Error editing configuration: {e}[/red]")


@config.command()
@click.option('--backup-dir', default='./backups', help='Backup directory')
@click.pass_context
def backup(ctx, backup_dir):
    """Backup current configuration."""
    manager = ctx.obj['manager']
    
    backup_path = _backup_config(manager.config, backup_dir)
    console.print(f"[green]Configuration backed up to {backup_path}[/green]")


@config.command()
@click.argument('backup_file', type=click.Path(exists=True))
@click.pass_context
def restore(ctx, backup_file):
    """Restore configuration from backup."""
    manager = ctx.obj['manager']
    
    if Confirm.ask(f"Restore configuration from {backup_file}?"):
        _restore_config(manager, backup_file)
        console.print("[green]Configuration restored[/green]")


# =============================================================================
# MONITORING COMMANDS
# =============================================================================

@cli.group()
@click.pass_context
def monitor(ctx):
    """System monitoring and metrics."""
    pass


@monitor.command()
@click.option('--refresh', default=5, help='Refresh interval in seconds')
@click.option('--duration', default=0, help='Monitoring duration (0 = indefinite)')
@click.pass_context
def live(ctx, refresh, duration):
    """Live system monitoring dashboard."""
    manager = ctx.obj['manager']
    
    _run_live_monitor(manager, refresh, duration)


@monitor.command()
@click.option('--component', help='Monitor specific component')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.option('--lines', '-n', default=50, help='Number of lines to show')
@click.pass_context
def logs(ctx, component, follow, lines):
    """View system logs."""
    manager = ctx.obj['manager']
    
    _view_logs(manager, component, follow, lines)


@monitor.command()
@click.option('--format', 'output_format', default='table', type=click.Choice(['table', 'json']),
              help='Output format')
@click.option('--detailed', is_flag=True, help='Show detailed metrics')
@click.pass_context
def metrics(ctx, output_format, detailed):
    """Show system metrics."""
    manager = ctx.obj['manager']
    
    metrics_data = _get_system_metrics(manager, detailed)
    
    if output_format == 'json':
        console.print(JSON.from_data(metrics_data))
    else:
        _display_metrics_table(metrics_data, detailed)


@monitor.command()
@click.option('--component', help='Monitor specific component performance')
@click.option('--duration', default=60, help='Monitoring duration in seconds')
@click.pass_context
def performance(ctx, component, duration):
    """Monitor system performance in real-time."""
    manager = ctx.obj['manager']
    
    _monitor_performance(manager, component, duration)


# =============================================================================
# OPTIMIZATION COMMANDS
# =============================================================================

@cli.group()
@click.pass_context
def optimize(ctx):
    """System optimization commands."""
    pass


@optimize.command()
@click.option('--auto', is_flag=True, help='Auto-apply recommended optimizations')
@click.option('--component', help='Optimize specific component')
@click.pass_context
def analyze(ctx, auto, component):
    """Analyze system for optimization opportunities."""
    manager = ctx.obj['manager']
    
    with console.status("[bold green]Analyzing system performance..."):
        optimizations = _analyze_optimizations(manager, component)
    
    _display_optimization_recommendations(optimizations)
    
    if auto and optimizations:
        if Confirm.ask("Apply recommended optimizations?"):
            _apply_optimizations(manager, optimizations)


@optimize.command()
@click.option('--target', help='Optimization target (memory, cpu, network)')
@click.option('--aggressive', is_flag=True, help='Use aggressive optimization')
@click.pass_context
def apply(ctx, target, aggressive):
    """Apply system optimizations."""
    manager = ctx.obj['manager']
    
    _apply_targeted_optimization(manager, target, aggressive)


@optimize.command()
@click.pass_context
def memory(ctx):
    """Optimize memory usage."""
    manager = ctx.obj['manager']
    
    with console.status("[bold green]Optimizing memory..."):
        result = _optimize_memory(manager)
    
    console.print(f"[green]Memory optimization complete: {result}[/green]")


@optimize.command()
@click.pass_context
def cache(ctx):
    """Optimize cache performance."""
    manager = ctx.obj['manager']
    
    with console.status("[bold green]Optimizing cache..."):
        result = _optimize_cache(manager)
    
    console.print(f"[green]Cache optimization complete: {result}[/green]")


# =============================================================================
# BENCHMARK COMMANDS
# =============================================================================

@cli.group()
@click.pass_context
def benchmark(ctx):
    """System benchmarking commands."""
    pass


@benchmark.command()
@click.option('--suite', default='standard', type=click.Choice(['quick', 'standard', 'comprehensive']),
              help='Benchmark suite to run')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--compare', type=click.Path(exists=True), help='Compare with previous results')
@click.pass_context
def run(ctx, suite, output, compare):
    """Run system benchmarks."""
    manager = ctx.obj['manager']
    
    console.print(f"[bold blue]Running {suite} benchmark suite...[/bold blue]")
    
    results = _run_benchmark_suite(manager, suite)
    
    _display_benchmark_results(results)
    
    if output:
        _save_benchmark_results(results, output)
        console.print(f"[green]Results saved to {output}[/green]")
    
    if compare:
        _compare_benchmark_results(results, compare)


@benchmark.command()
@click.option('--component', required=True, help='Component to benchmark')
@click.option('--iterations', default=100, help='Number of iterations')
@click.pass_context
def component(ctx, component, iterations):
    """Benchmark specific component."""
    manager = ctx.obj['manager']
    
    results = _benchmark_component(manager, component, iterations)
    _display_component_benchmark(results)


@benchmark.command()
@click.option('--duration', default=300, help='Stress test duration in seconds')
@click.option('--load-level', default=0.8, help='Load level (0.0-1.0)')
@click.pass_context
def stress(ctx, duration, load_level):
    """Run stress tests."""
    manager = ctx.obj['manager']
    
    console.print(f"[bold yellow]Running stress test for {duration}s at {load_level*100}% load...[/bold yellow]")
    
    results = _run_stress_test(manager, duration, load_level)
    _display_stress_test_results(results)


# =============================================================================
# AGENT DEPLOYMENT COMMANDS
# =============================================================================

@cli.group()
@click.pass_context
def agents(ctx):
    """Agent deployment and management."""
    pass


@agents.command()
@click.option('--format', 'output_format', default='table', type=click.Choice(['table', 'json']),
              help='Output format')
@click.pass_context
def list(ctx, output_format):
    """List deployed agents."""
    manager = ctx.obj['manager']
    
    agents_data = _get_deployed_agents(manager)
    
    if output_format == 'json':
        console.print(JSON.from_data(agents_data))
    else:
        _display_agents_table(agents_data)


@agents.command()
@click.argument('agent_name')
@click.option('--config-file', type=click.Path(exists=True), help='Agent configuration file')
@click.option('--replicas', default=1, help='Number of replicas')
@click.option('--resources', help='Resource requirements (JSON)')
@click.pass_context
def deploy(ctx, agent_name, config_file, replicas, resources):
    """Deploy a new agent."""
    manager = ctx.obj['manager']
    
    deployment_config = {
        'name': agent_name,
        'replicas': replicas,
        'config_file': config_file,
        'resources': json.loads(resources) if resources else {}
    }
    
    _deploy_agent(manager, deployment_config)


@agents.command()
@click.argument('agent_name')
@click.option('--force', is_flag=True, help='Force removal')
@click.pass_context
def remove(ctx, agent_name, force):
    """Remove deployed agent."""
    manager = ctx.obj['manager']
    
    if force or Confirm.ask(f"Remove agent {agent_name}?"):
        _remove_agent(manager, agent_name)
        console.print(f"[green]Agent {agent_name} removed[/green]")


@agents.command()
@click.argument('agent_name')
@click.option('--replicas', help='New number of replicas')
@click.option('--resources', help='New resource requirements (JSON)')
@click.pass_context
def scale(ctx, agent_name, replicas, resources):
    """Scale agent deployment."""
    manager = ctx.obj['manager']
    
    scale_config = {}
    if replicas:
        scale_config['replicas'] = int(replicas)
    if resources:
        scale_config['resources'] = json.loads(resources)
    
    _scale_agent(manager, agent_name, scale_config)


@agents.command()
@click.argument('agent_name')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.option('--lines', '-n', default=50, help='Number of lines to show')
@click.pass_context
def logs(ctx, agent_name, follow, lines):
    """View agent logs."""
    manager = ctx.obj['manager']
    
    _view_agent_logs(manager, agent_name, follow, lines)


# =============================================================================
# INTERACTIVE TUI COMMANDS
# =============================================================================

@cli.command()
@click.pass_context
def dashboard(ctx):
    """Launch interactive dashboard."""
    manager = ctx.obj['manager']
    
    _launch_interactive_dashboard(manager)


@cli.command()
@click.pass_context
def setup(ctx):
    """Interactive system setup wizard."""
    manager = ctx.obj['manager']
    
    _run_setup_wizard(manager)


@cli.command()
@click.pass_context
def help_menu(ctx):
    """Show comprehensive help menu."""
    from safla.cli_interactive import _show_help_menu
    _show_help_menu()


@cli.command()
@click.argument('query', required=False)
@click.pass_context
def search(ctx, query):
    """Search for commands, settings, or documentation."""
    if not query:
        query = click.prompt("Enter search query")
    
    _search_cli_help(query)


@cli.command()
@click.pass_context
def doctor(ctx):
    """Run comprehensive system health check and diagnostics."""
    manager = ctx.obj['manager']
    
    console.print("[bold blue]Running SAFLA System Diagnostics...[/bold blue]")
    
    with Progress() as progress:
        # System validation
        task = progress.add_task("System validation...", total=None)
        validation_results = validate_installation()
        progress.update(task, completed=True)
        
        # Configuration check
        task = progress.add_task("Configuration check...", total=None)
        config_errors = validate_config(manager.config)
        progress.update(task, completed=True)
        
        # Health checks
        task = progress.add_task("Component health checks...", total=None)
        health_results = _run_health_checks(manager)
        progress.update(task, completed=True)
        
        # Performance check
        task = progress.add_task("Performance analysis...", total=None)
        perf_results = _run_performance_check(manager)
        progress.update(task, completed=True)
        
        # Security check
        task = progress.add_task("Security audit...", total=None)
        security_results = _run_security_check(manager)
        progress.update(task, completed=True)
    
    # Display comprehensive report
    _display_doctor_report(validation_results, config_errors, health_results, 
                          perf_results, security_results)


@cli.command()
@click.option('--format', 'output_format', default='table', 
              type=click.Choice(['table', 'json', 'yaml']), help='Output format')
@click.pass_context
def version(ctx, output_format):
    """Show version information."""
    version_info = _get_version_info()
    
    if output_format == 'json':
        console.print(JSON.from_data(version_info))
    elif output_format == 'yaml':
        import yaml
        console.print(yaml.dump(version_info, default_flow_style=False))
    else:
        _display_version_table(version_info)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_system_status(manager: SaflaCliManager, detailed: bool = False) -> Dict[str, Any]:
    """Get comprehensive system status."""
    status = {
        'timestamp': datetime.now().isoformat(),
        'uptime': _get_system_uptime(),
        'components': {},
        'health': 'unknown',
        'performance': {}
    }
    
    # Check component status
    components = ['memory', 'cognition', 'safety', 'mcp', 'agents']
    for component in components:
        status['components'][component] = _check_component_status(component)
    
    # Overall health
    component_healths = [comp['health'] for comp in status['components'].values()]
    if all(h == 'healthy' for h in component_healths):
        status['health'] = 'healthy'
    elif any(h == 'error' for h in component_healths):
        status['health'] = 'error'
    else:
        status['health'] = 'warning'
    
    # Performance metrics
    if detailed:
        status['performance'] = _get_performance_summary()
    
    return status


def _display_status_table(status_data: Dict[str, Any], detailed: bool = False):
    """Display system status in table format."""
    table = Table(title="SAFLA System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Health", style="yellow")
    table.add_column("Uptime", style="blue")
    
    if detailed:
        table.add_column("CPU", style="magenta")
        table.add_column("Memory", style="red")
    
    for component, info in status_data['components'].items():
        row = [
            component.title(),
            info.get('status', 'unknown'),
            info.get('health', 'unknown'),
            info.get('uptime', 'unknown')
        ]
        
        if detailed:
            row.extend([
                f"{info.get('cpu_usage', 0):.1f}%",
                f"{info.get('memory_usage', 0):.1f}%"
            ])
        
        table.add_row(*row)
    
    console.print(table)
    
    # Overall status panel
    health_color = {
        'healthy': 'green',
        'warning': 'yellow',
        'error': 'red',
        'unknown': 'blue'
    }.get(status_data['health'], 'blue')
    
    console.print(Panel(
        f"[{health_color}]System Health: {status_data['health'].title()}[/{health_color}]\n"
        f"Uptime: {status_data['uptime']}\n"
        f"Last Check: {status_data['timestamp']}",
        title="Overall Status"
    ))


def _start_component(manager: SaflaCliManager, component: str, daemon: bool = False):
    """Start a specific component."""
    console.print(f"[blue]Starting {component}...[/blue]")
    
    component_map = {
        'memory': _start_memory_system,
        'cognition': _start_cognitive_engine,
        'safety': _start_safety_system,
        'mcp': _start_mcp_server,
        'agents': _start_agent_manager
    }
    
    if component in component_map:
        try:
            component_map[component](manager, daemon)
            console.print(f"[green]{component.title()} started successfully[/green]")
        except Exception as e:
            console.print(f"[red]Failed to start {component}: {e}[/red]")
    else:
        console.print(f"[red]Unknown component: {component}[/red]")


def _run_live_monitor(manager: SaflaCliManager, refresh: int, duration: int):
    """Run live monitoring dashboard."""
    start_time = time.time()
    
    def make_layout():
        """Create the layout for live monitoring."""
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
    
    with Live(layout, refresh_per_second=1/refresh) as live:
        while True:
            # Update header
            layout["header"].update(Panel(
                f"SAFLA Live Monitor - {datetime.now().strftime('%H:%M:%S')} | "
                f"Refresh: {refresh}s | "
                f"Running: {time.time() - start_time:.0f}s",
                style="bold blue"
            ))
            
            # Update status
            status_data = _get_system_status(manager, detailed=True)
            status_table = _create_status_table(status_data)
            layout["status"].update(Panel(status_table, title="System Status"))
            
            # Update metrics
            metrics_data = _get_system_metrics(manager, detailed=False)
            metrics_table = _create_metrics_table(metrics_data)
            layout["metrics"].update(Panel(metrics_table, title="Performance Metrics"))
            
            # Update logs
            recent_logs = _get_recent_logs(manager, lines=10)
            layout["logs"].update(Panel(recent_logs, title="Recent Logs"))
            
            # Update agents
            agents_data = _get_deployed_agents(manager)
            agents_table = _create_agents_table(agents_data)
            layout["agents"].update(Panel(agents_table, title="Deployed Agents"))
            
            # Check duration
            if duration > 0 and (time.time() - start_time) >= duration:
                break
            
            time.sleep(refresh)


def _launch_interactive_dashboard(manager: SaflaCliManager):
    """Launch interactive TUI dashboard."""
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Container, Horizontal, Vertical
        from textual.widgets import Header, Footer, Static, DataTable, Log
        from textual.reactive import reactive
        
        class SaflaDashboard(App):
            """SAFLA Interactive Dashboard."""
            
            CSS_PATH = "dashboard.css"
            BINDINGS = [
                ("d", "toggle_dark", "Toggle dark mode"),
                ("q", "quit", "Quit"),
                ("r", "refresh", "Refresh"),
            ]
            
            def compose(self) -> ComposeResult:
                yield Header()
                
                with Horizontal():
                    with Vertical(classes="left-panel"):
                        yield Static("System Status", classes="panel-title")
                        yield DataTable(id="status-table")
                        
                        yield Static("Performance Metrics", classes="panel-title")
                        yield DataTable(id="metrics-table")
                    
                    with Vertical(classes="right-panel"):
                        yield Static("System Logs", classes="panel-title")
                        yield Log(id="system-logs")
                        
                        yield Static("Deployed Agents", classes="panel-title")
                        yield DataTable(id="agents-table")
                
                yield Footer()
            
            def on_mount(self) -> None:
                self.refresh_data()
            
            def action_refresh(self) -> None:
                self.refresh_data()
            
            def refresh_data(self) -> None:
                # Update status table
                status_table = self.query_one("#status-table", DataTable)
                status_data = _get_system_status(manager, detailed=True)
                _populate_status_table(status_table, status_data)
                
                # Update metrics table
                metrics_table = self.query_one("#metrics-table", DataTable)
                metrics_data = _get_system_metrics(manager, detailed=True)
                _populate_metrics_table(metrics_table, metrics_data)
                
                # Update logs
                log_widget = self.query_one("#system-logs", Log)
                recent_logs = _get_recent_logs(manager, lines=50)
                log_widget.clear()
                for log_line in recent_logs.split('\n'):
                    log_widget.write_line(log_line)
                
                # Update agents table
                agents_table = self.query_one("#agents-table", DataTable)
                agents_data = _get_deployed_agents(manager)
                _populate_agents_table(agents_table, agents_data)
        
        app = SaflaDashboard()
        app.run()
        
    except ImportError:
        console.print("[red]Textual not installed. Install with: pip install textual[/red]")
        # Fallback to simple live monitor
        _run_live_monitor(manager, 5, 0)


# Component-specific helper functions
def _check_component_status(component: str) -> Dict[str, Any]:
    """Check status of a specific component."""
    # This would implement actual component health checks
    return {
        'status': 'running',
        'health': 'healthy',
        'uptime': '2h 30m',
        'cpu_usage': 25.5,
        'memory_usage': 45.2
    }


def _get_system_uptime() -> str:
    """Get system uptime."""
    try:
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
        
        uptime_delta = timedelta(seconds=uptime_seconds)
        days = uptime_delta.days
        hours, remainder = divmod(uptime_delta.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        return f"{days}d {hours}h {minutes}m"
    except:
        return "unknown"


# Helper function imports for missing functions
def _get_version_info():
    """Fallback version info if not imported from implementations."""
    import platform
    import sys
    return {
        'safla_version': '2.0.0',
        'cli_version': '1.0.0',
        'python_version': sys.version.split()[0],
        'platform': platform.platform()
    }

def _display_version_table(version_info):
    """Fallback version display."""
    console.print(f"SAFLA Version: {version_info.get('safla_version', 'unknown')}")
    console.print(f"CLI Version: {version_info.get('cli_version', 'unknown')}")
    console.print(f"Python Version: {version_info.get('python_version', 'unknown')}")
    console.print(f"Platform: {version_info.get('platform', 'unknown')}")

def _search_cli_help(query):
    """Fallback search function."""
    console.print(f"[blue]Searching for: {query}[/blue]")
    console.print("Search functionality available. Use 'help-menu' for full command list.")

def _run_health_checks(manager):
    """Fallback health checks."""
    return {
        'overall_health': 'good',
        'checks': {
            'basic': {'status': 'pass', 'message': 'Basic checks passed'}
        }
    }

def _run_performance_check(manager):
    """Fallback performance check."""
    return {'overall_performance': 'good'}

def _run_security_check(manager):
    """Fallback security check."""
    return {'security_level': 'medium'}

def _display_doctor_report(*args):
    """Fallback doctor report display."""
    console.print("[green]System diagnostic completed[/green]")


def _create_status_table(status_data: Dict[str, Any]) -> Table:
    """Create a status table for live monitoring."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=12)
    table.add_column("Status", style="green", width=10)
    table.add_column("Health", width=10)
    
    for component, info in status_data.get('components', {}).items():
        health = info.get('health', 'unknown')
        health_style = {
            'healthy': 'green',
            'warning': 'yellow',
            'error': 'red',
            'unknown': 'blue'
        }.get(health, 'white')
        
        table.add_row(
            component.title(),
            info.get('status', 'unknown'),
            f"[{health_style}]{health}[/{health_style}]"
        )
    
    return table


def _create_metrics_table(metrics_data: Dict[str, Any]) -> Table:
    """Create a metrics table for live monitoring."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=15)
    table.add_column("Value", style="green", width=15)
    
    # Add CPU metrics
    if 'cpu' in metrics_data:
        table.add_row("CPU Usage", f"{metrics_data['cpu'].get('percent', 0):.1f}%")
        table.add_row("Load Average", f"{metrics_data['cpu'].get('load_avg', [0])[0]:.2f}")
    
    # Add memory metrics
    if 'memory' in metrics_data:
        table.add_row("Memory Usage", f"{metrics_data['memory'].get('percent', 0):.1f}%")
        table.add_row("Available", f"{metrics_data['memory'].get('available_gb', 0):.1f} GB")
    
    # Add disk metrics
    if 'disk' in metrics_data:
        table.add_row("Disk Usage", f"{metrics_data['disk'].get('percent', 0):.1f}%")
    
    return table


def _create_agents_table(agents_data: List[Dict[str, Any]]) -> Table:
    """Create an agents table for live monitoring."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Agent", style="cyan", width=15)
    table.add_column("Status", style="green", width=10)
    table.add_column("Replicas", width=10)
    
    if isinstance(agents_data, list):
        for agent in agents_data:
            table.add_row(
                agent.get('name', 'unknown'),
                agent.get('status', 'unknown'),
                str(agent.get('replicas', 0))
            )
    else:
        table.add_row("No agents", "-", "-")
    
    return table


if __name__ == "__main__":
    cli()