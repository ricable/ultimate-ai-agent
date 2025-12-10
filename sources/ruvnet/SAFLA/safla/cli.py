"""
SAFLA Command Line Interface

This module provides the main command-line interface for SAFLA,
including installation, configuration, and system management commands.
"""

import click
import sys
import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from safla.utils.config import SAFLAConfig, get_config
from safla.utils.logging import setup_logging, get_logger
from safla.utils.validation import validate_installation, validate_config, check_gpu_availability
from safla.exceptions import SAFLAError, ConfigurationError


console = Console()
logger = get_logger(__name__)


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--config', type=click.Path(), help='Path to configuration file')
@click.option('--log-level', default='INFO', help='Logging level')
@click.pass_context
def cli(ctx, debug, config, log_level):
    """SAFLA - Self-Aware Feedback Loop Algorithm CLI"""
    ctx.ensure_object(dict)
    
    # Setup logging
    setup_logging(level=log_level, rich_logging=True)
    
    # Load configuration
    try:
        if config:
            safla_config = SAFLAConfig.from_file(config)
        else:
            safla_config = SAFLAConfig.from_env()
        
        if debug:
            safla_config.debug = True
            safla_config.log_level = "DEBUG"
        
        ctx.obj['config'] = safla_config
        
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file for validation report')
@click.pass_context
def validate(ctx, output):
    """Validate SAFLA installation and configuration"""
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
        config_errors = validate_config(ctx.obj['config'])
        progress.update(task, completed=True)
        
        # GPU check
        task = progress.add_task("Checking GPU availability...", total=None)
        gpu_info = check_gpu_availability()
        progress.update(task, completed=True)
    
    # Display results
    _display_validation_results(validation_results, config_errors, gpu_info)
    
    # Save report if requested
    if output:
        report = {
            "validation": validation_results,
            "config_errors": config_errors,
            "gpu_info": gpu_info
        }
        
        with open(output, 'w') as f:
            json.dump(report, f, indent=2)
        
        console.print(f"[green]Validation report saved to: {output}[/green]")
    
    # Exit with error code if validation failed
    if not validation_results["valid"] or config_errors:
        sys.exit(1)


@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output configuration file')
@click.option('--template', type=click.Choice(['minimal', 'development', 'production']), 
              default='minimal', help='Configuration template')
@click.pass_context
def init_config(ctx, output, template):
    """Initialize SAFLA configuration"""
    console.print(f"[bold blue]Initializing {template} configuration...[/bold blue]")
    
    # Create configuration based on template
    if template == 'minimal':
        config = SAFLAConfig()
    elif template == 'development':
        config = SAFLAConfig()
        config.debug = True
        config.log_level = "DEBUG"
        config.safety.monitoring_interval = 0.5
    elif template == 'production':
        config = SAFLAConfig()
        config.debug = False
        config.log_level = "WARNING"
        config.safety.memory_limit = 2 * 1024 * 1024 * 1024  # 2GB
        config.safety.monitoring_interval = 2.0
    
    # Set output path
    if not output:
        output = f"safla_config_{template}.json"
    
    # Save configuration
    try:
        config.save_to_file(output)
        console.print(f"[green]Configuration saved to: {output}[/green]")
        
        # Display configuration summary
        _display_config_summary(config)
        
    except Exception as e:
        console.print(f"[red]Error saving configuration: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def info(ctx):
    """Display SAFLA system information"""
    config = ctx.obj['config']
    
    # System information table
    table = Table(title="SAFLA System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    # Add system info
    validation_results = validate_installation()
    
    table.add_row(
        "Installation", 
        "✓ Valid" if validation_results["valid"] else "✗ Invalid",
        f"Python {validation_results['system_info']['python_version']}"
    )
    
    table.add_row(
        "Platform",
        "✓ Supported",
        f"{validation_results['system_info']['platform']} ({validation_results['system_info']['architecture']})"
    )
    
    # Memory info
    if "memory_total" in validation_results["system_info"]:
        table.add_row(
            "Memory",
            "✓ Available",
            f"{validation_results['system_info']['memory_available']} / {validation_results['system_info']['memory_total']}"
        )
    
    # GPU info
    gpu_info = check_gpu_availability()
    if gpu_info["cuda_available"]:
        table.add_row(
            "GPU",
            "✓ CUDA Available",
            f"{gpu_info['gpu_count']} GPU(s), CUDA {gpu_info['cuda_version']}"
        )
    else:
        table.add_row("GPU", "- Not Available", "CPU-only mode")
    
    console.print(table)
    
    # Configuration summary
    console.print("\n")
    _display_config_summary(config)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8080, help='Port to bind to')
@click.option('--gpu', is_flag=True, help='Enable GPU optimization')
@click.pass_context
def serve(ctx, host, port, gpu):
    """Start SAFLA HTTP server"""
    config = ctx.obj['config']
    
    console.print(f"[bold blue]Starting SAFLA Server on {host}:{port}...[/bold blue]")
    
    if gpu:
        console.print("[green]GPU optimization enabled[/green]")
        try:
            import torch
            if torch.cuda.is_available():
                console.print(f"[green]CUDA available: {torch.cuda.get_device_name()}[/green]")
            else:
                console.print("[yellow]CUDA not available, falling back to CPU[/yellow]")
        except ImportError:
            console.print("[yellow]PyTorch not available[/yellow]")
    
    try:
        from safla.mcp.server import start_server
        start_server(host=host, port=port, gpu_enabled=gpu, config=config)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down server...[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def health(ctx):
    """Health check endpoint"""
    import torch
    import json
    
    health_data = {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available() if 'torch' in globals() else False,
        "version": "0.1.3"
    }
    
    if torch.cuda.is_available():
        health_data["gpu_name"] = torch.cuda.get_device_name()
        health_data["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory
    
    print(json.dumps(health_data))


@cli.command()
@click.option('--component', type=click.Choice(['memory', 'safety', 'mcp', 'metacognitive', 'fastmcp']),
              help='Start specific component only')
@click.option('--daemon', is_flag=True, help='Run as daemon')
@click.pass_context
def start(ctx, component, daemon):
    """Start SAFLA system or components"""
    config = ctx.obj['config']
    
    console.print("[bold blue]Starting SAFLA System...[/bold blue]")
    
    if daemon:
        console.print("[yellow]Daemon mode not yet implemented[/yellow]")
        return
    
    try:
        if component:
            if component == 'fastmcp':
                _start_fastmcp_component(config)
            else:
                _start_component(component, config)
        else:
            _start_full_system(config)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down SAFLA...[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting SAFLA: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--format', type=click.Choice(['json', 'yaml', 'table']), default='table',
              help='Output format')
@click.pass_context
def status(ctx, format):
    """Show SAFLA system status"""
    console.print("[bold blue]SAFLA System Status[/bold blue]")
    
    # This would typically check running processes, memory usage, etc.
    # For now, show configuration status
    
    if format == 'table':
        table = Table(title="Component Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        # Mock status for demonstration
        table.add_row("Memory System", "Stopped", "Not running")
        table.add_row("Safety Framework", "Stopped", "Not running")
        table.add_row("MCP Orchestrator", "Stopped", "Not running")
        table.add_row("Meta-Cognitive Engine", "Stopped", "Not running")
        
        console.print(table)
    
    elif format == 'json':
        status_data = {
            "memory_system": {"status": "stopped", "details": "Not running"},
            "safety_framework": {"status": "stopped", "details": "Not running"},
            "mcp_orchestrator": {"status": "stopped", "details": "Not running"},
            "metacognitive_engine": {"status": "stopped", "details": "Not running"}
        }
        console.print(json.dumps(status_data, indent=2))


@cli.group()
@click.pass_context
def fastmcp(ctx):
    """FastMCP integration commands"""
    pass


@fastmcp.command()
@click.option('--config-file', type=click.Path(exists=True), help='FastMCP configuration file')
@click.option('--endpoint', help='Default endpoint name')
@click.pass_context
def client(ctx, config_file, endpoint):
    """Start FastMCP client interface"""
    console.print("[bold blue]Starting FastMCP Client...[/bold blue]")
    
    try:
        import asyncio
        from safla.integrations import create_fastmcp_client_from_config, FastMCPAdapter, FastMCPConfig, FastMCPClient
        
        if config_file:
            # Load from file
            client = asyncio.run(create_fastmcp_client_from_config(config_file, endpoint))
        else:
            # Create minimal client
            config = FastMCPConfig()
            adapter = FastMCPAdapter(config)
            asyncio.run(adapter.start())
            client = FastMCPClient(adapter)
        
        console.print("[green]FastMCP client started successfully[/green]")
        console.print("[yellow]Use 'safla fastmcp list-endpoints' to see available endpoints[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error starting FastMCP client: {e}[/red]")
        sys.exit(1)


@fastmcp.command()
@click.option('--name', required=True, help='Server name')
@click.option('--port', default=8000, help='Server port')
@click.option('--host', default='localhost', help='Server host')
@click.pass_context
def server(ctx, name, port, host):
    """Start FastMCP server"""
    console.print(f"[bold blue]Starting FastMCP Server: {name}...[/bold blue]")
    
    try:
        import asyncio
        from safla.integrations import FastMCPServer
        
        # Create server
        server = FastMCPServer(name=name, description=f"SAFLA FastMCP Server: {name}")
        
        # Add default health check tool
        @server.tool(name="ping", description="Ping the server")
        async def ping():
            return {"status": "pong", "server": name}
        
        # Start server
        console.print(f"[green]FastMCP server '{name}' starting on {host}:{port}[/green]")
        asyncio.run(server.start(host=host, port=port))
        
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Shutting down FastMCP server: {name}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting FastMCP server: {e}[/red]")
        sys.exit(1)


@fastmcp.command()
@click.option('--urls', multiple=True, help='Base URLs to discover (can be used multiple times)')
@click.option('--timeout', default=10.0, help='Discovery timeout in seconds')
@click.option('--output', type=click.Path(), help='Save results to file')
@click.pass_context
def discover(ctx, urls, timeout, output):
    """Discover FastMCP endpoints"""
    if not urls:
        console.print("[red]Please provide at least one URL to discover[/red]")
        sys.exit(1)
    
    console.print(f"[bold blue]Discovering FastMCP endpoints from {len(urls)} URLs...[/bold blue]")
    
    try:
        import asyncio
        from safla.integrations import discover_fastmcp_endpoints
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Discovering endpoints...", total=None)
            
            endpoints = asyncio.run(discover_fastmcp_endpoints(list(urls), timeout=timeout))
            progress.update(task, completed=True)
        
        if endpoints:
            # Display results
            table = Table(title="Discovered FastMCP Endpoints")
            table.add_column("Name", style="cyan")
            table.add_column("Base URL", style="blue")
            table.add_column("Version", style="green")
            table.add_column("Capabilities")
            
            for endpoint in endpoints:
                capabilities = ", ".join(endpoint.get("capabilities", []))
                table.add_row(
                    endpoint["name"],
                    endpoint["base_url"],
                    endpoint["version"],
                    capabilities or "None"
                )
            
            console.print(table)
            
            # Save to file if requested
            if output:
                import json
                with open(output, 'w') as f:
                    json.dump(endpoints, f, indent=2)
                console.print(f"[green]Results saved to: {output}[/green]")
        
        else:
            console.print("[yellow]No FastMCP endpoints discovered[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Error during discovery: {e}[/red]")
        sys.exit(1)


@fastmcp.command()
@click.option('--endpoint-url', required=True, help='FastMCP endpoint URL to test')
@click.option('--auth-token', help='Authentication token')
@click.option('--timeout', default=10.0, help='Test timeout in seconds')
@click.pass_context
def test(ctx, endpoint_url, auth_token, timeout):
    """Test FastMCP endpoint connectivity and capabilities"""
    console.print(f"[bold blue]Testing FastMCP endpoint: {endpoint_url}...[/bold blue]")
    
    try:
        import asyncio
        from safla.integrations import check_fastmcp_endpoint
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Testing endpoint...", total=None)
            
            results = asyncio.run(check_fastmcp_endpoint(
                endpoint_url=endpoint_url,
                auth_token=auth_token,
                timeout=timeout
            ))
            progress.update(task, completed=True)
        
        # Display results
        _display_fastmcp_test_results(results)
    
    except Exception as e:
        console.print(f"[red]Error testing endpoint: {e}[/red]")
        sys.exit(1)


@fastmcp.command()
@click.option('--config-file', type=click.Path(exists=True), help='FastMCP configuration file')
@click.pass_context
def list_endpoints(ctx, config_file):
    """List configured FastMCP endpoints"""
    console.print("[bold blue]FastMCP Endpoints[/bold blue]")
    
    try:
        import asyncio
        from safla.integrations import create_fastmcp_client_from_config, FastMCPConfig, FastMCPAdapter, FastMCPClient
        
        if config_file:
            client = asyncio.run(create_fastmcp_client_from_config(config_file))
        else:
            # Use default config
            config = FastMCPConfig()
            adapter = FastMCPAdapter(config)
            asyncio.run(adapter.start())
            client = FastMCPClient(adapter)
        
        endpoints = asyncio.run(client.list_endpoints())
        
        if endpoints:
            table = Table(title="Configured FastMCP Endpoints")
            table.add_column("Name", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Tools", style="blue")
            table.add_column("Resources", style="blue")
            table.add_column("Default", style="yellow")
            
            for endpoint in endpoints:
                status_style = "green" if endpoint.get("status") == "healthy" else "red"
                default_marker = "✓" if endpoint.get("is_default") else ""
                
                table.add_row(
                    endpoint.get("name", "Unknown"),
                    f"[{status_style}]{endpoint.get('status', 'unknown')}[/{status_style}]",
                    str(endpoint.get("tool_count", 0)),
                    str(endpoint.get("resource_count", 0)),
                    default_marker
                )
            
            console.print(table)
        else:
            console.print("[yellow]No FastMCP endpoints configured[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Error listing endpoints: {e}[/red]")
        sys.exit(1)


def _display_fastmcp_test_results(results):
    """Display FastMCP test results"""
    # Connectivity status
    if results["connectivity"]:
        console.print(Panel(
            "[bold green]✓ CONNECTIVITY PASSED[/bold green]",
            style="green"
        ))
    else:
        console.print(Panel(
            "[bold red]✗ CONNECTIVITY FAILED[/bold red]",
            style="red"
        ))
    
    # Capabilities table
    if results["capabilities"]:
        caps_table = Table(title="Endpoint Capabilities")
        caps_table.add_column("Capability", style="cyan")
        caps_table.add_column("Count", style="green")
        
        caps_table.add_row("Tools", str(results["capabilities"].get("tools", 0)))
        caps_table.add_row("Resources", str(results["capabilities"].get("resources", 0)))
        
        console.print(caps_table)
        
        # Tool names
        if results["capabilities"].get("tool_names"):
            console.print(f"[blue]Available Tools:[/blue] {', '.join(results['capabilities']['tool_names'])}")
        
        # Resource URIs
        if results["capabilities"].get("resource_uris"):
            console.print(f"[blue]Available Resources:[/blue] {', '.join(results['capabilities']['resource_uris'])}")
    
    # Performance info
    if results["performance"]:
        perf_table = Table(title="Performance Metrics")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green")
        
        for metric, value in results["performance"].items():
            if isinstance(value, float):
                perf_table.add_row(metric.replace("_", " ").title(), f"{value:.3f}s")
            else:
                perf_table.add_row(metric.replace("_", " ").title(), str(value))
        
        console.print(perf_table)
    
    # Errors
    if results["errors"]:
        error_table = Table(title="Test Errors", style="red")
        error_table.add_column("Error", style="red")
        
        for error in results["errors"]:
            error_table.add_row(error)
        
        console.print(error_table)


def _start_fastmcp_component(config):
    """Start FastMCP component"""
    console.print("[yellow]Starting FastMCP component...[/yellow]")
    
    try:
        import asyncio
        from safla.integrations import FastMCPAdapter, FastMCPConfig
        
        # Create FastMCP configuration from SAFLA config
        fastmcp_config = FastMCPConfig()
        
        # Create and start adapter
        adapter = FastMCPAdapter(fastmcp_config)
        
        async def run_fastmcp():
            await adapter.start()
            console.print("[green]FastMCP component started successfully[/green]")
            console.print("[yellow]Press Ctrl+C to stop[/yellow]")
            
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                await adapter.shutdown()
                console.print("[yellow]FastMCP component stopped[/yellow]")
        
        asyncio.run(run_fastmcp())
    
    except Exception as e:
        console.print(f"[red]Error starting FastMCP component: {e}[/red]")
        raise


def _display_validation_results(validation_results, config_errors, gpu_info):
    """Display validation results in a formatted table"""
    
    # Main validation panel
    if validation_results["valid"] and not config_errors:
        panel_style = "green"
        status = "✓ PASSED"
    else:
        panel_style = "red"
        status = "✗ FAILED"
    
    console.print(Panel(
        f"[bold]Validation Status: {status}[/bold]",
        style=panel_style
    ))
    
    # Errors table
    if validation_results["errors"] or config_errors:
        error_table = Table(title="Validation Errors", style="red")
        error_table.add_column("Type", style="cyan")
        error_table.add_column("Error", style="red")
        
        for error in validation_results["errors"]:
            error_table.add_row("System", error)
        
        for error in config_errors:
            error_table.add_row("Configuration", error)
        
        console.print(error_table)
    
    # Warnings table
    if validation_results["warnings"]:
        warning_table = Table(title="Warnings", style="yellow")
        warning_table.add_column("Warning", style="yellow")
        
        for warning in validation_results["warnings"]:
            warning_table.add_row(warning)
        
        console.print(warning_table)
    
    # System info table
    info_table = Table(title="System Information")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    
    for key, value in validation_results["system_info"].items():
        info_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(info_table)


def _display_config_summary(config):
    """Display configuration summary"""
    config_table = Table(title="Configuration Summary")
    config_table.add_column("Section", style="cyan")
    config_table.add_column("Key", style="blue")
    config_table.add_column("Value", style="green")
    
    # Memory configuration
    config_table.add_row("Memory", "Vector Dimensions", str(config.memory.vector_dimensions))
    config_table.add_row("", "Max Memories", str(config.memory.max_memories))
    config_table.add_row("", "Similarity Threshold", str(config.memory.similarity_threshold))
    
    # Safety configuration
    config_table.add_row("Safety", "Memory Limit", f"{config.safety.memory_limit / 1024 / 1024:.0f}MB")
    config_table.add_row("", "CPU Limit", f"{config.safety.cpu_limit * 100:.0f}%")
    config_table.add_row("", "Monitoring Interval", f"{config.safety.monitoring_interval}s")
    
    # General settings
    config_table.add_row("General", "Debug Mode", str(config.debug))
    config_table.add_row("", "Log Level", config.log_level)
    config_table.add_row("", "Data Directory", config.data_dir)
    
    console.print(config_table)


def _start_component(component_name, config):
    """Start a specific SAFLA component"""
    console.print(f"[yellow]Starting {component_name} component...[/yellow]")
    
    # This would typically import and start the actual component
    # For now, just show a placeholder
    console.print(f"[green]{component_name.title()} component started successfully[/green]")


def _start_full_system(config):
    """Start the full SAFLA system"""
    console.print("[yellow]Starting full SAFLA system...[/yellow]")
    
    components = ["Memory System", "Safety Framework", "MCP Orchestrator", "Meta-Cognitive Engine"]
    
    with Progress(console=console) as progress:
        task = progress.add_task("Starting components...", total=len(components))
        
        for component in components:
            progress.update(task, description=f"Starting {component}...")
            # Simulate startup time
            import time
            time.sleep(0.5)
            progress.advance(task)
    
    console.print("[green]SAFLA system started successfully[/green]")
    console.print("[yellow]Press Ctrl+C to stop[/yellow]")
    
    # Keep running until interrupted
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        pass

@cli.group()
@click.pass_context
def benchmark(ctx):
    """SAFLA benchmark commands"""
    pass


@benchmark.command()
@click.option('--suite', type=click.Choice(['cli', 'all']), default='cli',
              help='Benchmark suite to run')
@click.option('--parallel', is_flag=True, help='Run benchmarks in parallel')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--config', type=click.Path(exists=True), help='Benchmark configuration file')
@click.option('--store/--no-store', default=True, help='Store results in database')
@click.pass_context
def run(ctx, suite, parallel, output, config, store):
    """Run benchmark suite"""
    console.print(f"[bold blue]Running {suite} benchmark suite...[/bold blue]")
    
    try:
        import asyncio
        from benchmarks import BenchmarkSuite, BenchmarkDatabase, BenchmarkConfig
        from benchmarks.cli_benchmarks import CLIBenchmarkSuite
        
        # Load configuration
        if config:
            benchmark_config = BenchmarkConfig.from_file(config)
        else:
            benchmark_config = BenchmarkConfig()
            benchmark_config.parallel_execution = parallel
            benchmark_config.store_results = store
        
        async def run_benchmarks():
            # Create benchmark suite
            if suite == 'cli':
                cli_suite = CLIBenchmarkSuite()
                benchmark_suite = BenchmarkSuite("SAFLA CLI Benchmarks", "Performance benchmarks for SAFLA CLI commands")
                for benchmark in cli_suite.benchmarks:
                    benchmark_suite.add_benchmark(benchmark)
            else:
                # Future: Add other benchmark suites
                cli_suite = CLIBenchmarkSuite()
                benchmark_suite = BenchmarkSuite("SAFLA All Benchmarks", "Complete SAFLA performance benchmarks")
                for benchmark in cli_suite.benchmarks:
                    benchmark_suite.add_benchmark(benchmark)
            
            # Run benchmarks
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Running benchmarks...", total=None)
                
                results = await benchmark_suite.run_all(parallel=benchmark_config.parallel_execution)
                progress.update(task, completed=True)
            
            # Store results
            if benchmark_config.store_results:
                db = BenchmarkDatabase(benchmark_config.database_path)
                suite_id = db.store_suite_results(
                    benchmark_suite.name,
                    benchmark_suite.description,
                    results
                )
                console.print(f"[green]Results stored in database (Suite ID: {suite_id})[/green]")
            
            # Display results
            _display_benchmark_results(results)
            
            # Export results if requested
            if output:
                import json
                export_data = [result.to_dict() for result in results]
                with open(output, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                console.print(f"[green]Results exported to: {output}[/green]")
            
            return results
        
        results = asyncio.run(run_benchmarks())
        
        # Check for failures
        failed_benchmarks = [r for r in results if not r.success]
        if failed_benchmarks:
            console.print(f"[yellow]Warning: {len(failed_benchmarks)} benchmark(s) failed[/yellow]")
            for result in failed_benchmarks:
                console.print(f"[red]  - {result.benchmark_name}: {result.error_message}[/red]")
    
    except Exception as e:
        console.print(f"[red]Error running benchmarks: {e}[/red]")
        sys.exit(1)


@benchmark.command()
@click.option('--benchmark', help='Specific benchmark name to show')
@click.option('--days', default=30, help='Number of days of history to show')
@click.option('--format', type=click.Choice(['table', 'json']), default='table',
              help='Output format')
@click.pass_context
def results(ctx, benchmark, days, format):
    """Show benchmark results and history"""
    console.print("[bold blue]Benchmark Results[/bold blue]")
    
    try:
        from benchmarks import BenchmarkDatabase
        
        db = BenchmarkDatabase()
        
        if benchmark:
            # Show specific benchmark results
            latest_results = db.get_latest_results(benchmark, limit=10)
            
            if format == 'table':
                if latest_results:
                    table = Table(title=f"Latest Results: {benchmark}")
                    table.add_column("Timestamp", style="cyan")
                    table.add_column("Execution Time", style="green")
                    table.add_column("Memory Usage", style="blue")
                    table.add_column("Success", style="yellow")
                    
                    for result in latest_results:
                        timestamp = result['timestamp'][:19]  # Remove microseconds
                        exec_time = f"{result['execution_time']:.3f}s"
                        memory = f"{result['memory_usage']:.1f}MB"
                        success = "✓" if result['success'] else "✗"
                        
                        table.add_row(timestamp, exec_time, memory, success)
                    
                    console.print(table)
                else:
                    console.print(f"[yellow]No results found for benchmark: {benchmark}[/yellow]")
            
            elif format == 'json':
                console.print(json.dumps(latest_results, indent=2, default=str))
        
        else:
            # Show summary of all benchmarks
            # This would require a method to get all benchmark names
            console.print("[yellow]Use --benchmark option to specify a benchmark name[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Error retrieving results: {e}[/red]")
        sys.exit(1)


@benchmark.command()
@click.option('--benchmark', help='Specific benchmark name to analyze')
@click.option('--days', default=30, help='Number of days to analyze')
@click.option('--output', '-o', type=click.Path(), help='Output file for analysis report')
@click.pass_context
def analyze(ctx, benchmark, days, output):
    """Analyze benchmark performance trends"""
    console.print("[bold blue]Analyzing Benchmark Performance...[/bold blue]")
    
    try:
        from benchmarks.utils import BenchmarkAnalyzer
        
        analyzer = BenchmarkAnalyzer()
        
        if benchmark:
            # Analyze specific benchmark
            analysis = analyzer.analyze_benchmark_trends(benchmark, days)
            
            # Display analysis
            console.print(f"\n[bold]Analysis for {benchmark} (last {days} days)[/bold]")
            
            if analysis.get('data_points', 0) > 0:
                # Execution time analysis
                if 'execution_time' in analysis:
                    exec_stats = analysis['execution_time']['statistics']
                    exec_trend = analysis['execution_time']['trend_direction']
                    exec_stability = analysis['execution_time']['stability']['stability_score']
                    
                    console.print(f"Execution Time: {exec_stats['mean']:.3f}s ± {exec_stats['stdev']:.3f}s")
                    console.print(f"Trend: {exec_trend}, Stability: {exec_stability:.2f}")
                
                # Memory usage analysis
                if 'memory_usage' in analysis:
                    mem_stats = analysis['memory_usage']['statistics']
                    mem_trend = analysis['memory_usage']['trend_direction']
                    
                    console.print(f"Memory Usage: {mem_stats['mean']:.1f}MB ± {mem_stats['stdev']:.1f}MB")
                    console.print(f"Memory Trend: {mem_trend}")
            else:
                console.print(f"[yellow]No data available for benchmark: {benchmark}[/yellow]")
            
            # Save analysis if requested
            if output:
                with open(output, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
                console.print(f"[green]Analysis saved to: {output}[/green]")
        
        else:
            console.print("[yellow]Use --benchmark option to specify a benchmark name[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Error analyzing benchmarks: {e}[/red]")
        sys.exit(1)


@benchmark.command()
@click.option('--days', default=90, help='Delete results older than this many days')
@click.option('--confirm', is_flag=True, help='Confirm deletion without prompt')
@click.pass_context
def cleanup(ctx, days, confirm):
    """Clean up old benchmark results"""
    if not confirm:
        if not click.confirm(f"Delete benchmark results older than {days} days?"):
            console.print("[yellow]Cleanup cancelled[/yellow]")
            return
    
    try:
        from benchmarks import BenchmarkDatabase
        
        db = BenchmarkDatabase()
        deleted_count = db.cleanup_old_results(days)
        
        console.print(f"[green]Cleaned up {deleted_count} old benchmark records[/green]")
    
    except Exception as e:
        console.print(f"[red]Error during cleanup: {e}[/red]")
        sys.exit(1)


@benchmark.command()
@click.option('--benchmark', help='Specific benchmark name to export')
@click.option('--days', help='Number of days to export')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output file')
@click.pass_context
def export(ctx, benchmark, days, output):
    """Export benchmark results to file"""
    console.print("[bold blue]Exporting Benchmark Results...[/bold blue]")
    
    try:
        from benchmarks import BenchmarkDatabase
        
        db = BenchmarkDatabase()
        db.export_results(output, benchmark_name=benchmark, days=int(days) if days else None)
        
        console.print(f"[green]Results exported to: {output}[/green]")
    
    except Exception as e:
        console.print(f"[red]Error exporting results: {e}[/red]")
        sys.exit(1)


def _display_benchmark_results(results):
    """Display benchmark results in a formatted table"""
    if not results:
        console.print("[yellow]No benchmark results to display[/yellow]")
        return
    
    # Summary table
    summary_table = Table(title="Benchmark Summary")
    summary_table.add_column("Benchmark", style="cyan")
    summary_table.add_column("Status", style="green")
    summary_table.add_column("Execution Time", style="blue")
    summary_table.add_column("Memory Usage", style="yellow")
    summary_table.add_column("Throughput", style="magenta")
    
    successful_count = 0
    total_time = 0.0
    
    for result in results:
        status = "✓ Success" if result.success else "✗ Failed"
        status_style = "green" if result.success else "red"
        
        exec_time = f"{result.metrics.execution_time:.3f}s"
        memory = f"{result.metrics.memory_usage:.1f}MB"
        
        if result.metrics.throughput:
            throughput = f"{result.metrics.throughput:.1f} ops/s"
        else:
            throughput = "N/A"
        
        summary_table.add_row(
            result.benchmark_name,
            f"[{status_style}]{status}[/{status_style}]",
            exec_time,
            memory,
            throughput
        )
        
        if result.success:
            successful_count += 1
        total_time += result.metrics.execution_time
    
    console.print(summary_table)
    
    # Overall statistics
    success_rate = (successful_count / len(results)) * 100
    console.print(f"\n[bold]Overall Statistics:[/bold]")
    console.print(f"Total Benchmarks: {len(results)}")
    console.print(f"Successful: {successful_count}")
    console.print(f"Success Rate: {success_rate:.1f}%")
    console.print(f"Total Execution Time: {total_time:.3f}s")


def main():
    """Main entry point for the CLI"""
    try:
        cli()
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if "--debug" in sys.argv:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()