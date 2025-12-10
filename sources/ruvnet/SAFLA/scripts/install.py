#!/usr/bin/env python3
"""
SAFLA Installation Script

This script provides a comprehensive installation process for SAFLA,
including dependency management, configuration setup, and validation.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional

# Add the parent directory to the path to import SAFLA modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    
    class Console:
        def print(self, *args, **kwargs):
            print(*args)


console = Console()


def run_command(cmd: List[str], description: str = "", check: bool = True) -> subprocess.CompletedProcess:
    """Run a command with optional progress indication."""
    if RICH_AVAILABLE:
        console.print(f"[blue]Running:[/blue] {description or ' '.join(cmd)}")
    else:
        print(f"Running: {description or ' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        
        if result.returncode == 0:
            if RICH_AVAILABLE:
                console.print(f"[green]✓[/green] {description or 'Command completed'}")
            else:
                print(f"✓ {description or 'Command completed'}")
        else:
            if RICH_AVAILABLE:
                console.print(f"[red]✗[/red] {description or 'Command failed'}")
                console.print(f"[red]Error:[/red] {result.stderr}")
            else:
                print(f"✗ {description or 'Command failed'}")
                print(f"Error: {result.stderr}")
        
        return result
    
    except subprocess.CalledProcessError as e:
        if RICH_AVAILABLE:
            console.print(f"[red]✗ Command failed with exit code {e.returncode}[/red]")
            console.print(f"[red]Error:[/red] {e.stderr}")
        else:
            print(f"✗ Command failed with exit code {e.returncode}")
            print(f"Error: {e.stderr}")
        raise
    except FileNotFoundError:
        if RICH_AVAILABLE:
            console.print(f"[red]✗ Command not found: {cmd[0]}[/red]")
        else:
            print(f"✗ Command not found: {cmd[0]}")
        raise


def check_python_version():
    """Check if Python version meets requirements."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Checking Python version...[/bold blue]")
    else:
        print("Checking Python version...")
    
    version = sys.version_info
    if version < (3, 8):
        if RICH_AVAILABLE:
            console.print(f"[red]Python 3.8+ required, but {version.major}.{version.minor} found[/red]")
        else:
            print(f"Python 3.8+ required, but {version.major}.{version.minor} found")
        sys.exit(1)
    
    if RICH_AVAILABLE:
        console.print(f"[green]✓ Python {version.major}.{version.minor}.{version.micro}[/green]")
    else:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")


def install_build_dependencies():
    """Install build dependencies."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Installing build dependencies...[/bold blue]")
    else:
        print("Installing build dependencies...")
    
    build_deps = ["setuptools", "wheel", "build"]
    
    for dep in build_deps:
        run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", dep],
            f"Installing {dep}"
        )


def install_package(dev: bool = False, gpu: bool = False, editable: bool = True):
    """Install the SAFLA package."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Installing SAFLA package...[/bold blue]")
    else:
        print("Installing SAFLA package...")
    
    # Build the installation command
    cmd = [sys.executable, "-m", "pip", "install"]
    
    if editable:
        cmd.append("-e")
    
    # Add the current directory
    cmd.append(".")
    
    # Add optional dependencies
    extras = []
    if dev:
        extras.append("dev")
    if gpu:
        extras.append("gpu")
    
    if extras:
        cmd[-1] = f".[{','.join(extras)}]"
    
    run_command(cmd, "Installing SAFLA package")


def run_tests():
    """Run the test suite."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Running tests...[/bold blue]")
    else:
        print("Running tests...")
    
    try:
        run_command(
            [sys.executable, "-m", "pytest", "tests/", "-v"],
            "Running test suite"
        )
    except subprocess.CalledProcessError:
        if RICH_AVAILABLE:
            console.print("[yellow]Some tests failed, but installation can continue[/yellow]")
        else:
            print("Some tests failed, but installation can continue")


def validate_installation():
    """Validate the installation."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Validating installation...[/bold blue]")
    else:
        print("Validating installation...")
    
    try:
        # Try to import SAFLA
        import safla
        
        if RICH_AVAILABLE:
            console.print(f"[green]✓ SAFLA {safla.__version__} imported successfully[/green]")
        else:
            print(f"✓ SAFLA {safla.__version__} imported successfully")
        
        # Try to run validation command
        run_command(
            [sys.executable, "-m", "safla.cli", "validate"],
            "Running SAFLA validation",
            check=False  # Don't fail if validation has warnings
        )
        
    except ImportError as e:
        if RICH_AVAILABLE:
            console.print(f"[red]✗ Failed to import SAFLA: {e}[/red]")
        else:
            print(f"✗ Failed to import SAFLA: {e}")
        sys.exit(1)


def create_example_config():
    """Create example configuration files."""
    if RICH_AVAILABLE:
        console.print("[bold blue]Creating example configuration...[/bold blue]")
    else:
        print("Creating example configuration...")
    
    try:
        from safla.utils.config import SAFLAConfig
        
        # Create example configuration
        config = SAFLAConfig()
        config.save_to_file("safla_config_example.json")
        
        if RICH_AVAILABLE:
            console.print("[green]✓ Example configuration created: safla_config_example.json[/green]")
        else:
            print("✓ Example configuration created: safla_config_example.json")
        
        # Create .env example
        env_content = """# SAFLA Environment Configuration Example
# Copy this file to .env and modify as needed

# Memory Configuration
SAFLA_VECTOR_DIMENSIONS=512,768,1024,1536
SAFLA_MAX_MEMORIES=10000
SAFLA_SIMILARITY_THRESHOLD=0.8

# Safety Configuration
SAFLA_MEMORY_LIMIT=1000000000
SAFLA_CPU_LIMIT=0.9
SAFLA_SAFETY_MONITORING_INTERVAL=1.0

# MCP Configuration
SAFLA_MCP_TIMEOUT=30
SAFLA_MCP_MAX_RETRIES=3
SAFLA_MCP_HEALTH_CHECK_INTERVAL=60

# General Configuration
SAFLA_DEBUG=false
SAFLA_LOG_LEVEL=INFO
SAFLA_DATA_DIR=./data
SAFLA_CONFIG_DIR=./.roo
"""
        
        with open(".env.example", "w") as f:
            f.write(env_content)
        
        if RICH_AVAILABLE:
            console.print("[green]✓ Environment example created: .env.example[/green]")
        else:
            print("✓ Environment example created: .env.example")
    
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[yellow]Warning: Could not create example config: {e}[/yellow]")
        else:
            print(f"Warning: Could not create example config: {e}")


def show_completion_message():
    """Show installation completion message."""
    if RICH_AVAILABLE:
        completion_table = Table(title="🎉 SAFLA Installation Complete!")
        completion_table.add_column("Next Steps", style="cyan")
        completion_table.add_column("Command", style="green")
        
        completion_table.add_row("Validate installation", "safla validate")
        completion_table.add_row("Show system info", "safla info")
        completion_table.add_row("Initialize config", "safla init-config")
        completion_table.add_row("Start SAFLA", "safla start")
        completion_table.add_row("Run interactive installer", "safla-install")
        
        console.print(completion_table)
        
        console.print(Panel(
            "[bold green]Installation completed successfully![/bold green]\n\n"
            "You can now use SAFLA by running the commands shown above.\n"
            "For more information, check the documentation or run 'safla --help'.",
            title="Success",
            border_style="green"
        ))
    else:
        print("\n🎉 SAFLA Installation Complete!")
        print("\nNext Steps:")
        print("  Validate installation: safla validate")
        print("  Show system info: safla info")
        print("  Initialize config: safla init-config")
        print("  Start SAFLA: safla start")
        print("  Run interactive installer: safla-install")
        print("\nInstallation completed successfully!")


def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="SAFLA Installation Script")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--gpu", action="store_true", help="Install GPU support")
    parser.add_argument("--no-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--no-validation", action="store_true", help="Skip installation validation")
    parser.add_argument("--no-examples", action="store_true", help="Skip creating example files")
    
    args = parser.parse_args()
    
    try:
        # Show welcome message
        if RICH_AVAILABLE:
            console.print(Panel(
                "[bold blue]SAFLA Installation Script[/bold blue]\n"
                "This script will install SAFLA and its dependencies.",
                title="Welcome",
                border_style="blue"
            ))
        else:
            print("SAFLA Installation Script")
            print("This script will install SAFLA and its dependencies.")
        
        # Check Python version
        check_python_version()
        
        # Install build dependencies
        install_build_dependencies()
        
        # Install the package
        install_package(dev=args.dev, gpu=args.gpu)
        
        # Run tests (unless skipped)
        if not args.no_tests:
            try:
                run_tests()
            except subprocess.CalledProcessError:
                if RICH_AVAILABLE:
                    console.print("[yellow]Tests failed, but continuing with installation[/yellow]")
                else:
                    print("Tests failed, but continuing with installation")
        
        # Validate installation (unless skipped)
        if not args.no_validation:
            validate_installation()
        
        # Create example files (unless skipped)
        if not args.no_examples:
            create_example_config()
        
        # Show completion message
        show_completion_message()
        
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[yellow]Installation interrupted by user[/yellow]")
        else:
            print("\nInstallation interrupted by user")
        sys.exit(1)
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"\n[red]Installation failed: {e}[/red]")
        else:
            print(f"\nInstallation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()