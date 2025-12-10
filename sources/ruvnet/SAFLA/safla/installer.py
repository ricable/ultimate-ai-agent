"""
SAFLA Interactive Installer

This module provides an interactive installation experience with rich UI components,
progress tracking, and comprehensive system validation.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

import click
from rich.console import Console
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm, Prompt, IntPrompt
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.text import Text
from rich.tree import Tree

from safla.utils.config import SAFLAConfig
from safla.utils.validation import validate_installation, validate_dependencies, check_gpu_availability
from safla.utils.logging import setup_logging
from safla.exceptions import SAFLAError


console = Console()


class SAFLAInstaller:
    """Interactive SAFLA installer with rich UI."""
    
    def __init__(self):
        self.console = console
        self.config = SAFLAConfig()
        self.installation_path = Path.cwd()
        self.install_options = {
            "gpu_support": False,
            "dev_dependencies": False,
            "create_config": True,
            "setup_directories": True,
            "validate_installation": True
        }
    
    def run_interactive_install(self) -> bool:
        """Run the interactive installation process."""
        try:
            self._show_welcome()
            
            if not self._check_prerequisites():
                return False
            
            self._gather_installation_options()
            self._configure_installation()
            
            if not self._confirm_installation():
                self.console.print("[yellow]Installation cancelled by user.[/yellow]")
                return False
            
            return self._perform_installation()
        
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Installation interrupted by user.[/yellow]")
            return False
        except Exception as e:
            self.console.print(f"\n[red]Installation failed: {e}[/red]")
            return False
    
    def _show_welcome(self):
        """Display welcome screen."""
        welcome_text = Text()
        welcome_text.append("SAFLA", style="bold blue")
        welcome_text.append(" - Self-Aware Feedback Loop Algorithm\n", style="bold")
        welcome_text.append("Interactive Installation Wizard", style="italic")
        
        welcome_panel = Panel(
            Align.center(welcome_text),
            title="Welcome",
            border_style="blue",
            padding=(1, 2)
        )
        
        self.console.print(welcome_panel)
        self.console.print()
    
    def _check_prerequisites(self) -> bool:
        """Check system prerequisites."""
        self.console.print("[bold blue]Checking Prerequisites...[/bold blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            # Check Python version
            task = progress.add_task("Checking Python version...", total=None)
            validation_results = validate_installation()
            progress.update(task, completed=True)
            
            # Check dependencies
            task = progress.add_task("Checking dependencies...", total=None)
            dep_errors = validate_dependencies()
            progress.update(task, completed=True)
            
            # Check GPU availability
            task = progress.add_task("Checking GPU support...", total=None)
            gpu_info = check_gpu_availability()
            progress.update(task, completed=True)
        
        # Display results
        self._display_prerequisite_results(validation_results, dep_errors, gpu_info)
        
        # Check if we can continue
        if validation_results["errors"]:
            self.console.print("[red]Critical errors found. Please resolve them before continuing.[/red]")
            return False
        
        if dep_errors and not any("WARNING" in error for error in dep_errors):
            self.console.print("[red]Missing required dependencies. Please install them first.[/red]")
            return False
        
        return True
    
    def _display_prerequisite_results(self, validation_results, dep_errors, gpu_info):
        """Display prerequisite check results."""
        
        # Create results table
        table = Table(title="Prerequisite Check Results")
        table.add_column("Component", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Details")
        
        # Python version
        if validation_results["valid"]:
            table.add_row("Python", "✅", f"Version {validation_results['system_info']['python_version']}")
        else:
            table.add_row("Python", "❌", "Version requirements not met")
        
        # Dependencies
        critical_deps = [e for e in dep_errors if not e.startswith("WARNING")]
        if not critical_deps:
            table.add_row("Dependencies", "✅", "All required packages available")
        else:
            table.add_row("Dependencies", "❌", f"{len(critical_deps)} missing packages")
        
        # GPU support
        if gpu_info["cuda_available"]:
            table.add_row("GPU Support", "✅", f"CUDA {gpu_info['cuda_version']} with {gpu_info['gpu_count']} GPU(s)")
        else:
            table.add_row("GPU Support", "⚠️", "Not available (CPU-only mode)")
        
        # System resources
        if "memory_total" in validation_results["system_info"]:
            memory_gb = int(validation_results["system_info"]["memory_total"].replace("MB", "")) / 1024
            if memory_gb >= 2:
                table.add_row("Memory", "✅", f"{memory_gb:.1f}GB available")
            else:
                table.add_row("Memory", "⚠️", f"{memory_gb:.1f}GB (minimum 2GB recommended)")
        
        self.console.print(table)
        self.console.print()
    
    def _gather_installation_options(self):
        """Gather installation options from user."""
        self.console.print("[bold blue]Installation Options[/bold blue]")
        
        # GPU support
        gpu_info = check_gpu_availability()
        if gpu_info["cuda_available"]:
            self.install_options["gpu_support"] = Confirm.ask(
                "Install GPU support (CUDA)?", 
                default=True
            )
        
        # Development dependencies
        self.install_options["dev_dependencies"] = Confirm.ask(
            "Install development dependencies (testing, linting, etc.)?",
            default=False
        )
        
        # Configuration setup
        self.install_options["create_config"] = Confirm.ask(
            "Create default configuration file?",
            default=True
        )
        
        # Directory setup
        self.install_options["setup_directories"] = Confirm.ask(
            "Create data and configuration directories?",
            default=True
        )
        
        self.console.print()
    
    def _configure_installation(self):
        """Configure installation settings."""
        self.console.print("[bold blue]Installation Configuration[/bold blue]")
        
        # Installation path
        default_path = str(self.installation_path)
        path_input = Prompt.ask(
            "Installation directory",
            default=default_path
        )
        self.installation_path = Path(path_input)
        
        # Configuration options
        if self.install_options["create_config"]:
            self._configure_safla_settings()
        
        self.console.print()
    
    def _configure_safla_settings(self):
        """Configure SAFLA-specific settings."""
        self.console.print("[dim]Configuring SAFLA settings...[/dim]")
        
        # Memory settings
        max_memories = IntPrompt.ask(
            "Maximum number of memories to store",
            default=self.config.memory.max_memories
        )
        self.config.memory.max_memories = max_memories
        
        # Safety settings
        memory_limit_gb = IntPrompt.ask(
            "Memory limit (GB)",
            default=self.config.safety.memory_limit // (1024 * 1024 * 1024)
        )
        self.config.safety.memory_limit = memory_limit_gb * 1024 * 1024 * 1024
        
        # Data directory
        data_dir = Prompt.ask(
            "Data directory",
            default=self.config.data_dir
        )
        self.config.data_dir = data_dir
    
    def _confirm_installation(self) -> bool:
        """Show installation summary and confirm."""
        self.console.print("[bold blue]Installation Summary[/bold blue]")
        
        # Create summary table
        summary_table = Table(title="Installation Plan")
        summary_table.add_column("Setting", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Installation Path", str(self.installation_path))
        summary_table.add_row("GPU Support", "Yes" if self.install_options["gpu_support"] else "No")
        summary_table.add_row("Dev Dependencies", "Yes" if self.install_options["dev_dependencies"] else "No")
        summary_table.add_row("Create Config", "Yes" if self.install_options["create_config"] else "No")
        summary_table.add_row("Setup Directories", "Yes" if self.install_options["setup_directories"] else "No")
        
        if self.install_options["create_config"]:
            summary_table.add_row("Max Memories", str(self.config.memory.max_memories))
            summary_table.add_row("Memory Limit", f"{self.config.safety.memory_limit // (1024*1024*1024)}GB")
            summary_table.add_row("Data Directory", self.config.data_dir)
        
        self.console.print(summary_table)
        self.console.print()
        
        return Confirm.ask("Proceed with installation?", default=True)
    
    def _perform_installation(self) -> bool:
        """Perform the actual installation."""
        self.console.print("[bold blue]Installing SAFLA...[/bold blue]")
        
        installation_steps = [
            ("Creating directories", self._create_directories),
            ("Installing dependencies", self._install_dependencies),
            ("Setting up configuration", self._setup_configuration),
            ("Validating installation", self._validate_final_installation),
            ("Creating shortcuts", self._create_shortcuts)
        ]
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            
            main_task = progress.add_task("Installing SAFLA...", total=len(installation_steps))
            
            for step_name, step_func in installation_steps:
                step_task = progress.add_task(step_name, total=None)
                
                try:
                    success = step_func(progress, step_task)
                    if not success:
                        progress.update(step_task, description=f"❌ {step_name}")
                        self.console.print(f"[red]Installation failed at step: {step_name}[/red]")
                        return False
                    
                    progress.update(step_task, description=f"✅ {step_name}", completed=True)
                    progress.advance(main_task)
                    
                except Exception as e:
                    progress.update(step_task, description=f"❌ {step_name}")
                    self.console.print(f"[red]Error in {step_name}: {e}[/red]")
                    return False
        
        self._show_installation_complete()
        return True
    
    def _create_directories(self, progress, task) -> bool:
        """Create necessary directories."""
        if not self.install_options["setup_directories"]:
            return True
        
        try:
            self.config.create_directories()
            
            # Create additional directories
            directories = [
                self.installation_path / "logs",
                self.installation_path / "backups",
                self.installation_path / "plugins"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            
            return True
        
        except Exception as e:
            self.console.print(f"[red]Failed to create directories: {e}[/red]")
            return False
    
    def _install_dependencies(self, progress, task) -> bool:
        """Install Python dependencies."""
        try:
            # Determine which dependencies to install
            deps = ["safla"]
            
            if self.install_options["gpu_support"]:
                deps.append("safla[gpu]")
            
            if self.install_options["dev_dependencies"]:
                deps.append("safla[dev]")
            
            # Install dependencies
            for dep in deps:
                cmd = [sys.executable, "-m", "pip", "install", dep]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.console.print(f"[red]Failed to install {dep}: {result.stderr}[/red]")
                    return False
            
            return True
        
        except Exception as e:
            self.console.print(f"[red]Dependency installation failed: {e}[/red]")
            return False
    
    def _setup_configuration(self, progress, task) -> bool:
        """Set up configuration files."""
        if not self.install_options["create_config"]:
            return True
        
        try:
            # Save main configuration
            config_path = self.installation_path / "safla_config.json"
            self.config.save_to_file(str(config_path))
            
            # Create environment file template
            env_path = self.installation_path / ".env.example"
            self._create_env_template(env_path)
            
            # Create MCP configuration template
            mcp_config_path = self.installation_path / ".roo" / "mcp.json"
            self._create_mcp_config_template(mcp_config_path)
            
            return True
        
        except Exception as e:
            self.console.print(f"[red]Configuration setup failed: {e}[/red]")
            return False
    
    def _create_env_template(self, env_path: Path):
        """Create environment file template."""
        env_content = """# SAFLA Environment Configuration

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
        
        with open(env_path, 'w') as f:
            f.write(env_content)
    
    def _create_mcp_config_template(self, mcp_path: Path):
        """Create MCP configuration template."""
        mcp_config = {
            "mcpServers": {
                "context7": {
                    "command": "npx",
                    "args": ["-y", "@upstash/context7-mcp"]
                }
            }
        }
        
        mcp_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(mcp_path, 'w') as f:
            json.dump(mcp_config, f, indent=2)
    
    def _validate_final_installation(self, progress, task) -> bool:
        """Validate the completed installation."""
        try:
            validation_results = validate_installation()
            return validation_results["valid"]
        
        except Exception as e:
            self.console.print(f"[red]Installation validation failed: {e}[/red]")
            return False
    
    def _create_shortcuts(self, progress, task) -> bool:
        """Create command shortcuts and aliases."""
        try:
            # Create a simple startup script
            startup_script = self.installation_path / "start_safla.py"
            
            script_content = """#!/usr/bin/env python3
\"\"\"
SAFLA Startup Script
\"\"\"

import sys
from pathlib import Path

# Add SAFLA to path
safla_path = Path(__file__).parent
sys.path.insert(0, str(safla_path))

from safla.cli import main

if __name__ == "__main__":
    main()
"""
            
            with open(startup_script, 'w') as f:
                f.write(script_content)
            
            # Make executable on Unix systems
            if sys.platform != "win32":
                startup_script.chmod(0o755)
            
            return True
        
        except Exception as e:
            self.console.print(f"[red]Failed to create shortcuts: {e}[/red]")
            return False
    
    def _show_installation_complete(self):
        """Show installation completion message."""
        
        # Create completion tree
        tree = Tree("🎉 SAFLA Installation Complete!")
        
        tree.add("✅ Core system installed")
        tree.add("✅ Configuration files created")
        tree.add("✅ Directories set up")
        tree.add("✅ Dependencies installed")
        
        next_steps = tree.add("📋 Next Steps")
        next_steps.add("Run 'safla validate' to verify installation")
        next_steps.add("Run 'safla info' to see system information")
        next_steps.add("Run 'safla start' to start the SAFLA system")
        next_steps.add("Check the documentation for usage examples")
        
        completion_panel = Panel(
            tree,
            title="Installation Complete",
            border_style="green",
            padding=(1, 2)
        )
        
        self.console.print(completion_panel)


@click.command()
@click.option('--non-interactive', is_flag=True, help='Run non-interactive installation')
@click.option('--gpu', is_flag=True, help='Install with GPU support')
@click.option('--dev', is_flag=True, help='Install development dependencies')
def main(non_interactive, gpu, dev):
    """SAFLA Interactive Installer"""
    
    if non_interactive:
        console.print("[yellow]Non-interactive installation not yet implemented[/yellow]")
        return
    
    installer = SAFLAInstaller()
    
    # Set options from command line
    if gpu:
        installer.install_options["gpu_support"] = True
    if dev:
        installer.install_options["dev_dependencies"] = True
    
    success = installer.run_interactive_install()
    
    if success:
        console.print("[green]Installation completed successfully![/green]")
        sys.exit(0)
    else:
        console.print("[red]Installation failed![/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()