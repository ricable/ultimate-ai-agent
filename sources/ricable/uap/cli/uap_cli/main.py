# UAP CLI Main Application
"""
Main command-line interface for UAP platform management.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Any, Dict
import argparse
import logging
import json
import yaml
from datetime import datetime

# Add parent directories to path for SDK imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sdk"))

from uap_sdk import Configuration, UAPClient, PluginManager
from uap_sdk.exceptions import UAPException, UAPConnectionError, UAPAuthError

from .commands import (
    AuthCommands,
    AgentCommands, 
    DeploymentCommands,
    PluginCommands,
    ConfigCommands,
    ProjectCommands,
    MonitoringCommands
)
from .debug_commands import DebugCommands

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UAPCLIApplication:
    """Main UAP CLI application."""
    
    def __init__(self):
        self.config = None
        self.client = None
        self.plugin_manager = None
        
        # Command handlers
        self.auth_commands = AuthCommands(self)
        self.agent_commands = AgentCommands(self)
        self.deployment_commands = DeploymentCommands(self)
        self.plugin_commands = PluginCommands(self)
        self.config_commands = ConfigCommands(self)
        self.project_commands = ProjectCommands(self)
        self.monitoring_commands = MonitoringCommands(self)
        self.debug_commands = DebugCommands(self)
        
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser."""
        parser = argparse.ArgumentParser(
            prog='uap',
            description='UAP - Unified Agentic Platform CLI',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  uap auth login                    # Login to UAP
  uap agent create my-agent         # Create a new agent
  uap deploy start                  # Start local deployment
  uap plugin list                   # List available plugins
  uap config show                   # Show current configuration
  
For more help on specific commands, use:
  uap <command> --help
            """
        )
        
        parser.add_argument('--version', action='version', version='UAP CLI 1.0.0')
        parser.add_argument('--config', '-c', help='Configuration file path')
        parser.add_argument('--profile', '-p', help='Configuration profile to use')
        parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
        parser.add_argument('--quiet', '-q', action='store_true', help='Suppress non-essential output')
        parser.add_argument('--format', choices=['json', 'yaml', 'table'], default='table', help='Output format')
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Auth commands
        auth_parser = subparsers.add_parser('auth', help='Authentication commands')
        self.auth_commands.add_subcommands(auth_parser)
        
        # Agent commands
        agent_parser = subparsers.add_parser('agent', help='Agent management commands')
        self.agent_commands.add_subcommands(agent_parser)
        
        # Deployment commands
        deploy_parser = subparsers.add_parser('deploy', help='Deployment commands')
        self.deployment_commands.add_subcommands(deploy_parser)
        
        # Plugin commands
        plugin_parser = subparsers.add_parser('plugin', help='Plugin management commands')
        self.plugin_commands.add_subcommands(plugin_parser)
        
        # Configuration commands
        config_parser = subparsers.add_parser('config', help='Configuration management')
        self.config_commands.add_subcommands(config_parser)
        
        # Project commands
        project_parser = subparsers.add_parser('project', help='Project scaffolding commands')
        self.project_commands.add_subcommands(project_parser)
        
        # Monitoring commands
        monitor_parser = subparsers.add_parser('monitor', help='Monitoring and status commands')
        self.monitoring_commands.add_subcommands(monitor_parser)
        
        # Debug commands
        debug_parser = subparsers.add_parser('debug', help='Advanced debugging and diagnostics')
        self.debug_commands.add_subcommands(debug_parser)
        
        return parser
    
    async def initialize(self, args: argparse.Namespace) -> None:
        """Initialize the CLI application."""
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        elif args.quiet:
            logging.getLogger().setLevel(logging.WARNING)
        
        # Load configuration
        try:
            if args.config:
                self.config = Configuration(config_file=args.config)
            elif args.profile:
                from uap_sdk.config import ConfigurationProfile
                profile_manager = ConfigurationProfile()
                self.config = profile_manager.load_profile(args.profile)
            else:
                self.config = Configuration()
            
            # Validate configuration
            self.config.validate()
            
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            sys.exit(1)
        
        # Initialize client
        self.client = UAPClient(self.config)
        
        # Initialize plugin manager
        self.plugin_manager = PluginManager(self.config)
        await self.plugin_manager.discover_plugins()
    
    def format_output(self, data: Any, format_type: str = "table") -> str:
        """Format output according to the specified format."""
        if format_type == "json":
            return json.dumps(data, indent=2, default=str)
        elif format_type == "yaml":
            return yaml.dump(data, default_flow_style=False, indent=2)
        elif format_type == "table":
            return self._format_table(data)
        else:
            return str(data)
    
    def _format_table(self, data: Any) -> str:
        """Format data as a table."""
        if isinstance(data, dict):
            if not data:
                return "No data"
            
            max_key_len = max(len(str(k)) for k in data.keys())
            lines = []
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, default=str)
                lines.append(f"{str(key).ljust(max_key_len)} : {value}")
            return "\n".join(lines)
        
        elif isinstance(data, list):
            if not data:
                return "No items"
            
            if all(isinstance(item, dict) for item in data):
                # Table format for list of dicts
                if not data:
                    return "No items"
                
                headers = set()
                for item in data:
                    headers.update(item.keys())
                headers = sorted(headers)
                
                # Calculate column widths
                col_widths = {}
                for header in headers:
                    col_widths[header] = len(str(header))
                    for item in data:
                        if header in item:
                            col_widths[header] = max(col_widths[header], len(str(item[header])))
                
                # Build table
                lines = []
                
                # Header row
                header_row = " | ".join(str(h).ljust(col_widths[h]) for h in headers)
                lines.append(header_row)
                lines.append("-" * len(header_row))
                
                # Data rows
                for item in data:
                    row = " | ".join(str(item.get(h, "")).ljust(col_widths[h]) for h in headers)
                    lines.append(row)
                
                return "\n".join(lines)
            else:
                # Simple list
                return "\n".join(f"- {item}" for item in data)
        
        else:
            return str(data)
    
    def print_output(self, data: Any, format_type: str = "table") -> None:
        """Print formatted output."""
        output = self.format_output(data, format_type)
        print(output)
    
    def print_error(self, message: str) -> None:
        """Print error message."""
        print(f"Error: {message}", file=sys.stderr)
    
    def print_success(self, message: str) -> None:
        """Print success message."""
        print(f"✓ {message}")
    
    def print_warning(self, message: str) -> None:
        """Print warning message."""
        print(f"⚠ {message}")
    
    async def handle_command(self, args: argparse.Namespace) -> int:
        """Handle the parsed command."""
        try:
            # Route to appropriate command handler
            if args.command == 'auth':
                return await self.auth_commands.handle_command(args)
            elif args.command == 'agent':
                return await self.agent_commands.handle_command(args)
            elif args.command == 'deploy':
                return await self.deployment_commands.handle_command(args)
            elif args.command == 'plugin':
                return await self.plugin_commands.handle_command(args)
            elif args.command == 'config':
                return await self.config_commands.handle_command(args)
            elif args.command == 'project':
                return await self.project_commands.handle_command(args)
            elif args.command == 'monitor':
                return await self.monitoring_commands.handle_command(args)
            elif args.command == 'debug':
                return await self.debug_commands.handle_command(args)
            else:
                self.parser.print_help()
                return 0
                
        except UAPAuthError as e:
            self.print_error(f"Authentication error: {e.message}")
            self.print_warning("Try running 'uap auth login' first")
            return 1
        except UAPConnectionError as e:
            self.print_error(f"Connection error: {e.message}")
            self.print_warning("Make sure UAP backend is running")
            return 1
        except UAPException as e:
            self.print_error(f"UAP error: {e.message}")
            return 1
        except KeyboardInterrupt:
            self.print_warning("Operation cancelled by user")
            return 1
        except Exception as e:
            self.print_error(f"Unexpected error: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    async def run(self, argv: Optional[List[str]] = None) -> int:
        """Run the CLI application."""
        args = self.parser.parse_args(argv)
        
        # Initialize application
        await self.initialize(args)
        
        # Handle command
        result = await self.handle_command(args)
        
        # Cleanup
        if self.client:
            await self.client.cleanup()
        if self.plugin_manager:
            await self.plugin_manager.cleanup()
        
        return result


async def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    app = UAPCLIApplication()
    return await app.run(argv)


def sync_main(argv: Optional[List[str]] = None) -> int:
    """Synchronous wrapper for main function."""
    return asyncio.run(main(argv))


if __name__ == "__main__":
    sys.exit(sync_main())