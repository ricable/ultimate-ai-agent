# UAP CLI Commands Module
"""
Command handlers for UAP CLI operations.
"""

import asyncio
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
import subprocess
import sys
import os
from datetime import datetime

from uap_sdk import UAPClient, Configuration, PluginManager, UAPAgent, CustomAgentBuilder, SimpleAgent
from uap_sdk.exceptions import UAPException, UAPConnectionError, UAPAuthError


class BaseCommand:
    """Base class for CLI commands."""
    
    def __init__(self, app):
        self.app = app
    
    @property
    def client(self) -> UAPClient:
        return self.app.client
    
    @property
    def config(self) -> Configuration:
        return self.app.config
    
    def add_subcommands(self, parser: argparse.ArgumentParser) -> None:
        """Add subcommands to the parser. Override in subclasses."""
        pass
    
    async def handle_command(self, args: argparse.Namespace) -> int:
        """Handle the command. Override in subclasses."""
        return 0


class AuthCommands(BaseCommand):
    """Authentication commands."""
    
    def add_subcommands(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest='auth_command', help='Authentication actions')
        
        # Login command
        login_parser = subparsers.add_parser('login', help='Login to UAP')
        login_parser.add_argument('--username', '-u', help='Username')
        login_parser.add_argument('--password', '-p', help='Password')
        login_parser.add_argument('--save-profile', help='Save as configuration profile')
        
        # Register command
        register_parser = subparsers.add_parser('register', help='Register new user')
        register_parser.add_argument('username', help='Username')
        register_parser.add_argument('email', help='Email address')
        register_parser.add_argument('--password', '-p', help='Password')
        register_parser.add_argument('--roles', nargs='+', default=['user'], help='User roles')
        
        # Logout command
        subparsers.add_parser('logout', help='Logout and clear tokens')
        
        # Status command
        subparsers.add_parser('status', help='Show authentication status')
    
    async def handle_command(self, args: argparse.Namespace) -> int:
        if args.auth_command == 'login':
            return await self._handle_login(args)
        elif args.auth_command == 'register':
            return await self._handle_register(args)
        elif args.auth_command == 'logout':
            return await self._handle_logout(args)
        elif args.auth_command == 'status':
            return await self._handle_status(args)
        else:
            self.app.print_error("Unknown auth command")
            return 1
    
    async def _handle_login(self, args: argparse.Namespace) -> int:
        try:
            username = args.username or input("Username: ")
            
            if args.password:
                password = args.password
            else:
                import getpass
                password = getpass.getpass("Password: ")
            
            result = await self.client.login(username, password)
            
            self.app.print_success(f"Successfully logged in as {username}")
            
            if args.save_profile:
                from uap_sdk.config import ConfigurationProfile
                profile_manager = ConfigurationProfile()
                profile_manager.create_profile(args.save_profile, self.config)
                self.app.print_success(f"Profile '{args.save_profile}' saved")
            
            if args.format != 'table':
                self.app.print_output(result, args.format)
            
            return 0
            
        except UAPAuthError as e:
            self.app.print_error(f"Authentication failed: {e.message}")
            return 1
        except Exception as e:
            self.app.print_error(f"Login failed: {str(e)}")
            return 1
    
    async def _handle_register(self, args: argparse.Namespace) -> int:
        try:
            if args.password:
                password = args.password
            else:
                import getpass
                password = getpass.getpass("Password: ")
                confirm_password = getpass.getpass("Confirm password: ")
                if password != confirm_password:
                    self.app.print_error("Passwords do not match")
                    return 1
            
            result = await self.client.auth.register(args.username, password, args.email, args.roles)
            
            self.app.print_success(f"Successfully registered user {args.username}")
            self.app.print_output(result, args.format)
            
            return 0
            
        except UAPAuthError as e:
            self.app.print_error(f"Registration failed: {e.message}")
            return 1
        except Exception as e:
            self.app.print_error(f"Registration failed: {str(e)}")
            return 1
    
    async def _handle_logout(self, args: argparse.Namespace) -> int:
        try:
            await self.client.auth.logout()
            self.app.print_success("Successfully logged out")
            return 0
        except Exception as e:
            self.app.print_error(f"Logout failed: {str(e)}")
            return 1
    
    async def _handle_status(self, args: argparse.Namespace) -> int:
        try:
            token = self.client.auth.get_access_token()
            
            status = {
                "authenticated": bool(token),
                "backend_url": self.config.get("backend_url"),
                "websocket_url": self.config.get("websocket_url")
            }
            
            if token:
                try:
                    # Try to get system status to verify token
                    system_status = await self.client.get_status()
                    status["token_valid"] = True
                    status["system_status"] = system_status.get("status", "unknown")
                except UAPAuthError:
                    status["token_valid"] = False
                    status["token_expired"] = True
            
            self.app.print_output(status, args.format)
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to get auth status: {str(e)}")
            return 1


class AgentCommands(BaseCommand):
    """Agent management commands."""
    
    def add_subcommands(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest='agent_command', help='Agent actions')
        
        # List agents
        subparsers.add_parser('list', help='List available agents')
        
        # Create agent
        create_parser = subparsers.add_parser('create', help='Create a new agent')
        create_parser.add_argument('agent_id', help='Agent identifier')
        create_parser.add_argument('--framework', choices=['simple', 'copilot', 'agno', 'mastra'], 
                                   default='simple', help='Agent framework')
        create_parser.add_argument('--config-file', help='Agent configuration file')
        create_parser.add_argument('--start', action='store_true', help='Start agent after creation')
        
        # Chat with agent
        chat_parser = subparsers.add_parser('chat', help='Chat with an agent')
        chat_parser.add_argument('agent_id', help='Agent identifier')
        chat_parser.add_argument('--message', '-m', help='Message to send')
        chat_parser.add_argument('--websocket', action='store_true', help='Use WebSocket connection')
        chat_parser.add_argument('--framework', help='Override framework routing')
        
        # Agent status
        status_parser = subparsers.add_parser('status', help='Get agent status')
        status_parser.add_argument('agent_id', nargs='?', help='Agent identifier (optional)')
        
        # Test agent
        test_parser = subparsers.add_parser('test', help='Test agent with sample messages')
        test_parser.add_argument('agent_id', help='Agent identifier')
        test_parser.add_argument('--count', type=int, default=5, help='Number of test messages')
    
    async def handle_command(self, args: argparse.Namespace) -> int:
        if args.agent_command == 'list':
            return await self._handle_list(args)
        elif args.agent_command == 'create':
            return await self._handle_create(args)
        elif args.agent_command == 'chat':
            return await self._handle_chat(args)
        elif args.agent_command == 'status':
            return await self._handle_status(args)
        elif args.agent_command == 'test':
            return await self._handle_test(args)
        else:
            self.app.print_error("Unknown agent command")
            return 1
    
    async def _handle_list(self, args: argparse.Namespace) -> int:
        try:
            status = await self.client.get_status()
            frameworks = status.get("frameworks", {})
            
            agents = []
            for framework_name, framework_info in frameworks.items():
                agents.append({
                    "framework": framework_name,
                    "status": framework_info.get("status", "unknown"),
                    "initialized": framework_info.get("initialized", False),
                    "capabilities": framework_info.get("capabilities", [])
                })
            
            if not agents:
                self.app.print_warning("No agents available")
                return 0
            
            self.app.print_output(agents, args.format)
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to list agents: {str(e)}")
            return 1
    
    async def _handle_create(self, args: argparse.Namespace) -> int:
        try:
            # Load configuration if provided
            agent_config = self.config
            if args.config_file:
                agent_config = Configuration(config_file=args.config_file)
            
            # Create agent using builder
            builder = CustomAgentBuilder(args.agent_id)
            
            if args.framework == 'simple':
                builder = builder.with_simple_framework()
            else:
                # For production frameworks, we'll create a mock since they require backend integration
                self.app.print_warning(f"Framework '{args.framework}' requires backend integration")
                builder = builder.with_simple_framework()
            
            builder = builder.with_config(agent_config)
            agent = builder.build()
            
            if args.start:
                await agent.start()
                self.app.print_success(f"Agent '{args.agent_id}' created and started")
            else:
                self.app.print_success(f"Agent '{args.agent_id}' created")
            
            agent_info = agent.get_status()
            self.app.print_output(agent_info, args.format)
            
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to create agent: {str(e)}")
            return 1
    
    async def _handle_chat(self, args: argparse.Namespace) -> int:
        try:
            if args.message:
                message = args.message
            else:
                message = input("Message: ")
            
            framework = args.framework or "auto"
            
            response = await self.client.chat(
                args.agent_id, 
                message, 
                framework=framework,
                use_websocket=args.websocket
            )
            
            if args.websocket and response.get("websocket"):
                self.app.print_success("Message sent via WebSocket")
            else:
                self.app.print_output(response, args.format)
            
            return 0
            
        except Exception as e:
            self.app.print_error(f"Chat failed: {str(e)}")
            return 1
    
    async def _handle_status(self, args: argparse.Namespace) -> int:
        try:
            if args.agent_id:
                # Get specific agent status via backend
                try:
                    response = await self.client.chat(args.agent_id, "/status", framework="auto")
                    self.app.print_output(response, args.format)
                except Exception:
                    self.app.print_warning(f"Could not get status for agent '{args.agent_id}' via backend")
                    self.app.print_warning("Agent may not exist or backend may be unavailable")
                    return 1
            else:
                # Get system status
                status = await self.client.get_status()
                self.app.print_output(status, args.format)
            
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to get status: {str(e)}")
            return 1
    
    async def _handle_test(self, args: argparse.Namespace) -> int:
        try:
            test_messages = [
                "Hello, how are you?",
                "What is the weather like?",
                "Can you help me with a task?",
                "Tell me a joke",
                "What are your capabilities?",
                "How do you work?",
                "What can you do for me?",
                "Explain machine learning",
                "What is the meaning of life?",
                "Goodbye"
            ]
            
            results = []
            for i in range(min(args.count, len(test_messages))):
                message = test_messages[i]
                self.app.print_success(f"Sending: {message}")
                
                try:
                    response = await self.client.chat(args.agent_id, message)
                    results.append({
                        "message": message,
                        "response": response.get("content", ""),
                        "success": True,
                        "response_time": response.get("response_time", 0)
                    })
                    print(f"Response: {response.get('content', '')}\n")
                except Exception as e:
                    results.append({
                        "message": message,
                        "error": str(e),
                        "success": False
                    })
                    self.app.print_error(f"Failed: {str(e)}\n")
            
            # Summary
            successful = sum(1 for r in results if r.get("success"))
            self.app.print_success(f"Test completed: {successful}/{len(results)} messages successful")
            
            if args.format != 'table':
                self.app.print_output(results, args.format)
            
            return 0
            
        except Exception as e:
            self.app.print_error(f"Test failed: {str(e)}")
            return 1


class DeploymentCommands(BaseCommand):
    """Deployment commands."""
    
    def add_subcommands(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest='deploy_command', help='Deployment actions')
        
        # Start local deployment
        start_parser = subparsers.add_parser('start', help='Start local UAP deployment')
        start_parser.add_argument('--backend-only', action='store_true', help='Start only backend')
        start_parser.add_argument('--frontend-only', action='store_true', help='Start only frontend')
        start_parser.add_argument('--port', type=int, help='Backend port (default: 8000)')
        
        # Stop deployment
        subparsers.add_parser('stop', help='Stop local UAP deployment')
        
        # Status
        subparsers.add_parser('status', help='Check deployment status')
        
        # Cloud deployment
        cloud_parser = subparsers.add_parser('cloud', help='Deploy to cloud')
        cloud_parser.add_argument('--provider', choices=['aws', 'gcp', 'azure'], help='Cloud provider')
        cloud_parser.add_argument('--config', help='Deployment configuration file')
    
    async def handle_command(self, args: argparse.Namespace) -> int:
        if args.deploy_command == 'start':
            return await self._handle_start(args)
        elif args.deploy_command == 'stop':
            return await self._handle_stop(args)
        elif args.deploy_command == 'status':
            return await self._handle_status(args)
        elif args.deploy_command == 'cloud':
            return await self._handle_cloud(args)
        else:
            self.app.print_error("Unknown deployment command")
            return 1
    
    async def _handle_start(self, args: argparse.Namespace) -> int:
        try:
            # Check if we're in a UAP project directory
            if not Path("devbox.json").exists():
                self.app.print_error("Not in a UAP project directory")
                self.app.print_warning("Run this command from the UAP project root")
                return 1
            
            # Use devbox for development
            if not args.frontend_only:
                self.app.print_success("Starting UAP backend...")
                backend_cmd = ["devbox", "run", "backend"]
                if args.port:
                    os.environ["BACKEND_PORT"] = str(args.port)
                
                # Start backend in background
                try:
                    subprocess.Popen(backend_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    await asyncio.sleep(2)  # Give it time to start
                    self.app.print_success("Backend started")
                except FileNotFoundError:
                    self.app.print_error("devbox not found. Please install devbox or start services manually")
                    return 1
            
            if not args.backend_only:
                self.app.print_success("Starting UAP frontend...")
                frontend_cmd = ["devbox", "run", "frontend"]
                
                try:
                    subprocess.Popen(frontend_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    await asyncio.sleep(2)  # Give it time to start
                    self.app.print_success("Frontend started")
                except FileNotFoundError:
                    self.app.print_error("devbox not found. Please install devbox or start services manually")
                    return 1
            
            self.app.print_success("UAP deployment started successfully")
            self.app.print_success("Backend: http://localhost:8000")
            self.app.print_success("Frontend: http://localhost:3000")
            
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to start deployment: {str(e)}")
            return 1
    
    async def _handle_stop(self, args: argparse.Namespace) -> int:
        try:
            # Kill processes running on UAP ports
            import psutil
            
            ports_to_kill = [8000, 3000]  # Backend and frontend ports
            processes_killed = 0
            
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    if proc.info['connections']:
                        for conn in proc.info['connections']:
                            if hasattr(conn, 'laddr') and conn.laddr and conn.laddr.port in ports_to_kill:
                                proc.kill()
                                processes_killed += 1
                                self.app.print_success(f"Stopped process {proc.info['name']} (PID: {proc.info['pid']})")
                                break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if processes_killed == 0:
                self.app.print_warning("No UAP processes found running")
            else:
                self.app.print_success(f"Stopped {processes_killed} UAP processes")
            
            return 0
            
        except ImportError:
            self.app.print_error("psutil not available. Please stop processes manually")
            return 1
        except Exception as e:
            self.app.print_error(f"Failed to stop deployment: {str(e)}")
            return 1
    
    async def _handle_status(self, args: argparse.Namespace) -> int:
        try:
            status = {
                "backend": {"running": False, "url": None},
                "frontend": {"running": False, "url": None}
            }
            
            # Check backend
            try:
                backend_response = await self.client.get_status()
                status["backend"]["running"] = True
                status["backend"]["url"] = self.config.get("backend_url")
                status["backend"]["status"] = backend_response.get("status", "unknown")
            except Exception:
                pass
            
            # Check frontend (simple HTTP check)
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localhost:3000", timeout=5)
                    if response.status_code == 200:
                        status["frontend"]["running"] = True
                        status["frontend"]["url"] = "http://localhost:3000"
            except Exception:
                pass
            
            self.app.print_output(status, args.format)
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to check deployment status: {str(e)}")
            return 1
    
    async def _handle_cloud(self, args: argparse.Namespace) -> int:
        try:
            if not args.provider:
                self.app.print_error("Cloud provider required")
                return 1
            
            # Check if SkyPilot is available
            try:
                result = subprocess.run(["sky", "status"], capture_output=True, text=True)
                if result.returncode != 0:
                    self.app.print_error("SkyPilot not available or not configured")
                    return 1
            except FileNotFoundError:
                self.app.print_error("SkyPilot not found. Please install SkyPilot for cloud deployments")
                return 1
            
            # Use the appropriate SkyPilot configuration
            config_file = f"skypilot/uap-{args.provider}.yaml"
            if args.config:
                config_file = args.config
            
            if not Path(config_file).exists():
                self.app.print_error(f"Configuration file not found: {config_file}")
                return 1
            
            self.app.print_success(f"Deploying to {args.provider} using {config_file}")
            
            # Launch with SkyPilot
            cmd = ["sky", "up", config_file]
            result = subprocess.run(cmd)
            
            if result.returncode == 0:
                self.app.print_success("Cloud deployment completed successfully")
                return 0
            else:
                self.app.print_error("Cloud deployment failed")
                return 1
                
        except Exception as e:
            self.app.print_error(f"Cloud deployment failed: {str(e)}")
            return 1


class PluginCommands(BaseCommand):
    """Plugin management commands."""
    
    def add_subcommands(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest='plugin_command', help='Plugin actions')
        
        # List plugins
        subparsers.add_parser('list', help='List available plugins')
        
        # Install plugin
        install_parser = subparsers.add_parser('install', help='Install a plugin')
        install_parser.add_argument('plugin', help='Plugin name or path')
        install_parser.add_argument('--enable', action='store_true', help='Enable after installation')
        
        # Enable plugin
        enable_parser = subparsers.add_parser('enable', help='Enable a plugin')
        enable_parser.add_argument('plugin_name', help='Plugin name')
        
        # Disable plugin
        disable_parser = subparsers.add_parser('disable', help='Disable a plugin')
        disable_parser.add_argument('plugin_name', help='Plugin name')
        
        # Plugin info
        info_parser = subparsers.add_parser('info', help='Get plugin information')
        info_parser.add_argument('plugin_name', nargs='?', help='Plugin name (optional)')
    
    async def handle_command(self, args: argparse.Namespace) -> int:
        if args.plugin_command == 'list':
            return await self._handle_list(args)
        elif args.plugin_command == 'install':
            return await self._handle_install(args)
        elif args.plugin_command == 'enable':
            return await self._handle_enable(args)
        elif args.plugin_command == 'disable':
            return await self._handle_disable(args)
        elif args.plugin_command == 'info':
            return await self._handle_info(args)
        else:
            self.app.print_error("Unknown plugin command")
            return 1
    
    async def _handle_list(self, args: argparse.Namespace) -> int:
        try:
            plugin_manager = self.app.plugin_manager
            plugin_info = plugin_manager.get_plugin_info()
            
            discovered = plugin_info.get("discovered", {})
            enabled = plugin_info.get("enabled", {})
            
            plugins = []
            for name, info in discovered.items():
                plugins.append({
                    "name": name,
                    "version": info.get("version", "unknown"),
                    "type": info.get("type", "unknown"),
                    "enabled": name in enabled,
                    "description": info.get("description", "")
                })
            
            if not plugins:
                self.app.print_warning("No plugins found")
                return 0
            
            self.app.print_output(plugins, args.format)
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to list plugins: {str(e)}")
            return 1
    
    async def _handle_install(self, args: argparse.Namespace) -> int:
        try:
            plugin_path = Path(args.plugin)
            
            if plugin_path.exists():
                # Local plugin file/directory
                plugins_dir = Path.home() / ".uap" / "plugins"
                plugins_dir.mkdir(parents=True, exist_ok=True)
                
                if plugin_path.is_file():
                    # Copy file
                    import shutil
                    dest = plugins_dir / plugin_path.name
                    shutil.copy2(plugin_path, dest)
                    self.app.print_success(f"Plugin installed: {dest}")
                else:
                    # Copy directory
                    import shutil
                    dest = plugins_dir / plugin_path.name
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(plugin_path, dest)
                    self.app.print_success(f"Plugin installed: {dest}")
                
                # Rediscover plugins
                plugin_manager = self.app.plugin_manager
                plugin_manager.discover_plugins()
                
                if args.enable:
                    plugin_name = plugin_path.stem
                    success = await plugin_manager.enable_plugin(plugin_name)
                    if success:
                        self.app.print_success(f"Plugin '{plugin_name}' enabled")
                    else:
                        self.app.print_warning(f"Failed to enable plugin '{plugin_name}'")
                
                return 0
            else:
                self.app.print_error(f"Plugin path not found: {args.plugin}")
                return 1
                
        except Exception as e:
            self.app.print_error(f"Failed to install plugin: {str(e)}")
            return 1
    
    async def _handle_enable(self, args: argparse.Namespace) -> int:
        try:
            plugin_manager = self.app.plugin_manager
            success = await plugin_manager.enable_plugin(args.plugin_name)
            
            if success:
                self.app.print_success(f"Plugin '{args.plugin_name}' enabled")
                return 0
            else:
                self.app.print_error(f"Failed to enable plugin '{args.plugin_name}'")
                return 1
                
        except Exception as e:
            self.app.print_error(f"Failed to enable plugin: {str(e)}")
            return 1
    
    async def _handle_disable(self, args: argparse.Namespace) -> int:
        try:
            plugin_manager = self.app.plugin_manager
            success = await plugin_manager.disable_plugin(args.plugin_name)
            
            if success:
                self.app.print_success(f"Plugin '{args.plugin_name}' disabled")
                return 0
            else:
                self.app.print_error(f"Failed to disable plugin '{args.plugin_name}'")
                return 1
                
        except Exception as e:
            self.app.print_error(f"Failed to disable plugin: {str(e)}")
            return 1
    
    async def _handle_info(self, args: argparse.Namespace) -> int:
        try:
            plugin_manager = self.app.plugin_manager
            plugin_info = plugin_manager.get_plugin_info()
            
            if args.plugin_name:
                # Specific plugin info
                discovered = plugin_info.get("discovered", {})
                if args.plugin_name not in discovered:
                    self.app.print_error(f"Plugin '{args.plugin_name}' not found")
                    return 1
                
                info = discovered[args.plugin_name]
                self.app.print_output(info, args.format)
            else:
                # All plugin info
                self.app.print_output(plugin_info, args.format)
            
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to get plugin info: {str(e)}")
            return 1


class ConfigCommands(BaseCommand):
    """Configuration management commands."""
    
    def add_subcommands(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest='config_command', help='Configuration actions')
        
        # Show config
        subparsers.add_parser('show', help='Show current configuration')
        
        # Set config value
        set_parser = subparsers.add_parser('set', help='Set configuration value')
        set_parser.add_argument('key', help='Configuration key')
        set_parser.add_argument('value', help='Configuration value')
        
        # Get config value
        get_parser = subparsers.add_parser('get', help='Get configuration value')
        get_parser.add_argument('key', help='Configuration key')
        
        # Create default config
        create_parser = subparsers.add_parser('create', help='Create default configuration file')
        create_parser.add_argument('--file', '-f', help='Configuration file path')
        create_parser.add_argument('--format', choices=['json', 'yaml'], default='json', help='File format')
        
        # Profile management
        profile_parser = subparsers.add_parser('profile', help='Manage configuration profiles')
        profile_subparsers = profile_parser.add_subparsers(dest='profile_command', help='Profile actions')
        
        profile_subparsers.add_parser('list', help='List profiles')
        
        create_profile_parser = profile_subparsers.add_parser('create', help='Create profile')
        create_profile_parser.add_argument('name', help='Profile name')
        
        switch_parser = profile_subparsers.add_parser('switch', help='Switch to profile')
        switch_parser.add_argument('name', help='Profile name')
        
        delete_parser = profile_subparsers.add_parser('delete', help='Delete profile')
        delete_parser.add_argument('name', help='Profile name')
    
    async def handle_command(self, args: argparse.Namespace) -> int:
        if args.config_command == 'show':
            return await self._handle_show(args)
        elif args.config_command == 'set':
            return await self._handle_set(args)
        elif args.config_command == 'get':
            return await self._handle_get(args)
        elif args.config_command == 'create':
            return await self._handle_create(args)
        elif args.config_command == 'profile':
            return await self._handle_profile(args)
        else:
            self.app.print_error("Unknown config command")
            return 1
    
    async def _handle_show(self, args: argparse.Namespace) -> int:
        try:
            config_dict = self.config.to_dict()
            
            # Hide sensitive information
            if "access_token" in config_dict:
                config_dict["access_token"] = "***"
            if "refresh_token" in config_dict:
                config_dict["refresh_token"] = "***"
            
            self.app.print_output(config_dict, args.format)
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to show configuration: {str(e)}")
            return 1
    
    async def _handle_set(self, args: argparse.Namespace) -> int:
        try:
            # Try to convert value to appropriate type
            value = args.value
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
            
            self.config.set(args.key, value)
            self.app.print_success(f"Set {args.key} = {value}")
            
            # Save configuration if it has a file
            try:
                self.config.save_to_file()
                self.app.print_success("Configuration saved")
            except ValueError:
                self.app.print_warning("Configuration not saved to file (no file path)")
            
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to set configuration: {str(e)}")
            return 1
    
    async def _handle_get(self, args: argparse.Namespace) -> int:
        try:
            value = self.config.get(args.key)
            
            if value is None:
                self.app.print_warning(f"Configuration key '{args.key}' not found")
                return 1
            
            if args.format == 'json':
                print(json.dumps(value))
            elif args.format == 'yaml':
                print(yaml.dump({args.key: value}))
            else:
                print(value)
            
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to get configuration: {str(e)}")
            return 1
    
    async def _handle_create(self, args: argparse.Namespace) -> int:
        try:
            file_path = args.file or "uap-config.json"
            Configuration.create_default_config_file(file_path, args.format)
            self.app.print_success(f"Default configuration created: {file_path}")
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to create configuration: {str(e)}")
            return 1
    
    async def _handle_profile(self, args: argparse.Namespace) -> int:
        try:
            from uap_sdk.config import ConfigurationProfile
            profile_manager = ConfigurationProfile()
            
            if args.profile_command == 'list':
                profiles = profile_manager.list_profiles()
                current = profile_manager.get_current_profile()
                
                profile_list = []
                for profile in profiles:
                    profile_list.append({
                        "name": profile,
                        "current": profile == current
                    })
                
                self.app.print_output(profile_list, args.format)
                
            elif args.profile_command == 'create':
                profile_manager.create_profile(args.name, self.config)
                self.app.print_success(f"Profile '{args.name}' created")
                
            elif args.profile_command == 'switch':
                profile_manager.set_current_profile(args.name)
                self.app.print_success(f"Switched to profile '{args.name}'")
                
            elif args.profile_command == 'delete':
                profile_manager.delete_profile(args.name)
                self.app.print_success(f"Profile '{args.name}' deleted")
            
            return 0
            
        except Exception as e:
            self.app.print_error(f"Profile operation failed: {str(e)}")
            return 1


class ProjectCommands(BaseCommand):
    """Project scaffolding commands."""
    
    def add_subcommands(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest='project_command', help='Project actions')
        
        # Create new project
        create_parser = subparsers.add_parser('create', help='Create new UAP project')
        create_parser.add_argument('name', help='Project name')
        create_parser.add_argument('--template', choices=['basic', 'advanced', 'custom-agent'], 
                                   default='basic', help='Project template')
        create_parser.add_argument('--directory', help='Project directory (default: current directory)')
        
        # Initialize existing project
        init_parser = subparsers.add_parser('init', help='Initialize UAP in existing project')
        init_parser.add_argument('--template', choices=['basic', 'advanced', 'custom-agent'], 
                                 default='basic', help='Project template')
        
        # Add component
        add_parser = subparsers.add_parser('add', help='Add component to project')
        add_parser.add_argument('component', choices=['agent', 'plugin', 'middleware'], help='Component type')
        add_parser.add_argument('name', help='Component name')
    
    async def handle_command(self, args: argparse.Namespace) -> int:
        if args.project_command == 'create':
            return await self._handle_create(args)
        elif args.project_command == 'init':
            return await self._handle_init(args)
        elif args.project_command == 'add':
            return await self._handle_add(args)
        else:
            self.app.print_error("Unknown project command")
            return 1
    
    async def _handle_create(self, args: argparse.Namespace) -> int:
        try:
            project_dir = Path(args.directory) if args.directory else Path.cwd() / args.name
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Create project structure
            await self._create_project_structure(project_dir, args.name, args.template)
            
            self.app.print_success(f"Project '{args.name}' created in {project_dir}")
            self.app.print_success("Next steps:")
            self.app.print_success("  cd " + str(project_dir))
            self.app.print_success("  devbox shell")
            self.app.print_success("  uap deploy start")
            
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to create project: {str(e)}")
            return 1
    
    async def _handle_init(self, args: argparse.Namespace) -> int:
        try:
            project_dir = Path.cwd()
            project_name = project_dir.name
            
            await self._create_project_structure(project_dir, project_name, args.template)
            
            self.app.print_success(f"Initialized UAP project in {project_dir}")
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to initialize project: {str(e)}")
            return 1
    
    async def _create_project_structure(self, project_dir: Path, name: str, template: str) -> None:
        """Create the project structure based on template."""
        
        # Create basic structure
        (project_dir / "agents").mkdir(exist_ok=True)
        (project_dir / "plugins").mkdir(exist_ok=True)
        (project_dir / "config").mkdir(exist_ok=True)
        
        # Create configuration file
        config_file = project_dir / "config" / "uap.json"
        config = Configuration()
        config.save_to_file(config_file)
        
        # Create basic agent
        if template in ['basic', 'advanced']:
            agent_file = project_dir / "agents" / "example_agent.py"
            with open(agent_file, 'w') as f:
                f.write(self._get_example_agent_code(name))
        
        # Create custom agent for custom-agent template
        if template == 'custom-agent':
            agent_file = project_dir / "agents" / f"{name}_agent.py"
            with open(agent_file, 'w') as f:
                f.write(self._get_custom_agent_code(name))
        
        # Create main application file
        main_file = project_dir / "main.py"
        with open(main_file, 'w') as f:
            f.write(self._get_main_app_code(name, template))
        
        # Create requirements file
        requirements_file = project_dir / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write("uap-sdk>=1.0.0\naiofiles\nhttpx\nwebsockets\n")
        
        # Create README
        readme_file = project_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(self._get_readme_content(name, template))
    
    def _get_example_agent_code(self, project_name: str) -> str:
        return f'''# Example Agent for {project_name}

import asyncio
from uap_sdk import UAPAgent, CustomAgentBuilder, Configuration, SimpleAgent


async def main():
    """Main function to run the example agent."""
    
    # Load configuration
    config = Configuration(config_file="config/uap.json")
    
    # Create agent using builder
    agent = (CustomAgentBuilder("example-agent")
             .with_simple_framework()
             .with_config(config)
             .build())
    
    # Start the agent
    await agent.start()
    
    # Example interactions
    test_messages = [
        "Hello!",
        "How are you?",
        "What can you do?",
        "Goodbye"
    ]
    
    for message in test_messages:
        print(f"User: {{message}}")
        response = await agent.process_message(message)
        print(f"Agent: {{response.get('content', '')}}")
        print()
    
    # Stop the agent
    await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    def _get_custom_agent_code(self, project_name: str) -> str:
        return f'''# Custom Agent Framework for {project_name}

from typing import Dict, Any, List
from uap_sdk import AgentFramework, Configuration
from datetime import datetime


class {project_name.replace('-', '_').title()}Agent(AgentFramework):
    """Custom agent framework for {project_name}."""
    
    def __init__(self, config: Configuration = None):
        super().__init__("{project_name}", config)
        self.knowledge_base = {{
            "capabilities": [
                "Answer questions",
                "Process information", 
                "Provide assistance"
            ],
            "personality": "helpful and knowledgeable"
        }}
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message using custom logic."""
        
        message_lower = message.lower()
        
        # Custom responses based on keywords
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            response = f"Hello! I'm the {project_name} agent. How can I help you today?"
        
        elif any(word in message_lower for word in ["capabilities", "what can you do"]):
            capabilities = "\\n".join(f"- {{cap}}" for cap in self.knowledge_base["capabilities"])
            response = f"Here are my capabilities:\\n{{capabilities}}"
        
        elif any(word in message_lower for word in ["time", "date"]):
            now = datetime.now()
            response = f"The current time is {{now.strftime('%Y-%m-%d %H:%M:%S')}}"
        
        elif any(word in message_lower for word in ["bye", "goodbye"]):
            response = "Goodbye! It was nice talking with you."
        
        else:
            response = f"I understand you said: '{{message}}'. I'm still learning how to respond to that!"
        
        return {{
            "content": response,
            "metadata": {{
                "source": self.framework_name,
                "timestamp": datetime.utcnow().isoformat(),
                "message_length": len(message),
                "response_type": "custom"
            }}
        }}
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {{
            "status": self.status,
            "framework": self.framework_name,
            "initialized": self.is_initialized,
            "knowledge_base_size": len(self.knowledge_base),
            "capabilities": self.get_capabilities()
        }}
    
    async def initialize(self) -> None:
        """Initialize the custom agent."""
        self.is_initialized = True
        self.status = "active"
        # Add any custom initialization logic here
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities."""
        return self.knowledge_base["capabilities"]
'''
    
    def _get_main_app_code(self, project_name: str, template: str) -> str:
        if template == 'custom-agent':
            agent_class = f"{project_name.replace('-', '_').title()}Agent"
            return f'''# Main application for {project_name}

import asyncio
from pathlib import Path
from uap_sdk import UAPAgent, Configuration, UAPClient
from agents.{project_name}_agent import {agent_class}


async def main():
    """Main application entry point."""
    
    # Load configuration
    config_path = Path("config/uap.json")
    if config_path.exists():
        config = Configuration(config_file=config_path)
    else:
        config = Configuration()
    
    # Create custom agent
    framework = {agent_class}(config)
    agent = UAPAgent("{project_name}-agent", framework, config)
    
    # Start the agent
    await agent.start()
    
    print(f"{{project_name}} agent started successfully!")
    print("Agent status:", agent.get_status())
    
    # Interactive loop
    try:
        while True:
            user_input = input("\\nYou: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            
            response = await agent.process_message(user_input)
            print(f"Agent: {{response.get('content', '')}}")
    
    except KeyboardInterrupt:
        pass
    finally:
        await agent.stop()
        print("\\nAgent stopped. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
'''
        else:
            return f'''# Main application for {project_name}

import asyncio
from pathlib import Path
from uap_sdk import UAPClient, Configuration


async def main():
    """Main application entry point."""
    
    # Load configuration
    config_path = Path("config/uap.json")
    if config_path.exists():
        config = Configuration(config_file=config_path)
    else:
        config = Configuration()
    
    # Create client
    client = UAPClient(config)
    
    print(f"{{project_name}} application started!")
    
    # Example: Get system status
    try:
        status = await client.get_status()
        print("System status:", status)
    except Exception as e:
        print(f"Could not connect to UAP backend: {{e}}")
        print("Make sure the UAP backend is running with 'uap deploy start'")
        return
    
    # Interactive chat loop
    agent_id = "default"
    
    try:
        while True:
            user_input = input("\\nYou: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            
            response = await client.chat(agent_id, user_input)
            print(f"Agent: {{response.get('content', '')}}")
    
    except KeyboardInterrupt:
        pass
    finally:
        await client.cleanup()
        print("\\nApplication stopped. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    def _get_readme_content(self, project_name: str, template: str) -> str:
        return f'''# {project_name}

A UAP (Unified Agentic Platform) project created with template: {template}

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python main.py
   ```

## Project Structure

- `agents/` - Custom agent implementations
- `plugins/` - Custom plugins
- `config/` - Configuration files
- `main.py` - Main application entry point

## UAP CLI Commands

- `uap deploy start` - Start UAP backend and frontend
- `uap agent list` - List available agents
- `uap agent chat <agent-id>` - Chat with an agent
- `uap config show` - Show configuration

## Development

This project uses the UAP SDK for building AI agents and integrations. 

For more information, see the UAP documentation.
'''
    
    async def _handle_add(self, args: argparse.Namespace) -> int:
        try:
            if args.component == 'agent':
                await self._add_agent(args.name)
            elif args.component == 'plugin':
                await self._add_plugin(args.name)
            elif args.component == 'middleware':
                await self._add_middleware(args.name)
            
            self.app.print_success(f"{args.component.title()} '{args.name}' added to project")
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to add component: {str(e)}")
            return 1
    
    async def _add_agent(self, name: str) -> None:
        """Add a new agent to the project."""
        agents_dir = Path("agents")
        agents_dir.mkdir(exist_ok=True)
        
        agent_file = agents_dir / f"{name}_agent.py"
        with open(agent_file, 'w') as f:
            f.write(self._get_custom_agent_code(name))
    
    async def _add_plugin(self, name: str) -> None:
        """Add a new plugin to the project."""
        plugins_dir = Path("plugins")
        plugins_dir.mkdir(exist_ok=True)
        
        plugin_file = plugins_dir / f"{name}_plugin.py"
        with open(plugin_file, 'w') as f:
            f.write(self._get_plugin_template(name))
    
    def _get_plugin_template(self, name: str) -> str:
        return f'''# {name.title()} Plugin

from typing import Dict, Any, Optional
from uap_sdk.plugin import AgentPlugin, Configuration


class {name.title()}Plugin(AgentPlugin):
    """Custom plugin for {name} functionality."""
    
    PLUGIN_NAME = "{name}"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "Custom plugin for {name} functionality"
    
    def __init__(self):
        super().__init__(self.PLUGIN_NAME, self.PLUGIN_VERSION)
    
    async def initialize(self, config: Configuration) -> None:
        """Initialize the plugin."""
        # Add initialization logic here
        pass
    
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        # Add cleanup logic here
        pass
    
    async def process_message(self, agent_id: str, message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a message through the plugin."""
        
        # Example: Respond to specific keywords
        if "{name}" in message.lower():
            return {{
                "content": f"This is a response from the {{name}} plugin!",
                "metadata": {{
                    "plugin": self.name,
                    "processed_by": "{name}_plugin"
                }}
            }}
        
        return None  # Don't handle this message
    
    def should_handle_message(self, message: str, context: Dict[str, Any]) -> bool:
        """Determine if this plugin should handle the message."""
        return "{name}" in message.lower()
'''
    
    async def _add_middleware(self, name: str) -> None:
        """Add middleware to the project."""
        plugins_dir = Path("plugins")
        plugins_dir.mkdir(exist_ok=True)
        
        middleware_file = plugins_dir / f"{name}_middleware.py"
        with open(middleware_file, 'w') as f:
            f.write(self._get_middleware_template(name))
    
    def _get_middleware_template(self, name: str) -> str:
        return f'''# {name.title()} Middleware

from typing import Dict, Any, Tuple
from uap_sdk.plugin import MiddlewarePlugin, Configuration


class {name.title()}Middleware(MiddlewarePlugin):
    """Middleware plugin for {name} processing."""
    
    PLUGIN_NAME = "{name}_middleware"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "Middleware for {name} message processing"
    
    def __init__(self):
        super().__init__(self.PLUGIN_NAME, self.PLUGIN_VERSION)
    
    async def initialize(self, config: Configuration) -> None:
        """Initialize the middleware."""
        # Add initialization logic here
        pass
    
    async def cleanup(self) -> None:
        """Clean up middleware resources."""
        # Add cleanup logic here
        pass
    
    async def process_request(self, message: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Process incoming request."""
        
        # Example: Add metadata to context
        context["{name}_processed"] = True
        context["{name}_timestamp"] = "2023-01-01T00:00:00Z"
        
        # Example: Modify message
        processed_message = message
        
        return processed_message, context
    
    async def process_response(self, response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing response."""
        
        # Example: Add metadata to response
        if "metadata" not in response:
            response["metadata"] = {{}}
        
        response["metadata"]["{name}_middleware"] = "processed"
        
        return response
'''


class MonitoringCommands(BaseCommand):
    """Monitoring and status commands."""
    
    def add_subcommands(self, parser: argparse.ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest='monitor_command', help='Monitoring actions')
        
        # System status
        subparsers.add_parser('status', help='Get system status')
        
        # Health check
        subparsers.add_parser('health', help='Perform health check')
        
        # Metrics
        metrics_parser = subparsers.add_parser('metrics', help='Get system metrics')
        metrics_parser.add_argument('--format', choices=['json', 'prometheus'], default='json', help='Metrics format')
        
        # Logs
        logs_parser = subparsers.add_parser('logs', help='View system logs')
        logs_parser.add_argument('--lines', '-n', type=int, default=100, help='Number of lines to show')
        logs_parser.add_argument('--follow', '-f', action='store_true', help='Follow log output')
    
    async def handle_command(self, args: argparse.Namespace) -> int:
        if args.monitor_command == 'status':
            return await self._handle_status(args)
        elif args.monitor_command == 'health':
            return await self._handle_health(args)
        elif args.monitor_command == 'metrics':
            return await self._handle_metrics(args)
        elif args.monitor_command == 'logs':
            return await self._handle_logs(args)
        else:
            self.app.print_error("Unknown monitoring command")
            return 1
    
    async def _handle_status(self, args: argparse.Namespace) -> int:
        try:
            status = await self.client.get_status()
            self.app.print_output(status, args.format)
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to get system status: {str(e)}")
            return 1
    
    async def _handle_health(self, args: argparse.Namespace) -> int:
        try:
            # Perform comprehensive health check
            health_status = {
                "overall": "healthy",
                "components": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check backend connection
            try:
                status = await self.client.get_status()
                health_status["components"]["backend"] = {
                    "status": "healthy",
                    "response_time_ms": status.get("response_time", 0),
                    "version": status.get("version", "unknown")
                }
            except Exception as e:
                health_status["components"]["backend"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall"] = "degraded"
            
            # Check WebSocket connection
            try:
                # Quick WebSocket connection test
                import websockets
                ws_url = self.config.get("websocket_url", "ws://localhost:8000")
                
                async with websockets.connect(f"{ws_url}/ws/health", timeout=5) as websocket:
                    await websocket.send("ping")
                    response = await websocket.recv()
                    
                health_status["components"]["websocket"] = {
                    "status": "healthy",
                    "response": response
                }
            except Exception as e:
                health_status["components"]["websocket"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                if health_status["overall"] == "healthy":
                    health_status["overall"] = "degraded"
            
            # Check authentication
            try:
                token = self.client.auth.get_access_token()
                if token:
                    health_status["components"]["auth"] = {
                        "status": "authenticated",
                        "has_token": True
                    }
                else:
                    health_status["components"]["auth"] = {
                        "status": "unauthenticated", 
                        "has_token": False
                    }
            except Exception as e:
                health_status["components"]["auth"] = {
                    "status": "error",
                    "error": str(e)
                }
            
            self.app.print_output(health_status, args.format)
            
            # Return appropriate exit code
            if health_status["overall"] == "healthy":
                return 0
            elif health_status["overall"] == "degraded":
                return 1
            else:
                return 2
                
        except Exception as e:
            self.app.print_error(f"Health check failed: {str(e)}")
            return 2
    
    async def _handle_metrics(self, args: argparse.Namespace) -> int:
        try:
            # Get metrics from backend
            import httpx
            
            backend_url = self.config.get("backend_url", "http://localhost:8000")
            
            async with httpx.AsyncClient() as client:
                if args.format == 'prometheus':
                    response = await client.get(f"{backend_url}/metrics")
                    print(response.text)
                else:
                    response = await client.get(f"{backend_url}/api/metrics")
                    response.raise_for_status()
                    metrics = response.json()
                    self.app.print_output(metrics, args.format)
            
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to get metrics: {str(e)}")
            return 1
    
    async def _handle_logs(self, args: argparse.Namespace) -> int:
        try:
            # For now, just show a message about log location
            # In a real implementation, this would connect to the logging system
            
            log_info = {
                "message": "Log viewing functionality requires backend integration",
                "log_locations": [
                    "/var/log/uap/",
                    "~/.uap/logs/",
                    "./logs/"
                ],
                "alternatives": [
                    "Check backend logs directly",
                    "Use 'docker logs' if running in containers",
                    "Check devbox logs with 'devbox run logs'"
                ]
            }
            
            self.app.print_output(log_info, args.format)
            return 0
            
        except Exception as e:
            self.app.print_error(f"Failed to get logs: {str(e)}")
            return 1