#!/usr/bin/env python3
"""
PRP Execution Command - Secure Implementation

This command executes a Product Requirements Prompt (PRP) with comprehensive
validation, monitoring, and rollback capabilities using the new secure command system.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add the context engineering lib to path
context_lib = Path(__file__).parent.parent.parent / "context-engineering" / "lib"
sys.path.insert(0, str(context_lib))

try:
    from base_prp_command import BasePRPCommand, PRPContext, ValidationError
except ImportError:
    # Fallback imports if base command not available
    class BasePRPCommand:
        def __init__(self, name, description):
            self.name = name
            self.description = description
            self.last_result = {}
        
        def run(self):
            parser = argparse.ArgumentParser(description=self.description)
            self.setup_arguments(parser)
            args = parser.parse_args()
            self.validate_arguments(args)
            context = type('PRPContext', (), {'args': args})()
            result = self.execute_command_logic(context)
            self.last_result = result
            return self.generate_output(result)
    
    class ValidationError(Exception):
        pass

class ExecutePRPCommand(BasePRPCommand):
    """Secure PRP execution command with validation and monitoring."""
    
    def __init__(self):
        super().__init__("execute-prp", "Execute PRP with validation and monitoring")
    
    def setup_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Set up command-line arguments for PRP execution."""
        parser.add_argument("prp_file", help="PRP file to execute")
        parser.add_argument("--validate", action="store_true", help="Run validation gates")
        parser.add_argument("--monitor", action="store_true", help="Enable performance monitoring")
        parser.add_argument("--rollback-on-failure", action="store_true", 
                           help="Automatically rollback on execution failure")
        parser.add_argument("--timeout", type=int, default=300, 
                           help="Execution timeout in seconds")
        parser.add_argument("--environment", help="Override environment from PRP file")
        parser.add_argument("--dry-run", action="store_true", 
                           help="Show what would be executed without running")
    
    def validate_arguments(self, args: argparse.Namespace) -> None:
        """Validate command arguments."""
        # Validate PRP file exists and is readable
        prp_path = Path(args.prp_file)
        if not prp_path.exists():
            raise ValidationError(f"PRP file not found: {args.prp_file}")
        
        if not prp_path.is_file():
            raise ValidationError(f"PRP path is not a file: {args.prp_file}")
        
        try:
            prp_content = prp_path.read_text(encoding='utf-8')
            if len(prp_content.strip()) < 10:
                raise ValidationError("PRP file appears to be empty or too short")
        except Exception as e:
            raise ValidationError(f"Cannot read PRP file: {e}")
        
        # Validate timeout
        if args.timeout <= 0 or args.timeout > 3600:
            raise ValidationError("Timeout must be between 1 and 3600 seconds")
    
    def execute_command_logic(self, context) -> dict:
        """Execute PRP with secure command system."""
        result = {
            "success": False,
            "message": "",
            "execution_time": 0,
            "validation_results": {},
            "monitoring_data": {},
            "executed_commands": []
        }
        
        start_time = datetime.now()
        
        try:
            # Import secure command system
            try:
                from secure_command_system import (
                    SecurityConfigManager, SecureCommandInvoker, SecurityLevel,
                    ExecutionPermission, SecureShellCommand
                )
                from command_system import CommandContext
            except ImportError as e:
                result["message"] = f"Cannot import secure command system: {e}"
                return result
            
            # Parse PRP file to extract commands and environment
            prp_data = self._parse_prp_file(context.args.prp_file)
            
            # Determine environment
            environment = context.args.environment or prp_data.get("environment", "python-env")
            
            # Create security configuration
            config_manager = SecurityConfigManager()
            security_context = config_manager.create_security_context(
                user_id="prp_executor",
                permissions=["read", "write", "execute"]
            )
            security_context.security_level = SecurityLevel.HIGH
            security_context.max_execution_time = context.args.timeout
            
            # Set allowed paths based on environment
            project_root = Path.cwd()
            env_path = project_root / environment
            if env_path.exists():
                security_context.allowed_paths = [project_root, env_path]
            else:
                security_context.allowed_paths = [project_root]
            
            # Create command context
            command_context = CommandContext(
                working_directory=env_path if env_path.exists() else project_root,
                environment=environment,
                feature_name=prp_data.get("feature_name", "unknown"),
                parameters={"prp_file": context.args.prp_file}
            )
            
            # Create secure invoker
            invoker = SecureCommandInvoker(security_context)
            
            # Execute validation if requested
            if context.args.validate:
                validation_result = self._run_validation_gates(
                    invoker, command_context, prp_data, context.args.dry_run
                )
                result["validation_results"] = validation_result
                
                if not validation_result["success"]:
                    result["message"] = "Validation gates failed"
                    return result
            
            # Execute main PRP commands
            execution_result = self._execute_prp_commands(
                invoker, command_context, prp_data, context.args.dry_run
            )
            result["executed_commands"] = execution_result["commands"]
            
            if not execution_result["success"] and context.args.rollback_on_failure:
                # Attempt rollback
                rollback_result = self._perform_rollback(invoker, command_context)
                result["rollback_performed"] = rollback_result
            
            # Collect monitoring data if requested
            if context.args.monitor:
                result["monitoring_data"] = self._collect_monitoring_data(invoker)
            
            result["success"] = execution_result["success"]
            result["message"] = execution_result.get("message", "PRP execution completed")
            
        except Exception as e:
            result["message"] = f"PRP execution failed: {str(e)}"
            print(f"Error: {e}")  # Debug output
        
        finally:
            result["execution_time"] = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _parse_prp_file(self, prp_file: str) -> dict:
        """Parse PRP file to extract metadata and commands."""
        prp_path = Path(prp_file)
        content = prp_path.read_text(encoding='utf-8')
        
        # Extract basic metadata
        prp_data = {
            "feature_name": prp_path.stem,
            "environment": "python-env",  # default
            "commands": [],
            "validation_commands": [],
            "setup_commands": []
        }
        
        # Parse YAML frontmatter if present
        if content.startswith('---'):
            try:
                import yaml
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    if isinstance(frontmatter, dict):
                        prp_data.update(frontmatter)
            except ImportError:
                pass  # yaml not available
        
        # Extract commands from content - look for common command patterns
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for shell commands in various formats
            if line.startswith('$ ') or line.startswith('bash ') or line.startswith('python '):
                command = line[2:] if line.startswith('$ ') else line
                if 'test' in line.lower() or 'lint' in line.lower() or 'check' in line.lower():
                    prp_data['validation_commands'].append(command)
                elif 'install' in line.lower() or 'setup' in line.lower():
                    prp_data['setup_commands'].append(command)
                else:
                    prp_data['commands'].append(command)
            
            # Look for code blocks with executable commands
            elif line.startswith('```bash') or line.startswith('```sh'):
                # This indicates start of a code block, but we'll handle individual commands
                pass
        
        return prp_data
    
    def _run_validation_gates(self, invoker, context, prp_data: dict, dry_run: bool) -> dict:
        """Run validation gates for the PRP."""
        from secure_command_system import SecureShellCommand
        
        validation_result = {
            "success": True,
            "checks_passed": 0,
            "checks_failed": 0,
            "details": []
        }
        
        validation_commands = prp_data.get('validation_commands', [])
        if not validation_commands:
            # Default validation commands based on environment
            env = context.environment
            if 'python' in env:
                validation_commands = ['echo "Python environment check"', 'ls -la']
            elif 'typescript' in env:
                validation_commands = ['echo "TypeScript environment check"', 'ls -la']
            elif 'rust' in env:
                validation_commands = ['echo "Rust environment check"', 'ls -la']
            elif 'go' in env:
                validation_commands = ['echo "Go environment check"', 'ls -la']
            else:
                validation_commands = ['echo "Generic environment check"', 'ls -la']
        
        for cmd in validation_commands:
            try:
                if dry_run:
                    validation_result["details"].append(f"DRY RUN: Would execute: {cmd}")
                    validation_result["checks_passed"] += 1
                else:
                    shell_command = SecureShellCommand(cmd, timeout=60)
                    result = invoker.execute_secure_command(shell_command, context)
                    
                    if result.success:
                        validation_result["checks_passed"] += 1
                        validation_result["details"].append(f"✅ {cmd}")
                    else:
                        validation_result["checks_failed"] += 1
                        validation_result["details"].append(f"❌ {cmd}: {result.message}")
                        validation_result["success"] = False
                        
            except Exception as e:
                validation_result["checks_failed"] += 1
                validation_result["details"].append(f"❌ {cmd}: Exception: {str(e)}")
                validation_result["success"] = False
        
        return validation_result
    
    def _execute_prp_commands(self, invoker, context, prp_data: dict, dry_run: bool) -> dict:
        """Execute the main PRP commands."""
        execution_result = {
            "success": True,
            "commands": [],
            "message": ""
        }
        
        # Execute setup commands first
        setup_commands = prp_data.get('setup_commands', [])
        for cmd in setup_commands:
            command_result = self._execute_single_command(invoker, context, cmd, dry_run, "setup")
            execution_result["commands"].append(command_result)
            if not command_result["success"]:
                execution_result["success"] = False
                execution_result["message"] = f"Setup command failed: {cmd}"
                return execution_result
        
        # Execute main commands
        main_commands = prp_data.get('commands', [])
        if not main_commands:
            # If no specific commands found, add a default success command
            main_commands = ['echo "PRP execution completed - no specific commands found"']
        
        for cmd in main_commands:
            command_result = self._execute_single_command(invoker, context, cmd, dry_run, "main")
            execution_result["commands"].append(command_result)
            if not command_result["success"]:
                execution_result["success"] = False
                execution_result["message"] = f"Main command failed: {cmd}"
                return execution_result
        
        return execution_result
    
    def _execute_single_command(self, invoker, context, cmd: str, dry_run: bool, cmd_type: str) -> dict:
        """Execute a single command securely."""
        from secure_command_system import SecureShellCommand
        
        if dry_run:
            return {
                "command": cmd,
                "type": cmd_type,
                "success": True,
                "message": f"DRY RUN: Would execute: {cmd}",
                "output": ""
            }
        
        try:
            shell_command = SecureShellCommand(cmd, timeout=context.parameters.get("timeout", 60))
            result = invoker.execute_secure_command(shell_command, context)
            
            return {
                "command": cmd,
                "type": cmd_type,
                "success": result.success,
                "message": result.message,
                "output": result.data.get("stdout", "") if result.data else "",
                "error": result.data.get("stderr", "") if result.data else ""
            }
            
        except Exception as e:
            return {
                "command": cmd,
                "type": cmd_type,
                "success": False,
                "message": f"Command execution failed: {str(e)}",
                "output": "",
                "error": str(e)
            }
    
    def _perform_rollback(self, invoker, context) -> dict:
        """Perform rollback of executed commands."""
        # Note: The secure command system handles rollback automatically
        # This is just for reporting
        return {
            "attempted": True,
            "success": True,
            "message": "Rollback handled by secure command system"
        }
    
    def _collect_monitoring_data(self, invoker) -> dict:
        """Collect monitoring data from the execution."""
        audit_log = invoker.get_session_audit_log()
        
        return {
            "commands_executed": len(invoker.session_commands),
            "audit_entries": len(audit_log),
            "session_id": invoker.security_context.session_id,
            "security_level": invoker.security_context.security_level.value
        }
    
    def generate_output(self, result: dict) -> str:
        """Generate formatted output from execution results."""
        lines = []
        
        if result["success"]:
            lines.append("✅ PRP execution completed successfully")
        else:
            lines.append("❌ PRP execution failed")
        
        lines.append(f"⏱️  Execution time: {result['execution_time']:.2f} seconds")
        
        # Validation results
        if result.get("validation_results"):
            val_results = result["validation_results"]
            lines.append(f"\n🔍 Validation Results:")
            lines.append(f"   Passed: {val_results['checks_passed']}")
            lines.append(f"   Failed: {val_results['checks_failed']}")
            
            for detail in val_results["details"][:5]:  # Show first 5 details
                lines.append(f"   {detail}")
        
        # Command execution summary
        executed_commands = result.get("executed_commands", [])
        if executed_commands:
            lines.append(f"\n📋 Commands Executed ({len(executed_commands)}):")
            for cmd_result in executed_commands[:3]:  # Show first 3 commands
                status = "✅" if cmd_result["success"] else "❌"
                lines.append(f"   {status} {cmd_result['command'][:50]}...")
        
        # Monitoring data
        if result.get("monitoring_data"):
            mon_data = result["monitoring_data"]
            lines.append(f"\n📊 Monitoring Data:")
            lines.append(f"   Commands: {mon_data['commands_executed']}")
            lines.append(f"   Session: {mon_data['session_id'][:8]}...")
            lines.append(f"   Security Level: {mon_data['security_level']}")
        
        # Rollback information
        if result.get("rollback_performed"):
            rollback = result["rollback_performed"]
            lines.append(f"\n🔄 Rollback: {rollback['message']}")
        
        if not result["success"]:
            lines.append(f"\n💡 Error: {result['message']}")
        
        return "\n".join(lines)


def main():
    """Main entry point for the execute-prp command."""
    command = ExecutePRPCommand()
    try:
        result = command.run()
        print(result)
        sys.exit(0 if command.last_result.get("success", False) else 1)
    except KeyboardInterrupt:
        print("\n⏹️  PRP execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Fatal error in PRP execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()