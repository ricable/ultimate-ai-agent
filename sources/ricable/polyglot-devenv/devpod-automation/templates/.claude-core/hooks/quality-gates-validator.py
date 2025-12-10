#!/usr/bin/env python3
"""
Cross-Language Quality Gates Validator
Enforces polyglot development standards and runs appropriate linters before tool execution.

Features:
- Pre-execution validation for commands and file operations
- Environment-specific quality checks
- Cross-language consistency enforcement
- Integration with existing linting infrastructure
"""

import json
import sys
import subprocess
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

class QualityGatesValidator:
    """Validates operations against polyglot development standards."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
        # Environment paths
        self.environments = {
            "python-env": project_root / "python-env",
            "typescript-env": project_root / "typescript-env", 
            "rust-env": project_root / "rust-env",
            "go-env": project_root / "go-env",
            "nushell-env": project_root / "nushell-env"
        }
        
        # Quality rules for each environment
        self.quality_rules = self._init_quality_rules()
        
    def _init_quality_rules(self) -> Dict[str, Dict]:
        """Initialize quality rules for each environment."""
        return {
            "python-env": {
                "required_tools": ["uv", "ruff", "mypy", "pytest"],
                "file_patterns": [r"\.py$"],
                "anti_patterns": [
                    (r"pip install", "Use 'uv add' instead of 'pip install' for dependency management"),
                    (r"python -m venv", "Use devbox environment instead of manual venv creation"),
                    (r"from \* import", "Avoid wildcard imports for better code clarity"),
                ],
                "required_headers": ["#!/usr/bin/env python3", "\"\"\"", "import"],
                "lint_command": "devbox run lint",
                "test_command": "devbox run test",
                "format_command": "devbox run format"
            },
            "typescript-env": {
                "required_tools": ["node", "npm", "typescript", "eslint", "prettier"],
                "file_patterns": [r"\.(ts|tsx|js|jsx)$"],
                "anti_patterns": [
                    (r"\bany\b", "Avoid 'any' type, use proper TypeScript types"),
                    (r"console\.log\(", "Use proper logging instead of console.log in production code"),
                    (r"var\s+", "Use 'const' or 'let' instead of 'var'"),
                ],
                "required_headers": ["'use strict'", "import", "export"],
                "lint_command": "devbox run lint",
                "test_command": "devbox run test", 
                "format_command": "devbox run format"
            },
            "rust-env": {
                "required_tools": ["rustc", "cargo", "clippy", "rustfmt"],
                "file_patterns": [r"\.rs$"],
                "anti_patterns": [
                    (r"\.unwrap\(\)", "Avoid .unwrap(), use proper error handling with ? or match"),
                    (r"\.clone\(\)", "Minimize .clone() usage, consider borrowing instead"),
                    (r"unsafe\s*{", "Unsafe blocks require careful review and documentation"),
                ],
                "required_headers": ["//", "use", "fn main"],
                "lint_command": "devbox run lint",
                "test_command": "devbox run test",
                "format_command": "devbox run format"
            },
            "go-env": {
                "required_tools": ["go", "gofmt", "golangci-lint"],
                "file_patterns": [r"\.go$"],
                "anti_patterns": [
                    (r"panic\(", "Avoid panic(), use proper error handling"),
                    (r"fmt\.Print", "Use structured logging instead of fmt.Print"),
                    (r"_\s*=\s*err", "Don't ignore errors, handle them properly"),
                ],
                "required_headers": ["package", "import", "func"],
                "lint_command": "devbox run lint",
                "test_command": "devbox run test",
                "format_command": "devbox run format"
            },
            "nushell-env": {
                "required_tools": ["nu"],
                "file_patterns": [r"\.nu$"],
                "anti_patterns": [
                    (r"rm -rf", "Use careful file operations, consider safer alternatives"),
                    (r"curl.*password", "Avoid exposing credentials in shell commands"),
                ],
                "required_headers": ["#!/usr/bin/env nu", "def", "use"],
                "lint_command": "devbox run check",
                "test_command": "devbox run test",
                "format_command": "devbox run format"
            }
        }
    
    def detect_environment(self, file_path: str) -> Optional[str]:
        """Detect the environment based on file path."""
        for env_name, env_path in self.environments.items():
            if str(env_path) in file_path:
                return env_name
        
        # Fallback: detect by file extension
        if file_path.endswith('.py'):
            return "python-env"
        elif file_path.endswith(('.ts', '.tsx', '.js', '.jsx')):
            return "typescript-env"
        elif file_path.endswith('.rs'):
            return "rust-env"
        elif file_path.endswith('.go'):
            return "go-env"
        elif file_path.endswith('.nu'):
            return "nushell-env"
            
        return None
    
    def validate_file_content(self, file_path: str, content: str, environment: str) -> List[str]:
        """Validate file content against environment-specific rules."""
        issues = []
        
        if environment not in self.quality_rules:
            return issues
            
        rules = self.quality_rules[environment]
        
        # Check anti-patterns
        for pattern, message in rules["anti_patterns"]:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"❌ {message}")
        
        # Check file structure for new files
        if len(content.strip()) > 0:
            lines = content.split('\n')
            first_few_lines = '\n'.join(lines[:10]).lower()
            
            # Validate required headers/structure
            required_found = False
            for required in rules["required_headers"]:
                if required.lower() in first_few_lines:
                    required_found = True
                    break
            
            if not required_found and len(lines) > 5:
                issues.append(f"⚠️ Consider adding proper file headers/imports for {environment}")
        
        return issues
    
    def validate_bash_command(self, command: str) -> List[str]:
        """Validate bash commands for quality and security."""
        issues = []
        
        # Command quality rules
        command_rules = [
            (r"\bgrep\b(?!.*ripgrep|.*rg)", "Use 'rg' (ripgrep) instead of 'grep' for better performance"),
            (r"\bfind\s+.*-name", "Consider using 'rg --files' or 'fd' instead of 'find -name'"),
            (r"rm\s+-rf\s+/", "Dangerous: rm -rf with absolute paths requires careful review"),
            (r"chmod\s+777", "Avoid chmod 777, use more restrictive permissions"),
            (r"curl.*http://", "Prefer HTTPS over HTTP for security"),
            (r"pip\s+install(?!.*--user)", "Use 'uv add' instead of 'pip install' in this polyglot environment"),
            (r"npm\s+install\s+-g", "Avoid global npm installs, use devbox for tool management"),
        ]
        
        for pattern, message in command_rules:
            if re.search(pattern, command, re.IGNORECASE):
                issues.append(f"⚠️ {message}")
        
        # Environment-specific command validation
        if "devbox run" in command:
            # Validate devbox commands are in correct environment
            env = self._detect_environment_from_pwd()
            if env and not self._is_command_in_environment(command, env):
                issues.append(f"🔄 Consider running devbox command from {env} directory")
        
        return issues
    
    def _detect_environment_from_pwd(self) -> Optional[str]:
        """Detect current environment from working directory."""
        import os
        cwd = os.getcwd()
        
        for env_name, env_path in self.environments.items():
            if str(env_path) in cwd:
                return env_name
        return None
    
    def _is_command_in_environment(self, command: str, environment: str) -> bool:
        """Check if command is being run in the correct environment."""
        # This is a heuristic check
        if environment in command or f"cd {environment}" in command:
            return True
        return False
    
    def run_pre_validation(self, environment: str, file_path: str = None) -> Tuple[bool, List[str]]:
        """Run pre-execution validation for the environment."""
        issues = []
        
        if environment not in self.quality_rules:
            return True, []
        
        rules = self.quality_rules[environment]
        env_path = self.environments[environment]
        
        # Check if environment exists
        if not env_path.exists():
            issues.append(f"❌ Environment directory not found: {env_path}")
            return False, issues
        
        # Check required tools (quick validation)
        try:
            # Try to run a simple devbox command to verify environment
            result = subprocess.run(
                ["devbox", "version"],
                cwd=env_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                issues.append(f"⚠️ Devbox not accessible in {environment}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            issues.append(f"⚠️ Could not verify devbox availability in {environment}")
        
        return len(issues) == 0, issues
    
    def suggest_improvements(self, issues: List[str], environment: str) -> List[str]:
        """Suggest improvements based on identified issues."""
        suggestions = []
        
        if not issues:
            return suggestions
        
        # Environment-specific suggestions
        env_rules = self.quality_rules.get(environment, {})
        
        suggestions.append(f"🔧 To fix {environment} issues:")
        
        if environment == "python-env":
            suggestions.append("   • Run: cd python-env && devbox run lint")
            suggestions.append("   • Format: cd python-env && devbox run format") 
            suggestions.append("   • Type check: cd python-env && devbox run mypy")
        elif environment == "typescript-env":
            suggestions.append("   • Run: cd typescript-env && devbox run lint")
            suggestions.append("   • Format: cd typescript-env && devbox run format")
            suggestions.append("   • Type check: cd typescript-env && devbox run tsc")
        elif environment == "rust-env":
            suggestions.append("   • Run: cd rust-env && devbox run lint")
            suggestions.append("   • Format: cd rust-env && devbox run format")
            suggestions.append("   • Check: cd rust-env && cargo check")
        elif environment == "go-env":
            suggestions.append("   • Run: cd go-env && devbox run lint")
            suggestions.append("   • Format: cd go-env && devbox run format")
            suggestions.append("   • Test: cd go-env && devbox run test")
        elif environment == "nushell-env":
            suggestions.append("   • Run: cd nushell-env && devbox run check")
            suggestions.append("   • Validate: nu nushell-env/scripts/validate-all.nu")
        
        return suggestions
    
    def process_tool_event(self, hook_data: dict) -> dict:
        """Process tool event and return validation result."""
        try:
            tool_name = hook_data.get("tool_name", "")
            tool_input = hook_data.get("tool_input", {})
            
            result = {
                "continue": True,
                "issues": [],
                "suggestions": [],
                "environment": None
            }
            
            # Handle different tool types
            if tool_name == "Bash":
                command = tool_input.get("command", "")
                result["issues"] = self.validate_bash_command(command)
                
            elif tool_name in ["Edit", "MultiEdit", "Write"]:
                file_path = tool_input.get("file_path", "")
                content = tool_input.get("content", "")
                
                environment = self.detect_environment(file_path)
                if environment:
                    result["environment"] = environment
                    result["issues"] = self.validate_file_content(file_path, content, environment)
                    
                    # Run pre-validation
                    validation_passed, validation_issues = self.run_pre_validation(environment, file_path)
                    result["issues"].extend(validation_issues)
            
            # Generate suggestions if there are issues
            if result["issues"] and result["environment"]:
                result["suggestions"] = self.suggest_improvements(result["issues"], result["environment"])
            
            return result
            
        except Exception as e:
            return {
                "continue": True,
                "issues": [f"⚠️ Quality validation failed: {e}"],
                "suggestions": [],
                "environment": None
            }

def main():
    """Main hook entry point."""
    try:
        # Read hook input from stdin
        hook_input = json.load(sys.stdin)
        
        # Get project root
        project_root = Path.cwd()
        
        # Initialize validator
        validator = QualityGatesValidator(project_root)
        
        # Process the event
        result = validator.process_tool_event(hook_input)
        
        # Output issues to stderr if any (for blocking)
        if result["issues"]:
            print("🔍 Quality Gate Issues Found:")
            for issue in result["issues"]:
                print(f"  {issue}")
            
            # Show suggestions
            if result["suggestions"]:
                print("\n💡 Suggestions:")
                for suggestion in result["suggestions"]:
                    print(f"  {suggestion}")
            
            print("\n🎯 Quality gates help maintain polyglot development standards")
            
            # If critical issues, suggest blocking (exit code 2)
            critical_issues = [issue for issue in result["issues"] if "❌" in issue]
            if critical_issues:
                print("\n⚠️ Critical issues found - consider reviewing before proceeding", file=sys.stderr)
                sys.exit(2)  # Block execution
        
        # Return success (continue execution)
        sys.exit(0)
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Quality Gates Hook failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()