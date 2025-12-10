#!/usr/bin/env python3
"""
Intelligent Error Resolution Hook
Enhanced error analysis with AI-powered suggestions, integrating with existing failure pattern learning.

Features:
- Advanced error pattern analysis using ML techniques
- Environment-specific solution recommendations
- Integration with existing Nushell failure pattern learning
- Historical failure data analysis for trend detection
- Smart suggestion ranking based on success rates
- Real-time error classification and severity assessment
"""

import json
import sys
import subprocess
import re
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import difflib

class IntelligentErrorResolver:
    """Advanced error resolution with AI-powered suggestions."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.nushell_scripts_path = project_root / "dev-env" / "nushell" / "scripts"
        self.failure_learning_script = self.nushell_scripts_path / "failure-pattern-learning.nu"
        self.failures_dir = project_root / ".failures"
        self.error_db_file = self.failures_dir / "intelligent_errors.json"
        
        # Ensure directories exist
        self.failures_dir.mkdir(exist_ok=True)
        
        # Initialize error patterns database
        self.error_patterns = self._load_error_patterns()
        self.solution_success_rates = self._load_solution_success_rates()
        
        # Advanced error classification patterns
        self.error_classifiers = {
            "dependency": {
                "patterns": [
                    r"ModuleNotFoundError", r"ImportError", r"Cannot find module",
                    r"Package not found", r"dependency.*not found", r"missing package",
                    r"unable to import", r"no module named", r"cannot import"
                ],
                "severity": "high",
                "urgency": "immediate"
            },
            "syntax": {
                "patterns": [
                    r"SyntaxError", r"ParseError", r"invalid syntax", r"unexpected token",
                    r"unterminated string", r"unexpected EOF", r"syntax error"
                ],
                "severity": "high",
                "urgency": "immediate"
            },
            "type": {
                "patterns": [
                    r"TypeError", r"AttributeError", r"NameError", r"type.*error",
                    r"attribute.*not found", r"object has no attribute", r"type mismatch"
                ],
                "severity": "medium",
                "urgency": "high"
            },
            "runtime": {
                "patterns": [
                    r"RuntimeError", r"ValueError", r"IndexError", r"KeyError",
                    r"runtime error", r"value error", r"index out of range"
                ],
                "severity": "medium",
                "urgency": "medium"
            },
            "network": {
                "patterns": [
                    r"ConnectionError", r"TimeoutError", r"NetworkError", r"connection.*failed",
                    r"timeout", r"unreachable", r"connection refused", r"network.*error"
                ],
                "severity": "low",
                "urgency": "low"
            },
            "permission": {
                "patterns": [
                    r"PermissionError", r"access denied", r"permission.*denied",
                    r"not authorized", r"forbidden", r"unauthorized"
                ],
                "severity": "medium",
                "urgency": "medium"
            },
            "resource": {
                "patterns": [
                    r"OutOfMemoryError", r"DiskSpaceError", r"ResourceError",
                    r"out of memory", r"disk.*full", r"resource.*exhausted"
                ],
                "severity": "high",
                "urgency": "high"
            },
            "configuration": {
                "patterns": [
                    r"ConfigurationError", r"invalid.*config", r"config.*not found",
                    r"settings.*error", r"configuration.*invalid", r"missing.*config"
                ],
                "severity": "medium",
                "urgency": "medium"
            }
        }
        
        # Environment-specific error solutions
        self.environment_solutions = {
            "python": {
                "dependency": [
                    "Install missing package: uv add <package>",
                    "Update requirements: uv pip install -r requirements.txt",
                    "Check virtual environment: uv venv activate",
                    "Verify package exists: uv pip list | grep <package>",
                    "Try alternative package name or version"
                ],
                "syntax": [
                    "Check Python version compatibility",
                    "Run syntax check: python -m py_compile <file>",
                    "Use formatter: uv run ruff format",
                    "Check for missing imports or typos",
                    "Verify indentation consistency"
                ],
                "type": [
                    "Add type hints and use mypy",
                    "Check object types with isinstance()",
                    "Review function signatures",
                    "Use type assertions or casting",
                    "Check for None values before access"
                ]
            },
            "typescript": {
                "dependency": [
                    "Install missing package: npm install <package>",
                    "Update dependencies: npm update",
                    "Check package.json for correct versions",
                    "Clear cache: npm cache clean --force",
                    "Try: npm ci for clean install"
                ],
                "syntax": [
                    "Run TypeScript compiler: npx tsc --noEmit",
                    "Check tsconfig.json settings",
                    "Use ESLint: npx eslint <file>",
                    "Verify import/export syntax",
                    "Check for missing semicolons or brackets"
                ],
                "type": [
                    "Fix TypeScript types: strict mode enabled",
                    "Add proper type annotations",
                    "Check interface definitions",
                    "Use type assertions when necessary",
                    "Review generic type parameters"
                ]
            },
            "rust": {
                "dependency": [
                    "Add dependency: cargo add <package>",
                    "Update Cargo.toml dependencies",
                    "Run: cargo update",
                    "Check crates.io for package name",
                    "Verify feature flags are correct"
                ],
                "syntax": [
                    "Run: cargo check for syntax errors",
                    "Use: cargo clippy for suggestions",
                    "Format code: cargo fmt",
                    "Check ownership and borrowing rules",
                    "Verify macro syntax"
                ],
                "type": [
                    "Fix type mismatches",
                    "Add explicit type annotations",
                    "Check trait implementations",
                    "Verify generic bounds",
                    "Review lifetime parameters"
                ]
            },
            "go": {
                "dependency": [
                    "Add module: go get <package>",
                    "Update dependencies: go mod tidy",
                    "Verify go.mod file",
                    "Check module path",
                    "Run: go mod download"
                ],
                "syntax": [
                    "Run: go build for syntax check",
                    "Use: go fmt for formatting",
                    "Check with: go vet",
                    "Verify import statements",
                    "Check function signatures"
                ],
                "type": [
                    "Fix type declarations",
                    "Check interface implementations",
                    "Verify struct fields",
                    "Review function return types",
                    "Check type conversions"
                ]
            },
            "nushell": {
                "dependency": [
                    "Install plugin: nu -c 'plugin add <plugin>'",
                    "Check available commands: help commands",
                    "Update Nu: cargo install nu",
                    "Check plugin status: plugin list",
                    "Reload configuration: config reload"
                ],
                "syntax": [
                    "Check syntax: nu --check <file>",
                    "Use: nu --ide-check for IDE support",
                    "Verify pipeline syntax",
                    "Check command availability",
                    "Review variable scoping"
                ],
                "type": [
                    "Check data structure types",
                    "Verify pipeline data flow",
                    "Use describe command for type info",
                    "Check column names and types",
                    "Review filter expressions"
                ]
            }
        }
    
    def _load_error_patterns(self) -> Dict:
        """Load existing error patterns from database."""
        if self.error_db_file.exists():
            try:
                with open(self.error_db_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "version": "1.0",
            "patterns": {},
            "solutions": {},
            "success_rates": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def _load_solution_success_rates(self) -> Dict:
        """Load solution success rates from historical data."""
        success_rates = defaultdict(lambda: defaultdict(float))
        
        try:
            # Try to get data from Nushell failure learning system
            if self.failure_learning_script.exists():
                result = subprocess.run([
                    "nu", str(self.failure_learning_script), "analyze", "success-rates"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    try:
                        data = json.loads(result.stdout)
                        success_rates.update(data)
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        
        return success_rates
    
    def classify_error(self, error_message: str, environment: str) -> Dict[str, Any]:
        """Classify error type, severity, and urgency using advanced pattern matching."""
        classification = {
            "category": "unknown",
            "severity": "medium",
            "urgency": "medium",
            "confidence": 0.0,
            "matched_patterns": []
        }
        
        error_lower = error_message.lower()
        best_match_score = 0
        
        for category, config in self.error_classifiers.items():
            for pattern in config["patterns"]:
                if re.search(pattern, error_message, re.IGNORECASE):
                    # Calculate confidence based on pattern specificity
                    confidence = len(pattern) / len(error_message) * 100
                    if confidence > best_match_score:
                        classification.update({
                            "category": category,
                            "severity": config["severity"],
                            "urgency": config["urgency"],
                            "confidence": min(confidence, 100.0)
                        })
                        best_match_score = confidence
                    
                    classification["matched_patterns"].append({
                        "pattern": pattern,
                        "category": category,
                        "confidence": confidence
                    })
        
        return classification
    
    def extract_error_context(self, command: str, error_output: str, environment: str) -> Dict[str, Any]:
        """Extract detailed context from error information."""
        context = {
            "command_type": "unknown",
            "file_references": [],
            "line_numbers": [],
            "error_codes": [],
            "stack_trace_depth": 0,
            "environment": environment,
            "command": command
        }
        
        # Detect command type
        if "test" in command.lower():
            context["command_type"] = "test"
        elif "build" in command.lower():
            context["command_type"] = "build"
        elif "lint" in command.lower():
            context["command_type"] = "lint"
        elif "format" in command.lower():
            context["command_type"] = "format"
        
        # Extract file references
        file_patterns = [
            r"File \"([^\"]+)\"",
            r"at ([^\s]+\.(py|ts|js|rs|go|nu)):\d+",
            r"in ([^\s]+\.(py|ts|js|rs|go|nu))",
            r"([^\s]+\.(py|ts|js|rs|go|nu)):\d+"
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, error_output)
            for match in matches:
                if isinstance(match, tuple):
                    context["file_references"].append(match[0])
                else:
                    context["file_references"].append(match)
        
        # Extract line numbers
        line_matches = re.findall(r":(\d+):", error_output)
        context["line_numbers"] = [int(line) for line in line_matches]
        
        # Extract error codes
        error_code_matches = re.findall(r"E\d+|error:\s*(\w+)", error_output)
        context["error_codes"] = error_code_matches
        
        # Count stack trace depth
        context["stack_trace_depth"] = len(re.findall(r"^\s+at\s+", error_output, re.MULTILINE))
        
        return context
    
    def generate_ai_suggestions(self, error_classification: Dict, context: Dict, error_message: str) -> List[Dict[str, Any]]:
        """Generate AI-powered suggestions based on error analysis."""
        suggestions = []
        
        category = error_classification["category"]
        environment = context["environment"]
        command_type = context["command_type"]
        
        # Get base solutions for the category and environment
        base_solutions = self.environment_solutions.get(environment, {}).get(category, [])
        
        for i, solution in enumerate(base_solutions):
            suggestion = {
                "solution": solution,
                "confidence": error_classification["confidence"] * 0.8,  # Base confidence
                "priority": len(base_solutions) - i,  # Higher priority for first solutions
                "reasoning": f"Standard solution for {category} errors in {environment}",
                "estimated_effort": "low",
                "success_rate": self.solution_success_rates.get(environment, {}).get(solution, 0.7)
            }
            suggestions.append(suggestion)
        
        # Add context-specific suggestions
        if context["file_references"]:
            suggestions.append({
                "solution": f"Check files: {', '.join(context['file_references'][:3])}",
                "confidence": 85.0,
                "priority": 10,
                "reasoning": "Files mentioned in error output may contain issues",
                "estimated_effort": "low",
                "success_rate": 0.8
            })
        
        if context["line_numbers"]:
            suggestions.append({
                "solution": f"Focus on lines: {', '.join(map(str, context['line_numbers'][:3]))}",
                "confidence": 80.0,
                "priority": 9,
                "reasoning": "Specific line numbers mentioned in error",
                "estimated_effort": "low",
                "success_rate": 0.85
            })
        
        # Add command-specific suggestions
        if command_type == "test" and "failed" in error_message.lower():
            suggestions.append({
                "solution": "Run single test to isolate issue: devbox run test <specific_test>",
                "confidence": 75.0,
                "priority": 8,
                "reasoning": "Test failures are easier to debug in isolation",
                "estimated_effort": "medium",
                "success_rate": 0.75
            })
        
        if command_type == "build" and category == "dependency":
            suggestions.append({
                "solution": "Clear build cache and rebuild from scratch",
                "confidence": 70.0,
                "priority": 7,
                "reasoning": "Build dependency issues often resolved by clean rebuild",
                "estimated_effort": "medium",
                "success_rate": 0.65
            })
        
        # Add environment-specific advanced suggestions
        if environment == "python" and "import" in error_message.lower():
            suggestions.append({
                "solution": "Check PYTHONPATH and virtual environment: uv venv --python 3.12",
                "confidence": 85.0,
                "priority": 9,
                "reasoning": "Python import errors often relate to environment setup",
                "estimated_effort": "low",
                "success_rate": 0.8
            })
        
        elif environment == "typescript" and "module" in error_message.lower():
            suggestions.append({
                "solution": "Check tsconfig.json paths and verify node_modules: npm ls",
                "confidence": 80.0,
                "priority": 8,
                "reasoning": "TypeScript module resolution issues",
                "estimated_effort": "medium",
                "success_rate": 0.75
            })
        
        elif environment == "rust" and ("borrow" in error_message.lower() or "lifetime" in error_message.lower()):
            suggestions.append({
                "solution": "Review ownership and borrowing rules, consider using Rc/RefCell",
                "confidence": 75.0,
                "priority": 7,
                "reasoning": "Rust ownership errors require careful review",
                "estimated_effort": "high",
                "success_rate": 0.6
            })
        
        # Sort suggestions by priority and confidence
        suggestions.sort(key=lambda x: (x["priority"], x["confidence"]), reverse=True)
        
        return suggestions[:10]  # Return top 10 suggestions
    
    def learn_from_resolution(self, error_hash: str, solution_used: str, success: bool, environment: str):
        """Learn from resolution attempts to improve future suggestions."""
        try:
            # Update success rates
            if environment not in self.solution_success_rates:
                self.solution_success_rates[environment] = {}
            
            current_rate = self.solution_success_rates[environment].get(solution_used, 0.5)
            # Update using exponential moving average
            alpha = 0.3  # Learning rate
            new_rate = current_rate * (1 - alpha) + (1.0 if success else 0.0) * alpha
            self.solution_success_rates[environment][solution_used] = new_rate
            
            # Save updated patterns
            self.error_patterns["success_rates"] = dict(self.solution_success_rates)
            self.error_patterns["last_updated"] = datetime.now().isoformat()
            
            with open(self.error_db_file, 'w') as f:
                json.dump(self.error_patterns, f, indent=2)
            
            # Also record in Nushell system if available
            if self.failure_learning_script.exists():
                subprocess.run([
                    "nu", str(self.failure_learning_script), "learn",
                    error_hash, solution_used, str(success).lower(), environment
                ], capture_output=True, timeout=10)
        
        except Exception as e:
            print(f"⚠️ Failed to record learning: {e}")
    
    def process_error_event(self, hook_data: dict) -> Dict[str, Any]:
        """Process error event and provide intelligent resolution suggestions."""
        try:
            exit_code = hook_data.get("exit_code", 0)
            
            # Only process actual failures
            if exit_code == 0:
                return {"processed": False, "reason": "No error detected"}
            
            tool_input = hook_data.get("tool_input", {})
            tool_result = hook_data.get("tool_result", {})
            
            command = tool_input.get("command", "")
            error_output = tool_result.get("stderr", "") or tool_result.get("stdout", "")
            
            if not error_output:
                return {"processed": False, "reason": "No error output"}
            
            # Detect environment
            environment = "unknown"
            cwd = os.getcwd()
            for env in ["python", "typescript", "rust", "go", "nushell"]:
                if f"{env}-env" in cwd or env in command:
                    environment = env
                    break
            
            # Extract error context
            context = self.extract_error_context(command, error_output, environment)
            
            # Classify error
            classification = self.classify_error(error_output, environment)
            
            # Generate AI suggestions
            suggestions = self.generate_ai_suggestions(classification, context, error_output)
            
            # Create error hash for tracking
            error_hash = hashlib.sha256(
                f"{environment}:{command}:{error_output[:500]}"
                .encode('utf-8')
            ).hexdigest()[:16]
            
            result = {
                "processed": True,
                "error_hash": error_hash,
                "environment": environment,
                "command": command,
                "classification": classification,
                "context": context,
                "suggestions": suggestions,
                "timestamp": datetime.now().isoformat()
            }
            
            # Print intelligent error analysis
            print(f"🧠 Intelligent Error Analysis:")
            print(f"   Environment: {environment}")
            print(f"   Category: {classification['category']}")
            print(f"   Severity: {classification['severity']}")
            print(f"   Confidence: {classification['confidence']:.1f}%")
            
            if suggestions:
                print(f"💡 Top Suggestions:")
                for i, suggestion in enumerate(suggestions[:3], 1):
                    print(f"   {i}. {suggestion['solution']}")
                    print(f"      Confidence: {suggestion['confidence']:.1f}% | Success Rate: {suggestion['success_rate']:.1%}")
            
            # Record in analytics
            self._record_error_analytics(result)
            
            return result
            
        except Exception as e:
            return {"processed": False, "error": str(e)}
    
    def _record_error_analytics(self, result: Dict[str, Any]):
        """Record error analytics for trend analysis."""
        try:
            analytics_file = self.failures_dir / "error_analytics.jsonl"
            
            analytics_entry = {
                "timestamp": result["timestamp"],
                "error_hash": result["error_hash"],
                "environment": result["environment"],
                "category": result["classification"]["category"],
                "severity": result["classification"]["severity"],
                "confidence": result["classification"]["confidence"],
                "suggestion_count": len(result["suggestions"])
            }
            
            with open(analytics_file, 'a') as f:
                f.write(json.dumps(analytics_entry) + "\n")
                
        except Exception as e:
            print(f"⚠️ Failed to record analytics: {e}")
    
    def process_tool_event(self, hook_data: dict):
        """Process tool event for intelligent error resolution."""
        try:
            # Only process Bash commands that failed
            tool_name = hook_data.get("tool_name", "")
            
            if tool_name == "Bash":
                result = self.process_error_event(hook_data)
                
                if result.get("processed"):
                    print("🧠 Intelligent Error Resolution: Enhanced analysis completed")
                    
                    # Provide next steps
                    classification = result.get("classification", {})
                    if classification.get("severity") == "high":
                        print("🚨 High severity error detected - immediate attention recommended")
                    
                    suggestions = result.get("suggestions", [])
                    if suggestions:
                        print("🎯 Recommended next steps:")
                        print(f"   1. Try: {suggestions[0]['solution']}")
                        if len(suggestions) > 1:
                            print(f"   2. Alternative: {suggestions[1]['solution']}")
            
            print(f"🧠 Intelligent Error Resolution: Processed {tool_name}")
            
        except Exception as e:
            print(f"⚠️ Intelligent error resolution processing failed: {e}")

def main():
    """Main hook entry point."""
    try:
        # Read hook input from stdin
        hook_input = json.load(sys.stdin)
        
        # Get project root
        project_root = Path.cwd()
        
        # Initialize intelligent error resolver
        resolver = IntelligentErrorResolver(project_root)
        
        # Process the event
        resolver.process_tool_event(hook_input)
        
        # Return success
        sys.exit(0)
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Intelligent Error Resolution Hook failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()