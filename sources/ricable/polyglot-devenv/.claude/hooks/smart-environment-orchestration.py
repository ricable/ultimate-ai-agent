#!/usr/bin/env python3
"""
Smart Environment Orchestration Hook
Automatically orchestrates DevPod containers and environment switching based on file context and usage patterns.

Features:
- Intelligent DevPod provisioning based on file context
- Smart environment switching recommendations
- Resource optimization using existing monitoring
- Integration with centralized DevPod management
- Usage pattern learning for proactive provisioning
- Multi-environment project coordination
"""

import json
import sys
import subprocess
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import hashlib

class SmartEnvironmentOrchestrator:
    """Intelligent environment orchestration and DevPod management."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.devpod_management_script = project_root / "host-tooling" / "devpod-management" / "manage-devpod.nu"
        self.orchestration_state_file = project_root / ".claude" / "orchestration_state.json"
        self.usage_analytics_file = project_root / ".claude" / "environment_usage.jsonl"
        
        # Environment detection patterns
        self.file_environment_map = {
            ".py": "python",
            ".ts": "typescript", ".tsx": "typescript", ".js": "typescript", ".jsx": "typescript",
            ".rs": "rust",
            ".go": "go",
            ".nu": "nushell",
            ".md": None,  # Could be any environment based on content
            ".json": None, ".yaml": None, ".yml": None, ".toml": None
        }
        
        self.directory_environment_map = {
            "python-env": "python",
            "typescript-env": "typescript", 
            "rust-env": "rust",
            "go-env": "go",
            "nushell-env": "nushell"
        }
        
        # Environment resource requirements
        self.environment_resources = {
            "python": {"memory_mb": 2048, "cpu_cores": 2, "startup_time": 30},
            "typescript": {"memory_mb": 4096, "cpu_cores": 2, "startup_time": 45},
            "rust": {"memory_mb": 8192, "cpu_cores": 4, "startup_time": 60},
            "go": {"memory_mb": 1024, "cpu_cores": 2, "startup_time": 20},
            "nushell": {"memory_mb": 512, "cpu_cores": 1, "startup_time": 10}
        }
        
        # Load orchestration state
        self.orchestration_state = self._load_orchestration_state()
        
        # Session tracking
        self.current_session = {
            "start_time": time.time(),
            "environments_used": set(),
            "files_accessed": [],
            "operations_performed": []
        }
    
    def _load_orchestration_state(self) -> Dict:
        """Load current orchestration state."""
        if self.orchestration_state_file.exists():
            try:
                with open(self.orchestration_state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "version": "1.0",
            "active_environments": {},
            "provisioned_devpods": {},
            "usage_patterns": {},
            "last_optimization": None,
            "session_history": []
        }
    
    def _save_orchestration_state(self):
        """Save current orchestration state."""
        try:
            self.orchestration_state["last_updated"] = datetime.now().isoformat()
            with open(self.orchestration_state_file, 'w') as f:
                json.dump(self.orchestration_state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save orchestration state: {e}")
    
    def detect_environment_from_file(self, file_path: str) -> Optional[str]:
        """Detect target environment from file path and content."""
        path = Path(file_path)
        
        # Check directory context first
        for dir_pattern, env in self.directory_environment_map.items():
            if dir_pattern in str(path):
                return env
        
        # Check file extension
        suffix = path.suffix.lower()
        if suffix in self.file_environment_map:
            env = self.file_environment_map[suffix]
            if env:
                return env
        
        # For ambiguous files, try to detect from content
        if suffix in [".md", ".json", ".yaml", ".yml", ".toml"]:
            return self._detect_environment_from_content(path)
        
        return None
    
    def _detect_environment_from_content(self, file_path: Path) -> Optional[str]:
        """Detect environment from file content for ambiguous files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
            
            # Keywords that suggest specific environments
            environment_keywords = {
                "python": ["python", "pip", "pytest", "django", "flask", "fastapi", "uv", "requirements.txt"],
                "typescript": ["typescript", "react", "next", "npm", "node", "jest", "webpack", "package.json"],
                "rust": ["rust", "cargo", "tokio", "serde", "cargo.toml", "clippy"],
                "go": ["golang", "go mod", "go.mod", "goroutine", "gofmt"],
                "nushell": ["nushell", "nu ", ".nu", "pipeline", "table"]
            }
            
            scores = {}
            for env, keywords in environment_keywords.items():
                score = sum(1 for keyword in keywords if keyword in content)
                if score > 0:
                    scores[env] = score
            
            # Return environment with highest score
            if scores:
                return max(scores, key=scores.get)
                
        except Exception:
            pass
        
        return None
    
    def get_devpod_status(self, environment: str = None) -> Dict[str, Any]:
        """Get current DevPod status using centralized management."""
        try:
            cmd = ["nu", str(self.devpod_management_script), "status"]
            if environment:
                cmd.append(environment)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                # Parse the output to extract DevPod information
                status_info = {
                    "success": True,
                    "environments": {},
                    "total_workspaces": 0,
                    "active_workspaces": 0
                }
                
                # Basic parsing of status output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'workspace' in line.lower():
                        status_info["total_workspaces"] += 1
                        if 'running' in line.lower() or 'active' in line.lower():
                            status_info["active_workspaces"] += 1
                
                return status_info
            else:
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def provision_devpod_if_needed(self, environment: str, urgency: str = "normal") -> Dict[str, Any]:
        """Provision DevPod workspace if needed using centralized management."""
        try:
            # Check if already provisioned
            status = self.get_devpod_status(environment)
            if status.get("success") and status.get("active_workspaces", 0) > 0:
                return {
                    "action": "none",
                    "reason": f"DevPod already active for {environment}",
                    "success": True
                }
            
            print(f"🚀 Smart Orchestration: Provisioning DevPod for {environment}")
            
            # Use centralized DevPod management
            cmd = ["nu", str(self.devpod_management_script), "provision", environment]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for provisioning
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                provision_result = {
                    "action": "provisioned",
                    "environment": environment,
                    "success": True,
                    "urgency": urgency,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Update orchestration state
                self.orchestration_state["provisioned_devpods"][environment] = provision_result
                self._save_orchestration_state()
                
                print(f"✅ DevPod provisioned successfully for {environment}")
                return provision_result
            else:
                return {
                    "action": "failed",
                    "environment": environment,
                    "success": False,
                    "error": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                "action": "timeout",
                "environment": environment,
                "success": False,
                "error": "Provisioning timeout"
            }
        except Exception as e:
            return {
                "action": "error",
                "environment": environment,
                "success": False,
                "error": str(e)
            }
    
    def analyze_environment_usage_patterns(self) -> Dict[str, Any]:
        """Analyze usage patterns to optimize environment provisioning."""
        patterns = {
            "frequent_environments": [],
            "environment_transitions": {},
            "peak_usage_times": {},
            "resource_optimization_suggestions": []
        }
        
        try:
            # Load usage history
            usage_history = []
            if self.usage_analytics_file.exists():
                with open(self.usage_analytics_file, 'r') as f:
                    for line in f:
                        try:
                            usage_history.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
            
            if not usage_history:
                return patterns
            
            # Analyze frequent environments
            env_counter = Counter()
            for entry in usage_history[-100:]:  # Last 100 entries
                env = entry.get("environment")
                if env:
                    env_counter[env] += 1
            
            patterns["frequent_environments"] = [
                {"environment": env, "count": count}
                for env, count in env_counter.most_common(3)
            ]
            
            # Analyze environment transitions
            transitions = defaultdict(int)
            prev_env = None
            for entry in usage_history[-50:]:  # Last 50 entries
                env = entry.get("environment")
                if prev_env and env and prev_env != env:
                    transitions[f"{prev_env}->{env}"] += 1
                prev_env = env
            
            patterns["environment_transitions"] = dict(transitions)
            
            # Generate optimization suggestions
            if env_counter:
                most_used = env_counter.most_common(1)[0][0]
                patterns["resource_optimization_suggestions"].append(
                    f"Consider keeping {most_used} environment always active"
                )
            
            if len(env_counter) > 3:
                patterns["resource_optimization_suggestions"].append(
                    "Multi-environment usage detected - consider resource monitoring"
                )
            
        except Exception as e:
            patterns["analysis_error"] = str(e)
        
        return patterns
    
    def optimize_environment_resources(self) -> List[str]:
        """Provide resource optimization recommendations."""
        recommendations = []
        
        try:
            # Analyze current DevPod status
            status = self.get_devpod_status()
            active_workspaces = status.get("active_workspaces", 0)
            
            if active_workspaces > 5:
                recommendations.append(
                    "⚠️ Many active workspaces detected - consider stopping unused ones"
                )
            
            # Check usage patterns
            patterns = self.analyze_environment_usage_patterns()
            frequent_envs = patterns.get("frequent_environments", [])
            
            if len(frequent_envs) == 1:
                env = frequent_envs[0]["environment"]
                recommendations.append(
                    f"💡 Single environment usage ({env}) - consider native devbox for better performance"
                )
            
            elif len(frequent_envs) > 3:
                recommendations.append(
                    "🔄 Multi-environment workflow - DevPod containerization is optimal"
                )
            
            # Resource-specific recommendations
            resource_intensive = ["rust", "typescript"]
            for env_data in frequent_envs:
                env = env_data["environment"]
                if env in resource_intensive:
                    recommendations.append(
                        f"🔧 {env.title()} detected - ensure adequate memory allocation"
                    )
            
        except Exception as e:
            recommendations.append(f"⚠️ Optimization analysis failed: {e}")
        
        return recommendations
    
    def record_environment_usage(self, environment: str, operation: str, file_path: str = ""):
        """Record environment usage for pattern analysis."""
        try:
            usage_entry = {
                "timestamp": datetime.now().isoformat(),
                "environment": environment,
                "operation": operation,
                "file_path": file_path,
                "session_id": hashlib.sha256(
                    f"{self.current_session['start_time']}"
                    .encode('utf-8')
                ).hexdigest()[:16]
            }
            
            with open(self.usage_analytics_file, 'a') as f:
                f.write(json.dumps(usage_entry) + "\n")
                
        except Exception as e:
            print(f"⚠️ Failed to record usage: {e}")
    
    def suggest_environment_switch(self, target_environment: str, current_context: str) -> Dict[str, Any]:
        """Suggest optimal environment switching strategy."""
        suggestion = {
            "should_switch": False,
            "method": "none",
            "commands": [],
            "estimated_time": 0,
            "reasoning": ""
        }
        
        try:
            # Check if DevPod is already available
            status = self.get_devpod_status(target_environment)
            
            if status.get("success") and status.get("active_workspaces", 0) > 0:
                suggestion.update({
                    "should_switch": True,
                    "method": "devpod_connect",
                    "commands": [f"devpod ssh {target_environment}-workspace"],
                    "estimated_time": 5,
                    "reasoning": f"DevPod already available for {target_environment}"
                })
            else:
                # Suggest provisioning or native environment
                resources = self.environment_resources.get(target_environment, {})
                startup_time = resources.get("startup_time", 30)
                
                if startup_time <= 30:  # Quick environments
                    suggestion.update({
                        "should_switch": True,
                        "method": "devpod_provision",
                        "commands": [
                            f"nu host-tooling/devpod-management/manage-devpod.nu provision {target_environment}"
                        ],
                        "estimated_time": startup_time + 30,
                        "reasoning": f"Quick provisioning recommended for {target_environment}"
                    })
                else:  # Resource-intensive environments
                    suggestion.update({
                        "should_switch": True,
                        "method": "native_devbox",
                        "commands": [
                            f"cd dev-env/{target_environment}",
                            "devbox shell"
                        ],
                        "estimated_time": 10,
                        "reasoning": f"Native devbox recommended for resource-intensive {target_environment}"
                    })
            
        except Exception as e:
            suggestion["error"] = str(e)
        
        return suggestion
    
    def process_file_operation(self, file_path: str, operation: str) -> Dict[str, Any]:
        """Process file operation and provide smart orchestration."""
        result = {
            "processed": False,
            "environment": None,
            "actions_taken": [],
            "recommendations": []
        }
        
        try:
            # Detect environment from file
            environment = self.detect_environment_from_file(file_path)
            
            if not environment:
                return result
            
            result.update({
                "processed": True,
                "environment": environment,
                "file_path": file_path,
                "operation": operation
            })
            
            # Record usage
            self.record_environment_usage(environment, operation, file_path)
            
            # Get current context
            current_dir = os.getcwd()
            current_env = None
            for dir_pattern, env in self.directory_environment_map.items():
                if dir_pattern in current_dir:
                    current_env = env
                    break
            
            # Smart orchestration logic
            if current_env != environment:
                # Suggest environment switch
                switch_suggestion = self.suggest_environment_switch(environment, current_dir)
                result["recommendations"].append({
                    "type": "environment_switch",
                    "suggestion": switch_suggestion
                })
                
                # Auto-provision if it's a quick environment and operation is significant
                if operation in ["Edit", "Write"] and environment in ["python", "go", "nushell"]:
                    provision_result = self.provision_devpod_if_needed(environment, "normal")
                    if provision_result.get("success"):
                        result["actions_taken"].append(provision_result)
            
            # Check for multi-environment project patterns
            session_envs = set([environment])
            session_envs.update(self.current_session.get("environments_used", set()))
            
            if len(session_envs) > 2:
                result["recommendations"].append({
                    "type": "multi_environment_optimization",
                    "message": f"Multi-environment project detected: {', '.join(session_envs)}",
                    "suggestion": "Consider using DevPod for better isolation"
                })
            
            # Update session tracking
            self.current_session["environments_used"].add(environment)
            self.current_session["files_accessed"].append({
                "file": file_path,
                "environment": environment,
                "timestamp": datetime.now().isoformat()
            })
            
            # Print orchestration insights
            if result["actions_taken"]:
                print(f"🚀 Smart Orchestration: Auto-provisioned {environment}")
            
            if result["recommendations"]:
                rec = result["recommendations"][0]
                if rec["type"] == "environment_switch":
                    suggestion = rec["suggestion"]
                    if suggestion.get("should_switch"):
                        print(f"💡 Suggestion: Switch to {environment} environment")
                        print(f"   Method: {suggestion['method']}")
                        print(f"   Estimated time: {suggestion['estimated_time']}s")
            
            return result
            
        except Exception as e:
            result["error"] = str(e)
            return result
    
    def process_tool_event(self, hook_data: dict):
        """Process tool event for smart environment orchestration."""
        try:
            tool_name = hook_data.get("tool_name", "")
            tool_input = hook_data.get("tool_input", {})
            
            if tool_name in ["Edit", "MultiEdit", "Write", "Read"]:
                file_path = tool_input.get("file_path", "")
                
                if file_path:
                    result = self.process_file_operation(file_path, tool_name)
                    
                    if result.get("processed"):
                        environment = result.get("environment")
                        actions = result.get("actions_taken", [])
                        recommendations = result.get("recommendations", [])
                        
                        if actions or recommendations:
                            print(f"🔧 Smart Environment Orchestration: {environment}")
                            
                            # Show resource optimization tips periodically
                            if len(self.current_session["files_accessed"]) % 10 == 0:
                                optimizations = self.optimize_environment_resources()
                                if optimizations:
                                    print("💡 Resource optimization tips:")
                                    for tip in optimizations[:2]:
                                        print(f"   {tip}")
            
            print(f"🔧 Smart Environment Orchestration: Processed {tool_name}")
            
        except Exception as e:
            print(f"⚠️ Smart environment orchestration processing failed: {e}")

def main():
    """Main hook entry point."""
    try:
        # Read hook input from stdin
        hook_input = json.load(sys.stdin)
        
        # Get project root
        project_root = Path.cwd()
        
        # Initialize smart environment orchestrator
        orchestrator = SmartEnvironmentOrchestrator(project_root)
        
        # Process the event
        orchestrator.process_tool_event(hook_input)
        
        # Return success
        sys.exit(0)
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Smart Environment Orchestration Hook failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()