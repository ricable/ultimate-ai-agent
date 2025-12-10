#!/usr/bin/env python3
"""
DevPod Resource Management Hook
Smart container lifecycle management and load balancing for DevPod workspaces.

Features:
- Intelligent container provisioning and cleanup
- Resource usage monitoring and optimization
- Load balancing across available DevPod instances
- Integration with existing DevPod automation infrastructure
"""

import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import re
from typing import Dict, List, Optional, Tuple
import shutil

class DevPodResourceManager:
    """Manages DevPod resources intelligently."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.devpod_automation_path = project_root / "devpod-automation"
        self.scripts_path = self.devpod_automation_path / "scripts"
        
        # DevPod environment mapping
        self.devpod_environments = {
            "python-env": "python",
            "typescript-env": "typescript", 
            "rust-env": "rust",
            "go-env": "go",
            "nushell-env": "nushell"
        }
        
        # Resource thresholds
        self.resource_limits = {
            "max_containers_per_env": 5,
            "max_total_containers": 15,
            "idle_timeout_minutes": 30,
            "memory_threshold_mb": 8192,  # 8GB
            "cpu_threshold_percent": 80
        }
    
    def check_devpod_availability(self) -> bool:
        """Check if DevPod is available and configured."""
        try:
            result = subprocess.run(
                ["devpod", "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_active_workspaces(self) -> List[Dict]:
        """Get list of active DevPod workspaces."""
        try:
            result = subprocess.run(
                ["devpod", "list", "--output", "json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                try:
                    workspaces = json.loads(result.stdout)
                    return workspaces if isinstance(workspaces, list) else []
                except json.JSONDecodeError:
                    pass
            
            # Fallback: parse text output
            workspaces = []
            for line in result.stdout.split('\n'):
                if line.strip() and not line.startswith('NAME'):
                    parts = line.split()
                    if len(parts) >= 3:
                        workspaces.append({
                            "name": parts[0],
                            "status": parts[1],
                            "provider": parts[2] if len(parts) > 2 else "unknown"
                        })
            
            return workspaces
            
        except Exception as e:
            print(f"⚠️ Failed to get active workspaces: {e}")
            return []
    
    def get_workspace_resource_usage(self, workspace_name: str) -> Dict:
        """Get resource usage for a specific workspace."""
        try:
            # Use docker stats to get resource usage if workspace is running
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", 
                 "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n')[1:]:  # Skip header
                    if workspace_name in line:
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            return {
                                "container": parts[0].strip(),
                                "cpu_percent": parts[1].strip(),
                                "memory_usage": parts[2].strip()
                            }
            
            return {"container": workspace_name, "cpu_percent": "0%", "memory_usage": "0B / 0B"}
            
        except Exception as e:
            return {"error": str(e)}
    
    def cleanup_idle_workspaces(self) -> List[str]:
        """Clean up idle DevPod workspaces."""
        cleaned_workspaces = []
        
        try:
            workspaces = self.get_active_workspaces()
            
            for workspace in workspaces:
                if workspace.get("status") == "Running":
                    # Check if workspace is truly idle (simplified check)
                    usage = self.get_workspace_resource_usage(workspace["name"])
                    
                    cpu_percent = usage.get("cpu_percent", "0%").replace("%", "")
                    try:
                        cpu_value = float(cpu_percent)
                        if cpu_value < 1.0:  # Less than 1% CPU usage
                            print(f"🧹 Cleaning up idle workspace: {workspace['name']}")
                            
                            # Stop the workspace
                            result = subprocess.run(
                                ["devpod", "stop", workspace["name"]],
                                capture_output=True,
                                text=True,
                                timeout=60
                            )
                            
                            if result.returncode == 0:
                                cleaned_workspaces.append(workspace["name"])
                            else:
                                print(f"⚠️ Failed to stop workspace {workspace['name']}: {result.stderr}")
                                
                    except ValueError:
                        pass  # Could not parse CPU percentage
            
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")
        
        return cleaned_workspaces
    
    def optimize_resource_allocation(self) -> Dict:
        """Optimize resource allocation across DevPod workspaces."""
        optimization_result = {
            "actions_taken": [],
            "recommendations": [],
            "resource_status": {}
        }
        
        try:
            workspaces = self.get_active_workspaces()
            total_workspaces = len([w for w in workspaces if w.get("status") == "Running"])
            
            optimization_result["resource_status"] = {
                "total_workspaces": total_workspaces,
                "max_allowed": self.resource_limits["max_total_containers"]
            }
            
            # Check if we're at capacity
            if total_workspaces >= self.resource_limits["max_total_containers"]:
                cleaned = self.cleanup_idle_workspaces()
                if cleaned:
                    optimization_result["actions_taken"].append(f"Cleaned up {len(cleaned)} idle workspaces")
                else:
                    optimization_result["recommendations"].append(
                        "Consider manually reviewing workspace usage - at capacity"
                    )
            
            # Environment-specific optimizations
            env_counts = {}
            for workspace in workspaces:
                for env_name, devpod_name in self.devpod_environments.items():
                    if devpod_name in workspace.get("name", ""):
                        env_counts[env_name] = env_counts.get(env_name, 0) + 1
            
            for env_name, count in env_counts.items():
                if count > self.resource_limits["max_containers_per_env"]:
                    optimization_result["recommendations"].append(
                        f"Consider reducing {env_name} workspaces (current: {count}, limit: {self.resource_limits['max_containers_per_env']})"
                    )
            
        except Exception as e:
            optimization_result["actions_taken"].append(f"Optimization failed: {e}")
        
        return optimization_result
    
    def intelligent_provisioning(self, environment: str, count: int = 1) -> Dict:
        """Intelligently provision DevPod workspaces with load balancing."""
        result = {
            "success": False,
            "provisioned": [],
            "skipped": [],
            "error": None
        }
        
        try:
            # Check availability first
            if not self.check_devpod_availability():
                result["error"] = "DevPod not available"
                return result
            
            # Check current resource usage
            workspaces = self.get_active_workspaces()
            running_count = len([w for w in workspaces if w.get("status") == "Running"])
            
            # Calculate how many we can actually provision
            available_slots = self.resource_limits["max_total_containers"] - running_count
            actual_count = min(count, available_slots)
            
            if actual_count < count:
                result["skipped"].append(f"Reduced from {count} to {actual_count} due to resource limits")
            
            if actual_count == 0:
                result["error"] = "No available slots for new workspaces"
                return result
            
            # Use existing DevPod automation scripts
            devpod_env = self.devpod_environments.get(environment, "python")
            provision_script = self.scripts_path / f"provision-{devpod_env}.sh"
            
            if provision_script.exists():
                # Run the provision script
                cmd = ["bash", str(provision_script)]
                if actual_count > 1:
                    cmd.append(str(actual_count))
                
                provision_result = subprocess.run(
                    cmd,
                    cwd=self.devpod_automation_path,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout
                )
                
                if provision_result.returncode == 0:
                    result["success"] = True
                    result["provisioned"] = [f"{devpod_env}-workspace-{i}" for i in range(actual_count)]
                else:
                    result["error"] = f"Provisioning failed: {provision_result.stderr}"
            else:
                # Fallback: use direct DevPod commands
                print(f"🔄 Provisioning {actual_count} {environment} workspace(s)...")
                
                for i in range(actual_count):
                    workspace_name = f"polyglot-{devpod_env}-devpod-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{i+1}"
                    
                    provision_cmd = [
                        "devpod", "up",
                        workspace_name,
                        "--ide", "vscode",
                        "--source", str(self.project_root / environment)
                    ]
                    
                    provision_result = subprocess.run(
                        provision_cmd,
                        capture_output=True,
                        text=True,
                        timeout=180  # 3 minutes per workspace
                    )
                    
                    if provision_result.returncode == 0:
                        result["provisioned"].append(workspace_name)
                    else:
                        result["skipped"].append(f"Failed to provision workspace {i+1}: {provision_result.stderr}")
                
                result["success"] = len(result["provisioned"]) > 0
        
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def monitor_devpod_operations(self, command: str) -> Dict:
        """Monitor DevPod operations and provide insights."""
        monitoring_result = {
            "operation": "unknown",
            "environment": None,
            "recommendations": [],
            "resource_check": False
        }
        
        try:
            # Detect operation type
            if "devpod up" in command or "/devpod-" in command:
                monitoring_result["operation"] = "provision"
                monitoring_result["resource_check"] = True
                
                # Extract environment
                for env_name, devpod_name in self.devpod_environments.items():
                    if devpod_name in command or env_name in command:
                        monitoring_result["environment"] = env_name
                        break
                
                # Check if we should optimize before provisioning
                optimization = self.optimize_resource_allocation()
                if optimization["recommendations"]:
                    monitoring_result["recommendations"].extend(optimization["recommendations"])
                
            elif "devpod stop" in command or "devpod delete" in command:
                monitoring_result["operation"] = "cleanup"
                monitoring_result["recommendations"].append("Good! Cleaning up unused resources")
                
            elif "devpod list" in command:
                monitoring_result["operation"] = "status_check"
                
                # Provide status insights
                workspaces = self.get_active_workspaces()
                running_count = len([w for w in workspaces if w.get("status") == "Running"])
                
                if running_count > self.resource_limits["max_total_containers"] * 0.8:
                    monitoring_result["recommendations"].append(
                        f"High workspace usage ({running_count} active) - consider cleanup"
                    )
                elif running_count == 0:
                    monitoring_result["recommendations"].append(
                        "No active workspaces - ready for new development sessions"
                    )
            
        except Exception as e:
            monitoring_result["recommendations"].append(f"Monitoring failed: {e}")
        
        return monitoring_result
    
    def process_tool_event(self, hook_data: dict):
        """Process tool event for DevPod resource management."""
        try:
            tool_name = hook_data.get("tool_name", "")
            tool_input = hook_data.get("tool_input", {})
            
            if tool_name == "Bash":
                command = tool_input.get("command", "")
                
                # Monitor DevPod-related commands
                if "devpod" in command or "/devpod-" in command:
                    monitoring = self.monitor_devpod_operations(command)
                    
                    print(f"🐳 DevPod Operation: {monitoring['operation']}")
                    
                    if monitoring["environment"]:
                        print(f"   Environment: {monitoring['environment']}")
                    
                    if monitoring["resource_check"]:
                        optimization = self.optimize_resource_allocation()
                        if optimization["actions_taken"]:
                            print(f"   Resource actions: {', '.join(optimization['actions_taken'])}")
                    
                    if monitoring["recommendations"]:
                        print(f"   Recommendations:")
                        for rec in monitoring["recommendations"]:
                            print(f"     • {rec}")
                
                # Handle resource cleanup commands
                elif "docker" in command and ("stop" in command or "rm" in command):
                    print("🧹 Docker cleanup detected - good resource management!")
                    
            print(f"🎯 DevPod Resource Management: Processed {tool_name}")
            
        except Exception as e:
            print(f"⚠️ DevPod resource management failed: {e}")

def main():
    """Main hook entry point."""
    try:
        # Read hook input from stdin
        hook_input = json.load(sys.stdin)
        
        # Get project root
        project_root = Path.cwd()
        
        # Initialize resource manager
        manager = DevPodResourceManager(project_root)
        
        # Process the event
        manager.process_tool_event(hook_input)
        
        # Return success
        sys.exit(0)
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ DevPod Resource Management Hook failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()