#!/usr/bin/env python3
"""
Performance Analytics Integration Hook
Enhanced performance tracking and analytics for polyglot development environment.

Features:
- Advanced build time tracking across environments
- Resource usage analysis and optimization suggestions
- Performance regression detection
- Integration with existing Nushell analytics infrastructure
"""

import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import time
import psutil
import os
from typing import Dict, List, Optional, Tuple

class PerformanceAnalyticsIntegrator:
    """Integrates with existing performance analytics infrastructure."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.nushell_scripts_path = project_root / "nushell-env" / "scripts"
        self.analytics_script = self.nushell_scripts_path / "performance-analytics.nu"
        self.resource_monitor_script = self.nushell_scripts_path / "resource-monitor.nu"
        
        # Performance tracking
        self.operation_start_times = {}
        self.baseline_metrics = self._load_baseline_metrics()
        
        # Performance thresholds
        self.thresholds = {
            "build_time_warning_seconds": 60,
            "build_time_critical_seconds": 300,
            "test_time_warning_seconds": 30,
            "test_time_critical_seconds": 120,
            "lint_time_warning_seconds": 10,
            "lint_time_critical_seconds": 30,
            "memory_warning_mb": 4096,  # 4GB
            "memory_critical_mb": 8192,  # 8GB
            "cpu_warning_percent": 80,
            "cpu_critical_percent": 95
        }
    
    def _load_baseline_metrics(self) -> Dict:
        """Load baseline performance metrics."""
        try:
            # Try to get baseline from existing analytics
            if self.analytics_script.exists():
                result = subprocess.run(
                    ["nu", str(self.analytics_script), "baseline", "--format", "json"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    return json.loads(result.stdout)
        except Exception:
            pass
        
        # Default baseline metrics
        return {
            "python-env": {"build": 15, "test": 5, "lint": 3},
            "typescript-env": {"build": 20, "test": 8, "lint": 2},
            "rust-env": {"build": 45, "test": 10, "lint": 5},
            "go-env": {"build": 10, "test": 3, "lint": 2},
            "nushell-env": {"check": 2, "test": 1, "format": 1}
        }
    
    def detect_environment_from_path(self, file_path: str = None) -> str:
        """Detect environment from file path or current directory."""
        if file_path:
            if "python-env" in file_path:
                return "python-env"
            elif "typescript-env" in file_path:
                return "typescript-env"
            elif "rust-env" in file_path:
                return "rust-env"
            elif "go-env" in file_path:
                return "go-env"
            elif "nushell-env" in file_path:
                return "nushell-env"
        
        # Fallback to current directory
        cwd = os.getcwd()
        for env in ["python-env", "typescript-env", "rust-env", "go-env", "nushell-env"]:
            if env in cwd:
                return env
        
        return "unknown"
    
    def extract_operation_type(self, command: str) -> Optional[str]:
        """Extract operation type from command."""
        command_lower = command.lower()
        
        if "devbox run build" in command_lower or "cargo build" in command_lower or "npm run build" in command_lower:
            return "build"
        elif "devbox run test" in command_lower or "pytest" in command_lower or "npm test" in command_lower or "cargo test" in command_lower:
            return "test"
        elif "devbox run lint" in command_lower or "eslint" in command_lower or "ruff" in command_lower or "clippy" in command_lower:
            return "lint"
        elif "devbox run format" in command_lower or "prettier" in command_lower or "rustfmt" in command_lower:
            return "format"
        elif "devbox run check" in command_lower:
            return "check"
        
        return None
    
    def start_performance_tracking(self, environment: str, operation: str, command: str) -> str:
        """Start performance tracking for an operation."""
        try:
            tracking_id = f"{environment}_{operation}_{int(time.time())}"
            
            start_data = {
                "tracking_id": tracking_id,
                "environment": environment,
                "operation": operation,
                "command": command,
                "start_time": datetime.now().isoformat(),
                "start_memory_mb": psutil.virtual_memory().used // (1024 * 1024),
                "start_cpu_percent": psutil.cpu_percent(interval=1)
            }
            
            self.operation_start_times[tracking_id] = start_data
            
            # Use existing analytics infrastructure
            if self.analytics_script.exists():
                subprocess.run(
                    ["nu", str(self.analytics_script), "start-tracking", tracking_id, environment, operation],
                    check=False,
                    capture_output=True,
                    timeout=5
                )
            
            return tracking_id
            
        except Exception as e:
            print(f"⚠️ Failed to start performance tracking: {e}")
            return ""
    
    def end_performance_tracking(self, tracking_id: str) -> Dict:
        """End performance tracking and analyze results."""
        try:
            if tracking_id not in self.operation_start_times:
                return {"error": "Tracking ID not found"}
            
            start_data = self.operation_start_times[tracking_id]
            end_time = datetime.now()
            start_time = datetime.fromisoformat(start_data["start_time"])
            duration_seconds = (end_time - start_time).total_seconds()
            
            end_memory_mb = psutil.virtual_memory().used // (1024 * 1024)
            end_cpu_percent = psutil.cpu_percent(interval=1)
            
            performance_data = {
                "tracking_id": tracking_id,
                "environment": start_data["environment"],
                "operation": start_data["operation"],
                "command": start_data["command"],
                "duration_seconds": duration_seconds,
                "memory_delta_mb": end_memory_mb - start_data["start_memory_mb"],
                "peak_memory_mb": end_memory_mb,
                "avg_cpu_percent": (start_data["start_cpu_percent"] + end_cpu_percent) / 2,
                "completed_at": end_time.isoformat()
            }
            
            # Analyze performance
            analysis = self.analyze_performance(performance_data)
            performance_data.update(analysis)
            
            # Use existing analytics infrastructure
            if self.analytics_script.exists():
                subprocess.run(
                    ["nu", str(self.analytics_script), "end-tracking", tracking_id, 
                     str(duration_seconds), str(end_memory_mb)],
                    check=False,
                    capture_output=True,
                    timeout=5
                )
            
            # Cleanup
            del self.operation_start_times[tracking_id]
            
            return performance_data
            
        except Exception as e:
            return {"error": f"Failed to end performance tracking: {e}"}
    
    def analyze_performance(self, performance_data: Dict) -> Dict:
        """Analyze performance data and provide insights."""
        analysis = {
            "status": "normal",
            "warnings": [],
            "recommendations": [],
            "regression_detected": False
        }
        
        try:
            environment = performance_data["environment"]
            operation = performance_data["operation"]
            duration = performance_data["duration_seconds"]
            memory_delta = performance_data.get("memory_delta_mb", 0)
            cpu_percent = performance_data.get("avg_cpu_percent", 0)
            
            # Compare against thresholds
            warning_threshold = self.thresholds.get(f"{operation}_time_warning_seconds", 30)
            critical_threshold = self.thresholds.get(f"{operation}_time_critical_seconds", 60)
            
            if duration > critical_threshold:
                analysis["status"] = "critical"
                analysis["warnings"].append(f"Critical: {operation} took {duration:.1f}s (threshold: {critical_threshold}s)")
                analysis["recommendations"].append(f"Investigate {operation} performance bottlenecks")
            elif duration > warning_threshold:
                analysis["status"] = "warning"
                analysis["warnings"].append(f"Warning: {operation} took {duration:.1f}s (threshold: {warning_threshold}s)")
                analysis["recommendations"].append(f"Consider optimizing {operation} process")
            
            # Memory analysis
            if memory_delta > self.thresholds["memory_warning_mb"]:
                analysis["warnings"].append(f"High memory usage: +{memory_delta}MB")
                analysis["recommendations"].append("Check for memory leaks or optimize memory usage")
            
            # CPU analysis
            if cpu_percent > self.thresholds["cpu_warning_percent"]:
                analysis["warnings"].append(f"High CPU usage: {cpu_percent:.1f}%")
                analysis["recommendations"].append("Consider running operations during low-usage periods")
            
            # Regression detection
            baseline = self.baseline_metrics.get(environment, {}).get(operation)
            if baseline and duration > baseline * 1.5:  # 50% slower than baseline
                analysis["regression_detected"] = True
                analysis["warnings"].append(f"Performance regression: {duration:.1f}s vs baseline {baseline}s")
                analysis["recommendations"].append("Investigate recent changes that may have caused regression")
            
        except Exception as e:
            analysis["warnings"].append(f"Analysis failed: {e}")
        
        return analysis
    
    def generate_performance_report(self, environment: str = None) -> Dict:
        """Generate comprehensive performance report."""
        try:
            # Use existing analytics infrastructure
            if self.analytics_script.exists():
                cmd = ["nu", str(self.analytics_script), "report", "--days", "1", "--format", "json"]
                if environment:
                    cmd.extend(["--environment", environment])
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    try:
                        return json.loads(result.stdout)
                    except json.JSONDecodeError:
                        pass
            
            # Fallback: basic report
            return {
                "summary": "Performance analytics available through existing infrastructure",
                "recommendation": "Use: nu nushell-env/scripts/performance-analytics.nu report"
            }
            
        except Exception as e:
            return {"error": f"Failed to generate report: {e}"}
    
    def optimize_performance_settings(self, environment: str) -> List[str]:
        """Provide environment-specific performance optimization suggestions."""
        optimizations = []
        
        try:
            env_optimizations = {
                "python-env": [
                    "Use 'uv run' instead of activating virtual environments",
                    "Enable Python bytecode optimization with PYTHONOPTIMIZE=1",
                    "Consider using 'uv compile' for faster dependency resolution",
                    "Use pytest-xdist for parallel test execution"
                ],
                "typescript-env": [
                    "Enable TypeScript incremental compilation",
                    "Use 'npm ci' instead of 'npm install' in CI",
                    "Enable webpack caching for faster builds",
                    "Consider using 'ts-node' with SWC for faster execution"
                ],
                "rust-env": [
                    "Use 'cargo check' during development instead of full builds",
                    "Enable parallel compilation with CARGO_BUILD_JOBS",
                    "Use sccache for compilation caching",
                    "Consider using 'cargo-watch' for continuous testing"
                ],
                "go-env": [
                    "Use Go module proxy for faster dependency downloads",
                    "Enable build caching with GOCACHE",
                    "Use 'go build -i' for incremental builds",
                    "Consider using 'air' for live reloading during development"
                ],
                "nushell-env": [
                    "Use structured data pipelines instead of text processing",
                    "Enable command caching where possible",
                    "Optimize scripts with parallel processing using 'par-each'",
                    "Use built-in Nushell commands instead of external tools"
                ]
            }
            
            optimizations = env_optimizations.get(environment, [
                "Enable environment-specific optimizations",
                "Use existing performance analytics for insights"
            ])
            
        except Exception as e:
            optimizations.append(f"Optimization analysis failed: {e}")
        
        return optimizations
    
    def process_tool_event(self, hook_data: dict):
        """Process tool event for performance analytics."""
        try:
            tool_name = hook_data.get("tool_name", "")
            tool_input = hook_data.get("tool_input", {})
            
            if tool_name == "Bash":
                command = tool_input.get("command", "")
                environment = self.detect_environment_from_path()
                operation = self.extract_operation_type(command)
                
                if operation and environment != "unknown":
                    # Start performance tracking
                    tracking_id = self.start_performance_tracking(environment, operation, command)
                    
                    if tracking_id:
                        print(f"📊 Performance Tracking: Started for {environment} {operation}")
                        
                        # Provide immediate insights
                        baseline = self.baseline_metrics.get(environment, {}).get(operation)
                        if baseline:
                            print(f"   Expected duration: ~{baseline}s")
                        
                        # Store tracking ID for PostToolUse hook
                        # Note: In a real implementation, this would need to be stored persistently
                        print(f"   Tracking ID: {tracking_id}")
                
                # Provide performance insights for DevPod operations
                elif "devpod" in command:
                    print("🐳 DevPod Performance: Container operations may impact system resources")
                    
                    # Check current system resources
                    try:
                        memory_percent = psutil.virtual_memory().percent
                        cpu_percent = psutil.cpu_percent(interval=1)
                        
                        if memory_percent > 80:
                            print(f"   ⚠️ High memory usage: {memory_percent:.1f}%")
                        if cpu_percent > 80:
                            print(f"   ⚠️ High CPU usage: {cpu_percent:.1f}%")
                    except Exception:
                        pass
            
            elif tool_name in ["Edit", "MultiEdit", "Write"]:
                file_path = tool_input.get("file_path", "")
                environment = self.detect_environment_from_path(file_path)
                
                if environment != "unknown":
                    # Suggest performance optimizations for large files
                    content = tool_input.get("content", "")
                    if len(content) > 10000:  # Large file
                        optimizations = self.optimize_performance_settings(environment)
                        print(f"📊 Large file detected in {environment}")
                        print(f"   Performance tips: {optimizations[0] if optimizations else 'Use existing analytics'}")
            
            print(f"🎯 Performance Analytics: Processed {tool_name}")
            
        except Exception as e:
            print(f"⚠️ Performance analytics processing failed: {e}")

def main():
    """Main hook entry point."""
    try:
        # Read hook input from stdin
        hook_input = json.load(sys.stdin)
        
        # Get project root
        project_root = Path.cwd()
        
        # Initialize performance analytics integrator
        integrator = PerformanceAnalyticsIntegrator(project_root)
        
        # Process the event
        integrator.process_tool_event(hook_input)
        
        # Return success
        sys.exit(0)
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Performance Analytics Hook failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()