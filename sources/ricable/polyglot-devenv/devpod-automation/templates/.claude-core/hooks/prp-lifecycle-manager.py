#!/usr/bin/env python3
"""
PRP Lifecycle Management Hook
Integrates with the existing context-engineering system to automate PRP workflow.

Features:
- Auto-update PRP execution status based on file changes
- Generate execution reports when PRPs complete
- Sync PRP templates and track progress
- Integration with existing performance analytics
"""

import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import asyncio
import tempfile
import os

class PRPLifecycleManager:
    """Manages PRP lifecycle events through Claude Code hooks."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.context_engineering_path = project_root / "context-engineering"
        self.prp_path = self.context_engineering_path / "PRPs"
        self.execution_reports_path = self.context_engineering_path / "execution-reports"
        
    def get_environment_from_path(self, file_path: str) -> str:
        """Detect environment based on file path."""
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
        return "unknown"
    
    def is_prp_related_file(self, file_path: str) -> bool:
        """Check if file is PRP-related."""
        prp_indicators = [
            "/PRPs/",
            "/context-engineering/",
            "/features/",
            "prp",
            "PRP"
        ]
        return any(indicator in file_path for indicator in prp_indicators)
    
    def extract_prp_name(self, file_path: str) -> str:
        """Extract PRP name from file path."""
        path = Path(file_path)
        if "PRPs" in str(path):
            return path.stem
        elif "features" in str(path):
            return f"feature-{path.stem}"
        return path.stem
    
    def update_prp_status(self, prp_name: str, environment: str, status: str):
        """Update PRP execution status."""
        try:
            status_file = self.execution_reports_path / f"{prp_name}-{environment}-status.json"
            
            # Create execution reports directory if it doesn't exist
            self.execution_reports_path.mkdir(parents=True, exist_ok=True)
            
            status_data = {
                "prp_name": prp_name,
                "environment": environment,
                "status": status,
                "last_updated": datetime.now().isoformat(),
                "file_path": str(status_file)
            }
            
            if status_file.exists():
                try:
                    with open(status_file, 'r') as f:
                        existing_data = json.load(f)
                    status_data.update(existing_data)
                except (json.JSONDecodeError, KeyError):
                    pass  # Use new data if existing data is invalid
            
            status_data["last_updated"] = datetime.now().isoformat()
            status_data["status"] = status
            
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
            print(f"📋 PRP Status Updated: {prp_name} ({environment}) -> {status}")
            
        except Exception as e:
            print(f"⚠️ Failed to update PRP status: {e}")
    
    def generate_execution_report(self, prp_name: str, environment: str):
        """Generate comprehensive execution report for completed PRP."""
        try:
            report_file = self.execution_reports_path / f"{prp_name}-{environment}-report.md"
            
            # Create execution reports directory if it doesn't exist
            self.execution_reports_path.mkdir(parents=True, exist_ok=True)
            
            report_content = f"""# PRP Execution Report: {prp_name}

**Environment**: {environment}
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Status**: Completed

## Summary
PRP '{prp_name}' has been successfully executed in the {environment} environment.

## Execution Details
- **Start Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Environment**: {environment}
- **PRP Name**: {prp_name}

## Performance Metrics
Generated using integrated performance analytics system.

## Next Steps
- Review implementation for best practices
- Run cross-environment validation if applicable
- Update documentation if needed

---
*Report generated automatically by PRP Lifecycle Management Hook*
"""
            
            with open(report_file, 'w') as f:
                f.write(report_content)
                
            print(f"📊 Execution Report Generated: {report_file}")
            
            # Trigger performance analytics if available
            self.trigger_performance_analytics(prp_name, environment)
            
        except Exception as e:
            print(f"⚠️ Failed to generate execution report: {e}")
    
    def trigger_performance_analytics(self, prp_name: str, environment: str):
        """Trigger performance analytics for PRP completion."""
        try:
            analytics_script = self.project_root / "nushell-env" / "scripts" / "performance-analytics.nu"
            if analytics_script.exists():
                cmd = [
                    "nu", str(analytics_script),
                    "record", "prp_completion", prp_name, environment,
                    "--quiet"
                ]
                subprocess.run(cmd, check=False, capture_output=True)
                print(f"📈 Performance analytics triggered for {prp_name}")
        except Exception as e:
            print(f"⚠️ Performance analytics failed: {e}")
    
    def auto_generate_prp(self, file_path: str, environment: str):
        """Auto-generate PRP when feature files are modified."""
        try:
            if "features/" in file_path and file_path.endswith('.md'):
                print(f"🔄 Auto-generating PRP for feature file: {file_path}")
                
                # Use the enhanced PRP generation system
                generate_script = self.project_root / ".claude" / "commands" / "generate-prp-v2.py"
                if generate_script.exists():
                    cmd = [
                        "python", str(generate_script),
                        file_path,
                        "--env", environment,
                        "--workers", "2",
                        "--debug"
                    ]
                    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✅ PRP auto-generated successfully")
                    else:
                        print(f"⚠️ PRP auto-generation failed: {result.stderr}")
                        
        except Exception as e:
            print(f"⚠️ Auto-generation failed: {e}")
    
    def process_tool_event(self, hook_data: dict):
        """Process the tool event and take appropriate PRP actions."""
        try:
            tool_name = hook_data.get("tool_name", "")
            tool_input = hook_data.get("tool_input", {})
            file_path = tool_input.get("file_path", "")
            
            if not file_path:
                return
            
            environment = self.get_environment_from_path(file_path)
            
            # Handle PRP-related file changes
            if self.is_prp_related_file(file_path):
                prp_name = self.extract_prp_name(file_path)
                
                if tool_name in ["Edit", "MultiEdit", "Write"]:
                    if "PRPs/" in file_path:
                        self.update_prp_status(prp_name, environment, "in_progress")
                    elif "features/" in file_path:
                        self.auto_generate_prp(file_path, environment)
                        self.update_prp_status(prp_name, environment, "planned")
                
                # Check for completion indicators
                if "completed" in str(tool_input.get("content", "")).lower():
                    self.update_prp_status(prp_name, environment, "completed")
                    self.generate_execution_report(prp_name, environment)
            
            print(f"🎯 PRP Lifecycle: Processed {tool_name} for {file_path}")
            
        except Exception as e:
            print(f"⚠️ PRP Lifecycle processing failed: {e}")

def main():
    """Main hook entry point."""
    try:
        # Read hook input from stdin
        hook_input = json.load(sys.stdin)
        
        # Get project root (assuming hook runs from project directory)
        project_root = Path.cwd()
        
        # Initialize PRP lifecycle manager
        manager = PRPLifecycleManager(project_root)
        
        # Process the event
        manager.process_tool_event(hook_input)
        
        # Return success
        sys.exit(0)
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ PRP Lifecycle Hook failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()