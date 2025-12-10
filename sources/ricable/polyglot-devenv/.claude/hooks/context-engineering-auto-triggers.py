#!/usr/bin/env python3
"""
Context Engineering Auto-Triggers Hook
Automatically generates PRPs when feature files are edited to streamline context engineering workflow.

Features:
- Detects edits to feature files in context-engineering/workspace/features/
- Auto-generates PRPs using existing /generate-prp infrastructure
- Smart triggering to avoid excessive regeneration
- Environment detection for optimal PRP template selection
- Integration with existing context engineering framework
"""

import json
import sys
import subprocess
import os
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta

class ContextEngineeringAutoTrigger:
    """Automatically triggers PRP generation for context engineering workflow."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.context_engineering_root = project_root / "context-engineering"
        self.features_dir = self.context_engineering_root / "workspace" / "features"
        self.templates_dir = self.context_engineering_root / "workspace" / "templates"
        self.prps_dir = self.context_engineering_root / "workspace" / "PRPs"
        
        # State tracking for smart triggering
        self.last_generated = {}  # file_path -> timestamp
        self.file_hashes = {}     # file_path -> content_hash
        self.generation_cooldown = 60  # seconds between regenerations for same file
        
        # Environment mapping
        self.environment_patterns = {
            "python": ["python", "py", "fastapi", "django", "flask", "api"],
            "typescript": ["typescript", "ts", "react", "next", "ui", "frontend", "web"],
            "rust": ["rust", "rs", "tokio", "actix", "service", "performance"],
            "go": ["go", "golang", "api", "microservice", "cli", "server"],
            "nushell": ["nushell", "nu", "automation", "script", "pipeline", "data"]
        }
        
        # Create directories if they don't exist
        self.prps_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_environment_from_content(self, file_path: Path, content: str) -> List[str]:
        """Detect target environments from feature file content."""
        environments = []
        content_lower = content.lower()
        
        for env, keywords in self.environment_patterns.items():
            for keyword in keywords:
                if keyword in content_lower:
                    environments.append(env)
                    break
        
        # Default to Python if no specific environment detected
        if not environments:
            environments = ["python"]
        
        return list(set(environments))  # Remove duplicates
    
    def get_file_content_hash(self, file_path: Path) -> str:
        """Get SHA256 hash of file content for change detection."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return hashlib.sha256(content.encode('utf-8')).hexdigest()
        except Exception:
            return ""
    
    def should_generate_prp(self, file_path: Path) -> bool:
        """Determine if PRP should be generated based on smart triggering logic."""
        now = time.time()
        file_str = str(file_path)
        
        # Check cooldown period
        last_gen = self.last_generated.get(file_str, 0)
        if now - last_gen < self.generation_cooldown:
            return False
        
        # Check if content actually changed
        current_hash = self.get_file_content_hash(file_path)
        if not current_hash:
            return False
        
        previous_hash = self.file_hashes.get(file_str)
        if previous_hash == current_hash:
            return False
        
        # Update tracking
        self.file_hashes[file_str] = current_hash
        self.last_generated[file_str] = now
        
        return True
    
    def extract_feature_name(self, file_path: Path) -> str:
        """Extract feature name from file path."""
        return file_path.stem.replace('-', '_').replace(' ', '_')
    
    def generate_prp_for_environments(self, feature_file: Path, environments: List[str]) -> Dict[str, bool]:
        """Generate PRPs for specified environments using existing infrastructure."""
        results = {}
        feature_name = self.extract_feature_name(feature_file)
        
        for env in environments:
            try:
                # Use the existing /generate-prp command infrastructure
                # This integrates with their existing Claude Code slash command
                prp_command = [
                    "claude",
                    "/generate-prp",
                    str(feature_file),
                    "--env", f"dev-env/{env}",
                    "--output-dir", str(self.prps_dir),
                    "--template-dir", str(self.templates_dir),
                    "--quiet"
                ]
                
                print(f"🚀 Auto-generating PRP: {feature_name} for {env}")
                
                result = subprocess.run(
                    prp_command,
                    capture_output=True,
                    text=True,
                    timeout=120,  # 2 minute timeout
                    cwd=self.project_root
                )
                
                if result.returncode == 0:
                    results[env] = True
                    prp_file = self.prps_dir / f"{feature_name}_{env}.md"
                    print(f"✅ Generated: {prp_file}")
                    
                    # Log successful generation
                    self._log_generation(feature_file, env, prp_file, True)
                else:
                    results[env] = False
                    print(f"❌ Failed to generate PRP for {env}: {result.stderr}")
                    self._log_generation(feature_file, env, None, False, result.stderr)
                
            except subprocess.TimeoutExpired:
                results[env] = False
                print(f"⏱️ Timeout generating PRP for {env}")
                self._log_generation(feature_file, env, None, False, "Timeout")
            except Exception as e:
                results[env] = False
                print(f"⚠️ Error generating PRP for {env}: {e}")
                self._log_generation(feature_file, env, None, False, str(e))
        
        return results
    
    def _log_generation(self, feature_file: Path, environment: str, prp_file: Optional[Path], success: bool, error: str = ""):
        """Log PRP generation attempts for analytics."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "feature_file": str(feature_file.relative_to(self.project_root)),
                "environment": environment,
                "prp_file": str(prp_file.relative_to(self.project_root)) if prp_file else None,
                "success": success,
                "error": error
            }
            
            log_file = self.project_root / "dev-env" / "nushell" / "logs" / "context_engineering_auto.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            print(f"⚠️ Failed to log PRP generation: {e}")
    
    def suggest_environment_optimizations(self, environments: List[str]) -> List[str]:
        """Suggest optimizations based on detected environments."""
        suggestions = []
        
        if "python" in environments:
            suggestions.append("💡 Python: Consider using FastAPI for high-performance APIs")
        
        if "typescript" in environments:
            suggestions.append("💡 TypeScript: Use Next.js for full-stack development")
        
        if "rust" in environments:
            suggestions.append("💡 Rust: Leverage Tokio for async performance")
        
        if "go" in environments:
            suggestions.append("💡 Go: Design with microservices architecture")
        
        if "nushell" in environments:
            suggestions.append("💡 Nushell: Utilize structured data pipelines")
        
        if len(environments) > 1:
            suggestions.append("🔗 Multi-environment: Consider MCP integration for cross-language coordination")
        
        return suggestions
    
    def process_feature_file_edit(self, file_path: str, content: str = "") -> Dict:
        """Process a feature file edit and trigger PRP generation if appropriate."""
        try:
            feature_path = Path(file_path)
            
            # Only process files in the features directory
            if not str(feature_path).endswith('.md') or "features" not in str(feature_path):
                return {"processed": False, "reason": "Not a feature file"}
            
            # Ensure it's actually a feature file in the correct directory
            try:
                feature_path.relative_to(self.features_dir)
            except ValueError:
                return {"processed": False, "reason": "Not in features directory"}
            
            # Check if we should generate PRPs
            if not self.should_generate_prp(feature_path):
                return {"processed": False, "reason": "Cooldown period or no content changes"}
            
            # Read content if not provided
            if not content:
                try:
                    with open(feature_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    return {"processed": False, "reason": f"Failed to read file: {e}"}
            
            # Detect target environments
            environments = self.detect_environment_from_content(feature_path, content)
            
            print(f"🎯 Context Engineering Auto-Trigger: Processing {feature_path.name}")
            print(f"   Detected environments: {', '.join(environments)}")
            
            # Generate PRPs for detected environments
            generation_results = self.generate_prp_for_environments(feature_path, environments)
            
            # Provide optimization suggestions
            suggestions = self.suggest_environment_optimizations(environments)
            
            success_count = sum(1 for success in generation_results.values() if success)
            total_count = len(generation_results)
            
            result = {
                "processed": True,
                "feature_file": str(feature_path.relative_to(self.project_root)),
                "environments": environments,
                "generation_results": generation_results,
                "success_count": success_count,
                "total_count": total_count,
                "suggestions": suggestions
            }
            
            # Print summary
            if success_count > 0:
                print(f"✅ Successfully generated {success_count}/{total_count} PRPs")
            if suggestions:
                print("💡 Optimization suggestions:")
                for suggestion in suggestions:
                    print(f"   {suggestion}")
            
            return result
            
        except Exception as e:
            error_result = {"processed": False, "error": str(e)}
            print(f"❌ Context Engineering Auto-Trigger failed: {e}")
            return error_result
    
    def process_tool_event(self, hook_data: dict):
        """Process tool event for context engineering auto-triggers."""
        try:
            tool_name = hook_data.get("tool_name", "")
            tool_input = hook_data.get("tool_input", {})
            
            if tool_name in ["Edit", "MultiEdit", "Write"]:
                file_path = tool_input.get("file_path", "")
                content = tool_input.get("new_string", "") or tool_input.get("content", "")
                
                if file_path:
                    result = self.process_feature_file_edit(file_path, content)
                    
                    if result.get("processed"):
                        success_count = result.get("success_count", 0)
                        total_count = result.get("total_count", 0)
                        
                        if success_count > 0:
                            print(f"🚀 Context Engineering: Auto-generated {success_count}/{total_count} PRPs")
                        
                        # Suggest next steps
                        if success_count > 0:
                            print("🎯 Next steps:")
                            print("   • Review generated PRPs in context-engineering/workspace/PRPs/")
                            print("   • Execute PRPs with: /execute-prp <prp-file>")
                            print("   • Provision DevPod environments as needed")
            
            print(f"🎯 Context Engineering Auto-Trigger: Processed {tool_name}")
            
        except Exception as e:
            print(f"⚠️ Context Engineering Auto-Trigger processing failed: {e}")

def main():
    """Main hook entry point."""
    try:
        # Read hook input from stdin
        hook_input = json.load(sys.stdin)
        
        # Get project root
        project_root = Path.cwd()
        
        # Initialize context engineering auto-trigger
        auto_trigger = ContextEngineeringAutoTrigger(project_root)
        
        # Process the event
        auto_trigger.process_tool_event(hook_input)
        
        # Return success
        sys.exit(0)
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Context Engineering Auto-Trigger Hook failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()