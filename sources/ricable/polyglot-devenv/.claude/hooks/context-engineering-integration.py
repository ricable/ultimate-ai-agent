#!/usr/bin/env python3
"""
Context Engineering Integration Hook
Automates PRP generation, template syncing, and context management.

Features:
- Auto-generate PRPs when feature files are created/modified
- Sync PRP templates across environments
- Update context engineering artifacts
- Integration with existing PRP generation system
"""

import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import shutil
import re

class ContextEngineeringIntegrator:
    """Manages context engineering automation through Claude Code hooks."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.context_engineering_path = project_root / "context-engineering"
        self.templates_path = self.context_engineering_path / "templates"
        self.prp_path = self.context_engineering_path / "PRPs"
        self.lib_path = self.context_engineering_path / "lib"
        self.features_path = project_root / "features"
        
    def detect_environment_from_content(self, content: str) -> str:
        """Detect target environment from file content."""
        content_lower = content.lower()
        
        # Environment detection patterns
        if any(pattern in content_lower for pattern in ["python", "fastapi", "uvicorn", "pydantic", "pytest"]):
            return "python-env"
        elif any(pattern in content_lower for pattern in ["typescript", "node", "npm", "jest", "react", "express"]):
            return "typescript-env"
        elif any(pattern in content_lower for pattern in ["rust", "cargo", "tokio", "serde", "actix"]):
            return "rust-env"
        elif any(pattern in content_lower for pattern in ["golang", "go ", "gin", "gorilla", "gorm"]):
            return "go-env"
        elif any(pattern in content_lower for pattern in ["nushell", "nu ", "shell", "script"]):
            return "nushell-env"
        
        return "python-env"  # Default to Python
    
    def extract_feature_requirements(self, content: str) -> dict:
        """Extract feature requirements from markdown content."""
        requirements = {
            "title": "",
            "description": "",
            "acceptance_criteria": [],
            "technical_requirements": [],
            "environment": "python-env"
        }
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Extract title (first # header)
            if line.startswith('# ') and not requirements["title"]:
                requirements["title"] = line[2:].strip()
            
            # Detect sections
            elif line.startswith('## '):
                section = line[3:].lower().strip()
                if 'description' in section:
                    current_section = 'description'
                elif 'acceptance' in section or 'criteria' in section:
                    current_section = 'acceptance_criteria'
                elif 'technical' in section or 'requirements' in section:
                    current_section = 'technical_requirements'
                else:
                    current_section = None
            
            # Collect content for sections
            elif line and current_section:
                if line.startswith('- ') or line.startswith('* '):
                    if current_section in ['acceptance_criteria', 'technical_requirements']:
                        requirements[current_section].append(line[2:].strip())
                elif current_section == 'description':
                    if requirements["description"]:
                        requirements["description"] += " " + line
                    else:
                        requirements["description"] = line
        
        # Detect environment from content
        requirements["environment"] = self.detect_environment_from_content(content)
        
        return requirements
    
    def auto_generate_prp_from_feature(self, feature_file: str, content: str):
        """Auto-generate PRP from feature file using enhanced PRP system."""
        try:
            print(f"🔄 Auto-generating PRP from feature file: {feature_file}")
            
            # Extract requirements from feature content
            requirements = self.extract_feature_requirements(content)
            environment = requirements["environment"]
            
            # Use the enhanced PRP generation system
            generate_script = self.project_root / ".claude" / "commands" / "generate-prp-v2.py"
            if generate_script.exists():
                cmd = [
                    "python", str(generate_script),
                    feature_file,
                    "--env", environment,
                    "--template", "full",
                    "--workers", "2"
                ]
                
                result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ PRP auto-generated successfully for {environment}")
                    
                    # Update template if needed
                    self.sync_template_updates(environment)
                    
                    return True
                else:
                    print(f"⚠️ PRP auto-generation failed: {result.stderr}")
                    return False
            else:
                print(f"⚠️ PRP generation script not found: {generate_script}")
                return False
                
        except Exception as e:
            print(f"⚠️ Auto-generation failed: {e}")
            return False
    
    def sync_template_updates(self, environment: str):
        """Sync template updates across the context engineering system."""
        try:
            print(f"🔄 Syncing templates for {environment}")
            
            template_mapping = {
                "python-env": "python_prp.md",
                "typescript-env": "typescript_prp.md", 
                "rust-env": "rust_prp.md",
                "go-env": "go_prp.md",
                "nushell-env": "nushell_prp.md"
            }
            
            template_file = template_mapping.get(environment)
            if not template_file:
                print(f"⚠️ No template found for environment: {environment}")
                return
            
            template_path = self.templates_path / template_file
            if template_path.exists():
                # Update timestamp in template
                content = template_path.read_text()
                
                # Add or update last sync timestamp
                timestamp_marker = "<!-- Last synced: "
                new_timestamp = f"<!-- Last synced: {datetime.now().isoformat()} -->"
                
                if timestamp_marker in content:
                    # Replace existing timestamp
                    content = re.sub(
                        r'<!-- Last synced: [^>]+ -->',
                        new_timestamp,
                        content
                    )
                else:
                    # Add new timestamp at the top
                    content = new_timestamp + "\n\n" + content
                
                template_path.write_text(content)
                print(f"✅ Template synced: {template_file}")
            else:
                print(f"⚠️ Template not found: {template_path}")
                
        except Exception as e:
            print(f"⚠️ Template sync failed: {e}")
    
    def update_context_artifacts(self, file_path: str, environment: str):
        """Update context engineering artifacts based on file changes."""
        try:
            # Create features directory if it doesn't exist
            self.features_path.mkdir(exist_ok=True)
            
            # Update environment-specific configurations
            config_path = self.templates_path / "config" / f"{environment.replace('-env', '')}_config.yaml"
            if config_path.exists():
                print(f"📝 Updating environment config: {config_path}")
                
                # Read current config
                config_content = config_path.read_text()
                
                # Add last_updated timestamp
                timestamp_line = f"last_updated: {datetime.now().isoformat()}"
                
                if "last_updated:" in config_content:
                    # Replace existing timestamp
                    config_content = re.sub(
                        r'last_updated: [^\n]+',
                        timestamp_line,
                        config_content
                    )
                else:
                    # Add new timestamp
                    config_content = timestamp_line + "\n" + config_content
                
                config_path.write_text(config_content)
                print(f"✅ Environment config updated")
            
        except Exception as e:
            print(f"⚠️ Context artifact update failed: {e}")
    
    def validate_prp_consistency(self, file_path: str):
        """Validate PRP consistency with templates and standards."""
        try:
            if "/PRPs/" in file_path and file_path.endswith('.md'):
                print(f"🔍 Validating PRP consistency: {file_path}")
                
                # Use existing validation system if available
                validation_script = self.lib_path / "validation_specifications.py"
                if validation_script.exists():
                    cmd = [
                        "python", str(validation_script),
                        file_path
                    ]
                    
                    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✅ PRP validation passed")
                    else:
                        print(f"⚠️ PRP validation issues found: {result.stderr}")
                        
        except Exception as e:
            print(f"⚠️ PRP validation failed: {e}")
    
    def process_tool_event(self, hook_data: dict):
        """Process the tool event and take appropriate context engineering actions."""
        try:
            tool_name = hook_data.get("tool_name", "")
            tool_input = hook_data.get("tool_input", {})
            file_path = tool_input.get("file_path", "")
            content = tool_input.get("content", "")
            
            if not file_path:
                return
            
            print(f"🎯 Context Engineering: Processing {tool_name} for {file_path}")
            
            # Handle feature file changes
            if "/features/" in file_path and file_path.endswith('.md'):
                if tool_name in ["Write", "Edit", "MultiEdit"]:
                    self.auto_generate_prp_from_feature(file_path, content)
                    
            # Handle template changes
            elif "/templates/" in file_path:
                environment = self.detect_environment_from_content(content)
                self.sync_template_updates(environment)
                self.update_context_artifacts(file_path, environment)
                
            # Handle PRP changes
            elif "/PRPs/" in file_path:
                self.validate_prp_consistency(file_path)
                environment = self.detect_environment_from_content(content)
                self.update_context_artifacts(file_path, environment)
                
            # Handle context-engineering library changes
            elif "/context-engineering/lib/" in file_path:
                print(f"📚 Context engineering library updated: {Path(file_path).name}")
                # Trigger library validation or tests if available
                
            print(f"✅ Context Engineering: Completed processing for {file_path}")
            
        except Exception as e:
            print(f"⚠️ Context Engineering processing failed: {e}")

def main():
    """Main hook entry point."""
    try:
        # Read hook input from stdin
        hook_input = json.load(sys.stdin)
        
        # Get project root (assuming hook runs from project directory)
        project_root = Path.cwd()
        
        # Initialize context engineering integrator
        integrator = ContextEngineeringIntegrator(project_root)
        
        # Process the event
        integrator.process_tool_event(hook_input)
        
        # Return success
        sys.exit(0)
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Context Engineering Hook failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()